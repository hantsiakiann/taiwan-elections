#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cec_to_json.py
--------------
Convert a Taiwan Central Election Commission (中選會) village-level
("各村里得票數") result spreadsheet into a structured JSON file, with
English/romanized names attached to every 縣市 / 鄉鎮市區 / 村里.

USAGE
    python cec_to_json.py INPUT.xlsx OUTPUT.json --election-id 2024-president --election-type president

    INPUT may be either .xlsx or the legacy .xls format.

REQUIREMENTS
    pip install openpyxl pypinyin --break-system-packages
    pip install xlrd --break-system-packages   # only needed for .xls input

WHAT THIS HANDLES
    The standard CEC "各村里(得票數/得票率)一覽表" layout used for
    president, legislator-at-large, and local (mayor/council) races:
        Row 1: title (merged, ignored)
        Row 2: column headers (merged down through row 5 for most columns)
        Row 3: candidate number + name(s), one column per candidate,
               merged down through row 5
        Row 6+: data rows, one of three kinds per row:
            - grand total row      (col A filled with "總計"-like text, col B empty)
            - township/city total  (col A filled, col B empty)
            - village/li row       (col A empty, col B filled)

WHAT THIS DOES *NOT* HANDLE OUT OF THE BOX
    - Legislative district files that add an extra "選舉區" (district)
      column. If your file has one, adjust `COLS_BEFORE_CANDIDATES` below.
    - Referendum (公投) files, which have a different vote-count layout
      (agree/disagree instead of per-candidate columns).
    - Files where the sheet name is not the county/city name in Chinese.
    Read the parsing loop before trusting it on a new file layout --
    it is built and tested against the 2024 presidential Nantou County
    file, and CEC's own layout has drifted slightly between election
    cycles in the past.

ROMANIZATION CAVEAT (IMPORTANT)
    Village/township names are romanized automatically using Hanyu
    Pinyin (via `pypinyin`), which is NOT always what the place is
    officially/colloquially known as in English:
      - Six special municipalities + all 22 county/city names are
        hardcoded to their correct official English names below
        (Taipei, not "Taibei"; Kaohsiung, not "Gaoxiong"; etc.)
      - Township/village names are auto-romanized and will sometimes
        be wrong for places whose English name derives from Taiwanese
        Hokkien or a legacy romanization rather than Mandarin Pinyin
        (e.g. 鹿港 is officially "Lukang", not the Pinyin "Ludang").
    Treat the auto-generated `*_en` fields as a readable default, not
    ground truth. `OVERRIDES` below is the place to fix specific
    townships/villages as you notice mistakes -- it persists across runs.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import openpyxl
from pypinyin import lazy_pinyin

# ---------------------------------------------------------------------------
# 1. Name translation
# ---------------------------------------------------------------------------

# Official English names for Taiwan's 22 top-level divisions (縣市).
# These do NOT follow plain Hanyu Pinyin (Taipei != "Taibei"), so they are
# hardcoded rather than auto-romanized.
# Deprecated counties are also added for elections before 2014.
COUNTY_EN = {
    "臺北市": "Taipei City", "台北市": "Taipei City",
    "新北市": "New Taipei City",
    "臺北縣": "Taipei County", "台北縣": "Taipei County",
    "桃園市": "Taoyuan City",
    "桃園縣": "Taoyuan County",
    "臺中市": "Taichung City", "台中市": "Taichung City",
    "臺中縣": "Taichung County", "台中縣": "Taichung County",
    "臺南市": "Tainan City", "台南市": "Tainan City",
    "臺南縣": "Tainan County", "台南縣": "Tainan County",
    "高雄市": "Kaohsiung City", 
    "高雄縣": "Kaohsiung County",
    "基隆市": "Keelung City",
    "新竹市": "Hsinchu City",
    "新竹縣": "Hsinchu County",
    "苗栗縣": "Miaoli County",
    "彰化縣": "Changhua County",
    "南投縣": "Nantou County",
    "雲林縣": "Yunlin County",
    "嘉義市": "Chiayi City",
    "嘉義縣": "Chiayi County",
    "屏東縣": "Pingtung County",
    "宜蘭縣": "Yilan County",
    "花蓮縣": "Hualien County",
    "臺東縣": "Taitung County", "台東縣": "Taitung County",
    "澎湖縣": "Penghu County",
    "金門縣": "Kinmen County",
    "連江縣": "Lienchiang County",
}

# Manual corrections for townships/villages whose common English name is
# NOT plain Hanyu Pinyin of the Mandarin reading. Add to this as you spot
# mistakes -- key is the exact Chinese text (with suffix), value is the
# corrected English name (with suffix translated).
#
# Note: villages combining multiple 村/里 into one polling station (joined
# with '、' in the source cell, e.g. '復興村、福沃村') do NOT need an entry
# here -- see canonicalize_multi_village() and romanize()'s '、' handling
# below, which romanize and order those generically.
OVERRIDES = {
    "鹿港鎮": "Lukang Township",
}

# ISO 3166-2:TW codes for Taiwan's 22 counties/cities (statoids.com/ytw.html,
# matching the current ISO 3166-2 standard as of the 2015-11-27 update).
# Used as the county/city segment of every township and village identifier,
# e.g. "NAN-CAOTUN-YUFENG" instead of a romanized "NANTOU-CAOTUN-YUFENG".
# Deprecated counties are mapped onto the deprecated ISO codes, except for
# Taipei County, which is the equivalent of New Taipei City, and Taoyuan
# County, which is the equivalent of Taoyuan City.
COUNTY_ISO = {
    "彰化縣": "CHA",
    "嘉義縣": "CYQ",
    "嘉義市": "CYI",
    "新竹縣": "HSQ",
    "新竹市": "HSZ",
    "花蓮縣": "HUA",
    "宜蘭縣": "ILA",
    "高雄市": "KHH",
    "高雄縣": "KHQ",
    "基隆市": "KEE",
    "金門縣": "KIN",
    "連江縣": "LIE",
    "苗栗縣": "MIA",
    "南投縣": "NAN",
    "新北市": "NWT",
    "臺北縣": "NWT", "台北縣": "NWT", 
    "澎湖縣": "PEN",
    "屏東縣": "PIF",
    "臺中市": "TXG", "台中市": "TXG",
    "臺中縣": "TXQ", "台中縣": "TXQ",
    "臺南市": "TNN", "台南市": "TNN",
    "臺南縣": "TNQ", "台南縣": "TNQ",
    "臺北市": "TPE", "台北市": "TPE",
    "臺東縣": "TTT", "台東縣": "TTT",
    "桃園市": "TAO",
    "桃園縣": "TAO",
    "雲林縣": "YUN",
}

SUFFIX_EN = {
    "縣": "County", "市": "City", "鄉": "Township",
    "鎮": "Township", "區": "District", "村": "Village", "里": "Village",
}


MULTI_VILLAGE_SEP = "、"


def canonicalize_multi_village(zh_text: str) -> str:
    """Normalize a CEC row where one polling station serves multiple 村/里
    at once, joined with '、' in the source cell (e.g. '復興村、福沃村').
    Since the votes can't be split back out per village, this combined
    group has to be treated as its own synthetic 'village' -- but CEC
    doesn't list the group in a consistent order across years for the
    same real-world combination (e.g. '復興村、福沃村' one year,
    '福沃村、復興村' another). Sorting the parts into a fixed order makes
    the same combination produce the same village_zh / village_id no
    matter which order that year's file happened to list them in.

    A no-op for a plain single village/township name (no '、').
    """
    if MULTI_VILLAGE_SEP not in zh_text:
        return zh_text
    parts = [p.strip() for p in zh_text.split(MULTI_VILLAGE_SEP) if p.strip()]
    return MULTI_VILLAGE_SEP.join(sorted(parts))


def _join_with_and(items: list) -> str:
    """['A'] -> 'A'; ['A','B'] -> 'A and B'; ['A','B','C'] -> 'A, B, and C'."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _romanize_one(zh_text: str) -> tuple:
    """Romanize a single (non-'、'-joined) admin-unit name. Returns
    (romanized_root, suffix_word_or_None) so callers combining multiple
    names (see romanize()'s '、' handling) can join the roots and append
    one shared suffix word rather than repeating it per name."""
    if zh_text in OVERRIDES:
        return OVERRIDES[zh_text], None
    suffix = zh_text[-1] if zh_text[-1] in SUFFIX_EN else ""
    root = zh_text[:-1] if suffix else zh_text
    romanized_root = "".join(lazy_pinyin(root)).capitalize()
    return romanized_root, (SUFFIX_EN[suffix] if suffix else None)


def romanize(zh_text: str, level: str) -> str:
    """Return an English rendering of a Chinese admin-unit name.

    level is one of 'county', 'township', 'village' -- used to resolve the
    ambiguous 市 suffix (county-level City vs. not applicable at village
    level) and to pick the right lookup table.
    """
    zh_text = zh_text.strip()
    if level == "county" and zh_text in COUNTY_EN:
        return COUNTY_EN[zh_text]
    if zh_text in OVERRIDES:
        return OVERRIDES[zh_text]

    if MULTI_VILLAGE_SEP in zh_text:
        # A combined-village name (see canonicalize_multi_village) --
        # romanize each part on its own and join. The single-name path
        # below would otherwise pinyin-ize the '、' itself along with
        # everything after it, since the whole string ends in a valid
        # suffix character ('村'/'里') and looks like one (very wrong) root.
        parts = zh_text.split(MULTI_VILLAGE_SEP)
        romanized = [_romanize_one(p) for p in parts]
        suffix_word = next((s for _, s in romanized if s), None)
        joined = _join_with_and([root for root, _ in romanized])
        return f"{joined} {suffix_word}" if suffix_word else joined

    romanized_root, suffix_word = _romanize_one(zh_text)
    if suffix_word:
        return f"{romanized_root} {suffix_word}"
    return romanized_root


GENERIC_SUFFIX_WORDS = {"COUNTY", "CITY", "TOWNSHIP", "DISTRICT", "VILLAGE"}


def slug(en_text: str) -> str:
    """ASCII, upper-case, hyphenated identifier fragment, with the generic
    admin-level word (County/City/Township/District/Village) dropped so
    IDs read as e.g. NANTOU-CAOTUN-YUFENG rather than
    NANTOU-COUNTY-CAOTUN-TOWNSHIP-YUFENG-VILLAGE. The full English name
    with its suffix is still kept in the *_en fields."""
    words = re.sub(r"[^A-Za-z0-9]+", " ", en_text).strip().upper().split()
    if len(words) > 1 and words[-1] in GENERIC_SUFFIX_WORDS:
        words = words[:-1]
    return "-".join(words)


# ---------------------------------------------------------------------------
# 2. Header parsing (locate candidate columns + summary-stat columns)
# ---------------------------------------------------------------------------

# English abbreviations for party names as printed on CEC ballots. Add to
# this as you encounter new parties -- unmapped party names fall back to
# `party_en: None` with a warning rather than a guess (there's no reliable
# way to auto-derive "TPP" from "台灣民眾黨" the way romanize() handles
# place names).
PARTY_EN = {
    "中國國民黨": "KMT",
    "民主進步黨": "DPP",
    "台灣民眾黨": "TPP", "臺灣民眾黨": "TPP",
    "時代力量": "NPP",
    "台灣基進": "TSP", "臺灣基進": "TSP",
    "親民黨": "PFP",
    "新黨": "NP",
    "臺灣團結聯盟": "TSU", "台灣團結聯盟": "TSU",
    "臺聯黨": "TSU", "台聯黨": "TSU",
    "無黨團結聯盟": "NPSU",
    "臺灣綠黨": "GPT", "台灣綠黨": "GPT",
    "小民參政歐巴桑聯盟": "TOPEP",
    "臺灣吾黨": "TNOP", "台灣吾黨": "TNOP",
    "無黨籍": "IND", "無": "IND",
}

# Presidential ballots list a running mate on the second line instead of a
# party (see parse_candidates), so a ticket's party can't be read off the
# sheet -- it has to be hardcoded per election here, keyed by the
# presidential candidate's name. Extend this for each new presidential
# race as you add its data.
# Note that elections 2004 and prior are formatted differently.
# For 2008, use report 6. For 2012 and after, use A05.
# The subdivision level is 3.
#
# Keyed by year (matched against the leading 4 digits of --election-id),
# then by the presidential candidate's name -- NOT just by name, because
# the same person can run under different parties in different years (e.g.
# 宋楚瑜 ran PFP in 2012/2016/2020 but independent in 2000). A flat
# name -> party dict can't represent that: duplicate keys in a dict literal
# silently keep only the last one, which previously made every 宋楚瑜 entry
# resolve to whichever year happened to be listed last.
PRESIDENT_TICKET_PARTY = {
    "2024": {
        "柯文哲": "台灣民眾黨",
        "賴清德": "民主進步黨",
        "侯友宜": "中國國民黨",
    },
    "2020": {
        "宋楚瑜": "親民黨",
        "韓國瑜": "中國國民黨",
        "蔡英文": "民主進步黨",
    },
    "2016": {
        "朱立倫": "中國國民黨",
        "蔡英文": "民主進步黨",
        "宋楚瑜": "親民黨",
    },
    "2012": {
        "蔡英文": "民主進步黨",
        "馬英九": "中國國民黨",
        "宋楚瑜": "親民黨",
    },
    "2008": {
        "謝長廷": "民主進步黨",
        "馬英九": "中國國民黨",
    },
    "2004": {
        "陳水扁": "民主進步黨",
        "連戰": "中國國民黨",
    },
    "2000": {
        "宋楚瑜": "無黨籍",
        "連戰": "中國國民黨",
        "李敖": "新黨",
        "許信良": "無黨籍",
        "陳水扁": "民主進步黨",
    },
    "1996": {
        "陳履安": "無黨籍",
        "李登輝": "中國國民黨",
        "彭明敏": "民主進步黨",
        "林洋港": "無黨籍",
    },
}

STAT_LABELS = {
    "有效票數": "valid_votes",
    "無效票數": "invalid_votes",
    "投票數": "ballots_cast",
    "已領未投票數": "unreturned_ballots",
    "發出票數": "ballots_issued",
    "用餘票數": "unused_ballots",
    "選舉人數": "electors",
    "投票率": "turnout_rate",
}


def clean_header(raw: str) -> str:
    """'投票率H\\nH=C÷G' -> '投票率' ; '無效票數B' -> '無效票數'."""
    first_line = raw.split("\n")[0]
    first_line = re.sub(r"[A-Z]$", "", first_line)
    return first_line.strip()


def find_candidate_columns(ws):
    """Return (start_col, end_col) 1-indexed, inclusive, for the candidate
    vote columns, using the '各組候選人得票情形' merged header in row 2."""
    for rng in ws.merged_cells.ranges:
        top_left = ws.cell(row=rng.min_row, column=rng.min_col).value
        if top_left and "候選人" in str(top_left) and rng.min_row <= 2 <= rng.max_row:
            return rng.min_col, rng.max_col

    # Fallback for files with no merge info at all -- seen in older .xls
    # exports, where xlrd reports zero merged ranges even though the
    # candidate header visually spans several columns. Locate '候選人'
    # directly in row 2, then treat the run of columns after it (up to
    # the next recognized stat-column header, e.g. '有效票數') as the
    # candidate block.
    start_col = None
    max_col = ws.max_column
    for col in range(1, max_col + 1):
        raw = ws.cell(row=2, column=col).value
        if raw and "候選人" in str(raw):
            start_col = col
            break
    if start_col is not None:
        end_col = start_col
        for col in range(start_col + 1, max_col + 1):
            raw = ws.cell(row=2, column=col).value
            if raw and clean_header(str(raw)) in STAT_LABELS:
                break
            end_col = col
        return start_col, end_col

    raise ValueError("Could not locate the candidate header block (looked for "
                      "a merged cell, or else a plain cell, containing '候選人' "
                      "in row 2). This file's layout may differ from the "
                      "expected format.")


def parse_candidates(ws, start_col, end_col, election_type, year=None):
    """Read candidate number + name(s) from row 3.

    Row 3's second line means different things depending on race:
      - president: the VP running mate (not a party) -- CEC ballots don't
        print a presidential ticket's party anywhere in this sheet, so it
        has to come from the hardcoded PRESIDENT_TICKET_PARTY lookup, keyed
        by year since the same person can run under different parties in
        different years (e.g. 宋楚瑜: PFP in 2012/2016/2020, IND in 2000).
      - everything else (legislative, mayor, councillor): the candidate's
        party name (or blank/'無黨籍' for independents), printed directly.
    """
    candidates = {}
    for col in range(start_col, end_col + 1):
        raw = ws.cell(row=3, column=col).value
        if not raw:
            continue
        lines = [l.strip() for l in str(raw).split("\n") if l.strip()]
        m = re.match(r"\((\d+)\)", lines[0])
        number = m.group(1) if m else str(col - start_col + 1)
        name_lines = lines[1:] if m else lines
        name_zh = name_lines[0] if name_lines else ""

        running_mate_zh = None
        if election_type == "president":
            running_mate_zh = name_lines[1] if len(name_lines) > 1 else None
            year_tickets = PRESIDENT_TICKET_PARTY.get(year)
            if year_tickets is None:
                print(f"WARNING: no PRESIDENT_TICKET_PARTY entries for year "
                      f"'{year}' -- add a year block for it.", file=sys.stderr)
                year_tickets = {}
            party_zh = year_tickets.get(name_zh)
            if party_zh is None:
                print(f"WARNING: no party mapping for presidential candidate "
                      f"'{name_zh}' in {year} -- add an entry under "
                      f"PRESIDENT_TICKET_PARTY['{year}'].", file=sys.stderr)
        else:
            party_zh = name_lines[1] if len(name_lines) > 1 else None

        party_en = PARTY_EN.get(party_zh) if party_zh else None
        if party_zh and party_en is None:
            print(f"WARNING: no English abbreviation for party '{party_zh}' -- "
                  f"add an entry to PARTY_EN.", file=sys.stderr)

        candidates[number] = {
            "name_zh": name_zh,
            "running_mate_zh": running_mate_zh,
            "party_zh": party_zh,
            "party_en": party_en,
            "col": col,
        }
    return candidates


def parse_stat_columns(ws, after_col):
    """Map STAT_LABELS keys -> column index, scanning row 2 after the
    candidate block."""
    stat_cols = {}
    max_col = ws.max_column
    for col in range(after_col, max_col + 1):
        raw = ws.cell(row=2, column=col).value
        if not raw:
            continue
        label = clean_header(str(raw))
        if label in STAT_LABELS:
            stat_cols[STAT_LABELS[label]] = col
    return stat_cols


# ---------------------------------------------------------------------------
# 3. Main conversion
# ---------------------------------------------------------------------------

def parse_number(v, kind=int):
    """Convert a vote/stat cell to int/float, tolerating thousands-separator
    commas -- some older (e.g. 2020) .xls exports write counts as strings
    like '12,345' instead of a plain number, which int()/float() reject
    outright."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.replace(",", "").strip()
        if v == "":
            return None
    if kind is int:
        return int(float(v))  # int('12345.0') fails outright; float() first handles it
    return kind(v)


def load_worksheet(path: str):
    """Load the first sheet of an .xls or .xlsx file as an openpyxl Worksheet.

    openpyxl only reads the modern .xlsx/.xlsm format. Legacy CEC .xls files
    are read with xlrd instead and copied cell-by-cell -- including merged
    ranges, since find_candidate_columns() relies on those -- into an
    in-memory openpyxl workbook, so every function below can treat both
    formats identically.
    """
    suffix = Path(path).suffix.lower()

    if suffix in (".xlsx", ".xlsm"):
        wb = openpyxl.load_workbook(path, data_only=True)
        return wb[wb.sheetnames[0]]

    if suffix == ".xls":
        try:
            import xlrd
        except ImportError:
            raise SystemExit(
                "Reading .xls files requires xlrd "
                "(pip install xlrd --break-system-packages)."
            )
        book = xlrd.open_workbook(path)
        sheet = book.sheet_by_index(0)

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet.name

        for r in range(sheet.nrows):
            for c in range(sheet.ncols):
                value = sheet.cell_value(r, c)
                if value == "":
                    continue
                ws.cell(row=r + 1, column=c + 1, value=value)

        # xlrd merged_cells entries are (row_lo, row_hi, col_lo, col_hi),
        # 0-indexed with row_hi/col_hi exclusive -- which lines up exactly
        # with openpyxl's 1-indexed inclusive start/end once row_lo/col_lo
        # get a +1 and row_hi/col_hi are kept as-is.
        for row_lo, row_hi, col_lo, col_hi in sheet.merged_cells:
            ws.merge_cells(start_row=row_lo + 1, start_column=col_lo + 1,
                            end_row=row_hi, end_column=col_hi)

        return ws

    raise ValueError(f"Unsupported file type: {path!r} (expected .xls or .xlsx)")


def convert(xlsx_path: str, election_id: str, election_type: str):
    ws = load_worksheet(xlsx_path)

    county_zh = ws.title.strip()
    if county_zh not in COUNTY_EN and county_zh not in COUNTY_ISO:
        # The sheet tab often isn't the county name -- common in older .xls
        # exports, where it's a numeric CEC code ('307'), a generic default
        # ('Sheet1'), or garbled legacy-encoded text. Fall back to pulling a
        # recognized county/city name out of the row-1 title text instead,
        # e.g. '...在彰化縣各村（里）得票數一覽表' -> '彰化縣'.
        title_text = str(ws.cell(row=1, column=1).value or "")
        fallback = next((name for name in COUNTY_EN if name in title_text), None)
        if fallback:
            county_zh = fallback
        else:
            print(f"WARNING: worksheet tab name '{county_zh}' is not a "
                  f"recognized county/city, and no county name could be found "
                  f"in the row-1 title either -- county_zh/county_en/"
                  f"county_iso will likely be wrong.", file=sys.stderr)
    county_en = romanize(county_zh, "county")
    county_iso = COUNTY_ISO.get(county_zh)
    if county_iso is None:
        print(f"WARNING: no ISO 3166-2:TW code found for '{county_zh}' -- "
              f"falling back to a romanized slug for its identifiers. Check "
              f"COUNTY_ISO for a typo or missing entry.", file=sys.stderr)
        county_iso = slug(county_en)
    county_slug = county_iso

    year_match = re.match(r"(\d{4})", election_id)
    year = year_match.group(1) if year_match else None
    if year is None and election_type == "president":
        print(f"WARNING: could not read a leading 4-digit year out of "
              f"--election-id '{election_id}' -- presidential party lookups "
              f"will fail. --election-id should start with the year, e.g. "
              f"'2024-president'.", file=sys.stderr)

    cand_start, cand_end = find_candidate_columns(ws)
    candidates = parse_candidates(ws, cand_start, cand_end, election_type, year)
    stat_cols = parse_stat_columns(ws, cand_end + 1)

    result = {
        "election_id": election_id,
        "election_type": election_type,
        "source_file": Path(xlsx_path).name,
        "county_zh": county_zh,
        "county_en": county_en,
        "county_iso": county_iso,
        "candidates": {
            num: {
                "name_zh": c["name_zh"],
                "running_mate_zh": c["running_mate_zh"],
                "party_zh": c["party_zh"],
                "party_en": c["party_en"],
            }
            for num, c in candidates.items()
        },
        "county_total": None,
        "townships": {},
        "villages": {},
    }

    current_township_zh = None
    current_township_en = None
    seen_total_row = False

    FULLWIDTH_SPACE = "\u3000"

    for row in range(6, ws.max_row + 1):
        a_val = ws.cell(row=row, column=1).value
        b_val = ws.cell(row=row, column=2).value
        a_val = a_val.replace(FULLWIDTH_SPACE, "").strip() if a_val else ""
        b_val = canonicalize_multi_village(b_val.strip()) if b_val else ""

        if not a_val and not b_val:
            continue  # blank spacer row

        def read_record():
            votes = {}
            for num, c in candidates.items():
                v = ws.cell(row=row, column=c["col"]).value
                votes[num] = parse_number(v, int)
            stats = {}
            for key, col in stat_cols.items():
                v = ws.cell(row=row, column=col).value
                if key == "turnout_rate":
                    stats[key] = parse_number(v, float)
                else:
                    stats[key] = parse_number(v, int)
            return {"votes": votes, **stats}

        if a_val and not b_val:
            if not seen_total_row:
                # first A-filled/B-empty row = county grand total
                result["county_total"] = read_record()
                seen_total_row = True
            else:
                # township/city subtotal row
                current_township_zh = a_val
                current_township_en = romanize(current_township_zh, "township")
                township_id = f"{county_slug}-{slug(current_township_en)}"
                result["townships"][township_id] = {
                    "township_zh": current_township_zh,
                    "township_en": current_township_en,
                    **read_record(),
                }
        elif b_val and not a_val:
            if current_township_zh is None:
                print(f"WARNING: village row '{b_val}' at row {row} found before "
                      f"any township header -- skipping.", file=sys.stderr)
                continue
            village_zh = b_val
            village_en = romanize(village_zh, "village")
            village_id = f"{county_slug}-{slug(current_township_en)}-{slug(village_en)}"
            result["villages"][village_id] = {
                "county_zh": county_zh, "county_en": county_en,
                "township_zh": current_township_zh, "township_en": current_township_en,
                "village_zh": village_zh, "village_en": village_en,
                **read_record(),
            }
        else:
            print(f"WARNING: row {row} has both/neither A and B filled "
                  f"(A={a_val!r}, B={b_val!r}) -- skipping.", file=sys.stderr)

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_xlsx")
    ap.add_argument("output_json")
    ap.add_argument("--election-id", required=True,
                     help="e.g. 2024-president, 2024-legislative-nantou-1")
    ap.add_argument("--election-type", required=True,
                     choices=["president", "legislative", "mayor", "councillor", "referendum"])
    args = ap.parse_args()

    data = convert(args.input_xlsx, args.election_id, args.election_type)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_villages = len(data["villages"])
    n_townships = len(data["townships"])
    print(f"Wrote {args.output_json}: {n_townships} townships, {n_villages} villages.")


if __name__ == "__main__":
    main()
