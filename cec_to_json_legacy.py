#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cec_to_json_legacy.py
----------------------
Convert a Taiwan Central Election Commission (中選會) presidential-election
result file from 1996, 2000, or 2004 -- the only years that use an older,
polling-station-level layout -- into the same per-county JSON structure
that cec_to_json.py produces for later years.

USAGE
    python cec_to_json_legacy.py INPUT.xls OUTPUT_DIR --election-id 2004-president

    Writes one {year}-PRES-{ISO}.json file per county into OUTPUT_DIR
    (e.g. 2004-PRES-CHA.json), matching the modern per-county file naming.

WHY THIS IS A SEPARATE SCRIPT
    cec_to_json.py expects one file per county, with headers on fixed rows
    and one data row per township/village that CEC has already totaled.
    These three years' files are shaped completely differently:
      - The WHOLE COUNTRY is in a single sheet, not one file per county.
      - Each data row is one polling station (投開票所), not a
        township/village total -- rows have to be summed into village,
        then township, then county totals here.
      - There are two distinct sub-layouts across these three years (see
        detect_layout() below):
          * 1996 and 2004: title row, then headers, then a row of bare
            ballot numbers, then a row with each ticket's names -- but
            with the VP's name printed BEFORE the president's, run
            together with either no separator or a long run of spaces
            (e.g. "呂秀蓮陳水扁", "宋楚瑜<100+ spaces>連 戰"). Not safely
            splittable by string parsing alone in general.
          * 2000: a single flat header row; candidate columns are just
            labeled "1號得票數" / "2號得票數" / etc, with no candidate
            names anywhere in the file at all.
    Because names can't be reliably parsed (1996/2004) or aren't present
    at all (2000), which ticket is ballot number N in a given year comes
    from the hardcoded LEGACY_PRESIDENT_TICKETS table below, not from the
    spreadsheet text. See the comment above that table for how it was
    verified against each file's own vote totals.

REQUIREMENTS
    Same as cec_to_json.py: openpyxl, pypinyin; xlrd additionally for .xls
    input (all three of these years' files are .xls).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from cec_to_json import (
    COUNTY_EN,
    COUNTY_ISO,
    PARTY_EN,
    PRESIDENT_TICKET_PARTY,
    STAT_LABELS,
    canonicalize_multi_village,
    load_worksheet,
    parse_number,
    romanize,
    slug,
)

# ---------------------------------------------------------------------------
# 1. Ballot-number order for each legacy year
# ---------------------------------------------------------------------------

# (president_name, vp_name) per year, in ballot-number order (index 0 =
# ballot #1). Hardcoded rather than read from the sheet because:
#   - 1996/2004 print the VP's name before the president's, run together
#     with no reliable separator -- see the module docstring.
#   - 2000 doesn't print candidate names anywhere in the file at all;
#     ballot columns are just labeled "1號得票數", "2號得票數", etc.
# Verified against each file's own vote totals: summing every row's value
# in a given ballot column, nationwide, reproduces that ticket's official
# recorded vote total. For 2000 specifically, columns 1-5 sum to exactly
# 4,664,972 / 2,925,513 / 16,782 / 79,429 / 4,977,697, matching the
# published results for 宋楚瑜 / 連戰 / 李敖 / 許信良 / 陳水扁 in that
# order (zh.wikipedia.org/zh-tw/2000年中華民國總統選舉). 1996 and 2004
# were cross-checked the same way, and additionally against the (garbled
# but present) name text in those two years' files.
#
# Party comes from PRESIDENT_TICKET_PARTY (keyed by year, then by
# president name) so it's defined once, not duplicated here.
LEGACY_PRESIDENT_TICKETS = {
    "1996": [
        ("陳履安", "王清峰"),
        ("李登輝", "連戰"),
        ("彭明敏", "謝長廷"),
        ("林洋港", "郝柏村"),
    ],
    "2000": [
        ("宋楚瑜", "張昭雄"),
        ("連戰", "蕭萬長"),
        ("李敖", "馮滬祥"),
        ("許信良", "朱惠良"),
        ("陳水扁", "呂秀蓮"),
    ],
    "2004": [
        ("陳水扁", "呂秀蓮"),
        ("連戰", "宋楚瑜"),
    ],
}


# ---------------------------------------------------------------------------
# 2. Layout detection
# ---------------------------------------------------------------------------

# The 6 non-candidate stat columns, in the fixed order they appear in
# after the candidate block in the 1996/2004 layout. Note this layout's
# header text for these doesn't exactly match STAT_LABELS (e.g. "有效票"
# instead of "有效票數", and electors' header is a garbled, merged-looking
# "(原領票數）　　　選舉人數") -- so these are matched loosely, by
# substring, purely to sanity-check the positional assignment below rather
# than to drive it.
LEGACY_STAT_FIELD_ORDER = [
    ("valid_votes", "有效票"),
    ("invalid_votes", "無效票"),
    ("ballots_cast", "投票數"),
    ("unreturned_ballots", "已領未投票"),
    ("electors", "選舉人數"),
    ("turnout_rate", "投票率"),
]


def detect_layout(ws):
    """Identify which of the two legacy sub-layouts this file uses, and
    return a normalized column map:
        {
            "county_col", "township_col", "village_col": int,
            "candidate_cols": {ballot_number: votes_column},
            "stat_cols": {stat_field: column},  # turnout_rate excluded --
                                                 # recomputed, not summed
            "data_start_row": int,
        }
    so the rest of the script doesn't need to know which layout it's
    reading from.
    """
    row1 = str(ws.cell(row=1, column=1).value or "").strip()
    if row1 == "縣市別":
        return _detect_layout_2000(ws)
    return _detect_layout_1996_2004(ws)


def _detect_layout_1996_2004(ws):
    """Title row, header row 2, ballot-number row 3, candidate-name row 4,
    得票數/得票率 sub-header row 5, data from row 6."""
    header_row = 2
    if str(ws.cell(row=header_row, column=1).value or "").strip() != "縣市別":
        raise ValueError("Unrecognized legacy layout: expected '縣市別' in "
                          "row 1 or row 2, column 1.")

    max_col = ws.max_column
    start_col = None
    for col in range(1, max_col + 1):
        raw = ws.cell(row=header_row, column=col).value
        if raw and "候選人" in str(raw):
            start_col = col
            break
    if start_col is None:
        raise ValueError(f"Could not find the '候選人' header in row "
                          f"{header_row}.")

    # Candidate block: consecutive (votes, pct) column pairs. The
    # ballot-number row has a number under each candidate's votes column
    # and is blank under its pct column and everywhere past the block.
    candidate_cols = {}
    col = start_col
    number = 1
    while True:
        v = ws.cell(row=header_row + 1, column=col).value
        if v is None or str(v).strip() == "":
            break
        candidate_cols[number] = col
        number += 1
        col += 2
    if not candidate_cols:
        raise ValueError("Found the candidate header, but no ballot "
                          "numbers underneath it.")

    stat_cols = {}
    col = col  # first column after the last candidate's pct column
    for field, label in LEGACY_STAT_FIELD_ORDER:
        header = str(ws.cell(row=header_row, column=col).value or "")
        if label not in header:
            print(f"WARNING: expected a '{label}' column at column {col} "
                  f"but found {header!r} -- stat columns may be "
                  f"misaligned.", file=sys.stderr)
        if field != "turnout_rate":  # recomputed after aggregation instead
            stat_cols[field] = col
        col += 1

    return {
        "county_col": 1, "township_col": 2, "village_col": 3,
        "candidate_cols": candidate_cols,
        "stat_cols": stat_cols,
        "data_start_row": header_row + 4,
    }


def _detect_layout_2000(ws):
    """Single flat header row; candidate columns labeled 'N號得票數' /
    'N號得票率'; data from row 2. No candidate names in the file at all --
    see LEGACY_PRESIDENT_TICKETS."""
    candidate_cols = {}
    stat_cols = {}
    max_col = ws.max_column
    for col in range(1, max_col + 1):
        header = str(ws.cell(row=1, column=col).value or "").strip()
        m = re.match(r"^(\d+)號得票數$", header)
        if m:
            candidate_cols[int(m.group(1))] = col
            continue
        field = STAT_LABELS.get(header)
        if field and field != "turnout_rate":  # recomputed, not summed
            stat_cols[field] = col
    if not candidate_cols:
        raise ValueError("Could not find any 'N號得票數' columns in row 1.")

    return {
        "county_col": 1, "township_col": 2, "village_col": 3,
        "candidate_cols": candidate_cols,
        "stat_cols": stat_cols,
        "data_start_row": 2,
    }


# ---------------------------------------------------------------------------
# 3. Main conversion
# ---------------------------------------------------------------------------

def convert_legacy(input_path: str, election_id: str):
    """Return {county_iso: result_dict}, one entry per county found in the
    file, each in the same shape convert() in cec_to_json.py produces."""
    year_match = re.match(r"(\d{4})", election_id)
    if not year_match:
        raise ValueError(f"--election-id '{election_id}' must start with "
                          f"a 4-digit year, e.g. '2004-president'.")
    year = year_match.group(1)

    tickets = LEGACY_PRESIDENT_TICKETS.get(year)
    if tickets is None:
        raise ValueError(f"No LEGACY_PRESIDENT_TICKETS entry for year "
                          f"'{year}' -- add the ballot-number order for "
                          f"that year's tickets.")

    ws = load_worksheet(input_path)
    layout = detect_layout(ws)

    candidates = {}
    for number, (pres_name, vp_name) in enumerate(tickets, start=1):
        party_zh = PRESIDENT_TICKET_PARTY.get(year, {}).get(pres_name)
        if party_zh is None:
            print(f"WARNING: no party mapping for presidential candidate "
                  f"'{pres_name}' in {year} -- add an entry under "
                  f"PRESIDENT_TICKET_PARTY['{year}'].", file=sys.stderr)
        party_en = PARTY_EN.get(party_zh) if party_zh else None
        if party_zh and party_en is None:
            print(f"WARNING: no English abbreviation for party "
                  f"'{party_zh}' -- add an entry to PARTY_EN.",
                  file=sys.stderr)
        candidates[str(number)] = {
            "name_zh": pres_name,
            "running_mate_zh": vp_name,
            "party_zh": party_zh,
            "party_en": party_en,
        }

    file_ballot_numbers = set(layout["candidate_cols"])
    table_ballot_numbers = {n for n in range(1, len(tickets) + 1)}
    if file_ballot_numbers != table_ballot_numbers:
        print(f"WARNING: LEGACY_PRESIDENT_TICKETS['{year}'] lists ballot "
              f"numbers {sorted(table_ballot_numbers)} but the file has "
              f"{sorted(file_ballot_numbers)} -- these should match.",
              file=sys.stderr)

    def new_bucket():
        return {
            "votes": {num: 0 for num in candidates},
            "valid_votes": 0, "invalid_votes": 0, "ballots_cast": 0,
            "unreturned_ballots": 0, "electors": 0,
        }

    counties = {}    # county_zh -> bucket
    townships = {}   # (county_zh, township_zh) -> bucket
    villages = {}    # (county_zh, township_zh, village_zh) -> bucket

    for row in range(layout["data_start_row"], ws.max_row + 1):
        county_zh = str(ws.cell(row=row, column=layout["county_col"]).value or "").strip()
        if not county_zh:
            continue  # blank spacer row, if any
        township_zh = str(ws.cell(row=row, column=layout["township_col"]).value or "").strip()
        village_zh = canonicalize_multi_village(
            str(ws.cell(row=row, column=layout["village_col"]).value or "").strip()
        )

        row_votes = {
            str(num): parse_number(ws.cell(row=row, column=col).value, int) or 0
            for num, col in layout["candidate_cols"].items()
        }
        row_stats = {
            field: parse_number(ws.cell(row=row, column=col).value, int) or 0
            for field, col in layout["stat_cols"].items()
        }

        for key, bucket_map in (
            (county_zh, counties),
            ((county_zh, township_zh), townships),
            ((county_zh, township_zh, village_zh), villages),
        ):
            bucket = bucket_map.setdefault(key, new_bucket())
            for num, v in row_votes.items():
                bucket["votes"][num] += v
            for field, v in row_stats.items():
                bucket[field] += v

    def finalize(bucket):
        bucket = dict(bucket)
        electors = bucket["electors"]
        bucket["turnout_rate"] = (
            round(bucket["ballots_cast"] / electors * 100, 4) if electors else None
        )
        return bucket

    results = {}
    for county_zh, county_bucket in counties.items():
        county_en = romanize(county_zh, "county")
        county_iso = COUNTY_ISO.get(county_zh)
        if county_iso is None:
            print(f"WARNING: no ISO 3166-2:TW code found for '{county_zh}' "
                  f"-- falling back to a romanized slug for its "
                  f"identifiers. Check COUNTY_ISO for a typo or missing "
                  f"entry.", file=sys.stderr)
            county_iso = slug(county_en)

        result = {
            "election_id": election_id,
            "election_type": "president",
            "source_file": Path(input_path).name,
            "county_zh": county_zh,
            "county_en": county_en,
            "county_iso": county_iso,
            "candidates": candidates,
            "county_total": finalize(county_bucket),
            "townships": {},
            "villages": {},
        }

        for (c_zh, t_zh), bucket in townships.items():
            if c_zh != county_zh:
                continue
            t_en = romanize(t_zh, "township")
            township_id = f"{county_iso}-{slug(t_en)}"
            result["townships"][township_id] = {
                "township_zh": t_zh, "township_en": t_en,
                **finalize(bucket),
            }

        for (c_zh, t_zh, v_zh), bucket in villages.items():
            if c_zh != county_zh:
                continue
            t_en = romanize(t_zh, "township")
            v_en = romanize(v_zh, "village")
            village_id = f"{county_iso}-{slug(t_en)}-{slug(v_en)}"
            result["villages"][village_id] = {
                "county_zh": county_zh, "county_en": county_en,
                "township_zh": t_zh, "township_en": t_en,
                "village_zh": v_zh, "village_en": v_en,
                **finalize(bucket),
            }

        results[county_iso] = result

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_xls")
    ap.add_argument("output_dir")
    ap.add_argument("--election-id", required=True,
                     help="e.g. 2004-president -- must start with the "
                          "4-digit year")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = convert_legacy(args.input_xls, args.election_id)
    year = re.match(r"(\d{4})", args.election_id).group(1)

    for county_iso, data in results.items():
        out_path = out_dir / f"{year}-PRES-{county_iso}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}: {len(data['townships'])} townships, "
              f"{len(data['villages'])} villages.")


if __name__ == "__main__":
    main()
