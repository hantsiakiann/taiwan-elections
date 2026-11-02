#!/usr/bin/env python3
"""Extract county-level totals for mayor/magistrate races, party-list
legislative races and presidential races into data/county_totals.json.

Reads the raw CEC files already in data/:
  mayor      : 2005, 2006 (Taipei/Kaohsiung), 2009, 2010, 2014, 2018,
               2022 (+ Chiayi City rerun, Dec 2022)
  party list : 2008, 2012, 2016, 2020, 2024 (*-LEG-ATLARGE-*)
  president  : 2008-2024 (already-parsed per-county JSON)

Totals are computed by summing polling-station rows (rows whose station
column is a number), and cross-checked against the file's own 總計 row
when it has one. Pre-2010 counties that merged into special municipalities
(臺北縣, 臺中縣, 臺南縣, 高雄縣, 桃園縣) are keyed by their current ISO code
via the merge map, so their votes add up under the successor.

Usage:  python forecast/extract_county_totals.py
"""
import glob
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from cec_to_json import COUNTY_ISO  # noqa: E402

MERGE = {"KHQ": "KHH", "TNQ": "TNN", "TXQ": "TXG"}  # 2010 mergers

PARTY_CODE = {
    "民主進步黨": "DPP", "中國國民黨": "KMT", "台灣民眾黨": "TPP",
    "親民黨": "PFP", "新黨": "NP", "台灣團結聯盟": "TSU",
    "時代力量": "NPP", "台灣基進": "TSP", "無黨團結聯盟": "NPSU",
    "綠黨": "GPT", "台灣綠黨": "GPT", "社會民主黨": "SDP",
}
# Party-list camp: G = DPP camp, B = KMT camp, T = TPP, O = other
PL_CAMP = {"DPP": "G", "TSU": "G", "NPP": "G", "TSP": "G",
           "KMT": "B", "PFP": "B", "NP": "B", "TPP": "T"}


def num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def is_station(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return bool(re.fullmatch(r"\d+(\.0)?", str(v).strip()))


def county_from_name(path):
    base = os.path.basename(path)
    m = re.search(r"-([A-Z]{3})\.xlsx?$", base)
    if m:
        return m.group(1)
    # prefer names inside parentheses, e.g. 縣表3-1-302(宜蘭縣)
    m = re.search(r"\(([^)]+[縣市])\)", base)
    cand = [m.group(1)] if m else []
    cand += [k for k in COUNTY_ISO if k in base]
    for c in cand:
        if c in COUNTY_ISO:
            return COUNTY_ISO[c]
    raise ValueError(f"no county in {base}")


def parse_modern(path):
    """2008+ CEC layout: candidate header cells 'N\\nNAME\\nPARTY' (or '(N)\\n\\nPARTY')."""
    df = pd.read_excel(path, header=None, dtype=object)
    hdr_row, cand_cols = None, []
    for r in range(min(8, len(df))):
        cols = [c for c in range(df.shape[1])
                if re.match(r"^\(?\d+\)?\s*\n", str(df.iat[r, c]))]
        if len(cols) >= 1:
            hdr_row, cand_cols = r, cols
            break
        # 2008 party-list layout: '(1)' in one row, party name two rows below
        cols = [c for c in range(df.shape[1]) if re.fullmatch(r"\(\d+\)", str(df.iat[r, c]).strip())]
        if cols:
            for c in cols:
                below = next(str(df.iat[k, c]) for k in range(r + 1, r + 4)
                             if str(df.iat[k, c]) != "nan")
                df.iat[r, c] = f"{df.iat[r, c]}\n{below}"
            hdr_row, cand_cols = r, cols
            break
    if hdr_row is None:
        raise ValueError(f"no candidate header in {path}")
    cands = []
    for c in cand_cols:
        parts = [p.strip() for p in str(df.iat[hdr_row, c]).split("\n")]
        parts = [p for p in parts if p]
        number = re.sub(r"\D", "", parts[0])
        if len(parts) == 2:   # '(6)\n\n民主進步黨' (party list) or 'N\nNAME\n ' (independent)
            name = parts[1]
            party = parts[1] if (parts[1] in PARTY_CODE or parts[1].endswith(("黨", "聯盟", "連線", "會議"))) else "無黨籍"
        elif len(parts) >= 3:
            name, party = parts[1], parts[2]
        else:
            name, party = parts[1] if len(parts) > 1 else "", ""
        if party in ("無", "", "nan"):
            party = "無黨籍"
        cands.append({"no": number, "name": name, "party_zh": party,
                      "party": PARTY_CODE.get(party, "IND" if party == "無黨籍" else "OTH")})
    # station column = 3rd column (index 2) in every modern file
    votes = [0.0] * len(cand_cols)
    total_row = None
    for r in range(hdr_row + 1, len(df)):
        first = str(df.iat[r, 0]).replace("　", "").replace(" ", "")
        if first in ("總計", "合計"):
            total_row = [num(df.iat[r, c]) or 0 for c in cand_cols]
        if is_station(df.iat[r, 2]):
            for i, c in enumerate(cand_cols):
                votes[i] += num(df.iat[r, c]) or 0
    if total_row and any(abs(a - b) > 0.5 for a, b in zip(total_row, votes)):
        print(f"  NOTE {os.path.basename(path)}: station sum {votes[:3]} != 總計 {total_row[:3]}; using 總計",
              file=sys.stderr)
        votes = total_row
    for cd, v in zip(cands, votes):
        cd["votes"] = int(round(v))
    return cands


def parse_2005(path):
    df = pd.read_excel(path, header=None, dtype=object)
    hdr = [str(x) for x in df.iloc[1].tolist()]
    cols = [(i, h.replace("得票數", "").strip()) for i, h in enumerate(hdr) if h.startswith("得票數")]
    out = []
    for no, (i, name) in enumerate(cols, 1):
        v = sum(num(x) or 0 for x in df.iloc[2:, i])
        out.append({"no": str(no), "name": name, "party_zh": None, "party": None, "votes": int(v)})
    return out


def parse_tsv_total(path, parties):
    with open(path, "rb") as f:
        lines = f.read().decode("big5", errors="replace").splitlines()
    hdr = lines[1].split("\t")
    tot = next(l for l in lines if l.replace("　", "").startswith("總計")).split("\t")
    out = []
    for i in range(3, len(hdr)):
        name = hdr[i].replace(" ", "").replace("　", "")
        if name.startswith("有效票"):
            break
        out.append({"no": str(i - 2), "name": name, "party_zh": None,
                    "party": parties.get(name, "IND"), "votes": int(num(tot[i]))})
    return out


def parse_chiayi_2022_rerun(path):
    df = pd.read_excel(path, header=None, dtype=object)
    r = next(i for i in range(len(df)) if str(df.iat[i, 0]).strip() == "總計")
    out = []
    for c in range(2, 7):
        party = str(df.iat[5, c]).strip()
        party = "無黨籍" if party == "無" else party
        out.append({"no": str(df.iat[3, c]), "name": str(df.iat[4, c]).strip(), "party_zh": party,
                    "party": PARTY_CODE.get(party, "IND"), "votes": int(num(df.iat[r, c]))})
    return out


# 2005 files carry no party; hand-labelled (major candidates only; rest IND).
PARTY_2005 = {
    "林宗男": "IND", "李朝卿": "KMT", "蔡煌瑯": "DPP",
    "陳麗貞": "DPP", "黃敏惠": "KMT", "陳明文": "DPP", "翁重鈞": "KMT",
    "許財利": "KMT", "陳建銘": "DPP", "呂國華": "KMT", "陳定南": "DPP",
    "曹啟鴻": "DPP", "王進士": "KMT", "卓伯源": "KMT", "翁金珠": "DPP",
    "林政則": "KMT", "鄭貴蓮": "DPP", "鄭永金": "KMT", "邱太三": "DPP",
    "朱立倫": "KMT", "鄭寶清": "DPP", "王乾發": "KMT", "許素葉": "DPP",
    "胡志強": "KMT", "林佳龍": "DPP", "黃秀孟": "KMT", "邱垂貞": "DPP",
    "周錫瑋": "KMT", "羅文嘉": "DPP", "許添財": "DPP", "陳榮盛": "KMT",
    "蘇煥智": "DPP", "郭添財": "KMT", "鄺麗貞": "KMT", "劉櫂豪": "DPP",
    "謝深山": "KMT", "游盈隆": "DPP", "劉政鴻": "KMT", "楊長鎮": "DPP",
    "陳雪生": "PFP", "李炷烽": "NP", "張福興": "IND", "蘇治芬": "DPP",
    "張麗善": "IND", "楊秋興": "DPP", "林益世": "KMT", "陳秀卿": "IND",
    "吳威志": "IND", "邱鏡淳": "IND", "林光華": "DPP", "劉增應": "KMT",
    "黃仲生": "KMT", "鄭貴元": "DPP", "盧博基": "DPP", "陳光復": "DPP",
    "陳明振": "KMT", "許舒博": "KMT", "邱炳坤": "DPP", "劉文雄": "PFP",
}
PARTY_2006 = {"郝龍斌": "KMT", "謝長廷": "DPP", "宋楚瑜": "IND", "李敖": "IND",
              "周玉蔻": "TSU", "柯賜海": "IND", "黃俊英": "KMT", "陳菊": "DPP",
              "林志昇": "IND", "羅志明": "TSU", "林景元": "IND"}


def add(store, key, iso, cands, ):
    iso = MERGE.get(iso, iso)
    race = store.setdefault(key, {}).setdefault(iso, {"candidates": []})
    race["candidates"].extend(cands)


def main():
    out = {"mayor": {}, "partylist": {}, "president": {}}

    # ---- mayor ---------------------------------------------------------
    for f in sorted(glob.glob(os.path.join(DATA, "2005-mayor", "*.xls"))):
        cands = parse_2005(f)
        for c in cands:
            c["party"] = PARTY_2005.get(c["name"], "IND")
        add(out["mayor"], "2005", county_from_name(f), cands)
    d6 = os.path.join(DATA, "2006-mayor")
    add(out["mayor"], "2006", "TPE", parse_tsv_total(
        os.path.join(d6, "第04屆台北市長選舉得票概況", "各行政區開票結果表.xls"), PARTY_2006))
    add(out["mayor"], "2006", "KHH", parse_tsv_total(
        os.path.join(d6, "第04屆高雄市長選舉得票概況", "高雄市第4屆市長選舉各行政區開票結果表.xls"), PARTY_2006))
    for yr, pats in [("2009", ["2009-mayor/*.xls"]), ("2010", ["2010-mayor/*.xls"]),
                     ("2014", ["2014-mayor/*/*.xls"]), ("2018", ["2018-mayor/*/*.xls"]),
                     ("2022", ["2022-mayor/*/*.xls"])]:
        for p in pats:
            for f in sorted(glob.glob(os.path.join(DATA, p))):
                add(out["mayor"], yr, county_from_name(f), parse_modern(f))
    add(out["mayor"], "2022", "CYI", parse_chiayi_2022_rerun(
        os.path.join(DATA, "2022-mayor", "嘉義市第11屆市長重行選舉各村里得票數一覽表.xls")))

    # ---- party list ----------------------------------------------------
    for yr in ("2008", "2012", "2016", "2020", "2024"):
        for f in sorted(glob.glob(os.path.join(DATA, f"{yr}-legislature", f"{yr}-LEG-ATLARGE-*.xls*"))):
            iso = MERGE.get(county_from_name(f), county_from_name(f))
            cands = parse_modern(f)
            agg = out["partylist"].setdefault(yr, {}).setdefault(iso, {})
            for c in cands:
                p = PARTY_CODE.get(c["name"], "OTH")
                agg[p] = agg.get(p, 0) + c["votes"]

    # ---- president (already-parsed JSON) --------------------------------
    pres_camp = {"DPP": "G", "KMT": "B", "PFP": "B", "TPP": "T"}
    for yr in ("2008", "2012", "2016", "2020", "2024"):
        for f in sorted(glob.glob(os.path.join(DATA, f"{yr}-president", "*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            iso = MERGE.get(d["county_iso"], d["county_iso"])
            agg = out["president"].setdefault(yr, {}).setdefault(iso, {})
            for no, cd in d["candidates"].items():
                party = cd.get("party_en") or PARTY_CODE.get(cd.get("party_zh"), "OTH")
                agg[party] = agg.get(party, 0) + d["county_total"]["votes"][no]

    # merge duplicate candidates from merged counties (2005 TPQ+TXG etc. are
    # separate races, so keep them as separate candidate lists but tag source)
    with open(os.path.join(DATA, "county_totals.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for kind in out:
        print(kind, {y: len(v) for y, v in out[kind].items()})


if __name__ == "__main__":
    main()
