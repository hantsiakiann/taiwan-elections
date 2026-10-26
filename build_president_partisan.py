#!/usr/bin/env python3
"""Bundle the data/XXXX-president/*.json township results into a single file
(data/president_partisan.json) that the president-swing web tool consumes.

Handles the 2010 municipality mergers so every township/county has one stable
key across all eight presidential elections (1996-2024):

    KHQ-*  -> KHH-*      (Kaohsiung County folded into Kaohsiung City)
    TNQ-*  -> TNN-*      (Tainan County  -> Tainan City)
    TXQ-*  -> TXG-*      (Taichung County -> Taichung City)
    KHQ-SANMIN (三民鄉) -> KHH-NAMAXIA   (renamed 2008, kept distinct from the
                                          urban 三民區 = KHH-SANMIN)
    TNN-XI + TNN-ZHONG (西區, 中區, 1996/2000) -> TNN-ZHONGXI (中西區, merged 2004)
"""
import json
import glob
import os
import sys
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
YEARS = [1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]

COUNTY_MERGE = {"KHQ": "KHH", "TNQ": "TNN", "TXQ": "TXG"}

# camp assignment per (year, candidate number): G = Pan-Green, B = Pan-Blue,
# O = Other / non-aligned (dropped from the two-camp lean calculation).
CAMP = {
    "1996-1": "O", "1996-2": "B", "1996-3": "G", "1996-4": "B",
    "2000-1": "B", "2000-2": "B", "2000-3": "B", "2000-4": "O", "2000-5": "G",
    "2004-1": "G", "2004-2": "B",
    "2008-1": "G", "2008-2": "B",
    "2012-1": "G", "2012-2": "B", "2012-3": "B",
    "2016-1": "B", "2016-2": "G", "2016-3": "B",
    "2020-1": "B", "2020-2": "B", "2020-3": "G",
    "2024-1": "O", "2024-2": "G", "2024-3": "B",
}

# three-force bucket per (year, candidate number), for the DPP / KMT / TPP lean:
#   D = DPP camp, K = KMT camp (incl. New Party AND James Soong every time he ran),
#   T = TPP camp (only Ko Wen-je, 2024), X = excluded (non-aligned independents).
# So 1996-2020 this view is a plain DPP-vs-KMT (Green-vs-Blue) lean; only 2024 is
# genuinely three-way.
THIRD = {
    "1996-1": "X", "1996-2": "K", "1996-3": "D", "1996-4": "K",
    "2000-1": "K", "2000-2": "K", "2000-3": "K", "2000-4": "X", "2000-5": "D",
    "2004-1": "D", "2004-2": "K",
    "2008-1": "D", "2008-2": "K",
    "2012-1": "D", "2012-2": "K", "2012-3": "K",
    "2016-1": "K", "2016-2": "D", "2016-3": "K",
    "2020-1": "K", "2020-2": "K", "2020-3": "D",
    "2024-1": "T", "2024-2": "D", "2024-3": "K",
}

# candidates the user can reclassify from the UI (id -> affected year/number)
TOGGLE = {
    "1996-1": "chenlian1996",   # 陳履安  Other <-> Blue
    "2000-4": "hsu2000",        # 許信良  Other <-> Green
    "2024-1": "ko2024",         # 柯文哲  Other <-> Blue / Green
}

# hand romanisation for the English UI (source JSON has zh names only)
NAME_EN = {
    "陳履安": "Chen Li-an", "李登輝": "Lee Teng-hui", "彭明敏": "Peng Ming-min",
    "林洋港": "Lin Yang-kang", "宋楚瑜": "James Soong", "連戰": "Lien Chan",
    "李敖": "Li Ao", "許信良": "Hsu Hsin-liang", "陳水扁": "Chen Shui-bian",
    "謝長廷": "Frank Hsieh", "馬英九": "Ma Ying-jeou", "蔡英文": "Tsai Ing-wen",
    "朱立倫": "Eric Chu", "韓國瑜": "Han Kuo-yu", "柯文哲": "Ko Wen-je",
    "賴清德": "Lai Ching-te", "侯友宜": "Hou Yu-ih",
}
MATE_EN = {
    "王清峰": "Wang Ching-feng", "連戰": "Lien Chan", "謝長廷": "Frank Hsieh",
    "郝柏村": "Hau Pei-tsun", "張昭雄": "Chang Chao-hsiung", "蕭萬長": "Vincent Siew",
    "馮滬祥": "Feng Hu-hsiang", "朱惠良": "Chu Hui-liang", "呂秀蓮": "Annette Lu",
    "蘇貞昌": "Su Tseng-chang", "蘇嘉全": "Su Chia-chyuan", "吳敦義": "Wu Den-yih",
    "林瑞雄": "Lin Rui-hsiung", "陳建仁": "Chen Chien-jen", "徐欣瑩": "Hsu Hsin-ying",
    "余湘": "Sandra Yu", "張善政": "Chang San-cheng", "賴清德": "Lai Ching-te",
    "吳欣盈": "Cynthia Wu", "蕭美琴": "Hsiao Bi-khim", "趙少康": "Jaw Shaw-kong",
}


def canon(twkey):
    """pre-merger township key -> stable canonical key + canonical county iso."""
    pfx, rest = twkey.split("-", 1)
    if twkey == "KHQ-SANMIN":
        return "KHH-NAMAXIA", "KHH"
    if twkey in ("TNN-XI", "TNN-ZHONG"):
        return "TNN-ZHONGXI", "TNN"
    cpfx = COUNTY_MERGE.get(pfx, pfx)
    return cpfx + "-" + rest, cpfx


def add_votes(dst, src):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + (v or 0)


def main():
    candidates_by_year = {}
    national_by_year = {}
    counties = {}
    counties_by_year = {}
    tw = {}          # canon key -> {zh,en,county, rows:{year:{...}}}

    for year in YEARS:
        files = sorted(glob.glob(os.path.join(ROOT, "data", f"{year}-president", "*.json")))
        if not files:
            print(f"!! no files for {year}", file=sys.stderr)
            continue
        ystr = str(year)
        nat_cand, nat_valid = {}, 0
        cby = {}

        first = json.load(open(files[0], encoding="utf-8"))
        clist = []
        for n, c in first["candidates"].items():
            key = f"{year}-{n}"
            clist.append({
                "n": n,
                "name": c.get("name_zh"),
                "name_en": NAME_EN.get(c.get("name_zh"), c.get("name_zh")),
                "mate": c.get("running_mate_zh"),
                "mate_en": MATE_EN.get(c.get("running_mate_zh"), c.get("running_mate_zh")),
                "party_zh": c.get("party_zh"),
                "party": c.get("party_en"),
                "camp": CAMP[key],
                "tf": THIRD[key],
                "toggle": TOGGLE.get(key),
            })
        clist.sort(key=lambda x: int(x["n"]))
        candidates_by_year[ystr] = clist

        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            ciso = d["county_iso"]
            cc = COUNTY_MERGE.get(ciso, ciso)
            # overwrite each year so the newest (2024) name wins -> modern names
            # e.g. NWT = "New Taipei City" not the 1996 "Taipei County"
            counties[cc] = {"zh": d["county_zh"].replace("﻿", "").strip(),
                            "en": d["county_en"].strip()}
            # county totals (authoritative for the county view)
            ct = d["county_total"]
            slot = cby.setdefault(cc, {"valid": 0, "electors": 0, "cast": 0, "cand": {}})
            slot["valid"] += ct.get("valid_votes", 0) or 0
            slot["electors"] += ct.get("electors", 0) or 0
            slot["cast"] += ct.get("ballots_cast", 0) or 0
            add_votes(slot["cand"], ct.get("votes", {}))
            add_votes(nat_cand, ct.get("votes", {}))
            nat_valid += ct.get("valid_votes", 0) or 0

            for rawkey, t in d["townships"].items():
                ckey, ccty = canon(rawkey)
                rec = tw.setdefault(ckey, {"zh": None, "en": None, "county": ccty,
                                           "rows": {}})
                rec["zh"] = t.get("township_zh")
                rec["en"] = t.get("township_en")
                rec["county"] = ccty
                row = rec["rows"].setdefault(ystr, {
                    "valid": 0, "electors": 0, "cast": 0, "cand": {}})
                row["valid"] += t.get("valid_votes", 0) or 0
                row["electors"] += t.get("electors", 0) or 0
                row["cast"] += t.get("ballots_cast", 0) or 0
                add_votes(row["cand"], t.get("votes", {}))

        for cc, s in cby.items():
            s["turnout"] = round(s["cast"] / s["electors"] * 100, 2) if s["electors"] else None
            s.pop("cast", None)
        counties_by_year[ystr] = cby
        national_by_year[ystr] = {"valid": nat_valid, "cand": nat_cand}

    for rec in tw.values():
        for row in rec["rows"].values():
            row["turnout"] = round(row["cast"] / row["electors"] * 100, 2) if row["electors"] else None
            row.pop("cast", None)

    townships = [dict(key=k, **v) for k, v in sorted(tw.items())]

    bundle = {
        "meta": {
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": "Central Election Commission results, parsed by cec_to_json.py",
            "note": "Township/county keys normalized across the 2010 municipality mergers.",
        },
        "elections": YEARS,
        "camps": {"G": "Pan-Green", "B": "Pan-Blue", "O": "Other / non-aligned"},
        "forces": {"D": "DPP camp", "K": "KMT camp", "T": "TPP camp",
                   "X": "excluded"},
        "toggles": {
            "chenlian1996": {"label": "Chen Li-an (1996)", "options": ["O", "B"],
                             "default": "O"},
            "hsu2000": {"label": "Hsu Hsin-liang (2000)", "options": ["O", "G"],
                        "default": "O"},
            "ko2024": {"label": "Ko Wen-je / TPP (2024)", "options": ["O", "B", "G"],
                       "default": "O"},
        },
        "candidatesByYear": candidates_by_year,
        "nationalByYear": national_by_year,
        "counties": counties,
        "countiesByYear": counties_by_year,
        "townships": townships,
    }

    out = os.path.join(ROOT, "data", "president_partisan.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out}")
    print(f"  {len(townships)} townships, {len(counties)} counties, {len(YEARS)} elections")
    print(f"  {os.path.getsize(out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
