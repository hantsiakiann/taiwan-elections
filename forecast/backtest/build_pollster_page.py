#!/usr/bin/env python3
"""Merge mayoral and presidential pollster records into forecast/pollster_page.json
for the Pollster Ratings page.

Combined accuracy = poll-weighted average of
  - mayoral "vs rivals" (two-camp share points), and
  - presidential "vs rivals" / 2 (margin points -> share points),
both negative when the pollster was closer to the result than its competitors.
Accuracy alone grades A-C (thirds of the combined score, pollsters with >= MIN_POLLS
rated polls), so a pollster that simply missed is a C at worst. An F is given only
when forecast/pollster_methods.json marks a pollster's method as flawed, with or
without a track record.
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
sys.path.insert(0, FC)
from pollsters import canon  # noqa: E402

MIN_POLLS = 8
SPONSOR_OVERRIDE = {"民眾黨": "T", "國民黨": "B", "民進黨": "G"}


def main():
    loc = json.load(open(os.path.join(FC, "pollster_ratings.json"), encoding="utf-8"))
    pres = json.load(open(os.path.join(FC, "presidential_ratings.json"), encoding="utf-8"))
    p26 = Counter(canon(p["pollster"]) for p in csv.DictReader(open(os.path.join(FC, "polls_2026.csv"), encoding="utf-8")))
    lean26 = {canon(p["pollster"]): p["sponsor_lean"] for p in csv.DictReader(open(os.path.join(FC, "polls_2026.csv"), encoding="utf-8"))}
    meth = json.load(open(os.path.join(FC, "pollster_methods.json"), encoding="utf-8"))["pollsters"]
    names = set(loc["pollsters"]) | set(pres["pollsters"]) | set(p26)
    rows = []
    for n in names:
        L = loc["pollsters"].get(n)
        P = pres["pollsters"].get(n)
        nl = L["n_compared"] if L else 0
        npr = P["n_polls"] if P else 0
        parts = []
        if L and nl:
            parts.append((L["excess_err_pts"], nl))
        if P:
            parts.append((P["vs_rivals"] / 2, npr))
        score = sum(s * w for s, w in parts) / sum(w for _, w in parts) if parts else None
        years = sorted(set((L or {}).get("years", [])) | set((P or {}).get("years", [])))
        rows.append({
            "name": n, "n_rated": nl + npr, "score": score, "polls_2026": p26.get(n, 0),
            "sponsor": SPONSOR_OVERRIDE.get(n) or lean26.get(n) or (L or {}).get("sponsor_lean") or "N",
            "method": meth.get(n),
            "elections": years,
            "local": None if not L else {"polls": L["n_polls"], "house_pts": L["house_pts"],
                                         "abs_err_pts": L["abs_err_pts"], "vs_rivals_pts": L["excess_err_pts"],
                                         "years": L["years"]},
            "pres": None if not P else {"polls": P["n_polls"], "margin_err": P["mean_margin_err"],
                                        "abs_margin_err": P["mean_abs_margin_err"], "vs_rivals": P["vs_rivals"],
                                        "by_year": P["by_year"]},
        })
    graded = [r for r in rows if r["n_rated"] >= MIN_POLLS and r["score"] is not None]
    cuts = np.percentile([r["score"] for r in graded], [100 / 3, 200 / 3])
    for r in rows:
        if (r["method"] or {}).get("flawed"):
            r["grade"] = "F"
        elif r in graded:
            s = r["score"]
            r["grade"] = "A" if s <= cuts[0] else "B" if s <= cuts[1] else "C"
        else:
            r["grade"] = None
    rows.sort(key=lambda r: (r["grade"] is None, r["grade"] or "", r["score"] if r["score"] is not None else 99, -r["polls_2026"]))
    out = {"min_polls": MIN_POLLS,
           "field_local": loc["year_miss_pts"],
           "field_pres": {y: v["mean_margin_err"] for y, v in pres["years"].items()},
           "pollsters": rows}
    json.dump(out, open(os.path.join(FC, "pollster_page.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in rows[:16]:
        print(r["grade"], r["name"], r["n_rated"], None if r["score"] is None else round(r["score"], 2), r["polls_2026"])
    print(len(rows), "pollsters,", len(graded), "graded")


if __name__ == "__main__":
    main()
