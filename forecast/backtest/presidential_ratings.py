#!/usr/bin/env python3
"""How pollsters did in presidential elections (2012, 2016, 2020, 2024).

Each poll is reduced to shares among respondents who named a candidate
(undecided removed, the three candidates renormalized to 100), then compared
with the official result:

  margin error  = (DPP - KMT) poll  -  (DPP - KMT) actual      (+ = too DPP)
  candidate err = mean |poll share - actual share| over the three candidates
  vs rivals     = this poll's |margin error| minus the mean |margin error|
                  of the other polls in that election (negative = better)

Writes forecast/presidential_ratings.json.
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
D = json.load(open(os.path.join(HERE, "presidential_polls.json"), encoding="utf-8"))
K = 4  # shrink "vs rivals" toward 0 with weight n / (n + K)


def main():
    rows = []
    for year, polls in D["polls"].items():
        act = D["elections"][year]["actual"]
        tot = act["DPP"] + act["KMT"] + act["third"]
        a = {k: act[k] / tot * 100 for k in ("DPP", "KMT", "third")}
        for name, end, g, b, t, u in polls:
            s = g + b + t
            p = {"DPP": g / s * 100, "KMT": b / s * 100, "third": t / s * 100}
            rows.append({"year": year, "pollster": D["name_map"].get(name, name), "en": name, "end": end,
                         "margin_err": (p["DPP"] - p["KMT"]) - (a["DPP"] - a["KMT"]),
                         "cand_err": float(np.mean([abs(p[k] - a[k]) for k in p])),
                         "third_err": p["third"] - a["third"]})
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r)
    for yr in by_year.values():
        for r in yr:
            others = [abs(o["margin_err"]) for o in yr if o is not r]
            r["vs_rivals"] = abs(r["margin_err"]) - float(np.mean(others)) if others else 0.0

    years = {y: {"n_polls": len(v), "mean_margin_err": float(np.mean([r["margin_err"] for r in v])),
                 "mean_abs_margin_err": float(np.mean([abs(r["margin_err"]) for r in v]))} for y, v in by_year.items()}
    pol = {}
    for name in sorted({r["pollster"] for r in rows}):
        rs = [r for r in rows if r["pollster"] == name]
        n = len(rs)
        pol[name] = {"english": rs[0]["en"], "n_polls": n, "years": sorted({r["year"] for r in rs}),
                     "mean_margin_err": float(np.mean([r["margin_err"] for r in rs])),
                     "mean_abs_margin_err": float(np.mean([abs(r["margin_err"]) for r in rs])),
                     "cand_err": float(np.mean([r["cand_err"] for r in rs])),
                     "vs_rivals": n / (n + K) * float(np.mean([r["vs_rivals"] for r in rs])),
                     "by_year": {y: float(np.mean([r["margin_err"] for r in rs if r["year"] == y]))
                                 for y in sorted({r["year"] for r in rs})}}
    json.dump({"years": years, "pollsters": pol, "source_note": D["_note"]},
              open(os.path.join(FC, "presidential_ratings.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("Average DPP-KMT margin error by election (+ = polls too DPP):")
    for y, v in sorted(years.items()):
        print(f"  {y}: {v['mean_margin_err']:+5.1f} pts  (mean |error| {v['mean_abs_margin_err']:.1f}, {v['n_polls']} polls)")
    print(f"\n{'pollster':26} {'polls':>5} {'years':20} {'margin err':>10} {'|margin err|':>12} {'cand err':>8} {'vs rivals':>9}  by year")
    for k in sorted(pol, key=lambda k: pol[k]["vs_rivals"]):
        v = pol[k]
        by = ", ".join(f"{y}:{e:+.1f}" for y, e in v["by_year"].items())
        print(f"{k:26} {v['n_polls']:>5} {','.join(v['years']):20} {v['mean_margin_err']:>+10.1f} {v['mean_abs_margin_err']:>12.1f} {v['cand_err']:>8.1f} {v['vs_rivals']:>+9.1f}  {by}")


if __name__ == "__main__":
    main()
