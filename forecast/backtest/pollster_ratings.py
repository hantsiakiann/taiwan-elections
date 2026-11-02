#!/usr/bin/env python3
"""Pollster ratings from past mayor/magistrate elections (2014, 2018, 2022).

Every final-matchup poll fielded in the last WINDOW days before an election
is put on the same basis the model uses: the DPP share of the two main camps,
with undecided respondents allocated toward the county's fundamentals-based
lean. Races where a third candidate took over 20% are skipped.

For each poll:
  error  = logit(poll) - logit(actual result)            (+ = too DPP)
  rel    = error - mean error of the OTHER polls of that race
           (how the pollster differed from its competitors on the same race,
           which removes misses every pollster shared)

For each pollster:
  house    = mean rel, shrunk toward 0 with weight n / (n + K)
  scatter  = SD of rel around its house effect, shrunk toward the field
  abs_err  = mean |error| against the actual result, in points
  excess   = mean of |error| minus the mean |error| of the other polls of the
             same race (negative = closer to the result than its competitors),
             shrunk toward 0
  grade    = A-F by excess error, only with >= MIN_GRADED compared polls

Usage: python forecast/backtest/pollster_ratings.py
Writes forecast/pollster_ratings.json (all years), read by model.py.
"""
import datetime as dt
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, FC)
import backtest as B          # noqa: E402
from pollsters import canon   # noqa: E402

WINDOW = 45
K = 6
MIN_GRADED = 5


def expit(x):
    return 1 / (1 + math.exp(-x))


def logit(p):
    return math.log(p / (1 - p))


def collect(years):
    rows = []
    for year in years:
        eday = B.ELECTIONS[year]
        races, polls, actual = B.build_year(year)
        fund, nat_prior, _ = B.fundamentals_for(year)
        for p in polls:
            r, a = races[p["race"]], actual[p["race"]]
            if r.get("nonpartisan") or not p["green"] or a["votes"].get("T", 0) / a["total"] > 0.20:
                continue
            d = dt.date.fromisoformat(p["field_end"])
            if not (0 <= (eday - d).days <= WINDOW):
                continue
            g, b = float(p["green"]), float(p["blue"])
            if g <= 0 or b <= 0:
                continue
            t = float(p["third"]) if p["third"] else 0.0
            u = float(p["undecided"]) if p["undecided"] else max(100 - g - b - t, 0.0)
            f = expit(nat_prior + fund["features_2026"][p["race"]]["pred_dev"])
            w = u / (g + b + u)
            adj = (1 - w) * (g / (g + b)) + w * f
            rows.append({"year": year, "race": p["race"], "pollster": canon(p["pollster"]),
                         "lean": p["sponsor_lean"], "online": p["mode"] == "online",
                         "err": logit(adj) - logit(a["two_camp_green"])})
    # relative to the other polls of the same race
    by_race = {}
    for i, r in enumerate(rows):
        by_race.setdefault((r["year"], r["race"]), []).append(i)
    for idx in by_race.values():
        for i in idx:
            others = [rows[j]["err"] for j in idx if j != i]
            rows[i]["rel"] = rows[i]["err"] - np.mean(others) if len(others) >= 2 else None
            rows[i]["excess"] = (abs(rows[i]["err"]) - np.mean([abs(x) for x in others])) if len(others) >= 2 else None
    return rows


def compute(years):
    rows = collect(years)
    rel = [r for r in rows if r["rel"] is not None]
    field = float(np.sqrt(np.mean([r["rel"] ** 2 for r in rel]))) if rel else 0.2
    out = {}
    for name in sorted({r["pollster"] for r in rows}):
        rs = [r for r in rows if r["pollster"] == name]
        rr = [r["rel"] for r in rs if r["rel"] is not None]
        n = len(rr)
        w = n / (n + K)
        house_raw = float(np.mean(rr)) if rr else 0.0
        scatter_raw = float(np.sqrt(np.mean([(x - house_raw) ** 2 for x in rr]))) if n > 1 else field
        v = {"n_polls": len(rs), "n_compared": n,
             "n_races": len({(r["year"], r["race"]) for r in rs}),
             "years": sorted({r["year"] for r in rs}),
             "house_logit": w * house_raw,
             "scatter_logit": math.sqrt(w * scatter_raw ** 2 + (1 - w) * field ** 2),
             "abs_err_pts": float(np.mean([abs(r["err"]) for r in rs])) * 25,
             "sponsor_lean": rs[0]["lean"], "online": rs[0]["online"]}
        v["house_pts"] = v["house_logit"] * 25
        v["scatter_pts"] = v["scatter_logit"] * 25
        ex = [r["excess"] for r in rs if r["excess"] is not None]
        v["excess_err_pts"] = (len(ex) / (len(ex) + K)) * float(np.mean(ex)) * 25 if ex else 0.0
        v["score_pts"] = v["excess_err_pts"]
        out[name] = v
    graded = sorted((k for k in out if out[k]["n_compared"] >= MIN_GRADED), key=lambda k: out[k]["score_pts"])
    if graded:
        cuts = np.percentile([out[k]["score_pts"] for k in graded], [20, 40, 60, 80])
        for k in graded:
            sc = out[k]["score_pts"]
            out[k]["grade"] = "A" if sc <= cuts[0] else "B" if sc <= cuts[1] else "C" if sc <= cuts[2] else "D" if sc <= cuts[3] else "F"
    for k in out:
        out[k].setdefault("grade", None)
    year_miss = {y: float(np.mean([r["err"] for r in rows if r["year"] == y])) for y in years}
    return {"years": list(years), "window_days": WINDOW, "shrinkage_K": K, "field_scatter_logit": field,
            "year_miss_pts": {y: m * 25 for y, m in year_miss.items()}, "pollsters": out}


def main():
    res = compute(list(B.ELECTIONS))
    json.dump(res, open(os.path.join(FC, "pollster_ratings.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("shared miss by year (pts, + = polls too DPP):", {y: round(v, 1) for y, v in res["year_miss_pts"].items()},
          " field scatter vs same-race polls:", round(res["field_scatter_logit"] * 25, 1), "pts")
    ps = res["pollsters"]
    order = sorted(ps, key=lambda k: (ps[k]["grade"] is None, ps[k]["score_pts"]))
    print(f"{'pollster':22} {'grade':5} {'polls':>5} {'races':>5} {'years':16} {'house(+DPP)':>11} {'scatter':>7} {'abs err':>7} {'vs rivals':>9}")
    for k in order:
        v = ps[k]
        print(f"{k:22} {v['grade'] or '–':5} {v['n_polls']:>5} {v['n_races']:>5} {','.join(v['years']):16} {v['house_pts']:>+11.1f} {v['scatter_pts']:>7.1f} {v['abs_err_pts']:>7.1f} {v['excess_err_pts']:>+9.1f}")


if __name__ == "__main__":
    main()
