#!/usr/bin/env python3
"""Backtest the forecast model on the 2018 and 2022 mayor/magistrate elections.

For each year, the 2026 model is run exactly as it would have been before
that election:
  - polls: the final-matchup tables from zh.wikipedia (wiki_polls_<year>.json),
    only polls finished on or before the forecast date
  - fundamentals: refit without the target cycle (no peeking at its result);
    the residual SD comes from leave-one-out on the other cycles only
  - national prior: preceding presidential two-camp share minus the average
    governing-party penalty from the earlier mayoral cycles
Then scored against the actual CEC results in data/county_totals.json.

Usage:  python forecast/backtest/backtest.py
Writes forecast/backtest/results.json and prints a summary.
"""
import contextlib
import csv
import datetime as dt
import io
import json
import math
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
ROOT = os.path.dirname(FC)
sys.path.insert(0, FC)
import fundamentals as FU  # noqa: E402
import model as M          # noqa: E402

ELECTIONS = {"2018": dt.date(2018, 11, 24), "2022": dt.date(2022, 11, 26)}
CT = json.load(open(os.path.join(ROOT, "data", "county_totals.json"), encoding="utf-8"))

# Sponsor leans and online panels (same rules as the 2026 poll file)
LEAN_G = {"新台灣國策智庫", "民進黨", "民主進步黨", "台灣世代智庫", "臺灣世代智庫", "台灣智庫"}
LEAN_B = {"國民黨", "張善政團隊", "柯志恩團隊", "謝國樑團隊"}
ONLINE = {"網路溫度計", "蕃新聞", "QuickseeK", "皮爾森數據", "鉅聞天下"}
PARTY_PREFIX = {"民進": "DPP", "國民": "KMT", "民眾": "TPP", "時力": "NPP", "無黨": "IND",
                "民國": "MKT", "樹黨": "TREE"}
# Independents effectively backed by a camp
CAMP_BACKED = {("2018", "徐定禎"): "G"}
# Races not forecast: both main candidates KMT (Lienchiang 2022)
SKIP = {("2022", "LIE")}
TIER = {"TPE", "NWT", "TAO", "TXG", "TNN", "KHH"}


def cn_date(s):
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
    return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def num(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_cols(hdr):
    """header like ['民調來源','完成日期','國民 楊鎮浯',...,'未表態'] -> [(party, name)]"""
    cols = []
    for h in hdr[2:-1]:
        pre, _, name = h.partition(" ")
        name = re.sub(r"（.*?）", "", name).strip()
        cols.append((PARTY_PREFIX.get(pre, "IND"), name))
    return cols


def build_year(year):
    wiki = json.load(open(os.path.join(HERE, f"wiki_polls_{year}.json"), encoding="utf-8"))
    races, polls, actual = {}, [], {}
    for iso, tab in wiki.items():
        if (year, iso) in SKIP:
            continue
        cols = parse_cols(tab["hdr"])
        rows = [r for r in tab["rows"] if cn_date(r[1])]
        # slots: G = DPP (or DPP-backed), B = KMT; T = largest other candidate on average
        slot = {}
        for j, (party, name) in enumerate(cols):
            if CAMP_BACKED.get((year, name)) == "G" or (party == "DPP" and "G" not in slot.values()):
                slot[j] = "G"
            elif party == "KMT" and "B" not in slot.values():
                slot[j] = "B"
        others = [j for j in range(len(cols)) if j not in slot]
        if others:
            def avg(j):
                v = [num(r[2 + j]) for r in rows]
                v = [x for x in v if x is not None]
                return sum(v) / len(v) if v else 0
            slot[max(others, key=avg)] = "T"
        inv = {v: k for k, v in slot.items()}
        nonpartisan = "G" not in inv
        cand = lambda s: [cols[inv[s]][1], cols[inv[s]][1], cols[inv[s]][0]]  # noqa: E731
        cfg = {"en": iso, "zh": iso, "tier": "municipality" if iso in TIER else "county", "incumbent": "0"}
        if nonpartisan:
            cfg.update({"nonpartisan": True, "blue": cand("B"), "indep": cand("T"),
                        "kmt_vs_top_indep_prior": [0.55, 0.10]})
        else:
            cfg.update({"green": cand("G"), "blue": cand("B")})
            if "T" in inv:
                cfg["third"] = cand("T")
        races[iso] = cfg
        # polls
        for r in rows:
            d = cn_date(r[1])
            g = num(r[2 + inv["G"]]) if "G" in inv else None
            b = num(r[2 + inv["B"]])
            t = num(r[2 + inv["T"]]) if "T" in inv else None
            u = num(r[-1])
            if b is None or (not nonpartisan and g is None) or (nonpartisan and t is None):
                continue
            if (g or 0) + b + (t or 0) <= 0:
                continue
            name = r[0]
            polls.append({"race": iso, "pollster": name,
                          "sponsor_lean": "G" if name in LEAN_G else "B" if name in LEAN_B else "N",
                          "kind": "candidate", "field_end": d.isoformat(), "n": "1000",
                          "mode": "online" if name in ONLINE else "phone",
                          "green": "" if g is None else str(g), "blue": str(b),
                          "third": "" if t is None else str(t), "undecided": "" if u is None else str(u),
                          "source": "", "notes": ""})
        # actual result, by slot
        res = CT["mayor"][year][iso]["candidates"]
        def votes(nm):
            # CEC files store some rare characters as private-use glyphs (e.g. 江聰淵), so fall
            # back to matching the first and last characters
            hit = [c["votes"] for c in res if c["name"][:2] == nm[:2]] or \
                  [c["votes"] for c in res if c["name"][0] == nm[0] and c["name"][-1] == nm[-1]]
            return hit[0] if hit else 0
        tot = sum(c["votes"] for c in res)
        av = {s: votes(cols[j][1]) for j, s in slot.items()}
        top = max(c["votes"] for c in res)
        winner = next((s for s, v in av.items() if v == top), "X")
        actual[iso] = {"votes": av, "total": tot, "winner": winner,
                       "two_camp_green": av["G"] / (av["G"] + av["B"]) if not nonpartisan else None,
                       "kmt_vs_top_indep": av["B"] / (av["B"] + av["T"]) if nonpartisan else None}
    return races, polls, actual


def fundamentals_for(year):
    rows, nats = FU.build_rows()
    train = [r for r in rows if r["cycle"] not in (year, "2026") and r["y"] is not None and not r["distorted"]]
    beta, _ = FU.fit(train)
    # residual SD: leave-one-out over the remaining cycles only
    errs = []
    for cy in sorted({r["cycle"] for r in train}):
        tr = [r for r in train if r["cycle"] != cy]
        te = [r for r in train if r["cycle"] == cy]
        b, _ = FU.fit(tr)
        errs += list(np.array([r["y"] for r in te]) - FU.design(te) @ b)
    sd = float(np.sqrt(np.mean(np.square(errs))))
    feats = {}
    for r in rows:
        if r["cycle"] == year:
            feats[r["iso"]] = {k: r[k] for k in ("pres", "pl", "prev", "prev_imputed", "inc")}
            feats[r["iso"]]["pred_dev"] = float((FU.design([r]) @ beta)[0])
    # national prior: preceding presidential share minus mean governing-party penalty in earlier cycles
    fund_all = json.load(open(os.path.join(FC, "output", "fundamentals.json")))
    hist = [h for h in fund_all["env_history"] if h["cycle"] < year]
    pen = float(np.mean([h["ruling_party_penalty_logit"] for h in hist]))
    pres = {"2018": "2016", "2022": "2020"}[year]
    _, pres_share = FU.lean("president", pres)
    ruling = FU.CYCLES[year]["ruling"]
    nat_prior = math.log(pres_share / (1 - pres_share)) - pen * (1 if ruling == "DPP" else -1)
    fund = {"features_2026": feats, "resid_sd_out_of_sample": sd, "coef": dict(zip(["lean", "prev", "inc"], beta.tolist())),
            "backtest": {}, "env_history": hist}
    return fund, nat_prior, nats[year][1]


VARIANTS = {
    "fundamentals only":        {"no_polls": True},
    "polls, undecideds dropped": {"ALLOCATE_UNDECIDED": False, "FUND_INFLATE_MAX": 1.0},
    "current model":             {},
    "current + regional":        {"REGIONAL": True},
}


def run(year, as_of, variant):
    races, polls, actual = build_year(year)
    fund, nat_prior, _ = fundamentals_for(year)
    opts = VARIANTS[variant]
    saved = {k: getattr(M, k) for k in ("ALLOCATE_UNDECIDED", "FUND_INFLATE_MAX", "REGIONAL", "ELECTION", "NAT_PRIOR", "OUT", "load")}
    try:
        for k, v in opts.items():
            if k != "no_polls":
                setattr(M, k, v)
        M.ELECTION = ELECTIONS[year]
        M.NAT_PRIOR = (nat_prior, 0.20)
        tmp = tempfile.mkdtemp()
        M.OUT = tmp
        use_polls = [] if opts.get("no_polls") else polls
        if not use_polls:
            # model needs at least one row; a zero-weight dummy poll keeps the algebra intact
            use_polls = [{"race": next(iter(r for r in races if not races[r].get("nonpartisan"))), "pollster": "dummy",
                          "sponsor_lean": "N", "kind": "camp", "field_end": "1900-01-01", "n": "1",
                          "mode": "online", "green": "50", "blue": "50", "third": "", "undecided": "", "source": "", "notes": ""}]
        M.load = lambda: (fund, {"races": races}, use_polls)
        with contextlib.redirect_stdout(io.StringIO()):
            M.main(as_of)
        out = json.load(open(os.path.join(tmp, "forecast_2026.json")))
    finally:
        for k, v in saved.items():
            setattr(M, k, v)
    return out, actual


def score(out, actual):
    rows = []
    for iso, a in actual.items():
        r = out["races"][iso]
        wp = r["win_prob"]
        key = "I" if r.get("nonpartisan") and a["winner"] == "T" else a["winner"]
        p_win = wp.get(key, 0.0)
        fav = max(wp, key=wp.get)
        fav = "T" if fav == "I" else fav
        brier = sum((wp.get(k, 0) - (1.0 if k == key else 0.0)) ** 2 for k in set(wp) | {key})
        row = {"race": iso, "winner": a["winner"], "p_winner": p_win, "correct": fav == a["winner"],
               "brier": brier, "logloss": -math.log(max(p_win, 0.01))}
        third_share = a["votes"].get("T", 0) / a["total"]
        if not r.get("nonpartisan") and third_share <= 0.20:   # two-camp share is meaningless when a third candidate is a main contender
            q = r["two_camp_green"]["p10_p50_p90"]
            act = a["two_camp_green"]
            row.update({"share_err_pts": (q[1] - act) * 100, "in_80": q[0] <= act <= q[2]})
        rows.append(row)
    return rows


def main():
    results = {}
    for year, eday in ELECTIONS.items():
        _, _, nat_actual = fundamentals_for(year)
        for days_out in (28, 10):
            as_of = eday - dt.timedelta(days=days_out)
            for variant in VARIANTS:
                out, actual = run(year, as_of, variant)
                rows = score(out, actual)
                g_actual = sum(1 for a in actual.values() if a["winner"] == "G")
                results[f"{year}|{days_out}|{variant}"] = {
                    "rows": rows, "nat_projected": out["national"]["env_green_two_camp"]["mean"],
                    "nat_actual": nat_actual, "dpp_seats_mean": out["party_seats"]["DPP"]["mean"],
                    "dpp_seats_80": out["party_seats"]["DPP"]["p10_p90"], "dpp_seats_actual": g_actual,
                    "n_polls": out["method"]["n_polls_used"]}
    json.dump(results, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"{'year':5} {'out':>4} {'variant':27} {'called':>7} {'brier':>6} {'logloss':>7} {'shareMAE':>8} {'80%cov':>6} {'nat proj/act':>13} {'DPP seats':>16}")
    for key, v in results.items():
        year, days, variant = key.split("|")
        rs = v["rows"]
        sh = [r for r in rs if "share_err_pts" in r]
        print(f"{year:5} {days:>4} {variant:27} {sum(r['correct'] for r in rs):>3}/{len(rs):<3} "
              f"{np.mean([r['brier'] for r in rs]):6.3f} {np.mean([r['logloss'] for r in rs]):7.3f} "
              f"{np.mean([abs(r['share_err_pts']) for r in sh]):8.1f} {np.mean([r['in_80'] for r in sh]):6.0%} "
              f"{v['nat_projected']:6.3f}/{v['nat_actual']:.3f} "
              f"{v['dpp_seats_mean']:5.1f} {str(v['dpp_seats_80']):>7} ({v['dpp_seats_actual']})")


if __name__ == "__main__":
    main()
