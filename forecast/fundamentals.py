#!/usr/bin/env python3
"""Fundamentals model for Taiwan mayor/magistrate races.

Builds county-level partisan-lean features from data/county_totals.json and
fits

    dev_ct = b1*lean + b2*prev_dev + b3*incumbency + e

with lean = mean(presidential lean, party-list lean). The two leans correlate
at r=0.997 across counties, so fitting them separately just splits one effect
into offsetting coefficients (it gave pres -0.24 / party-list +0.51) with no
out-of-sample gain.

where dev_ct = logit(green two-camp share of county c's mayoral vote in
cycle t) minus that cycle's national mayoral environment (vote-weighted
logit across contested races). Leans are logit(county) - logit(national) of
the green/(green+blue) share in the most recent presidential and party-list
votes.

Camps:  green = DPP (+TSU/NPP/TSP on the party list) and DPP-endorsed
        independents;  blue = KMT, PFP, NP and blue-aligned independents.
Races where non-camp candidates took > 20% are "distorted": dropped as
targets and their prev_dev replaced by the presidential lean.
Kinmen and Lienchiang are excluded (no meaningful DPP contest).

Writes forecast/output/fundamentals.json with coefficients, backtests and
the 2026 feature table.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CT = json.load(open(os.path.join(ROOT, "data", "county_totals.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

COUNTIES = ["TPE", "NWT", "TAO", "TXG", "TNN", "KHH", "KEE", "ILA", "HSQ", "HSZ", "MIA",
            "CHA", "NAN", "YUN", "CYQ", "CYI", "PIF", "HUA", "TTT", "PEN"]
EXCLUDED = ["KIN", "LIE"]

# mayoral cycles -> (source years, prior presidential/party-list year, prior cycle)
CYCLES = {
    "2005": {"years": ["2005", "2006"]},
    "2009": {"years": ["2009", "2010"], "pres": "2008", "pl": "2008", "prev": "2005", "ruling": "KMT"},
    "2014": {"years": ["2014"], "pres": "2012", "pl": "2012", "prev": "2009", "ruling": "KMT"},
    "2018": {"years": ["2018"], "pres": "2016", "pl": "2016", "prev": "2014", "ruling": "DPP"},
    "2022": {"years": ["2022"], "pres": "2020", "pl": "2020", "prev": "2018", "ruling": "DPP"},
    "2026": {"years": [], "pres": "2024", "pl": "2024", "prev": "2022", "ruling": "DPP"},
}

GREEN_PARTIES = {"DPP"}
BLUE_PARTIES = {"KMT", "PFP", "NP"}
# (year, name) -> camp, for independents / minor-party candidates who were
# effectively a camp's candidate.
CAMP_OVERRIDE = {
    ("2014", "柯文哲"): "G",   # DPP stood aside and endorsed Ko
    ("2018", "徐定禎"): "G",   # DPP-backed independent (Miaoli)
    ("2005", "吳俊立"): "B", ("2005", "徐耀昌"): "B",
    ("2009", "張碧琴"): "B", ("2014", "鄭永金"): "B",
    ("2018", "徐欣瑩"): "B", ("2018", "蕭淑麗"): "B",
    ("2022", "鍾東錦"): "B",
}
# Fu Kun-chi (傅崐萁) appears with a PUA glyph in the CEC files; match by prefix.
BLUE_PREFIX = {("2005", "傅"), ("2009", "傅"), ("2014", "傅")}

# incumbency on the ballot, per (cycle, county): +1 green incumbent, -1 blue
INCUMBENT = {
    "2009": {"CHA": -1, "CYI": -1, "PIF": 1, "YUN": 1, "ILA": -1, "MIA": -1, "NAN": -1, "PEN": -1,
             "TPE": -1, "TXG": -1, "TNN": 0, "KHH": 1, "NWT": 0, "TAO": 0, "KEE": -1, "HSQ": 0,
             "HSZ": 0, "CYQ": 0, "HUA": 0, "TTT": 0},
    "2014": {"ILA": 1, "CYQ": 1, "PIF": 0, "YUN": 0, "TNN": 1, "KHH": 1, "NWT": -1, "TAO": -1,
             "TXG": -1, "HSQ": -1, "CHA": 0, "NAN": 0, "TTT": -1, "KEE": 0, "HSZ": -1, "MIA": 0,
             "CYI": 0, "PEN": 0, "HUA": -1, "TPE": 0},
    "2018": {"TAO": 1, "KEE": 1, "HSZ": 1, "PIF": 1, "YUN": 1, "CHA": 1, "PEN": 1, "CYI": 1,
             "MIA": -1, "NAN": -1, "HSQ": 0, "TTT": 0, "HUA": 0, "TXG": 1, "TNN": 0, "KHH": 0,
             "NWT": 0, "ILA": 0, "CYQ": 0, "TPE": 0},
    "2022": {"KHH": 1, "TNN": 1, "CYQ": 1, "PEN": -1, "PIF": 0, "TXG": -1, "NWT": -1, "CHA": -1,
             "YUN": -1, "ILA": -1, "HSQ": -1, "HUA": -1, "TTT": -1, "CYI": -1, "TPE": 0, "TAO": 0,
             "KEE": 0, "HSZ": 0, "MIA": 0, "NAN": 0},
    "2026": {"TPE": -1, "TAO": -1, "KEE": -1, "HSZ": -1, "MIA": -1, "NAN": -1, "PIF": 1},
}


def logit(p):
    return math.log(p / (1 - p))


def camp_of(year, c):
    if (year, c["name"]) in CAMP_OVERRIDE:
        return CAMP_OVERRIDE[(year, c["name"])]
    if any(year == y and c["name"].startswith(p) for y, p in BLUE_PREFIX):
        return "B"
    if c["party"] in GREEN_PARTIES:
        return "G"
    if c["party"] in BLUE_PARTIES:
        return "B"
    return "O"


def mayor_camps(cycle):
    """county -> dict(G, B, O votes) for a mayoral cycle (sums merged races)."""
    res = {}
    for yr in CYCLES[cycle]["years"]:
        for iso, race in CT["mayor"].get(yr, {}).items():
            d = res.setdefault(iso, {"G": 0, "B": 0, "O": 0})
            for c in race["candidates"]:
                d[camp_of(yr, c)] += c["votes"]
    return res


def mayor_dev(cycle):
    """county -> (dev, distorted?) and the national environment for a cycle."""
    camps = mayor_camps(cycle)
    G = sum(v["G"] for k, v in camps.items() if k not in EXCLUDED and v["G"] and v["B"])
    B = sum(v["B"] for k, v in camps.items() if k not in EXCLUDED and v["G"] and v["B"])
    nat = logit(G / (G + B))
    out = {}
    for iso, v in camps.items():
        if iso in EXCLUDED or not v["G"] or not v["B"]:
            out[iso] = (None, True)
            continue
        tot = v["G"] + v["B"] + v["O"]
        out[iso] = (logit(v["G"] / (v["G"] + v["B"])) - nat, v["O"] / tot > 0.20)
    return out, nat, G / (G + B)


def lean(kind, year):
    data = CT[kind][year]
    g_par = {"DPP"} if kind == "president" else {"DPP", "TSU", "NPP", "TSP"}
    b_par = {"KMT", "PFP", "NP"}
    G = sum(sum(v for p, v in d.items() if p in g_par) for d in data.values())
    B = sum(sum(v for p, v in d.items() if p in b_par) for d in data.values())
    nat = logit(G / (G + B))
    out = {}
    for iso, d in data.items():
        g = sum(v for p, v in d.items() if p in g_par)
        b = sum(v for p, v in d.items() if p in b_par)
        out[iso] = logit(g / (g + b)) - nat
    return out, G / (G + B)


def build_rows():
    rows = []
    devs = {cy: mayor_dev(cy) for cy in CYCLES if cy != "2026"}
    for cy in ["2009", "2014", "2018", "2022", "2026"]:
        meta = CYCLES[cy]
        pres, _ = lean("president", meta["pres"])
        pl, _ = lean("partylist", meta["pl"])
        prev, _, _ = devs[meta["prev"]]
        cur = devs.get(cy, ({}, None, None))[0]
        inc = INCUMBENT.get(cy, {})
        for iso in COUNTIES:
            p_dev, p_dist = prev.get(iso, (None, True))
            prev_val = pres[iso] if (p_dev is None or p_dist) else p_dev
            y, dist = cur.get(iso, (None, True)) if cy != "2026" else (None, False)
            rows.append({"cycle": cy, "iso": iso, "pres": pres[iso], "pl": pl[iso],
                         "prev": prev_val, "prev_imputed": p_dev is None or p_dist,
                         "inc": inc.get(iso, 0), "y": y, "distorted": dist})
    return rows, {cy: (v[1], v[2]) for cy, v in devs.items()}


def design(rs):
    return np.array([[(r["pres"] + r["pl"]) / 2, r["prev"], r["inc"]] for r in rs])


def fit(rs):
    X, y = design(rs), np.array([r["y"] for r in rs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = math.sqrt(resid @ resid / (len(y) - X.shape[1]))
    return beta, sigma


def main():
    rows, nats = build_rows()
    train = [r for r in rows if r["cycle"] != "2026" and r["y"] is not None and not r["distorted"]]
    beta, sigma = fit(train)
    names = ["lean", "prev_mayor_dev", "incumbency"]
    print("n =", len(train), " coefficients:", dict(zip(names, np.round(beta, 3))), " resid SD:", round(sigma, 3))

    # leave-one-cycle-out backtest; national environment taken as known so
    # the test isolates the fundamentals' cross-county skill
    bt = {}
    all_err = []
    for cy in ["2009", "2014", "2018", "2022"]:
        tr = [r for r in train if r["cycle"] != cy]
        te = [r for r in train if r["cycle"] == cy]
        b, _ = fit(tr)
        pred = design(te) @ b
        err = np.array([r["y"] for r in te]) - pred
        nat = nats[cy][0]
        hits = sum((p + nat > 0) == (r["y"] + nat > 0) for p, r in zip(pred, te))
        # pres-lean-only baseline
        base_err = np.array([r["y"] - r["pres"] for r in te])
        bt[cy] = {"n": len(te), "rmse_logit": float(np.sqrt((err ** 2).mean())),
                  "rmse_pts": float(np.sqrt((err ** 2).mean()) * 25),
                  "baseline_pres_only_rmse_logit": float(np.sqrt((base_err ** 2).mean())),
                  "winner_hits": int(hits)}
        all_err.extend(err.tolist())
        print(f"  backtest {cy}: n={len(te)} rmse={bt[cy]['rmse_logit']:.3f} logit (~{bt[cy]['rmse_pts']:.1f} pts share)"
              f"  pres-only baseline={bt[cy]['baseline_pres_only_rmse_logit']:.3f}  winners {hits}/{len(te)}")
    oos_sd = float(np.sqrt(np.mean(np.square(all_err))))
    print("  pooled out-of-sample SD:", round(oos_sd, 3))

    # national environment history: mayoral nat vs preceding presidential nat
    hist = []
    for cy in ["2009", "2014", "2018", "2022"]:
        _, pres_share = lean("president", CYCLES[cy]["pres"])
        m_share = nats[cy][1]
        sign = 1 if CYCLES[cy]["ruling"] == "DPP" else -1
        shift = logit(m_share) - logit(pres_share)
        hist.append({"cycle": cy, "ruling": CYCLES[cy]["ruling"], "mayor_green_share": m_share,
                     "prior_pres_green_share": pres_share, "shift_logit": shift,
                     "ruling_party_penalty_logit": -shift * sign})
        print(f"  env {cy}: mayors {m_share:.3f} vs pres {pres_share:.3f}  shift {shift:+.3f} (ruling {CYCLES[cy]['ruling']})")

    feats26 = {r["iso"]: {k: r[k] for k in ("pres", "pl", "prev", "prev_imputed", "inc")}
               for r in rows if r["cycle"] == "2026"}
    for iso, f in feats26.items():
        f["pred_dev"] = float((design([f]) @ beta)[0])
    _, pres24 = lean("president", "2024")
    json.dump({"coef": dict(zip(names, beta.tolist())), "resid_sd_in_sample": sigma,
               "resid_sd_out_of_sample": oos_sd, "backtest": bt, "env_history": hist,
               "pres2024_green_two_camp": pres24, "features_2026": feats26,
               "training_rows": [{k: r[k] for k in ("cycle", "iso", "y", "pres", "pl", "prev", "inc")}
                                 for r in train]},
              open(os.path.join(OUT, "fundamentals.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
