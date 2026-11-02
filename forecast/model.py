#!/usr/bin/env python3
"""Polls + fundamentals forecast for the 2026 Taiwan mayor/magistrate races,
simulated 10,000 times.

Model (logit scale, x = green share of the two-camp vote):

    theta_r = nat + mu_r + delta_r                     true result, race r
    y_i     = theta_r(i) + h_lean(i) + b + c_r(i) + e_i  poll i in race r(i)
    theta at election = theta_r + nu_r                  forward drift

  nat      national environment       prior N(logit .46, .20^2)
  mu_r     fundamentals prediction    from fundamentals.py (lean, prior
                                      mayoral deviation, incumbency)
  delta_r  race-specific deviation    prior N(0, sd_oos^2), sd_oos from the
                                      leave-one-cycle-out backtest
  h_G/h_B  house effect of DPP- / KMT-aligned sponsors
                                      priors N(+.15,.10^2) / N(-.15,.10^2);
                                      media/academic polls are the reference
  b        polling miss shared by all polls          prior N(0, .08^2)
  c_r      polling miss shared by the polls in race r prior N(0, .12^2)
           (undecided allocation, turnout, local frame problems: averaging
           more polls of the same race cannot shrink this)
  undecided  each poll's undecided respondents are allocated as if they break
           like the county's partisanship (fundamentals + estimated national
           environment), not like the decided respondents; the break itself
           carries SD .15 of DPP share, scaled by the undecided weight. This
           keeps low-name-recognition candidates in a poll with many
           undecideds from being written off.
  e_i      poll error: sampling (design effect 1.5, decided respondents only)
           + non-sampling (.10; .30 for party-camp questions; +.10 online
           opt-in) + drift from fieldwork to today (.020^2 per day; Taipei moved
           ~.8 logit May-Oct 2026, so campaign drift is large)
  nu_r     drift from today to election day, N(0, .020^2 * days)

Every term is Gaussian and enters linearly, so the posterior over
(nat, h_G, h_B, b, delta_1..R) is computed exactly. Each simulation is one
joint posterior draw, so national swings and house-effect uncertainty are
shared across races. The national environment used to allocate undecideds
comes from a first pass without the allocation, so it moves with the polls.

Weighting polls over fundamentals as election day nears: the prior SD on
each race's deviation from its fundamentals (delta_r) is inflated by
fund_inflation(days_left), which ramps linearly from 1.0 at 100+ days out to
1.6 on election day, for races with at least one poll. A wider prior means
the race's polls pull it further from the fundamentals; races with no polls
keep the backtested fundamentals uncertainty. Older polls are also discounted by the drift term,
and undecided shares fall as the campaign ends, so poll weight rises on
both counts.

Third candidates (where polled, plus Penghu's Yeh Chu-lin from a prior)
take a logit-normal share off the top; the camps split the rest by theta.
Kinmen and Lienchiang have no DPP contest and use a KMT-vs-top-independent
prior only.
"""
import csv
import datetime as dt
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
N_SIMS = 10_000
SEED = 2026

ELECTION = dt.date(2026, 11, 28)
NAT_PRIOR = (math.log(0.46 / 0.54), 0.20)
HOUSE_PRIOR = {"G": (0.15, 0.10), "B": (-0.15, 0.10)}
MISS_SD = 0.08
MISS_MEAN = 0.0   # prior mean of the shared polling miss (logit; + = polls overstate the DPP)
RACE_MISS_SD = 0.12
DEFF = 1.5
NONSAMPLE = {"candidate": 0.10, "camp": 0.30}
ONLINE_EXTRA = 0.10
DRIFT_PER_DAY = 0.020
THIRD_SD = 0.40  # logit SD on a third candidate's share
UND_BREAK_SD = 0.15   # SD of the DPP share of the undecided break, around the county's lean
ALLOCATE_UNDECIDED = True   # switches, so backtest.py can compare model versions
REGIONAL = False            # regional polling misses and correlated late drift
REGIONS = {"North": ["TPE", "NWT", "KEE", "TAO", "HSQ", "HSZ", "ILA"],
           "Central": ["MIA", "TXG", "CHA", "NAN", "YUN"],
           "South": ["CYQ", "CYI", "TNN", "KHH", "PIF"],
           "East": ["HUA", "TTT"], "Islands": ["PEN", "KIN", "LIE"]}
REG_MISS_SD = 0.08          # polling miss shared within a region (REGIONAL only)
RACE_MISS_SD_REG = 0.09     # race miss when REGIONAL (0.08^2 + 0.09^2 ~ 0.12^2)
DRIFT_SPLIT = (0.5, 0.25, 0.25)   # national / regional / race share of drift variance (REGIONAL only)
FUND_RAMP_DAYS = 100  # fundamentals start losing weight this many days before the election
FUND_INFLATE_MAX = 1.6


def fund_inflation(days_left):
    """Multiplier on the fundamentals prior SD: 1.0 far out, FUND_INFLATE_MAX on election day."""
    frac = min(max((FUND_RAMP_DAYS - days_left) / FUND_RAMP_DAYS, 0.0), 1.0)
    return 1.0 + (FUND_INFLATE_MAX - 1.0) * frac


def logit(p):
    return math.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def load():
    fund = json.load(open(os.path.join(OUT, "fundamentals.json")))
    races = json.load(open(os.path.join(HERE, "races_2026.json"), encoding="utf-8"))
    polls = list(csv.DictReader(open(os.path.join(HERE, "polls_2026.csv"), encoding="utf-8")))
    return fund, races, polls


def poll_obs(p, as_of, fund=None):
    """Poll -> (logit DPP two-camp share, variance, days to election, age).
    fund: the race's fundamentals-based DPP two-camp share; when given,
    undecided respondents are allocated toward it."""
    g, b = float(p["green"]), float(p["blue"])
    t = float(p["third"]) if p.get("third") else 0.0
    u = float(p["undecided"]) if p.get("undecided") else max(100 - g - b - t, 0.0)
    share = g / (g + b)
    n2 = float(p["n"]) * (g + b) / 100
    und_var = 0.0
    if fund is not None and u > 0:
        w = u / (g + b + u)                 # undecideds' weight in the eventual two-camp vote
        adj = (1 - w) * share + w * fund
        und_var = (w * UND_BREAK_SD / (adj * (1 - adj))) ** 2
        share = adj
    end = dt.date.fromisoformat(p["field_end"])
    days = (ELECTION - end).days
    var = DEFF / (n2 * share * (1 - share))
    var += NONSAMPLE[p["kind"]] ** 2
    if "online" in p["mode"]:
        var += ONLINE_EXTRA ** 2
    age = max((as_of - end).days, 0)
    var += DRIFT_PER_DAY ** 2 * age
    var += und_var
    return logit(share), var, days, age


def main(as_of=None):
    as_of = as_of or dt.date.today()
    fund, races, polls = load()
    feats = fund["features_2026"]
    sd_f = fund["resid_sd_out_of_sample"]
    partisan = [r for r in races["races"] if not races["races"][r].get("nonpartisan")]
    R = len(partisan)
    idx = {r: i for i, r in enumerate(partisan)}
    # parameter vector: [nat, hG, hB, b, delta_1..R, c_1..R]
    reg_list = list(REGIONS)
    region_of = {c: g for g, cs in REGIONS.items() for c in cs}
    G_ = len(reg_list) if REGIONAL else 0
    P = 4 + 2 * R + G_
    m0 = np.zeros(P)
    v0 = np.zeros(P)
    m0[0], v0[0] = NAT_PRIOR[0], NAT_PRIOR[1] ** 2
    m0[1], v0[1] = HOUSE_PRIOR["G"][0], HOUSE_PRIOR["G"][1] ** 2
    m0[2], v0[2] = HOUSE_PRIOR["B"][0], HOUSE_PRIOR["B"][1] ** 2
    m0[3], v0[3] = MISS_MEAN, MISS_SD ** 2
    days_left = (ELECTION - as_of).days
    infl = fund_inflation(days_left)
    # inflate only where there are polls to take the weight; unpolled races keep
    # the backtested fundamentals uncertainty
    polled = {p["race"] for p in polls if p["race"] in idx and dt.date.fromisoformat(p["field_end"]) <= as_of}
    race_infl = np.array([infl if r in polled else 1.0 for r in partisan])
    v0[4:4 + R] = (sd_f * race_infl) ** 2
    v0[4 + R:4 + 2 * R] = (RACE_MISS_SD_REG if REGIONAL else RACE_MISS_SD) ** 2
    if REGIONAL:
        v0[4 + 2 * R:] = REG_MISS_SD ** 2
    mu = np.array([feats[r]["pred_dev"] for r in partisan])

    def posterior(nat_for_undecided):
        """Exact Gaussian posterior. nat_for_undecided=None skips the undecided allocation."""
        rows, ys, vs, used = [], [], [], []
        for p in polls:
            if p["race"] not in idx or dt.date.fromisoformat(p["field_end"]) > as_of:
                continue
            f = None if nat_for_undecided is None else float(expit(nat_for_undecided + mu[idx[p["race"]]]))
            y, v, days, age = poll_obs(p, as_of, fund=f)
            a = np.zeros(P)
            a[0] = 1
            a[3] = 1
            a[4 + idx[p["race"]]] = 1
            a[4 + R + idx[p["race"]]] = 1
            if REGIONAL:
                a[4 + 2 * R + reg_list.index(region_of[p["race"]])] = 1
            if p["sponsor_lean"] == "G":
                a[1] = 1
            elif p["sponsor_lean"] == "B":
                a[2] = 1
            rows.append(a)
            ys.append(y - mu[idx[p["race"]]])
            vs.append(v)
            used.append({**p, "logit": y, "var": v, "days_to_election": days})
        A = np.array(rows)
        W = np.diag(1 / np.array(vs))
        cov = np.linalg.inv(np.diag(1 / v0) + A.T @ W @ A)
        mean = cov @ (m0 / v0 + A.T @ W @ np.array(ys))
        return mean, cov, used

    # pass 1: national environment from the polls as reported; pass 2: undecideds
    # allocated toward each county's lean under that environment
    mean1, _, _ = posterior(None)
    mean, cov, used = posterior(mean1[0]) if ALLOCATE_UNDECIDED else posterior(None)

    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(mean, cov, size=N_SIMS)
    nat, hG, hB = draws[:, 0], draws[:, 1], draws[:, 2]
    dsd = DRIFT_PER_DAY * math.sqrt(days_left)
    if REGIONAL:
        rg = np.array([reg_list.index(region_of[r]) for r in partisan])
        fwd = dsd * (math.sqrt(DRIFT_SPLIT[0]) * rng.standard_normal((N_SIMS, 1))
                     + math.sqrt(DRIFT_SPLIT[1]) * rng.standard_normal((N_SIMS, len(reg_list)))[:, rg]
                     + math.sqrt(DRIFT_SPLIT[2]) * rng.standard_normal((N_SIMS, R)))
    else:
        fwd = dsd * rng.standard_normal((N_SIMS, R))
    theta = nat[:, None] + mu[None, :] + draws[:, 4:4 + R] + fwd   # (sims, R)

    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "as_of": as_of.isoformat(), "election_date": ELECTION.isoformat(),
           "n_sims": N_SIMS, "national": {}, "races": {}, "seats": {}}
    winners = np.empty((N_SIMS, len(races["races"])), dtype=object)
    order = list(races["races"])

    for r in partisan:
        cfg = races["races"][r]
        i = idx[r]
        two = expit(theta[:, i])
        third_share = np.zeros(N_SIMS)
        third_info = None
        if "third" in cfg:
            tp = [p for p in used if p["race"] == r and p["kind"] == "candidate" and p["third"]]
            if tp:
                latest = max(tp, key=lambda p: p["field_end"])
                g, b, t = float(latest["green"]), float(latest["blue"]), float(latest["third"])
                center = t / (g + b + t)
                sd = math.sqrt(THIRD_SD ** 2 + DRIFT_PER_DAY ** 2 * latest["days_to_election"])
            elif "third_prior" in cfg:
                center, sd = cfg["third_prior"][0], cfg["third_prior"][1] / (cfg["third_prior"][0] * (1 - cfg["third_prior"][0]))
            else:
                center, sd = None, None
            if center:
                third_share = expit(logit(center) + sd * rng.standard_normal(N_SIMS))
                third_info = {"name": cfg["third"], "share_center": center}
        g_share = (1 - third_share) * two
        b_share = (1 - third_share) * (1 - two)
        stack = np.stack([g_share, b_share, third_share], axis=1)
        win = np.argmax(stack, axis=1)
        winners[:, order.index(r)] = np.array(["G", "B", "T"])[win]

        # display aids: fundamentals-only estimate and house-adjusted poll avg
        fund_only = expit(mean[0] + mu[i])
        rp = [p for p in used if p["race"] == r]
        poll_avg = None
        if rp:
            adj = []
            for p in rp:
                h = mean[1] if p["sponsor_lean"] == "G" else mean[2] if p["sponsor_lean"] == "B" else 0
                adj.append((p["logit"] - h, 1 / p["var"]))
            poll_avg = float(expit(sum(y * w for y, w in adj) / sum(w for _, w in adj)))
        prior_var = (sd_f * race_infl[i]) ** 2
        post_var = cov[4 + i, 4 + i]
        q = lambda x: [float(np.percentile(x, k)) for k in (10, 50, 90)]
        out["races"][r] = {
            "name_en": cfg["en"], "name_zh": cfg["zh"], "tier": cfg["tier"],
            "incumbent": cfg["incumbent"],
            "candidates": {"G": cfg["green"], "B": cfg["blue"], **({"T": cfg["third"]} if third_info else {})},
            "win_prob": {k: float(np.mean(win == j)) for j, k in enumerate(["G", "B", "T"]) if k != "T" or third_info},
            "two_camp_green": {"p10_p50_p90": q(two), "mean": float(two.mean())},
            "vote_share": {"G": q(g_share), "B": q(b_share), **({"T": q(third_share)} if third_info else {})},
            "fundamentals_only_green": float(fund_only),
            "poll_avg_green": poll_avg,
            "n_polls": len(rp),
            "poll_weight": float(1 - post_var / prior_var),
            "features": {k: feats[r][k] for k in ("pres", "pl", "prev", "inc", "pred_dev")},
            "polls": [{k: p[k] for k in ("pollster", "sponsor_lean", "kind", "field_end", "n", "green",
                                         "blue", "third", "undecided", "source", "notes")} for p in rp],
        }

    # Kinmen / Lienchiang: KMT vs strongest independent
    for r in order:
        cfg = races["races"][r]
        if not cfg.get("nonpartisan"):
            continue
        m, s = cfg["kmt_vs_top_indep_prior"]
        center, sd_l = logit(m), s / (m * (1 - m))
        np_polls = [p for p in polls if p["race"] == r and dt.date.fromisoformat(p["field_end"]) <= as_of]
        note = "No DPP contest; no published polls found. Prior only."
        if np_polls:
            # combine prior with the polls (KMT vs strongest independent), with the
            # same error terms as partisan polls plus the race-level miss and drift
            prec, acc = 1 / sd_l ** 2, center / sd_l ** 2
            for p in np_polls:
                b_, t_ = float(p["blue"]), float(p["third"])
                sh = b_ / (b_ + t_)
                n2 = float(p["n"]) * (b_ + t_) / 100
                end = dt.date.fromisoformat(p["field_end"])
                var = DEFF / (n2 * sh * (1 - sh)) + NONSAMPLE["candidate"] ** 2 + RACE_MISS_SD ** 2 \
                    + DRIFT_PER_DAY ** 2 * max((as_of - end).days, 0)
                prec += 1 / var
                acc += logit(sh) / var
            center, sd_l = acc / prec, math.sqrt(1 / prec)
            note = "No DPP contest. KMT vs. the strongest independent, from the polls below."
        sd_l = math.sqrt(sd_l ** 2 + DRIFT_PER_DAY ** 2 * (ELECTION - as_of).days)
        x = expit(center + sd_l * rng.standard_normal(N_SIMS))
        winners[:, order.index(r)] = np.where(x > 0.5, "B", "I")
        out["races"][r] = {
            "name_en": cfg["en"], "name_zh": cfg["zh"], "tier": cfg["tier"], "incumbent": cfg["incumbent"],
            "nonpartisan": True, "candidates": {"B": cfg["blue"], "I": cfg["indep"]},
            "win_prob": {"B": float((x > 0.5).mean()), "I": float((x <= 0.5).mean())},
            "kmt_vs_top_indep": {"p10_p50_p90": [float(np.percentile(x, k)) for k in (10, 50, 90)]},
            "n_polls": len(np_polls), "note": note,
            "polls": [{k: p[k] for k in ("pollster", "sponsor_lean", "kind", "field_end", "n", "green",
                                         "blue", "third", "undecided", "source", "notes")} for p in np_polls],
        }

    g_seats = (winners == "G").sum(axis=1)
    b_seats = (winners == "B").sum(axis=1)
    o_seats = len(order) - g_seats - b_seats
    muni = [order.index(r) for r in order if races["races"][r]["tier"] == "municipality"]
    gm = (winners[:, muni] == "G").sum(axis=1)
    # party-level seats: map each race's winning slot to its candidate's party
    def party_of(label):
        for k in ("DPP", "KMT", "TPP"):
            if label.startswith(k):
                return k
        return "IND"
    slot_party = np.empty_like(winners)
    for j, r in enumerate(order):
        cfg = races["races"][r]
        lab = {"G": cfg.get("green", [None, None, "DPP"])[2], "B": cfg["blue"][2],
               "T": cfg.get("third", [None, None, "IND"])[2], "I": "IND"}
        col = winners[:, j]
        slot_party[:, j] = np.vectorize(lambda s: party_of(lab[s]))(col)
    party_seats = {}
    for k in ("DPP", "KMT", "TPP", "IND"):
        sc = (slot_party == k).sum(axis=1)
        party_seats[k] = {"mean": float(sc.mean()), "hist": np.bincount(sc, minlength=23).tolist(),
                          "p10_p90": [int(np.percentile(sc, 10)), int(np.percentile(sc, 90))]}
    out["party_seats"] = party_seats
    out["party_seats_2022"] = {"DPP": 5, "KMT": 14, "TPP": 1, "IND": 2}
    out["seats"] = {
        "green_hist": np.bincount(g_seats, minlength=23).tolist(),
        "blue_hist": np.bincount(b_seats, minlength=23).tolist(),
        "other_hist": np.bincount(o_seats, minlength=23).tolist(),
        "green_mean": float(g_seats.mean()), "blue_mean": float(b_seats.mean()),
        "green_p10_p90": [int(np.percentile(g_seats, 10)), int(np.percentile(g_seats, 90))],
        "blue_p10_p90": [int(np.percentile(b_seats, 10)), int(np.percentile(b_seats, 90))],
        "p_green_more": float((g_seats > b_seats).mean()),
        "municipalities_green_hist": np.bincount(gm, minlength=7).tolist(),
        "seats_2022": {"G": 5, "B": 14, "O": 3},
    }
    out["national"] = {
        "env_green_two_camp": {"mean": float(expit(nat).mean()),
                               "p10_p90": [float(np.percentile(expit(nat), 10)), float(np.percentile(expit(nat), 90))]},
        "env_prior": float(expit(NAT_PRIOR[0])),
        "house_effect_dpp_aligned_logit": float(hG.mean()),
        "house_effect_kmt_aligned_logit": float(hB.mean()),
        "history": fund["env_history"],
    }
    out["method"] = {
        "fundamentals_coef": fund["coef"], "fundamentals_oos_sd_logit": sd_f,
        "backtest": fund["backtest"], "nat_prior": NAT_PRIOR, "house_prior": HOUSE_PRIOR,
        "miss_sd": MISS_SD, "race_miss_sd": RACE_MISS_SD, "drift_per_day": DRIFT_PER_DAY, "n_polls_used": len(used),
        "undecided_break_sd": UND_BREAK_SD, "fund_inflation": infl, "days_left": days_left,
        "national_env_pass1": float(expit(mean1[0])),
    }
    json.dump(out, open(os.path.join(OUT, "forecast_2026.json"), "w"), ensure_ascii=False, indent=1)

    print(f"national env: {out['national']['env_green_two_camp']}  house G {hG.mean():+.3f} B {hB.mean():+.3f}")
    for r in order:
        d = out["races"][r]
        wp = " ".join(f"{k}:{v:.0%}" for k, v in d["win_prob"].items())
        extra = ""
        if "two_camp_green" in d:
            extra = (f"  2camp G {d['two_camp_green']['p10_p50_p90'][1]:.3f}"
                     f"  fund {d['fundamentals_only_green']:.3f}"
                     f"  polls {d['poll_avg_green'] if d['poll_avg_green'] is None else round(d['poll_avg_green'], 3)}"
                     f"  w_polls {d['poll_weight']:.2f} (n={d['n_polls']})")
        print(f"{r} {d['name_en']:15s} {wp:28s}{extra}")
    s = out["seats"]
    print("party seats:", {k: (round(v["mean"], 2), v["p10_p90"]) for k, v in party_seats.items()})
    print(f"seats: DPP mean {s['green_mean']:.1f} {s['green_p10_p90']}  KMT-camp {s['blue_mean']:.1f} {s['blue_p10_p90']}  P(DPP>KMT) {s['p_green_more']:.2f}")


if __name__ == "__main__":
    import sys
    main(dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None)
