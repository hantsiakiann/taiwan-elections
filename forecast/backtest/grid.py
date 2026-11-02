import sys, json, datetime as dt, itertools, numpy as np
sys.path.insert(0, '.')
import backtest as B
B.VARIANTS = {}
grid = []
for race_sd, miss_sd, miss_mean in itertools.product([0.12, 0.18, 0.24], [0.08, 0.12], [0.0, 0.08]):
    name = f"race{race_sd}_nat{miss_sd}_bias{miss_mean}"
    B.VARIANTS[name] = {"RACE_MISS_SD": race_sd, "MISS_SD": miss_sd, "MISS_MEAN": miss_mean}
import model as M
for k in ("RACE_MISS_SD", "MISS_SD", "MISS_MEAN"):
    pass
# run() only restores a fixed set of attributes; save/restore these too
base = {k: getattr(M, k) for k in ("RACE_MISS_SD", "MISS_SD", "MISS_MEAN")}
out = []
for name, opts in B.VARIANTS.items():
    m = {"brier": [], "logloss": [], "mae": [], "cov": [], "called": 0, "n": 0}
    for year, eday in B.ELECTIONS.items():
        for days in (28, 10):
            o, actual = B.run(year, eday - dt.timedelta(days=days), name)
            for k, v in base.items(): setattr(M, k, v)
            rs = B.score(o, actual)
            m["brier"] += [r["brier"] for r in rs]; m["logloss"] += [r["logloss"] for r in rs]
            m["mae"] += [abs(r["share_err_pts"]) for r in rs if "share_err_pts" in r]
            m["cov"] += [r["in_80"] for r in rs if "in_80" in r]
            m["called"] += sum(r["correct"] for r in rs); m["n"] += len(rs)
    out.append((name, np.mean(m["brier"]), np.mean(m["logloss"]), np.mean(m["mae"]), np.mean(m["cov"]), m["called"], m["n"]))
print(f"{'variant':32} {'brier':>6} {'logloss':>7} {'MAE':>5} {'80%cov':>6} called")
for r in sorted(out, key=lambda r: r[2]):
    print(f"{r[0]:32} {r[1]:6.3f} {r[2]:7.3f} {r[3]:5.1f} {r[4]:6.0%} {r[5]}/{r[6]}")
