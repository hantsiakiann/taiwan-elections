# 2026 local election forecast

Polls + fundamentals model for the 22 mayor/magistrate races on 28 Nov 2026,
simulated 10,000 times. Page: `forecast/index.html` (reads `output/forecast_2026.json`).

Rebuild from the repo root:

```
python forecast/extract_county_totals.py   # CEC files -> data/county_totals.json
python forecast/fundamentals.py            # fit + backtest -> output/fundamentals.json
python forecast/model.py                   # polls + sims -> output/forecast_2026.json
```

Inputs you edit by hand:
- `polls_2026.csv` — one row per poll. `green`/`blue` are the two main camps'
  candidates (see `races_2026.json` for who fills each slot), `third` an optional
  third candidate. `sponsor_lean` is G (DPP-aligned sponsor), B (KMT-aligned) or
  N (media/academic). `kind` is `candidate` or `camp` (party-camp question).
- `races_2026.json` — candidates, incumbency, third-candidate priors.

`model.py` takes the forecast date as its argument to `main()`; polls fielded
after that date are ignored, so past forecasts can be reproduced.
