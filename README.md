# Taiwan elections — historical data

Precinct/township-level Taiwan election results (1994–2024) from the Central Election Commission (中選會), the parsing pipeline that converts them to JSON, and the Vote Explorer.

- **Vote Explorer:** https://hantsiakiann.github.io/taiwan-elections/
- `data/<year>-<race>/`: CEC source `.xls` files and converted per-county JSON
- `cec_to_json.py` (2008+ layout), `cec_to_json_legacy.py` (1996–2004 layout), `run_cec_to_json.sh`: CEC → JSON
- `build_president_partisan.py` → `data/president_partisan.json`; `build_taiwan_map.py` → `data/taiwan_map.json` (both consumed by the explorer)
