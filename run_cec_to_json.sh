#!/usr/bin/env bash
# run_cec_to_json.sh
# ------------------
# Convert every presidential-election data folder under data/ to JSON:
#   - 2008-2024: cec_to_json.py, one .xls/.xlsx per county, modern CEC
#     layout -- output written next to each source file.
#   - 1996-2004: cec_to_json_legacy.py, one nationwide polling-station-
#     level .xls per year, older CEC layout (see that script's docstring)
#     -- output written as one {year}-PRES-{ISO}.json per county, into
#     the same folder as the source file.
#
# USAGE
#     ./run_cec_to_json.sh [DATA_DIR]
#
#     DATA_DIR defaults to "data". Expects DATA_DIR/{year}-president/
#     subfolders, matching this repo's layout; a missing year is skipped
#     with a note rather than failing the whole run.

set -euo pipefail

data_dir="${1:-data}"

LEGACY_YEARS=(1996 2000 2004)
MODERN_YEARS=(2008 2012 2016 2020 2024)

for year in "${LEGACY_YEARS[@]}"; do
    folder="$data_dir/${year}-president"
    if [ ! -d "$folder" ]; then
        echo "Skipping $folder (not found)"
        continue
    fi

    shopt -s nullglob
    srcs=("$folder"/*.xls "$folder"/*.xlsx)
    shopt -u nullglob
    if [ "${#srcs[@]}" -eq 0 ]; then
        echo "WARNING: no .xls/.xlsx file found in $folder -- skipping." >&2
        continue
    fi
    if [ "${#srcs[@]}" -gt 1 ]; then
        echo "WARNING: multiple source files found in $folder; using the" \
             "first (${srcs[0]}) -- this folder should have one nationwide" \
             "file, not one per county." >&2
    fi

    echo "Converting (legacy): ${srcs[0]} -> $folder/"
    python cec_to_json_legacy.py "${srcs[0]}" "$folder" \
        --election-id "${year}-president"
done

for year in "${MODERN_YEARS[@]}"; do
    folder="$data_dir/${year}-president"
    if [ ! -d "$folder" ]; then
        echo "Skipping $folder (not found)"
        continue
    fi

    shopt -s nullglob
    for src in "$folder"/*.xls "$folder"/*.xlsx; do
        json="${src%.*}.json"
        echo "Converting: $src -> $json"
        python cec_to_json.py "$src" "$json" \
            --election-id "${year}-president" \
            --election-type president
    done
    shopt -u nullglob
done
