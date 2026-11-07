#!/usr/bin/env bash
# Download the build inputs into RAW_DIR (default: raw/). Season logic matches build_data.py.
set -euo pipefail
RAW="${1:-raw}"
mkdir -p "$RAW"
Y=$(date -u +%Y); M=$((10#$(date -u +%m)))
CUR=$(( M >= 9 ? Y : Y - 1 ))
DP=https://raw.githubusercontent.com/dynastyprocess/data/master/files
NV=https://github.com/nflverse/nflverse-data/releases/download
get() { curl -fsSL --retry 3 -o "$RAW/$2" "$1" && echo "got $2 ($(wc -c < "$RAW/$2") bytes)"; }
get "$DP/values-players.csv" values-players.csv
get "$DP/db_playerids.csv" db_playerids.csv
get "$NV/players/players.parquet" players.parquet
get "$NV/stats_player/stats_player_reg_$((CUR - 1)).parquet" "stats_player_reg_$((CUR - 1)).parquet"
# the current season's file appears after week 1; the build copes without it
get "$NV/stats_player/stats_player_reg_$CUR.parquet" "stats_player_reg_$CUR.parquet" || echo "no $CUR stats yet"
