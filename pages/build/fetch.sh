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

# in-season extras; each is optional and the build skips what is missing
get "$DP/db_fpecr_latest.csv" db_fpecr_latest.csv || echo "no FP ROS ranks"
get "$DP/db_fpecr.parquet" db_fpecr.parquet || echo "no FP rank history, so no signals that need a preseason rank"
get "$DP/fp_latest_weekly.csv" fp_latest_weekly.csv || echo "no FP weekly ranks"
FC="https://api.fantasycalc.com/values/current?isDynasty=true&numTeams=12&ppr=1"
get "$FC&numQbs=1" fc_1qb.json || echo "no FantasyCalc 1QB values"
get "$FC&numQbs=2" fc_sf.json || echo "no FantasyCalc superflex values"
get https://api.sleeper.app/v1/players/nfl sleeper_players.json || echo "no Sleeper player list"
get https://api.sleeper.app/v1/state/nfl sleeper_state.json || echo "no Sleeper state"
