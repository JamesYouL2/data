#!/usr/bin/env bash
# Download nflverse inputs (regular-season player stats 1999+, player bio/draft table).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw/stats_player_reg raw/players
gh release download stats_player -R nflverse/nflverse-data -p 'stats_player_reg_*.parquet' -D raw/stats_player_reg --clobber
gh release download players -R nflverse/nflverse-data -p 'players.parquet' -D raw/players --clobber
