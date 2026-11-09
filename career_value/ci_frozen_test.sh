#!/usr/bin/env bash
# The 2026 test's scheduled step, run by .github/workflows/frozen-test-2026.yml from the repo root. Safe to rerun: predict and score each do nothing
# until something new is due (analysis/frozen_test_2026.py). Versions are pinned so week 12 is fit like weeks 4 and 8.
set -euo pipefail
cd career_value

# Fixed inputs that aren't in git: the panel, the player table and Dynasty Daddy's past prices. They're kept as of the freeze, so an archive change
# or a rebuilt panel can't change how week 12 is made.
gh release download frozen-test-inputs --repo "$GITHUB_REPOSITORY" --pattern frozen_test_inputs.tgz --output - | tar xz -C ..

# The live inputs: FP ranks and this season's nflverse files
DP=https://raw.githubusercontent.com/dynastyprocess/data/master/files
for f in db_fpecr.parquet db_playerids.csv; do curl -fsSL --retry 3 -o "../files/$f" "$DP/$f"; done
NV=https://github.com/nflverse/nflverse-data/releases/download
FFO=https://github.com/ffverse/ffopportunity/releases/download/latest-data
get() { mkdir -p "raw/nflverse/$2"; curl -fsSL --retry 3 -o "raw/nflverse/$2/$(basename "$1")" "$1"; }
get "$NV/stats_player/stats_player_week_2026.parquet" stats_player_week
get "$FFO/ep_weekly_2026.parquet" ep_weekly
get "$NV/injuries/injuries_2026.parquet" injuries
get "$NV/weekly_rosters/roster_weekly_2026.parquet" roster_weekly
get "$NV/schedules/games.parquet" schedules

RUN="uv run -q --no-project --python 3.14 --with pandas==3.1.1 --with pyarrow==26.0.1 --with scikit-learn==1.9.1 --with scipy==1.18.1 --with numpy==2.5.4 python -I -W ignore analysis/frozen_test_2026.py"
for league in 1qb sf; do
  for source in ktc fantasycalc; do
    echo "== $league $source"
    LEAGUE=$league SOURCE=$source $RUN predict
    LEAGUE=$league SOURCE=$source $RUN score
  done
done
if [ -n "$(git status --porcelain -- outputs | grep _score.json || true)" ]; then $RUN site; fi
