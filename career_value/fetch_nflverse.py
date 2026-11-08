# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download weekly nflverse / ffverse data for the in-season riser-faller work to raw/nflverse/<dataset>/.

Per season (2017-2026; the 2026 files are partial and skipped when not yet published):
  stats_player_week  weekly box-score stats + PPR points        (nflverse-data, tag stats_player)
  ep_weekly          expected fantasy points by week (usage)    (ffverse/ffopportunity, tag latest-data)
  snap_counts        offensive snap share by game (PFR ids)     (tag snap_counts)
  injuries           weekly injury report + practice status     (tag injuries)
  depth_charts       weekly depth chart position                (tag depth_charts)
  roster_weekly      weekly team / roster status (IR, PUP, ...) (tag weekly_rosters)
  advstats_week_*    PFR weekly receiving / rushing / passing   (tag pfr_advstats; 2018+)
Single files (all seasons): ngs_receiving / ngs_rushing / ngs_passing (tag nextgen_stats; weekly rows, 2016+);
schedules/games.parquet (tag schedules: game dates, to map a ranking date to the weeks played).
Existing files are kept (delete to refresh). Polite: one request at a time, a short pause between.
"""
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "raw" / "nflverse"
UA = {"User-Agent": "career-value-research/0.1"}
NV = "https://github.com/nflverse/nflverse-data/releases/download"
FFO = "https://github.com/ffverse/ffopportunity/releases/download/latest-data"
YEARS = range(2017, 2027)
PER_YEAR = {
    "stats_player_week": f"{NV}/stats_player/stats_player_week_{{y}}.parquet",
    "ep_weekly": f"{FFO}/ep_weekly_{{y}}.parquet",
    "snap_counts": f"{NV}/snap_counts/snap_counts_{{y}}.parquet",
    "injuries": f"{NV}/injuries/injuries_{{y}}.parquet",
    "depth_charts": f"{NV}/depth_charts/depth_charts_{{y}}.parquet",
    "roster_weekly": f"{NV}/weekly_rosters/roster_weekly_{{y}}.parquet",
    "advstats_week_rec": f"{NV}/pfr_advstats/advstats_week_rec_{{y}}.parquet",
    "advstats_week_rush": f"{NV}/pfr_advstats/advstats_week_rush_{{y}}.parquet",
    "advstats_week_pass": f"{NV}/pfr_advstats/advstats_week_pass_{{y}}.parquet",
}
SINGLE = {"nextgen": [f"{NV}/nextgen_stats/ngs_{k}.parquet" for k in ("receiving", "rushing", "passing")],
          "schedules": [f"{NV}/schedules/games.parquet"]}


def fetch(url: str, f: Path) -> str:
    if f.exists():
        return "cached"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "not published"
        raise
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(data)
    time.sleep(1)
    return f"{len(data) / 1e6:.1f} MB"


def main():
    jobs = [(u.format(y=y), OUT / k / u.format(y=y).rsplit("/", 1)[1]) for k, u in PER_YEAR.items() for y in YEARS]
    jobs += [(u, OUT / k / u.rsplit("/", 1)[1]) for k, us in SINGLE.items() for u in us]
    for url, f in jobs:
        print(f"{f.relative_to(OUT)}: {fetch(url, f)}", flush=True)


if __name__ == "__main__":
    main()
