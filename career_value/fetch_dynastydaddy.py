# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download Dynasty Daddy archived market snapshots (KTC, FantasyCalc, DP, ...).

dynasty-daddy.com/api/v1/player/all/prev/{N} returns all players' values as of
N days ago. We take one post-draft snapshot per year (May 15), in-season snapshots
(inseason_*.json, Sept-Feb of 2021-2025), the dates of the FP week 4 / 8 / 12
snapshots of inseason_movers.py (simweek_*.json, for market_price_sim.py), plus
the current id table (/player/all/today) for sleeper/mfl id mapping. Polite: one
request every few seconds, cached on disk.
"""
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "raw" / "dynastydaddy"
BASE = "https://dynasty-daddy.com/api/v1/player/all"
UA = {"User-Agent": "career-value-research/0.1"}
SNAP_DATES = [dt.date(y, 5, 15) for y in range(2021, 2027)]
# in-season snapshots for season Y (preseason, ~week 6, ~week 11, ~week 16, mid-Feb), matching FP scrape dates
INSEASON_DATES = [d for y in range(2021, 2026) for d in
                  (dt.date(y, 9, 1), dt.date(y, 10, 18), dt.date(y, 11, 17), dt.date(y, 12, 22), dt.date(y + 1, 2, 17))]
# inseason_movers.py's FP scrape dates at weeks 4, 8 and 12 (the same in 1QB and superflex)
SIM_DATES = [dt.date.fromisoformat(d) for d in (
    "2021-10-05", "2021-11-05", "2021-12-03", "2022-10-07", "2022-11-04", "2022-12-02", "2023-10-06", "2023-11-03",
    "2023-12-01", "2024-10-04", "2024-11-01", "2024-11-27", "2025-10-03", "2025-10-31", "2025-11-28")]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.load(r)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    f = OUT / f"today_{today}.json"
    if not f.exists():
        f.write_text(json.dumps(get(f"{BASE}/today")))
        time.sleep(3)
    for kind, dates in (("snapshot", SNAP_DATES), ("inseason", INSEASON_DATES), ("simweek", SIM_DATES)):
        for d in dates:
            f = OUT / f"{kind}_{d}.json"
            if f.exists():
                continue
            n = (today - d).days
            data = get(f"{BASE}/prev/{n}")
            got = sorted({str(r.get("date"))[:10] for r in data})
            print(kind, d, "->", got[:3], len(data), flush=True)
            f.write_text(json.dumps(data))
            time.sleep(3)


if __name__ == "__main__":
    main()
