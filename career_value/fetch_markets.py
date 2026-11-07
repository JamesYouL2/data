# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download historical ADP: MFL keeper/dynasty ADP and FantasyFootballCalculator PPR + 2QB ADP.

Saved as raw JSON under raw/markets/ (untrusted; parsed by market_residuals.py).
Also copies DynastyProcess values-players.csv from this repo's git history (first commit on/after May 10 of 2019
and 2020) to raw/markets/dp_values_<date>.csv: DP values then were FantasyPros dynasty ECR (1QB and 2QB), which gives
a superflex FP price for t = 2018-19, before FP's own superflex ECR (Oct 2020).
"""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "raw" / "markets"
UA = {"User-Agent": "career-value-research/0.1"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def dp_history():
    for y in (2019, 2020):
        log = subprocess.run(["git", "log", "--reverse", "--format=%H %cs", f"--since={y}-05-10", f"--until={y}-06-30",
                              "--", "files/values-players.csv"], cwd=OUT.parents[2], capture_output=True, text=True,
                             check=True).stdout.split()
        sha, date = log[0], log[1]
        f = OUT / f"dp_values_{date}.csv"
        if not f.exists():
            f.write_bytes(subprocess.run(["git", "show", f"{sha}:files/values-players.csv"], cwd=OUT.parents[2],
                                         capture_output=True, check=True).stdout)
        print("dp values", date, "ok", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dp_history()
    for y in range(2011, 2026):
        f = OUT / f"mfl_keeper_ppr_{y}.json"
        if not f.exists():
            d = get(f"https://api.myfantasyleague.com/{y}/export?TYPE=adp&IS_KEEPER=K&IS_PPR=1&JSON=1")
            f.write_text(json.dumps(d))
            time.sleep(1)
        f = OUT / f"ffc_ppr_{y}.json"
        if not f.exists():
            d = get(f"https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={y}&position=all")
            f.write_text(json.dumps(d))
            time.sleep(1)
        f = OUT / f"ffc_2qb_{y}.json"
        if not f.exists():
            d = get(f"https://fantasyfootballcalculator.com/api/v1/adp/2qb?teams=12&year={y}&position=all")
            f.write_text(json.dumps(d))
            time.sleep(1)
        print(y, "ok", flush=True)


if __name__ == "__main__":
    main()
