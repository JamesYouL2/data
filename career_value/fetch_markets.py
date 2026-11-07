# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download historical ADP: MFL keeper/dynasty ADP and FantasyFootballCalculator PPR + 2QB ADP.

Saved as raw JSON under raw/markets/ (untrusted; parsed by market_residuals.py).
"""
import json
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "raw" / "markets"
UA = {"User-Agent": "career-value-research/0.1"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
