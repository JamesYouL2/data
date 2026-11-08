"""Q: do the site's calls hold up on seasons they weren't checked on? Splits the cohorts in two and scores every call
group on each half with call_strength.py's lenses:
    3-year (keep3):           2018-19 (fit) vs 2020-21 (holdout)
    1-year (keep1, prod1):    2018-20 (fit) vs 2021-23 (holdout)
The lens is the one each call answers to: buys by the mode's buy lens (Rebuilding keep3, Default keep1, Contending
prod1), sells and cuts by keep3 (the rebuilder's lens, every mode), the rebuilder's holds by keep3 and the contender's
by prod1. A call holds up on the holdout half if:
    buy   >= 0.9 (BUY_HOLD_BELOW);  sell < 1.1, or still under 1 without the best player;
    cut   < 0.5, or under 0.5 without the best player;
    hold  0.65-1.5, or above 1.5 only through one player (a hold that should be a sell or a strong buy fails).
Lottery tickets and "use, then sell" are not scored. Groups with fewer than MIN_N players in the holdout half are
listed as too few. Calls come from rules.classify before the strength split, as call_strength.py.

Caveat: this is a split-half check, not a clean holdout. The rules and the price-band calls were picked looking at all
of 2018-23, so a call that passes here passed on data it had seen. A call that fails on the later half is still a
warning: what made it work early on has not lasted. Writes outputs/[sf/]holdout.csv. LEAGUE=sf for superflex.

Findings (2026-11, action calls that failed on the later half):
- 1QB contenders' "Sell high now" on Day-2+ rookie RBs priced 101-200 (keep3 1.61 / 1.71 without the best, both
  halves) is now a Lottery ticket, as for rebuilders. League sim, ranks 4-8: Contending 5-year titles 20.2 -> 23.2,
  value 3 years on 186 -> 220; Default and Rebuilding unchanged. Superflex keeps the sell (it held).
- 1QB "RB 30+, fading" Cut (2.03 / 0.68) stays: the 201-300 ones all returned 0 (14 of 14; 9 of 9 in superflex); the
  holdout gain is two waiver-level RBs past 300 (Cordarrelle Patterson 2020, Jerick McKinnon 2021) on prices near 0.
- Superflex: no action call fails.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import out_dir  # noqa: E402
import call_strength as cs  # noqa: E402

R, SF, MIN_N = cs.R, cs.SF, cs.MIN_N
FIT = {3: (2018, 2019), 1: (2018, 2019, 2020)}
HOLD = {3: (2020, 2021), 1: (2021, 2022, 2023)}
LENS = {"keep3": (3, "hold", "fair"), "keep1": (1, "hold", "fair"), "prod1": (1, "vorp1", "v1fair")}


def lens_for(mode, k):
    if k == "buy":
        return cs.BUY_LENS[mode]
    if k in ("sell", "cut"):
        return "keep3"
    if k == "hold" and mode in ("reb", "con"):
        return "keep3" if mode == "reb" else "prod1"
    return None


def ratio(q, num, den):
    if len(q) == 0 or q[den].sum() <= 0:
        return np.nan, np.nan
    k = q[num].sum() / q[den].sum()
    if len(q) < 2:
        return k, np.nan
    b = (q[num] - q[den]).idxmax()
    return k, (q[num].sum() - q[num][b]) / (q[den].sum() - q[den][b])


def verdict(k, v, solo, n):
    if n < MIN_N or np.isnan(v):
        return "too few"
    if k == "buy":
        return "holds" if v >= R.BUY_HOLD_BELOW else "fails"
    if k == "sell":
        return "holds" if v < 1.1 or solo < 1 else "fails"
    if k == "cut":                                  # one hit doesn't save a cut (rebucket.py's rule)
        return "holds" if (solo if solo < 0.5 else v) < 0.5 else "fails"
    if v < R.HEAVY_BELOW:
        return "fails (sell?)"
    return "fails (buy?)" if v >= R.STRONG_AT and solo >= 1 else "holds"


def main():
    H = cs.horizons()
    H[1] = cs.calls(H[1])
    cols = ["grp", "reb", "con", "mid", "pb"]
    H[3] = H[3].join(H[1].set_index(["gsis_id", "t"])[cols], on=["gsis_id", "t"])
    out = []
    for g in sorted(H[1].grp.unique()):
        for mode in ("reb", "mid", "con"):
            for act in H[1][H[1].grp == g][mode].unique():
                k = R.kind(act)
                lens = lens_for(mode, k)
                if lens is None:
                    continue
                h, num, den = LENS[lens]
                q = H[h][(H[h].grp == g) & (H[h][mode] == act)]
                row = {"grp": g, "mode": mode, "call": act, "lens": lens}
                for half, ts in (("fit", FIT[h]), ("hold", HOLD[h])):
                    s = q[q.t.isin(ts)]
                    row[f"{half}_n"] = len(s)
                    row[f"{half}"], row[f"{half}_solo"] = ratio(s, num, den)
                row["fit_verdict"] = verdict(k, row["fit"], row["fit_solo"], row["fit_n"])
                row["verdict"] = verdict(k, row["hold"], row["hold_solo"], row["hold_n"])
                out.append(row)
    tab = pd.DataFrame(out).drop_duplicates(["grp", "call", "lens"])
    pd.set_option("display.width", 250, "display.max_colwidth", 42, "display.max_rows", 500)
    print(f"== {'Superflex' if SF else '1QB'}: split-half check of every call group")
    print(tab.verdict.value_counts().to_string())
    print("\n-- calls that fail on the holdout half")
    print(tab[tab.verdict.str.startswith("fails")].round(2).to_string(index=False))
    tab.to_csv(out_dir(SF) / "holdout.csv", index=False)


if __name__ == "__main__":
    main()
