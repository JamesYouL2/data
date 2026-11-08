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
  halves) is now Hold (to summer). League sim, ranks 4-8: Contending 5-year titles 20.2 -> 23.2, value 3 years on
  186 -> 220; Default and Rebuilding unchanged. "Use, then sell" did worse than holding (titles yrs 1-3 15.4 vs 15.9,
  value 3 years on 205 vs 220). Superflex keeps the sell (it held). The Hold shows here as "fails (sell?)": contender
  holds are scored on year-1 production (prod1 0.41), and both ways of selling did worse in the sim.
- 1QB "RB 30+, fading" Cut (2.03 / 0.68) stays: the 201-300 ones all returned 0 (14 of 14; 9 of 9 in superflex); the
  holdout gain is two waiver-level RBs past 300 (Cordarrelle Patterson 2020, Jerick McKinnon 2021) on prices near 0.
- Superflex: no action call fails.

Bucket audit (2026-11): every call group against "young: keep unless a sale returns more; old: contenders buy,
rebuilders sell". Each candidate change ran through the league sim (middle_mode.py) and was kept only if the gain held
in both the 2018-20 and the 2021-24 cohorts (sim deltas per team, clustered by league):
- Adopted: 1QB contender Buy on "Young WR, NFL R1-3" 101-200 (prod1 1.58 / 1.41 without the best, n=57). Contending
  and Default value 3 years on +65-93; by cohort, Contending +73 / +72, Default +80 / +52. Year-1 titles rose only in
  2018-20 (+2.3 / 0.0).
- Adopted: superflex rebuilder Buy on rookie WRs 51-100 (keep3 1.62 / 1.31, n=12). Default value 3 years on +160 /
  +167 and titles yrs 1-3 +7.8 / +2.3 by cohort; Rebuilding 5-year titles +13 (2018-20), value 3 years on -14 ± 27 in
  2021-24.
- Dropped, gain only in 2018-20: 1QB rookie WRs 51-100 rebuilder Buy (Rebuilding value 3 years on +17 / -107, Default
  +169 / -28); 1QB Day-2+ rookie RBs 101-200 rebuilder Buy (+159 / -325); selling young Day-3 WRs 101-200 (both
  formats; Default and Contending -52 to -71 in 2021-24). The rookie RB Hold shows here as "fails (buy?)" (keep3
  1.61 / 1.71); the sim says the Buy lost on the later cohorts.
- Both adopted buys hold on the holdout half (1QB Contending prod1 1.50, Default keep1 1.32; superflex Rebuilding
  keep3 1.28, n=8).
- Dropped outright: superflex rebuilder Buy on young Day-3 WRs 201-300; Lottery tickets on young TEs 201-300 (about
  neutral).
- Exceptions kept: contender Cut on fading 30+ RBs / WRs / TEs 201-300 (prod1 2.15 on WRs is Thielen 2022 and Beasley
  2019, 2 of 26); contender Hold, not Buy, on old 1QB QBs (later half 0.56 / 0.00).
- Superflex RBs and WRs: no group besides rookie WRs 51-100 beat its price in both halves; the young NFL R1-3 WRs
  101-200 that 1QB contenders now buy had prod1 1.09 / 0.98 in superflex.

Lottery tickets (2026-11): buy, hold or sell? Sims as above, ranks 4-8, by cohort 2018-20 / 2021-24:
- No buy. Rebuilder Buy on rookie WRs 201-300 (keep3 3.82 1QB, 3.93 SF; 2018 McLaurin and Diontae Johnson at 17-33x):
  Rebuilding value 3 years on +182 / -267 (1QB), +148 / -88 (SF). SF Day-3 rookie RBs 201-300 (keep3 2.68, 0.47 in
  2018-19): +26 / -109.
- Adopted: rebuilder Sell on young Day-3 RBs 201-300 (keep3 0.53 / 0.56; 2020-21 0.03 / 0.05). Rebuilding value 3
  years on +43 / +35 (1QB), +52 / +45 (SF); Default +57 / +68, +70 / +115. Titles flat.
- Adopted: rebuilder Sell on rookie TEs 201-300 (keep3 0.54 / 0.74; 0.18 / 0.29 without Dawson Knox 2018). Rebuilding
  +74 / +111 (1QB), +31 / +57 (SF); Default +23 / +17 and +8 / -8. Contenders sell them too unless producing (the
  contender rule; Contending +19 / +28 in 1QB, 0 / -8 in SF) and keep the RBs as Lottery tickets.
- Adopted, 1QB only: rebuilder Buy on rookie TEs 101-200 (keep3 1.84 / 0.96 without the best, n=8; 5 of 8 beat their
  price). Rebuilding value 3 years on +63 / +53, 5-year titles +5.1 (2018-20), 3-year titles +4.0 (2021-24), year-1
  -0.9 / -0.3; Default +18 / +8. Superflex gained only in 2018-20 (+108 / +2, 3-year titles -8.0 in 2021-24). A
  contender Buy added little (1QB Contending +10 / +19; prod1 0.81 would make it a Hold).
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
