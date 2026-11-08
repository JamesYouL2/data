"""Q: are the site's QB Sell calls right, for a rebuilder, a contender and the default mode, in 1QB and superflex?

Same measure as call_strength.py, at several horizons. For FP dynasty top-340 QBs (May snapshots after season t):
    hold_h = VORP over seasons t+1..t+h + value at t+h
(value = expected 3-year VORP at that FP rank; 0 outside the top 340), and fair_h = expected hold_h at his FP rank
at t over all players, i.e. what a fair-price trade gets on average. keep_h = sum(hold_h) / sum(fair_h).
    prod1: year-1 VORP alone against the year-1 VORP of the average player at his price (a contender's lens).
    keep1: play him one season, then hold or sell (the "Use, then sell" lens).
    keep3, keep5: a rebuilder's lens (keep5 has 2 cohorts, 2018-19).
Cohorts: h=1 2018-23, h=3 2018-21, h=5 2018-19. Groups are rules.base_calls groups (the backtested calls) and the
calls each mode now gives (rules.classify). "start" = PPG rank in season t inside the starter line (QB12, QB24 in SF),
which is what the default mode keys on. LEAGUE=sf for superflex.

Findings (2026-11, FP top 340; the first pass at top 200 led to the two Holds below):
- QBs as a whole beat their FP price at every horizon (keep3 1.39 1QB, 1.18 SF; non-QBs 0.96 / 0.94).
- Most QB sells rest on a few hits; 26-35% of sold QBs beat their price. Kept as light sells. At top 340, two 1QB
  groups look even less like sells: "QB 26-29" (keep3 1.78) and "QB 36+" (1.43; Brady 2018-20 carries it). Both
  stay light sells: call_strength.py marks them "close call".
- Two failed in both the mean and the share, and are Hold in every mode: SF "Young QB priced as a QB1" (keep3 1.18,
  keep5 1.40) and "QB 30+, mid-tier" (keep3 1.23 1QB / 1.18 SF, only ~5 distinct QBs).
- Contender / default "Use, then sell" on QBs: prod1 1.65 (1QB) / 1.43 (SF) backs the use; keep5 0.03 / 0.86 backs
  the sell.
- Buys by each mode's lens (call_strength.py): SF "QB 26-29, 3rd-round price" prod1 2.47, keep1 1.30, keep3 0.51:
  strong for a contender, weak for Default, Hold for a rebuilder. "QB 30+, long-term" prod1 1.12 / 1.27, keep1
  0.87 / 0.98: Hold in 1QB Default, weak elsewhere. SF rookie QB1 Sell: no rookies in the sample.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import out_dir  # noqa: E402
import call_strength as cs  # noqa: E402

R, SF, F = cs.R, cs.SF, cs.F


def main():
    H = cs.horizons((1, 3, 5))                             # FP top 340, as call_strength.py
    base = cs.calls(H[1])                                  # h=1 has every cohort 2018-23; calls are per (player, t)
    pr = cs.prod_ranks()
    line = R.START[F]["QB"]
    base["start"] = [pd.notna(pr[4].get(t, {}).get(g)) and pr[4][t][g] <= line for g, t in zip(base.gsis_id, base.t)]
    key = base.set_index(["gsis_id", "t"])[["grp", "reb", "con", "mid", "band", "start"]]
    for h in (3, 5):
        H[h] = H[h].join(key, on=["gsis_id", "t"])

    def score(mask_fn, label):
        out = {"group": label}
        q1 = base[mask_fn(base)]
        out["n"] = len(q1)
        out["prod1"] = q1.vorp1.sum() / q1.v1fair.sum() if len(q1) else np.nan
        out["keep1"] = q1.hold.sum() / q1.fair.sum() if len(q1) else np.nan
        coh = q1.groupby("t")[["hold", "fair"]].sum()
        out["keep1<1 coh"] = f"{(coh.hold < coh.fair).sum()}/{len(coh)}"
        for h in (3, 5):
            q = H[h][mask_fn(H[h])]
            out[f"n{h}"] = len(q)
            out[f"keep{h}"] = q.hold.sum() / q.fair.sum() if len(q) and q.fair.sum() > 0 else np.nan
            coh = q.groupby("t")[["hold", "fair"]].sum()
            out[f"keep{h}<1 coh"] = f"{(coh.hold < coh.fair).sum()}/{len(coh)}"
        return out

    qb = lambda d: d.position == "QB"  # noqa: E731
    res = [score(lambda d: d.position != "QB", "all non-QBs"), score(qb, "all QBs")]
    for lab, m in (("QB, starting", lambda d: qb(d) & d.start), ("QB, not starting", lambda d: qb(d) & ~d.start)):
        res.append(score(m, lab))
    for g in sorted(base[qb(base)].grp.dropna().unique()):
        res.append(score(lambda d, g=g: qb(d) & (d.grp == g), g))
        for st in (True, False):
            res.append(score(lambda d, g=g, st=st: qb(d) & (d.grp == g) & (d.start == st), f"  {g} | {'starting' if st else 'not starting'}"))
    sells = base[qb(base) & base.reb.map(R.kind).eq("sell")].grp.unique()
    for g in sorted(sells):                                  # just the players a rebuilder is told to sell
        m = lambda d, g=g: qb(d) & (d.grp == g) & d.reb.map(R.kind).eq("sell")  # noqa: E731
        res.append(score(m, f"{g} [rebuilder sells]"))
        for st in (True, False):
            res.append(score(lambda d, m=m, st=st: m(d) & (d.start == st), f"  {g} [rebuilder sells] | {'starting' if st else 'not starting'}"))
    for col, mode in (("reb", "rebuilder"), ("con", "contender"), ("mid", "default")):
        for k in ("sell", "hold", "buy", "use", "cut"):
            res.append(score(lambda d, col=col, k=k: qb(d) & d[col].map(R.kind).eq(k), f"[{mode} call: {k}]"))
    tab = pd.DataFrame(res).set_index("group")
    tab = tab[tab.n >= 3]
    calls = base[qb(base)].groupby("grp")[["reb", "con", "mid"]].agg(lambda s: "/".join(f"{a}:{b}" for a, b in s.value_counts().items()))
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(f"\n== {'Superflex' if SF else '1QB'} QBs, FP top 340. keep = hold / fair (1 = the average player at his price)")
    print(tab.round(2).to_string())
    print("\n-- calls per group (count)")
    print(calls.to_string())
    tab.to_csv(out_dir(SF) / "qb_sells.csv")


if __name__ == "__main__":
    main()
