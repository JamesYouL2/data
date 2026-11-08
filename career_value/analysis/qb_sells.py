"""Q: are the site's QB Sell calls right, for a rebuilder, a contender and the default mode, in 1QB and superflex?

Same measure as sell_strength.py, at several horizons. For FP dynasty top-200 QBs (May snapshots after season t):
    hold_h = VORP over seasons t+1..t+h + value at t+h
(value = expected 3-year VORP at that FP rank; 0 outside the top 200), and fair_h = expected hold_h at his FP rank
at t over all players, i.e. what a fair-price trade gets on average. keep_h = sum(hold_h) / sum(fair_h).
    prod1: year-1 VORP alone against the year-1 VORP of the average player at his price (a contender's lens).
    keep1: play him one season, then hold or sell (the "Use, then sell" lens).
    keep3, keep5: a rebuilder's lens (keep5 has 2 cohorts, 2018-19).
Cohorts: h=1 2018-23, h=3 2018-21, h=5 2018-19. Groups are rules.base_calls groups (the backtested calls) and the
calls each mode now gives (rules.classify). "start" = PPG rank in season t inside the starter line (QB12, QB24 in SF),
which is what the default mode keys on. LEAGUE=sf for superflex.

Findings (2026-11, before the rule changes they led to):
- QBs as a whole beat their FP price at every horizon (keep3 1.37 1QB, 1.19 SF; non-QBs 0.96 / 0.93).
- Most QB sells still hold up for the typical player: 26-35% of sold QBs beat their price; group means above 1 come
  from a few hits (Brady 2018-20 alone lifts 1QB "QB 36+" from 0.64 to 1.58). Kept as light sells.
- Two failed in both the mean and the share: SF "Young QB priced as a QB1" (keep3 1.18, keep5 1.40, 64% beat; still
  1.09 without the best) and "QB 30+, mid-tier" (keep3 1.27 / 1.25, 67% / 78% beat; 1.08 / 1.16 without the best,
  though only ~5 distinct QBs). Both are now Hold in every mode.
- Contender / default "Use, then sell" on QBs: prod1 1.66 (1QB) / 1.58 (SF) backs the use, keep5 0.53 / 1.00 the sell.
- Not changed, thin or mixed: SF "QB 26-29, 3rd-round price" Buy (keep3 0.46 over 2018-21, 1 of 8 beat, but keep1 2.01
  with the 2022-23 cohorts); "QB 30+, long-term" Buy (prod1 1.13 / 1.27 backs it for a contender, keep3 0.76 / 0.90
  costs a little dynasty value); SF rookie QB1 Sell (no rookies in the sample).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import out_dir, rows  # noqa: E402
import sell_strength as ss  # noqa: E402

R, SF, F = ss.R, ss.SF, ss.F


def horizon(h, v3):
    x = rows(["fp_dynasty"], h, SF)
    r1 = rows(["fp_dynasty"], 1, SF)[["gsis_id", "t", "mkt_rank"]]
    x = x.merge(r1.assign(t=r1.t - h).rename(columns={"mkt_rank": "rank_h"}), on=["gsis_id", "t"], how="left")
    x = x[x.t + h <= r1.t.max()].reset_index(drop=True)
    x["hold"] = x[f"vorp{h}"] + np.where(x.rank_h.notna(), v3.predict(np.log(x.rank_h.fillna(200))), 0.0)
    x["fair"] = ss.iso(x.mkt_rank, x.hold).predict(np.log(x.mkt_rank))
    if h == 1:
        x["v1fair"] = ss.iso(x.mkt_rank, x.vorp1).predict(np.log(x.mkt_rank))
    return x


def main():
    x3 = rows(["fp_dynasty"], 3, SF)
    v3 = ss.iso(x3.mkt_rank, x3.vorp3)
    H = {h: horizon(h, v3) for h in (1, 3, 5)}
    base = ss.calls(H[1])                                  # h=1 has every cohort 2018-23; calls are per (player, t)
    pr = ss.prod_ranks()
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
    print(f"\n== {'Superflex' if SF else '1QB'} QBs, FP top 200. keep = hold / fair (1 = the average player at his price)")
    print(tab.round(2).to_string())
    print("\n-- calls per group (count)")
    print(calls.to_string())
    tab.to_csv(out_dir(SF) / "qb_sells.csv")


if __name__ == "__main__":
    main()
