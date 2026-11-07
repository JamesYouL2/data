"""Q: is consolidating 2-for-1 worth it?

KTC / FantasyCalc / DP raw trade values (market_rows: mkt = -value), as a share of each cohort's top-150 value sum.
  price(rank)   : isotonic value share on overall market rank (pooled cohorts)
  outcome(rank) : isotonic h-year VORP on the same rank
For a stud at rank R: the player(s) with price(R)/2 (pair) and price(R)/3 (trio). Ratio = outcome(R) / (2 x
outcome(pair rank)) etc. > 1: the stud returns more VORP than the package of equal raw value -> consolidating wins
(and the trade calculator's 'value adjustment' for the 1-side is justified). Also VORP per unit of value by tier.
Caveat: VORP counts only above replacement (RB31/WR43/TE13/QB13) so it roughly measures starter value; the pair
also uses an extra roster spot and its second player may sit on your bench, which favours the stud further.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import DD, rows

STUDS = (1, 3, 6, 12, 24, 36, 48)
TIERS = [0, 6, 12, 24, 48, 100, 200]


def curves(x, y):
    p = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x.mkt_rank, x.share)
    o = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x.mkt_rank, x[y])
    grid = np.arange(1, 201)
    return pd.DataFrame({"rank": grid, "price": p.predict(grid), "out": o.predict(grid)})


def match(c, target):
    """first rank whose price is <= target"""
    i = np.searchsorted(-c.price.to_numpy(), -target, side="left")
    return int(c["rank"].iloc[min(i, len(c) - 1)])


def main():
    for sf in (False, True):
        lg = "SF" if sf else "1QB"
        res, per = [], []
        for h in (1, 2, 3):
            for m in DD:
                x = rows([m], h, sf)
                x["v"] = -x.mkt
                x["share"] = x.v / x.groupby("t").v.transform(lambda s: s.nlargest(150).sum()) * 100
                c = curves(x, f"vorp{h}")
                for R in STUDS:
                    v, e = c.price.iloc[R - 1], c.out.iloc[R - 1]
                    r2, r3 = match(c, v / 2), match(c, v / 3)
                    res.append({"h": h, "mk": m, "stud rank": R, "pair rank": r2, "trio rank": r3,
                                "stud/pair": e / (2 * c.out.iloc[r2 - 1]), "stud/trio": e / (3 * c.out.iloc[r3 - 1])})
                x["tier"] = pd.cut(x.mkt_rank, TIERS).astype(str)
                g = x.groupby("tier")
                per.append(pd.DataFrame({"h": h, "mk": m, "vorp per value": g[f"vorp{h}"].sum() / g.share.sum()}))
        r = pd.DataFrame(res)
        print(f"\n== {lg}: stud vs equal-raw-value package (KTC/FC/DP 2019-24; >1 = stud returns more VORP)")
        print("   ranks are overall market rank; pair/trio = rank whose value is 1/2 or 1/3 of the stud's (median of 3 markets)")
        t = r.groupby(["stud rank", "h"]).agg({"pair rank": "median", "trio rank": "median", "stud/pair": "mean",
                                               "stud/trio": "mean"}).unstack("h")
        print(t.round(2).to_string())
        p = pd.concat(per).reset_index()
        p["rel"] = p["vorp per value"] / p.groupby(["h", "mk"])["vorp per value"].transform(
            lambda s: s[p.loc[s.index, "tier"] == "(24, 48]"].iloc[0])
        order = [f"({a}, {b}]" for a, b in zip(TIERS[:-1], TIERS[1:])]
        print(f"\n== {lg}: VORP per unit of raw trade value by market-rank tier (each market relative to its ranks 25-48)")
        print(p.pivot_table(index="tier", columns=["h", "mk"], values="rel").reindex(order).round(2).to_string())
        # by market, to show the calculator spread
        print(f"\n== {lg}: stud/pair at h=2 by market")
        print(r[r.h == 2].pivot(index="stud rank", columns="mk", values="stud/pair").round(2).to_string())


if __name__ == "__main__":
    main()
