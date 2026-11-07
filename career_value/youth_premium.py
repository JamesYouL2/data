# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "scikit-learn"]
# ///
"""Is the KTC / FantasyCalc youth premium justified over 5 or 10 years?

KTC (from 2021) and FantasyCalc (from 2022) have almost no 5-year and no 10-year outcomes yet.
MFL keeper ADP has them (5-yr: 11 cohorts, 10-yr: 6), so we use it as a bridge:

  justified[band] = sum(actual h-yr VORP) / sum(expected from MFL rank)  in MFL cohorts with outcomes
                    -> how much more (or less) than its MFL price a band actually produced
  charged[band]   = sum(expected at the market's re-ordered rank) / sum(expected at MFL rank)
                    in cohorts both markets cover (2020-2024)
                    -> how much more than MFL the market pays for that band, in MFL value units

The market's ordering is mapped onto MFL's rank slots within the shared player set, so the
total is unchanged and only the re-ordering by band moves value. If charged ~= justified, the
market's age premium (relative to MFL) is about right at horizon h. Implied residual =
justified / charged - 1 (negative = the market overpays that band).

Validation: at h = 1-3 the market's actual residual is known, so we compare implied vs actual.
Veterans only (MFL ADP excludes rookies). Assumes MFL's age pricing in 2010-2015 resembles
2020-2024; checked by splitting MFL's justified ratios by era at h = 3.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from market_residuals import OUT  # outputs/ or outputs/sf/ (LEAGUE=sf)
ROWS = OUT / "market_rows"
MARKETS = ["dd_ktc", "dd_fantasycalc", "dd_dynastyprocess", "fp_dynasty"]


def band(x):
    a = x.age
    pos_age = x.position + " " + pd.cut(a, [0, 24.5, 27.5, 99], labels=["<=24", "25-27", "28+"]).astype(str)
    age = pd.cut(a, [0, 23.5, 25.5, 27.5, 29.5, 99], labels=["<=23", "24-25", "26-27", "28-29", "30+"]).astype(str)
    return pd.concat([pd.DataFrame({"grouping": "age", "level": age, "i": x.index}),
                      pd.DataFrame({"grouping": "pos_x_age", "level": pos_age, "i": x.index})])


def iso_curve(h):
    x = pd.read_parquet(ROWS / f"mfl_dynasty_h{h}.parquet")
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x.mkt_rank, x[f"vorp{h}"])
    return x, iso


def justified(x, iso, h):
    x = x.assign(exp=iso.predict(x.mkt_rank), act=x[f"vorp{h}"])
    b = band(x).merge(x[["exp", "act", "t"]], left_on="i", right_index=True)
    return b.groupby(["grouping", "level"]).agg(n=("act", "size"), act=("act", "sum"), exp=("exp", "sum"),
                                                 cohorts=("t", "nunique"))


def charged(mkt, iso):
    """Re-order MFL's rank slots by the other market within the shared veteran set."""
    m = pd.read_parquet(ROWS / "mfl_dynasty_h1.parquet")[["gsis_id", "t", "mkt_rank", "age", "position"]]
    o = pd.read_parquet(ROWS / f"{mkt}_h1.parquet")
    o = o[o.row_type == "veteran"][["gsis_id", "t", "mkt"]]
    s = m.merge(o, on=["gsis_id", "t"])
    s["slot"] = s.groupby("t").mkt_rank.rank(method="first")
    slots = s.sort_values(["t", "slot"]).groupby("t").mkt_rank.apply(list)
    s["their_rank"] = [slots[t][int(r) - 1] for t, r in
                       zip(s.t, s.groupby("t").mkt.rank(method="first"))]
    s = s.reset_index(drop=True)
    s["e_mfl"], s["e_mkt"] = iso.predict(s.mkt_rank), iso.predict(s.their_rank)
    b = band(s).merge(s[["e_mfl", "e_mkt"]], left_on="i", right_index=True)
    g = b.groupby(["grouping", "level"])[["e_mfl", "e_mkt"]].sum()
    return (g.e_mkt / g.e_mfl).rename("charged"), s.t.nunique(), len(s)


def actual_resid(mkt, h):
    f = OUT / "market_residuals_by_group.csv"
    r = pd.read_csv(f)
    r = r[(r.market == mkt) & (r.horizon == h)]
    r["grouping"] = r.grouping.map({"age_band": "age", "pos_x_age": "pos_x_age"})
    return r.dropna(subset=["grouping"]).set_index(["grouping", "level"]).resid_pct_of_avg_outcome


def main():
    rows = []
    for h in [1, 2, 3, 5, 10]:
        x, iso = iso_curve(h)
        j = justified(x, iso, h)
        j["justified"] = j.act / j.exp
        for mkt in MARKETS:
            c, nc, ns = charged(mkt, iso)
            d = j.join(c, how="inner")
            d["implied_resid"] = d.justified / d.charged - 1
            # the market's own measured residual (veterans + rookies in its top 200; approximate check)
            d["actual_resid"] = actual_resid(mkt, h).reindex(d.index)
            rows.append(d.reset_index().assign(market=mkt, h=h, overlap_cohorts=nc, overlap_players=ns))
    r = pd.concat(rows, ignore_index=True)
    r.round(3).to_csv(OUT / "youth_premium_bridge.csv", index=False)

    # validation of the bridge where the market's own h-year residual exists
    v = r.dropna(subset=["actual_resid"])
    v = v[v.n >= 30]
    print("Bridge check (implied vs actual residual, same market & horizon):")
    for (mkt, h), g in v.groupby(["market", "h"]):
        print(f"  {mkt:18s} h={h}: corr {np.corrcoef(g.implied_resid, g.actual_resid)[0, 1]:+.2f}  "
              f"over {len(g)} bands; mean abs gap {(g.implied_resid - g.actual_resid).abs().mean()*100:.0f} pts")

    for grp in ["age", "pos_x_age"]:
        s = r[(r.grouping == grp) & (r.n >= 30)].copy()
        s["cell"] = (s.implied_resid * 100).round(0).astype(int).map("{:+d}".format)
        print(f"\nImplied residual % by {grp} (negative = market overpays vs what MFL history says the band produced)")
        print(s.pivot_table(index=["level", "market"], columns="h", values="cell", aggfunc="first").fillna("").to_string())
        jj = r[(r.grouping == grp)].drop_duplicates(["level", "h"])
        jj = jj.assign(cell=((jj.justified - 1) * 100).round(0).astype(int).map("{:+d}".format)
                       + "/" + jj.cohorts.astype(str))
        print(f"\nMFL justified adjustment % by {grp} (actual vs MFL-rank expectation / cohorts)")
        print(jj.pivot(index="level", columns="h", values="cell").to_string())
        cc = r[(r.grouping == grp) & (r.h == 1)]
        print(f"\nCharged premium vs MFL ranking, % (2020-2024 shared veterans)")
        print(((cc.pivot(index="level", columns="market", values="charged") - 1) * 100).round(0).to_string())

    # era stability of MFL's age pricing (h = 3)
    x, iso = iso_curve(3)
    x["era"] = np.where(x.t <= 2015, "2010-2015", "2016-2022")
    out = {}
    for era, g in x.groupby("era"):
        jj = justified(g, iso, 3)
        out[era] = ((jj.act / jj.exp - 1) * 100).round(0)
    print("\nMFL justified adjustment at h=3 by era (stability check):")
    print(pd.DataFrame(out).loc["age"].to_string())


if __name__ == "__main__":
    main()
