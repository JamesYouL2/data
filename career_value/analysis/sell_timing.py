"""Q: the rebuilder's side -- when to sell?

1. Price retention: what share of a player's trade price is left 1 and 2 years later, by position x age?
   KTC / FantasyCalc / DP: value share = raw trade value / sum of that date's top-150 values (removes scale drift);
   retention = sum(share at May t+1) / sum(share at May t) over players top-100 overall at t (dropped = 0).
   MFL keeper ADP (2010-24): ADP rank mapped to a price-implied value (isotonic h3 VORP on rank), same retention.
2. In-season price path (DD, cycles 2021-24): share at Sep 1 / Oct 18 / Nov 17 / Dec 22 / Feb 17 / May Y+1
   relative to May Y, by position x age. When in the year is a player's price highest?
3. Is the market wrong about them? h1/h2/h3 residual % by position x age (same buckets).
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import DD, DYN, HERE, rows  # noqa: F401
import market_residuals as mr  # noqa: E402

TOP = 100


def agegrp(pos, age):
    if pos == "RB":
        b = [0, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 99]
        lab = ["<=23", "24", "25", "26", "27", "28", "29+"]
    elif pos == "QB":
        b, lab = [0, 25.5, 28.5, 31.5, 34.5, 99], ["<=25", "26-28", "29-31", "32-34", "35+"]
    else:
        b, lab = [0, 23.5, 25.5, 27.5, 29.5, 31.5, 99], ["<=23", "24-25", "26-27", "28-29", "30-31", "32+"]
    return pd.cut(age, b, labels=lab).astype(str)


def with_age(x, panel):
    x = x.merge(panel[["gsis_id", "t", "position", "age", "row_type"]], on=["gsis_id", "t"])
    x["ag"] = pd.concat([agegrp(p, g.age) for p, g in x.groupby("position")]).reindex(x.index)
    return x


def retention(cur, nxt, keys=("position", "ag")):
    """cur: gsis_id, t, v (+keys); nxt: gsis_id, t, v. ratio of summed value; median per-player ratio."""
    out = []
    for lag in (1, 2):
        n = nxt.assign(t=nxt.t - lag).rename(columns={"v": "v2"})
        z = cur.merge(n[["gsis_id", "t", "v2"]], on=["gsis_id", "t"], how="left")
        z = z[z.t + lag <= nxt.t.max()]
        z["v2"] = z.v2.fillna(0)
        g = z.groupby(list(keys) + ["src"]).apply(lambda d: d.v2.sum() / d.v.sum(), include_groups=False)
        med = z.assign(q=z.v2 / z.v).groupby(list(keys)).q.median()
        n_ = z.groupby(list(keys)).gsis_id.size()
        out.append(pd.DataFrame({f"keep {lag}y": g.groupby(list(keys)).mean(), f"median {lag}y": med,
                                 f"n {lag}y": n_}))
    return pd.concat(out, axis=1)


def main():
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    # ---- 1a. DD raw values (May snapshots)
    dd = mr.dd_markets(panel)
    cur, nxt = [], []
    for k in DD:
        m = dd[k].assign(v=lambda d: -d.mkt)
        m["share"] = m.v / m.groupby("t").v.transform(lambda s: s.nlargest(150).sum()) * 100
        m["rank"] = m.groupby("t").v.rank(ascending=False, method="first")
        nxt.append(m.assign(src=k)[["gsis_id", "t", "share", "src"]].rename(columns={"share": "v"}))
        cur.append(m[m["rank"] <= TOP].assign(src=k)[["gsis_id", "t", "share", "src"]].rename(columns={"share": "v"}))
    cur, nxt = with_age(pd.concat(cur), panel), pd.concat(nxt)
    ret_dd = pd.concat([retention(cur[cur.src == k], nxt[nxt.src == k]).assign(src=k) for k in DD])
    ret_dd = ret_dd.groupby(level=[0, 1]).agg({"keep 1y": "mean", "median 1y": "mean", "n 1y": "sum",
                                               "keep 2y": "mean", "median 2y": "mean", "n 2y": "sum"})
    # ---- 1b. MFL ADP -> implied value
    mfl = mr.mfl_dynasty(panel)[["gsis_id", "t", "mkt"]].dropna().drop_duplicates(["gsis_id", "t"])
    mfl["rank"] = mfl.groupby("t").mkt.rank(method="first")
    h3 = rows(["mfl_dynasty"], 3)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(h3.mkt_rank, h3.vorp3)
    mfl["v"] = np.where(mfl["rank"] <= 200, iso.predict(mfl["rank"].clip(upper=200)), 0)
    mc = with_age(mfl[mfl["rank"] <= TOP].assign(src="mfl"), panel)
    ret_mfl = retention(mc, mfl.assign(src="mfl"))
    for lab, r in (("KTC/FC/DP value share, May->May (cohorts 2020-24)", ret_dd),
                   ("MFL keeper ADP -> implied value (cohorts 2010-24)", ret_mfl)):
        print(f"\n== Price retention, top-{TOP} players at t: {lab}")
        print("   keep = sum(value t+lag)/sum(value t), median = median player ratio")
        for p in ["QB", "RB", "WR", "TE"]:
            print(p); print(r.loc[p].round(2).to_string())

    # ---- 2. in-season path (cycle Y: May Y .. May Y+1)
    ins = {}
    for pat in ("snapshot_*.json", "inseason_*.json"):
        for k, v in mr.dd_markets(panel, pat).items():
            if k in DD:
                ins.setdefault(k, []).append(v)
    path = []
    labels = {"05-15a": "0 May Y", "09-01": "1 Sep 1", "10-18": "2 Oct 18", "11-17": "3 Nov 17", "12-22": "4 Dec 22",
              "02-17": "5 Feb 17", "05-15b": "6 May Y+1"}
    for k, vs in ins.items():
        m = pd.concat(vs).assign(v=lambda d: -d.mkt)
        m["share"] = m.v / m.groupby("snap_date").v.transform(lambda s: s.nlargest(150).sum()) * 100
        for Y in range(2021, 2025):
            dates = {f"{Y}-05-15": "0 May Y", f"{Y}-09-01": "1 Sep 1", f"{Y}-10-18": "2 Oct 18",
                     f"{Y}-11-17": "3 Nov 17", f"{Y}-12-22": "4 Dec 22", f"{Y + 1}-02-17": "5 Feb 17",
                     f"{Y + 1}-05-15": "6 May Y+1"}
            s = m[m.snap_date.isin(dates)].assign(snap=lambda d: d.snap_date.map(dates))
            base = s[s.snap == "0 May Y"]
            base = base[base.v.rank(ascending=False) <= TOP][["gsis_id", "share"]].rename(columns={"share": "s0"})
            w = s.pivot_table(index="gsis_id", columns="snap", values="share").reindex(columns=sorted(set(dates.values())))
            w = base.set_index("gsis_id").join(w).fillna(0)
            if w.shape[1] < 8 or (w.drop(columns="s0") == 0).all().any():
                continue
            w = w.drop(columns="s0").reset_index().assign(src=k, Y=Y, t=Y - 1)
            path.append(w)
    P = with_age(pd.concat(path, ignore_index=True), panel)
    snaps = sorted(set(labels.values()))
    print(f"\n== In-season price path (KTC/FC/DP value share, top-{TOP} at May Y, cycles 2021-24): "
          "sum(share at date) / sum(share at May Y)")
    print("   (age = age at May Y; 'rookies' = that May's rookie class)")
    P["ag2"] = np.where(P.row_type == "rookie", "rookie", P.ag)
    for p in ["QB", "RB", "WR", "TE"]:
        g = P[P.position == p].groupby("ag2")
        tab = g[snaps].sum().div(g["0 May Y"].sum(), axis=0).round(2)
        tab["n"] = g.size()
        tab["peak"] = tab[snaps].idxmax(axis=1).str[2:]
        print(p); print(tab.to_string())

    # ---- 3. residual by age
    print("\n== Residual % by position x age (is the market wrong about them?)  mean (cohorts positive/total)")
    for fam, mks, hs in (("Dynasty pooled", DYN, (1, 2, 3)), ("MFL", ["mfl_dynasty"], (1, 2, 3, 5))):
        out = {}
        for h in hs:
            x = rows(mks, h)
            x = x[x.mkt_rank <= TOP].copy()
            x["ag"] = pd.concat([agegrp(p, g.age) for p, g in x.groupby("position")]).reindex(x.index)
            out[f"h{h}"] = x.groupby(["position", "ag"]).apply(
                lambda d: f"{d.r.mean():+.0f} ({(d.groupby('t').r.mean() > 0).sum()}/{d.t.nunique()}) n{len(d)}",
                include_groups=False)
        print(f"{fam} (top-{TOP} priced)"); print(pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
