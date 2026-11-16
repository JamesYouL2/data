"""Q: is "rest-of-season rank well below the market's rank" a sell call, and does it need an injury control?

Signal: a player on FP's rest-of-season list whose positional rank there is a quarter or more worse than his positional rank in the
trade market (KTC or FantasyCalc) and at least 3 places worse. Snapshots after weeks 4, 8 and 12, market top POOL, seasons from 2021.
1. Players (`check`): what they returned against the market's price (isotonic on the market rank): rest-of-season VORP, value at
   next May's market rank, and the one-year hold (inseason_movers.py), split by hurt (IR / reserve / Out that week) or healthy, by
   age, by how large the gap is, and by whether the site's rules already sell him.
2. The league sim of market_price_sim.py at market prices. The site's calls (live), then the same with the signal laid on top the
   way the site applies a sell signal: Hold and Use, then sell become Sell, Buy becomes Hold, Sell and Cut stay.
     low         every flagged player           low_well / low_hurt   healthy / hurt ones only
     low_y / low_o   25 and under / 26+         low150                the gap at a half or more
     rand        control: as many Hold or Use players on the rest-of-season list turned to Sell, picked at random
Caveats as market_price_sim.py: few cohorts, and KTC and FantasyCalc share players and seasons.
Writes outputs/[sf/]ros_low_sim_<source>.csv.gz. LEAGUE=sf; SOURCE=ktc or fantasycalc; REPS, WEEKS, JOBS.
"""
import os
import pathlib
import sys

os.environ["ARMS"] = "live,nosig"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import single_model_sim as sm  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

mps, ss, MODES = sm.mps, sm.ss, sm.MODES
ARMS = ("live", "low", "low_well", "low_hurt", "low_y", "low_o", "low150", "rand")
MOVE, FLOOR = 4 / 3, 3


def flags(ros_p, mkt_p, on_list):
    ratio = ros_p / mkt_p
    return on_list & (ratio >= MOVE) & (ros_p - mkt_p >= FLOOR), ratio


def check():
    pd.set_option("display.width", 250, "display.max_rows", 500)
    z = sm.panel()
    on = z.ros_rank.notna() & (z.ros_rank < 500)
    z["low"], z["ratio"] = flags(z.ros_prank, z.mpr, on)
    for y in ("ros", "mv1", "hold1"):
        d = z[z[y].notna()]
        z.loc[d.index, "r_" + y] = d[y] - IsotonicRegression(increasing=False, out_of_bounds="clip").fit(d.mlog, d[y]).predict(d.mlog)
    z["health"] = np.where(~on, "off the list", np.where(z.hurt == 1, "hurt", "healthy"))
    z["age_c"] = np.where(z.row_type == "rookie", "rookie", np.where(z.age < 26, "25-", np.where(z.age < 30, "26-29", "30+")))
    z["size"] = np.where(~z.low, "not flagged", np.where(z.ratio >= 2, "2x or more", np.where(z.ratio >= 1.5, "1.5-2x", "1.33-1.5x")))
    z["rules"] = np.where(z.kl_mid == "sell", "site sells (default)", "site doesn't")
    z["signal"] = np.where((z.kl_mid == "sell") & (z.kn_mid != "sell"), "site's sell signal fires", "no site sell signal")

    def cell(g, c):
        g = g.dropna(subset=[c])
        if g.drop_duplicates(["gsis_id", "season"]).shape[0] < 15:
            return ""
        by = g.groupby("season")[c].mean()
        return f"{g[c].mean():+6.1f} {(g[c] > 0).mean() * 100:2.0f}% {(by < 0).sum()}/{len(by)}"

    def table(q, by, title):
        t = pd.DataFrame({(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): {**{c[2:]: cell(g, c) for c in ("r_ros", "r_mv1", "r_hold1")},
                                                                                 "player-weeks": len(g), "a snapshot": round(len(g) / q.groupby(["season", "target"]).ngroups, 1)}
                          for k, g in q.groupby(by)}).T
        print(f"-- {title}\n{t.to_string()}")
    print(f"== {'Superflex' if sm.SF else '1QB'}, {mps.SOURCE}: mean return less the market's fair value, share above it, seasons below zero / seasons. "
          "ros = rest-of-season VORP, mv1 = value at next May's market rank (340 - rank), hold1 = one-year hold")
    table(z, "health", "everyone, by health (for scale)")
    table(z[on], ["low", "health"], "on the rest-of-season list: flagged or not, by health")
    f = z[z.low]
    table(z[on], "size", "by the size of the gap")
    table(f, "age_c", "flagged, by age")
    table(f, "position", "flagged, by position")
    table(f, "target", "flagged, by week")
    table(f, "rules", "flagged: does the site's default call already sell him?")
    table(f, "signal", "flagged: does one of the site's sell signals fire?")
    table(f, ["health", "rules"], "flagged, by health and the site's default call")
    nf = z[on & ~z.low]
    table(nf, ["health"], "not flagged, by health (what an injury alone does)")


def snapshot_calls(q, last, w, mk, mkpre):
    c = ss.snapshot_calls(q, last, w, mk, mkpre)
    on = q.ros_rank.notna() & (q.ros_rank < 500)
    low, ratio = flags(q.ros_prank, q.gsis_id.map(mk["prank"]), on)
    hurt, young = q.hurt.astype(bool), q.age < 26
    pick = {"low": low, "low_well": low & ~hurt, "low_hurt": low & hurt, "low_y": low & young, "low_o": low & ~young, "low150": low & (ratio >= 1.5)}
    rng = np.random.default_rng(int(q.season.iloc[0]) * 100 + int(q.target.iloc[0]))
    out = {}
    for m in MODES:
        live = c["live", m]
        out["live", m] = live

        def lay(ids):
            k = dict(live)
            for g in ids:
                k[g] = {"hold": "sell", "use": "sell", "buy": "hold"}.get(k.get(g), k.get(g))
            return k
        for arm, mask in pick.items():
            out[arm, m] = lay(q.gsis_id[mask])
        n = sum(live.get(g) in ("hold", "use") for g in q.gsis_id[low])
        pool_ = [g for g in q.gsis_id[on & ~low] if live.get(g) in ("hold", "use")]
        out["rand", m] = lay(rng.choice(pool_, size=min(n, len(pool_)), replace=False))
    return out


def summarize(df):
    ss.ARMS = ARMS
    ss.summarize(df)
    key = ["t", "rep", "wk", "rank"]
    own = {m: list(r) for m, r in mps.LENS.values()}
    z = df[[r in own.get(m, ()) for r, m in zip(df["rank"], df["mode"])]].copy()
    z["yr1"] = z.p1 * 100
    z["yrs3"] = np.where(z.t + 3 <= mps.su.LAST, (z.p1 + z.p2 + z.p3) * 100, np.nan)
    z = z.set_index(key + ["mode", "arm"])[["yr1", "yrs3", "mval1", "mval3"]]

    def cell(v):
        v = v.dropna()
        if v.empty:
            return ""
        c, by = v.groupby(level=["t", "rep", "wk"]).mean(), v.groupby(level="t").mean()
        return f"{c.mean():+6.1f}±{1.96 * c.std(ddof=1) / np.sqrt(len(c)):4.1f} {int((by > 0).sum())}/{int((by != 0).sum())}"
    print("\n=== Against the random control (as many Hold or Use players sold)")
    for arm in ("low", "low_well", "low_hurt"):
        d = z.xs(f"m:{arm}", level="arm") - z.xs("m:rand", level="arm")
        print(f"\n-- {arm} - rand")
        print(pd.DataFrame([{"mode": m, **{c: cell(d.xs(m, level="mode")[c]) for c in d.columns}} for m in MODES]).to_string(index=False))
    print("\n=== By cohort: low - live")
    d = z.xs("m:low", level="arm") - z.xs("m:live", level="arm")
    print(d.groupby(level=["mode", "t"]).mean().round(1).to_string())


mps.OUT = "ros_low_sim"
mps.F_ARMS, mps.M_ARMS, mps.RANDOM, mps.KIND = (), ARMS, (), ()
mps.snapshot_calls, mps.summarize = snapshot_calls, summarize

if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
    else:
        mps.main()
