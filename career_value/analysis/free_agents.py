"""Which free agents are worth a roster spot? Backtest of the site's free-agent list (pages/app.js).

Two questions:
  1. Plays now: is a free agent ranked inside some multiple of the starter line on FantasyPros' rest-of-season (ROS) list worth
     adding for this season, and where should the line be (1x, 1.25x, 1.5x, 2x the starter line)?
  2. Free dynasty value: is an unrostered player worth adding for his dynasty price, and down to which price?

Panel: inseason_movers.py's sources (FP dynasty rank and the ROS list at weeks 4 / 8 / 12, 2019-25, the injury report), but
without its top-340 cap: every player on the dynasty list (about 280 deep in 2019, 400-460 later) or the ROS list (about 370).
Not on the dynasty list = unranked.

Who is free: nobody knows who was rostered, so a roster line L stands in. Players priced past L (or unranked) are free agents;
the player you'd drop for one is a rostered player just inside the line, priced L-49 to L. L = 200 / 250 / 300.
Each free agent is scored against the average of that drop tier at the same snapshot:
  ros   = rest-of-season VORP                                              Contending's lens, 2019-25
  hold1 = ros + next season's VORP + value at the May after it              Default's lens, 2019-24
  hold3 = ros + the next three seasons' VORP + value at the May after them  Rebuilding's lens, 2019-22
(VORP and values as inseason_movers.py: replacement QB13/RB31/WR43/TE13, or the superflex fill; value at May = expected
3-season VORP at that FP rank, zero past 340.) Cells: mean outcome vs drop tier, difference +- 2 standard errors (player-cluster
bootstrap, 400 draws; * = beyond them), seasons where the group beat the drop tier, distinct players.
Plays-now rows also give the share with any rest-of-season VORP and the share of the remaining weeks spent inside the starter
line that week (QB12 or QB24 in superflex, RB24, WR30, TE12). Healthy = not on reserve or ruled Out at the snapshot.

  [LEAGUE=sf] uv run --with pandas --with pyarrow --with scikit-learn --with scipy --with statsmodels --with rapidfuzz \
      python analysis/free_agents.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import call_strength as cs  # noqa: E402
import inseason_movers as IM  # noqa: E402
from common import HERE, out_dir  # noqa: E402

SF, M = IM.SF, IM.M
LINE = {"QB": 24 if SF else 12, "RB": 24, "WR": 30, "TE": 12}
TARGETS = (4, 8, 12)
OFF = 999                        # not on the list
MIN_DEPTH = 250                  # snapshots whose dynasty list is shallower are left out: "unranked" there includes starters
LENSES = ("ros", "hold1", "hold3")
rng = np.random.default_rng(0)


def build():
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    pos = panel.sort_values("t").groupby("gsis_id").position.last()
    P = {k: IM.ranked(v, pos) for k, v in IM.prices(panel).items()}
    snaps = IM.snapshots(P, IM.week_ends())
    snaps = snaps[snaps.target.isin(TARGETS)].copy()
    rd = pd.Series(sorted(P["ros"].date.unique()))
    near = {d: rd.iloc[(pd.to_datetime(rd) - pd.Timestamp(d)).abs().argmin()] for d in snaps.date}
    snaps["ros_date"] = snaps.date.map({d: r for d, r in near.items() if abs((pd.Timestamp(r) - pd.Timestamp(d)).days) <= 4})
    dyn = snaps.merge(P["dyn"][["gsis_id", "date", "rank", "prank"]], on="date")
    ros = snaps.dropna(subset=["ros_date"]).merge(
        P["ros"][["gsis_id", "date", "rank", "prank"]].rename(columns={"date": "ros_date", "rank": "ros_rank", "prank": "ros_prank"}),
        on="ros_date")
    key = ["season", "target", "gsis_id"]
    x = dyn[key + ["rank", "prank"]].merge(ros[key + ["ros_rank", "ros_prank"]], on=key, how="outer")
    x = x.merge(snaps, on=["season", "target"])
    f = panel[["gsis_id", "t", "position", "age", "row_type", "draft_round_f"]].assign(season=lambda d: d.t + 1)
    x = x.merge(f.drop(columns="t"), on=["gsis_id", "season"])
    depth = x.groupby(["season", "target"])["rank"].count()             # superflex 2019 week 4: 211, with starters unmatched
    x = x.join((depth >= MIN_DEPTH).rename("deep"), on=["season", "target"])
    x = x[x.deep].drop(columns="deep")
    x["rank"] = x["rank"].fillna(OFF)
    x.loc[x.ros_date.notna() & x.ros_prank.isna(), ["ros_rank", "ros_prank"]] = OFF
    x["line"] = x.position.map(LINE)
    # production: rest-of-season VORP, points per game to date (rank among everyone with 4+ games, fewer before week 4),
    # weeks inside the starter line from here on
    wk = IM.weekly()
    x = x.merge(IM.ros_vorp(wk, snaps), on=["gsis_id", "season", "w"], how="left")
    x["ros"] = x.ros.fillna(0.0)
    pg = []
    for (y, w) in snaps[["season", "w"]].drop_duplicates().itertuples(index=False):
        s = wk[(wk.season == y) & (wk.week <= w)].groupby(["gsis_id", "position"]).agg(g=("week", "nunique"), pts=("ppr", "sum")).reset_index()
        s = s[s.g >= min(4, w)].assign(ppg=lambda d: d.pts / d.g)
        s["prk"] = s.groupby("position").ppg.rank(ascending=False, method="min")
        pg.append(s[["gsis_id", "prk"]].assign(season=y, w=w))
    x = x.merge(pd.concat(pg), on=["gsis_id", "season", "w"], how="left")
    wk["wrank"] = wk.groupby(["season", "week", "position"]).ppr.rank(ascending=False, method="min")
    wk["start"] = wk.wrank <= wk.position.map(LINE)
    nweeks = wk.groupby("season").week.max()
    st = []
    for (y, w) in snaps[["season", "w"]].drop_duplicates().itertuples(index=False):
        s = wk[(wk.season == y) & (wk.week > w)].groupby("gsis_id").start.sum().rename("starts").reset_index()
        st.append(s.assign(season=y, w=w))
    x = x.merge(pd.concat(st), on=["gsis_id", "season", "w"], how="left")
    x["starts"] = x.starts.fillna(0) / (x.season.map(nweeks) - x.w)
    hs = IM.hurt()
    x["hurt"] = [(g, y, w) in hs or (g, y, w + 1) in hs for g, y, w in zip(x.gsis_id, x.season, x.w)]
    # what holding him returns (as inseason_movers.build)
    o = M.outcomes()[["gsis_id", "t", "vorp1", "vorp3"]].rename(columns={"t": "season"})
    x = x.merge(o, on=["gsis_id", "season"], how="left")
    may = M.fp_dynasty(panel).dropna(subset=["mkt"]).merge(panel[["gsis_id", "t"]], on=["gsis_id", "t"])
    may["mrank"] = may.groupby("t").mkt.rank(method="first")
    X3 = cs.deep(3)
    v3 = cs.iso(X3.mkt_rank, X3.vorp3)
    may["val"] = np.where(may.mrank <= cs.POOL, v3.predict(np.log(may.mrank.clip(upper=cs.POOL))), 0.0)
    val = may.set_index(["gsis_id", "t"]).val
    last_may = may.t.max()
    for h in (1, 3):
        v = pd.Series([val.get((g, y + h), 0.0) for g, y in zip(x.gsis_id, x.season)], index=x.index)
        ok = (x.season + h <= M.LAST) & (x.season + h <= last_may)
        x[f"hold{h}"] = np.where(ok, x.ros + x[f"vorp{h}"].fillna(0.0) + v, np.nan)
    x["ros"] = np.where(x.season <= M.LAST, x.ros, np.nan)
    rookie = x.row_type == "rookie"
    x["grp"] = np.where(x.position == "QB", np.where(x.age >= 33, "QB 33+", "QB under 33"),
                        np.where(rookie, "rookie", np.where(x.age < 26, "RB/WR/TE 25-", "RB/WR/TE 26+")))
    return x


def stat(q, lens):
    q = q[q[lens].notna() & q[f"b_{lens}"].notna()]
    if q.gsis_id.nunique() < 5:
        return ""
    d = (q[lens] - q[f"b_{lens}"]).to_numpy()
    pl = pd.DataFrame({"g": q.gsis_id.to_numpy(), "d": d}).groupby("g").d.agg(["sum", "size"])
    s, c = pl["sum"].to_numpy(), pl["size"].to_numpy()
    idx = rng.integers(0, len(s), (400, len(s)))
    se = (s[idx].sum(1) / c[idx].sum(1)).std()
    m = d.mean()
    ss = q.assign(d=d).groupby("season").d.mean()
    flag = "*" if abs(m) > 2 * se else " "
    return (f"{q[lens].mean():5.1f} vs {q[f'b_{lens}'].mean():5.1f}: {m:+6.1f}{flag}+-{2 * se:4.1f} "
            f"s{(ss > 0).sum()}/{len(ss)} n{q.gsis_id.nunique()}")


def table(x, rows, title, extra=False):
    out = {}
    for lab, mask in rows.items():
        q = x[mask]
        r = {lens: stat(q, lens) for lens in LENSES}
        if extra:
            g = q[q.ros.notna()]
            r["any ROS VORP"] = f"{(g.ros > 0).mean():.0%}" if len(g) else ""
            r["weeks as starter"] = f"{g.starts.mean():.0%}" if len(g) else ""
        out[lab] = r
    print(f"\n== {title}")
    print(pd.DataFrame(out).T.to_string())


def run(x, L):
    x = x.copy()
    drop = x[(x["rank"] > L - 50) & (x["rank"] <= L)]
    b = drop.groupby(["season", "target"])[list(LENSES)].mean().add_prefix("b_")
    x = x.join(b, on=["season", "target"])
    fa = x[x["rank"] > L]
    print(f"\n\n######## {'Superflex' if SF else '1QB'}, roster line {L}: free agents = priced past {L} or unranked "
          f"(about {fa.groupby(['season', 'target']).size().median():.0f} a snapshot); drop tier = priced {L - 49}-{L} "
          f"(about {drop.groupby(['season', 'target']).size().median():.0f})")
    print("drop tier: ROS VORP {:.1f}, hold1 {:.1f}, hold3 {:.1f}".format(*[drop[c].mean() for c in LENSES]))
    h = fa[~fa.hurt & fa.ros_prank.notna()]
    rel = h.ros_prank / h.line
    rows = {
        "ROS rank within 1x the starter line": rel <= 1,
        "  1-1.25x": (rel > 1) & (rel <= 1.25),
        "  1.25-1.5x": (rel > 1.25) & (rel <= 1.5),
        "  1.5-2x": (rel > 1.5) & (rel <= 2),
        "  2-3x": (rel > 2) & (rel <= 3),
        "  past 3x, on the ROS list": (rel > 3) & (h.ros_prank < OFF),
        "  off the ROS list": h.ros_prank == OFF,
        "within 1.25x": rel <= 1.25,
        "within 1.5x": rel <= 1.5,
        "within 2x": rel <= 2,
        "producing (points per game inside the line)": h.prk <= h.line,
        "  producing, ROS past 1.5x": (h.prk <= h.line) & (rel > 1.5),
        "producing or ROS within 1.5x": (h.prk <= h.line) | (rel <= 1.5),
    }
    table(h, rows, "PLAYS NOW: healthy free agents by ROS positional rank / starter line", extra=True)
    for p in ("QB", "RB", "WR", "TE"):
        hp = h[h.position == p]
        rp = hp.ros_prank / hp.line
        table(hp, {f"{p} within 1x": rp <= 1, f"{p} 1-1.5x": (rp > 1) & (rp <= 1.5), f"{p} 1.5-2x": (rp > 1.5) & (rp <= 2),
                   f"{p} 2-3x": (rp > 2) & (rp <= 3)}, f"plays now, {p}", extra=True)
    # free dynasty value: by price band and the group the site's rules treat alike past 200
    bands = {f"priced {L + 1}-{L + 50}": (fa["rank"] <= L + 50), f"priced {L + 51}-{L + 100}": (fa["rank"] > L + 50) & (fa["rank"] <= L + 100),
             f"priced {L + 101}+": (fa["rank"] > L + 100) & (fa["rank"] < OFF), "not on the dynasty list": fa["rank"] == OFF}
    table(fa, {**bands, "  all free agents": fa["rank"] > L}, "FREE DYNASTY VALUE: free agents by price (healthy and hurt)")
    rows = {}
    for g in ("rookie", "QB under 33", "QB 33+", "RB/WR/TE 25-", "RB/WR/TE 26+"):
        rows[g] = fa.grp == g
        for lab, m in list(bands.items())[:3]:
            rows[f"  {g}, {lab}"] = (fa.grp == g) & m
    table(fa, rows, "free dynasty value by group (past 200 the rules hold rookies and QBs under 33 and sell the rest)")
    table(fa, {"hurt rookies": fa.hurt & (fa.grp == "rookie"), "hurt, others": fa.hurt & (fa.grp != "rookie"),
               "healthy rookies": ~fa.hurt & (fa.grp == "rookie")}, "free dynasty value by health")


def main():
    pd.set_option("display.width", 260, "display.max_colwidth", 70, "display.max_rows", 300)
    x = build()
    x.to_csv(out_dir(SF) / "free_agents.csv.gz", index=False)
    print("rows per snapshot week:", x.groupby("target").size().to_dict(), " distinct players:", x.gsis_id.nunique())
    for L in (200, 250, 300):
        run(x, L)


if __name__ == "__main__":
    main()
