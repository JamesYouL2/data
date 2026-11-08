"""Q: should in-season risers and fallers change the call? Price after week w of season Y = FP dynasty ECR then; the
question is whether the move since the preseason is fully priced (keep ~1), overpriced (risers < 1, fallers > 1:
sell high, buy low) or underpriced (momentum).

Prices (rank among QB/RB/WR/TE by ECR, top POOL at the snapshot):
  1QB dynasty   2019: files/archives/fantasypros weekly CSVs (name-matched); 2020+: db_fpecr ("do")
  SF dynasty    2019-Oct 2020: DynastyProcess values-players.csv git history (FP 2QB ECR, about monthly in 2019);
                then db_fpecr "dsf"
  1QB ROS       2019: files/archives/fantasypros ecr_*.csv (roECR, Aug-Nov); 2020+: db_fpecr "ro" (redraft = rest of
                season in-season; players out for the year drop off the list -> unranked)
  SF ROS        none in-season (FP's superflex redraft list stops each September): the 1QB ROS list is used, so
                ROS moves count for every position; the ROS-priced check leaves out QBs
w = regular-season weeks completed before the scrape (nflverse schedules). Snapshots: preseason (w = 0) and w = 4, 8,
12, 16 (nearest scrape within a week).
What holding him from the snapshot returns (season VORP units, as call_strength.py):
  ros   = VORP over the rest of season Y (weeks > w; replacement QB13/RB31/WR43/TE13 on rest-of-season points, or the
          superflex greedy fill)                                                  contender's lens, seasons 2019-25
  hold1 = ros + VORP Y+1 + value at May after Y+1                                 Default's lens, seasons 2019-24
  hold3 = ros + VORP Y+1..Y+3 + value at May after Y+3                            rebuilder's lens, seasons 2019-22
value = expected 3-year VORP at that May FP rank (call_strength.py's fit). Benchmark = expected outcome at his
snapshot price (isotonic on log rank, per snapshot week, seasons pooled); keep = sum outcome / sum benchmark.
move = log(preseason rank / rank now) (unranked preseason = 500): riser >= +0.29 (rank a quarter better), faller
<= -0.29. Riser kinds by expected fantasy points to date (ffopportunity): usage (points within 20% of expected or
below) vs luck (points > 1.2x expected). Faller kinds: hurt (on IR / reserve or ruled Out in week w) vs healthy.
ROS vs dynasty: positional rank in each list (RB12 in dynasty, RB5 rest of season = "ROS > dynasty" by the same
quarter margin).
Writes outputs/[sf/]inseason_movers.csv (one row per player-snapshot). LEAGUE=sf for superflex.

Findings (2026-11; keep vs flat-dynasty players of the same position x age class, player-cluster bootstrap, 400
draws, 90% CI, halves ros 2019-21 / 2022-25, hold1 2019-21 / 2022-24, hold3 2019-20 / 2021-22):
- No reversal. Fallers are not a buy-low: all fallers hold1 0.89 [0.81-0.97] 1QB, 0.88 [0.80-0.95] SF; hurt fallers
  0.72 [0.58-0.88] / 0.69 [0.53-0.83], both halves; healthy fallers about neutral (0.96 / 0.95). Risers about fairly
  priced (hold1 1.04 / 1.00), except SF RB risers hold3 0.67 [0.53-0.80] and SF TE risers hold3 0.49 [0.31-0.67].
- The ROS rank tracks rest-of-season VORP better than the dynasty rank at every snapshot week (Spearman, 26 of 27
  season-weeks in 1QB; SF similar). Priced by ROS rank, contenders' risers are about neutral (usage 0.95 / 0.91).
- The ROS move is underpriced in dynasty: ROS up hold1 1.10 [1.04-1.17] 1QB, 1.11 [1.05-1.17] SF; ROS down 0.86
  [0.80-0.93] / 0.87 [0.79-0.92]; both halves, both formats. Flat dynasty, ROS up: 1.14 / 1.13.
- ROS rank above dynasty rank: rookies hold1 1.51 [1.25-1.81] 1QB (n=39), 1.61 [1.38-1.90] SF (n=37); 25 and under
  hold3 1.20 in both formats (CI low end about 1.0); 26+ hold1 1.13 1QB, 1.22 SF. ROS below dynasty, 26-29: 0.78
  [0.64-0.93] 1QB; SF and 30+ halves disagree.
Caveats: a player appears at several weeks (hence the cluster bootstrap); SF 2019 snapshots are weeks 4 and 17 only;
the SF ROS list is the 1QB one; nothing here has been run through the league sim (it has no in-season trades).
"""
import glob
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import HERE, out_dir  # noqa: E402
import call_strength as cs  # noqa: E402
import market_residuals as M  # noqa: E402
import build_dataset as B  # noqa: E402

SF, POS, POOL = M.SF, M.POS, cs.POOL
NV, FILES, ARCH = HERE / "raw" / "nflverse", M.FILES, M.FILES / "archives" / "fantasypros"
SEASONS = range(2019, 2026)
WEEKS = (0, 4, 8, 12, 16)
UNRANKED = 500
RISE = np.log(4 / 3)
HALVES = {"ros": ((2019, 2020, 2021), (2022, 2023, 2024, 2025)), "hold1": ((2019, 2020, 2021), (2022, 2023, 2024)),
          "hold3": ((2019, 2020), (2021, 2022))}


# ---------------------------------------------------------------- calendar
def week_ends():
    g = pd.read_parquet(NV / "schedules" / "games.parquet", columns=["season", "game_type", "week", "gameday"])
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS)]
    return g.groupby(["season", "week"]).gameday.max().reset_index()


def weeks_played(dates, ends):
    """date (YYYY-MM-DD) -> (season, weeks completed before it); dates Aug 15 - Jan 20 only"""
    out = {}
    for d in dates:
        y = int(d[:4]) - (d[5:7] == "01")
        if y not in SEASONS or not ("08-15" <= d[5:] or d[5:] <= "01-20"):
            continue
        e = ends[ends.season == y]
        out[d] = (y, int((e.gameday < d).sum()))
    return out


# ---------------------------------------------------------------- prices
def fp_ids():
    return (pd.read_csv(FILES / "db_playerids.csv", dtype=str, usecols=["fantasypros_id", "gsis_id"])
            .dropna().drop_duplicates("fantasypros_id"))


def by_name(x, panel):
    """x(name, pos, ...) -> gsis_id via market_residuals.match_names, matched once per distinct name"""
    u = x[["name", "pos"]].drop_duplicates()
    m = M.match_names(u, 2018, panel)[["name", "pos", "gsis_id"]]
    return x.merge(m, on=["name", "pos"])


def db_prices(e, ecr_type, pages):
    x = e[(e.ecr_type == ecr_type) & e.page_type.isin(pages) & e.pos.str[:2].isin(POS)]
    x = x.merge(fp_ids(), left_on="id", right_on="fantasypros_id")
    return x[["gsis_id", "scrape_date", "ecr"]].rename(columns={"scrape_date": "date"})


def prices(panel):
    """kind ('dyn', 'ros') -> gsis_id, date, ecr"""
    e = pd.read_parquet(FILES / "db_fpecr.parquet", columns=["page_type", "ecr_type", "scrape_date", "id", "pos", "ecr"])
    e = e[e.scrape_date >= "2019-08-01"]
    if SF:
        dyn = [db_prices(e, "dsf", ["dynasty-op"]), dp_history_sf(panel)]
        ros = [db_prices(e, "ro", ["redraft-overall", "redraft-offense"]), archive_ros(panel)]
    else:
        dyn = [db_prices(e, "do", ["dynasty-overall", "dynasty-offense"]), archive_dyn(panel)]
        ros = [db_prices(e, "ro", ["redraft-overall", "redraft-offense"]), archive_ros(panel)]
    out = {}
    for k, parts in (("dyn", dyn), ("ros", ros)):
        p = pd.concat(parts, ignore_index=True).dropna(subset=["ecr"])
        p = p[p.gsis_id.isin(panel.gsis_id)].drop_duplicates(["gsis_id", "date"])
        out[k] = p
    return out


def archive_dyn(panel):
    fs = [f for f in sorted(glob.glob(str(ARCH / "dynasty-overall-*.csv"))) if "20190815" <= f[-12:-4] <= "20200120"]
    xs = []
    for f in fs:
        x = pd.read_csv(f).dropna(subset=["Overall (Team)"])
        d = f[-12:-4]
        xs.append(pd.DataFrame({"name": x["Overall (Team)"].map(M.fp_archive_name), "pos": x.Pos.str.extract(r"^([A-Z]+)")[0],
                                "ecr": pd.to_numeric(x.Avg, errors="coerce"), "date": f"{d[:4]}-{d[4:6]}-{d[6:]}"}))
    x = pd.concat(xs)
    return by_name(x[x.pos.isin(POS)], panel)[["gsis_id", "date", "ecr"]]


def archive_ros(panel):
    xs = []
    for f in sorted(glob.glob(str(ARCH / "ecr_2019*.csv"))):
        x = pd.read_csv(f)
        xs.append(pd.DataFrame({"name": x.Player, "pos": x.Pos, "ecr": pd.to_numeric(x.roECR, errors="coerce"),
                                "date": x.ScrapeDate.astype(str)}))
    x = pd.concat(xs)
    return by_name(x[x.pos.isin(POS)], panel)[["gsis_id", "date", "ecr"]]


def dp_history_sf(panel):
    """FP 2QB ECR from DynastyProcess values-players.csv commits, Aug 2019 - Oct 15 2020"""
    repo = HERE.parent
    log = subprocess.run(["git", "log", "--format=%H %cs", "--since=2019-08-15", "--until=2020-10-15", "--",
                          "files/values-players.csv"], cwd=repo, capture_output=True, text=True, check=True).stdout.split()
    xs = []
    for sha, d in zip(log[::2], log[1::2]):
        if not ("08-15" <= d[5:] or d[5:] <= "01-20"):
            continue
        raw = subprocess.run(["git", "show", f"{sha}:files/values-players.csv"], cwd=repo, capture_output=True,
                             check=True).stdout
        x = pd.read_csv(pd.io.common.BytesIO(raw), encoding="utf-8-sig")
        x = x.rename(columns={"mergename": "name", "player": "name", "dyno2QBECR": "ecr", "ecr_2qb": "ecr"})
        x = x[x.pos.isin(POS)].assign(date=d)
        xs.append(x[["name", "pos", "ecr", "date"] + (["fp_id"] if "fp_id" in x else [])])
    x = pd.concat(xs, ignore_index=True)
    a = x[x.fp_id.notna()].assign(fp_id=lambda d: d.fp_id.astype("Int64").astype(str)) if "fp_id" in x else x.iloc[:0]
    a = a.merge(fp_ids(), left_on="fp_id", right_on="fantasypros_id")
    b = by_name(x[~x.index.isin(a.index) & ~x.name.isin(a.name)], panel)
    return pd.concat([a, b])[["gsis_id", "date", "ecr"]]


def ranked(p, pos):
    """overall rank and positional rank (position: the player's latest panel position) per scrape date"""
    p = p.assign(position=p.gsis_id.map(pos))
    p["rank"] = p.groupby("date").ecr.rank(method="first")
    p["prank"] = p.groupby(["date", "position"]).ecr.rank(method="first")
    return p.drop(columns="position")


def snapshots(P, ends):
    """season, w, date per target week (preseason = last scrape before week 1 from Aug 15)"""
    wk = weeks_played(sorted(P["dyn"].date.unique()), ends)
    d = pd.DataFrame([(k, *v) for k, v in wk.items()], columns=["date", "season", "w"])
    rows = []
    for y, g in d.groupby("season"):
        for w in WEEKS:
            c = g[g.w == w] if w == 0 else g[(g.w - w).abs() <= 1]
            if len(c):
                c = c.iloc[[-1]] if w == 0 else c.iloc[[(c.w - w).abs().argmin()]]
                rows.append({"season": y, "w": int(c.w.iloc[0]), "target": w, "date": c.date.iloc[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- production
def weekly():
    s = pd.concat([pd.read_parquet(f, columns=["player_id", "position", "season", "week", "season_type",
                                               "fantasy_points_ppr"]) for f in sorted((NV / "stats_player_week").glob("*.parquet"))])
    s = s[(s.season_type == "REG") & s.season.isin(SEASONS) & s.position.isin(POS)]
    e = pd.concat([pd.read_parquet(f, columns=["player_id", "season", "week", "total_fantasy_points_exp"])
                   for f in sorted((NV / "ep_weekly").glob("*.parquet"))])
    e["season"] = e.season.astype(int)
    s = s.merge(e.groupby(["player_id", "season", "week"]).total_fantasy_points_exp.sum().reset_index(),
                on=["player_id", "season", "week"], how="left")
    return s.rename(columns={"player_id": "gsis_id", "fantasy_points_ppr": "ppr", "total_fantasy_points_exp": "xppr"})


def hurt():
    """(gsis_id, season, week) on reserve (IR/PUP/NFI) or ruled Out"""
    r = pd.concat([pd.read_parquet(f, columns=["gsis_id", "season", "week", "status"]) for f in
                   sorted((NV / "roster_weekly").glob("*.parquet"))])
    i = pd.concat([pd.read_parquet(f, columns=["gsis_id", "season", "week", "report_status", "game_type"]) for f in
                   sorted((NV / "injuries").glob("*.parquet"))])
    a = r[r.status == "RES"][["gsis_id", "season", "week"]]
    b = i[(i.game_type == "REG") & (i.report_status == "Out")][["gsis_id", "season", "week"]]
    return set(map(tuple, pd.concat([a, b]).dropna().astype({"season": int, "week": int}).itertuples(index=False)))


def ros_vorp(wk, snaps):
    """rest-of-season VORP for every (season, w) in snaps, and points / expected points to date"""
    out = []
    for (y, w) in snaps[["season", "w"]].drop_duplicates().itertuples(index=False):
        s = wk[wk.season == y]
        rest = s[s.week > w].groupby(["gsis_id", "position"]).ppr.sum().reset_index()
        rest["season"] = y * 100 + w                         # add_vorp / add_vorp_sf group by "season"
        v = B.add_vorp_sf(rest) if SF else B.add_vorp(rest)
        v = v.assign(ros=v.vorp_sf if SF else v.vorp)
        past = s[s.week <= w].groupby("gsis_id").agg(g=("week", "nunique"), pts=("ppr", "sum"), xpts=("xppr", "sum"))
        v = v[["gsis_id", "ros"]].merge(past.reset_index(), on="gsis_id", how="outer")
        out.append(v.assign(season=y, w=w))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- assemble
def build():
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    ends = week_ends()
    pos = panel.sort_values("t").groupby("gsis_id").position.last()
    P = {k: ranked(v, pos) for k, v in prices(panel).items()}
    snaps = snapshots(P, ends)
    print("snapshots (season x target week -> weeks played):")
    print(snaps.pivot(index="season", columns="target", values="w").to_string())
    x = snaps.merge(P["dyn"], on="date")
    pre = x[x.target == 0][["gsis_id", "season", "rank"]].rename(columns={"rank": "pre"})
    x = x[x["rank"] <= POOL].merge(pre, on=["gsis_id", "season"], how="left")
    x["pre"] = x.pre.fillna(UNRANKED).clip(upper=UNRANKED)
    x["move"] = np.log(x.pre / x["rank"])
    # ROS rank at the nearest ROS scrape within 4 days (no ROS list that week -> NaN; off the list -> unranked)
    rd = pd.Series(sorted(P["ros"].date.unique()))
    near = {d: (rd.iloc[(pd.to_datetime(rd) - pd.Timestamp(d)).abs().argmin()] if len(rd) else None) for d in snaps.date}
    near = {d: r for d, r in near.items() if r is not None and abs((pd.Timestamp(r) - pd.Timestamp(d)).days) <= 4}
    x["ros_date"] = x.date.map(near)
    x = x.merge(P["ros"][["gsis_id", "date", "rank", "prank"]].rename(
        columns={"date": "ros_date", "rank": "ros_rank", "prank": "ros_prank"}), on=["gsis_id", "ros_date"], how="left")
    x.loc[x.ros_date.notna() & x.ros_rank.isna(), ["ros_rank", "ros_prank"]] = UNRANKED
    rpre = x[x.target == 0][["gsis_id", "season", "ros_rank"]].rename(columns={"ros_rank": "ros_pre"})
    x = x.merge(rpre, on=["gsis_id", "season"], how="left")
    # features at the preseason panel row (t = Y - 1)
    f = panel[["gsis_id", "t", "position", "age", "row_type", "draft_round_f"]].assign(season=lambda d: d.t + 1)
    x = x.merge(f.drop(columns="t"), on=["gsis_id", "season"])
    x["acls"] = np.where(x.row_type == "rookie", "rookie", np.where(x.age < 26, "25-", np.where(x.age >= 30, "30+", "26-29")))
    # production
    wk = weekly()
    x = x.merge(ros_vorp(wk, snaps), on=["gsis_id", "season", "w"], how="left")
    x["ros"] = x.ros.fillna(0.0)
    hs = hurt()
    x["hurt"] = [(g, y, w) in hs or (g, y, w + 1) in hs for g, y, w in zip(x.gsis_id, x.season, x.w)]
    # future seasons and May values
    o = M.outcomes()[["gsis_id", "t", "vorp1", "vorp3"]].rename(columns={"t": "season"})
    x = x.merge(o, on=["gsis_id", "season"], how="left")      # outcomes at t = Y: seasons Y+1..
    may = M.fp_dynasty(panel).dropna(subset=["mkt"]).merge(panel[["gsis_id", "t"]], on=["gsis_id", "t"])
    may["mrank"] = may.groupby("t").mkt.rank(method="first")
    X3 = cs.deep(3)
    v3 = cs.iso(X3.mkt_rank, X3.vorp3)
    may["val"] = np.where(may.mrank <= POOL, v3.predict(np.log(may.mrank.clip(upper=POOL))), 0.0)
    val = may.set_index(["gsis_id", "t"]).val
    last_may = may.t.max()
    for h in (1, 3):
        v = pd.Series([val.get((g, y + h), 0.0) for g, y in zip(x.gsis_id, x.season)], index=x.index)
        ok = (x.season + h <= M.LAST) & (x.season + h <= last_may)
        x[f"hold{h}"] = np.where(ok, x.ros + x[f"vorp{h}"] + v, np.nan)
    x["ros"] = np.where(x.season <= M.LAST, x.ros, np.nan)
    return x[x.target > 0].reset_index(drop=True)


def add_fair(x):
    for lens in ("ros", "hold1", "hold3"):
        x[f"f_{lens}"] = np.nan
        for t, g in x[x[lens].notna()].groupby("target"):
            x.loc[g.index, f"f_{lens}"] = cs.iso(g["rank"], g[lens]).predict(np.log(g["rank"]))
    r = x[x.ros_rank.notna() & x.ros.notna() & ~(SF & (x.position == "QB"))]
    x["f_ros_by_ros"] = np.nan
    for t, g in r.groupby("target"):
        x.loc[g.index, "f_ros_by_ros"] = cs.iso(g.ros_rank, g.ros).predict(np.log(g.ros_rank))
    return x


def groups(x):
    x["mv"] = np.where(x.move >= RISE, "riser", np.where(x.move <= -RISE, "faller", "flat"))
    luck = (x.pts > 1.2 * x.xpts.clip(lower=1)) & (x.g > 0)
    x["kind"] = np.where(x.mv == "riser", np.where(luck, "riser: luck", "riser: usage"),
                         np.where(x.mv == "faller", np.where(x.hurt, "faller: hurt", "faller: healthy"), "flat"))
    x["rosmv"] = np.where(x.ros_pre.isna() | x.ros_rank.isna(), "n/a",
                          np.where(np.log(x.ros_pre / x.ros_rank) >= RISE, "ROS up",
                                   np.where(np.log(x.ros_pre / x.ros_rank) <= -RISE, "ROS down", "ROS flat")))
    gap = np.log(x.prank / x.ros_prank)                     # positional ranks: format-neutral
    x["gap"] = np.where(x.ros_prank.isna(), "n/a", np.where(gap >= RISE, "ROS > dynasty",
                                                            np.where(gap <= -RISE, "ROS < dynasty", "ROS ~ dynasty")))
    x["phase"] = x.target.map({4: "wk4", 8: "wk8", 12: "wk12", 16: "wk16"})
    return x


# ---------------------------------------------------------------- report
def keep(q, lens, fair=None):
    fair = fair or f"f_{lens}"
    q = q[q[lens].notna() & q[fair].notna()]
    if len(q) == 0 or q[fair].sum() <= 0:
        return None
    num, den = q[lens], q[fair]
    b = (num - den).idxmax()
    solo = (num.sum() - num[b]) / (den.sum() - den[b]) if len(q) > 1 else np.nan
    a, z = HALVES[lens]
    h = [q[q.season.isin(s)] for s in (a, z)]
    hk = [s[lens].sum() / s[fair].sum() if len(s) and s[fair].sum() > 0 else np.nan for s in h]
    return {"keep": num.sum() / den.sum(), "solo": solo, "beat": (num > den).mean(), "n": len(q),
            "players": q.gsis_id.nunique(), "early": hk[0], "late": hk[1]}


def fmt(k):
    return "" if k is None else (f"{k['keep']:.2f} ({k['early']:.2f}/{k['late']:.2f}) s{k['solo']:.2f} "
                                 f"n{k['players']}")


def table(x, by, title, lenses=("ros", "hold1", "hold3"), rows_by=None):
    out = {}
    for key, q in x.groupby(by):
        out[key] = {lens: fmt(keep(q, lens)) for lens in lenses}
    t = pd.DataFrame(out).T
    print(f"\n== {title}  [keep (2019-21 / later) solo-without-best, distinct players]")
    print(t.to_string())


def main():
    x = groups(add_fair(build()))
    pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 300)
    lg = "Superflex" if SF else "1QB"
    print(f"\n######## {lg}: in-season movers vs their new price (FP top {POOL} at the snapshot)")
    print("rows per week:", x.groupby("target").size().to_dict(), " distinct players:", x.gsis_id.nunique())
    table(x, ["mv"], "move since preseason, all weeks pooled")
    table(x, ["phase", "mv"], "by week")
    table(x, ["position", "mv"], "by position")
    table(x, ["acls", "mv"], "by age class")
    table(x, ["kind"], "riser: usage vs luck; faller: hurt vs healthy")
    table(x, ["position", "kind"], "kind by position")
    table(x[x.rosmv != "n/a"], ["mv", "rosmv"], "dynasty move x ROS move (ROS list weeks only)")
    table(x[x.gap != "n/a"], ["gap"], "ROS rank vs dynasty rank now")
    table(x[x.gap != "n/a"], ["acls", "gap"], "ROS vs dynasty, by age class")
    # ROS rank vs dynasty rank as a forecast of rest-of-season production
    r = x[x.ros_rank.notna() & x.ros.notna() & ~(SF & (x.position == "QB"))]
    rows = []
    for (t, y), g in r.groupby(["target", "season"]):
        rows.append({"target": t, "season": y, "dyn": spearmanr(-g["rank"], g.ros).statistic,
                     "ros_rank": spearmanr(-g.ros_rank, g.ros).statistic})
    s = pd.DataFrame(rows)
    s["ros better"] = s.ros_rank > s.dyn
    print("\n== Spearman with rest-of-season VORP: dynasty rank vs ROS rank (seasons where ROS beats dynasty)")
    print(s.groupby("target").agg(dyn=("dyn", "mean"), ros_rank=("ros_rank", "mean"),
                                  ros_better=("ros better", "sum"), seasons=("season", "size")).round(3).to_string())
    k = {lab: fmt(keep(q, "ros", "f_ros_by_ros")) for lab, q in r.groupby("kind")}
    print("\n== contender lens priced by ROS rank instead (keep = rest-of-season VORP / expected at ROS rank)")
    print(pd.Series(k).to_string())
    x.to_csv(out_dir(SF) / "inseason_movers.csv", index=False)


if __name__ == "__main__":
    main()
