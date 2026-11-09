"""Which variables earn a test? Screened only on seasons before the holdout, using no KTC/FantasyCalc price.

DEV DATA (every outcome season <= 2020; the script stops if not):
  may    MFL keeper ADP top 200, veterans, cohorts 2010-17. Outcomes vorp1, vorp3.
  week   same players at weeks 4/8/12 of seasons 2011-18, price = their preseason MFL rank. Outcomes ros, next1, next3 (next3 only through 2017).
  fp     FP dynasty in-season, seasons 2019-20 only. Outcomes ros, hold1. Two folds: shown for size, can't pass the rule.
  nop    no price at all: box scores from 1999, trained on cohorts <= T-h, scored on the may rows.

RERUN NOTE: the first run was thrown out. roc_vorp (rest-of-career VORP, a future outcome) was in "yr before"; weeks before 2017 had no local stats;
a pooled baseline on a 1QB price scored QB-ness as skill in superflex. The pass rule did not change. In superflex the MFL baseline is now fit per position.

METHOD: leave one season out, dropping seasons whose outcomes overlap the test season. Model settings are single_model_sim.py's, untuned.
Skill = % of the squared error of isotonic(log rank) removed. Floor: same fit with that group's columns shuffled within season, 10 times.

DATA: raw/nflverse_old/ holds the nflverse stats_player_week, ep_weekly, injuries and roster_weekly files for 2010-18 (raw/ isn't in git; the fetch
scripts stop at 2017, so download the older years). Saved output: outputs/[sf/]variable_screen.txt.

PASS RULE, fixed before the first run. A group earns a test on an outcome if, added alone to the rank:
  1. its mean skill beats all 10 of its shuffled copies, and
  2. skill is positive in at least 75% of test seasons, and
  3. the mean is positive in both the early and the late half of the test seasons.
"""
import itertools
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import inseason_movers as IM  # noqa: E402
import market_residuals as M  # noqa: E402
from common import out_dir, rows  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

SF = os.environ.get("LEAGUE") == "sf"
JOBS, PERMS = int(os.environ.get("JOBS", 6)), int(os.environ.get("PERMS", 10))
LAST_DEV_OUTCOME = 2020
POSMAP = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
GROUPS = {
    "pos+age": ["pos", "age", "exp"],
    "draft": ["draft_round_f", "draft_pick_f"],
    "last yr": ["games", "ppg", "vorp", "ppg_pos_rank"],
    "yr before": ["games_prev", "ppg_prev", "vorp_prev"],
    "career": ["career_vorp", "career_best_vorp", "career_best_ppg", "seasons_played"],
    "usage": ["att_pg", "carries_pg", "targets_pg", "target_share", "air_yards_share", "wopr"],
    "yards": ["pass_yds_pg", "rush_yds_pg", "rec_yds_pg"],
    "epa": ["epa_pg"],
    "team change": ["changed_team"],
    "body": ["height", "weight", "bmi"],
}
ALLC = [c for g in GROUPS.values() for c in g]
assert not any(c.startswith(("roc", "next", "vorp1", "vorp3", "followup")) for c in ALLC), "future outcome in the features"


def hgb(iters=150):
    return HGB(max_depth=3, learning_rate=0.03, max_iter=iters, min_samples_leaf=40, l2_regularization=5, random_state=0)


def price(tr, y, te_list):
    """isotonic(log rank) on tr, per position when d has bypos set, evaluated on each frame of te_list"""
    keys = sorted(tr.bypos.unique())
    out = [np.full(len(q), np.nan) for q in te_list]
    for k in keys:
        a = tr[tr.bypos == k]
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(a.lr, a[y])
        for i, q in enumerate(te_list):
            m = (q.bypos == k).to_numpy()
            if m.any():
                out[i][m] = iso.predict(q.lr[m])
    return out


def one_fold(tr, te, y, cols):
    bt, b = price(tr, y, [tr, te])
    cols = [c for c in cols if tr[c].notna().sum() > 40 and tr[c].nunique() > 1]
    if not cols:
        return 0.0
    c = ["lr"] + cols
    p = b + hgb().fit(tr[c].to_numpy(float), tr[y] - bt).predict(te[c].to_numpy(float))
    return 100 * (1 - ((te[y] - p) ** 2).mean() / ((te[y] - b) ** 2).mean())


def shuffled(d, cols, seed):
    rng = np.random.default_rng(seed)
    d = d.copy()
    for _, i in d.groupby(["t", "w"]).indices.items():
        d.iloc[i, [d.columns.get_loc(c) for c in cols]] = d.iloc[rng.permutation(i)][cols].to_numpy()
    return d


def folds(d, y, cols, gap, perm=None, seed=0, shuffle_cols=None):
    if perm:
        d = shuffled(d, shuffle_cols, seed)
    return {T: one_fold(d[(d.t - T).abs() >= gap], d[d.t == T], y, cols) for T in sorted(d.t.unique()) if ((d.t - T).abs() >= gap).any()}


def screen(d, y, gap, groups, title):
    d = d[d[y].notna()]
    tasks = {}
    for g, cols in groups.items():
        tasks[g, "add"] = delayed(folds)(d, y, cols, gap)
        tasks[g, "add0"] = None
        for s in range(PERMS):
            tasks[g, f"p{s}"] = delayed(folds)(d, y, cols, gap, True, s, cols)
    allc = [c for cols in groups.values() for c in cols]
    tasks["ALL", "add"] = delayed(folds)(d, y, allc, gap)
    for g, cols in groups.items():
        tasks[g, "drop"] = delayed(folds)(d, y, [c for c in allc if c not in cols], gap)
    tasks = {k: v for k, v in tasks.items() if v is not None}
    res = dict(zip(tasks, Parallel(n_jobs=JOBS)(tasks.values())))
    ts = sorted(res["ALL", "add"])
    full = np.mean(list(res["ALL", "add"].values()))
    out = []
    for g in groups:
        a = pd.Series(res[g, "add"])
        perm = [np.mean(list(res[g, f"p{s}"].values())) for s in range(PERMS)]
        early, late = a.iloc[: len(a) // 2].mean(), a.iloc[len(a) // 2:].mean()
        ok = a.mean() > max(perm) and (a > 0).mean() >= 0.75 and early > 0 and late > 0
        sd = a.std(ddof=1)
        out.append({"group": g, "alone": a.mean(), "floor": max(perm), "seasons+": f"{(a > 0).sum()}/{len(a)}", "early": early, "late": late,
                    "drop": full - pd.Series(res[g, "drop"]).mean(), "seasons for CI>0": f"{(1.96 * sd / a.mean()) ** 2:.0f}" if a.mean() > 0 else "-",
                    "PASS": "yes" if ok else ""})
    print(f"\n-- {title}: {y}, {len(d)} rows, {d.gsis_id.nunique()} players, test seasons {ts}, overlap gap {gap}. All groups together {full:+.1f}")
    print(pd.DataFrame(out).sort_values("alone", ascending=False).round(1).to_string(index=False))


def lc_task(d, y, cols, T, train):
    return one_fold(d[d.t.isin(train)], d[d.t == T], y, cols)


def learning_curve(d, y, gap, sets, title, draws=8):
    d = d[d[y].notna()]
    rng = np.random.default_rng(0)
    plan = []
    for T in sorted(d.t.unique()):
        el = sorted(t for t in d.t.unique() if abs(t - T) >= gap)
        for k in range(1, len(el) + 1):
            combos = list(itertools.combinations(el, k))
            for tr in ([combos[i] for i in rng.choice(len(combos), min(draws, len(combos)), replace=False)]):
                for name, cols in sets.items():
                    plan.append((name, k, T, tr, cols))
    vals = Parallel(n_jobs=JOBS)(delayed(lc_task)(d, y, c, T, tr) for _, _, T, tr, c in plan)
    z = pd.DataFrame({"set": [p[0] for p in plan], "k": [p[1] for p in plan], "T": [p[2] for p in plan], "s": vals})
    z = z.groupby(["set", "k", "T"]).s.mean().reset_index()
    n = z.groupby("k")["T"].nunique()
    t = z[z.k.isin(n[n >= max(2, 0.5 * n.max())].index)].pivot_table(index="set", columns="k", values="s").round(1)
    print(f"\n-- {title}: {y}, skill by number of training seasons (columns), mean over test seasons and {draws} random picks of training seasons")
    print(t.to_string())


def load_may(h):
    x = rows(["mfl_dynasty"], h, SF)
    x = x[x.t <= 2017].copy()
    assert (x.t + h).max() <= LAST_DEV_OUTCOME
    x["lr"], x["pos"] = np.log(x.mkt_rank), x.position.map(POSMAP)
    x["changed_team"] = x.changed_team.astype(float)
    x["w"] = 0
    x["bypos"] = x.position if SF else "all"
    return x


def no_price(h):
    """box scores only, trained on cohorts up to T-h, scored on the MFL rows; negative = worse than the price"""
    y = f"vorp{h}"
    p = pd.read_parquet(ROOT / "data" / "panel.parquet")
    p = p[(p.row_type == "veteran") & p.position.isin(POSMAP)].drop(columns=["next_vorp"]).merge(M.outcomes()[["gsis_id", "t", y]], on=["gsis_id", "t"], how="left")
    p["pos"], p["changed_team"] = p.position.map(POSMAP), p.changed_team.astype(float)
    p["ppg_pos_rank"] = p.groupby(["t", "position"]).ppg.rank(ascending=False, method="min")
    d = load_may(h)
    out = []

    def fold(T, k, iters):
        te = d[d.t == T]
        tr = p[(p.t <= T - h) & (p.t > T - h - k) & p[y].notna()]
        el = d[(d.t - T).abs() >= h]
        b = price(el, y, [te])[0]
        q = hgb(iters).fit(tr[ALLC].to_numpy(float), tr[y]).predict(te[ALLC].to_numpy(float))
        return {"T": T, "k": k, "iters": iters, "train rows": len(tr), "skill": 100 * (1 - ((te[y] - q) ** 2).mean() / ((te[y] - b) ** 2).mean()),
                "corr": np.corrcoef(q, te[y])[0, 1], "corr price": np.corrcoef(b, te[y])[0, 1]}
    out = [fold(T, k, it) for T in sorted(d.t.unique()) for k in (1, 3, 5, 8, 30) for it in (150, 400)]
    z = pd.DataFrame(out)
    print(f"\n-- No price at all: vorp{h}, box-score model vs the MFL rank on the same rows. Negative skill = worse than the price. Rows: training seasons (30 = all back to 1999)")
    print(z.groupby(["iters", "k"])[["train rows", "skill", "corr", "corr price"]].mean().round(2).to_string())
    print("  seasons where it beat the price, iters 400, all history:", int((z[(z.iters == 400) & (z.k == 30)].skill > 0).sum()), "of", z["T"].nunique())


def load_week():
    IM.SEASONS = list(range(2011, 2019))
    IM.NV = ROOT / "raw" / "nflverse_old"
    wk = IM.weekly()
    hurt = IM.hurt()
    snaps = pd.DataFrame([(y, w) for y in IM.SEASONS for w in (4, 8, 12)], columns=["season", "w"])
    v = IM.ros_vorp(wk, snaps)
    m = load_may(1).drop(columns=["w", "vorp1", "vorp2", "vorp3", "vorp5", "vorp10", "next_vorp", "resid", "r"], errors="ignore")
    m = m.assign(season=m.t + 1).merge(snaps, on="season")
    z = m.merge(v, on=["gsis_id", "season", "w"], how="left")
    z[["ros", "g", "pts"]] = z[["ros", "g", "pts"]].fillna(0.0)
    z["ppg_td"], z["xppg_td"] = (z.pts / z.g).where(z.g >= 2), (z.xpts / z.g).where((z.g >= 2) & (z.xpts > 0))
    z["hurt_now"] = [float((g, y, w) in hurt) for g, y, w in zip(z.gsis_id, z.season, z.w)]
    z["gone"] = z.w - z.g
    o = M.outcomes()[["gsis_id", "t", "vorp1", "vorp3"]].rename(columns={"t": "season", "vorp1": "next1", "vorp3": "next3"})
    z = z.merge(o, on=["gsis_id", "season"], how="left")
    z[["next1", "next3"]] = z[["next1", "next3"]].fillna(0.0)
    z.loc[z.season + 3 > LAST_DEV_OUTCOME, "next3"] = np.nan
    assert z.season.max() + 1 <= LAST_DEV_OUTCOME
    z["t"] = z.season
    print("week rows: seasons", z.groupby("season").ros.agg(lambda r: round(float(r.mean()), 1)).to_dict(), "| xppg known", z.groupby("season").xppg_td.apply(lambda r: round(float(r.notna().mean()), 2)).to_dict(), "| hurt share", z.groupby("season").hurt_now.mean().round(3).to_dict())
    return z


def load_fp():
    x = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    x = x[x.season.between(2019, 2020) & x.target.isin((4, 8, 12)) & (x["rank"] <= 340)].copy()
    on = x.ros_rank.notna() & (x.ros_rank < 500)
    x = x.assign(t=x.season, lr=np.log(x["rank"]), pos=x.position.map(POSMAP), fpp=np.log(x.prank), pre_=np.log(x.pre.clip(lower=1)),
                 rosr=np.log(x.ros_rank.where(on)), rosp=np.log(x.ros_prank.where(on)), rospre=np.log(x.ros_pre), rd=x.draft_round_f.fillna(8),
                 rookie=(x.row_type == "rookie").astype(float), hurt_=x.hurt.astype(float), ppg_td=(x.pts / x.g).where(x.g >= 2),
                 xppg_td=(x.xpts / x.g).where(x.g >= 2), bypos="all")
    return x


if __name__ == "__main__":
    lg = "Superflex" if SF else "1QB"
    print(f"===== {lg}. Pass rule fixed before the run; see the script's docstring. drop = what all-groups skill loses without the group.")
    for h in (1, 3):
        d = load_may(h)
        y = f"vorp{h}"
        screen(d, y, h, GROUPS, f"May, MFL price, cohorts 2010-17")
        learning_curve(d, y, h, {"pos+age": GROUPS["pos+age"], "all box": ALLC}, "May")
        no_price(h)
    w = load_week()
    wg = {**GROUPS, "form": ["g", "gone", "ppg_td"], "xform": ["xppg_td"], "hurt": ["hurt_now"]}
    print("\nNote: week rows use a preseason price, so in-season news is new to the baseline. A pass here can't show it beats a current market price; a fail is informative.")
    for y, gap in (("ros", 1), ("next1", 2), ("next3", 4)):
        screen(w, y, gap, wg, "In season, preseason MFL price, seasons 2011-18")
    f = load_fp()
    fg = {"pos+age": ["pos", "age"], "draft": ["rd", "rookie"], "ros ranks": ["rosr", "rosp", "rospre"], "pos rank+pre": ["fpp", "pre_"],
          "form": ["g", "ppg_td"], "xform": ["xppg_td"], "hurt": ["hurt_"]}
    for y, gap in (("ros", 1), ("hold1", 1)):
        screen(f, y, gap, fg, "In season, FP dynasty rank, seasons 2019-20 ONLY (2 folds; size check)")
