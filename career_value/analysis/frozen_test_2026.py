"""Q: do this season's production, injury status and rest-of-season ranks predict beyond the current market price, on a season nobody has seen?

WHY THESE VARIABLES. variable_screen.py picked them on seasons with outcomes ending by 2020, with no KTC or FantasyCalc price: production and injury
status passed for the rest of the season in both formats. ROS ranks start in 2019, so the screen could barely see them; they are on the list on a
promise from two seasons in 1QB, and on what the market-era model found on seasons this test can't call clean.
Expected points also passed but added nothing once production was in, so it stays off. Nothing passed for next season's points; mv1 is here because
next May's price is what a holder can actually sell at, not because the screen backed it. Nothing else is on the list.

FROZEN, committed and pushed before any 2026 outcome was scored:
  Rows      seasons' snapshots at weeks 4, 8 and 12, the market's top 340. 2026 is the test; training is the market era, through 2025.
  Base      isotonic(log market rank), fit per snapshot week. The market rank is that source's on the FP scrape date.
  Groups    form  (games, games missed, PPR per game with 2+ games)
            hurt  (on reserve or ruled out, this week or next)
            ros   (FP ROS overall and position rank, preseason ROS rank)
            all   (the three together)
  Model     gradient boosting on the gap from the base, features log market rank + the group. Settings as single_model_sim.py, untuned.
  Outcomes  ros  rest-of-season VORP (known when the 2026 regular season ends, January 2027)
            mv1  value at the market's rank on May 15, 2027 (POOL - rank)
  Score     skill = % of the base's squared error removed on 2026, all snapshots pooled, read from the saved predictions with no refit.
  Floor     the same fit with the group's columns shuffled within each snapshot, 10 times.
  Cells     1QB and superflex, KTC and FantasyCalc: four, not independent, since they share players and the season.
  PASS      a group passes an outcome if, in all four cells, skill > 0 and skill beats all 10 shuffled copies.
  Meaning   one season can't change a call. A pass goes to 2027 as the next test and to the league sim. A fail comes off the list for that outcome.

score --interim is not the test. It retrains on the rest of the season through the last finished week, the same way for past seasons and for 2026, and
scores that. It reads outcomes the final score will also read, but the frozen predictions can't change: they were saved first.

Usage: LEAGUE=sf SOURCE=ktc python analysis/frozen_test_2026.py predict | score | score --interim | site
predict only adds snapshot weeks missing from the file, so a week 12 run leaves weeks 4 and 8 as they were.
Writes outputs/[sf/]frozen_test_2026_<source>.csv.gz. site collects all four cells' scores for the web page into pages/build/frozen_test.json.
"""
import datetime as dt
import json
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import inseason_movers as IM  # noqa: E402
import market_price_sim as mps  # noqa: E402
import market_residuals as M  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import HERE, out_dir  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

SF, POOL, SOURCE = mps.SF, mps.POOL, mps.SOURCE
TEST, TARGETS, PERMS, OUTCOMES = 2026, (4, 8, 12), 10, ("ros", "mv1")
GROUPS = {"form": ["g", "gone", "ppg"], "hurt": ["hurt"], "ros": ["rosr", "rosp", "rospre"]}
GROUPS["all"] = [c for g in GROUPS.values() for c in g]
OUT = out_dir(SF) / f"frozen_test_2026_{SOURCE}.csv.gz"
SITE = HERE.parent / "pages" / "build" / "frozen_test.json"
DD = HERE / "raw" / "dynastydaddy"
# the page's reading of the interim table, which only shows until the first final score: rewrite it if the interim is rerun
INTERIM_NOTE = ("Injury status leads in every column, but that's partly built in: a dynasty price isn't a forecast of the next four weeks, and a player "
                "ruled out for next week scores nothing in one of them. Production to date is below zero in all four, so the market seems to have priced it "
                "already. One snapshot and four weeks is very little, and the result that bears on dynasty value is next May's.")


def rows_2026():
    """2026 snapshots with no outcome in them: M.LAST is 2025, so build() blanks ros, hold1 and hold3"""
    IM.SEASONS = range(TEST, TEST + 1)
    x = IM.build()
    assert x.ros.isna().all() and x.hold1.isna().all(), "a 2026 outcome reached the predict step"
    return x[(x.w - x.target).abs() <= 1]                     # target 12 has no snapshot until week 11 or 12 is played


def fetch_market(dates):
    for d in dates:
        f = DD / f"simweek_{d}.json"
        if not f.exists():
            n = (dt.date.today() - dt.date.fromisoformat(d)).days
            req = urllib.request.Request(f"https://dynasty-daddy.com/api/v1/player/all/prev/{n}", headers={"User-Agent": "career-value-research/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                f.write_bytes(r.read())
            print("fetched market snapshot", d)


def features(x):
    """market rank, the groups' columns and mv1, for every snapshot the source has a price on"""
    mkt = mps.market_ranks(pd.read_parquet(HERE / "data" / "panel.parquet"))
    out = []
    for (y, tg), q in x[x.target.isin(TARGETS)].groupby(["season", "target"]):
        mk = mkt.get(str(q.date.iloc[0]))
        if mk is None:
            continue
        may = mkt.get(f"{y + 1}-05-15")
        q = q.assign(mr=q.gsis_id.map(mk["rank"]))
        q = q[q.mr <= POOL].copy()
        q["mv1"] = np.maximum(0, POOL - q.gsis_id.map(may["rank"]).fillna(POOL)) if may is not None else np.nan
        out.append(q)
    z = pd.concat(out, ignore_index=True)
    on = z.ros_rank.notna() & (z.ros_rank < 500)
    return z.assign(lr=np.log(z.mr), g=z.g.fillna(0), gone=z.w - z.g.fillna(0), ppg=(z.pts / z.g).where(z.g >= 2), hurt=z.hurt.astype(float),
                    rosr=np.log(z.ros_rank.where(on)), rosp=np.log(z.ros_prank.where(on)), rospre=np.log(z.ros_pre))


def base(tr, y, te):
    """(base on tr, base on te), fit per snapshot week"""
    bt, be = pd.Series(np.nan, index=tr.index), pd.Series(np.nan, index=te.index)
    for t in TARGETS:
        a = tr[tr.target == t]
        if a.empty:
            continue
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(a.lr, a[y])
        bt[a.index] = iso.predict(a.lr)
        i = te.index[te.target == t]
        if len(i):
            be[i] = iso.predict(te.loc[i, "lr"])
    return bt, be


def fit_predict(tr, te, y, cols, bt, be):
    c = ["lr"] + cols
    m = HGB(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=40, l2_regularization=5, random_state=0)
    return be + m.fit(tr[c].to_numpy(float), tr[y] - bt).predict(te[c].to_numpy(float))


def shuffled(d, cols, rng):
    d = d.copy()
    for _, i in d.groupby(["season", "target"]).indices.items():
        d.iloc[i, [d.columns.get_loc(c) for c in cols]] = d.iloc[rng.permutation(i)][cols].to_numpy()
    return d


def predictions(tr, te, y):
    tr, te = tr[tr[y].notna()], te.copy()
    bt, be = base(tr, y, te)
    out = {f"b_{y}": be}
    for g, cols in GROUPS.items():
        out[f"p_{y}_{g}"] = fit_predict(tr, te, y, cols, bt, be)
        rng = np.random.default_rng(0)
        for k in range(PERMS):
            out[f"s{k}_{y}_{g}"] = fit_predict(shuffled(tr, cols, rng), shuffled(te, cols, rng), y, cols, bt, be)
    return pd.DataFrame(out, index=te.index)


def predict():
    new = rows_2026()
    fetch_market(sorted(str(d) for d in new[new.target.isin(TARGETS)].date.unique()))
    old = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    z = features(pd.concat([old[old.season < TEST], new], ignore_index=True))
    tr, te = z[z.season < TEST], z[z.season == TEST]
    print(f"training seasons {sorted(int(s) for s in tr.season.unique())}, {len(tr)} rows; 2026: {te.groupby('target').size().to_dict()}")
    have = pd.read_csv(OUT) if OUT.exists() else None
    if have is not None:
        te = te[~te.target.isin(have.target.unique())]
    if te.empty:
        print("nothing new to predict")
        return
    keep = ["season", "target", "w", "date", "gsis_id", "position", "mr", "g", "gone", "ppg", "hurt", "ros_rank", "ros_prank", "ros_pre"]
    out = pd.concat([te[keep]] + [predictions(tr, te, y) for y in OUTCOMES], axis=1)
    pd.concat([have, out.round(4)], ignore_index=True).to_csv(OUT, index=False) if have is not None else out.round(4).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(HERE)}: snapshot weeks {sorted(int(t) for t in out.target.unique())}, dates {sorted(str(d) for d in out.date.unique())}")


def skill(d, y, p):
    return 100 * (1 - ((d[y] - d[p]) ** 2).mean() / ((d[y] - d[f"b_{y}"]) ** 2).mean())


def weeks_done():
    """regular-season weeks finished by today, from the schedule: a week's stats arrive game by game, so what's on disk can be partial"""
    IM.SEASONS = range(TEST, TEST + 1)
    e = IM.week_ends()
    return int((e.gameday < str(dt.date.today())).sum())


def actual_weekly():
    IM.SEASONS = range(2019, TEST + 1)
    return IM.weekly()


def table(d, y):
    rows = []
    for g in GROUPS:
        s = skill(d, y, f"p_{y}_{g}")
        floor = max(skill(d, y, f"s{k}_{y}_{g}") for k in range(PERMS))
        rows.append({"group": g, "skill": round(s, 1), "floor": round(floor, 1), "ok": bool(s > 0 and s > floor)})
    return rows


def score_interim(res, played):
    """Not the test: refit with the target cut at the last finished week, past seasons too, so the scales match"""
    wk = actual_weekly()
    old = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    x = pd.concat([old[old.season < TEST], rows_2026()], ignore_index=True)
    x = x[x.target.isin(TARGETS) & (x.w < played)].drop(columns="ros")
    v = IM.ros_vorp(wk[wk.week <= played], x)[["gsis_id", "season", "w", "ros"]].rename(columns={"ros": "ros_i"})
    x = x.merge(v, on=["gsis_id", "season", "w"], how="left")
    z = features(x.assign(ros_i=x.ros_i.fillna(0.0)))
    tr, te = z[z.season < TEST], z[z.season == TEST]
    p = pd.concat([te, predictions(tr, te, "ros_i")], axis=1)
    res["interim"] = {"through": played, "rows": len(p), "snapshots": sorted(int(t) for t in p.target.unique()), "ros": table(p, "ros_i")}


def score_final(res, f, played):
    if played >= 18:
        s = IM.ros_vorp(actual_weekly(), f[["season", "w"]].drop_duplicates())
        d = f.merge(s[["gsis_id", "season", "w", "ros"]], on=["gsis_id", "season", "w"], how="left")
        res["ros"] = table(d.assign(ros=d.ros.fillna(0.0)), "ros")
    else:
        print(f"ros waits for the regular season: {played} weeks finished")
    may = mps.market_ranks(pd.read_parquet(HERE / "data" / "panel.parquet")).get(f"{TEST + 1}-05-15")
    if may is not None:
        res["mv1"] = table(f.assign(mv1=np.maximum(0, POOL - f.gsis_id.map(may["rank"]).fillna(POOL))), "mv1")


def score(interim=False):
    f, played = pd.read_csv(OUT), weeks_done()
    p = out_dir(SF) / f"frozen_test_2026_{SOURCE}_score.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    res.update({"format": "sf" if SF else "1qb", "source": SOURCE, "played": played, "snapshots": sorted(int(t) for t in f.target.unique())})
    if interim:
        score_interim(res, played)
    else:
        score_final(res, f, played)
    print(json.dumps(res, indent=1))
    p.write_text(json.dumps(res, indent=1) + "\n")


def site():
    """the four cells' saved scores -> pages/build/frozen_test.json, which the page renders"""
    cells = []
    for sf in (False, True):
        for src in ("ktc", "fantasycalc"):
            f = out_dir(sf) / f"frozen_test_2026_{src}_score.json"
            if f.exists():
                cells.append(json.loads(f.read_text()))
    SITE.write_text(json.dumps({"built": str(dt.date.today()), "note": INTERIM_NOTE, "cells": cells}, indent=1) + "\n")
    print("wrote", SITE, len(cells), "cells")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"predict": predict, "score": lambda: score("--interim" in a), "site": site}[a[0]]()
