"""Q: would one model of all the in-season inputs beat the site's rules, at market prices?

The site's calls are hand-set rules over many inputs. The alternative: predict what a player is worth from the market rank, position, age,
draft round, FP dynasty rank, preseason FP and market ranks, ROS ranks, this season's and last season's production and injury status, then act
where the prediction is furthest from what the market rank alone implies. The model fits that gap (gradient boosting over isotonic(market rank)), so it
starts from the market's price and only moves off it where the inputs earned it.

Two checks:
1. Prediction, by season: how much of the error left by the market rank alone does the model remove, and do its picks beat the rules' signal picks?
   Outcomes: mv1 (value at the market's rank next May), ros (rest-of-season VORP), hold1 and hold3 (inseason_movers.py: production plus FP May value).
2. The league sim of market_price_sim.py, all trades at the market rank. Each call list has as many sells, buys and sell-next-Mays as live has
   for that team type at that snapshot, so the arms differ in who gets traded rather than how much:
     regsig   the rules without their signals, plus model sells and buys in place of the signals
     regall   no rules: sells and buys off the model's edge, sell-next-Mays off its mv1 edge
   Contenders act on the ros edge, rebuilders on the mv1 edge, the default team on both, z-scored and added.
   _wf trains only on seasons before the snapshot's. _lo trains on every other season, which sees the future the way the rules' own fit does.

Caveats: market data starts in 2021, so 3-year sim results rest on 1-2 walk-forward cohorts. hold3 can't be walk-forward tested at all. The KTC and FantasyCalc
runs share players and seasons. The model isn't tuned; its settings are one conservative guess.
The saved outputs ran before the site's rules changed on the back of signal_sim.py, so live here is the older rules.
Writes outputs/[sf/]single_model_sim_<source>.csv.gz. LEAGUE=sf for superflex; SOURCE=ktc or fantasycalc; REPS, WEEKS, JOBS.
"""
import json
import os
import pathlib
import sys

os.environ["ARMS"] = "live,nosig"                            # all signal_sim needs to give here
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import market_price_sim as mps  # noqa: E402
import middle_mode as mm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rank_input_sim as rs  # noqa: E402
import signal_sim as ss  # noqa: E402
from common import HERE, out_dir  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

SF, POOL, MODES = mps.SF, mps.POOL, ("con", "mid", "reb")
ARMS = ("live", "regsig_wf", "regall_wf", "regsig_lo", "regall_lo")
SETS = {"pos+age": ["mp", "pos", "age"]}
SETS["+draft"] = SETS["pos+age"] + ["rd", "rookie"]
SETS["+ranks"] = SETS["+draft"] + ["fp", "fpp", "pre", "mpre", "rosr", "rosp", "rospre"]
SETS["+all"] = SETS["+ranks"] + ["g", "ppg", "xppg", "prk", "last", "hurt", "w"]
EMBARGO = {"mv1": 1, "ros": 1, "hold1": 2, "hold3": 4}       # season s's outcome is known by the fall of s + this
EDGES, PICKS = {}, []


def panel():
    """one row per player and snapshot in the market's top POOL, with the features, outcomes and the rules' calls"""
    rs.LOOK.update({(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"]) for r in
                    json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]})
    x = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    mkt = mps.market_ranks(pd.read_parquet(HERE / "data" / "panel.parquet"))
    pr6 = mm.prod_ranks(pd.read_parquet(HERE / "data" / "player_seasons.parquet"))[6]
    out = []
    for (y, tg), q in x[x.target.isin(mps.WEEKS)].groupby(["season", "target"]):
        d, sep = str(q.date.iloc[0]), mkt.get(f"{y}-09-01")
        if d not in mkt or sep is None:
            continue
        mk, last, may = mkt[d], pr6.get(y - 1, {}), mkt.get(f"{y + 1}-05-15")
        q = q.assign(mr=q.gsis_id.map(mk["rank"]), mpr=q.gsis_id.map(mk["prank"]), mpre_=q.gsis_id.map(sep["rank"]).fillna(mps.UNLISTED),
                     last=q.gsis_id.map(last))
        q = q[q.mr <= POOL].copy()
        q["mv1"] = np.maximum(0, POOL - q.gsis_id.map(may["rank"]).fillna(POOL)) if may is not None else np.nan
        c = ss.snapshot_calls(q, last, int(q.w.iloc[0]), mk, sep)
        for m in MODES:
            q[f"kl_{m}"], q[f"kn_{m}"] = q.gsis_id.map(c["live", m]), q.gsis_id.map(c["nosig", m])
        out.append(q)
    z = pd.concat(out, ignore_index=True)
    on_list = z.ros_rank.notna() & (z.ros_rank < 500)
    z = z.assign(mlog=np.log(z.mr), mp=np.log(z.mpr), fp=np.log(z["rank"]), fpp=np.log(z.prank), pre=np.log(z.pre.clip(lower=1)),
                 mpre=np.log(z.mpre_), rosr=np.log(z.ros_rank.where(on_list)), rosp=np.log(z.ros_prank.where(on_list)),
                 rospre=np.log(z.ros_pre), pos=z.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3}), rd=z.draft_round_f.fillna(8),
                 rookie=(z.row_type == "rookie").astype(int), hurt=z.hurt.astype(int), ppg=(z.pts / z.g).where(z.g >= 4),
                 xppg=(z.xpts / z.g).where(z.g >= 4))
    z["prk"] = z.groupby(["season", "target", "position"]).ppg.rank(ascending=False, method="min")
    return z


def fit_pred(tr, te, y, cols):
    """(prediction, what the market rank alone implies) for te"""
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(tr.mlog, tr[y])
    m = HGB(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=40, l2_regularization=5, random_state=0,
            categorical_features=[cols.index("pos") + 1])
    m.fit(tr[["mlog"] + cols].to_numpy(float), tr[y] - iso.predict(tr.mlog))
    b = iso.predict(te.mlog)
    return b + m.predict(te[["mlog"] + cols].to_numpy(float)), b


def prediction_check(z):
    pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 40)
    for y in EMBARGO:
        d = z[z[y].notna()]
        for how, name in (("wf", "walk-forward"), ("lo", "every other season")):
            skill, picks, rows = [], [], []
            for Y in sorted(d.season.unique()):
                te = d[d.season == Y]
                tr = d[d.season <= Y - EMBARGO[y]] if how == "wf" else d[d.season != Y]
                if len(tr) < 200:
                    continue
                for s, cols in SETS.items():
                    p, b = fit_pred(tr, te, y, cols)
                    skill.append({"set": s, "Y": Y, "skill": 100 * (1 - ((te[y] - p) ** 2).mean() / ((te[y] - b) ** 2).mean()),
                                  "rho": spearmanr(p - b, te[y] - b).statistic})
                te = te.assign(res=te[y] - b, e=p - b)                # the last set: +all
                for m in MODES:
                    sell = te[(te[f"kl_{m}"] == "sell") & (te[f"kn_{m}"] != "sell")]
                    buy = te[(te[f"kl_{m}"] == "buy") & (te[f"kn_{m}"] != "buy")]
                    rows.append({"mode": m, "Y": Y, "n sell": len(sell), "rule sell": sell.res.mean(), "model sell": te.nsmallest(len(sell), "e").res.mean(),
                                 "n buy": len(buy), "rule buy": buy.res.mean(), "model buy": te.nlargest(len(buy), "e").res.mean()})
            if not skill:
                print(f"\n-- {y}, {name}: no season can be tested")
                continue
            s = pd.DataFrame(skill)
            print(f"\n-- {y}, {name}: % of the squared error left by the market rank that the model removes, by season, then mean rank correlation of edge and residual")
            print(s.pivot(index="set", columns="Y", values="skill").reindex(SETS).round(1).assign(rho=s.groupby("set").rho.mean().round(2)).to_string())
            print("mean outcome less the market's fair value, signal picks vs the +all model's picks of the same number, per season:")
            print(pd.DataFrame(rows).groupby("mode").mean(numeric_only=True).drop(columns="Y").round(1).to_string())


def edges(z):
    """(how, outcome, season, target) -> Series gsis_id -> the +all model's edge over the market's fair value"""
    cols = SETS["+all"]
    out = {}
    for how in ("wf", "lo"):
        for y in ("ros", "mv1"):
            d = z[z[y].notna()]
            for Y in sorted(z.season.unique()):
                tr = d[d.season <= Y - EMBARGO[y]] if how == "wf" else d[d.season != Y]
                if len(tr) < 200:
                    continue
                te = z[z.season == Y]
                p, b = fit_pred(tr, te, y, cols)
                e = pd.Series(p - b, index=te.index)
                for tg, g in te.groupby("target"):
                    out[how, y, Y, tg] = pd.Series(e[g.index].to_numpy(), index=g.gsis_id)
    return out


def zs(s):
    return (s - s.mean()) / s.std()


def model_calls(arm, m, kl, kn, ros, mv):
    """calls of this arm for team type m; kl, kn = the rules' calls with and without signals"""
    e = {"con": ros, "reb": mv, "mid": zs(ros) + zs(mv)}[m].sort_values()
    if arm.startswith("regall"):
        n_s, n_b, n_u = (sum(kl.get(g) == c for g in e.index) for c in ("sell", "buy", "use"))
        k = dict.fromkeys(e.index, "hold")
        k.update(dict.fromkeys(e.index[:n_s], "sell"))
        k.update(dict.fromkeys(e.index[len(e) - n_b:], "buy"))
        rest = [g for g in mv.sort_values().index if k[g] == "hold"]
        k.update(dict.fromkeys(rest[:n_u], "use"))
        return k
    n_s, n_b = (sum(kl.get(g) == c and kn.get(g) != c for g in e.index) for c in ("sell", "buy"))
    k = dict(kn)
    k.update(dict.fromkeys([g for g in e.index if kn.get(g) != "sell"][:n_s], "sell"))
    k.update(dict.fromkeys([g for g in e.index[::-1] if k.get(g) not in ("sell", "buy")][:n_b], "buy"))
    return k


def snapshot_calls(q, last, w, mk, mkpre):
    """(arm, mode) -> {gsis_id: kind}"""
    c = ss.snapshot_calls(q, last, w, mk, mkpre)
    y, tg = int(q.season.iloc[0]), int(q.target.iloc[0])
    out = {}
    for m in MODES:
        out["live", m] = c["live", m]
        for arm in ARMS[1:]:
            how = arm[-2:]
            ros, mv = EDGES.get((how, "ros", y, tg)), EDGES.get((how, "mv1", y, tg))
            out[arm, m] = {} if ros is None or mv is None else model_calls(arm, m, c["live", m], c["nosig", m], ros, mv)
    PICKS.extend((y, tg, arm, m, g, k) for (arm, m), v in out.items() for g, k in v.items() if k in ("sell", "buy"))
    return out


def who(x):
    """what each list of calls is made of"""
    p = pd.DataFrame(PICKS, columns=["season", "target", "arm", "mode", "gsis_id", "k"])
    p = p.merge(x[["season", "target", "gsis_id", "position", "age", "rank", "hurt", "row_type"]], on=["season", "target", "gsis_id"])
    p = p[p.arm.isin(("live", "regall_lo"))]
    t = p.groupby(["mode", "k", "arm"]).agg(n=("age", "size"), age=("age", "mean"), rank=("rank", "median"), QB=("position", lambda s: (s == "QB").mean()),
                                            RB=("position", lambda s: (s == "RB").mean()), WR=("position", lambda s: (s == "WR").mean()),
                                            hurt=("hurt", "mean"), rookie=("row_type", lambda s: (s == "rookie").mean()))
    print("\n== who gets called: the rules and the no-rules model, every snapshot pooled\n" + t.round(2).to_string())


def summarize(df):
    wf = {t for t in df.t.unique() if ((df.t == t) & (df.arm == "m:regall_wf") & (df.sell + df.buy + df.later > 0)).any()}
    df = df[~(df.arm.str.endswith("_wf") & ~df.t.isin(wf))]
    print(f"\nwalk-forward arms cover cohorts {sorted(int(t) for t in wf)}")
    ss.ARMS = ARMS                                           # its calls are done: now it tables these arms
    ss.summarize(df)
    key = ["t", "rep", "wk", "rank"]
    base = df[df.arm == "none"].set_index(key)[[f"p{k}" for k in range(1, mps.H + 1)] + ["mval1", "mval3"]].add_suffix("_0")
    z = df[df.arm != "none"].join(base, on=key)
    z["yr1"] = (z.p1 - z.p1_0) * 100
    z["yrs3"] = np.where(z.t + 3 <= mps.su.LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in (1, 2, 3)) * 100, np.nan)
    z["v1"], z["v3"] = z.mval1 - z.mval1_0, z.mval3 - z.mval3_0
    own = {m: list(r) for m, r in mps.LENS.values()}
    z = z[[r in own.get(m, ()) for r, m in zip(z["rank"], z["mode"])]].set_index(key + ["mode", "arm"])[["yr1", "yrs3", "v1", "v3"]]
    print("\n=== By cohort, against live")
    for arm in ARMS[1:]:
        d = z.xs(f"m:{arm}", level="arm") - z.xs("m:live", level="arm")
        print(f"\n-- {arm} - live\n{d.groupby(level=['mode', 't']).mean().round(1).to_string()}")
    who(pd.read_csv(out_dir(SF) / "inseason_movers.csv"))


mps.OUT = "single_model_sim"
mps.F_ARMS, mps.M_ARMS, mps.RANDOM, mps.KIND = (), ARMS, (), ()
mps.snapshot_calls, mps.summarize = snapshot_calls, summarize

if __name__ == "__main__":
    z = panel()
    print(f"== {'Superflex' if SF else '1QB'}, {mps.SOURCE}: {len(z)} player-snapshots, seasons {sorted(int(s) for s in z.season.unique())}")
    prediction_check(z)
    EDGES.update(edges(z))
    mps.main()
