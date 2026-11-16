"""Q: does the redraft gap earn a place in a model of dynasty price, position, age and draft capital, at market prices?

single_model_sim.py's panel (players in the market's top POOL after weeks 4, 8 and 12, seasons from 2021) and its model (gradient boosting
over isotonic(market rank), so it starts from the price). Feature sets:
  base            market positional rank, position, age, NFL draft round, rookie or not
  base+gap        + gap: log(FP rest-of-season positional rank / market positional rank), and whether he is off the rest-of-season list
  base+gap+prod   + the same gap for this season's points-per-game positional rank
  ablations       base+gap without age, without draft capital, without position; the gap with the price only; base + production gap
                  only; base + FP dynasty vs market gap; single_model_sim.py's +all (every input)
   `ablate` runs five follow-ups on the same check: the gap's two directions apart, the gap for 25-and-under or 26+ only, base against
   base+gap by snapshot week, the off-list flag without the gap, and +all with each group of inputs dropped (and base+gap with each added).
1. Prediction (`check`): % of the squared error the market rank leaves that each set removes, and the rank correlation of its edge with
   the residual, by season, walk-forward (wf) and trained on every other season (lo). Outcomes as single_model_sim.py: ros (rest-of-season
   VORP), mv1 (value at the market's rank next May), hold1, hold3.
2. The league sim of market_price_sim.py, trades at the market rank, as many sells, buys and sell-next-Mays as the site's rules make:
     base_*, gap_*   no rules: calls off the base and the base+gap model's edge
     siggap_*        the rules without their signals, plus base+gap model picks in place of the signals
Caveats as single_model_sim.py: few seasons (walk-forward covers 2022 on for 1-year outcomes), untuned model, KTC and FantasyCalc share
players. Writes outputs/[sf/]gap_model_sim_<source>.csv.gz. LEAGUE=sf; SOURCE=ktc or fantasycalc; REPS, WEEKS, JOBS.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import single_model_sim as sm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

mps, ss, MODES, EMBARGO = sm.mps, sm.ss, sm.MODES, sm.EMBARGO
BASE = ["mp", "pos", "age", "rd", "rookie"]
GAP = ["gap", "off"]
SETS = {"base": BASE, "base+gap": BASE + GAP, "base+gap+prod": BASE + GAP + ["pgap"],
        "base+gap, no age": [c for c in BASE if c != "age"] + GAP, "base+gap, no draft": ["mp", "pos", "age"] + GAP,
        "base+gap, no position": ["mp", "age", "rd", "rookie"] + GAP, "price+gap only": ["mp"] + GAP,
        "base+prod gap": BASE + ["pgap"], "base+FP gap": BASE + ["fgap"], "+all": sm.SETS["+all"]}
ARMS = ("live", "base_wf", "gap_wf", "siggap_wf", "base_lo", "gap_lo", "siggap_lo")
SET_OF = {"base": "base", "gap": "base+gap", "siggap": "base+gap"}
EDGES = {}


def panel():
    z = sm.panel()
    z["gap"], z["off"] = z.rosp - z.mp, z.rosp.isna().astype(int)
    z["pgap"], z["fgap"] = np.log(z.prk) - z.mp, z.fpp - z.mp
    z["gup"], z["gdn"] = z.gap.clip(upper=0), z.gap.clip(lower=0)          # ROS rank better / worse than the market's
    z["gyoung"], z["gold"] = z.gap.where(z.age < 26, 0), z.gap.where(z.age >= 26, 0)
    return z


ALL = sm.SETS["+all"]
GROUPS = {"FP dynasty rank": ["fp", "fpp"], "preseason ranks": ["pre", "mpre", "rospre"], "ROS ranks": ["rosr", "rosp"],
          "production": ["g", "ppg", "xppg", "prk", "last"], "games played": ["g"], "points per game": ["ppg", "xppg", "prk"],
          "last season": ["last"], "injury status": ["hurt"], "week": ["w"]}
ABL = {"base": BASE, "base+gap": BASE + GAP,
       "1 gap up only (ROS better)": BASE + ["gup"], "1 gap down only (ROS worse)": BASE + ["gdn"], "1 gap down + off list": BASE + ["gdn", "off"],
       "2 gap for 25- only": BASE + ["gyoung", "off"], "2 gap for 26+ only": BASE + ["gold", "off"],
       "4 gap, no off-list flag": BASE + ["gap"], "4 off-list flag, no gap": BASE + ["off"],
       "5 +all": ALL, **{f"5 +all less {k}": [c for c in ALL if c not in v] for k, v in GROUPS.items()},
       **{f"5 base+gap plus {k}": BASE + GAP + [c for c in v if c not in BASE + GAP] for k, v in GROUPS.items()}}


def ablate(z):
    """the five follow-ups: gap direction, gap by age class, by week, the off-list flag, and which of +all's inputs carry it"""
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)
    for y in ("ros", "mv1"):
        d = z[z[y].notna()]
        for how, name in (("wf", "walk-forward"), ("lo", "every other season")):
            rows, wk = [], []
            for Y in sorted(d.season.unique()):
                te, tr = d[d.season == Y], split(d, Y, y, how)
                if len(tr) < 200:
                    continue
                for s_, cols in ABL.items():
                    p, b = fit_pred(tr, te, y, cols)
                    e, res = pd.Series(p - b, index=te.index), te[y] - b
                    k = max(1, len(te) // 10)
                    o = np.argsort(e.to_numpy())
                    rows.append({"set": s_, "Y": Y, "skill": 100 * (1 - ((te[y] - p) ** 2).mean() / (res ** 2).mean()), "rho": spearmanr(e, res).statistic,
                                 "top10": res.to_numpy()[o[-k:]].mean(), "bot10": res.to_numpy()[o[:k]].mean()})
                    if s_ in ("base", "base+gap"):
                        for tg, g in te.groupby("target"):
                            r, pp = res[g.index], p[te.index.get_indexer(g.index)]
                            wk.append({"set": s_, "Y": Y, "wk": tg, "skill": 100 * (1 - ((g[y] - pp) ** 2).mean() / (r ** 2).mean()),
                                       "rho": spearmanr(e[g.index], r).statistic})
            if not rows:
                continue
            s_ = pd.DataFrame(rows)
            g = s_.groupby("set")
            rho = s_.pivot(index="set", columns="Y", values="rho")
            t = pd.DataFrame({"skill": g.skill.mean().round(1), "rho": g.rho.mean().round(3), "top10": g.top10.mean().round(1),
                              "bot10": g.bot10.mean().round(1)}).reindex(ABL)
            ref = {k: ("5 +all" if k.startswith("5 +all less") else "base+gap" if k.startswith(("5 base+gap plus", "4 ")) else "base") for k in ABL}
            t["rho vs"] = [f"{int((rho.loc[k] > rho.loc[ref[k]]).sum())}/{rho.shape[1]} above {ref[k]}" if k != ref[k] and k != "base" else "" for k in ABL]
            print(f"\n-- {y}, {name}, seasons {sorted(int(v) for v in s_.Y.unique())}: skill = % of the squared error left by the market rank removed; rho = rank "
                  "correlation of edge and residual; top10 / bot10 = mean residual of the model's top / bottom tenth")
            print(t.to_string())
            w = pd.DataFrame(wk).groupby(["wk", "set"])[["skill", "rho"]].mean().unstack("set").round(3)
            ww = pd.DataFrame(wk).pivot_table(index=["wk", "Y"], columns="set", values="rho")
            w.columns = [f"{m} {c}" for m, c in w.columns]
            w["gap above base in"] = (ww["base+gap"] > ww["base"]).groupby("wk").sum().astype(str) + "/" + ww.groupby("wk").size().astype(str)
            print("3 by week of the snapshot:")
            print(w.to_string())


def fit_pred(tr, te, y, cols):
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(tr.mlog, tr[y])
    m = HGB(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=40, l2_regularization=5, random_state=0,
            categorical_features=[cols.index("pos") + 1] if "pos" in cols else None)
    m.fit(tr[["mlog"] + cols].to_numpy(float), tr[y] - iso.predict(tr.mlog))
    b = iso.predict(te.mlog)
    return b + m.predict(te[["mlog"] + cols].to_numpy(float)), b


def split(d, Y, y, how):
    return d[d.season <= Y - EMBARGO[y]] if how == "wf" else d[d.season != Y]


def check(z):
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)
    for y in EMBARGO:
        d = z[z[y].notna()]
        for how, name in (("wf", "walk-forward"), ("lo", "every other season")):
            rows = []
            for Y in sorted(d.season.unique()):
                te, tr = d[d.season == Y], split(d, Y, y, how)
                if len(tr) < 200:
                    continue
                for s, cols in SETS.items():
                    p, b = fit_pred(tr, te, y, cols)
                    e, res = p - b, te[y] - b
                    k = max(1, len(te) // 10)
                    o = np.argsort(e)
                    rows.append({"set": s, "Y": Y, "skill": 100 * (1 - ((te[y] - p) ** 2).mean() / (res ** 2).mean()), "rho": spearmanr(e, res).statistic,
                                 "top10": res.to_numpy()[o[-k:]].mean(), "bot10": res.to_numpy()[o[:k]].mean()})
            if not rows:
                print(f"\n-- {y}, {name}: no season can be tested")
                continue
            s = pd.DataFrame(rows)
            t = s.pivot(index="set", columns="Y", values="skill").reindex(SETS).round(1)
            g = s.groupby("set")
            t = t.assign(mean=g.skill.mean().round(1), rho=g.rho.mean().round(3), top10=g.top10.mean().round(1), bot10=g.bot10.mean().round(1))
            rho = s.pivot(index="set", columns="Y", values="rho")
            t["gap adds rho in"] = ""
            for a, b in (("base+gap", "base"), ("base+gap+prod", "base+gap"), ("base+prod gap", "base"), ("base+FP gap", "base")):
                t.loc[a, "gap adds rho in"] = f"{int((rho.loc[a] > rho.loc[b]).sum())}/{rho.shape[1]} vs {b}"
            print(f"\n-- {y}, {name}: % of the squared error left by the market rank that the model removes, by season; mean; mean rank correlation of "
                  "edge and residual; mean residual of the model's top and bottom tenth")
            print(t.to_string())


def edges(z):
    out = {}
    for name in set(SET_OF.values()):
        for how in ("wf", "lo"):
            for y in ("ros", "mv1"):
                d = z[z[y].notna()]
                for Y in sorted(z.season.unique()):
                    tr = split(d, Y, y, how)
                    if len(tr) < 200:
                        continue
                    te = z[z.season == Y]
                    p, b = fit_pred(tr, te, y, SETS[name])
                    e = pd.Series(p - b, index=te.index)
                    for tg, g in te.groupby("target"):
                        out[name, how, y, Y, tg] = pd.Series(e[g.index].to_numpy(), index=g.gsis_id)
    return out


def snapshot_calls(q, last, w, mk, mkpre):
    c = ss.snapshot_calls(q, last, w, mk, mkpre)
    y, tg = int(q.season.iloc[0]), int(q.target.iloc[0])
    out = {}
    for m in MODES:
        out["live", m] = c["live", m]
        for arm in ARMS[1:]:
            kind, how = arm.split("_")
            ros, mv = EDGES.get((SET_OF[kind], how, "ros", y, tg)), EDGES.get((SET_OF[kind], how, "mv1", y, tg))
            out[arm, m] = ({} if ros is None or mv is None else
                           sm.model_calls("regsig" if kind == "siggap" else "regall", m, c["live", m], c["nosig", m], ros, mv))
    return out


def summarize(df):
    wf = {t for t in df.t.unique() if ((df.t == t) & (df.arm == "m:gap_wf") & (df.sell + df.buy + df.later > 0)).any()}
    df = df[~(df.arm.str.endswith("_wf") & ~df.t.isin(wf))]
    print(f"\nwalk-forward arms cover cohorts {sorted(int(t) for t in wf)}")
    ss.ARMS = ARMS
    ss.summarize(df)
    key = ["t", "rep", "wk", "rank"]
    base = df[df.arm == "none"].set_index(key)[[f"p{k}" for k in range(1, mps.H + 1)] + ["mval1", "mval3"]].add_suffix("_0")
    z = df[df.arm != "none"].join(base, on=key)
    z["yr1"] = (z.p1 - z.p1_0) * 100
    z["yrs3"] = np.where(z.t + 3 <= mps.su.LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in (1, 2, 3)) * 100, np.nan)
    z["v1"], z["v3"] = z.mval1 - z.mval1_0, z.mval3 - z.mval3_0
    own = {m: list(r) for m, r in mps.LENS.values()}
    z = z[[r in own.get(m, ()) for r, m in zip(z["rank"], z["mode"])]].set_index(key + ["mode", "arm"])[["yr1", "yrs3", "v1", "v3"]]

    def cell(v):
        v = v.dropna()
        if v.empty:
            return ""
        c, by = v.groupby(level=["t", "rep", "wk"]).mean(), v.groupby(level="t").mean()
        return f"{c.mean():+6.1f}±{1.96 * c.std(ddof=1) / np.sqrt(len(c)):4.1f} {int((by > 0).sum())}/{int((by != 0).sum())}"
    print("\n=== What the gap adds: the base+gap model's calls less the base model's (no rules in either)")
    for how in ("wf", "lo"):
        d = z.xs(f"m:gap_{how}", level="arm") - z.xs(f"m:base_{how}", level="arm")
        print(f"\n-- gap_{how} - base_{how}")
        print(pd.DataFrame([{"mode": m, **{c: cell(d.xs(m, level="mode")[c]) for c in d.columns}} for m in MODES]).to_string(index=False))
        print(d.groupby(level=["mode", "t"]).mean().round(1).to_string())


mps.OUT = "gap_model_sim"
mps.F_ARMS, mps.M_ARMS, mps.RANDOM, mps.KIND = (), ARMS, (), ()
mps.snapshot_calls, mps.summarize = snapshot_calls, summarize

if __name__ == "__main__":
    z = panel()
    print(f"== {'Superflex' if sm.SF else '1QB'}, {mps.SOURCE}: {len(z)} player-snapshots, seasons {sorted(int(s) for s in z.season.unique())}")
    if sys.argv[1:] == ["check"]:
        check(z)
    elif sys.argv[1:] == ["ablate"]:
        ablate(z)
    else:
        EDGES.update(edges(z))
        mps.main()
