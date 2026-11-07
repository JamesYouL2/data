import pandas as pd, numpy as np, statsmodels.formula.api as smf, sys
from pathlib import Path
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
BASE = sys.argv[1] if len(sys.argv) > 1 else "outputs"
def load(mks, h):
    xs = []
    for m in mks:
        f = Path(BASE) / "market_rows" / f"{m}_h{h}.parquet"
        if f.exists():
            x = pd.read_parquet(f); x["r"] = x.resid / x[f"vorp{h}"].mean() * 100; x["mk"] = m; xs.append(x)
    return pd.concat(xs) if xs else None
def feats(x):
    x = x[x.row_type == "veteran"].copy()
    # "injured": short season but established (market still has him top-36 at position, or was a starter the year before)
    x["short"] = x.games <= 9
    x["est"] = (x.mkt_pos_rank <= 36) | ((x.games_prev >= 10) & (x.ppg_prev >= 10))
    x = x[x.est]
    x["grp"] = np.where(x.short, "INJ", "healthy")
    x["pos"] = x.position
    x["age_b"] = pd.cut(x.age, [0, 25.5, 28.5, 99], labels=["<=25", "26-28", "29+"]).astype(str)
    # pre-injury production: per-game in the short season (>=3 games) else prior season
    x["ppg_any"] = np.where(x.games >= 3, x.ppg, x.ppg_prev)
    x["ppgr"] = x.groupby(["t", "position", "mk"]).ppg_any.rank(ascending=False)
    x["prod"] = np.where(x.ppgr <= 12, "ppg pos1-12", "ppg pos13+")
    x["prone"] = np.where(x.games_prev <= 9, "2nd short yr (prone)", np.where(x.games_prev.isna(), "n/a", "healthy yr before"))
    x["missed"] = np.where(x.games <= 4, "0-4 g", "5-9 g")
    x["price"] = pd.cut(x.mkt_pos_rank, [0, 12, 24, 999], labels=["mkt pos1-12", "pos13-24", "pos25+"]).astype(str)
    x["drop"] = np.where(x.ppgr + 8 <= x.mkt_pos_rank, "mkt discounts (priced >=8 below ppg)", "mkt holds price")
    return x
def cell(g):
    coh = g.groupby("t").r.mean(); top = g.groupby("display_name").r.sum().idxmax()
    return f"{g.r.mean():+.0f}/{g.r.median():+.0f} {(g.r>0).mean()*100:.0f}%b ex{g[g.display_name!=top].r.mean():+.0f} c{(coh>0).sum()}/{len(coh)} n{g.gsis_id.nunique()}"
def table(mks, hs, label):
    cells = {}
    for h in hs:
        x = load(mks, h)
        if x is None: continue
        x = feats(x)
        for k, g in x.groupby("grp"): cells[("all", k, h)] = cell(g)
        for k, g in x.groupby(["pos", "grp"]): cells[("pos", " ".join(k), h)] = cell(g)
        i = x[x.short]
        for s in ["age_b", "prod", "prone", "missed", "price", "drop"]:
            for k, g in i.groupby(s):
                if g.gsis_id.nunique() >= 6 and k not in ("n/a", "nan"): cells[(s, k, h)] = cell(g)
        for k, g in i.groupby(["pos", "age_b"]):
            if g.gsis_id.nunique() >= 6: cells[("pos x age", " ".join(k), h)] = cell(g)
        for k, g in i.groupby(["pos", "prod"]):
            if g.gsis_id.nunique() >= 6: cells[("pos x prod", " ".join(k), h)] = cell(g)
    print(f"\n== {label}  [mean/median %beat ex-best cohorts+ players]")
    print(pd.Series(cells).unstack().fillna("").to_string())
def names(mks, h, label):
    x = feats(load(mks, h)); i = x[x.short]
    g = i.groupby(["display_name", "t"]).agg(pos=("pos", "first"), age=("age", "first"), g=("games", "first"), gprev=("games_prev", "first"),
        ppg=("ppg_any", "first"), mkt=("mkt_pos_rank", "median"), r=("r", "mean")).round(1).reset_index()
    print(f"\n-- {label}: injured, best 15 / worst 10")
    print(g.sort_values("r", ascending=False).head(15).to_string(index=False))
    print(g.sort_values("r").head(10).to_string(index=False))
table(DYN, [1, 2, 3, 5], "dynasty pooled")
table(["mfl_dynasty"], [1, 3, 5, 10], "MFL keeper ADP")
for mks, h, lab in [(DYN, 3, "dynasty"), (["mfl_dynasty"], 5, "MFL")]:
    x = feats(load(mks, h)); x = x[x.short].dropna(subset=["age", "ppg_any", "mkt_pos_rank"])
    x["log_disc"] = np.log(x.mkt_pos_rank) - np.log(x.ppgr)
    x["prone_i"] = (x.games_prev <= 9).astype(int)
    fit = smf.ols("r ~ age + C(pos) + log_disc + prone_i + I(games<=4)", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.gsis_id)[0]})
    print(f"\n-- joint among injured, {lab} h={h}, n={len(x)} players={x.gsis_id.nunique()}")
    print(pd.DataFrame({"coef": fit.params.round(1), "p": fit.pvalues.round(3)}).to_string())
names(DYN, 3, "dynasty h3"); names(["mfl_dynasty"], 5, "MFL h5")
