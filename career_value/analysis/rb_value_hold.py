import pandas as pd, numpy as np, statsmodels.formula.api as smf
from pathlib import Path
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
def load(base, mks, h):
    xs = []
    for m in mks:
        f = Path(base) / "market_rows" / f"{m}_h{h}.parquet"
        if f.exists():
            x = pd.read_parquet(f); x["r"] = x.resid / x[f"vorp{h}"].mean() * 100; x["mk"] = m; xs.append(x)
    return pd.concat(xs) if xs else None
def feats(x):
    x = x[x.position == "RB"].copy()
    v = x.row_type == "veteran"
    x["age_b"] = pd.cut(x.age, [0, 23.5, 25.5, 27.5, 99], labels=["<=23", "24-25", "26-27", "28+"]).astype(str)
    x["draft"] = np.where(x.draft_round_f == 1, "rd1", np.where(x.draft_round_f == 2, "rd2", np.where(x.draft_round_f.between(3, 7), "rd3-7", "UDFA")))
    x["tgt"] = np.where(~v, "rookie", np.where(x.targets_pg >= 4, "tgt 4+", np.where(x.targets_pg >= 2.5, "tgt 2.5-4", "tgt <2.5")))
    x["carries"] = np.where(~v, "rookie", np.where(x.carries_pg >= 15, "car 15+", np.where(x.carries_pg >= 10, "car 10-15", "car <10")))
    x["ypc"] = np.where(~v | (x.carries_pg < 5), "n/a", np.where(x.rush_yds_pg / x.carries_pg >= 4.6, "ypc 4.6+", np.where(x.rush_yds_pg / x.carries_pg >= 4.1, "ypc 4.1-4.6", "ypc <4.1")))
    x["size"] = pd.cut(x.weight, [0, 204.5, 214.5, 999], labels=["<205lb", "205-214lb", "215+lb"]).astype(str)
    x["prod"] = np.where(~v, "rookie", np.where(x.ppg_pos_rank <= 12, "ppg RB1-12", np.where(x.ppg_pos_rank <= 24, "ppg RB13-24", "ppg RB25+")))
    x["price"] = pd.cut(x.mkt_pos_rank, [0, 6, 12, 24, 999], labels=["mkt RB1-6", "mkt RB7-12", "mkt RB13-24", "mkt RB25+"]).astype(str)
    # priced below production (discount) vs above (hype)
    gap = x.ppg_pos_rank - x.mkt_pos_rank
    x["gap"] = np.where(~v, "rookie", np.where(gap >= 8, "priced above prod (hype)", np.where(gap <= -8, "priced below prod (discount)", "priced ~prod")))
    x["games_b"] = np.where(~v, "rookie", np.where(x.games <= 9, "<=9 g", "10+ g"))
    x["exp_b"] = np.where(~v, "rookie", np.where(x.exp <= 2, "yr2-3", np.where(x.exp <= 4, "yr4-5", "yr6+")))
    return x
SPLITS = ["age_b", "exp_b", "draft", "tgt", "carries", "ypc", "size", "prod", "price", "gap", "games_b"]
def table(base, mks, hs, label):
    cells = {}
    for h in hs:
        x = load(base, mks, h)
        if x is None: continue
        x = feats(x)
        for s in SPLITS:
            for k, g in x.groupby(s):
                if len(g) < 15 or k in ("nan", "n/a") or (k == "rookie" and s != "exp_b"): continue
                top = g.groupby("display_name").r.sum().idxmax()
                coh = g.groupby("t").r.mean()
                cells[(s, k, h)] = f"{g.r.mean():+.0f}/{g.r.median():+.0f} {(g.r>0).mean()*100:.0f}%b ex{g[g.display_name!=top].r.mean():+.0f} c{(coh>0).sum()}/{len(coh)}"
    t = pd.Series(cells).unstack()
    print(f"\n== {label}: mean/median %beat ex-best cohorts+")
    print(t.fillna("").to_string())
def joint(base, mks, h, label):
    x = feats(load(base, mks, h)); x = x[x.row_type == "veteran"].dropna(subset=["targets_pg", "carries_pg", "weight", "age", "ppg_pos_rank", "mkt_pos_rank"])
    x["ypc_c"] = (x.rush_yds_pg / x.carries_pg.clip(lower=1)).clip(2, 7)
    x["rd1"] = (x.draft_round_f == 1).astype(int)
    x["log_gap"] = np.log(x.ppg_pos_rank) - np.log(x.mkt_pos_rank)  # + = priced above production
    f = "r ~ age + targets_pg + carries_pg + ypc_c + weight + rd1 + log_gap + I(games<=9)"
    fit = smf.ols(f, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.gsis_id)[0]})
    print(f"\n-- joint {label} h={h} (n={len(x)}, players={x.gsis_id.nunique()}): coef % per unit (p)")
    print(pd.DataFrame({"coef": fit.params.round(1), "p": fit.pvalues.round(3)}).drop("Intercept").to_string())
table("outputs", DYN, [1, 2, 3, 5], "1QB RBs, dynasty markets pooled (2019-24)")
table("outputs", ["mfl_dynasty"], [1, 3, 5, 10], "1QB RBs, MFL keeper ADP (2010-24, veterans)")
for h in [2, 3]: joint("outputs", DYN, h, "dynasty pooled")
for h in [3, 5]: joint("outputs", ["mfl_dynasty"], h, "MFL")

print("\n\n######## NAMES")
for base, mks, h, lab in [("outputs", DYN, 3, "dynasty pooled h3"), ("outputs", ["mfl_dynasty"], 5, "MFL h5")]:
    x = feats(load(base, mks, h))
    x = x[x.mkt_pos_rank <= 24]
    g = x.groupby(["display_name", "t"]).agg(age=("age", "first"), exp=("exp_b", "first"), car=("carries_pg", "first"), tgt=("targets_pg", "first"),
        draft=("draft", "first"), prod=("ppg_pos_rank", "first"), mkt=("mkt_pos_rank", "median"), r=("r", "mean")).round(1).reset_index()
    print(f"\n-- {lab}: top-24-priced RBs, best 20 / worst 12 player-years")
    print(g.sort_values("r", ascending=False).head(20).to_string(index=False))
    print(g.sort_values("r").head(12).to_string(index=False))
    print("share beating price:", (g.r > 0).mean().round(2), " n", len(g))
