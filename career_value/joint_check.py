# /// script
# dependencies = ["pandas", "pyarrow", "numpy", "statsmodels", "scikit-learn"]
# ///
"""Joint OLS of market residual on key flags (player-clustered SEs), per market x horizon.

Reads outputs/market_rows/{market}_h{h}.parquet written by market_residuals.py.
"""
from pathlib import Path
import pandas as pd, statsmodels.formula.api as smf
from market_residuals import OUT  # outputs/ or outputs/sf/ (LEAGUE=sf)
rows = []
for fpath in sorted((OUT / "market_rows").glob("*_h*.parquet")):
    m, h = fpath.stem.rsplit("_h", 1)
    x = pd.read_parquet(fpath)
    if x.t.nunique() < 3:  # too few cohorts for a joint fit
        continue
    outcome = f"vorp{h}"
    x["r"] = x.resid / x[outcome].mean() * 100  # % of average outcome
    x["short_season"] = (x.g_prior_games == "<=9 games").astype(int)
    x["rush_qb"] = (x.g_qb_rushing == "QB rush >=25 yd/g").astype(int)
    x["rb_old"] = (x.g_pos_x_age == "RB old(28+)").astype(int)
    x["rb_lowtgt"] = (x.g_rb_receiving == "RB <4 tgt/g").astype(int)
    x["wr_young"] = (x.g_pos_x_age == "WR young(<=24)").astype(int)
    x["mkt_fades"] = x.g_mkt_vs_last_year.str.startswith("market fades").astype(int)
    x["rookie"] = (x.row_type == "rookie").astype(int)
    x["age30p"] = (x.age >= 29.5).astype(int)
    f = "r ~ short_season + rush_qb + rb_old + rb_lowtgt + wr_young + mkt_fades + age30p + C(position)"
    if x.rookie.sum() > 0:
        f += " + rookie"
    fit = smf.ols(f, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.gsis_id)[0]})
    for k in fit.params.index:
        if k.startswith("C(") or k == "Intercept":
            continue
        rows.append({"market": m, "h": int(h), "term": k, "coef_pct": fit.params[k], "p": fit.pvalues[k]})
t = pd.DataFrame(rows)
t["cell"] = t.coef_pct.round(0).astype(int).astype(str) + t.p.map(lambda p: "**" if p < .01 else "*" if p < .05 else "")
for h, g in t.groupby("h"):
    print(f"\n== horizon {h} years: coefficient, % of average outcome (* p<.05, ** p<.01)")
    print(g.pivot(index="term", columns="market", values="cell").fillna("").to_string())
t.round(4).to_csv(OUT / "joint_residual_regression.csv", index=False)
