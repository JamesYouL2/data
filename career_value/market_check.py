# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "scikit-learn", "scipy"]
# ///
"""Model vs market: does FantasyPros dynasty ECR (this repo) beat the stats model?

Snapshots t = 2019, 2020 (post-draft dynasty ECR scraped mid-May of t+1).
Outcome: roc5_vorp = VORP over seasons t+1..t+5 (fully observed through 2025).
The model is trained only on snapshots 2002-2014, i.e. data a 2020 analyst had.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from model import FEATURES, load, make_model, top_k_hit

HERE = Path(__file__).parent
FILES = HERE.parent / "files"
OUT = HERE / "outputs"


def dynasty_ecr(snap_year: int) -> pd.DataFrame:
    """Dynasty overall ECR from the first scrape on/after May 10 of snap_year."""
    e = pd.read_parquet(FILES / "db_fpecr.parquet",
                        columns=["page_type", "ecr_type", "scrape_date", "id", "pos", "ecr"])
    e = e[(e.ecr_type == "do") & e.page_type.isin(["dynasty-overall", "dynasty-offense"])
          & e.pos.str[:2].isin(["QB", "RB", "WR", "TE"])]
    date = min(x for x in e.scrape_date.unique() if x >= f"{snap_year}-05-10")
    e = e[e.scrape_date == date]
    # one page per date; if both exist prefer dynasty-overall
    page = "dynasty-overall" if (e.page_type == "dynasty-overall").any() else "dynasty-offense"
    e = e[e.page_type == page]
    ids = pd.read_csv(FILES / "db_playerids.csv", dtype=str, usecols=["fantasypros_id", "gsis_id"])
    ids = ids.dropna().drop_duplicates("fantasypros_id")
    e = e.merge(ids, left_on="id", right_on="fantasypros_id", how="inner")
    print(f"  ECR scrape {date}: {len(e)} ranked skill players matched to gsis_id")
    return e[["gsis_id", "ecr"]].drop_duplicates("gsis_id")


def main():
    d = load()
    train = d[d.t.between(2002, 2014)]
    m = make_model(FEATURES).fit(train[FEATURES], train["roc5_vorp"])

    rows = []
    for t in [2019, 2020]:
        snap = d[d.t == t].copy()
        snap["pred"] = m.predict(snap[FEATURES])
        snap = snap.merge(dynasty_ecr(t + 1), on="gsis_id", how="inner")
        rows.append(snap)
    x = pd.concat(rows, ignore_index=True)

    y = x.roc5_vorp
    x["mkt"] = -np.log(x.ecr)
    x["blend"] = (x.groupby("t").pred.rank(pct=True) + x.groupby("t").mkt.rank(pct=True)) / 2
    res = []
    for name in ["pred", "mkt", "blend"]:
        res.append({
            "signal": {"pred": "stats model", "mkt": "FP dynasty ECR", "blend": "rank average"}[name],
            "spearman": spearmanr(y, x[name]).statistic,
            "top24_hit": np.mean([top_k_hit(g.roc5_vorp, g[name]) for _, g in x.groupby("t")]),
            "top50_hit": np.mean([top_k_hit(g.roc5_vorp, g[name], 50) for _, g in x.groupby("t")]),
        })
    res = pd.DataFrame(res).round(3)
    print(f"\nn={len(x)} player-snapshots in market universe")
    print(res.to_string(index=False))

    # incremental information: standardized OLS of log1p(outcome) on both signals
    Z = pd.DataFrame({"model": np.log1p(x.pred), "market": x.mkt})
    Z = (Z - Z.mean()) / Z.std()
    ols = LinearRegression().fit(Z, np.log1p(y))
    print("standardized coefs on log1p(roc5):", dict(zip(Z.columns, ols.coef_.round(3))),
          "R2", round(ols.score(Z, np.log1p(y)), 3))
    for c in Z:
        r2 = LinearRegression().fit(Z[[c]], np.log1p(y)).score(Z[[c]], np.log1p(y))
        print(f"  R2 {c} alone: {r2:.3f}")

    res.to_csv(OUT / "market_check.csv", index=False)
    by_rt = x.groupby("row_type").apply(lambda g: pd.Series({
        "n": len(g), "model_spearman": spearmanr(g.roc5_vorp, g.pred).statistic,
        "ecr_spearman": spearmanr(g.roc5_vorp, g.mkt).statistic}), include_groups=False).round(3)
    print(by_rt)
    by_rt.to_csv(OUT / "market_check_by_rowtype.csv")


if __name__ == "__main__":
    main()
