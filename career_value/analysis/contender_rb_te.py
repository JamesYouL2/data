"""Contender view: which RBs / TEs beat their dynasty price over the next 1-2 seasons? Pooled dynasty markets, 1QB and SF."""
import pandas as pd, numpy as np
from pathlib import Path
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
def load(base, h):
    xs = []
    for m in DYN:
        f = Path(base) / "market_rows" / f"{m}_h{h}.parquet"
        if f.exists():
            x = pd.read_parquet(f); x["r"] = x.resid / x[f"vorp{h}"].mean() * 100; x["mk"] = m; xs.append(x)
    return pd.concat(xs)
def feats(x, pos):
    x = x[x.position == pos].copy(); v = x.row_type == "veteran"
    x["age_b"] = np.where(~v, "rookie", pd.cut(x.age, [0, 24.5, 26.5, 28.5, 99], labels=["<=24", "25-26", "27-28", "29+"]).astype(str))
    x["prod"] = np.where(~v, "rookie", np.where(x.ppg_pos_rank <= 6, "ppg 1-6", np.where(x.ppg_pos_rank <= 12, "ppg 7-12", np.where(x.ppg_pos_rank <= 24, "ppg 13-24", "ppg 25+/none"))))
    x["price"] = pd.cut(x.mkt_pos_rank, [0, 3, 6, 12, 24, 999], labels=["mkt 1-3", "mkt 4-6", "mkt 7-12", "mkt 13-24", "mkt 25+"]).astype(str)
    gap = x.ppg_pos_rank - x.mkt_pos_rank
    x["gap"] = np.where(~v, "rookie", np.where(gap <= -6, "discount (mkt >=6 below ppg)", np.where(gap >= 6, "hype (mkt >=6 above ppg)", "~fair")))
    t = x.targets_pg
    x["tgt"] = np.where(~v, "rookie", (np.where(t >= 4, "tgt 4+", np.where(t >= 2.5, "tgt 2.5-4", "tgt <2.5")) if pos == "RB"
                                      else np.where(t >= 7, "tgt 7+", np.where(t >= 5, "tgt 5-7", "tgt <5"))))
    x["games_b"] = np.where(~v, "rookie", np.where(x.games <= 9, "<=9 g", "10+ g"))
    x["old_prod"] = np.where(v & (x.age >= 27.5) & (x.ppg_pos_rank <= 12), "27.5+ & ppg top-12", "")
    x["old_prod_cheap"] = np.where(v & (x.age >= 27.5) & (x.ppg_pos_rank <= 12) & (gap <= -6), "27.5+ & top-12 ppg & discount", "")
    x["young_hype"] = np.where(v & (x.age < 25) & (gap >= 6), "<25 & hype", "")
    return x
SPL = ["age_b", "prod", "price", "gap", "tgt", "games_b", "old_prod", "old_prod_cheap", "young_hype"]
def cell(g):
    coh = g.groupby("t").r.mean(); top = g.groupby("display_name").r.sum().idxmax()
    return f"{g.r.mean():+.0f}/{g.r.median():+.0f} {(g.r>0).mean()*100:.0f}%b ex{g[g.display_name!=top].r.mean():+.0f} c{(coh>0).sum()}/{len(coh)} n{g.gsis_id.nunique()}"
for base, lg in [("outputs", "1QB"), ("outputs/sf", "SF")]:
    for pos in ["RB", "TE"]:
        cells = {}
        for h in [1, 2]:
            x = feats(load(base, h), pos)
            cells[("all", pos, h)] = cell(x)
            for s in SPL:
                for k, g in x.groupby(s):
                    if k in ("", "nan") or g.gsis_id.nunique() < 5 or (k == "rookie" and s != "age_b"): continue
                    cells[(s, k, h)] = cell(g)
            # by market for the contender-relevant groups
            for k, g in x[x.old_prod != ""].groupby("mk"): cells[("mkt: 27.5+ top-12", k, h)] = cell(g)
            for k, g in x[x.age_b == "29+"].groupby("mk"): cells[("mkt: 29+", k, h)] = cell(g)
        print(f"\n==== {lg} {pos}: mean/median %beat ex-best cohorts+ players")
        print(pd.Series(cells).unstack().fillna("").to_string())
        if pos == "TE":
            x = feats(load(base, 2), pos); x = x[x.row_type == "veteran"]
            g = x.groupby(["display_name", "t"]).agg(age=("age", "first"), tgt=("targets_pg", "first"), prod=("ppg_pos_rank", "first"),
                                                       mkt=("mkt_pos_rank", "median"), r=("r", "mean")).round(1).reset_index()
            print(f"-- {lg} TE h2 best 12 / worst 8 (top-12 priced)"); g = g[g.mkt <= 12]
            print(g.sort_values("r", ascending=False).head(12).to_string(index=False)); print(g.sort_values("r").head(8).to_string(index=False))
