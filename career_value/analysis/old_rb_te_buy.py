import sys, pandas as pd, numpy as np
from pathlib import Path
base = Path(sys.argv[1])
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
def load(mk, h):
    f = base / "market_rows" / f"{mk}_h{h}.parquet"
    if not f.exists(): return None
    x = pd.read_parquet(f); y = f"vorp{h}"
    x["r"] = x.resid / x[y].mean() * 100
    return x
def slices(o):
    o = o[(o.position.isin(["RB", "TE"])) & (o.age >= 27.5) & (o.row_type == "veteran")].copy()
    P = o.position
    out = []
    out.append(("all old", P + " 28+"))
    out.append(("age", P + " " + np.where(o.age >= 29.5, "30+", "28-29")))
    out.append(("price", P + " " + np.where(o.mkt_rank <= 60, "pricey (top60)", "cheap (61+)")))
    out.append(("prod", P + " " + np.where(o.ppg_pos_rank <= 6, "top6 ppg", "not top6 ppg")))
    out.append(("role", pd.Series(np.where(P == "RB", np.where(o.targets_pg >= 4, "RB >=4 tgt/g", "RB <4 tgt/g"), "TE"), index=o.index)))
    out.append(("elite+cheap", P + " " + np.where((o.ppg_pos_rank <= 12) & (o.mkt_rank > 60), "top12 ppg & cheap", "other")))
    rows = []
    for name, lab in out:
        for k, g in o.groupby(lab):
            rows.append((name, k, g))
    return rows
for label, mks in [("DYNASTY pooled FP/KTC/FC/DP", DYN), ("MFL keeper ADP 2010+", ["mfl_dynasty"])]:
    cells = {}
    for h in [1, 2, 3, 5]:
        xs = [load(m, h) for m in mks]; xs = [x for x in xs if x is not None]
        if not xs: continue
        for name, k, g in slices(pd.concat(xs)):
            if len(g) < 15 or k.endswith("other") or k == "TE": continue
            top = g.groupby("display_name").r.sum().idxmax()
            cells[(k, h)] = f"{g.r.mean():+.0f}/{g.r.median():+.0f}/{(g.r>0).mean()*100:.0f}% ({g[g.display_name!=top].r.mean():+.0f}) n{g.gsis_id.nunique()}"
    t = pd.Series(cells).unstack()
    print(f"\n== {label}: mean/median/%beat (mean excl. best player) n players")
    print(t.fillna("").to_string())
