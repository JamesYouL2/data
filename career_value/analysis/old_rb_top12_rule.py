import sys, pandas as pd, numpy as np
from pathlib import Path
base = Path(sys.argv[1])
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
for h in [1, 2]:
    xs = []
    for m in DYN:
        f = base / "market_rows" / f"{m}_h{h}.parquet"
        if f.exists():
            x = pd.read_parquet(f); x["r"] = x.resid / x[f"vorp{h}"].mean() * 100; x["mk"] = m; xs.append(x)
    x = pd.concat(xs)
    o = x[(x.position == "RB") & (x.age >= 27.5) & (x.row_type == "veteran") & (x.ppg_pos_rank <= 12)]
    o = o.assign(cheap=o.mkt_rank > 60)
    print(f"\n-- h={h}: old RB, top-12 PPG last season; mean resid % across markets per player-year")
    g = o.groupby(["t", "display_name", "cheap"]).agg(age=("age", "first"), ppg_rk=("ppg_pos_rank", "first"),
        mkt_rank=("mkt_rank", "median"), r=("r", "mean")).round(0).reset_index().sort_values(["cheap", "t"])
    print(g.to_string(index=False))
    print("by cohort (all old top-12 PPG RBs):", g.groupby("t").r.median().to_dict())
