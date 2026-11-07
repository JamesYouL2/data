import sys, pandas as pd, numpy as np
from pathlib import Path
base = Path(sys.argv[1])
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
def load(mk, h):
    f = base / "market_rows" / f"{mk}_h{h}.parquet"
    if not f.exists(): return None
    x = pd.read_parquet(f); y = f"vorp{h}"
    x["r"] = x.resid / x[y].mean() * 100
    x["mk"] = mk
    return x
def slices(x):
    o = x[(x.position.isin(["RB", "TE"])) & (x.age >= 27.5) & (x.row_type == "veteran")].copy()
    o["agegrp"] = np.where(o.age >= 29.5, "30+", "28-29")
    o["tier"] = o.g_market_tier
    o["prod"] = np.where(o.ppg_pos_rank <= 6, "top6 ppg last yr", np.where(o.ppg_pos_rank <= 12, "7-12 ppg", "13+ ppg"))
    o["role"] = np.where(o.position == "RB", o.g_rb_receiving, "TE")
    o["short"] = o.g_prior_games
    return o
def summ(o, keys):
    rows = []
    for k, g in o.groupby(keys):
        if len(g) < 12: continue
        # robustness: drop the single best player's rows
        top = g.groupby("display_name").r.sum().idxmax()
        rows.append({"slice": " | ".join(map(str, k if isinstance(k, tuple) else (k,))), "n": len(g),
                     "players": g.gsis_id.nunique(), "mean%": round(g.r.mean()), "median%": round(g.r.median()),
                     "beat%": round((g.r > 0).mean() * 100), "ex_top%": round(g[g.display_name != top].r.mean()),
                     "top": top})
    return pd.DataFrame(rows)
for label, mks in [("DYNASTY (FP/KTC/FC/DP pooled)", DYN), ("MFL", ["mfl_dynasty"])]:
    for h in [1, 2, 3, 5]:
        xs = [load(m, h) for m in mks]; xs = [x for x in xs if x is not None]
        if not xs: continue
        o = slices(pd.concat(xs))
        print(f"\n===== {label} h={h}  cohorts {sorted(o.t.unique())}")
        for keys in [["position", "agegrp"], ["position", "tier"], ["position", "prod"], ["role"], ["position", "short"]]:
            s = summ(o, keys)
            if len(s): print(s.to_string(index=False))
