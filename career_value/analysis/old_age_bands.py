import sys, glob, pandas as pd, numpy as np
base = sys.argv[1]
rows = []
for f in glob.glob(f"{base}/market_rows/*_h*.parquet"):
    m, h = f.rsplit("/", 1)[1][:-8].rsplit("_h", 1)
    h = int(h)
    if m not in ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess", "mfl_dynasty"]:
        continue
    x = pd.read_parquet(f, columns=["gsis_id", "t", "position", "age", "resid", f"vorp{h}"])
    avg = x[f"vorp{h}"].mean()
    x["band"] = pd.cut(x.age, [27.5, 29.5, 31.5, 99], labels=["28-29", "30-31", "32+"])
    x = x[x.band.notna() & x.position.isin(["RB", "WR", "TE"])]
    for (p, b), g in x.groupby(["position", "band"], observed=True):
        # player-cluster bootstrap CI
        rng = np.random.default_rng(0)
        pid = g.gsis_id.unique()
        by = g.groupby("gsis_id").resid.agg(["sum", "size"])
        bs = []
        for _ in range(300):
            s = by.loc[rng.choice(pid, len(pid))]
            bs.append(s["sum"].sum() / s["size"].sum())
        lo, hi = np.percentile(bs, [5, 95]) / avg * 100
        rows.append({"market": m, "h": h, "pos": p, "band": b, "n": len(g), "cohorts": g.t.nunique(),
                     "pct": g.resid.mean() / avg * 100, "sig": (lo > 0) or (hi < 0)})
r = pd.DataFrame(rows)
r["dyn"] = np.where(r.market == "mfl_dynasty", "MFL", "dynasty value mkts")
# dynasty value markets: average of per-market pct; show range and n of markets
out = []
for (d, p, b, h), g in r.groupby(["dyn", "pos", "band", "h"]):
    g = g[g.n >= 8]
    if g.empty: continue
    cell = f"{g.pct.mean():+.0f}" + ("*" if g.sig.mean() >= .5 else "") + (f" [{g.pct.min():+.0f},{g.pct.max():+.0f}]" if len(g) > 1 else "") + f" n={int(g.n.mean())}"
    out.append({"dyn": d, "pos": p, "band": b, "h": h, "cell": cell})
o = pd.DataFrame(out).pivot(index=["dyn", "pos", "band"], columns="h", values="cell").fillna("")
print(o.to_string())
