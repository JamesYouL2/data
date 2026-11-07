import sys, pandas as pd, numpy as np
from pathlib import Path
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
def load(base, mk, h):
    f = Path(base) / "market_rows" / f"{mk}_h{h}.parquet"
    if not f.exists(): return None
    x = pd.read_parquet(f); x["r"] = x.resid / x[f"vorp{h}"].mean() * 100; x["mk"] = mk; return x
def cell(g):
    top = g.groupby("display_name").r.sum().idxmax()
    coh = g.groupby("t").r.mean()
    return (f"{g.r.mean():+.0f}/{g.r.median():+.0f} {(g.r>0).mean()*100:.0f}%b "
            f"ex{g[g.display_name!=top].r.mean():+.0f} c{(coh>0).sum()}/{len(coh)} n{g.gsis_id.nunique()}")
def run(base, label, pos, cuts, labels, mks, hs=(1,2,3,5), extra=None):
    cells = {}
    for h in hs:
        xs = [load(base, m, h) for m in mks]; xs = [x for x in xs if x is not None]
        if not xs: continue
        x = pd.concat(xs)
        x = x[x.position == pos] if pos else x
        if extra == "tier":
            x = x.assign(b=x.position + " " + pd.cut(x.mkt_pos_rank, [0, 3, 6, 12, 24], labels=["1-3", "4-6", "7-12", "13-24"]).astype(str))
            x = x[~x.b.str.endswith("nan")]
        else:
            x = x.assign(b=pd.cut(x.age, cuts, labels=labels).astype(str))
            x.loc[x.row_type == "rookie", "b"] = "rookie"
        for b, g in x.groupby("b"):
            if len(g) >= 10: cells[(b, h)] = cell(g)
    print(f"\n== {label}   mean/median, %beat, mean ex-best-player, cohorts positive/total, n players")
    print(pd.Series(cells).unstack().fillna("").to_string())
S, Q = "outputs/sf", "outputs"
QBc, QBl = [0, 24.5, 27.5, 31.5, 99], ["QB <=24", "QB 25-27", "QB 28-31", "QB 32+"]
WRc, WRl = [0, 23.5, 25.5, 27.5, 29.5, 99], ["WR <=23", "WR 24-25", "WR 26-27", "WR 28-29", "WR 30+"]
run(S, "SF QBs, dynasty markets pooled", "QB", QBc, QBl, DYN)
for m in DYN: run(S, f"SF QBs, {m}", "QB", QBc, QBl, [m], hs=(1, 2, 3))
run(S, "SF QBs, MFL keeper ADP (mixed-format pool, veterans)", "QB", QBc, QBl, ["mfl_dynasty"], hs=(1, 3, 5, 10))
run(Q, "1QB WRs, dynasty markets pooled", "WR", WRc, WRl, DYN)
for m in DYN: run(Q, f"1QB WRs, {m}", "WR", WRc, WRl, [m], hs=(1, 2, 3))
run(Q, "1QB WRs, MFL keeper ADP (veterans)", "WR", WRc, WRl, ["mfl_dynasty"], hs=(1, 3, 5, 10))
for base, lab in [(Q, "1QB"), (S, "SF")]:
    run(base, f"{lab} ELITE tiers by positional market rank, dynasty pooled", None, None, None, DYN, extra="tier")
    run(base, f"{lab} ELITE tiers, MFL", None, None, None, ["mfl_dynasty"], hs=(1, 3, 5, 10), extra="tier")
