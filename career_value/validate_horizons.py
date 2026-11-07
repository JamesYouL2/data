# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "scikit-learn", "scipy", "statsmodels"]
# ///
"""How well does each market (and the stats model) rank players for value over the next h seasons?

Horizons h = 1, 2, 3, 5, 10. A cohort (snapshot t) is scored at horizon h only if t+h <= 2025.

Universe per cohort: union of every available market's top 150 at that snapshot, restricted to
players in the panel (vets who played in season t, plus that year's drafted rookies). A player
missing from a market gets that market's worst rank. Two panels:
  all       - every market except MFL (MFL ADP is veterans only)
  veterans  - veteran rows only; MFL included

The stats model is trained strictly in real time: for snapshot T and horizon h it only sees
snapshots t <= T-h (whose h-year outcome was complete by season T).

Because markets cover different cohorts, a mean over each market's own cohorts is not
comparable across markets. Head-to-head tables compare each market with KTC and with the model
on exactly the cohorts they share.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import market_residuals as mr
from model import FEATURES, load, make_model, top_k_hit

UNIV_TOP = 150
FEATS = FEATURES + (["vorp_sf"] if mr.SF else [])
OUT = mr.OUT


def model_scores(panel, outc, cohorts_by_h):
    d = panel.merge(outc, on=["gsis_id", "t"], how="left")
    rows = []
    for h, cohorts in cohorts_by_h.items():
        y = f"vorp{h}"
        for T in sorted(cohorts):
            tr = d[(d.t <= T - h) & d[y].notna()]
            m = make_model(FEATS).fit(tr[FEATS], tr[y])
            te = d[d.t == T]
            rows.append(pd.DataFrame({"gsis_id": te.gsis_id, "t": T, "h": h, "mkt": -m.predict(te[FEATS])}))
    return pd.concat(rows, ignore_index=True)


def evaluate(markets, model, panel, outc, vets_only):
    pan = panel[["gsis_id", "t", "row_type"]]
    if vets_only:
        pan = pan[pan.row_type == "veteran"]
    rows = []
    for h in mr.HORIZONS:
        y = f"vorp{h}"
        mk = {k: v for k, v in markets.items() if (h == 1 or k not in mr.REDRAFT)}
        if not vets_only:
            mk.pop("mfl_dynasty", None)
        for t in sorted(set().union(*[set(v.t) for v in mk.values()])):
            if t + h > mr.LAST:
                continue
            avail = {k: v[v.t == t].merge(pan, on=["gsis_id", "t"]) for k, v in mk.items()}
            avail = {k: v for k, v in avail.items() if len(v) >= 100}
            if not avail:
                continue
            univ = set()
            for v in avail.values():
                univ |= set(v.nsmallest(UNIV_TOP, "mkt").gsis_id)
            u = pd.DataFrame({"gsis_id": sorted(univ), "t": t}).merge(outc[["gsis_id", "t", y]], on=["gsis_id", "t"])
            mod = model[(model.t == t) & (model.h == h)]
            for name, v in list(avail.items()) + [("model", mod)]:
                s = u.merge(v[["gsis_id", "mkt"]], on="gsis_id", how="left").mkt
                s = s.fillna(s.max() + 1)  # missing from market -> worst rank
                rows.append({"panel": "veterans" if vets_only else "all", "h": h, "t": t, "market": name,
                             "n": len(u), "spearman": spearmanr(-s, u[y]).statistic,
                             "top24_hit": top_k_hit(u[y].to_numpy(), -s.to_numpy())})
    return pd.DataFrame(rows)


def head_to_head(r, ref):
    """Mean (market - ref) Spearman on shared cohorts."""
    a = r.pivot_table(index=["panel", "h", "t"], columns="market", values="spearman")
    out = []
    for (p, h), g in a.groupby(level=[0, 1]):
        for mkt in g.columns:
            if mkt == ref:
                continue
            both = g[[mkt, ref]].dropna()
            if len(both):
                out.append({"panel": p, "h": h, "market": mkt, "vs": ref, "cohorts": len(both),
                            "market_rho": both[mkt].mean(), "ref_rho": both[ref].mean(),
                            "diff": (both[mkt] - both[ref]).mean(),
                            "wins": int((both[mkt] > both[ref]).sum())})
    return pd.DataFrame(out)


def main():
    panel = load()
    outc = mr.outcomes()
    markets = mr.all_markets(panel)
    for k, v in markets.items():
        markets[k] = v[["gsis_id", "t", "mkt"]].dropna()
    cohorts = sorted(set().union(*[set(v.t) for v in markets.values()]))
    cohorts_by_h = {h: [t for t in cohorts if t + h <= mr.LAST] for h in mr.HORIZONS}
    print("training real-time model ...", flush=True)
    model = model_scores(panel, outc, cohorts_by_h)

    r = pd.concat([evaluate(markets, model, panel, outc, False),
                   evaluate(markets, model, panel, outc, True)], ignore_index=True)
    r.to_csv(OUT / "horizon_validation_by_cohort.csv", index=False)

    summ = (r.groupby(["panel", "market", "h"])
            .agg(cohorts=("t", "nunique"), first=("t", "min"), last=("t", "max"),
                 spearman=("spearman", "mean"), top24=("top24_hit", "mean")).reset_index())
    summ.round(3).to_csv(OUT / "horizon_validation.csv", index=False)
    for p in ["all", "veterans"]:
        s = summ[summ.panel == p]
        tab = s.pivot(index="market", columns="h", values="spearman").round(2).astype(str) + \
            " (" + s.pivot(index="market", columns="h", values="cohorts").astype("Int64").astype(str) + ")"
        print(f"\n[{p}] mean Spearman per cohort (number of cohorts) — own cohorts, not comparable across rows")
        print(tab.replace("nan (<NA>)", "").to_string())

    h2h = pd.concat([head_to_head(r, "dd_ktc"), head_to_head(r, "model"), head_to_head(r, "mfl_dynasty")])
    h2h.round(3).to_csv(OUT / "horizon_head_to_head.csv", index=False)
    for p, ref in [("all", "dd_ktc"), ("all", "model"), ("veterans", "model"), ("veterans", "mfl_dynasty")]:
        s = h2h[(h2h.panel == p) & (h2h.vs == ref)]
        if s.empty:
            continue
        tab = s.assign(cell=lambda d: d["diff"].map("{:+.3f}".format) + " (" + d.wins.astype(str) + "/"
                       + d.cohorts.astype(str) + ")").pivot(index="market", columns="h", values="cell")
        print(f"\n[{p}] Spearman minus {ref} on shared cohorts (cohorts market wins / shared)")
        print(tab.fillna("").to_string())


if __name__ == "__main__":
    main()
