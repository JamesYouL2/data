# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "scikit-learn", "scipy", "statsmodels"]
# ///
"""ATC-style test: does a weighted blend of sources beat the best single source?

Part A (all players, 2020-2024): FP dynasty, KTC, FantasyCalc, DP, real-time stats model.
  - equal-weight consensus of available markets' log-ranks (+/- the model)
  - learned non-negative weights on log-rank, leave-one-cohort-out (few cohorts -> noisy)
Part B (veterans, MFL keeper ADP 2010-2024): bias layer. Multiply MFL's rank-implied value by
  position x age and short-season factors learned only from cohorts whose h-year outcome was
  complete before the snapshot (t <= T-h). Real-time, so any gain is out of sample.

Score: per-cohort Spearman vs h-year VORP on the validate_horizons universe (union of top 150s).
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

import market_residuals as mr
from model import load
from validate_horizons import UNIV_TOP, model_scores

OUT = mr.OUT
A_SOURCES = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]


def universe_ranks(markets, pan, outc, t, h, names):
    y = f"vorp{h}"
    avail = {k: markets[k][markets[k].t == t].merge(pan, on=["gsis_id", "t"]) for k in names}
    avail = {k: v for k, v in avail.items() if len(v) >= 100}
    univ = set()
    for k, v in avail.items():
        if k != "model":
            univ |= set(v.nsmallest(UNIV_TOP, "mkt").gsis_id)
    u = pd.DataFrame({"gsis_id": sorted(univ), "t": t}).merge(outc[["gsis_id", "t", y]], on=["gsis_id", "t"])
    for k, v in avail.items():
        s = u.merge(v[["gsis_id", "mkt"]], on="gsis_id", how="left").mkt
        u[k] = np.log(s.rank(method="average").fillna(len(u) + 1))  # missing -> worst
    return u, [k for k in names if k in avail]


def rho(u, col, y):
    return spearmanr(-u[col], u[y]).statistic


def part_a(markets, pan, outc, hs):
    rows = []
    for h in hs:
        y = f"vorp{h}"
        cohorts = {}
        for t in range(2020, mr.LAST - h + 1):
            u, av = universe_ranks(markets, pan, outc, t, h, A_SOURCES + ["model"])
            if {"dd_ktc", "fp_dynasty", "model"} <= set(av):
                cohorts[t] = (u, av)
        for t, (u, av) in cohorts.items():
            mk = [a for a in av if a != "model"]
            u["consensus"] = u[mk].mean(axis=1)
            u["consensus+model"] = u[mk + ["model"]].mean(axis=1)
            # LOCO weights over fp, ktc, model (available in every cohort 2020+)
            fit_cols = ["fp_dynasty", "dd_ktc", "model"]
            tr = pd.concat([c[0] for s, c in cohorts.items() if s != t])
            w = None
            if len(tr):
                A = np.column_stack([-tr[c] for c in fit_cols] + [np.ones(len(tr))])
                w, _ = nnls(A, np.log1p(tr[y].to_numpy()))
                u["weighted(fp,ktc,model)"] = (u[fit_cols] * w[:3]).sum(axis=1)
            for c in av + ["consensus", "consensus+model"] + (["weighted(fp,ktc,model)"] if w is not None else []):
                rows.append({"h": h, "t": t, "source": c, "spearman": rho(u, c, y),
                             "weights": None if c != "weighted(fp,ktc,model)" else np.round(w[:3] / w[:3].sum(), 2)})
    return pd.DataFrame(rows)


def part_b(markets, panel, outc):
    """MFL vets: real-time pos x age + short-season bias layer on MFL's rank-implied value."""
    m = markets["mfl_dynasty"].merge(panel[panel.row_type == "veteran"], on=["gsis_id", "t"])
    m = m.merge(outc, on=["gsis_id", "t"], how="left")
    m = m[m.groupby("t").mkt.rank(method="first") <= mr.TOP_N].copy()
    m["rank"] = m.groupby("t").mkt.rank(method="first")
    age = pd.cut(m.age, [0, 24.5, 27.5, 99], labels=["y", "p", "o"]).astype(str)
    m["cell"] = m.position + age + np.where(m.games <= 9, "_short", "")
    rows = []
    for h in [1, 2, 3, 5]:
        y = f"vorp{h}"
        for T in sorted(m.t.unique()):
            if T + h > mr.LAST:
                continue
            tr = m[(m.t <= T - h) & m[y].notna()]
            if tr.t.nunique() < 3:
                continue
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(tr["rank"], tr[y])
            tr = tr.assign(e=iso.predict(tr["rank"]))
            k = 20  # shrink each cell's actual/expected ratio toward 1 with k pseudo-players
            g = tr.groupby("cell").agg(a=(y, "sum"), e=("e", "sum"), n=(y, "size"))
            mean_e = tr.e.mean()
            g["f"] = (g.a + k * mean_e) / (g.e + k * mean_e)
            te = m[m.t == T].copy()
            te["raw"] = iso.predict(te["rank"])
            te["adj"] = te.raw * te.cell.map(g.f).fillna(1.0)
            for c in ["raw", "adj"]:
                rows.append({"h": h, "t": T, "source": "mfl" if c == "raw" else "mfl+bias layer",
                             "spearman": spearmanr(te[c], te[y]).statistic})
    return pd.DataFrame(rows)


def main():
    panel = load()
    outc = mr.outcomes()
    markets = {k: v[["gsis_id", "t", "mkt"]].dropna() for k, v in mr.all_markets(panel).items()}
    pan = panel[["gsis_id", "t"]]
    print("training real-time model ...", flush=True)
    mod = model_scores(panel, outc, {h: list(range(2020, mr.LAST - h + 1)) for h in [1, 2, 3]})

    ra = []
    for h in [1, 2, 3]:
        markets["model"] = mod[mod.h == h][["gsis_id", "t", "mkt"]]
        ra.append(part_a(markets, pan, outc, [h]))
    a = pd.concat(ra, ignore_index=True)
    a.to_csv(OUT / "blend_test_markets.csv", index=False)
    print("\nPart A: mean Spearman on cohorts with both FP and KTC (cohorts in brackets)")
    s = a.groupby(["source", "h"]).agg(r=("spearman", "mean"), n=("t", "nunique")).reset_index()
    s["cell"] = s.r.round(3).astype(str) + " (" + s.n.astype(str) + ")"
    print(s.pivot(index="source", columns="h", values="cell").to_string())
    print("weights (fp, ktc, model) per held-out cohort:")
    print(a[a.weights.notna()][["h", "t", "weights"]].to_string(index=False))

    b = part_b(markets, panel, outc)
    b.to_csv(OUT / "blend_test_bias_layer.csv", index=False)
    piv = b.pivot_table(index=["h", "t"], columns="source", values="spearman")
    d = (piv["mfl+bias layer"] - piv["mfl"]).groupby(level=0)
    print("\nPart B: MFL vets, real-time bias layer minus raw MFL (Spearman)")
    print(pd.DataFrame({"mean_gain": d.mean().round(3), "cohorts_improved": d.apply(lambda s: (s > 0).sum()),
                        "cohorts": d.size()}).to_string())


if __name__ == "__main__":
    main()
