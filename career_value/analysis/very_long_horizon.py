"""Q: what do 10, 15 and 20 year outcomes say about young players, young QBs above all?

Outcome: VORP summed over seasons t+1..t+h, undiscounted unless a column says otherwise, so a retired player
counts as zero. Season stats start in 1999, so with 2025 the last season:
  h = 20: cohorts 1999-2005    h = 15: cohorts 1999-2010    h = 10: cohorts 1999-2015

Real prices barely reach that far. MFL keeper ADP starts with cohort 2010 and has no rookies, the dynasty markets
start in 2018. So 10-year outcomes have a real price for 2010-15, 15-year for 2010 alone, 20-year for no cohort.

Modelled price: a gradient-boosted model learns the dynasty consensus (common.DYN, mean log rank, outside a top 200
set to 250) on cohorts 2018-24 from age, position, experience, draft capital and last season, with no market input.
Applied to earlier cohorts, it says what today's dynasty pricing style would have paid for that profile. It never
sees an outcome. The real markets know more than it does, and football changed, so the checks come first:
  1. train 2018-21, test 2022-24 against the real consensus;
  2. train 2018-24, test 2010-17 against real MFL rank;
  3. on the rows with a real long-run price, do modelled and real prices tell the same story?

Residual %: (actual - isotonic curve on rank) / mean outcome x 100, as in long_horizon.py, among each cohort's
modelled top 200. Each row of a table is fit on one cohort set, so compare horizons within a set. Brackets are a
95% bootstrap interval over players, since one player fills several overlapping cohorts.
LEAGUE=sf: superflex VORP and superflex dynasty prices (outputs/sf).
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import HERE, out_dir, rows  # noqa: E402
from long_horizon import dyn_target, model, resid_pct, target_for  # noqa: E402

SF = os.environ.get("LEAGUE", "1qb").lower() == "sf"
VALUE = "vorp_sf" if SF else "vorp"
LAST = 2025
TOP_N = 200
FEATS = ["pos", "age", "exp", "rookie", "games", "ppg", "ppg_pos_rank", "val", "vorp_prev", "career_best_ppg",
         "draft_round_f", "draft_pick_f", "drafted", "targets_pg", "carries_pg", "att_pg", "changed_team"]
BANDS = [("yrs 1-3", 1, 3), ("yrs 4-5", 4, 5), ("yrs 6-10", 6, 10), ("yrs 11-15", 11, 15), ("yrs 16-20", 16, 20)]


def load():
    p = pd.read_parquet(HERE / "data" / "panel.parquet")
    p = p[p.t.notna()].copy()
    p["t"] = p.t.astype(int)
    p["pos"] = p.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
    p["rookie"] = (p.row_type == "rookie").astype(int)
    p["changed_team"] = p.changed_team.astype(float)
    p["drafted"] = p.drafted.astype(float)
    p["val"] = p[VALUE]
    p["ppg_pos_rank"] = p.groupby(["t", "position"]).ppg.rank(ascending=False)
    p = p.reset_index(drop=True)
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet", columns=["gsis_id", "season", VALUE])
    v = s.pivot_table(index="gsis_id", columns="season", values=VALUE, fill_value=0.0)
    return p, v


def outcome(p, v, h, disc=1.0, first=1):
    """VORP over seasons t+first..t+h, each discounted from t+1; NaN when season t+h hasn't been played."""
    out = np.full(len(p), np.nan)
    for t, g in p.groupby("t"):
        if t + h > LAST:
            continue
        years = list(range(t + first, t + h + 1))
        w = np.array([disc ** (y - t - 1) for y in years])
        a = (v.reindex(columns=years).fillna(0.0) * w).sum(axis=1)
        out[g.index] = g.gsis_id.map(a).fillna(0.0).to_numpy()
    return out


def groups(x):
    pos, age, vet, rook = x.position, x.age, x.row_type == "veteran", x.row_type == "rookie"
    young = age < 25
    starter = (x.games >= 8) & (x.att_pg >= 20)
    return {
        "QB <=24": (pos == "QB") & young,
        "  rookie, round 1": (pos == "QB") & young & rook & (x.draft_round_f == 1),
        "  rookie, later": (pos == "QB") & young & rook & (x.draft_round_f != 1),
        "  vet, started last yr": (pos == "QB") & young & vet & starter,
        "  vet, didn't start": (pos == "QB") & young & vet & ~starter,
        "QB 25-27": (pos == "QB") & age.between(25, 27.99),
        "QB 28-31": (pos == "QB") & age.between(28, 31.99),
        "QB 32+": (pos == "QB") & (age >= 32),
        "WR <=24": (pos == "WR") & young,
        "WR 25-27": (pos == "WR") & age.between(25, 27.99),
        "WR 28+": (pos == "WR") & (age >= 28),
        "RB <=24": (pos == "RB") & young,
        "RB 25-27": (pos == "RB") & age.between(25, 27.99),
        "RB 28+": (pos == "RB") & (age >= 28),
        "TE <=24": (pos == "TE") & young,
        "TE 25-28": (pos == "TE") & age.between(25, 28.99),
    }


def boot(g, col, n=1000, seed=0):
    """95% interval of the row mean, resampling players."""
    by = g.groupby("gsis_id")[col].agg(["sum", "size"]).to_numpy()
    if len(by) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(by), (n, len(by)))
    m = by[pick, 0].sum(axis=1) / by[pick, 1].sum(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def stat(g, col):
    lo, hi = boot(g, col)
    coh = g.groupby("t")[col].mean()
    return {"mean": g[col].mean(), "median": g[col].median(), "lo": lo, "hi": hi, "cohorts_pos": int((coh > 0).sum()),
            "cohorts": len(coh), "players": g.gsis_id.nunique(), "rows": len(g)}


def cellfmt(s):
    if s["rows"] < 5:
        return ""
    return f"{s['mean']:+.0f} [{s['lo']:+.0f},{s['hi']:+.0f}] c{s['cohorts_pos']}/{s['cohorts']} p{s['players']}"


def priced(p, f):
    x = p.copy()
    x["pred"] = f.predict(x[FEATS])
    x["mrank"] = x.groupby("t").pred.rank(method="first")
    return x


def validate(p, target):
    print("== Check 1: train 2018-21, test 2022-24, Spearman with the real dynasty consensus (rows in a real top 200)")
    ov = p.merge(target, on=["gsis_id", "t"])
    tr, te = ov[ov.t <= 2021], ov[ov.t >= 2022].copy()
    te["pred"] = model().fit(tr[FEATS], tr.y).predict(te[FEATS])
    te = te[te.y < np.log(250) - 1e-9]
    masks = {"all": te.age > 0, "rookies": te.rookie == 1, "QB": te.position == "QB",
             "QB <=24": (te.position == "QB") & (te.age < 25), "WR <=24": (te.position == "WR") & (te.age < 25)}
    for k, m in masks.items():
        g = te[m]
        print(f"  {k:10s} n={len(g):4d}  Spearman {spearmanr(g.pred, g.y).statistic:.3f}  "
              f"mean modelled - real log rank {(g.pred - g.y).mean():+.2f}")
    return ov


def check_real(x, v):
    print("\n== Check 2: model trained 2018-24, Spearman with real MFL rank, by cohort")
    m = rows(["mfl_dynasty"], 1, SF)
    z = x.merge(m[["gsis_id", "t", "mkt_rank"]], on=["gsis_id", "t"])
    print("  " + "  ".join(f"{t}:{spearmanr(g.pred, g.mkt_rank).statistic:.2f}" for t, g in z.groupby("t")))
    old = z[z.t <= 2017]
    print(f"  cohorts 2010-17, QB <=24 only: {spearmanr(*old.loc[(old.position == 'QB') & (old.age < 25), ['pred', 'mkt_rank']].T.to_numpy()).statistic:.2f}")

    print("\n== Check 3: same rows, real price vs modelled price, residual %  [mean, bootstrap 95%, cohorts+, players]")
    out = {}
    for label, mk, h, lo, hi in [("MFL h10 2010-15", "mfl_dynasty", 10, 2010, 2015), ("FP h5 2018-20", "fp_dynasty", 5, 2018, 2020)]:
        z = rows([mk], h, SF)[["gsis_id", "t", "mkt_rank"]]
        z = x[x.t.between(lo, hi)].merge(z, on=["gsis_id", "t"])
        z["y"] = outcome(z, v, h)
        z["real"], z["modelled"] = resid_pct(z, "mkt_rank", "y"), resid_pct(z, "pred", "y")
        for k, mask in groups(z).items():
            g = z[mask.fillna(False)]
            if len(g) >= 5:
                out.setdefault(k, {})[f"{label} real"] = cellfmt(stat(g, "real"))
                out.setdefault(k, {})[f"{label} modelled"] = cellfmt(stat(g, "modelled"))
    print(pd.DataFrame(out).T.fillna("").to_string())

    print("\n== Cohort 2010 alone at a real 15-year price (MFL, no rookies): QBs and young players")
    z = x[x.t == 2010].merge(rows(["mfl_dynasty"], 1, SF)[["gsis_id", "t", "mkt_rank"]], on=["gsis_id", "t"])
    z["y"] = outcome(z, v, 15)
    z["real"] = resid_pct(z, "mkt_rank", "y")
    print({k: f"{z.loc[m.fillna(False), 'real'].mean():+.0f} (n{int(m.fillna(False).sum())})" for k, m in groups(z).items() if m.fillna(False).sum() >= 3})
    print(z[(z.position == "QB") & (z.age < 28)].sort_values("mkt_rank")[["display_name", "age", "mkt_rank", "y", "real"]].round(0).to_string(index=False))


def horizon_table(x, v, lo, hi, specs, title, store):
    """Residual % for each (label, h, discount) among the top 200 of cohorts lo..hi."""
    z = x[x.t.between(lo, hi) & (x.mrank <= TOP_N)].reset_index(drop=True)
    res, raw = {}, {}
    masks = {k: m.fillna(False) for k, m in groups(z).items()}
    for label, h, disc in specs:
        z[label] = outcome(z, v, h, disc)
        z["_r"] = resid_pct(z, "mrank", label)
        for k, m in masks.items():
            g = z[m]
            s = stat(g, "_r")
            res.setdefault(k, {})[label] = cellfmt(s)
            raw.setdefault(k, {})[label] = g[label].mean()
            store.append({"league": "sf" if SF else "1qb", "cohorts": f"{lo}-{hi}", "group": k.strip(), "outcome": label,
                          "raw_mean": g[label].mean(), **s})
        z[f"_r_{label}"] = z["_r"]
    print(f"\n== {title}: residual % vs modelled price [mean, bootstrap 95%, cohorts+, players]")
    print(pd.DataFrame(res).T.to_string())
    print(f"-- same cohorts, mean raw VORP per row, undiscounted unless labelled (top 200 mean: "
          + ", ".join(f"{lab} {z[lab].mean():.0f}" for lab, _, _ in specs) + ")")
    print(pd.DataFrame(raw).T.round(0).to_string())
    return z


def concentration(z, label):
    col = f"_r_{label}"
    g = z[(z.position == "QB") & (z.age < 25)]
    by = g.sort_values("t").groupby("gsis_id").agg(name=("display_name", "first"), first_t=("t", "first"), age=("age", "first"),
                                                   rows=("t", "size"), raw=(label, "mean"), r=(col, "mean"), rsum=(col, "sum"))
    by = by.sort_values("rsum", ascending=False)
    print(f"\n== QB <=24, who drives it, {label} ({len(by)} players, {len(g)} rows)")
    print(by.head(8).round(0).to_string())
    print("... least")
    print(by.tail(4).round(0).to_string())
    for drop in (0, 1, 3, 5):
        keep = g[~g.gsis_id.isin(by.index[:drop])]
        print(f"  mean residual without top {drop}: {keep[col].mean():+.0f}, median {keep[col].median():+.0f}, "
              f"rows beating the curve {(keep[col] > 0).mean():.0%}")
    print(f"  players with zero VORP over {label}: {(by.raw == 0).mean():.0%}")


def shape(p, v):
    z = p[p.t.between(1999, 2005)].reset_index(drop=True)
    masks = {k: m.fillna(False) for k, m in groups(z).items()}
    out = {}
    for lab, a, b in BANDS:
        z[lab] = outcome(z, v, b, 1.0, a)
    tot = z[[b[0] for b in BANDS]].sum(axis=1)
    for k, m in masks.items():
        out[k] = {b[0]: z.loc[m, b[0]].sum() / tot[m].sum() * 100 for b in BANDS}
    print("\n== When the value arrives, cohorts 1999-2005, % of 20-year VORP by years after t (all panel rows)")
    print(pd.DataFrame(out).T.round(0).to_string())


def rookies(p, v):
    print("\n== Rookies by draft slot, no price at all: mean VORP, median, share who never returned value, players")
    bands = [("picks 1-10", 1, 10), ("picks 11-32", 11, 32), ("picks 33-64", 33, 64)]
    for h, hi in ((10, 2015), (15, 2010), (20, 2005)):
        z = p[(p.row_type == "rookie") & (p.age < 25) & (p.t <= hi)].reset_index(drop=True)
        z["y"] = outcome(z, v, h)
        res = {}
        for lab, a, b in bands:
            g0 = z[z.draft_pick_f.between(a, b)]
            for pos in ("QB", "WR", "RB", "TE"):
                g = g0[g0.position == pos]
                res.setdefault(pos, {})[lab] = (f"{g.y.mean():.0f}/{g.y.median():.0f} zero {(g.y == 0).mean():.0%} n{len(g)}"
                                                if len(g) else "")
        print(f"h{h}, draft classes through t={hi}")
        print(pd.DataFrame(res).to_string())


def main():
    p, v = load()
    print(f"League {'superflex' if SF else '1QB'}. Panel {len(p)} rows, cohorts {p.t.min()}-{p.t.max()}")
    d, live = dyn_target()
    cur = p[p.t.between(2018, 2024)]
    target = target_for(cur, d, live)
    ov = validate(p, target)
    x = priced(p, model().fit(ov[FEATS], ov.y))
    check_real(x, v)
    shape(p, v)
    rookies(p, v)

    disc = [("h3", 3, 1), ("h5", 5, 1), ("h10", 10, 1), ("h15", 15, 1), ("h20", 20, 1), ("h20 d10%", 20, .9), ("h20 d15%", 20, .85)]
    store = []
    z = horizon_table(x, v, 1999, 2005, disc, "Cohorts 1999-2005, every horizon", store)
    concentration(z, "h20")
    z = horizon_table(x, v, 1999, 2010, [("h3", 3, 1), ("h5", 5, 1), ("h10", 10, 1), ("h15", 15, 1), ("h15 d10%", 15, .9)],
                      "Cohorts 1999-2010", store)
    concentration(z, "h15")
    z = horizon_table(x, v, 2006, 2015, [("h3", 3, 1), ("h5", 5, 1), ("h10", 10, 1), ("h10 d10%", 10, .9)],
                      "Cohorts 2006-2015, a later era", store)
    concentration(z, "h10")
    horizon_table(x, v, 2016, 2020, [("h3", 3, 1), ("h5", 5, 1)], "Cohorts 2016-2020, recent (2018-20 also trained the price)", store)
    f = out_dir(SF) / "very_long_horizon.csv"
    pd.DataFrame(store).round(2).to_csv(f, index=False)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
