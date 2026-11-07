"""Q: what holds up over 5 and 10 years, and can MFL keeper ADP be translated into a dynasty-market price?

Only MFL keeper ADP reaches back far enough for long outcomes: its API has no keeper ADP before 2011 (cohort
t = 2010), so 10-year outcomes exist for cohorts 2010-15 and 5-year for 2010-20. The dynasty trade-value markets
start in 2018 (FP, via files/archives), 2020 (KTC), 2021 (FC) and 2022 (DP).

Translation ("synthetic dynasty price"): on the overlap cohorts 2018-24, for every MFL top-200 row, the target is
the dynasty consensus price = mean over the dynasty markets live that year of log(overall market rank), with a
player outside a market's top 200 set to rank 250. A gradient-boosted model learns the target from log MFL rank
plus what the dynasty markets react to differently (age, position, experience, last season, draft capital).
  validation : train on 2018-21, test on 2022-24 -- rank correlation with the real dynasty price, and whether
               the synthetic price reproduces the real dynasty residuals of the playbook rules
  application: model trained on 2018-24, applied to MFL cohorts 2010-17; residuals of h-year VORP against an
               isotonic curve on the synthetic rank, next to residuals against raw MFL ADP.
Assumption: dynasty markets' pricing relative to ADP was the same in 2010-17 as in 2018-24.
Direct check: FP dynasty itself now has three 5-year cohorts (2018-20). Read the synthetic
results as "how today's dynasty pricing style would have done", not as what traders paid then.
LEAGUE=sf: superflex VORP and superflex dynasty prices (outputs/sf).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import DYN, out_dir, rows
from out_of_sample import vet_rules

SF = os.environ.get("LEAGUE", "1qb").lower() == "sf"  # LEAGUE=sf: superflex outcomes and prices


def rules(x):
    """playbook rules + position-by-age bands"""
    pos, age, vet = x.position, x.age, x.row_type == "veteran"
    r = vet_rules(x)
    r.update({"QB <=24": vet & (pos == "QB") & (age < 25), "QB 25-27": (pos == "QB") & age.between(25, 27.99),
              "QB 28-31": (pos == "QB") & age.between(28, 31.99), "QB 32+": (pos == "QB") & (age >= 32),
              "WR <=24": vet & (pos == "WR") & (age < 25), "WR 25-27": (pos == "WR") & age.between(25, 27.99),
              "RB <=24": vet & (pos == "RB") & (age < 25), "RB 28+": (pos == "RB") & (age >= 28),
              "TE 25-28": (pos == "TE") & age.between(25, 28.99)})
    return r

FEATS = ["lmfl", "mkt_pos_rank", "pos", "age", "exp", "rookie", "games", "ppg", "ppg_pos_rank", "vorp", "vorp_prev",
         "career_best_ppg", "draft_round_f", "draft_pick_f", "targets_pg", "carries_pg", "att_pg", "changed_team"]


def prep(x):
    x = x.copy()
    x["lmfl"] = np.log(x.mkt_rank)
    x["pos"] = x.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
    x["rookie"] = (x.row_type == "rookie").astype(int)
    x["changed_team"] = x.changed_team.astype(float)
    return x


def dyn_target():
    """mean log overall rank across dynasty markets live in cohort t (outside top 200 -> 250)."""
    d = rows(DYN, 1, SF)[["mk", "gsis_id", "t", "mkt_rank"]]
    live = d.groupby("mk").t.unique()
    return d, live


def target_for(x, d, live):
    out = []
    for m in DYN:
        s = x[x.t.isin(live[m])][["gsis_id", "t"]].merge(d[d.mk == m][["gsis_id", "t", "mkt_rank"]],
                                                         on=["gsis_id", "t"], how="left")
        s["lr"] = np.log(s.mkt_rank.fillna(250))
        out.append(s[["gsis_id", "t", "lr"]])
    return pd.concat(out).groupby(["gsis_id", "t"]).lr.mean().rename("y").reset_index()


def model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


def resid_pct(x, rank, y):
    out = pd.Series(index=x.index, dtype=float)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x[rank], x[y])
    out[:] = (x[y] - iso.predict(x[rank])) / x[y].mean() * 100
    return out


def rule_table(x, h, cols):
    rl = rules(x)
    res = {}
    for k, v in rl.items():
        v = v.fillna(False).to_numpy()
        if v.sum() < 8:
            continue
        g = x[v]
        res[k] = {c: f"{g[c].mean():+.0f} ({(g.groupby('t')[c].mean() > 0).sum()}/{g.t.nunique()})" for c in cols}
        res[k]["n"] = len(g)
    return pd.DataFrame(res).T


def main():
    d, live = dyn_target()
    m1 = prep(rows(["mfl_dynasty"], 1, SF))
    ov = m1[m1.t >= 2018].merge(target_for(m1[m1.t >= 2018], d, live), on=["gsis_id", "t"])
    print(f"overlap rows (MFL top-200 x cohorts 2018-24): {len(ov)}; in at least one dynasty top-200: "
          f"{(ov.y < np.log(250) - 1e-9).mean():.0%}")

    # ---- validation: train 2018-21, test 2022-24
    tr, te = ov[ov.t <= 2021], ov[ov.t >= 2022].copy()
    f = model().fit(tr[FEATS], tr.y)
    te["pred"] = f.predict(te[FEATS])
    lin = np.polyfit(tr.lmfl, tr.y, 1)
    te["pred_lin"] = np.polyval(lin, te.lmfl)
    print("\n== Validation (train 2018-21, test 2022-24): Spearman with real dynasty consensus rank, per cohort")
    for t, g in te.groupby("t"):
        print(f"  {t}: raw MFL rank {spearmanr(g.lmfl, g.y).statistic:.3f}  ->  translated {spearmanr(g.pred, g.y).statistic:.3f}")
    # do synthetic residuals reproduce the dynasty residual pattern in the test years?
    print("\n== Validation: playbook rules in test cohorts 2022-24, residual % vs three prices "
          "(real dynasty consensus / translated MFL / raw MFL)")
    for h in (1, 2):
        mh = prep(rows(["mfl_dynasty"], h, SF))
        z = mh[mh.t >= 2022].merge(te[["gsis_id", "t", "y", "pred"]], on=["gsis_id", "t"])
        if z.empty:
            continue
        y = f"vorp{h}"
        z["real dynasty"] = resid_pct(z, "y", y)
        z["translated"] = resid_pct(z, "pred", y)
        z["raw MFL"] = resid_pct(z, "mkt_rank", y)
        print(f"h{h} (cohorts {sorted(z.t.unique())})")
        print(rule_table(z, h, ["real dynasty", "translated", "raw MFL"]).to_string())

    # ---- application: train on all overlap, translate 2010-18
    f = model().fit(ov[FEATS], ov.y)
    imp = {}
    base = spearmanr(f.predict(ov[FEATS]), ov.y).statistic
    rng = np.random.default_rng(0)
    for c in FEATS:
        z = ov[FEATS].copy()
        z[c] = rng.permutation(z[c].to_numpy())
        imp[c] = base - spearmanr(f.predict(z), ov.y).statistic
    print("\nwhat the translation uses (drop in in-sample Spearman when shuffled):",
          {k: round(v, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8]})

    for h, cohorts in ((1, range(2010, 2018)), (3, range(2010, 2018)), (5, range(2010, 2018)), (10, range(2010, 2016))):
        mh = prep(rows(["mfl_dynasty"], h, SF))
        z = mh[mh.t.isin(cohorts)].copy()
        z["pred"] = f.predict(z[FEATS])
        y = f"vorp{h}"
        z["vs translated (dynasty-style)"] = resid_pct(z, "pred", y)
        z["vs raw MFL ADP"] = resid_pct(z, "mkt_rank", y)
        print(f"\n== h={h}: cohorts {min(cohorts)}-{max(cohorts)} (pre-overlap), residual % (cohorts positive/total)")
        print(rule_table(z, h, ["vs translated (dynasty-style)", "vs raw MFL ADP"]).to_string())
        if h == 10:
            # how the translation reprices groups: mean synthetic minus raw log-rank (positive = dynasty prices lower)
            z["dshift"] = z.pred - np.log(z.mkt_rank)
            print("price shift, translated vs raw (mean log-rank change; + = dynasty-style ranks him lower):",
                  {k: round(z[v.fillna(False).to_numpy()].dshift.mean(), 2) for k, v in vet_rules(z).items()})

    # ---- direct: FP dynasty's own 5-year residuals (cohorts 2018-20), next to MFL on the same cohorts
    fp5, mf5 = rows(["fp_dynasty"], 5, SF), prep(rows(["mfl_dynasty"], 5, SF))
    mf5 = mf5[mf5.t.isin(fp5.t.unique())]
    a = rule_table(fp5.assign(**{"FP dynasty": fp5.r}), 5, ["FP dynasty"])
    b = rule_table(mf5.assign(**{"MFL same cohorts": mf5.r}), 5, ["MFL same cohorts"])
    print(f"\n== Direct 5-year check: FP dynasty residual % (cohorts {sorted(fp5.t.unique())}), MFL on the same cohorts")
    print(a.join(b.drop(columns="n"), how="left").to_string())

    # ---- rookies at 10 years (MFL rookie order; no translation)
    r = pd.read_csv(out_dir(SF) / "rookie_position_residuals.csv")
    r = r[(r.market == "mfl") & (r.h == 10)]
    print(f"\n== MFL rookie order, 10-year residual % (classes {r.t.min() + 1}-{r.t.max() + 1})")
    g = {"WR NFL R1": (r.position == "WR") & (r.nfl == "NFL R1"), "WR NFL R2": (r.position == "WR") & (r.nfl == "NFL R2"),
         "WR NFL R3+": (r.position == "WR") & (r.nfl == "NFL R3+"), "RB NFL R1": (r.position == "RB") & (r.nfl == "NFL R1"),
         "RB NFL R2": (r.position == "RB") & (r.nfl == "NFL R2"), "RB NFL R3+": (r.position == "RB") & (r.nfl == "NFL R3+"),
         "TE all": r.position == "TE", "TE rookie R1-R2 slot": (r.position == "TE") & (r.slot <= 24),
         "QB all": r.position == "QB", "QB rookie R3-R4 slot": (r.position == "QB") & (r.slot > 24),
         "age <=21": r.age < 21.5, "age 23+": r.age >= 22.5}
    print(pd.DataFrame({k: {"mean": f"{r[v].res.mean():+.0f}", "median": f"{r[v].res.median():+.0f}",
                            "classes +": f"{(r[v].groupby('t').res.mean() > 0).sum()}/{r[v].t.nunique()}", "n": v.sum()}
                        for k, v in g.items()}).T.to_string())


if __name__ == "__main__":
    main()
