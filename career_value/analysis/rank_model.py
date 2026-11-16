"""Would a regression order the Rankings tab better than the calls' factors? Out-of-cohort test.

Target: what a player returned over h seasons (call_strength.horizons: VORP + the value of the rank held after), FP rows ranked
1-300, 2018-24. Each cohort is predicted from the others:
  fp        fair(rank): the average return at that FP rank, nothing else (the FantasyPros order)
  factors   fair(rank x factor for his call): the site's method, with the site's factors (app.js LEAN) and with factors refit on the
            training cohorts (rank_factors.factor, over Hold's)
  linear    least squares on fair(rank) and its products with position, age and age squared by position, rookie, producing,
            NFL round 4+ : a price-relative model, so it re-weights the FP rank and can't ignore it
  ridge     the same, shrunk (alpha by leave-one-cohort-out inside the training cohorts is overkill here: fixed 10 on standardised x)
  boost     gradient-boosted trees on the raw features
Scores, mean over held-out cohorts: Spearman of predicted against realised, and the share of the variance in realised return left
unexplained by fp that the method explains (skill, 0 = no better than the FP order). 1 season is what Contending is judged on, 3
Default and Rebuilding; factors uses that team type's calls.

  LEAGUE=sf uv run --with pandas --with pyarrow --with scikit-learn --with scipy --with statsmodels python analysis/rank_model.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import call_strength as cs  # noqa: E402
import rank_factors as rf  # noqa: E402

F, POOL = cs.F, cs.POOL
SITE = {"con": {"buy": 0.9, "use": 1.05, "sell": 1.1, "cut": 1.1}, "mid": {"buy": 0.7, "use": 1.3, "sell": 1.15, "cut": 1.15},
        "reb": {"buy": 0.85, "use": 1.2, "sell": 1.2, "cut": 1.2}}
CELLS = [("Contending", "con", 1), ("Default", "mid", 3), ("Rebuilding", "reb", 3)]
POS = ["QB", "RB", "WR", "TE"]


def frame(x):
    """the site's feature rows beside each FP row"""
    f = pd.DataFrame([(i, row.age, row.rd, row.dy != 0, row.prod_rank) for i, row, _ in cs.features(x)],
                     columns=["i", "age_f", "rd", "rookie", "prod_rank"]).set_index("i")
    x = x.join(f)
    start = {"QB": 24 if F == "sf" else 12, "RB": 24, "WR": 30, "TE": 12}
    x["producing"] = (x.prod_rank <= x.position.map(start)).fillna(False)
    x["late"] = x.rd >= 4
    return x


def design(x, base):
    a = (x.age_f.fillna(x.age_f.median()) - 26) / 4
    d = {"base": base}
    for p in POS:
        m = (x.position == p).to_numpy()
        d[f"{p}"] = base * m
        d[f"{p}_age"] = base * m * a
        d[f"{p}_age2"] = base * m * a ** 2
        d[f"{p}_prod"] = base * m * x.producing
        d[f"{p}_rook"] = base * m * x.rookie
    d["late"] = base * x.late
    return pd.DataFrame(d, index=x.index).drop(columns=["QB"]).astype(float)


def raw(x):
    d = pd.DataFrame({"lr": np.log(x.mkt_rank), "age": x.age_f, "rd": x.rd, "rookie": x.rookie.astype(float), "prod": x.prod_rank,
                      "producing": x.producing.astype(float)}, index=x.index)
    for p in POS:
        d[p] = (x.position == p).astype(float)
    return d


def main():
    H = {h: frame(rf.live_calls(x)) for h, x in cs.horizons(hs=(1, 3)).items()}
    rows, coefs = [], []
    for name, m, h in CELLS:
        x = H[h][H[h].mkt_rank <= 300]
        sc = {}
        for t in sorted(x.t.unique()):
            tr, te = x[x.t != t], x[x.t == t]
            fair = cs.iso(tr.mkt_rank, tr.hold)
            fp = lambda g, k=1.0: fair.predict(np.log(np.clip(g.mkt_rank * k, 1, POOL * 1.5)))  # noqa: E731
            pred = {"fp": fp(te), "factors (site)": fp(te, te[m].map(SITE[m]).fillna(1.0).to_numpy())}
            hold_k = rf.factor(tr[tr[m] == "hold"], fair)
            fit = {kd: rf.factor(tr[tr[m] == kd], fair) / hold_k for kd in ("buy", "use", "sell") if (tr[m] == kd).sum() >= 10}
            pred["factors (refit, full)"] = fp(te, te[m].map(fit).fillna(1.0).to_numpy())
            Xtr, Xte = design(tr, fp(tr)), design(te, fp(te))
            pred["linear"] = LinearRegression().fit(Xtr, tr.hold).predict(Xte)
            pred["ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, tr.hold).predict(Xte)
            pred["boost"] = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=30,
                                                          random_state=0).fit(raw(tr), tr.hold).predict(raw(te))
            sse0 = ((te.hold - pred["fp"]) ** 2).sum()
            for k, p in pred.items():
                top = te.assign(p=p).nlargest(100, "p").hold.mean()
                sc.setdefault(k, []).append((pd.Series(p, index=te.index).corr(te.hold, method="spearman"), 1 - ((te.hold - p) ** 2).sum() / sse0, top))
        for k, v in sc.items():
            v = np.array(v)
            rows.append({"team": name, "seasons": h, "method": k, "spearman": round(v[:, 0].mean(), 3), "skill over fp": round(v[:, 1].mean(), 3),
                         "cohorts better than fp": f"{(v[:, 1] > 0).sum()}/{len(v)}", "return of its top 100": round(v[:, 2].mean(), 1)})
        fair = cs.iso(x.mkt_rank, x.hold)
        base = fair.predict(np.log(x.mkt_rank))
        X = design(x, base)
        c = pd.Series(make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, x.hold)[-1].coef_ / X.std().to_numpy(), index=X.columns)
        coefs.append(c.rename(f"{name} ({h})"))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"== {F}: each cohort predicted from the others"); print(out.to_string(index=False))
    print("\nridge coefficients on all cohorts (multipliers on the fair return at his rank; age in units of 4 years from 26)")
    print(pd.concat(coefs, axis=1).round(2).to_string())
    out.to_csv(HERE.parent / "outputs" / f"rank_model_{F}.csv", index=False)


if __name__ == "__main__":
    main()
