# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "scikit-learn", "scipy"]
# ///
"""Rest-of-career (ROC) value model and primary-variable analysis.

Target: roc_vorp = sum of future-season PPR VORP (see build_dataset.py).
Train: snapshots 2002-2010. Test: snapshots 2011-2014 (>= 11 seasons of
follow-up, so careers are essentially complete).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent
OUT = HERE / "outputs"
RNG = np.random.default_rng(0)

GROUPS = {
    "position": ["pos_code"],
    "age": ["age"],
    "experience": ["exp", "seasons_played"],
    "draft_capital": ["draft_pick_f", "draft_round_f", "drafted"],
    "size": ["height", "weight", "bmi"],
    "current_season_production": ["ppg", "vorp", "ppr", "pos_rank", "games"],
    "prior_season_production": ["ppg_prev", "vorp_prev", "games_prev"],
    "career_production": ["career_vorp", "career_best_vorp", "career_best_ppg"],
    "usage_volume": ["att_pg", "pass_yds_pg", "carries_pg", "rush_yds_pg", "targets_pg",
                     "rec_yds_pg", "target_share", "air_yards_share", "wopr"],
    "efficiency_epa": ["epa_pg"],
    "team_change": ["changed_team"],
}
FEATURES = [f for fs in GROUPS.values() for f in fs]
POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def make_model(features, max_iter=300):
    cat = [f == "pos_code" for f in features]
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=cat,
        random_state=0,
    )


def score(y, p):
    return {"r2": r2_score(y, p), "spearman": spearmanr(y, p).statistic,
            "top24_hit": top_k_hit(y, p)}


def top_k_hit(y, p, k=24):
    """Share of the true top-k ROC players that the model also ranks top-k (per test year avg)."""
    y, p = np.asarray(y), np.asarray(p)
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k


def load():
    d = pd.read_parquet(HERE / "data" / "panel.parquet")
    d["pos_code"] = d.position.map(POS_CODE)
    return d


def cv_r2(df, features, target="roc_vorp", n_splits=5, max_iter=200):
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(df))
    for tr, va in gkf.split(df, groups=df.gsis_id):
        m = make_model(features, max_iter).fit(df.iloc[tr][features], df.iloc[tr][target])
        pred[va] = m.predict(df.iloc[va][features])
    return r2_score(df[target], pred)


def group_permutation(model, X, y, base, n_rep=10):
    rows = []
    for g, cols in GROUPS.items():
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            idx = RNG.permutation(len(X))
            Xp[cols] = X[cols].values[idx]  # permute the group jointly
            drops.append(base - r2_score(y, model.predict(Xp)))
        rows.append({"group": g, "perm_r2_drop": np.mean(drops), "perm_sd": np.std(drops)})
    return pd.DataFrame(rows)


def drop_group(train, test, target, base):
    rows = []
    for g, cols in GROUPS.items():
        feats = [f for f in FEATURES if f not in cols]
        m = make_model(feats).fit(train[feats], train[target])
        rows.append({"group": g, "dropgroup_r2_loss": base - r2_score(test[target], m.predict(test[feats]))})
    return pd.DataFrame(rows)


def forward_select(train, test, target, steps=8):
    chosen, out = [], []
    remaining = list(FEATURES)
    for step in range(steps):
        trials = {f: cv_r2(train, chosen + [f], target) for f in remaining}
        best = max(trials, key=trials.get)
        chosen.append(best)
        remaining.remove(best)
        m = make_model(chosen).fit(train[chosen], train[target])
        test_r2 = r2_score(test[target], m.predict(test[chosen]))
        out.append({"step": step + 1, "added": best, "cv_r2_train": trials[best], "test_r2": test_r2})
        print(f"  step {step + 1}: +{best:<20} cv={trials[best]:.3f} test={test_r2:.3f}", flush=True)
    return pd.DataFrame(out)


def baselines(train, test, target):
    res = {}
    # position x age-bucket mean
    key = lambda d: d.position + "_" + d.age.round().clip(21, 36).astype(int).astype(str)
    means = train.groupby(key(train))[target].mean()
    res["pos_x_age_mean"] = key(test).map(means).fillna(train[target].mean())
    # last season VORP x position-specific remaining-career multiplier (naive "rank by current value")
    for name, feats in {"current_vorp_only": ["pos_code", "vorp"],
                        "age_pos_draft": ["pos_code", "age", "draft_pick_f", "exp"]}.items():
        m = make_model(feats).fit(train[feats], train[target])
        res[name] = m.predict(test[feats])
    return res


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    target = "roc_vorp"
    train = d[d.t.between(2002, 2010)].reset_index(drop=True)
    test = d[d.t.between(2011, 2014)].reset_index(drop=True)
    print(f"train rows {len(train)}  test rows {len(test)}")
    censored = d[(d.t == 2014) & (d.row_type == "veteran")]
    print("share of 2014 snapshot players still active in 2025:",
          round(d[(d.t == 2025)].gsis_id.isin(censored.gsis_id).mean() * 0 +
                censored.gsis_id.isin(d[d.t == 2025].gsis_id).mean(), 3))

    # ---- full model + baselines --------------------------------------------
    full = make_model(FEATURES).fit(train[FEATURES], train[target])
    p_full = full.predict(test[FEATURES])
    perf = {"full_model": score(test[target], p_full)}
    for k, v in baselines(train, test, target).items():
        perf[k] = score(test[target], v)
    perf = pd.DataFrame(perf).T.round(3)
    print("\nTest performance (snapshots 2011-2014):\n", perf)
    perf.to_csv(OUT / "performance.csv")

    # per-subset performance
    sub = []
    for (rt, pos), g in test.assign(p=p_full).groupby(["row_type", "position"]):
        sub.append({"row_type": rt, "position": pos, "n": len(g), **score(g[target], g.p)})
    sub = pd.DataFrame(sub).round(3)
    print(sub)
    sub.to_csv(OUT / "performance_by_subset.csv", index=False)

    # ---- importance: permutation + drop-group -----------------------------
    base = r2_score(test[target], p_full)
    imp = group_permutation(full, test[FEATURES], test[target], base).merge(
        drop_group(train, test, target, base), on="group")
    imp = imp.sort_values("perm_r2_drop", ascending=False).round(4)
    print("\nGroup importance (test R2 lost):\n", imp)
    imp.to_csv(OUT / "importance_groups.csv", index=False)

    # importance separately for veterans and rookies
    for rt in ["veteran", "rookie"]:
        te = test[test.row_type == rt]
        b = r2_score(te[target], full.predict(te[FEATURES]))
        gi = group_permutation(full, te[FEATURES], te[target], b).sort_values("perm_r2_drop", ascending=False)
        gi.round(4).to_csv(OUT / f"importance_groups_{rt}.csv", index=False)
        print(f"\n{rt} (test R2 {b:.3f}):\n", gi.head(6).round(4))

    # ---- forward selection: minimal primary-variable set -----------------
    print("\nForward selection (all rows):")
    fs = forward_select(train, test, target)
    fs.to_csv(OUT / "forward_selection.csv", index=False)
    print("\nForward selection (veterans only):")
    fsv = forward_select(train[train.row_type == "veteran"].reset_index(drop=True),
                         test[test.row_type == "veteran"].reset_index(drop=True), target, steps=6)
    fsv.to_csv(OUT / "forward_selection_veterans.csv", index=False)

    # ---- robustness: same importance for alternative targets ---------------
    rob = []
    for tgt in ["roc_vorp_disc", "roc5_vorp", "next_vorp"]:
        m = make_model(FEATURES).fit(train[FEATURES], train[tgt])
        b = r2_score(test[tgt], m.predict(test[FEATURES]))
        gi = group_permutation(m, test[FEATURES], test[tgt], b, n_rep=5)
        gi["target"], gi["test_r2"] = tgt, b
        rob.append(gi)
    rob = pd.concat(rob).pivot(index="group", columns="target", values="perm_r2_drop")
    rob = rob.sort_values("roc_vorp_disc", ascending=False).round(4)
    print("\nPermutation importance by target:\n", rob)
    rob.to_csv(OUT / "importance_by_target.csv")

    # ---- final model on all complete-career snapshots; score current players
    allc = d[d.t.between(2002, 2014)]
    final = make_model(FEATURES).fit(allc[FEATURES], allc[target])
    cur = d[d.t == 2025].copy()
    cur["pred_roc_vorp"] = final.predict(cur[FEATURES])
    cur = cur.sort_values("pred_roc_vorp", ascending=False)
    cur[["gsis_id", "display_name", "position", "row_type", "recent_team", "age", "exp",
         "draft_pick_f", "ppg", "vorp", "pred_roc_vorp"]].round(2).to_csv(
        OUT / "projections_after_2025.csv", index=False)
    print("\nTop 25 projected ROC VORP (snapshot after 2025 season):\n",
          cur[["display_name", "position", "age", "vorp", "pred_roc_vorp"]].head(25).round(1).to_string())

    # age effect: partial dependence by position (veterans, holding other features at observed values)
    rows = []
    vets = allc[allc.row_type == "veteran"]
    for pos, g in vets.groupby("position"):
        g = g[g.vorp > 0]  # productive players: the interesting case for age
        for a in range(22, 36):
            X = g[FEATURES].copy()
            X["age"] = a
            rows.append({"position": pos, "age": a, "pd_pred_roc_vorp": final.predict(X).mean()})
    pd.DataFrame(rows).pivot(index="age", columns="position", values="pd_pred_roc_vorp").round(1).to_csv(
        OUT / "age_partial_dependence.csv")


if __name__ == "__main__":
    main()
