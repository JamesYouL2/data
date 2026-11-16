"""Q: does redraft rank plus age rank players better than a dynasty rank does?

Model: VORP over the next h seasons on the summer redraft rank (overall and positional; off the list is its own case), age,
position and rookie or not. Gradient boosting, monotone in the redraft rank, trained in real time: for cohort T and horizon h it
sees only cohorts t <= T - h, from 2012 (FFC redraft ADP; training universe = FFC's list, MFL's dynasty top 200 and rookies
drafted in NFL rounds 1-3).
Test: each dynasty market's top 200 after season T. Spearman with the h-year VORP for the dynasty rank, the redraft rank alone,
the model, and rank blends of the model and the dynasty rank; and the mean VORP of each ranking's top 50, as a % of the best
possible top 50.
  May markets (FP, KTC, FantasyCalc): the model reads FFC ADP, taken in late August, three months after the price.
  FP August: FP dynasty ECR in the same week as FP redraft ECR, which the model reads there. The fair comparison.
LEAGUE=sf for superflex. Prints only.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E402
import market_residuals as mr  # noqa: E402
from model import load  # noqa: E402

HS = (1, 2, 3, 5)
X = ["lrd", "lrd_pos", "off", "age", "rookie", "qb", "rb", "wr", "te"]
OFF, OFF_POS = 250, 80


def feats(u, rd):
    """u: gsis_id, t, position, age, row_type; rd: gsis_id, t, mkt (redraft, lower is better)"""
    rd = rd.merge(u[["gsis_id", "t", "position"]], on=["gsis_id", "t"])
    rd["rd"] = rd.groupby("t").mkt.rank(method="first")
    rd["rd_pos"] = rd.groupby(["t", "position"]).mkt.rank(method="first")
    u = u.merge(rd[["gsis_id", "t", "rd", "rd_pos"]], on=["gsis_id", "t"], how="left")
    u["off"] = u.rd.isna().astype(int)
    u["lrd"], u["lrd_pos"] = np.log(u.rd.fillna(OFF)), np.log(u.rd_pos.fillna(OFF_POS))
    u["rookie"] = (u.row_type == "rookie").astype(int)
    for q in ("QB", "RB", "WR", "TE"):
        u[q.lower()] = (u.position == q).astype(int)
    return u


def main():
    pd.set_option("display.width", 250, "display.max_rows", 400)
    panel, outc = load(), mr.outcomes()
    base = panel[["gsis_id", "t", "position", "age", "row_type", "draft_round_f"]]
    ffc, mfl = mr.ffc_redraft(panel), mr.mfl_dynasty(panel)
    mfl = mfl[mfl.groupby("t").mkt.rank(method="first") <= 200]
    keys = pd.concat([ffc[["gsis_id", "t"]], mfl[["gsis_id", "t"]],
                      base[(base.row_type == "rookie") & (base.draft_round_f <= 3)][["gsis_id", "t"]]]).drop_duplicates()
    train = feats(base.merge(keys, on=["gsis_id", "t"]), ffc).merge(outc, on=["gsis_id", "t"], how="left")

    def aug(ds, y):
        c = [d for d in ds if f"{y}-07-15" <= d <= f"{y}-09-05"]
        return c[-1] if c else None
    et, pages = ("dsf", ["dynasty-op"]) if mr.SF else ("do", ["dynasty-overall", "dynasty-offense"])
    dd = mr.dd_markets(panel)
    tests = {"FP May": (mr.fp_dynasty(panel), ffc), "KTC May": (dd["dd_ktc"], ffc), "FantasyCalc May": (dd["dd_fantasycalc"], ffc),
             "FP August (same date)": (mr.fp_market(et, pages, aug), mr.fp_redraft())}
    models = {}

    def model(T, h):
        if (T, h) not in models:
            q = train[(train.t <= T - h) & train[f"vorp{h}"].notna()]
            models[(T, h)] = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0,
                                                           monotonic_cst=[-1, -1, 0, 0, 0, 0, 0, 0, 0]).fit(q[X], q[f"vorp{h}"])
        return models[(T, h)]

    def top(q, score, y, k=50):
        return q.nlargest(k, score)[y].sum() / q.nlargest(k, y)[y].sum() * 100

    for name, (m, rd) in tests.items():
        m = m[m.groupby("t").mkt.rank(method="first") <= 200].merge(base, on=["gsis_id", "t"])
        u = feats(m, rd[rd.t.isin(m.t.unique())]).merge(outc, on=["gsis_id", "t"], how="left")
        rows, parts = [], []
        for h in HS:
            y = f"vorp{h}"
            for T, q in u[u[y].notna() & u.t.isin(rd.t.unique())].groupby("t"):
                if T - h < 2014 or len(q) < 100:
                    continue
                q = q.assign(dyn=-q.mkt.rank(method="first"), red=-q.lrd, mod=model(T, h).predict(q[X]))
                rk = {c: q[c].rank() for c in ("dyn", "mod")}
                for w in (25, 50, 75):
                    q[f"blend{w}"] = (1 - w / 100) * rk["dyn"] + w / 100 * rk["mod"]
                for c in ("dyn", "red", "mod", "blend25", "blend50", "blend75"):
                    rows.append({"h": h, "t": T, "rank": c, "rho": spearmanr(q[c], q[y])[0], "top50": top(q, c, y)})
                parts.append(q.assign(h=h, y=q[y]))
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        print(f"\n==== {'Superflex' if mr.SF else '1QB'}, {name}: top 200. dyn = dynasty rank, red = redraft rank alone, mod = redraft + age model, "
              "blendW = W% model / rest dynasty (ranks)")
        for met in ("rho", "top50"):
            w = r.pivot_table(index=["h", "t"], columns="rank", values=met)
            tab = w.groupby("h").mean().round(3 if met == "rho" else 1)
            tab["cohorts"] = w.groupby("h").size()
            for c in ("red", "mod", "blend50"):
                tab[f"{c} > dyn"] = (w[c] > w.dyn).groupby("h").sum()
            print(f"-- {'Spearman with h-year VORP' if met == 'rho' else 'VORP of the top 50 by each ranking, % of the best possible'}, mean over cohorts")
            print(tab[["cohorts", "dyn", "red", "mod", "blend25", "blend50", "blend75", "red > dyn", "mod > dyn", "blend50 > dyn"]].to_string())
        a = pd.concat(parts)
        a["grp"] = np.where(a.rookie == 1, "rookie", np.where(a.age < 26, "25-", np.where(a.age < 30, "26-29", "30+")))
        g = (a.groupby(["h", "grp", "t"]).apply(lambda q: pd.Series({c: spearmanr(q[c], q.y)[0] for c in ("dyn", "mod", "blend50")})
                                                 if len(q) >= 15 else None, include_groups=False).dropna())
        t2 = g.groupby(["h", "grp"]).mean().round(3)
        t2["mod > dyn"] = (g["mod"] > g.dyn).groupby(["h", "grp"]).sum().astype(str) + "/" + g.groupby(["h", "grp"]).size().astype(str)
        on = a[(a.rookie == 0) & (a.off == 0)]
        g2 = on.groupby(["h", "t"]).apply(lambda q: pd.Series({c: spearmanr(q[c], q.y)[0] for c in ("dyn", "red", "mod", "blend50")}),
                                          include_groups=False)
        t3 = g2.groupby("h").mean().round(3)
        t3["n a cohort"] = (on.groupby(["h", "t"]).size().groupby("h").mean()).round(0)
        for c in ("red", "mod", "blend50"):
            t3[f"{c} > dyn"] = (g2[c] > g2.dyn).groupby("h").sum().astype(str) + "/" + g2.groupby("h").size().astype(str)
        print("-- Spearman among veterans on the redraft list")
        print(t3.to_string())
        print("-- Spearman within age class")
        print(t2.unstack("grp").to_string())


if __name__ == "__main__":
    main()
