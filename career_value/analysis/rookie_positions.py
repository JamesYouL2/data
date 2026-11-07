"""Which positions to take with rookie picks? Within each class, rank rookies by market price -> rookie-draft slot
(1-48 = 4 rounds x 12). Expected outcome = isotonic curve of h-year VORP on slot (per market, pooled over classes).
Residual = actual - expected, % of the mean outcome of slots 1-48. Splits by position, slot round, NFL draft round.
Markets: MFL keeper ADP (rookie order within class, classes 2011-2025), FP dynasty, KTC, FantasyCalc, DP (2020-2025).
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))  # project modules
import market_residuals as mr  # noqa: E402
from model import load  # noqa: E402

SLOTS = 48


def mfl_rookies():
    ids = (pd.read_csv(mr.FILES / "db_playerids.csv", dtype=str, usecols=["mfl_id", "gsis_id"])
           .dropna().drop_duplicates("mfl_id"))
    out = []
    for y in range(2011, mr.LAST + 1):
        d = json.loads((mr.RAW / "markets" / f"mfl_keeper_ppr_{y}.json").read_text())["adp"]
        x = pd.DataFrame(d.get("player", []))
        if x.empty:
            continue
        x = x.merge(ids, left_on="id", right_on="mfl_id").assign(t=y - 1)
        x["mkt"] = x.averagePick.astype(float)
        out.append(x[["gsis_id", "t", "mkt"]])
    return pd.concat(out, ignore_index=True)


def main():
    panel = load()
    outc = mr.outcomes()
    rk = panel[panel.row_type == "rookie"][["gsis_id", "t", "display_name", "position", "draft_round_f",
                                           "draft_pick_f", "age"]]
    # panel position = current position; group rookies by their draft-year position (MFL player list of class year)
    ids = (pd.read_csv(mr.FILES / "db_playerids.csv", dtype=str, usecols=["mfl_id", "gsis_id"])
           .dropna().drop_duplicates("mfl_id"))
    dp = []
    for y in range(2011, mr.LAST + 1):
        x = pd.DataFrame(json.loads((mr.RAW / "mfl_players" / f"players_{y}.json").read_text())["players"]["player"])
        dp.append(x[["id", "position"]].merge(ids, left_on="id", right_on="mfl_id").assign(t=y - 1)
                  .rename(columns={"position": "mpos"})[["gsis_id", "t", "mpos"]])
    rk = rk.merge(pd.concat(dp).drop_duplicates(["gsis_id", "t"]), on=["gsis_id", "t"], how="left")
    fix = rk.mpos.isin(["QB", "RB", "WR", "TE"]) & (rk.mpos != rk.position)
    print("relabelled to draft-year position:", fix.sum())
    rk.loc[fix, "position"] = rk.loc[fix, "mpos"]
    mk = {"mfl": mfl_rookies(), "fp_dynasty": mr.fp_dynasty(panel)}
    dd = mr.dd_markets(panel)
    for k in ["dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]:
        mk[k] = dd[k]
    rows = []
    for name, m in mk.items():
        x = m[["gsis_id", "t", "mkt"]].dropna().drop_duplicates(["gsis_id", "t"]).merge(rk, on=["gsis_id", "t"])
        x["slot"] = x.groupby("t").mkt.rank(method="first")
        x = x[x.slot <= SLOTS].merge(outc, on=["gsis_id", "t"], how="left")
        for h in mr.HORIZONS:
            y = f"vorp{h}"
            s = x.dropna(subset=[y]).copy()
            if s.t.nunique() < 2:
                continue
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(s.slot, s[y])
            s["exp"] = iso.predict(s.slot)
            scale = s[y].mean()
            s["res"] = (s[y] - s.exp) / scale * 100
            s["act"] = s[y]
            rows.append(s.assign(market=name, h=h, scale=scale)[
                ["market", "h", "t", "gsis_id", "display_name", "position", "slot", "draft_round_f", "age",
                 "act", "exp", "res"]])
    r = pd.concat(rows, ignore_index=True)
    r["rd"] = "R" + ((r.slot - 1) // 12 + 1).astype(int).astype(str)
    r["nfl"] = np.select([r.draft_round_f == 1, r.draft_round_f == 2], ["NFL R1", "NFL R2"], "NFL R3+")
    r["agegrp"] = pd.cut(r.age, [0, 21.5, 22.5, 99], labels=["<=21", "22", "23+"]).astype(str)
    r["dyn"] = np.where(r.market == "mfl", "MFL 2011-25", "Dynasty 2020-25")
    r.to_csv(sys.argv[1] if len(sys.argv) > 1 else mr.OUT / "rookie_position_residuals.csv", index=False)

    def cell(g):
        cl = g.groupby("t").res.mean()
        return (f"{g.res.mean():+4.0f}/{g.res.median():+4.0f} {(g.res > 0).mean()*100:3.0f}%b "
                f"c{(cl > 0).sum()}/{len(cl)} n{len(g)}")

    def tab(df, by, title, hs=(1, 2, 3, 5, 10)):
        print(f"\n== {title}  [mean/median residual %, % beat, classes positive/total, n]")
        out = {}
        for h in hs:
            d = df[df.h == h]
            if d.empty:
                continue
            out[f"h{h}"] = d.groupby(by).apply(cell, include_groups=False)
        print(pd.DataFrame(out).fillna("").to_string())

    mfl, dyn = r[r.market == "mfl"], r[r.market != "mfl"]
    for lab, d in [("MFL rookie order (classes 2011-2025)", mfl), ("Dynasty markets pooled (classes 2020-2025)", dyn)]:
        tab(d, ["position"], f"{lab}: by position")
        tab(d, ["position", "rd"], f"{lab}: position x rookie round", hs=(2, 3, 5, 10) if d is mfl else (1, 2, 3))
        tab(d, ["position", "nfl"], f"{lab}: position x NFL draft round", hs=(3, 5, 10) if d is mfl else (1, 2, 3))
    tab(mfl, ["position", "agegrp"], "MFL: position x age at draft", hs=(3, 5))
    tab(dyn, ["market", "position"], "Dynasty by market", hs=(1, 2, 3))
    # 1st-round slot share by position and mean outcome (what you get per slot)
    print("\n== MFL: mean h-year VORP by position x rookie round (actual / expected from slot)")
    for h in (3, 5):
        d = mfl[mfl.h == h]
        print(f"h{h}\n" + d.groupby(["rd", "position"]).agg(n=("act", "size"), act=("act", "mean"), exp=("exp", "mean"),
                                                               bust=("act", lambda v: (v == 0).mean() * 100)).round(1).to_string())
    print("\n== MFL era split, position residual at h=3")
    d = mfl[mfl.h == 3].assign(era=lambda z: np.where(z.t <= 2015, "classes 2011-16", "classes 2017-22"))
    print(d.groupby(["era", "position"]).res.agg(["mean", "median", "size"]).round(0).unstack(0).to_string())
    import statsmodels.formula.api as smf
    print("\n== Joint OLS on MFL residual %: position x NFL round + rookie round + age, SE clustered by class")
    for h in (3, 5):
        d = mfl[mfl.h == h].copy()
        d["pn"] = d.position + "_" + d.nfl.str.replace("NFL ", "").str.replace("+", "p")
        f = smf.ols("res ~ C(pn, Treatment('WR_R3p')) + C(rd) + age", d).fit(cov_type="cluster", cov_kwds={"groups": d.t})
        print(f"h{h} n={len(d)}"); print(pd.DataFrame({"coef": f.params, "p": f.pvalues}).round(3).to_string())
    print("\n== Names: RBs and WRs by NFL round, MFL h=5 residual (class, rookie round)")
    d = mfl[(mfl.h == 5) & mfl.position.isin(["RB", "WR", "TE"])]
    for (p, n), g in d.groupby(["position", "nfl"]):
        g = g.sort_values("res")
        f = lambda z: ", ".join(f"{a} {int(b)+1} {r:+.0f}" for a, b, r in zip(z.display_name, z.t, z.res))
        print(f"{p} {n}: best {f(g.tail(5)[::-1])} | worst {f(g.head(4))}")
    print("\n== Biggest positive / negative MFL residuals at h=5 by position (names)")
    d = mfl[mfl.h == 5]
    for p in ["QB", "RB", "WR", "TE"]:
        g = d[d.position == p].sort_values("res")
        f = lambda z: ", ".join(f"{a} {int(b)+1} (R{int((c-1)//12+1)})" for a, b, c in zip(z.display_name, z.t, z.slot))
        print(p, "best:", f(g.tail(6)[::-1]), "| worst:", f(g.head(5)))


if __name__ == "__main__":
    main()
