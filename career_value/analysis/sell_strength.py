"""Q: how hard should each Sell call sell? Split the site's plain Sell calls (pages/build/rules.py) into a heavy
sell (take any fair offer, even below market) and a light sell (sell at the market price, hold otherwise).

For every FP dynasty top-200 player in cohorts 2018-21 (May snapshots after season t), the site's calls are
rebuilt from his features at t as in middle_mode.py (rules.base_calls: before the split). What a rebuilder gets by holding him for H = 3 years:
    hold = VORP over seasons t+1..t+3 + his value at t+3,
where value = the expected 3-year VORP of a player at that FP rank (isotonic fit of vorp3 on log rank, all
players; 0 once he drops out of the top 200). A fair-price trade gets, on average, what the typical player at
the same price returns, so the benchmark is
    fair = expected hold at his FP rank at t (isotonic fit of hold on log rank, all players).
keep = sum(hold) / sum(fair) for a group of players with the same call. keep < 1 backs the sell; how far below 1
says how hard. Small groups: n players, and the share of cohorts whose own keep is below 1.
LEAGUE=sf for superflex.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import HERE, out_dir, rows  # noqa: E402
from long_horizon import SF  # noqa: E402

sys.path.insert(0, str(HERE.parent / "pages" / "build"))
import rules as R  # noqa: E402

H = 3
F = "sf" if SF else "1qb"
POS = ["QB", "RB", "WR", "TE"]


def iso(x, y):
    return IsotonicRegression(increasing=False, out_of_bounds="clip").fit(np.log(x), y)


def prod_ranks():
    """season -> {gsis_id: PPG positional rank}, 4+ games and 6+ games, as build_data.py"""
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    out = {}
    for min_g in (4, 6):
        q = s[s.position.isin(POS) & (s.games >= min_g)].copy()
        q["prk"] = q.groupby(["season", "position"]).ppg.rank(ascending=False, method="min")
        out[min_g] = {y: z.set_index("gsis_id").prk.to_dict() for y, z in q.groupby("season")}
    return out


def load():
    x = rows(["fp_dynasty"], H, SF)
    r1 = rows(["fp_dynasty"], 1, SF)[["gsis_id", "t", "mkt_rank"]]
    later = r1.assign(t=r1.t - H).rename(columns={"mkt_rank": "rank_h"})
    x = x.merge(later, on=["gsis_id", "t"], how="left")
    x = x[x.t + H <= r1.t.max()].reset_index(drop=True)
    v = iso(x.mkt_rank, x[f"vorp{H}"])
    x["val_h"] = np.where(x.rank_h.notna(), v.predict(np.log(x.rank_h.fillna(200))), 0.0)
    x["hold"] = x[f"vorp{H}"] + x.val_h
    x["fair"] = iso(x.mkt_rank, x.hold).predict(np.log(x.mkt_rank))
    return x


def calls(x):
    pr = prod_ranks()
    look = {(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"])
            for r in json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]}
    out = []
    for t, g in x.groupby("t"):
        f = pd.DataFrame({
            "position": g.position, "age": g.age, f"r_{F}": g.mkt_rank.rank(method="first"),
            f"pr_{F}": g.groupby("position").mkt_rank.rank(method="first"),
            "rd": g.draft_round_f.fillna(8), "dy": np.where(g.row_type == "rookie", t + 1, 0),
            "prk": g.gsis_id.map(pr[4].get(t, {})),
        })
        f["prod_rank"] = pd.concat([f.prk, g.gsis_id.map(pr[6].get(t - 1, {}))], axis=1).mean(axis=1)
        ctx = {"ROOKIE": t + 1, "LOOK": look}
        for i, row in zip(g.index, f.itertuples(index=False)):
            grp, (ra, _), _ = R.base_calls(row, F, ctx)        # before the heavy/light split this script sets
            _, _, (ca, _), (ma, _) = R.classify(row, F, ctx)
            out.append((i, grp, ra, ca, ma, R.band(getattr(row, f"r_{F}"))))
    c = pd.DataFrame(out, columns=["i", "grp", "reb", "con", "mid", "band"]).set_index("i")
    return x.join(c)


def keep(q):
    coh = q.groupby("t")[["hold", "fair"]].sum()
    return pd.Series({"n": len(q), "keep": q.hold.sum() / q.fair.sum(), "coh_below": (coh.hold < coh.fair).mean(),
                      "cohorts": len(coh), "vorp3": q[f"vorp{H}"].mean(), "val_h": q.val_h.mean(),
                      "fair": q.fair.mean(), "rank": q.mkt_rank.median()})


def main():
    x = calls(load())
    pd.set_option("display.width", 200)
    print(f"== {'Superflex' if SF else '1QB'}: keep = (3-yr VORP + value at t+3) / same for the average player at his price, "
          f"cohorts {x.t.min()}-{x.t.max()}")
    print("\n-- all players, by call (rebuilder)")
    print(x.groupby("reb").apply(keep, include_groups=False).round(2).to_string())
    sells = x[x.reb.isin([R.SELL, R.SELLH])]
    print("\n-- rebuilder Sell / Sell high now, by group")
    tab = sells.groupby(["reb", "grp"]).apply(keep, include_groups=False).sort_values("keep")
    print(tab.round(2).to_string())
    tab.to_csv(out_dir(SF) / "sell_strength_groups.csv")
    x[["gsis_id", "t", "display_name", "position", "age", "mkt_rank", "grp", "reb", "con", "mid", "band",
       f"vorp{H}", "rank_h", "val_h", "hold", "fair"]].to_csv(out_dir(SF) / "sell_strength_rows.csv", index=False)


if __name__ == "__main__":
    main()
