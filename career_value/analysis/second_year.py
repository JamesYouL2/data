"""Q: second-year players -- are quiet-rookie-year WRs/RBs underpriced (or hit rookies overpriced)?

Veterans one season into their career (exp == 1) in market_rows, split by
  rookie-year production : hit (PPG top-24 RB/WR, top-12 TE/QB, 6+ games) / quiet (6+ games, below that) /
                           barely played (< 6 games)
  price move since rookie snapshot (same market, position rank t-1 -> t): rose >= 5 spots / flat / fell >= 5 /
                           not top-200 as a rookie
  NFL draft round.
Benchmark: the same production split for exp 2 and exp 3+ (is "quiet year" generally cheap, or only in year 2?).
Residual % from market_rows (outcome minus pooled isotonic curve on market rank).
"""
import numpy as np
import pandas as pd

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import DYN, rows, table


def tag(x, h):
    x = x.copy()
    hit_rank = np.where(x.position.isin(["RB", "WR"]), 24, 12)
    x["prod"] = np.select([x.games < 6, x.ppg_pos_rank <= hit_rank], ["barely played", "hit"], "quiet")
    # price move vs rookie snapshot in the same market (use h1 rows: they cover the most cohorts)
    prev = []
    for m in x.mk.unique():
        p = rows([m], 1)
        if p is not None:
            prev.append(p[["gsis_id", "t", "mkt_pos_rank"]].assign(mk=m, t=lambda d: d.t + 1)
                        .rename(columns={"mkt_pos_rank": "prev_pos_rank"}))
    x = x.merge(pd.concat(prev), on=["gsis_id", "t", "mk"], how="left")
    d = x.prev_pos_rank - x.mkt_pos_rank
    x["move"] = np.select([x.prev_pos_rank.isna(), d >= 5, d <= -5], ["unranked t-1", "rose", "fell"], "flat")
    x["nfl"] = np.select([x.draft_round_f == 1, x.draft_round_f == 2], ["R1", "R2"], "R3+")
    x["yr"] = np.select([x.exp == 1, x.exp == 2], ["yr2", "yr3"], "yr4+")
    return x[x.row_type == "veteran"]


def main():
    for fam, mks, hs in (("Dynasty pooled 2019-24", DYN, (1, 2, 3)), ("MFL keeper ADP 2010-24", ["mfl_dynasty"], (1, 2, 3, 5))):
        X = {h: tag(rows(mks, h), h) for h in hs}
        for pos in ["WR", "RB", "TE", "QB"]:
            masks = {}
            for yr in ["yr2", "yr3", "yr4+"]:
                for pr in ["hit", "quiet", "barely played"]:
                    masks[f"{yr} {pr}"] = (lambda yr, pr: lambda d: (d.position == pos) & (d.yr == yr) & (d["prod"] == pr))(yr, pr)
            table({f"h{h}": X[h] for h in hs}, masks, f"{fam} {pos}: entering year 2/3/4+ by last season's production")
        for pos in ["WR", "RB", "TE"]:
            masks = {}
            for pr in ["hit", "quiet", "barely played"]:
                for mv in ["rose", "flat", "fell", "unranked t-1"]:
                    masks[f"{pr} / price {mv}"] = (lambda pr, mv: lambda d: (d.position == pos) & (d.yr == "yr2") & (d["prod"] == pr) & (d.move == mv))(pr, mv)
            for n in ["R1", "R2", "R3+"]:
                masks[f"quiet+barely, NFL {n}"] = (lambda n: lambda d: (d.position == pos) & (d.yr == "yr2") & (d["prod"] != "hit") & (d.nfl == n))(n)
                masks[f"hit, NFL {n}"] = (lambda n: lambda d: (d.position == pos) & (d.yr == "yr2") & (d["prod"] == "hit") & (d.nfl == n))(n)
            table({f"h{h}": X[h] for h in hs}, masks, f"{fam} {pos} entering year 2: production x price move since rookie snapshot / NFL round")
        # names: quiet second-year WR/RB with biggest h2 (dynasty) or h3 (MFL) residuals
        h = 2 if fam.startswith("Dyn") else 3
        z = X[h][(X[h].yr == "yr2") & (X[h]["prod"] != "hit") & X[h].position.isin(["WR", "RB"])]
        z = z.groupby(["gsis_id", "display_name", "position", "t"]).r.mean().reset_index().sort_values("r")
        f = lambda d: ", ".join(f"{a} {p} {t + 1} {r:+.0f}" for a, p, t, r in zip(d.display_name, d.position, d.t, d.r))
        print(f"\n{fam}: quiet/barely-played year-2 WR/RB, h{h} best: {f(z.tail(8)[::-1])}\n   worst: {f(z.head(6))}")


if __name__ == "__main__":
    main()
