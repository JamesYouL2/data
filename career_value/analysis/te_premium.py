"""Q: does TE premium (1.5 PPR for TEs) change the TE answer?

Rescores TE seasons as ppr + 0.5 * receptions with replacement re-set at TE13 under that scoring (other positions
unchanged), rebuilds h-year outcomes, and refits each market's isotonic price curve on the new outcomes. Market
prices are the same (1QB PPR prices; Dynasty Daddy has no TEP values), so a TE residual near zero means "the
standard price is already right in TEP", and a positive one means TEs need a bump.

Break-even bump (KTC / FantasyCalc / DP raw trade values): multiply every TE's value by 1 + b, re-rank, refit,
and report the TE residual for each b. The b where it crosses zero is the TEP premium the outcomes justify.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import DD, DYN, HERE, table

H = (1, 2, 3, 5)


def tep_outcomes(bonus=0.5):
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    te = s.position == "TE"
    s["pts"] = s.ppr + np.where(te, bonus * s.receptions.fillna(0), 0)
    s["rk"] = s.groupby(["season", "position"]).pts.rank(ascending=False, method="first")
    s["v"] = np.where(te, (s.pts - s.season.map(repl_te(s))).clip(0), s.vorp)
    v = s.pivot_table(index="gsis_id", columns="season", values="v", fill_value=0.0)
    last = 2025
    for y in range(1999, last + 2):
        if y not in v:
            v[y] = 0.0
    out = []
    for t in range(1999, last + 1):
        o = pd.DataFrame({"gsis_id": v.index, "t": t})
        for h in H:
            o[f"tep{h}"] = v[list(range(t + 1, t + h + 1))].sum(axis=1).to_numpy() if t + h <= last else np.nan
        out.append(o)
    return pd.concat(out, ignore_index=True)


def repl_te(s):
    return s[(s.position == "TE") & (s.rk == 13)].set_index("season").pts


def resid(x, y, rank="mkt_rank"):
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x[rank], x[y])
    return (x[y] - iso.predict(x[rank])) / x[y].mean() * 100


def load(markets, h, tep):
    out = []
    for m in markets:
        f = HERE / "outputs" / "market_rows" / f"{m}_h{h}.parquet"
        if not f.exists():
            continue
        x = pd.read_parquet(f).merge(tep[["gsis_id", "t", f"tep{h}"]], on=["gsis_id", "t"], how="left")
        x[f"tep{h}"] = x[f"tep{h}"].fillna(0)
        x["r_ppr"], x["r_tep"], x["mk"] = resid(x, f"vorp{h}"), resid(x, f"tep{h}"), m
        out.append(x)
    return pd.concat(out, ignore_index=True)


def main():
    tep = tep_outcomes()
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    s = s[(s.position == "TE") & s.season.between(2010, 2025)]
    print("TE rescoring: mean replacement TE13 ppr", s[s.pos_rank == 13].ppr.mean().round(0),
          "| TE1 VORP gain from +0.5/rec (TE ranks 1-6 vs 7-12 vs 13+, per season):",
          s.assign(g=pd.cut(s.pos_rank, [0, 6, 12, 99])).groupby("g", observed=True).receptions.mean().mul(0.5).round(1).to_dict())
    masks = {
        "TE all": lambda d: d.position == "TE",
        "TE priced TE1-3": lambda d: (d.position == "TE") & (d.mkt_pos_rank <= 3),
        "TE priced TE4-12": lambda d: (d.position == "TE") & d.mkt_pos_rank.between(4, 12),
        "TE priced TE13+": lambda d: (d.position == "TE") & (d.mkt_pos_rank > 12),
        "TE <25": lambda d: (d.position == "TE") & (d.age < 25),
        "TE 25-28": lambda d: (d.position == "TE") & d.age.between(25, 28.99),
        "TE 29+": lambda d: (d.position == "TE") & (d.age >= 29),
        "TE rookies": lambda d: (d.position == "TE") & (d.row_type == "rookie"),
        "TE top-6 PPG last yr": lambda d: (d.position == "TE") & (d.ppg_pos_rank <= 6),
        "WR all (for scale)": lambda d: d.position == "WR",
    }
    for fam, mks, hs in (("Dynasty pooled 2019-24", DYN, (1, 2, 3)), ("MFL keeper ADP 2010-24", ["mfl_dynasty"], H)):
        frames = {}
        for h in hs:
            x = load(mks, h, tep)
            frames[f"h{h} PPR"] = x.assign(r=x.r_ppr)
            frames[f"h{h} TEP"] = x.assign(r=x.r_tep)
        table(frames, masks, f"{fam}: residual % in standard PPR vs TE premium (same prices)")

    # break-even TE value bump on raw trade values
    print("\n== Break-even TE value bump (KTC/FC/DP raw values x (1+b) for TEs, re-ranked): TE residual mean %")
    rows = []
    for h in (1, 2, 3):
        x = load(DD, h, tep)
        for b in (-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2):
            z = x.copy()
            z["mkt2"] = np.where(z.position == "TE", z.mkt * (1 + b), z.mkt)
            z["rk2"] = z.groupby(["mk", "t"]).mkt2.rank(method="first")
            for sc, y in (("PPR", f"vorp{h}"), ("TEP", f"tep{h}")):
                r = z.groupby("mk", group_keys=False).apply(lambda d: resid(d, y, "rk2"), include_groups=False)
                te = z.position == "TE"
                top = te & (z.mkt_pos_rank <= 12)
                rows.append({"h": h, "scoring": sc, "bump": f"{int(round(b * 100)):+d}%", "TE all": r[te].mean(),
                             "TE priced TE1-12": r[top].mean()})
    print(pd.DataFrame(rows).pivot_table(index=["scoring", "bump"], columns="h", values=["TE all", "TE priced TE1-12"])
          .round(0).to_string())


if __name__ == "__main__":
    main()
