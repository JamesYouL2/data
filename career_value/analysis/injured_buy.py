"""Backtest of the site's "Buy low (injured)" call: is a player who is out for the season worth buying, and for whom?

The site (pages/build/rules.situational) tells a rebuilding team to buy a player on IR / PUP / Out who has dropped off the
FantasyPros rest-of-season list when the trade market has marked him well below his FP dynasty rank (rules.gap: market rank at
least 25% worse, 12 places minimum). Here: the in-season panel (inseason_movers.csv: FP dynasty rank, ROS list, the injury report,
at weeks 4 / 8 / 12, 2019-25), each player-season once, at the first snapshot where he is hurt and off the ROS list, FP top 200.
Return = what he was worth from then on (next season's VORP for hold1, the next three for hold3, plus the value of the rank held
after), against the average player at the price paid: his FP rank ("fp") or the market's rank that day ("mkt", KTC, 2020 on).
Elite = his position rank on the preseason FP dynasty list. Rows: n, returned / price, the same without the best player, share
who beat the price, cohorts above 1.

  LEAGUE=sf uv run --with pandas --with pyarrow --with scikit-learn --with scipy --with statsmodels python analysis/injured_buy.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import call_strength as cs  # noqa: E402
import market_price_sim as mp  # noqa: E402
from common import HERE, out_dir  # noqa: E402
from long_horizon import SF  # noqa: E402

sys.path.insert(0, str(HERE.parent / "pages" / "build"))
import rules as LIVE  # noqa: E402

WEEKS = (4, 8, 12)


def load():
    x = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    x = x[x.target.isin(WEEKS)].copy()
    mkt = mp.market_ranks(pd.read_parquet(HERE / "data" / "panel.parquet"))
    x["mrank"] = [mkt[d]["rank"].get(g, np.nan) if d in mkt else np.nan for d, g in zip(x.date.astype(str), x.gsis_id)]
    x["mgap"] = [LIVE.gap(r, m) for r, m in zip(x["rank"], x.mrank)]
    pre = x[x.pre < 500].drop_duplicates(["season", "gsis_id"])          # off the preseason list: no preseason position rank
    x["pre_pos"] = x.set_index(["season", "gsis_id"]).index.map(
        pre.assign(pp=pre.groupby(["season", "position"]).pre.rank(method="first")).set_index(["season", "gsis_id"]).pp).to_numpy()
    for h in (1, 3):
        x[f"m_hold{h}"] = np.nan                       # the average return at the market's rank
        for t, g in x[x[f"hold{h}"].notna()].groupby("target"):
            ok = g[g.mrank.notna()]
            x.loc[ok.index, f"m_hold{h}"] = cs.iso(g["rank"], g[f"hold{h}"]).predict(np.log(ok.mrank.clip(1, mp.POOL)))
    x["out"] = x.hurt & (x.ros_rank == 500)
    return x


def first(x, mask):
    return x[mask].sort_values("target").drop_duplicates(["season", "gsis_id"])


def cell(g, h, price):
    col, fair = f"hold{h}", (f"f_hold{h}" if price == "fp" else f"m_hold{h}")
    g = g[g[col].notna() & g[fair].notna()]
    if len(g) < 5:
        return ""
    best = (g[col] - g[fair]).idxmax()
    ct = g.groupby("season")[[col, fair]].sum()
    return (f"n {len(g):3d}  {g[col].sum() / g[fair].sum():.2f}  ex-best {(g[col].sum() - g[col][best]) / (g[fair].sum() - g[fair][best]):.2f}  "
            f"{(g[col] > g[fair]).mean():.0%} beat  {(ct[col] > ct[fair]).sum()}/{len(ct)}")


def main():
    x = load()
    top = x["rank"] <= 200
    o = first(x, x.out & top)
    well = first(x, ~x.hurt & top & (x.ros_rank != 500))
    young, mid, old = o.age < 26, (o.age >= 26) & (o.age < 28), o.age >= 28
    e5, e10 = o.pre_pos <= 5, o.pre_pos <= 10
    rb, wr = o.position == "RB", o.position == "WR"
    cuts = {
        "healthy and on the ROS list (baseline)": well,
        "out for the season, all": o,
        "  the site's call: market marked him down": o[o.mgap == "less"],
        "  market not marked down": o[o.mrank.notna() & (o.mgap != "less")],
        "  top 5 at position preseason": o[e5], "  top 10": o[e10], "  11-24": o[(o.pre_pos > 10) & (o.pre_pos <= 24)], "  25+": o[o.pre_pos > 24],
        "  age 25-": o[young], "  age 26-27": o[mid], "  age 28+": o[old],
        "  top 10, age 27-": o[e10 & ~old], "  top 10, age 28+": o[e10 & old], "  not top 10, age 27-": o[~e10 & ~old], "  not top 10, age 28+": o[~e10 & old],
        "  RB": o[rb], "  WR": o[wr], "  QB": o[o.position == "QB"], "  TE": o[o.position == "TE"],
        "  RB top 10": o[rb & e10], "  RB top 10, age 27-": o[rb & e10 & ~old], "  RB not top 10": o[rb & ~e10], "  RB age 25-": o[rb & young],
        "  WR top 10": o[wr & e10], "  WR top 10, age 27-": o[wr & e10 & ~old], "  WR not top 10": o[wr & ~e10], "  WR age 25-": o[wr & young],
        "  not RB: top 10, age 27-": o[~rb & e10 & ~old],
        "  rookie": o[o.row_type == "rookie"],
    }
    pd.set_option("display.width", 260)
    for h in (1, 3):
        t = pd.DataFrame({f"at his FP rank": {k: cell(g, h, "fp") for k, g in cuts.items()},
                          f"at the market's rank (KTC)": {k: cell(g, h, "mkt") for k, g in cuts.items()}})
        print(f"\n== {'Superflex' if SF else '1QB'}, {h}-season hold: n, returned / price, without the best, share beating, cohorts above 1")
        print(t.to_string())
    g = o[e10 & o.hold3.notna()].assign(ratio=lambda d: d.hold3 / d.f_hold3)
    panel = pd.read_parquet(HERE / "data" / "panel.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
    nm = next((c for c in ("display_name", "name", "player") if c in panel.columns), None)
    if nm:
        g["name"] = g.gsis_id.map(panel[nm])
        print("\ntop 10 preseason, out for the season, 3-season hold / FP price:")
        print(g.sort_values("ratio")[["name", "season", "position", "age", "rank", "mrank", "ratio"]].round(2).to_string(index=False))
    o.to_csv(out_dir(SF) / "injured_buy.csv", index=False)


if __name__ == "__main__":
    main()
