# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "scikit-learn"]
# ///
"""Are rookie draft picks priced correctly? (Dynasty Daddy May-15 snapshots, 2021-2026)

Each snapshot lists pick assets ("2023 Early 1st", ...) next to players for KTC (all years),
FantasyCalc (2022+) and DynastyProcess (2023+), in 1QB and superflex (LEAGUE=sf).

Price.   A pick's value is mapped to the player-rank scale of the same market and snapshot:
         pick_rank = 1 + number of panel players valued above it. Its expected h-year VORP is
         the market's own isotonic rank -> outcome curve (fit on players, pooled over cohorts),
         exactly how player residuals are computed in market_residuals.py.
Outcome. What the pick turned into: the rookies at those slots of a 12-team rookie draft
         (round r, Early = slots 1-4, Mid 5-8, Late 9-12), ordered by the same market's own
         rookie ranking at that class's May snapshot. Outcome = their mean VORP over seasons
         Y..Y+h-1, where Y is the snapshot year. A future pick earns 0 for seasons before its
         class arrives, so the delay is priced in. Future-pick tiers are not known when bought,
         so future picks are also scored at the round level (mean of the 12 slots).
Residual = outcome - expected, as % of the average top-200 player's h-year outcome
         (same scale as market_residuals_by_group.csv).
Drift.   Pick price (May Y) vs the price of the rookies it became (May of the class
         year), both in expected-value units of the market's h=3 curve. > 1 means the pick was
         priced above what its rookies were later priced at.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import market_residuals as mr
from model import load

OUT = mr.OUT
SRC = {f"dd_{v}": k for k, v in mr.DD_SOURCES.items() if v in ("ktc", "fantasycalc", "dynastyprocess")}
TIERS = {"early": range(1, 5), "mid": range(5, 9), "late": range(9, 13)}
ROUNDS = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
PICK = re.compile(r"^(\d{4})(early|mid|late)(1st|2nd|3rd|4th)pi$")


def pick_prices():
    rows = []
    for f in sorted((mr.RAW / "dynastydaddy").glob("snapshot_*.json")):
        y = int(f.stem.split("_")[1][:4])
        x = pd.DataFrame(json.loads(f.read_text()))
        for _, r in x[x.name_id.str.match(PICK)].iterrows():
            py, tier, rnd = PICK.match(r.name_id).groups()
            for mkt, col in SRC.items():
                v = pd.to_numeric(r.get(col), errors="coerce")
                if v and v > 0:
                    rows.append({"market": mkt, "Y": y, "t": y - 1, "class": int(py), "tier": tier,
                                 "round": ROUNDS[rnd], "value": float(v)})
    p = pd.DataFrame(rows)
    # a market that prices every tier of a round identically (FC future picks) is fine; drop
    # placeholder years where the market lists a pick at the same value as the next round
    return p


def main():
    panel = load()
    outc = mr.outcomes()
    mk = mr.dd_markets(panel)
    picks = pick_prices()
    pan = panel[["gsis_id", "t", "row_type"]]
    rows, drift = [], []
    for mname in SRC:
        if mname not in mk:
            continue
        m = mk[mname].merge(pan, on=["gsis_id", "t"])
        m["value"] = -m.mkt
        # rookie order per class (class C rookies are row_type rookie at t = C-1)
        rk = m[m.row_type == "rookie"].copy()
        rk["slot"] = rk.groupby("t").mkt.rank(method="first")
        rk["class"] = rk.t + 1
        pk = picks[picks.market == mname]
        for h in [1, 2, 3, 5]:
            y = f"vorp{h}"
            pl = m.merge(outc[["gsis_id", "t", y]], on=["gsis_id", "t"]).dropna(subset=[y])
            if pl.empty:
                continue
            # curve fit on every ranked player (late picks can sit below the top 200);
            # residual scale stays the top-200 mean, as in market_residuals.py
            pl["rank"] = pl.groupby("t").mkt.rank(method="first")
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(pl["rank"], pl[y])
            scale = pl[pl["rank"] <= mr.TOP_N][y].mean()
            for _, p in pk[pk.t.isin(pl.t.unique())].iterrows():
                vals = pl[pl.t == p.t].value
                prank = 1 + (vals > p.value).sum()
                slots = [(p["round"] - 1) * 12 + s for s in TIERS[p.tier]]
                got = rk[(rk["class"] == p["class"]) & rk.slot.isin(slots)][["gsis_id"]].assign(t=p.t)
                got = got.merge(outc[["gsis_id", "t", y]], on=["gsis_id", "t"])
                if len(got) < 3:
                    continue
                act = got[y].mean()
                exp = float(iso.predict([prank])[0])
                rows.append({"market": mname, "h": h, "Y": p.Y, "n_players_ranked": len(vals), "class": p["class"], "years_out": p["class"] - p.Y,
                             "round": p["round"], "tier": p.tier, "pick_rank": prank, "exp": exp, "act": act,
                             "resid_pct": (act - exp) / scale * 100})
        # price drift: pick (May Y) vs the price of its rookies at May of the class year (years_out 0 = same day)
        x3 = m.merge(outc[["gsis_id", "t", "vorp3"]], on=["gsis_id", "t"]).dropna()
        x3["rank"] = x3.groupby("t").mkt.rank(method="first")
        iso3 = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x3["rank"], x3.vorp3)
        allm = m.assign(rank=m.groupby("t").mkt.rank(method="first"))
        for _, p in pk[pk["class"] >= pk.Y].iterrows():
            vals = allm[allm.t == p.t].value
            prank = 1 + (vals > p.value).sum()
            slots = [(p["round"] - 1) * 12 + s for s in TIERS[p.tier]]
            later = rk[(rk["class"] == p["class"]) & rk.slot.isin(slots)].merge(allm[["gsis_id", "t", "rank"]],
                                                                                on=["gsis_id", "t"])
            den = iso3.predict(later["rank"]).mean() if len(later) else 0.0
            if len(later) < 3 or den <= 0:  # rookies priced where the curve is 0 (deep 4ths): ratio undefined
                continue
            drift.append({"market": mname, "Y": p.Y, "class": p["class"], "years_out": p["class"] - p.Y,
                          "round": p["round"], "tier": p.tier, "pick_rank": prank,
                          "rookie_rank_later": later["rank"].mean(),
                          "price_ratio": float(iso3.predict([prank])[0]) / den})
    r = pd.DataFrame(rows)
    d = pd.DataFrame(drift)
    r.round(3).to_csv(OUT / "picks_residuals.csv", index=False)
    d.round(3).to_csv(OUT / "picks_price_drift.csv", index=False)

    def tab(s, by, title):
        g = s.groupby(by + ["h"]).agg(res=("resid_pct", "mean"), beat=("resid_pct", lambda v: (v > 0).mean() * 100),
                                      n=("resid_pct", "size"), classes=("class", "nunique"))
        g["cell"] = g.res.round(0).astype(int).map("{:+d}".format) + " (" + g.beat.round(0).astype(int).astype(str) \
            + "% beat, " + g.classes.astype(str) + " cl)"
        print(f"\n{title}\n" + g.cell.unstack("h").fillna("").to_string())

    cur = r[r.years_out == 0]
    tab(cur, ["round", "tier"], "CURRENT-YEAR picks (May, after the NFL draft): residual % (share of pick-years that beat price, classes)")
    tab(cur, ["market", "round"], "Current-year picks by market")
    fut = r[r.years_out > 0].groupby(["market", "h", "Y", "class", "years_out", "round"]).agg(
        resid_pct=("resid_pct", "mean")).reset_index()
    tab(fut, ["years_out", "round"], "FUTURE picks, round level (mean of early/mid/late): residual %")
    tab(fut, ["market", "round"], "Future picks by market")
    print("\nPRICE DRIFT: pick price / price of the rookies it became (h=3 value units; >1 = pick cost more)")
    print(d.groupby(["years_out", "round"]).price_ratio.agg(["median", "mean", "size"]).round(2).to_string())
    print(d.groupby(["market", "years_out"]).price_ratio.median().round(2).unstack().to_string())


if __name__ == "__main__":
    main()
