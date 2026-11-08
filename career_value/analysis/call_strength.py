"""Q: how hard should each Buy and Sell call push? Splits the site's calls (pages/build/rules.py) into
    Buy: strong (pay the market price, or a little over) / Buy: weak (market price or below) / Hold (not backed),
    Sell: heavy (take any fair offer, even below market) / Sell: light (market price, else hold).

Players: FP dynasty top 340 per cohort (May snapshots after season t; market_residuals.prepare, as middle_mode.py),
since most buys are priced 101-300. The site's calls are rebuilt from each player's features at t (rules.classify
with split=False: before this script's split). What holding him returns over h years:
    hold_h = VORP over seasons t+1..t+h + his value at t+h,
value = expected 3-year VORP at that FP rank (isotonic fit on log rank; 0 once he drops out of the top 340). A fair
trade gets, on average, what the typical player at the same price returns:
    fair_h = expected hold_h at his FP rank at t (isotonic fit on log rank, all players).
Lenses (sum over a group of players, divided):
    keep3 = hold_3 / fair_3            cohorts 2018-21   the rebuilder's lens; every mode's sells
    keep1 = hold_1 / fair_1            cohorts 2018-23   one season, then what he's worth; Default's buys
    prod1 = VORP t+1 / same at price   cohorts 2018-23   this season's production; Contending's buys
Sells keep the rebuilder lens in every mode: when each mode sells was settled by the middle_mode.py league sim; this
only says how hard. beat = share of players above their benchmark; solo = keep without the group's best player
(cheap players' means often rest on one hit). Cells are call group x price (FP rank 1-50, 51-100, 101-200, 201+). A
cell with fewer than MIN_N players is pooled: position x age class (rookie, 25 or younger, 26-29, 30+) at the same
price, then position x age class at any price, then the age class across RB / WR / TE (QBs beat their price as a group,
so they are not pooled with the rest); past that, the site says there are too few players to tell (weak buy, light sell). The pools are stored too, for cells the sample never saw. Writes
pages/build/strength.json and outputs/[sf/]call_strength.csv.
LEAGUE=sf for superflex.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import HERE, out_dir  # noqa: E402
from long_horizon import SF  # noqa: E402

sys.path.insert(0, str(HERE.parent / "pages" / "build"))
import rules as R  # noqa: E402

F = "sf" if SF else "1qb"
POS = ["QB", "RB", "WR", "TE"]
POOL = 340
MIN_N = 8
BUY_LENS = {"reb": "keep3", "mid": "keep1", "con": "prod1"}
NAME = {"reb": "Rebuilding", "mid": "Default", "con": "Contending"}
JSON = HERE.parent / "pages" / "build" / "strength.json"


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


def deep(h):
    """FP dynasty rows (market_rows layout) for the top POOL of each cohort at horizon h"""
    sys.path.insert(0, str(HERE))
    import market_residuals as M
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    seasons = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    return M.prepare(M.fp_dynasty(panel), panel, M.outcomes(), seasons, h, top_n=POOL)[0]


def horizons(hs=(1, 3)):
    """h -> rows with hold, fair (and v1fair at h=1), cohorts whose t+h rank is known"""
    X = {h: deep(h) for h in sorted(set(hs) | {1, 3})}
    v3 = iso(X[3].mkt_rank, X[3].vorp3)
    r1 = X[1][["gsis_id", "t", "mkt_rank"]]
    H = {}
    for h in hs:
        x = X[h].merge(r1.assign(t=r1.t - h).rename(columns={"mkt_rank": "rank_h"}), on=["gsis_id", "t"], how="left")
        x = x[x.t + h <= r1.t.max()].reset_index(drop=True)
        x["val_h"] = np.where(x.rank_h.notna(), v3.predict(np.log(x.rank_h.fillna(POOL))), 0.0)
        x["hold"] = x[f"vorp{h}"] + x.val_h
        x["fair"] = iso(x.mkt_rank, x.hold).predict(np.log(x.mkt_rank))
        if h == 1:
            x["v1fair"] = iso(x.mkt_rank, x.vorp1).predict(np.log(x.mkt_rank))
        H[h] = x
    return H


def features(x):
    """yield (row index, the site's feature row, ctx) for each FP row: ranks within the cohort's top POOL, age, NFL
    round, rookie flag, PPG rank in season t and the t / t-1 blend, as build_data.py"""
    pr = prod_ranks()
    look = {(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"])
            for r in json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]}
    for t, g in x.groupby("t"):
        f = pd.DataFrame({
            "position": g.position, "age": g.age, f"r_{F}": g.mkt_rank.rank(method="first"),
            f"pr_{F}": g.groupby("position").mkt_rank.rank(method="first"),
            "rd": g.draft_round_f.fillna(8), "dy": np.where(g.row_type == "rookie", t + 1, 0),
            "prk": g.gsis_id.map(pr[4].get(t, {})),
        })
        f["prod_rank"] = pd.concat([f.prk, g.gsis_id.map(pr[6].get(t - 1, {}))], axis=1).mean(axis=1)
        ctx = {"ROOKIE": t + 1, "LOOK": look}
        yield from ((i, row, ctx) for i, row in zip(g.index, f.itertuples(index=False)))


def calls(x):
    """the site's calls for each row (before the strength split), plus its age class"""
    out = []
    for i, row, ctx in features(x):
        grp, (ra, _), (ca, _), (ma, _) = R.classify(row, F, ctx, split=False)
        r = getattr(row, f"r_{F}")
        out.append((i, grp, ra, ca, ma, R.band(r), R.price_bucket(r), R.age_class(row, ctx)))
    c = pd.DataFrame(out, columns=["i", "grp", "reb", "con", "mid", "band", "pb", "acls"]).set_index("i")
    return x.join(c)


def stat(q, lens):
    num, den = (q.vorp1, q.v1fair) if lens == "prod1" else (q.hold, q.fair)
    coh = pd.DataFrame({"t": q.t, "n": num, "d": den}).groupby("t").sum()
    best = (num - den).idxmax() if len(q) else None
    return {"n": len(q), "keep": num.sum() / den.sum() if den.sum() > 0 else np.nan, "beat": (num > den).mean(),
            "solo": (num.sum() - num[best]) / (den.sum() - den[best]) if len(q) > 1 else np.nan,
            "coh_below": f"{(coh.n < coh.d).sum()}/{len(coh)}", "rank": q.mkt_rank.median()}


def score(b, lens):
    """b: the rows with one kind of call in one mode -> {key: [keep, beat, n, key used, solo]} for every cell
    ("group | price") and pool with MIN_N players, and a table of the cells"""
    pools = {}
    for a, q in b[b.position != "QB"].groupby("acls"):    # QBs age apart: no cross-position pool
        pools[f"all {a}"] = q
    for (p, a), q in b.groupby(["position", "acls"]):
        pools[f"all {p} {a}"] = q
    for (p, a, pb), q in b.groupby(["position", "acls", "pb"]):
        pools[f"all {p} {a} | {pb}"] = q
    out, tab = {}, []
    for key, q in pools.items():
        if len(q) >= MIN_N:
            s = stat(q, lens)
            out[key] = [round(s["keep"], 2), round(s["beat"], 2), s["n"], key, round(s["solo"], 2)]
    for (g, pb), q in b.groupby(["grp", "pb"]):
        p, a, cell = q.position.iloc[0], q.acls.mode().iloc[0], f"{g} | {pb}"
        if len(q) >= MIN_N:
            used = cell
        else:
            used = next((k for k in (f"all {p} {a} | {pb}", f"all {p} {a}", f"all {a}" if p != "QB" else "") if k in out), None)
        s = stat(q, lens)
        if used == cell:
            out[cell] = [round(s["keep"], 2), round(s["beat"], 2), s["n"], cell, round(s["solo"], 2)]
        e = out.get(used, [np.nan, np.nan, 0, None, np.nan])
        tab.append({"grp": g, "price": pb, "calls": "/".join(sorted(q.iloc[:, 0].unique())),
                    **{f"own_{k}": v for k, v in s.items()},
                    "used": used or "(too few)", "keep": e[0], "beat": e[1], "n_used": e[2], "solo": e[4]})
    return out, pd.DataFrame(tab)


def label(tab, side):
    if side == "buy":
        return np.where(tab.keep.isna(), "weak (too few)", np.where(tab.keep < R.BUY_HOLD_BELOW, "Hold (not backed)",
                        np.where((tab.keep >= R.STRONG_AT) & (tab.solo >= 1) & (tab.beat >= R.STRONG_BEAT), "strong", "weak")))
    return np.where(tab.keep < R.HEAVY_BELOW, "heavy", np.where((tab.keep >= 1.1) & (tab.solo >= 1), "light (sell not backed)", "light"))


def main():
    H = horizons()
    H[1] = calls(H[1])
    H[3] = H[3].join(H[1].set_index(["gsis_id", "t"])[["grp", "reb", "con", "mid", "band", "pb", "acls"]], on=["gsis_id", "t"])
    js = json.loads(JSON.read_text()) if JSON.exists() else {}
    js[F] = {"buy": {}, "sell": {}}
    tabs = []
    pd.set_option("display.width", 250, "display.max_colwidth", 34, "display.max_rows", 500)
    print(f"== {'Superflex' if SF else '1QB'}, FP top {POOL}. keep = lens / the same for the average player at his price")
    for col, lens in BUY_LENS.items():
        d = H[3] if lens == "keep3" else H[1]
        b = d[d[col].map(R.kind).eq("buy")].copy()
        b = b[[col] + [c for c in b.columns if c != col]]
        js[F]["buy"][col], t = score(b, lens)
        t.insert(0, "mode", NAME[col]); t.insert(1, "side", "buy"); t["lens"] = lens; t["strength"] = label(t, "buy")
        tabs.append(t)
    s = H[3][H[3].reb.eq(R.SELL)].copy()                      # plain sells, by the rebuilder's call
    s = s[["reb"] + [c for c in s.columns if c != "reb"]]
    js[F]["sell"], t = score(s, "keep3")
    t.insert(0, "mode", "all (rebuilder sells)"); t.insert(1, "side", "sell"); t["lens"] = "keep3"; t["strength"] = label(t, "sell")
    tabs.append(t)
    tab = pd.concat(tabs, ignore_index=True)
    for (m, side), t in tab.groupby(["mode", "side"], sort=False):
        print(f"\n-- {side}: {m} ({t.lens.iloc[0]})")
        print(t.drop(columns=["mode", "side", "lens", "own_coh_below"]).sort_values("keep").round(2).to_string(index=False))
    tab.to_csv(out_dir(SF) / "call_strength.csv", index=False)
    JSON.write_text(json.dumps(js, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
