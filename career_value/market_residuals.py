# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "scikit-learn"]
# ///
"""Which player groupings beat / trail the market (consensus rankings, ADP)?

For each market snapshot we map market rank -> expected outcome with an
isotonic fit (pooled over years), then average residuals (actual - expected)
within groupings defined from information available at snapshot time.

Markets (snapshot t = offseason after NFL season t):
  fp_dynasty   FantasyPros dynasty ECR, first scrape May 10-Aug 31 of t+1     (this repo)
               t=2018 (1QB): files/archives/fantasypros weekly CSV of 2019-05-14
               t=2018-19 (LEAGUE=sf): FP 2QB ECR carried in DynastyProcess values-players.csv (git history,
               May 2019 / May 2020; see fetch_markets.py), as FP's own superflex ECR starts Oct 2020
  mfl_dynasty  MFL keeper-league PPR ADP for year t+1, veterans only          (MFL API)
  fp_redraft   FantasyPros redraft ECR, last scrape before Sept 9 of t+1      (this repo)
  ffc_redraft  FantasyFootballCalculator 12-team PPR ADP, final preseason      (FFC API)

Outcomes: VORP over seasons t+1..t+h for h in 1,2,3,5,10 (cohorts with t+h <= 2025)
          redraft markets use h=1 only. Also: Dynasty Daddy archive (KTC, FantasyCalc, DP, ...)
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).parent
FILES = HERE.parent / "files"
RAW = HERE / "raw"
# LEAGUE=sf -> superflex: outcome = superflex VORP, superflex versions of each market
# (same market names so downstream scripts work), outputs under outputs/sf/.
SF = os.environ.get("LEAGUE", "1qb").lower() == "sf"
VALUE = "vorp_sf" if SF else "vorp"
OUT = HERE / "outputs" / ("sf" if SF else "")
POS = ["QB", "RB", "WR", "TE"]
LAST = 2025
TOP_N = 200
HORIZONS = [1, 2, 3, 5, 10]
RNG = np.random.default_rng(0)

# Dynasty Daddy archive columns (1QB values) -> market name. Labels from dynasty-daddy.com bundle.
DD_SOURCES_1QB = {
    "trade_value": "ktc", "fc_trade_value": "fantasycalc", "dp_trade_value": "dynastyprocess",
    "ds_trade_value": "dynastysuperflex", "pfn_trade_value": "pfn", "dss_trade_value": "draftsharks",
    "adp_trade_value": "dd_adp",
}
DD_SOURCES = {k.replace("trade_value", "sf_trade_value") if k != "trade_value" else "sf_trade_value": v
              for k, v in DD_SOURCES_1QB.items()} if SF else DD_SOURCES_1QB


# ---------------------------------------------------------------- outcomes
def outcomes() -> pd.DataFrame:
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")[["gsis_id", "season", VALUE]]
    v = s.pivot_table(index="gsis_id", columns="season", values=VALUE, fill_value=0.0)
    for y in range(1999, LAST + 2):
        if y not in v:
            v[y] = 0.0
    rows = []
    for t in range(1999, LAST + 1):
        o = pd.DataFrame({"gsis_id": v.index, "t": t})
        for h in HORIZONS:
            o[f"vorp{h}"] = v[list(range(t + 1, t + h + 1))].sum(axis=1).to_numpy() if t + h <= LAST else np.nan
        rows.append(o)
    o = pd.concat(rows, ignore_index=True)
    o["next_vorp"] = o.vorp1
    return o


# ---------------------------------------------------------------- markets
def fp_market(ecr_type, pages, pick) -> pd.DataFrame:
    e = pd.read_parquet(FILES / "db_fpecr.parquet",
                        columns=["page_type", "ecr_type", "scrape_date", "id", "pos", "ecr", "sd", "tm"])
    e = e[(e.ecr_type == ecr_type) & e.page_type.isin(pages) & e.pos.str[:2].isin(POS)]
    ids = (pd.read_csv(FILES / "db_playerids.csv", dtype=str, usecols=["fantasypros_id", "gsis_id"])
           .dropna().drop_duplicates("fantasypros_id"))
    out = []
    for y in range(2020, LAST + 2):
        dates = sorted(e.scrape_date.unique())
        date = pick(dates, y)
        if date is None:
            continue
        x = e[e.scrape_date == date]
        page = pages[0] if (x.page_type == pages[0]).any() else pages[-1]
        x = x[x.page_type == page].merge(ids, left_on="id", right_on="fantasypros_id")
        x = x.drop_duplicates("gsis_id").assign(t=y - 1, scrape_date=date)
        out.append(x[["gsis_id", "t", "ecr", "sd", "tm", "scrape_date"]].rename(columns={"ecr": "mkt"}))
    return pd.concat(out, ignore_index=True)


def fp_offseason(ds, y):
    """first scrape May 10-Aug 31 of year y (none -> skip the cohort rather than use an in-season scrape)"""
    return next((d for d in ds if f"{y}-05-10" <= d <= f"{y}-08-31"), None)


NAME_ALIAS = {"robby anderson": "robbie chosen", "mitch trubisky": "mitchell trubisky"}


def match_names(x, t, panel):
    """x(name, pos) -> gsis_id by normalized name + position; duplicates resolved by a panel row at t.
    Fallback for players whose current position differs (e.g. Cordarrelle Patterson): name only, if the candidate
    has a panel row at t."""
    pl = pd.read_parquet(RAW / "players" / "players.parquet", columns=["gsis_id", "display_name", "position"])
    pl = pl[pl.position.isin(POS)].assign(nm=lambda d: d.display_name.map(norm_name))
    pl["key"] = pl.nm + "|" + pl.position
    ok = panel[panel.t == t][["gsis_id"]].drop_duplicates().assign(ok=1)
    x = x.assign(nm=x.name.map(norm_name).replace(NAME_ALIAS))
    x["key"] = x.nm + "|" + x.pos
    m = x.merge(pl[["gsis_id", "key"]], on="key").merge(ok, on="gsis_id", how="left")
    m = m.sort_values("ok", na_position="last").drop_duplicates("key")
    r = x[~x.key.isin(m.key)].merge(pl[["gsis_id", "nm"]], on="nm").merge(ok, on="gsis_id")
    r = r.drop_duplicates("key", keep=False)  # ambiguous -> drop
    m = pd.concat([m, r]).drop_duplicates("gsis_id").drop(columns=["ok", "nm"])
    print(f"  name match t={t}: {len(m)}/{len(x)}")
    return m


def fp_archive_name(s):
    """'Odell Beckham Jr.O. Beckham NYG' -> 'Odell Beckham Jr.' (full name, then abbreviated name, then team)"""
    s = s.strip().rsplit(" ", 1)[0] if re.search(r" [A-Z]{2,3}$", s.strip()) else s.strip()
    for mt in reversed(list(re.finditer(r"(?<=[A-Za-z.'])[A-Z]\. ", s))):
        k = mt.start()
        if s[k + 3:] and s[k + 3:] in s[:k]:
            return s[:k]
    return s


def fp_dynasty_early(panel):
    if SF:
        out = []
        for f, t, col in (("dp_values_2019-05-27.csv", 2018, "dyno2QBECR"), ("dp_values_2020-05-15.csv", 2019, "ecr_2qb")):
            x = pd.read_csv(RAW / "markets" / f, encoding="utf-8-sig")
            x = x.rename(columns={"mergename": "name", "player": "name"})
            x = x[x.pos.isin(POS)]
            if "fp_id" in x:
                ids = (pd.read_csv(FILES / "db_playerids.csv", dtype=str, usecols=["fantasypros_id", "gsis_id"])
                       .dropna().drop_duplicates("fantasypros_id"))
                a = x.assign(fp_id=x.fp_id.astype(str)).merge(ids, left_on="fp_id", right_on="fantasypros_id")
                b = match_names(x[~x.name.isin(a.name)], t, panel)
                m = pd.concat([a, b]).drop_duplicates("gsis_id")
                print(f"  DP {f}: fp_id + name match {len(m)}/{len(x)}")
            else:
                m = match_names(x, t, panel)
            out.append(pd.DataFrame({"gsis_id": m.gsis_id, "t": t, "mkt": m[col].astype(float),
                                     "scrape_date": f[10:20]}))
        return pd.concat(out, ignore_index=True)
    date = "20190514"
    x = pd.read_csv(FILES / "archives" / "fantasypros" / f"dynasty-overall-{date}.csv").dropna(subset=["Overall (Team)"])
    x = x.assign(name=x["Overall (Team)"].map(fp_archive_name), pos=x.Pos.str.extract(r"^([A-Z]+)")[0])
    x = x[x.pos.isin(POS)]
    m = match_names(x, 2018, panel)
    return pd.DataFrame({"gsis_id": m.gsis_id, "t": 2018, "mkt": m.Rank.astype(float), "sd": m["Std Dev"],
                         "scrape_date": f"{date[:4]}-{date[4:6]}-{date[6:]}"})


def fp_dynasty(panel):
    if SF:  # FP superflex dynasty ECR ("OP" page), from Oct 2020
        cur = fp_market("dsf", ["dynasty-op"], fp_offseason)
    else:
        cur = fp_market("do", ["dynasty-overall", "dynasty-offense"], fp_offseason)
    early = fp_dynasty_early(panel)
    return pd.concat([early[~early.t.isin(cur.t)], cur], ignore_index=True)


def fp_redraft():
    def pick(ds, y):
        c = [d for d in ds if f"{y}-08-15" <= d <= f"{y}-09-08"]
        return c[-1] if c else None
    if SF:
        return fp_market("rsf", ["redraft-op"], pick)
    return fp_market("ro", ["redraft-overall", "redraft-offense"], pick)


def mfl_dynasty(panel):
    ids = (pd.read_csv(FILES / "db_playerids.csv", dtype=str, usecols=["mfl_id", "gsis_id"])
           .dropna().drop_duplicates("mfl_id"))
    out = []
    for y in range(2011, LAST + 1):
        d = json.loads((RAW / "markets" / f"mfl_keeper_ppr_{y}.json").read_text())["adp"]
        x = pd.DataFrame(d.get("player", []))
        if x.empty:
            continue
        x = x.merge(ids, left_on="id", right_on="mfl_id").assign(t=y - 1)
        x["mkt"] = x.averagePick.astype(float)
        out.append(x[["gsis_id", "t", "mkt"]])
    m = pd.concat(out, ignore_index=True)
    # keeper ADP mixes rookie-only drafts in -> rookie ADP not comparable. Veterans only.
    vets = panel[panel.row_type == "veteran"][["gsis_id", "t"]]
    return m.merge(vets, on=["gsis_id", "t"])


def norm_name(s: str) -> str:
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def ffc_redraft(panel):
    pl = pd.read_parquet(RAW / "players" / "players.parquet", columns=["gsis_id", "display_name", "position"])
    pl = pl[pl.position.isin(POS)]
    pl["key"] = pl.display_name.map(norm_name) + "|" + pl.position
    out, n_raw, n_hit = [], 0, 0
    fmt, first = ("2qb", 2014) if SF else ("ppr", 2013)
    for y in range(first, LAST + 1):
        d = json.loads((RAW / "markets" / f"ffc_{fmt}_{y}.json").read_text())
        x = pd.DataFrame(d.get("players", []))
        x = x[x.position.isin(POS)]
        x["key"] = x.name.map(norm_name) + "|" + x.position
        m = x.merge(pl[["gsis_id", "key"]], on="key")
        # resolve duplicate names: keep candidate who has a panel row at t=y-1 (played the prior season)
        # or is a rookie that year
        cand = panel[panel.t == y - 1][["gsis_id"]].assign(ok=1)
        m = m.merge(cand, on="gsis_id", how="left")
        m = m.sort_values("ok", na_position="last").drop_duplicates("key")
        n_raw += len(x); n_hit += len(m)
        out.append(m.assign(t=y - 1, mkt=m.adp.astype(float))[["gsis_id", "t", "mkt"]])
    print(f"  FFC name match: {n_hit}/{n_raw} = {n_hit / n_raw:.1%}")
    return pd.concat(out, ignore_index=True)


def dd_key(name: str, pos: str, strip_suffix=True) -> str:
    s = name.lower()
    if strip_suffix:
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s) + pos.lower()


def dd_markets(panel, pattern="snapshot_*.json") -> dict[str, pd.DataFrame]:
    """Dynasty Daddy archive -> {source: df(gsis_id, t, mkt, snap_date)}.

    Default: one May snapshot per year. pattern="inseason_*.json" reads the Sept-Feb snapshots.
    t is the season the snapshot follows or falls in (Jul-Dec of Y -> Y, Jan-Jun of Y -> Y-1).
    """
    d = RAW / "dynastydaddy"
    ids = pd.read_csv(FILES / "db_playerids.csv", dtype=str, usecols=["mfl_id", "sleeper_id", "gsis_id"])
    today = pd.DataFrame(json.loads(sorted(d.glob("today_*.json"))[-1].read_text()))
    m1 = today.merge(ids.dropna(subset=["mfl_id", "gsis_id"]).drop_duplicates("mfl_id"), on="mfl_id")
    m2 = today.merge(ids.dropna(subset=["sleeper_id", "gsis_id"]).drop_duplicates("sleeper_id"), on="sleeper_id")
    by_id = pd.concat([m1, m2])[["name_id", "gsis_id"]].drop_duplicates("name_id")

    pl = pd.read_parquet(RAW / "players" / "players.parquet", columns=["gsis_id", "display_name", "position"])
    pl = pl[pl.position.isin(POS)]
    keys = pd.concat([
        pl.assign(name_id=[dd_key(n, p) for n, p in zip(pl.display_name, pl.position)]),
        pl.assign(name_id=[dd_key(n, p, False) for n, p in zip(pl.display_name, pl.position)]),
    ])[["name_id", "gsis_id"]].drop_duplicates()

    frames = {v: [] for v in DD_SOURCES.values()}
    n_raw = n_hit = 0
    for f in sorted(d.glob(pattern)):
        date = f.stem.split("_")[1]
        year, month = int(date[:4]), int(date[5:7])
        t = year if month >= 7 else year - 1
        x = pd.DataFrame(json.loads(f.read_text()))
        x = x[x.name_id.str[-2:].isin([p.lower() for p in POS])]
        a = x.merge(by_id, on="name_id", how="inner")
        rest = x[~x.name_id.isin(a.name_id)]
        # name fallback; disambiguate duplicates by a panel row at t (played season t or rookie)
        b = rest.merge(keys, on="name_id").merge(
            panel[panel.t == t][["gsis_id"]].assign(ok=1), on="gsis_id", how="left")
        b = b.sort_values("ok", na_position="last").drop_duplicates("name_id").drop(columns="ok")
        m = pd.concat([a, b], ignore_index=True).drop_duplicates("gsis_id")
        n_raw += len(x); n_hit += len(m)
        for col, src in DD_SOURCES.items():
            v = pd.to_numeric(m[col], errors="coerce").where(lambda s: s > 0)
            if v.notna().sum() < 100:  # source not live yet at this date (placeholders)
                continue
            frames[src].append(pd.DataFrame({"gsis_id": m.gsis_id, "t": t, "mkt": -v, "snap_date": date}).dropna())
    print(f"  Dynasty Daddy id match: {n_hit}/{n_raw} = {n_hit / n_raw:.1%}")
    return {f"dd_{k}": pd.concat(v, ignore_index=True) for k, v in frames.items() if v}


# ---------------------------------------------------------------- groupings
def add_groups(x: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["mkt_rank"] = x.groupby("t").mkt.rank(method="first")
    x["mkt_pos_rank"] = x.groupby(["t", "position"]).mkt.rank(method="first")

    # last-season per-game finish at position (players with >= 6 games)
    s = seasons[seasons.games >= 6].copy()
    s["ppg_pos_rank"] = s.groupby(["season", "position"]).ppg.rank(ascending=False, method="first")
    x = x.merge(s[["gsis_id", "season", "ppg_pos_rank"]].rename(columns={"season": "t"}),
                on=["gsis_id", "t"], how="left")

    g = pd.DataFrame(index=x.index)
    g["position"] = x.position
    age = x.age
    g["age_band"] = pd.cut(age, [0, 23.5, 25.5, 27.5, 29.5, 99],
                           labels=["<=23", "24-25", "26-27", "28-29", "30+"]).astype(str)
    g["pos_x_age"] = x.position + " " + pd.cut(age, [0, 24.5, 27.5, 99],
                                               labels=["young(<=24)", "prime(25-27)", "old(28+)"]).astype(str)
    g["experience"] = pd.cut(x.exp, [-1, 0, 1, 2, 5, 99],
                             labels=["rookie", "year2", "year3", "year4-6", "year7+"]).astype(str)
    g["draft_round"] = np.select(
        [x.drafted == 0, x.draft_round_f == 1, x.draft_round_f == 2, x.draft_round_f == 3],
        ["UDFA", "R1", "R2", "R3"], "R4-7")
    g["prior_games"] = np.where(x.row_type == "rookie", "rookie",
                                pd.cut(x.games.fillna(0), [-1, 9, 13, 99],
                                       labels=["<=9 games", "10-13", "14+"]).astype(str))
    gap = x.mkt_pos_rank - x.ppg_pos_rank  # >0: market ranks him worse than last year's per-game finish
    g["mkt_vs_last_year"] = np.select(
        [x.ppg_pos_rank.isna(), gap >= 10, gap <= -10],
        ["n/a (rookie/<6g)", "market fades last yr (>=10 spots)", "market bullish vs last yr (>=10)"],
        "aligned (+/-9)")
    g["breakout_last_yr"] = np.where(
        x.row_type == "rookie", "rookie",
        np.where((x.vorp > 0) & (x.vorp_prev.fillna(0) == 0), "first VORP season", "other"))
    qb = x.position == "QB"
    g["qb_rushing"] = np.where(~qb, "non-QB", np.where(x.rush_yds_pg >= 25, "QB rush >=25 yd/g",
                                                       np.where(x.row_type == "rookie", "QB rookie",
                                                                "QB pocket")))
    g["rb_receiving"] = np.where(x.position != "RB", "non-RB",
                                 np.where(x.targets_pg >= 4, "RB >=4 tgt/g",
                                          np.where(x.row_type == "rookie", "RB rookie", "RB <4 tgt/g")))
    g["market_tier"] = pd.cut(x.mkt_rank, [0, 24, 60, 120, TOP_N],
                              labels=["1-24", "25-60", "61-120", "121-200"]).astype(str)
    if "tm" in x and x.tm.notna().any():
        changed = (x.tm.notna() & x.recent_team.notna()
                   & (x.tm.replace({"JAC": "JAX", "LA": "LAR", "FA": "FA"}) != x.recent_team)
                   & (x.row_type == "veteran"))
        g["offseason_team_change"] = np.where(x.row_type == "rookie", "rookie",
                                              np.where(x.tm == "FA", "free agent (unsigned)",
                                                       np.where(changed, "new team", "same team")))
    if "sd" in x and x.sd.notna().sum() >= 100:
        # expert disagreement relative to players ranked nearby (SD grows with rank); rows without SD -> NaN group
        ok = x.sd.notna()
        lsd, lr = np.log(x.sd[ok].clip(lower=0.3)), np.log(x.mkt_rank[ok])
        resid_sd = lsd - np.poly1d(np.polyfit(lr, lsd, 2))(lr)
        g["expert_disagreement"] = pd.qcut(resid_sd, 3, labels=["low", "mid", "high"]).astype(str).reindex(x.index)
    return x, g


# ---------------------------------------------------------------- analysis
def residuals(x: pd.DataFrame, outcome: str) -> pd.Series:
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(x.mkt_rank, x[outcome])
    return x[outcome] - iso.predict(x.mkt_rank)


def summarize(x, g, market, outcome, n_boot=500):
    rows = []
    players = x.gsis_id.values
    uniq = np.unique(players)
    idx_by_player = pd.Series(np.arange(len(x))).groupby(players).apply(np.array)
    boot_idx = [np.concatenate(idx_by_player.loc[RNG.choice(uniq, len(uniq))].values) for _ in range(n_boot)]
    r = x.resid.values
    scale = x[outcome].mean()
    for col in g.columns:
        lev = g[col].values
        for level in pd.unique(lev):
            mask = lev == level
            n = mask.sum()
            if n < 25:
                continue
            mean = r[mask].mean()
            bs = [r[b][mask[b]].mean() for b in boot_idx if mask[b].any()]
            by_year = pd.Series(r[mask]).groupby(x.t.values[mask]).mean()
            by_year = by_year[pd.Series(mask).groupby(x.t.values).sum().loc[by_year.index] >= 5]
            rows.append({
                "market": market, "outcome": outcome, "grouping": col, "level": level, "n": int(n),
                "players": len(np.unique(players[mask])),
                "mean_resid": mean, "ci05": np.percentile(bs, 5), "ci95": np.percentile(bs, 95),
                "resid_pct_of_avg_outcome": mean / scale,
                "years": len(by_year), "years_same_sign": float((np.sign(by_year) == np.sign(mean)).mean()),
            })
    return pd.DataFrame(rows)


def main():
    (OUT / "market_rows").mkdir(parents=True, exist_ok=True)
    print(f"League: {'superflex' if SF else '1QB'}; value = {VALUE}; outputs -> {OUT}")
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    seasons = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    outc = outcomes()

    markets = all_markets(panel)
    allres, coverage = [], []
    for name, m in markets.items():
        horizons = [1] if name in REDRAFT else HORIZONS
        for h in horizons:
            x = prepare(m, panel, outc, seasons, h)
            if x is None:
                continue
            x, g = x
            outcome = f"vorp{h}"
            coverage.append({"market": name, "horizon": h, "snapshots": f"{x.t.min()}-{x.t.max()}",
                             "cohorts": x.t.nunique(), "rows": len(x)})
            res = summarize(x, g, name, outcome).assign(horizon=h)
            allres.append(res)
            x.assign(**{f"g_{c}": g[c] for c in g}).to_parquet(
                OUT / "market_rows" / f"{name}_h{h}.parquet", index=False)
    cov = pd.DataFrame(coverage)
    print(cov.pivot(index="market", columns="horizon", values="cohorts").fillna(0).astype(int).to_string())
    cov.to_csv(OUT / "market_coverage.csv", index=False)
    res = pd.concat(allres, ignore_index=True)
    res["robust"] = ((np.sign(res.ci05) == np.sign(res.ci95)) & (res.years >= 3)
                     & (res.years_same_sign >= 0.7))
    res.round(3).to_csv(OUT / "market_residuals_by_group.csv", index=False)

    # cross-market summary per horizon (dynasty markets): median effect, sign agreement
    dyn = res[~res.market.isin(REDRAFT) & (res.grouping != "market_tier")].copy()
    dyn["pct"] = dyn.resid_pct_of_avg_outcome * 100

    def agg(s):
        med = s.pct.median()
        same = (np.sign(s.pct) == np.sign(med)).sum()
        return f"{med:+.0f} ({same}/{len(s)}{', ' + str(int(s.robust.sum())) + '*' if s.robust.any() else ''})"

    summ = dyn.groupby(["grouping", "level", "horizon"]).apply(agg, include_groups=False).unstack("horizon")
    summ.to_csv(OUT / "market_residuals_summary_by_horizon.csv")
    print("\nDynasty markets: median residual % across markets "
          "(markets agreeing in sign / markets available, * = robust within market):")
    print(summ.fillna("").to_string())


REDRAFT = {"fp_redraft", "ffc_redraft"}


def all_markets(panel) -> dict[str, pd.DataFrame]:
    m = {"fp_dynasty": fp_dynasty(panel), "mfl_dynasty": mfl_dynasty(panel)}
    m.update(dd_markets(panel))
    m.update({"fp_redraft": fp_redraft(), "ffc_redraft": ffc_redraft(panel)})
    return m


def prepare(m, panel, outc, seasons, h, top_n=TOP_N):
    """Join a market to snapshot features + complete h-year outcomes; keep its top-N per cohort."""
    outcome = f"vorp{h}"
    x = (m.merge(panel.drop(columns=["next_vorp"]), on=["gsis_id", "t"], how="inner")
         .merge(outc, on=["gsis_id", "t"], how="left"))
    x = x[x[outcome].notna()]
    if x.empty:
        return None
    x = x[x.groupby("t").mkt.rank(method="first") <= top_n].reset_index(drop=True)
    x, g = add_groups(x, seasons)
    x["resid"] = residuals(x, outcome)
    return x, g


if __name__ == "__main__":
    main()
