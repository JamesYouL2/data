"""Q: do the site's calls still work if trades clear at trade-market prices, and should the calls read those prices?

rank_input_sim.py's league, cohorts with market prices (t = 2020-24, seasons 2021-25; FantasyCalc 2021-24). Every trade there
clears at the in-season FP rank. If FP lags the market, some of its gains come from prices no one would pay.
Market prices: Dynasty Daddy's archive on the dates of the FP snapshots at weeks 4, 8 and 12 (fetch_dynastydaddy.py), as the
rank among QB / RB / WR / TE by value. SOURCE=ktc (default) or fantasycalc, in the league's format.
Two price worlds, same drafts, same random numbers:
  f:  trades clear at the FP rank, players off the FP top POOL aren't tradeable, value scored at FP ranks at the next Mays
  m:  trades clear at the market rank, players off the market top POOL aren't tradeable, value scored at market May ranks
Scoring each world in its own currency matters: buying at the market price and scoring at FP ranks would count every FP-market
gap as a free gain. Title odds come from points and don't depend on either.
Arms (all as rank_input_sim.py's "live" apart from what's named; the market's rank also feeds rules.situational's mr, as on
the site):
  live      the site's calls from FP ranks
  nosig     no signals
  noqbx     no QB exemption
  noyoungx  no young-hurt exemption
  mrank     (m only) classify on the market rank, tilted to the FP ROS rank; situational and signals on FP
  msig      (m only) classify on FP; situational and signals on the market rank (dynasty fall since Sept 1 and the young gap;
            the ROS fall stays FP's, as the market has no ROS list here)
  mall      (m only) both
  X~        placebo: live~ and mall~ as many sells and buys on random players; noqbx~ and noyoungx~ as many sells of random
            own players of the same kind (rank_input_sim.py)
Caveats: the rules were fit on FP May ranks over these cohorts, which favours FP-input arms. Few cohorts: value 3 years on
exists for 3 cohorts at market ranks and 2 at FP ranks; 3-year titles for 3. Market lists are shorter than 340 in the early
years (FantasyCalc about 300 in 2022), so fewer trades clear there.
The committed outputs ran on the sell exemptions from before they were cut back in superflex (rules.apply_signal).
Writes outputs/[sf/]market_price_sim_<source>.csv.gz. LEAGUE=sf for superflex; REPS (default 10), WEEKS (default 4,8,12), JOBS.
"""
import json
import os
import pathlib
import sys
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import inseason_sim as isim  # noqa: E402  (sets POOL = 340 for middle_mode)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rank_input_sim as rs  # noqa: E402
from common import HERE, out_dir  # noqa: E402
import middle_mode as mm  # noqa: E402
import startup as su  # noqa: E402
import win_now as wn  # noqa: E402
from long_horizon import SF  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

R, F = mm.R, mm.F
REPS, WEEKS, JOBS = isim.REPS, isim.WEEKS, isim.JOBS
POOL, H = mm.POOL, mm.H
LAST_WK, REG = isim.LAST_WK, isim.REG
MODES = rs.MODES
SOURCE = os.environ.get("SOURCE", "ktc")
F_ARMS = ("live", "nosig", "noqbx", "noyoungx")
M_ARMS = F_ARMS + ("mrank", "msig", "mall")
RANDOM = ("live", "mall")
KIND = ("noqbx", "noyoungx")
LENS = {"1-3": ("con", [1, 2, 3]), "4-8": ("mid", range(4, 9)), "9-12": ("reb", range(9, 13))}
COHORTS = range(2020, 2025)
OUT = "market_price_sim"
UNLISTED = 500          # preseason rank of a player off the market's Sept 1 list, as in inseason_movers.py


def market_ranks(panel):
    """snap_date -> frame indexed by gsis_id: rank, prank among QB / RB / WR / TE at the source's value that day"""
    import market_residuals as M
    pos = panel.sort_values("t").groupby("gsis_id").position.last()
    out = {}
    for pat in ("simweek_*.json", "inseason_*-09-01.json", "snapshot_*.json"):
        v = M.dd_markets(panel, pat).get(f"dd_{SOURCE}")
        if v is None:
            continue
        v = v.assign(position=v.gsis_id.map(pos)).dropna(subset=["position"])
        for d, z in v.groupby("snap_date"):
            z = z.assign(rank=z.mkt.rank(method="first"), prank=z.groupby("position").mkt.rank(method="first"))
            out[d] = z.set_index("gsis_id")[["rank", "prank"]]
    return out


def market_rows(q, rows, mk, mkpre):
    """feature rows with the market rank as mr; and with the market rank in place of every dynasty rank"""
    mr = q.gsis_id.map(mk["rank"]).to_numpy(float)
    mp = q.gsis_id.map(mk["prank"]).to_numpy(float)
    pr = q.gsis_id.map(mkpre["rank"]).fillna(UNLISTED).to_numpy(float)
    rows = [x._replace(**{f"mr_{F}": m}) for x, m in zip(rows, mr)]
    mx = []
    for x, a, b, c in zip(rows, mr, mp, pr):                 # off the market's list: the FP rank, and he can't trade at m prices
        if a == a:
            x = x._replace(**{f"r_{F}": a, f"pr_{F}": b, "do_now": a, "dsf_now": a, "do_pre": c, "dsf_pre": c})
        mx.append(x)
    return rows, mx


def snapshot_calls(q, last, w, mk, mkpre):
    """(arm, mode) -> {gsis_id: kind}; market rows mk and preseason mkpre are indexed by gsis_id"""
    rows, inputs = rs.features(q, last)
    has = q.ros_rank.notna().any()
    y = int(q.season.iloc[0])
    base = {"ROOKIE": y, "LOOK": rs.LOOK, "EARLY": False}
    live = {**base, "WEEK": w + 1}
    rows, mx = market_rows(q, rows, mk, mkpre)
    mtilt = [x._replace(**{f"r_{F}": getattr(x, f"r_{F}") * k, f"pr_{F}": getattr(x, f"pr_{F}") * k}) for x, k in
             zip(mx, (R.ros_tilt(getattr(x, f"pr_{F}"), x.ros_p) if has else 1.0 for x in mx))]
    spec = {"live": (inputs["tilt"], rows, live, "live"), "nosig": (inputs["tilt"], rows, base, "live"),
            "noqbx": (inputs["tilt"], rows, live, "noqbx"), "noyoungx": (inputs["tilt"], rows, live, "noyoungx"),
            "mrank": (mtilt, rows, live, "live"), "msig": (inputs["tilt"], mx, live, "live"), "mall": (mtilt, mx, live, "live")}
    out = {}
    for arm, (cl, si, ctx, sig) in spec.items():
        res = {m: {} for m in MODES}
        with rs.signal_rules(sig):
            for g, xs, xc in zip(q.gsis_id, si, cl):
                c = R.situational(xs, F, R.classify(xc, F, ctx), ctx)
                res["reb"][g], res["con"][g], res["mid"][g] = rs.act_kind(c[1]), rs.act_kind(c[3]), rs.act_kind(c[7])
        for m in MODES:
            out[arm, m] = res[m]
    return out


def run_cohort(args):
    t, g, snaps, calls, wkp, pts, rank_at, mrank_at, proj_by_w, seed = args
    rng = np.random.default_rng(seed)
    Y = t + 1
    pos, age = g.position.to_numpy(), g.age.to_numpy()
    yrs = [y for y in range(Y + 1, Y + H) if y <= su.LAST]
    P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1) if yrs else np.zeros((len(g), 0))

    def at(table, k):
        return g.gsis_id.map(table[k]).to_numpy(float) if k in table else None
    rk1, rk3, mk1, mk3 = at(rank_at, t + 1), at(rank_at, t + 3), at(mrank_at, t + 1), at(mrank_at, t + 3)

    def capped(a):
        return np.full(len(g), np.nan) if a is None else np.where(a <= POOL, a, np.nan)
    nxt = {"f": capped(rk1), "m": capped(mk1)}
    out = []
    for rep in range(REPS):
        team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
        strength = np.array([su.lineup(g.proj.to_numpy()[team == i], pos[team == i]) for i in range(su.TEAMS)])
        rkk = (-strength).argsort().argsort() + 1
        for w_target, m in snaps.items():
            w = int(m.w.dropna().iloc[0])
            wk = wkp[Y]
            pre = g.gsis_id.map(wk.loc[:, wk.columns <= w].sum(axis=1)).fillna(0).to_numpy()
            post = g.gsis_id.map(wk.loc[:, (wk.columns > w) & (wk.columns <= LAST_WK)].sum(axis=1)).fillna(0).to_numpy()
            Lpre = np.array([su.lineup(pre[team == k], pos[team == k]) for k in range(su.TEAMS)]) / w
            sd = wn.SD * (Lpre.mean() * w + np.array([su.lineup(post[team == k], pos[team == k])
                                                      for k in range(su.TEAMS)]).mean()) / LAST_WK
            worlds = {"f": {"price": m["rank"].to_numpy(float), "age": age, "pos": pos, "proj": proj_by_w[w_target]},
                      "m": {"price": capped(m["mrank"].to_numpy(float)), "age": age, "pos": pos, "proj": proj_by_w[w_target]}}
            young = m.acls.isin(["rookie", "25-"]).to_numpy()
            seeds = rng.integers(1 << 31, size=H)
            kinds = {key: np.array([v.get(x, "none") for x in g.gsis_id], dtype="U5") for key, v in calls[w_target].items()}
            memo = {}

            def evaluate(T):
                key = T.tobytes()
                if key not in memo:
                    Lpost = np.array([su.lineup(post[T[:, 0] == k], pos[T[:, 0] == k]) for k in range(su.TEAMS)]) / (LAST_WK - w)
                    p = [isim.titles_split(Lpre, Lpost, min(w, REG), sd, np.random.default_rng(seeds[0]))]
                    if P.shape[1]:
                        p += list(mm.season_titles(T[:, 1:1 + P.shape[1]], P, pos, seeds[1:]))
                    p = np.array(p)
                    memo[key] = np.pad(p, ((0, H - len(p)), (0, 0)), constant_values=np.nan)
                return memo[key]

            def val(T, col, rk, i):
                return np.nansum(np.maximum(0, max(200, POOL) - rk[T[:, col] == i])) if rk is not None else np.nan

            def row(T, i, arm, mode, n):
                return (t, rep, w_target, rkk[i], arm, mode, *evaluate(T)[:, i], val(T, 1, rk1, i), val(T, 2, rk3, i),
                        val(T, 1, mk1, i), val(T, 2, mk3, i), *n)
            T0 = np.repeat(team[:, None], H, 1)
            for i in range(su.TEAMS):
                out.append(row(T0, i, "none", "-", (0, 0, 0)))
                for mode in MODES:
                    for world, arms in (("f", F_ARMS), ("m", M_ARMS)):
                        S, n1 = worlds[world], nxt[world]
                        ok = ~np.isnan(S["price"])
                        for arm in arms:
                            k = kinds[arm, mode]
                            T, n = rs.policy(i, mode, team, k, S, n1)
                            out.append(row(T, i, f"{world}:{arm}", mode, n))
                            if arm in RANDOM:
                                sell, buy = isim.placebo(i, team, k == "sell", k == "buy", ok, rng)
                                kp = np.where(sell, "sell", np.where(buy, "buy", "hold"))
                            elif arm in KIND:
                                kl = kinds["live", mode]
                                new = (k == "sell") & (kl != "sell")
                                kp = np.where(new, kl, k)
                                same = pos == "QB" if arm == "noqbx" else (pos != "QB") & young
                                c = np.flatnonzero((team == i) & ok & same & ~new & ~np.isin(kp, ("sell", "cut")))
                                kp[rng.choice(c, size=min(int((new & (team == i)).sum()), len(c)), replace=False)] = "sell"
                            else:
                                continue
                            T, n = rs.policy(i, mode, team, kp, S, n1)
                            out.append(row(T, i, f"{world}:{arm}~", mode, n))
    print(f"cohort {t} done", flush=True)
    return out


def diagnostics(x, mkt):
    """how far the market sits from FP at the snapshots, and whether it has already marked injured players down"""
    rows = []
    for (y, tg), q in x[x.season >= COHORTS[0] + 1].groupby(["season", "target"]):
        d = str(q.date.iloc[0])
        if d not in mkt or tg not in WEEKS:
            continue
        pos = q[(q.ros_rank != 500) | q.ros_rank.isna()].copy()
        pos["mr"] = pos.gsis_id.map(mkt[d]["rank"])
        pos = pos[pos.mr.notna() & (pos["rank"] <= 200)]
        lr = np.log(pos.mr / pos["rank"])
        yg = pos.acls.isin(["rookie", "25-"]) & (pos.position != "QB")
        rows.append({"season": y, "wk": tg, "n": len(pos), "spearman": spearmanr(pos["rank"], pos.mr).statistic,
                     "young hurt": np.exp(lr[pos.hurt & yg].mean()), "n yh": int((pos.hurt & yg).sum()),
                     "older hurt": np.exp(lr[pos.hurt & ~yg & (pos.position != "QB")].mean()),
                     "healthy": np.exp(lr[~pos.hurt & (pos.position != "QB")].mean()),
                     "young hurt faller": np.exp(lr[(pos.kind == "faller: hurt") & yg].mean()),
                     "QB": np.exp(lr[pos.position == "QB"].mean())})
    d = pd.DataFrame(rows)
    print(f"\n== {SOURCE} rank / FP rank, geometric mean (below 1 = the market prices him higher), FP top 200, off-ROS-list dropped")
    print(d.round(2).to_string(index=False))
    print("mean:", d.drop(columns=["season", "wk"]).mean().round(2).to_dict())


def main():
    import inseason_movers as im
    rs.LOOK.update({(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"]) for r in
                    json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]})
    x = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    wk = im.weekly()
    wkp = {y: z.pivot_table(index="gsis_id", columns="week", values="ppr", aggfunc="sum").fillna(0)
           for y, z in wk.groupby("season")}
    proj = isim.ros_projector(x, wk)
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    mkt = market_ranks(panel)
    diagnostics(x, mkt)
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    pr6 = mm.prod_ranks(s)[6]
    d = mm.pool()
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    mrank_at = {y - 1: mkt[f"{y}-05-15"]["rank"] for y in range(2021, 2027) if f"{y}-05-15" in mkt}   # t's May is t + 1's
    d = d[d.t <= 2024]
    pproj = mm.projector(pts, d)
    rng = np.random.default_rng(13)
    jobs = []
    for t in COHORTS:
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
        g["proj"] = pproj(g)
        snaps, pbw, calls = {}, {}, {}
        sep = mkt.get(f"{t + 1}-09-01")
        for w_target in WEEKS:
            q = x[(x.season == t + 1) & (x.target == w_target)]
            if q.empty or int(q.w.iloc[0]) >= LAST_WK or sep is None or str(q.date.iloc[0]) not in mkt:
                continue
            mk = mkt[str(q.date.iloc[0])]
            m = g[["gsis_id", "position"]].merge(q.drop(columns="position"), on="gsis_id", how="left")
            m["w"] = int(q.w.iloc[0])
            m["mrank"] = m.gsis_id.map(mk["rank"])
            snaps[w_target], pbw[w_target] = m, proj(m)
            calls[w_target] = snapshot_calls(q, pr6.get(t, {}), int(q.w.iloc[0]), mk, sep)
        if snaps:
            jobs.append((t, g, snaps, calls, wkp, pts, rank_at, mrank_at, pbw, int(rng.integers(1 << 31))))
    print(f"cohorts {[j[0] for j in jobs]}", flush=True)
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = (["t", "rep", "wk", "rank", "arm", "mode"] + [f"p{k}" for k in range(1, H + 1)]
            + ["val1", "val3", "mval1", "mval3", "sell", "later", "buy"])
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / f"{OUT}_{SOURCE}.csv.gz", index=False)
    summarize(df)


def summarize(df):
    key = ["t", "rep", "wk", "rank"]
    pc = [f"p{k}" for k in range(1, H + 1)]
    vals = ["val1", "val3", "mval1", "mval3"]
    base = df[df.arm == "none"].set_index(key)[pc + vals].add_suffix("_0")
    z = df[df.arm != "none"].join(base, on=key)
    z["yr1"] = (z.p1 - z.p1_0) * 100
    z["yrs3"] = np.where(z.t + 3 <= su.LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in (1, 2, 3)) * 100, np.nan)
    fw = z.arm.str.startswith("f:")
    for a, b, c in (("v1", "val1", "mval1"), ("v3", "val3", "mval3")):
        z[a] = np.where(fw, z[b] - z[b + "_0"], z[c] - z[c + "_0"])     # each world in its own currency
    z["trades"] = z.sell + z.later + z.buy
    z["bothv1"] = np.where(z.val1_0.notna() & z.mval1_0.notna(), z.v1, np.nan)
    z["bothv3"] = np.where(z.val3_0.notna() & z.mval3_0.notna(), z.v3, np.nan)
    idx = z.set_index(key + ["mode", "arm"])
    mets = ["yr1", "yrs3", "v1", "v3"]
    pd.set_option("display.width", 250, "display.max_rows", 800, "display.max_columns", 40)

    def sel(arm, cols):
        return idx.xs(arm, level="arm")[cols]

    def cell(d):
        v = d.dropna()
        if v.empty:
            return ""
        c = v.groupby(level=["t", "rep", "wk"]).mean()
        by = v.groupby(level="t").mean()
        return f"{c.mean():+6.1f}±{1.96 * c.std(ddof=1) / np.sqrt(len(c)):4.1f} {int((by > 0).sum())}/{int((by != 0).sum())}"

    def table(title, a, b=None, cols=mets, trades=False):
        d = sel(a, cols) if b is None else sel(a, cols) - sel(b, cols)
        rows = []
        for grp, (mode, ranks) in LENS.items():
            q = d.xs(mode, level="mode")
            q = q[np.isin(q.index.get_level_values("rank"), ranks)]
            r = {"ranks": grp, "mode": mode, **{c: cell(q[c]) for c in cols}}
            if trades:
                n = sel(a, ["trades"]).xs(mode, level="mode")
                r["trades"] = round(n[np.isin(n.index.get_level_values("rank"), ranks)].trades.mean(), 1)
            rows.append(r)
        print(f"\n-- {title}")
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n== {'Superflex' if SF else '1QB'}, {SOURCE}, cohorts {sorted(z.t.unique())}, weeks {WEEKS}, {REPS} reps. Own-mode teams. "
          "mean ± 95% CI, cohorts above zero / cohorts that differ. Titles in points; v1, v3 = value at the May 1 and 3 years on, "
          "in the world's own ranks")
    print("\n=== 1. Do the calls survive market prices? Against not trading; v on cohorts with both FP and market Mays")
    both = ["yr1", "yrs3", "bothv1", "bothv3"]
    for arm in ("live", "nosig", "live~"):
        table(f"f:{arm} - no trades", f"f:{arm}", cols=both, trades=True)
        table(f"m:{arm} - no trades", f"m:{arm}", cols=both, trades=True)
    print("\n=== 2. What the signals and the calls are worth at each price")
    table("f: live - nosig", "f:live", "f:nosig")
    table("m: live - nosig", "m:live", "m:nosig")
    table("f: live - placebo", "f:live", "f:live~")
    table("m: live - placebo", "m:live", "m:live~")
    print("\n=== 3. Exemptions at each price (positive = removing the exemption is better)")
    for arm in KIND:
        for w in ("f", "m"):
            table(f"{w}: {arm} - live", f"{w}:{arm}", f"{w}:live")
            table(f"{w}: {arm} - random same-kind sells", f"{w}:{arm}", f"{w}:{arm}~")
    print("\n=== 4. Should the calls read the market? All at market prices")
    for arm in ("mrank", "msig", "mall"):
        table(f"m: {arm} - live", f"m:{arm}", "m:live", trades=True)
    table("m: mall - its placebo", "m:mall", "m:mall~")
    print("\n=== 5. By cohort, own-mode teams")
    for a, b in (("m:live", "m:nosig"), ("m:live", "m:live~"), ("m:noyoungx", "m:live"), ("m:noyoungx", "m:noyoungx~"),
                 ("m:noqbx", "m:live"), ("m:noqbx", "m:noqbx~"), ("m:mall", "m:live")):
        d = sel(a, mets) - sel(b, mets)
        parts = []
        for grp, (mode, ranks) in LENS.items():
            q = d.xs(mode, level="mode")
            q = q[np.isin(q.index.get_level_values("rank"), ranks)]
            parts.append(q.groupby(level="t").mean().round(1).assign(group=grp))
        print(f"\n-- {a} - {b}")
        print(pd.concat(parts).reset_index().set_index(["group", "t"]).to_string())


if __name__ == "__main__":
    main()
