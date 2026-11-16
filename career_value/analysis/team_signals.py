"""Q: which says most about whether to buy for this season: your record, your rest-of-season strength or your dynasty value?

inseason_sim.py's league (12 teams draft the FP dynasty top POOL of cohort t, real PPR scoring) at one window in season Y = t + 1,
after week w = 4, 8 or 12. Each team then has three measures, each a z-score within its league:
  record  wins through week w. Weeks 1..w are played DRAWS times under the title model (the team's real points to date per week,
          plus weekly noise, random schedule), so the same team is seen with a lucky and an unlucky record
  ros     projected rest-of-season lineup points (inseason_sim.ros_projector: FP rest-of-season rank -> points, by position)
  value   dynasty value, the sum of POOL - in-season FP dynasty rank over the roster (the site's roster-strength measure)
Outcomes, title odds in points:
  now     P(title) in season Y given the banked wins and points: weeks w+1..14 and the playoffs at the real rest-of-season lineup
  later   P(title) summed over seasons Y+1 and Y+2, roster frozen
  and the change in each from a win-now trade: up to NTRADE swaps, each the one that most raises the projected rest-of-season
  lineup, for a player the in-season FP dynasty rank puts level with or up to 20% below the one given (so the team never takes
  back more dynasty value than it gives); and the dynasty value it costs by the next May.
Reported as the change per 1 SD of each measure with the other two held fixed (OLS, 95% CI from a bootstrap over leagues), and
by thirds of each measure.
Caveats: FP prices only; rosters otherwise frozen, so no waivers or rookies; one team trades; record's noise is the title model's
(weekly SD 0.2 of the league mean), and real schedules are no luckier or unluckier than that only if the model is right.
Writes outputs/[sf/]team_signals.csv.gz. LEAGUE=sf for superflex; REPS (default 10), WEEKS, DRAWS (default 6), NTRADE (default 3), JOBS.
"""
import os
import pathlib
import sys
from multiprocessing import Pool

os.environ.setdefault("POOL", "340")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import HERE, out_dir  # noqa: E402
import middle_mode as mm  # noqa: E402
import startup as su  # noqa: E402
import win_now as wn  # noqa: E402
import inseason_sim as ins  # noqa: E402
from long_horizon import SF  # noqa: E402

REPS = int(os.environ.get("REPS", 10))
WEEKS = tuple(int(w) for w in os.environ.get("WEEKS", "4,8,12").split(","))
JOBS = int(os.environ.get("JOBS", 6))
DRAWS = int(os.environ.get("DRAWS", 6))
NTRADE = int(os.environ.get("NTRADE", 3))
POOL, REG, LAST_WK, WIN = mm.POOL, ins.REG, ins.LAST_WK, ins.WIN
MEAS = ["record", "ros", "value"]
DECAY = float(os.environ.get("DECAY", 60))                 # top-heavy dynasty value: sum of exp(-rank / DECAY); see decay()
TAUS = (10, 15, 20, 30, 45, 60, 90, 120, 180, 300)


def play(wins, pf, m, weeks, sd, rng):
    """add `weeks` regular-season weeks at per-week means m to wins / pf (sims x teams)"""
    NS, n = wins.shape
    for _ in range(weeks):
        perm = np.argsort(rng.random((NS, n)), 1)
        sc = m + sd * rng.standard_normal((NS, n))
        a, b = perm[:, 0::2], perm[:, 1::2]
        sa, sb = np.take_along_axis(sc, a, 1), np.take_along_axis(sc, b, 1)
        np.put_along_axis(wins, a, np.take_along_axis(wins, a, 1) + (sa > sb), 1)
        np.put_along_axis(wins, b, np.take_along_axis(wins, b, 1) + (sb > sa), 1)
        pf += sc
    return wins, pf


def finish(wins0, pf0, Lpost, w, sd, rng):
    """P(title) per team given banked wins / points after week w, the rest at Lpost"""
    n = len(Lpost)
    wins, pf = play(np.tile(wins0, (wn.NSIM, 1)), np.tile(pf0, (wn.NSIM, 1)), Lpost, REG - w, sd, rng)
    seed = np.argsort(-(wins * 1e4 + pf), 1)[:, :6]

    def game(x, y):
        sx = Lpost[x] + sd * rng.standard_normal(len(x))
        sy = Lpost[y] + sd * rng.standard_normal(len(y))
        return np.where(sx > sy, x, y)
    q1, q2 = game(seed[:, 2], seed[:, 5]), game(seed[:, 3], seed[:, 4])
    ch = game(game(seed[:, 0], q2), game(seed[:, 1], q1))
    return np.bincount(ch, minlength=n) / wn.NSIM


def win_now_trade(i, team, price, pj, pos):
    """up to NTRADE equal-price swaps, each the one that most raises team i's projected rest-of-season lineup"""
    cur, moved, n = team.copy(), np.zeros(len(team), bool), 0
    ok = ~np.isnan(price)
    for _ in range(NTRADE):
        mine = np.flatnonzero(cur == i)
        base, best = su.lineup(pj[mine], pos[mine]), (1e-9, None, None)
        for j in mine[ok[mine] & ~moved[mine]]:
            rest = mine[mine != j]
            for x in np.flatnonzero((cur >= 0) & (cur != i) & ~moved & ok & (price >= price[j]) & (price <= price[j] * (1 + WIN))):
                if pj[x] <= pj[j] and pos[x] == pos[j]:
                    continue
                ids = np.append(rest, x)
                gain = su.lineup(pj[ids], pos[ids]) - base
                if gain > best[0]:
                    best = (gain, j, x)
        if best[1] is None:
            break
        _, j, x = best
        cur[j], cur[x] = cur[x], i
        moved[[j, x]] = True
        n += 1
    return cur, n


def run_cohort(args):
    t, g, snaps, wkp, pts, rank_at, proj_by_w, seed = args
    rng = np.random.default_rng(seed)
    Y = t + 1
    pos = g.position.to_numpy()
    yrs = [y for y in (Y + 1, Y + 2) if y <= su.LAST]
    P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1) if yrs else np.zeros((len(g), 0))
    rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
    out = []

    def lus(vals, T):
        return np.array([su.lineup(vals[T == k], pos[T == k]) for k in range(su.TEAMS)])

    for rep in range(REPS):
        team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
        may = (-lus(g.proj.to_numpy(), team)).argsort().argsort() + 1
        for w_target in WEEKS:
            m = snaps.get(w_target)
            if m is None:
                continue
            w = min(int(m.w.dropna().iloc[0]), REG - 1)
            wk = wkp[Y]
            pre = g.gsis_id.map(wk.loc[:, wk.columns <= w].sum(axis=1)).fillna(0).to_numpy()
            post = g.gsis_id.map(wk.loc[:, (wk.columns > w) & (wk.columns <= LAST_WK)].sum(axis=1)).fillna(0).to_numpy()
            Lpre = lus(pre, team) / w
            sd = wn.SD * (Lpre.mean() * w + lus(post, team).mean()) / LAST_WK
            price, pj = m["rank"].to_numpy(float), proj_by_w[w_target]
            ros = lus(pj, team)
            value = np.array([np.nansum(np.maximum(0, POOL - price[team == k])) for k in range(su.TEAMS)])
            seeds = rng.integers(1 << 31, size=3)

            def later(T):
                p = mm.season_titles(np.repeat(T[:, None], P.shape[1], 1), P, pos, seeds[1:]) if P.shape[1] else np.zeros((0, su.TEAMS))
                return np.pad(p, ((0, 2 - len(p)), (0, 0)), constant_values=np.nan)

            def val1(T, i):
                return np.nansum(np.maximum(0, POOL - rk1[T == i])) if rk1 is not None else np.nan

            Lpost0, late0 = lus(post, team) / (LAST_WK - w), later(team)
            arms = []
            for i in range(su.TEAMS):
                T, n = win_now_trade(i, team, price, pj, pos)
                arms.append((lus(post, T) / (LAST_WK - w), later(T)[:, i] if n else late0[:, i], val1(T, i) - val1(team, i), n))
            for d in range(DRAWS):
                w0, pf0 = play(np.zeros((1, su.TEAMS)), np.zeros((1, su.TEAMS)), Lpre, w, sd, rng)
                s = int(rng.integers(1 << 31))
                p0 = finish(w0[0], pf0[0], Lpost0, w, sd, np.random.default_rng(s))
                for i, (Lp, late, dv, n) in enumerate(arms):
                    p = finish(w0[0], pf0[0], Lp, w, sd, np.random.default_rng(s))[i] if n else p0[i]
                    out.append((t, rep, w_target, d, i, may[i], w0[0, i], Lpre[i], ros[i], value[i], p0[i], *late0[:, i],
                                p, *late, dv, n))
    print(f"cohort {t} done", flush=True)
    return out


def main():
    import inseason_movers as im
    x = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    wk = im.weekly()
    wkp = {y: z.pivot_table(index="gsis_id", columns="week", values="ppr", aggfunc="sum").fillna(0) for y, z in wk.groupby("season")}
    proj = ins.ros_projector(x, wk)
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    d = mm.pool()
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    d = d[d.t <= 2024]
    pproj = mm.projector(pts, d)
    rng = np.random.default_rng(41)
    jobs = []
    for t in range(2019 if SF else 2018, 2025):          # superflex 2018: the only in-season list looks corrupt
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["proj"] = pproj(g)
        snaps, pbw = {}, {}
        for w_target in WEEKS:
            q = x[(x.season == t + 1) & (x.target == w_target)]
            if q.empty or int(q.w.iloc[0]) >= LAST_WK:
                continue
            m = g[["gsis_id", "position"]].merge(q.drop(columns="position"), on="gsis_id", how="left")
            m["w"] = int(q.w.iloc[0])
            snaps[w_target], pbw[w_target] = m, proj(m)
        jobs.append((t, g, snaps, wkp, pts, rank_at, pbw, int(rng.integers(1 << 31))))
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "wk", "draw", "team", "may_rank", "wins", "pf", "ros", "value", "now0", "y2_0", "y3_0", "now1", "y2_1", "y3_1",
            "dval1", "n"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / "team_signals.csv.gz", index=False)
    summarize(df)


def summarize(df):
    pd.set_option("display.width", 250, "display.max_rows", 400)
    lg = ["t", "rep", "wk", "draw"]
    z = df.copy()
    for c, src in zip(MEAS + ["points", "may"], ["wins", "ros", "value", "pf", "may_rank"]):
        gb = z.groupby(lg)[src]
        z[c] = (z[src] - gb.transform("mean")) / gb.transform("std").replace(0, np.nan)
    z["may"] = -z["may"]
    z["now"], z["later"] = z.now0 * 100, (z.y2_0 + z.y3_0) * 100
    z["d_now"], z["d_later"] = (z.now1 - z.now0) * 100, (z.y2_1 + z.y3_1 - z.y2_0 - z.y3_0) * 100
    z["d_3yr"] = z.d_now + z.d_later
    outs = ["now", "later", "d_now", "d_later", "d_3yr", "dval1"]
    z = z.dropna(subset=MEAS)
    z["lgid"] = z.t * 100 + z.rep

    def fit(q, y, xs):
        q = q.dropna(subset=[y])
        X = np.column_stack([np.ones(len(q))] + [q[c].to_numpy() for c in xs])
        return np.linalg.lstsq(X, q[y].to_numpy(), rcond=None)[0][1:]

    def table(q, xs, title):
        ids = q.lgid.unique()
        parts = {k: v for k, v in q.groupby("lgid")}
        rng = np.random.default_rng(3)
        boots = [pd.concat([parts[k] for k in rng.choice(ids, len(ids))]) for _ in range(200)]
        rows = []
        for y in outs:
            b = fit(q, y, xs)
            bs = np.array([fit(bq, y, xs) for bq in boots])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            by = np.array([fit(g, y, xs) for _, g in q.dropna(subset=[y]).groupby("t")])
            rows.append({"outcome": y, "cohorts": len(by),
                         **{c: f"{b[k]:+6.1f} [{lo[k]:+.1f}, {hi[k]:+.1f}] {int((np.sign(by[:, k]) == np.sign(b[k])).sum())}/{len(by)}"
                            for k, c in enumerate(xs)}})
        print(f"\n-- {title}")
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"== {'Superflex' if SF else '1QB'}: cohorts {sorted(z.t.unique())}, {z.rep.nunique()} leagues each, weeks {sorted(z.wk.unique())}, "
          f"{z.draw.nunique()} records a league. Change per 1 SD of each measure, the others held fixed [95% CI over leagues] cohorts with "
          "that sign. now / later = title odds this season / the next two (points); d_ = the win-now trade's change; dval1 = its "
          f"dynasty value next May. Trades made: {df.n.mean():.2f} a team")
    print("correlations of the measures:")
    print(z[MEAS + ["points", "may"]].corr().round(2).to_string())
    table(z, MEAS, "all weeks")
    for w, q in z.groupby("wk"):
        table(q, MEAS, f"after week {w}")
    table(z, MEAS + ["points"], "all weeks, with points scored to date added")
    print("\n-- by thirds of each measure within the league (bottom 4 / middle 4 / top 4 teams), all weeks")
    for c in MEAS:
        z["third"] = z.groupby(lg)[c].rank(method="first").pipe(lambda r: pd.cut(r, [0, 4, 8, 12], labels=["bottom", "middle", "top"]))
        print(c)
        print(z.groupby("third", observed=True)[outs].mean().round(1).to_string())
    print("\n-- record x rest-of-season strength (thirds), all weeks: title odds now / the trade's change now / over 3 seasons")
    for c in ("record", "ros"):
        z[c + "3"] = z.groupby(lg)[c].rank(method="first").pipe(lambda r: pd.cut(r, [0, 4, 8, 12], labels=["bottom", "middle", "top"]))
    print(z.pivot_table(index="record3", columns="ros3", values=["now", "d_now", "d_3yr"], observed=True).round(1).to_string())


def may_cohort(args):
    t, g, P, seed = args
    rng = np.random.default_rng(seed)
    pos, pj, rk, age = g.position.to_numpy(), g.proj.to_numpy(), g.mkt_rank.to_numpy(float), g.age.to_numpy(float)
    w = np.exp(-rk / DECAY)
    out = []
    for rep in range(REPS * 4):
        team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
        p = mm.season_titles(np.repeat(team[:, None], P.shape[1], 1), P, pos, rng.integers(1 << 31, size=P.shape[1]))
        for i in range(su.TEAMS):
            m = team == i
            out.append((t, rep, i, su.lineup(pj[m], pos[m]), np.maximum(0, POOL - rk[m]).sum(), np.exp(-rk[m] / DECAY).sum(),
                        *[np.exp(-rk[m] / tau).sum() for tau in TAUS],
                        np.sort(w[m])[-3:].sum() / w[m].sum(), w[m & (age <= 25)].sum() / w[m].sum(),
                        *[w[m & (pos == q)].sum() / w[m].sum() for q in ("QB", "RB", "WR", "TE")],
                        *np.pad(p[:, i], (0, mm.H - P.shape[1]), constant_values=np.nan)))
    return out


COMP = ["top3", "young", "QB", "RB", "WR", "TE"]


def may(scan=False):
    """May of t + 1, no trades: title odds in seasons 1..5 on projected lineup points this season (redraft ADP) and dynasty value
    (sum of POOL - FP rank; and a top-heavy sum of exp(-rank / DECAY)), z-scores within the league, both in one OLS."""
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    d = mm.pool()
    d = d[d.t <= 2024]
    proj = mm.projector(pts, d)
    rng = np.random.default_rng(43)
    jobs = []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        yrs = [y for y in range(t + 1, t + 1 + mm.H) if y <= su.LAST]
        jobs.append((t, g, np.stack([g.gsis_id.map(pts[y]).fillna(0).to_numpy() for y in yrs], 1), int(rng.integers(1 << 31))))
    with Pool(JOBS) as pool:
        res = pool.map(may_cohort, jobs)
    pc = [f"p{k}" for k in range(1, mm.H + 1)]
    z = pd.DataFrame([r for part in res for r in part], columns=["t", "rep", "team", "redraft", "value", "value_top"] + [f"tau{x}" for x in TAUS] + COMP + pc)
    for c in ["redraft", "value", "value_top"] + [f"tau{x}" for x in TAUS] + COMP:
        gb = z.groupby(["t", "rep"])[c]
        z[c] = (z[c] - gb.transform("mean")) / gb.transform("std")
    z["lgid"] = z.t * 1000 + z.rep
    if scan == "splits":
        return splits(z)
    if scan:
        z["s1"], z["s23"], z["s45"], z["s25"] = z.p1, z.p2 + z.p3, z.p4 + z.p5, z.p2 + z.p3 + z.p4 + z.p5
        rows = []
        for c in ["value"] + [f"tau{x}" for x in TAUS]:
            r = {"measure": "rank sum" if c == "value" else f"decay {c[3:]}"}
            for y in ("s1", "s23", "s45", "s25"):
                q = z.dropna(subset=[y])
                by = q.groupby("t").apply(lambda a: a[c].corr(a[y]), include_groups=False)
                r[y] = round(by.mean(), 3)
            rows.append(r)
        print(f"== {'Superflex' if SF else '1QB'}, May: correlation of dynasty value with title odds (mean over cohorts), by decay. "
              "s1 = season 1, s23 = seasons 2-3, s45 = 4-5, s25 = 2-5")
        print(pd.DataFrame(rows).to_string(index=False))
        return
    print(f"== {'Superflex' if SF else '1QB'}, May, {REPS * 4} leagues a cohort, rosters frozen. Title odds (points) per 1 SD, both in the model "
          "[95% CI over leagues] cohorts with that sign. corr(redraft, value) = "
          f"{z.redraft.corr(z.value):.2f}, corr(redraft, value_top) = {z.redraft.corr(z.value_top):.2f}")
    rng = np.random.default_rng(5)
    for vc in ("value", "value_top"):
        rows = []
        for k, c in enumerate(pc, 1):
            q = z.dropna(subset=[c])

            def fit(q):
                X = np.column_stack([np.ones(len(q)), q.redraft, q[vc]])
                return np.linalg.lstsq(X, q[c].to_numpy() * 100, rcond=None)[0][1:]
            b, parts, ids = fit(q), dict(tuple(q.groupby("lgid"))), q.lgid.unique()
            bs = np.array([fit(pd.concat([parts[j] for j in rng.choice(ids, len(ids))])) for _ in range(200)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            by = np.array([fit(g) for _, g in q.groupby("t")])
            alone = [np.polyfit(q[x], q[c] * 100, 1)[0] for x in ("redraft", vc)]
            rows.append({"season": k, "cohorts": len(by),
                         **{n: f"{b[j]:+5.1f} [{lo[j]:+.1f}, {hi[j]:+.1f}] {int((by[:, j] > 0).sum())}/{len(by)}" for j, n in enumerate(("redraft", vc))},
                         "value ahead in": f"{int((by[:, 1] > by[:, 0]).sum())}/{len(by)}",
                         "redraft alone": round(alone[0], 1), "value alone": round(alone[1], 1)})
        print(pd.DataFrame(rows).to_string(index=False))
    z["rd3"] = z.groupby(["t", "rep"]).redraft.rank(method="first").pipe(lambda r: pd.cut(r, [0, 4, 8, 12], labels=["bottom", "middle", "top"]))
    z["v3"] = z.groupby(["t", "rep"]).value.rank(method="first").pipe(lambda r: pd.cut(r, [0, 4, 8, 12], labels=["bottom", "middle", "top"]))
    print("title odds by thirds of redraft strength (rows) and dynasty value (columns): season 1 / seasons 2-3 / seasons 4-5")
    z["s23"], z["s45"] = (z.p2 + z.p3) * 100, (z.p4 + z.p5) * 100
    z["s1"] = z.p1 * 100
    print(z.pivot_table(index="rd3", columns="v3", values=["s1", "s23", "s45"], observed=True).round(1).to_string())


def splits(z):
    """Does what a roster's dynasty value is made of matter, at the same strength this season and the same total value? Shares of
    the top-heavy value: in the best three players (top3), in players 25 or younger (young), and at each position. Title odds per
    1 SD of the share, with redraft strength and total top-heavy value in the model; and by thirds of the share."""
    z = z.copy()
    wins = {"s1": ["p1"], "s23": ["p2", "p3"], "s45": ["p4", "p5"]}
    for k, cs in wins.items():
        z[k] = z[cs].sum(axis=1, skipna=False) * 100
    rng = np.random.default_rng(11)
    print(f"== {'Superflex' if SF else '1QB'}, May, rosters frozen: title odds (points) per 1 SD of each share of top-heavy value, strength this season and "
          "total value held fixed [95% CI over leagues] cohorts with that sign; then mean title odds by thirds of the share "
          "(bottom / middle / top). s1 = season 1, s23 = seasons 2-3, s45 = seasons 4-5")
    print("correlations:")
    print(z[["redraft", "value_top"] + COMP].corr().round(2).to_string())
    rows = []
    for c in COMP:
        if c == "QB" and not SF:
            continue
        r = {"share": c}
        third = z.groupby(["t", "rep"])[c].rank(method="first").pipe(lambda x: pd.cut(x, [0, 4, 8, 12], labels=["b", "m", "t"]))
        for y in wins:
            q = z.dropna(subset=[y])

            def fit(q):
                X = np.column_stack([np.ones(len(q)), q.redraft, q.value_top, q[c]])
                return np.linalg.lstsq(X, q[y].to_numpy(), rcond=None)[0][3]
            b, parts, ids = fit(q), dict(tuple(q.groupby("lgid"))), q.lgid.unique()
            bs = [fit(pd.concat([parts[j] for j in rng.choice(ids, len(ids))])) for _ in range(200)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            by = np.array([fit(g) for _, g in q.groupby("t")])
            r[y] = f"{b:+5.1f} [{lo:+.1f}, {hi:+.1f}] {int((np.sign(by) == np.sign(b)).sum())}/{len(by)}"
            r[y + " thirds"] = " / ".join(f"{v:.1f}" for v in q.groupby(third[q.index], observed=True)[y].mean())
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def decay():
    """What decay do trade markets use, and which predicts titles best? Fits value = a * exp(-rank / decay) to each May list of KTC
    and FantasyCalc values (top POOL players, least squares on the value's share of the #1 player), then scans decays in may()."""
    import market_residuals as M
    from scipy.optimize import curve_fit
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    rows = []
    for src in ("ktc", "fantasycalc"):
        v = M.dd_markets(panel)[f"dd_{src}"]
        for t, q in v.groupby("t"):
            val = np.sort(-q.mkt.to_numpy())[::-1][:POOL]
            val, r = val / val[0], np.arange(1, len(val[:POOL]) + 1)
            (a, tau), _ = curve_fit(lambda r, a, tau: a * np.exp(-r / tau), r, val, p0=(1, 60))
            fit = a * np.exp(-r / tau)
            rows.append({"source": src, "May": t + 1, "players": len(r), "decay": round(tau, 1),
                         "R2": round(1 - ((val - fit) ** 2).sum() / ((val - val.mean()) ** 2).sum(), 3),
                         **{f"#{k} / #1": round(val[k - 1], 2) for k in (12, 60, 120, 200) if k <= len(r)}})
    print(f"== {'Superflex' if SF else '1QB'}: exponential fit to market values by rank")
    print(pd.DataFrame(rows).to_string(index=False))
    may(scan=True)


def modes():
    """middle_mode.py's trades for each of the site's modes (and its random-swap control), with the dynasty value held 1 and 3
    years on scored two ways: the site's rank sum (POOL - FP rank) and the top-heavy sum of exp(-rank / DECAY), each as a % of the
    no-trade roster's value. Does a top-heavy measure change which moves gain value?"""
    import json
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    pr = mm.prod_ranks(s)
    look = {(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"])
            for r in json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]}
    d = mm.pool()
    d = d[d.t <= 2024]
    proj = mm.projector(pts, d)
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    rng = np.random.default_rng(1)
    out = []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        act = mm.calls(g, t, pr, look)
        pos = g.position.to_numpy()
        nxt1 = g.gsis_id.map(rank_at.get(t + 1, pd.Series(dtype=float))).to_numpy(float)
        rks = [g.gsis_id.map(rank_at[y]).to_numpy(float) if y in rank_at else None for y in (t + 1, t + 3)]
        yrs = [y for y in range(t + 1, t + 1 + mm.H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        now = g.mkt_rank.to_numpy(float)

        def vals(T, i):
            r = []
            for col, rk in zip((1, 2), rks):
                x = rk[T[:, col] == i] if rk is not None else None
                r += [np.nan, np.nan] if x is None else [np.nansum(np.maximum(0, POOL - x)), np.nansum(np.exp(-x / DECAY))]
            return r
        for rep in range(REPS * 2):
            team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
            strength = np.array([su.lineup(g.proj.to_numpy()[team == i], pos[team == i]) for i in range(su.TEAMS)])
            rkk = (-strength).argsort().argsort() + 1
            T0 = np.repeat(team[:, None], mm.H, 1)
            top = np.array([np.exp(-now[team == i] / DECAY).sum() for i in range(su.TEAMS)])
            vrk = (-top).argsort().argsort() + 1
            seeds = rng.integers(1 << 31, size=mm.H)
            p0 = mm.season_titles(T0[:, :len(yrs)], P, pos, seeds)

            def pad(x):
                return np.pad(x * 100, (0, mm.H - len(x)), constant_values=np.nan)
            for i in range(su.TEAMS):
                v0 = vals(T0, i)
                for m in ("rebuild", "middle", "contend", "random"):
                    T, _ = mm.policy(i, m, team, g, act, nxt1, rng)
                    pm = mm.season_titles(T[:, :len(yrs)], P, pos, seeds)[:, i]
                    out.append((t, rkk[i], vrk[i], m, *v0, *vals(T, i), *pad(p0[:, i]), *pad(pm)))
    c = ["lin1", "top1", "lin3", "top3"]
    z = pd.DataFrame(out, columns=["t", "rank", "vrank", "mode"] + [x + "_0" for x in c] + c + [f"b{k}" for k in range(1, 6)]
                     + [f"p{k}" for k in range(1, 6)])
    z["grp"] = pd.cut(z["rank"], [0, 3, 8, 12], labels=["1-3", "4-8", "9-12"])
    rows = []
    for (grp, m), q in z.groupby(["grp", "mode"], observed=True):
        r = {"ranks": grp, "mode": m}
        for x in c:
            qq = q.dropna(subset=[x])
            by = qq.groupby("t").apply(lambda a: (a[x].mean() / a[x + "_0"].mean() - 1) * 100, include_groups=False)
            r[x] = f"{(qq[x].mean() / qq[x + '_0'].mean() - 1) * 100:+5.1f}% {int((by > 0).sum())}/{len(by)}"
        rows.append(r)
    print(f"== {'Superflex' if SF else '1QB'}, FP prices: dynasty value against no trades, % of the roster's value, cohorts above zero / cohorts. "
          "lin = rank sum, top = top-heavy; 1 and 3 years on")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- teams projected 4th-8th this season, by rank in top-heavy dynasty value: title odds with no trades (base) and each mode's "
          "change, season 1 / seasons 2-3 / seasons 4-5 (points; cohorts with a gain), and top-heavy value 3 years on")
    z["vg"] = pd.cut(z.vrank, [0, 2, 6, 12], labels=["value 1-2", "value 3-6", "value 7-12"])
    rows = []
    for (vg, m), q in z[z.grp == "4-8"].groupby(["vg", "mode"], observed=True):
        r = {"value rank": vg, "mode": m, "teams": len(q)}
        for name, ks in (("s1", (1,)), ("s23", (2, 3)), ("s45", (4, 5))):
            qq = q.dropna(subset=[f"p{k}" for k in ks])
            d = sum(qq[f"p{k}"] - qq[f"b{k}"] for k in ks)
            by = d.groupby(qq.t).mean()
            r[name + " base"] = round(sum(qq[f"b{k}"] for k in ks).mean(), 1)
            r[name] = f"{d.mean():+5.1f} {int((by > 0).sum())}/{len(by)}"
        qq = q.dropna(subset=["top3"])
        r["top3"] = f"{(qq.top3.mean() / qq.top3_0.mean() - 1) * 100:+5.1f}%"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["may"]:
        may()
    elif sys.argv[1:] == ["modes"]:
        modes()
    elif sys.argv[1:] == ["decay"]:
        decay()
    elif sys.argv[1:] == ["splits"]:
        may(scan="splits")
    elif len(sys.argv) > 1:
        summarize(pd.read_csv(out_dir(SF) / "team_signals.csv.gz"))
    else:
        main()
