"""Q: do the in-season riser / faller signals (inseason_movers.py) pay off as trades inside the league sim?

League as middle_mode.py: 12 teams draft the FP dynasty top POOL (default 340) of cohort t before season Y = t + 1.
At one trade window in season Y (after week w = 4, 8 or 12; the inseason_movers.py snapshot) one team follows a rule
and makes equal-price trades at the in-season FP dynasty rank then (other side within x0.8-1.25; players out of the
top POOL at the snapshot are not tradeable). Everyone else stands pat; rosters are otherwise frozen.
Rules (signal from inseason_movers.py's columns at that snapshot):
  sell_hurt      sell own fallers on IR / reserve or ruled Out ("faller: hurt")
  sell_faller    sell every own faller (rank down a quarter or more since the preseason)
  ros_down_sell  sell own players whose ROS rank fell a quarter or more since the preseason
  ros_up_buy     buy up to NBUY others whose ROS rank rose a quarter or more
  ros_both       both of the above
  young_gap_buy  buy up to NBUY rookies / 25-and-under whose positional ROS rank beats their dynasty one by a quarter
  rookie_gap_buy the same, rookies only
  riser_rbte_sell sell own RB / TE risers (superflex only)
  random         control: NRAND equal-price swaps with random partners
Each rule runs with two pickers (what to take back / give away), as middle_mode.py: "young" (rebuilder: take the
youngest, give the oldest), "lineup" (take the biggest rest-of-season lineup gain, give the smallest loss; rest-of-season
points projected from the ROS rank, isotonic by position, or from the dynasty rank where there is no ROS list) and
"value" (the site's in-season price: FP dynasty rank x (ROS positional / dynasty positional rank) ^ 0.1; take the best,
give the worst, buy the most underpriced first). A sell never takes back a player with a sell signal; a buy never gives one with a buy signal.
Season Y title odds: regular-season weeks 1..w at the pre-trade lineup's per-week mean (actual points, best ball),
weeks w+1..14 and three playoff weeks at the post-trade lineup's mean over weeks w+1..17; later seasons as
middle_mode.py. Outcomes vs the same team not trading (common random numbers): title odds change, season Y and
summed over Y..Y+2 / Y..Y+4, and dynasty value (sum of POOL - FP rank) at the next two Mays (t + 1, t + 3).
Cohort halves 2018-20 / 2021-24 (seasons 2019-21 / 2022-25); league-clustered 95% CIs.
Caveat: the signals were found on these same seasons, so this checks how they trade, not out of sample.
Writes outputs/[sf/]inseason_sim[_offdrop].csv. LEAGUE=sf for superflex; REPS (default 10), WEEKS (default 4,8,12), JOBS;
OFFLIST=drop: players off the ROS list (mostly out for the season) get no signal, in case the dynasty price lags the
news there.
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
import middle_mode as mm  # noqa: E402  (scales su.CAP / ROUNDS to POOL)
import startup as su  # noqa: E402
import win_now as wn  # noqa: E402
from long_horizon import SF  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

REPS = int(os.environ.get("REPS", 10))
WEEKS = tuple(int(w) for w in os.environ.get("WEEKS", "4,8,12").split(","))
JOBS = int(os.environ.get("JOBS", 6))
POOL, H, NBUY, NRAND = mm.POOL, mm.H, 3, 3
RULES = ["random", "sell_hurt", "sell_faller", "ros_down_sell", "ros_up_buy", "ros_both", "young_gap_buy",
         "rookie_gap_buy"] + (["riser_rbte_sell"] if SF else [])
PICKERS = ("young", "lineup", "value")
OFFLIST = os.environ.get("OFFLIST", "keep")      # "drop": players off the ROS list get no signal (stale-price check)
TILT = 0.1                                       # pages/build/rules.py ROS_TILT
REG, LAST_WK, WIN = 14, 17, 0.2


def signals(m):
    """rule -> (sell mask, buy mask) over the snapshot rows m (aligned to the pool)"""
    hurt = m.kind.eq("faller: hurt").to_numpy()
    fall = m.mv.eq("faller").to_numpy()
    rd, ru = m.rosmv.eq("ROS down").to_numpy(), m.rosmv.eq("ROS up").to_numpy()
    gap = m.gap.eq("ROS > dynasty").to_numpy()
    young = gap & m.acls.isin(["rookie", "25-"]).to_numpy()
    rook = gap & m.acls.eq("rookie").to_numpy()
    rbte = m.mv.eq("riser").to_numpy() & m.position.isin(["RB", "TE"]).to_numpy()
    no = np.zeros(len(m), bool)
    if OFFLIST == "drop":
        on = ~(m.ros_rank >= 500).to_numpy()
        hurt, fall, rd, ru, gap, young, rook, rbte = (a & on for a in (hurt, fall, rd, ru, gap, young, rook, rbte))
    return {"sell_hurt": (hurt, no), "sell_faller": (fall, no), "ros_down_sell": (rd, no), "ros_up_buy": (no, ru),
            "ros_both": (rd, ru), "young_gap_buy": (no, young), "rookie_gap_buy": (no, rook),
            "riser_rbte_sell": (rbte, no)}


def ros_projector(x, wk):
    """per-week rest-of-season points from ROS rank (or dynasty rank), isotonic by position, pooled over seasons"""
    rest = {}
    for (y, w) in x[["season", "w"]].drop_duplicates().itertuples(index=False):
        s = wk[(wk.season == y) & (wk.week > w) & (wk.week <= LAST_WK)]
        rest[(y, w)] = s.groupby("gsis_id").ppr.sum() / (LAST_WK - w)
    x = x.assign(rate=[rest[(y, w)].get(g, 0.0) for g, y, w in zip(x.gsis_id, x.season, x.w)])
    fits = {}
    for col in ("ros_rank", "rank"):
        for q, g in x[x[col].notna()].groupby("position"):
            fits[(col, q)] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(np.log(g[col]), g.rate)

    def proj(m):
        out = np.zeros(len(m))
        for k, (r, rr, q) in enumerate(zip(m.ros_rank, m["rank"], m.position)):
            if isinstance(q, str):
                col, v = ("ros_rank", r) if r == r else ("rank", rr)
                if v == v:
                    out[k] = fits[(col, q)].predict([np.log(v)])[0]
        return out
    return proj


def titles_split(Lpre, Lpost, w, sd, rng):
    """wn.titles with weeks 1..w at Lpre (per-week means) and the rest (incl. playoffs) at Lpost"""
    n, NS = len(Lpre), wn.NSIM
    wins, pf = np.zeros((NS, n)), np.zeros((NS, n))
    for k in range(REG):
        m = Lpre if k < w else Lpost
        perm = np.argsort(rng.random((NS, n)), 1)
        sc = m + sd * rng.standard_normal((NS, n))
        a, b = perm[:, 0::2], perm[:, 1::2]
        sa, sb = np.take_along_axis(sc, a, 1), np.take_along_axis(sc, b, 1)
        np.put_along_axis(wins, a, np.take_along_axis(wins, a, 1) + (sa > sb), 1)
        np.put_along_axis(wins, b, np.take_along_axis(wins, b, 1) + (sb > sa), 1)
        pf += sc
    seed = np.argsort(-(wins * 1e4 + pf), 1)[:, :6]

    def game(x, y):
        sx = Lpost[x] + sd * rng.standard_normal(len(x))
        sy = Lpost[y] + sd * rng.standard_normal(len(y))
        return np.where(sx > sy, x, y)
    q1, q2 = game(seed[:, 2], seed[:, 5]), game(seed[:, 3], seed[:, 4])
    ch = game(game(seed[:, 0], q2), game(seed[:, 1], q1))
    win = np.zeros(n)
    np.add.at(win, ch, 1)
    return win / NS


def trade(i, team, rule, picker, S, rng, masks=None):
    """team assignment after team i trades at the window; and the number of trades. masks: (sell, buy) override"""
    price, age, pj, pos = S["price"], S["age"], S["proj"], S["pos"]
    cur, moved = team.copy(), np.zeros(len(team), bool)
    ok = ~np.isnan(price)

    def window(p):
        return ok & (np.abs(price - p) <= p * WIN)

    def other():
        return (cur >= 0) & (cur != i) & ~moved

    def lu(ids):
        return su.lineup(pj[ids], pos[ids])

    if rule == "random":
        n = 0
        for j in rng.permutation(np.flatnonzero((team == i) & ok)):
            if n >= NRAND:
                break
            c = np.flatnonzero(other() & window(price[j]))
            if len(c):
                get = rng.choice(c)
                cur[j], cur[get] = cur[get], i
                moved[[j, get]] = True
                n += 1
        return cur, n
    sell, buy = masks if masks is not None else S["sig"][rule]
    n = 0
    for j in sorted(np.flatnonzero((cur == i) & sell & ok), key=lambda j: price[j]):
        c = np.flatnonzero(other() & window(price[j]) & ~sell)
        if not len(c):
            continue
        if picker == "young":
            get = c[np.lexsort((price[c], age[c]))[0]]
        elif picker == "value":
            get = c[np.argmin(S["val"][c])]
        else:
            mine = np.flatnonzero((cur == i) & (np.arange(len(cur)) != j))
            get = c[int(np.argmax([lu(np.append(mine, x)) for x in c]))]
        cur[j], cur[get] = cur[get], i
        moved[[j, get]] = True
        n += 1
    tg = np.flatnonzero(other() & buy & ok)
    if picker == "young":
        tg = tg[np.argsort(age[tg])]
    elif picker == "value":
        tg = tg[np.argsort(S["val"][tg] / price[tg])]
    else:
        mine = np.flatnonzero(cur == i)
        base = lu(mine)
        tg = tg[np.argsort([-(lu(np.append(mine, x)) - base) for x in tg])]
    nb = 0
    for b in tg:
        if nb >= NBUY:
            break
        give = np.flatnonzero((cur == i) & ~moved & window(price[b]) & ~buy)
        if not len(give):
            continue
        if picker == "young":
            gv = give[np.argmax(age[give])]
        elif picker == "value":
            gv = give[np.argmax(S["val"][give])]
        else:
            mine = np.flatnonzero(cur == i)
            gv = give[int(np.argmax([lu(np.append(mine[mine != x], b)) for x in give]))]
        cur[gv], cur[b] = cur[b], i
        moved[[gv, b]] = True
        nb += 1
    return cur, n + nb


def placebo(i, team, sell, buy, ok, rng):
    """the rule's masks with its players swapped for as many random tradeable ones without the signal"""
    def draw(pool_, k):
        m = np.zeros(len(team), bool)
        c = np.flatnonzero(pool_)
        m[rng.choice(c, size=min(k, len(c)), replace=False)] = True
        return m
    mine, oth = (team == i) & ok, (team >= 0) & (team != i) & ok
    return draw(mine & ~sell, int((mine & sell).sum())), draw(oth & ~buy, int((oth & buy).sum()))


def value(team, i, rk):
    return np.nansum(np.maximum(0, max(200, POOL) - rk[team == i])) if rk is not None else np.nan


def run_cohort(args):
    t, g, snaps, wkp, pts, rank_at, proj_by_w, seed = args
    rng = np.random.default_rng(seed)
    Y = t + 1
    pos = g.position.to_numpy()
    yrs = [y for y in range(Y + 1, Y + H) if y <= su.LAST]
    P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1) if yrs else np.zeros((len(g), 0))
    rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
    rk3 = g.gsis_id.map(rank_at[t + 3]).to_numpy(float) if t + 3 in rank_at else None
    out = []
    for rep in range(REPS):
        team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
        strength = np.array([su.lineup(g.proj.to_numpy()[team == i], pos[team == i]) for i in range(su.TEAMS)])
        rkk = (-strength).argsort().argsort() + 1
        for w_target in WEEKS:
            m = snaps.get(w_target)
            if m is None:
                continue
            w = int(m.w.dropna().iloc[0])
            wk = wkp[Y]
            pre = g.gsis_id.map(wk.loc[:, wk.columns <= w].sum(axis=1)).fillna(0).to_numpy()
            post = g.gsis_id.map(wk.loc[:, (wk.columns > w) & (wk.columns <= LAST_WK)].sum(axis=1)).fillna(0).to_numpy()
            Lpre = np.array([su.lineup(pre[team == k], pos[team == k]) for k in range(su.TEAMS)]) / w
            sd = wn.SD * (Lpre.mean() * w + np.array([su.lineup(post[team == k], pos[team == k])
                                                      for k in range(su.TEAMS)]).mean()) / LAST_WK
            S = {"price": m["rank"].to_numpy(float), "age": g.age.to_numpy(), "pos": pos, "proj": proj_by_w[w_target],
                 "sig": signals(m)}
            has_ros = m.ros_rank.notna().any()
            rp = m.ros_prank.fillna(500).to_numpy(float) if has_ros else m.prank.to_numpy(float)
            S["val"] = S["price"] * (rp / m.prank.to_numpy(float)) ** TILT     # in-season dynasty, small ROS tilt
            seeds = rng.integers(1 << 31, size=H)

            def evaluate(T):
                Lpost = np.array([su.lineup(post[T == k], pos[T == k]) for k in range(su.TEAMS)]) / (LAST_WK - w)
                p = [titles_split(Lpre, Lpost, min(w, REG), sd, np.random.default_rng(seeds[0]))]
                if P.shape[1]:
                    p += list(mm.season_titles(np.repeat(T[:, None], P.shape[1], 1), P, pos, seeds[1:]))
                p = np.array(p)                                   # seasons x teams
                return np.pad(p, ((0, H - len(p)), (0, 0)), constant_values=np.nan)
            p0 = evaluate(team)
            for i in range(su.TEAMS):
                out.append((t, rep, w_target, rkk[i], "none", "-", *p0[:, i], value(team, i, rk1),
                            value(team, i, rk3), 0))
                tradeable = ~np.isnan(S["price"])
                for rule in RULES:
                    for picker in (PICKERS if rule != "random" else ("-",)):
                        T, n = trade(i, team, rule, picker, S, rng)
                        p = evaluate(T)[:, i] if n else p0[:, i]
                        out.append((t, rep, w_target, rkk[i], rule, picker, *p, value(T, i, rk1), value(T, i, rk3), n))
                        if rule == "random":
                            continue
                        T, n = trade(i, team, rule, picker, S, rng, placebo(i, team, *S["sig"][rule], tradeable, rng))
                        p = evaluate(T)[:, i] if n else p0[:, i]
                        out.append((t, rep, w_target, rkk[i], rule + "~", picker, *p, value(T, i, rk1),
                                    value(T, i, rk3), n))
    print(f"cohort {t} done", flush=True)
    return out


def main():
    import inseason_movers as im
    x = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    wk = im.weekly()
    wkp = {y: z.pivot_table(index="gsis_id", columns="week", values="ppr", aggfunc="sum").fillna(0)
           for y, z in wk.groupby("season")}
    proj = ros_projector(x, wk)
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    d = mm.pool()
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    d = d[d.t <= 2024]
    pproj = mm.projector(pts, d)
    rng = np.random.default_rng(7)
    jobs = []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
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
    cols = ["t", "rep", "wk", "rank", "rule", "picker"] + [f"p{k}" for k in range(1, H + 1)] + ["val1", "val3", "n"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.to_csv(out_dir(SF) / f"inseason_sim{'_offdrop' if OFFLIST == 'drop' else ''}.csv", index=False)
    summarize(df)


def ci(d):
    d = d.dropna()
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan


def summarize(df):
    LAST = su.LAST
    key = ["t", "rep", "wk", "rank"]
    base = df[df.rule == "none"].set_index(key)
    z = df[df.rule != "none"].join(base[[f"p{k}" for k in range(1, H + 1)] + ["val1", "val3"]], on=key, rsuffix="_0")  # noqa
    z["yr1"] = (z.p1 - z.p1_0) * 100
    z["yrs3"] = np.where(z.t + 3 <= LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in (1, 2, 3)) * 100, np.nan)
    z["yrs5"] = np.where(z.t + 5 <= LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in range(1, 6)) * 100, np.nan)
    z["dval1"], z["dval3"] = z.val1 - z.val1_0, z.val3 - z.val3_0
    z["grp"] = pd.cut(z["rank"], [0, 3, 8, 12], labels=["1-3", "4-8", "9-12"])
    z["half"] = np.where(z.t <= 2020, "a", "b")
    mets = ["yr1", "yrs3", "yrs5", "dval1", "dval3"]
    lens = {"1-3": ["yr1", "yrs3", "dval1"], "4-8": ["yrs3", "dval1", "dval3"], "9-12": ["dval3", "yrs5", "yrs3"]}
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 40)

    def cell(d):                                          # d: Series indexed by key
        mu, e = ci(d.groupby(level=["t", "rep", "wk"]).mean())
        h = d.groupby(np.where(d.index.get_level_values("t") <= 2020, "a", "b")).mean()
        return f"{mu:+6.1f}±{e:4.1f} [{h.get('a', np.nan):+.1f}/{h.get('b', np.nan):+.1f}]"
    print(f"\n== {'Superflex' if SF else '1QB'}, FP top {POOL}: one in-season trade window (weeks {WEEKS} pooled). "
          "mean ± 95% CI (league-clustered) [cohorts 2018-20 / 2021-24]; title odds in percentage points summed, "
          "value = sum of POOL - FP rank")
    for grp, cols in lens.items():
        q0 = z[z.grp == grp]
        rows = []
        for (rule, picker), q in q0[~q0.rule.str.endswith("~")].groupby(["rule", "picker"]):
            r = {"rule": rule, "picker": picker, "trades": round(q.n.mean(), 2)}
            pl = q0[(q0.rule == rule + "~") & (q0.picker == picker)].set_index(key)[mets]
            qq = q.set_index(key)
            for c in cols:
                r[f"{c} vs none"] = cell(qq[c])
                if len(pl):
                    d = (qq[c] - pl[c]).dropna()
                    r[f"{c} vs placebo"] = cell(d)
            rows.append(r)
        print(f"\n-- projected ranks {grp}")
        print(pd.DataFrame(rows).fillna("").to_string(index=False))
    print("\n-- vs placebo by window week, ranks 4-8 (yrs3 / dval1 / dval3)")
    q = z[z.grp == "4-8"].set_index(key + ["picker"])
    a, b = q[~q.rule.str.endswith("~") & (q.rule != "random")], q[q.rule.str.endswith("~")]
    b = b.assign(rule=b.rule.str[:-1])
    d = a.set_index("rule", append=True)[["yrs3", "dval1", "dval3"]] - b.set_index("rule", append=True)[["yrs3", "dval1", "dval3"]]
    print(d.groupby(["rule", "picker", "wk"]).mean().unstack("wk").round(1).to_string())

if __name__ == "__main__":
    main()
