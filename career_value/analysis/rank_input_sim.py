"""Q: mid-season, should the site's calls come from the current dynasty rank or the preseason one? And did the sell-signal
exemptions and floors in pages/build/rules.py help?

League as inseason_sim.py: 12 teams draft the FP dynasty top POOL of cohort t before season Y = t + 1; at one window
(after week 4, 8 or 12) one team follows the site's full calls in one mode and trades at the in-season FP dynasty rank
(other side within x0.8-1.25; players off the snapshot are not tradeable). Rosters are otherwise frozen.
The calls are the live ones: rules.classify on the arm's rank, then rules.situational on the current rank, as
pages/build/build_data.py does (rest-of-season role, injury calls and signals). No market prices: no notes from them.
Arms:
  live      current rank tilted toward the ROS rank, signals with floors and exemptions: what the site does
  now       current rank, no tilt
  pre       preseason rank (classify only; the price, the signals and the rest-of-season calls still use the current one)
  nosig     as live, no signals
  noexempt  as live, no QB or young-hurt exemption
  noqbx     as live, no QB exemption
  noyoungx  as live, no young-hurt exemption
  nofloor   as live, no rank-place floors
  live~ pre~  placebo: as many sells and buys as that arm, on random players without the call
ONLY=exempt runs live, noqbx and noyoungx with a tighter placebo, noqbx~ / noyoungx~: instead of the signal's extra sells,
as many sells of random own players of the same kind (QBs; RB / WR / TE 25 or younger). Any extra equal-price trade gains from
the pickers, so noqbx or noyoungx minus live alone doesn't say the signal found anything. Writes rank_input_sim_exempt.csv.gz.
What the team does, as middle_mode.py: Sell (any strength) trades now for a player the mode wouldn't sell (rebuild: the
youngest; contend / default: the biggest rest-of-season lineup gain); Buy, up to 3, gives the own player valued least
(rebuild: the oldest; else the smallest lineup loss); Use, then sell and Sell when healthy keep him for season Y and
trade him at next May's price; Hold, Stash, Lottery ticket and Cut make no move.
Outcomes vs the same team not trading (common random numbers), as inseason_sim.py: title odds change in season Y and summed
over Y..Y+2 / Y..Y+4; dynasty value (sum of POOL - FP rank) at the next two Mays.
The CIs treat league sims as independent, but every sim of a cohort shares its players, so also read the per-cohort rows.
Caveat: the rules were fit on May ranks over these cohorts, and the preseason rank is nearly a May rank, so "pre" gets
more of the in-sample edge than "live" or "now". Cohorts after the fit are the fairer test.
Writes outputs/[sf/]rank_input_sim.csv.gz. LEAGUE=sf for superflex; REPS (default 10), WEEKS (default 4,8,12), JOBS.
"""
import json
import os
import pathlib
import sys
from collections import namedtuple
from contextlib import contextmanager
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import inseason_sim as isim  # noqa: E402  (sets POOL = 340 for middle_mode)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import HERE, out_dir  # noqa: E402
import middle_mode as mm  # noqa: E402
import startup as su  # noqa: E402
import win_now as wn  # noqa: E402
from long_horizon import SF  # noqa: E402

R = mm.R
F = mm.F
REPS, WEEKS, JOBS = isim.REPS, isim.WEEKS, isim.JOBS
POOL, H, NBUY = mm.POOL, mm.H, 3
LAST_WK, REG = isim.LAST_WK, isim.REG
MODES = ("reb", "mid", "con")
ONLY = os.environ.get("ONLY", "")
ARMS = ("live", "noqbx", "noyoungx") if ONLY == "exempt" else ("live", "now", "pre", "nosig", "noexempt", "noqbx", "noyoungx", "nofloor")
PLACEBO = () if ONLY == "exempt" else ("live", "pre")
KIND_PLACEBO = ("noqbx", "noyoungx") if ONLY == "exempt" else ()
LENS_MODE = {"1-3": "con", "4-8": "mid", "9-12": "reb"}
Row = namedtuple("Row", ["position", "age", f"r_{F}", f"pr_{F}", "rd", "dy", "prk", "prk_last", "prod_rank", f"mr_{F}",
                         f"rv_{F}", "inj", "inj_part", "gone", "ros_r", "ros_p", "ro_pre", "ro_now", "do_pre", "do_now",
                         "dsf_pre", "dsf_now"])


@contextmanager
def signal_rules(arm):
    """rules.py with this arm's signal variant"""
    apply0, floors = R.apply_signal, (R.SIG_FLOOR_ROS, R.SIG_FLOOR_DYN)

    def wrapped(act, mode, x, f, s):
        if x.position == "QB":
            if arm in ("noexempt", "noqbx"):
                x, s = x._replace(position="RB"), {**s, "young": False}
        elif arm in ("noexempt", "noyoungx"):
            s = {**s, "young": False}
        return apply0(act, mode, x, f, s)
    R.apply_signal = wrapped
    if arm == "nofloor":
        R.SIG_FLOOR_ROS, R.SIG_FLOOR_DYN = 0, 0
    try:
        yield
    finally:
        R.apply_signal = apply0
        R.SIG_FLOOR_ROS, R.SIG_FLOOR_DYN = floors


def features(q, last):
    """site feature rows at one snapshot; q is one (season, target) of inseason_movers.csv"""
    q = q.copy()
    played = q.g.fillna(0) >= 4
    q["ppg"] = (q.pts / q.g).where(played)
    q["prk"] = q.groupby("position").ppg.rank(ascending=False, method="min")
    q["ppre"] = q.groupby("position").pre.rank(method="first")
    y = int(q.season.iloc[0])
    has = q.ros_rank.notna().any()                          # no ROS list for this snapshot: no tilt, no signals
    rows, inputs = [], {}
    for t in q.itertuples(index=False):
        prk_last = last.get(t.gsis_id, np.nan)
        blend = np.nanmean([t.prk, prk_last]) if t.prk == t.prk or prk_last == prk_last else np.nan
        off = has and (t.ros_rank != t.ros_rank or t.ros_rank >= R.ROS_OFF)
        ros_r = np.nan if off or not has else t.ros_rank
        ros_p = np.nan if off or not has else t.ros_prank
        rows.append(Row(t.position, t.age + 1, t.rank, t.prank, 8 if t.draft_round_f != t.draft_round_f else t.draft_round_f,
                        y if t.row_type == "rookie" else 0, t.prk, prk_last, blend, np.nan, np.nan,
                        "Out" if t.hurt else None, None, bool(off), ros_r, ros_p, t.ros_pre, ros_r, t.pre, t.rank,
                        t.pre, t.rank))
    inputs["tilt"] = [
        (x._replace(**{f"r_{F}": getattr(x, f"r_{F}") * k, f"pr_{F}": getattr(x, f"pr_{F}") * k}))
        for x, k in zip(rows, (R.ros_tilt(getattr(x, f"pr_{F}"), x.ros_p) if has else 1.0 for x in rows))]
    inputs["now"] = rows
    inputs["pre"] = [x._replace(**{f"r_{F}": t.pre, f"pr_{F}": t.ppre}) for x, t in zip(rows, q.itertuples(index=False))]
    return rows, inputs


def act_kind(a):
    return "use" if a == R.SELLHEALTHY else R.kind(a)       # "sell once he's playing again": next May, as Use, then sell


def snapshot_calls(q, last, w):
    """(arm, mode) -> {gsis_id: kind}"""
    rows, inputs = features(q, last)
    y = int(q.season.iloc[0])
    base = {"ROOKIE": y, "LOOK": LOOK, "EARLY": False}
    live = {**base, "WEEK": w + 1}                          # Sleeper's current week right after week w
    out = {}
    for arm in ARMS:
        src = {"now": "now", "pre": "pre"}.get(arm, "tilt")
        ctx = base if arm == "nosig" else live
        res = {m: {} for m in MODES}
        with signal_rules(arm):
            for g, x, xv in zip(q.gsis_id, rows, inputs[src]):
                c = R.situational(x, F, R.classify(xv, F, ctx), ctx)
                res["reb"][g], res["con"][g], res["mid"][g] = act_kind(c[1]), act_kind(c[3]), act_kind(c[7])
        for m in MODES:
            out[arm, m] = res[m]
    return out


def policy(i, mode, team, k, S, nxt1):
    """team assignment per season (players x H) after team i follows the calls k; and (sells, sells later, buys)"""
    price, age, pj, pos = S["price"], S["age"], S["proj"], S["pos"]
    cur, moved = team.copy(), np.zeros(len(team), bool)
    no_take = ("sell", "cut") if mode == "con" else ("sell", "cut", "use")
    taken = np.isin(k, no_take)

    def window(p, pr):
        return (pr >= p * 0.8) & (pr <= p * 1.25)

    def other():
        return (cur >= 0) & (cur != i) & ~moved

    def lu(ids):
        return su.lineup(pj[ids], pos[ids])

    n_now = n_buy = n_later = 0
    for j in sorted(np.flatnonzero((cur == i) & (k == "sell") & ~np.isnan(price)), key=lambda j: price[j]):
        c = np.flatnonzero(other() & window(price[j], price) & ~taken)
        if not len(c):
            continue
        if mode == "reb":
            get = c[np.lexsort((price[c], age[c]))[0]]
        else:
            mine = np.flatnonzero((cur == i) & (np.arange(len(cur)) != j))
            get = c[int(np.argmax([lu(np.append(mine, x)) for x in c]))]
        cur[j], cur[get] = cur[get], i
        moved[[j, get]] = True
        n_now += 1
    tg = np.flatnonzero(other() & (k == "buy") & ~np.isnan(price))
    if mode == "reb":
        tg = tg[np.argsort(age[tg])]
    elif len(tg):
        mine = np.flatnonzero(cur == i)
        b0 = lu(mine)
        tg = tg[np.argsort([-(lu(np.append(mine, x)) - b0) for x in tg])]
    for b in tg:
        if n_buy >= NBUY:
            break
        give = np.flatnonzero((cur == i) & ~moved & window(price[b], price) & (k != "buy"))
        if not len(give):
            continue
        if mode == "reb":
            gv = give[np.argmax(age[give])]
        else:
            mine = np.flatnonzero(cur == i)
            gv = give[int(np.argmax([lu(np.append(mine[mine != x], b)) for x in give]))]
        cur[gv], cur[b] = cur[b], i
        moved[[gv, b]] = True
        n_buy += 1
    T = np.repeat(cur[:, None], H, 1)
    for j in np.flatnonzero((cur == i) & (k == "use") & ~moved):
        p1 = nxt1[j]
        if np.isnan(p1):
            continue                                        # out of the top POOL a year on: nothing to sell
        c = np.flatnonzero(other() & window(p1, nxt1) & ~taken)
        if not len(c):
            continue
        get = c[np.lexsort((nxt1[c], age[c]))[0]] if mode == "reb" else c[np.argmin(nxt1[c])]
        T[j, 1:], T[get, 1:] = cur[get], i
        moved[[j, get]] = True
        n_later += 1
    return T, (n_now, n_later, n_buy)


def run_cohort(args):
    t, g, snaps, calls, wkp, pts, rank_at, proj_by_w, seed = args
    rng = np.random.default_rng(seed)
    Y = t + 1
    pos, age = g.position.to_numpy(), g.age.to_numpy()
    yrs = [y for y in range(Y + 1, Y + H) if y <= su.LAST]
    P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1) if yrs else np.zeros((len(g), 0))
    rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
    rk3 = g.gsis_id.map(rank_at[t + 3]).to_numpy(float) if t + 3 in rank_at else None
    nxt1 = rk1 if rk1 is not None else np.full(len(g), np.nan)
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
            S = {"price": m["rank"].to_numpy(float), "age": age, "pos": pos, "proj": proj_by_w[w_target]}
            ok = ~np.isnan(S["price"])
            young = m.acls.isin(["rookie", "25-"]).to_numpy()
            seeds = rng.integers(1 << 31, size=H)
            kinds = {key: np.array([v.get(x, "none") for x in g.gsis_id], dtype="U5")
                     for key, v in calls[w_target].items()}
            memo = {}

            def evaluate(T):
                key = T.tobytes()
                if key not in memo:
                    Lpost = np.array([su.lineup(post[T[:, 0] == k], pos[T[:, 0] == k]) for k in range(su.TEAMS)]) / (LAST_WK - w)
                    p = [isim.titles_split(Lpre, Lpost, min(w, REG), sd, np.random.default_rng(seeds[0]))]
                    if P.shape[1]:
                        p += list(mm.season_titles(T[:, 1:1 + P.shape[1]], P, pos, seeds[1:]))
                    p = np.array(p)                         # seasons x teams
                    memo[key] = np.pad(p, ((0, H - len(p)), (0, 0)), constant_values=np.nan)
                return memo[key]

            def val(T, col, rk):
                return np.nansum(np.maximum(0, max(200, POOL) - rk[T[:, col] == i])) if rk is not None else np.nan

            T0 = np.repeat(team[:, None], H, 1)
            p0 = evaluate(T0)
            for i in range(su.TEAMS):
                out.append((t, rep, w_target, rkk[i], "none", "-", *p0[:, i], val(T0, 1, rk1), val(T0, 2, rk3), 0, 0, 0))
                for mode in MODES:
                    for arm in ARMS:
                        k = kinds[arm, mode]
                        T, n = policy(i, mode, team, k, S, nxt1)
                        out.append((t, rep, w_target, rkk[i], arm, mode, *evaluate(T)[:, i], val(T, 1, rk1),
                                    val(T, 2, rk3), *n))
                        if arm in PLACEBO:
                            sell, buy = isim.placebo(i, team, k == "sell", k == "buy", ok, rng)
                            kp = np.where(sell, "sell", np.where(buy, "buy", "hold"))
                            T, n = policy(i, mode, team, kp, S, nxt1)
                            out.append((t, rep, w_target, rkk[i], arm + "~", mode, *evaluate(T)[:, i], val(T, 1, rk1),
                                        val(T, 2, rk3), *n))
                        if arm in KIND_PLACEBO:
                            kl = kinds["live", mode]
                            new = (k == "sell") & (kl != "sell")
                            kp = np.where(new, kl, k)
                            same = pos == "QB" if arm == "noqbx" else (pos != "QB") & young
                            c = np.flatnonzero((team == i) & ok & same & ~new & ~np.isin(kp, ("sell", "cut")))
                            pick = rng.choice(c, size=min(int((new & (team == i)).sum()), len(c)), replace=False)
                            kp[pick] = "sell"
                            T, n = policy(i, mode, team, kp, S, nxt1)
                            out.append((t, rep, w_target, rkk[i], arm + "~", mode, *evaluate(T)[:, i], val(T, 1, rk1),
                                        val(T, 2, rk3), *n))
    print(f"cohort {t} done", flush=True)
    return out


LOOK = {}


def main():
    import inseason_movers as im
    LOOK.update({(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"]) for r in
                 json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]})
    x = pd.read_csv(out_dir(SF) / "inseason_movers.csv")
    wk = im.weekly()
    wkp = {y: z.pivot_table(index="gsis_id", columns="week", values="ppr", aggfunc="sum").fillna(0)
           for y, z in wk.groupby("season")}
    proj = isim.ros_projector(x, wk)
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    pr6 = mm.prod_ranks(s)[6]
    d = mm.pool()
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    d = d[d.t <= 2024]
    pproj = mm.projector(pts, d)
    rng = np.random.default_rng(11)
    jobs = []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
        g["proj"] = pproj(g)
        snaps, pbw, calls = {}, {}, {}
        for w_target in WEEKS:
            q = x[(x.season == t + 1) & (x.target == w_target)]
            if q.empty or int(q.w.iloc[0]) >= LAST_WK:
                continue
            m = g[["gsis_id", "position"]].merge(q.drop(columns="position"), on="gsis_id", how="left")
            m["w"] = int(q.w.iloc[0])
            snaps[w_target], pbw[w_target] = m, proj(m)
            calls[w_target] = snapshot_calls(q, pr6.get(t, {}), int(q.w.iloc[0]))
        jobs.append((t, g, snaps, calls, wkp, pts, rank_at, pbw, int(rng.integers(1 << 31))))
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "wk", "rank", "arm", "mode"] + [f"p{k}" for k in range(1, H + 1)] + ["val1", "val3", "sell", "later", "buy"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / f"rank_input_sim{'_exempt' if ONLY == 'exempt' else ''}.csv.gz", index=False)
    summarize_exempt(df) if ONLY == "exempt" else summarize(df)


def summarize_exempt(df):
    key = ["t", "rep", "wk", "rank", "mode"]
    mets = ["yr1", "yrs3", "yrs5", "dval1", "dval3"]
    z = df[df.arm != "none"].copy()
    z["yr1"] = z.p1 * 100
    z["yrs3"] = np.where(z.t + 3 <= su.LAST, (z.p1 + z.p2 + z.p3) * 100, np.nan)
    z["yrs5"] = np.where(z.t + 5 <= su.LAST, sum(z[f"p{k}"] for k in range(1, 6)) * 100, np.nan)
    z["dval1"], z["dval3"] = z.val1, z.val3
    idx = z.set_index(key)
    live = idx[idx.arm == "live"]
    groups = {"1-3": [1, 2, 3], "4-8": range(4, 9), "9-12": range(9, 13)}

    def cell(d):
        v = d.dropna()
        if v.empty:
            return ""
        c = v.groupby(level=["t", "rep", "wk"]).mean()
        by = v.groupby(level="t").mean()
        return f"{c.mean():+6.1f}±{1.96 * c.std(ddof=1) / np.sqrt(len(c)):4.1f} {int((by > 0).sum())}/{int((by != 0).sum())}"
    pd.set_option("display.width", 250, "display.max_rows", 800, "display.max_columns", 40)
    print(f"\n== {'Superflex' if SF else '1QB'}: mean ± 95% CI, cohorts above zero / cohorts that differ")
    for arm in KIND_PLACEBO:
        a, p = idx[idx.arm == arm], idx[idx.arm == arm + "~"]
        for lab, d in (("signal's sells - live", a[mets] - live[mets]), ("random same-kind sells - live", p[mets] - live[mets]),
                       ("signal's sells - random same-kind sells", a[mets] - p[mets])):
            rows = []
            for grp, ranks in groups.items():
                for mode in MODES:
                    q = d.xs(mode, level="mode")
                    q = q[np.isin(q.index.get_level_values("rank"), ranks)]
                    extra = (a.xs(mode, level="mode").sell - live.xs(mode, level="mode").sell)
                    extra = extra[np.isin(extra.index.get_level_values("rank"), ranks)].mean()
                    rows.append({"ranks": grp, "mode": mode, "extra sells": round(extra, 2), **{c: cell(q[c]) for c in mets}})
            print(f"\n-- {arm}: {lab}")
            print(pd.DataFrame(rows).to_string(index=False))


def summarize(df):
    key = ["t", "rep", "wk", "rank"]
    pc = [f"p{k}" for k in range(1, H + 1)]
    base = df[df.arm == "none"].set_index(key)[pc + ["val1", "val3"]].add_suffix("_0")
    z = df[df.arm != "none"].join(base, on=key)
    z["yr1"] = (z.p1 - z.p1_0) * 100
    z["yrs3"] = np.where(z.t + 3 <= su.LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in (1, 2, 3)) * 100, np.nan)
    z["yrs5"] = np.where(z.t + 5 <= su.LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in range(1, 6)) * 100, np.nan)
    z["dval1"], z["dval3"] = z.val1 - z.val1_0, z.val3 - z.val3_0
    z["grp"] = pd.cut(z["rank"], [0, 3, 8, 12], labels=["1-3", "4-8", "9-12"]).astype(str)
    z["trades"] = z.sell + z.later + z.buy
    mets = ["yr1", "yrs3", "yrs5", "dval1", "dval3"]
    pd.set_option("display.width", 250, "display.max_rows", 800, "display.max_columns", 40)
    print(f"\n== {'Superflex' if SF else '1QB'}, FP top {POOL}, weeks {WEEKS}, {REPS} reps: vs no trades (title points summed; value = POOL - FP rank)")
    print(z.groupby(["grp", "mode", "arm"])[mets + ["sell", "later", "buy"]].mean().round(1).to_string())

    def cell(d):
        v = d.dropna()
        if v.empty:
            return ""
        c = v.groupby(level=["t", "rep", "wk"]).mean()
        h = v.groupby(np.where(v.index.get_level_values("t") <= 2020, "a", "b")).mean()
        by = v.groupby(level="t").mean()
        return (f"{c.mean():+6.1f}±{1.96 * c.std(ddof=1) / np.sqrt(len(c)):4.1f} "
                f"[{h.get('a', np.nan):+.1f}/{h.get('b', np.nan):+.1f}] {int((by > 0).sum())}/{len(by)}")
    print("\n== arm minus live, same team and window. mean ± 95% CI [cohorts 2018-20 / 2021-24] cohorts above zero / cohorts")
    idx = z.set_index(key + ["mode"])
    live = idx[idx.arm == "live"][mets]
    for arm in [a for a in ARMS if a != "live"]:
        d = idx[idx.arm == arm][mets] - live
        rows = []
        for grp, mode in [(g, m) for g in LENS_MODE for m in MODES]:
            q = d.xs(mode, level="mode")
            q = q[np.isin(q.index.get_level_values("rank"), {"1-3": [1, 2, 3], "4-8": range(4, 9), "9-12": range(9, 13)}[grp])]
            rows.append({"ranks": grp, "mode": mode, "share differ": round((q.abs().sum(axis=1) > 0).mean(), 2),
                         **{c: cell(q[c]) for c in mets}})
        print(f"\n-- {arm} - live")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n== arm minus its placebo, same team and window")
    for arm in PLACEBO:
        d = idx[idx.arm == arm][mets] - idx[idx.arm == arm + "~"][mets]
        rows = []
        for grp, mode in LENS_MODE.items():
            q = d.xs(mode, level="mode")
            q = q[np.isin(q.index.get_level_values("rank"), {"1-3": [1, 2, 3], "4-8": range(4, 9), "9-12": range(9, 13)}[grp])]
            rows.append({"ranks": grp, "mode": mode, **{c: cell(q[c]) for c in mets}})
        print(f"\n-- {arm} vs placebo")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n== pre - live and now - live by cohort, each group in its own mode (later cohorts are further from the rule fit)")
    for arm in ("pre", "now", "nosig", "noexempt"):
        d = idx[idx.arm == arm][mets] - live
        parts = []
        for grp, mode in LENS_MODE.items():
            q = d.xs(mode, level="mode")
            q = q[np.isin(q.index.get_level_values("rank"), {"1-3": [1, 2, 3], "4-8": range(4, 9), "9-12": range(9, 13)}[grp])]
            parts.append(q.groupby(level="t").mean().round(1).assign(group=grp))
        print(f"\n-- {arm} - live")
        print(pd.concat(parts).reset_index().set_index(["group", "t"]).to_string())
    print("\n== pre - live by window, each group in its own mode")
    d = idx[idx.arm == "pre"][mets] - live
    for grp, mode in LENS_MODE.items():
        q = d.xs(mode, level="mode")
        q = q[np.isin(q.index.get_level_values("rank"), {"1-3": [1, 2, 3], "4-8": range(4, 9), "9-12": range(9, 13)}[grp])]
        print(grp, mode)
        print(q.groupby(level="wk").mean().round(1).to_string())


if __name__ == "__main__":
    main()
