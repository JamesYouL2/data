"""Q: should a middle-of-the-pack team (projected 4th-8th of 12) follow the buy/sell site's contender calls, its
rebuilder calls, or a proposed middle mode? Backtest of pages/build/rules.py inside the win_now.py league sim.

Leagues as in win_now.py, but deeper: 12 teams draft all of the FP dynasty top POOL (default 300, so 25-man rosters)
of cohort t (2018-24), position caps scaled up from startup.py's 16 rounds; projected strength from redraft ADP, real PPR scoring seasons t+1..t+5, title odds from the head-to-head playoff model. Every player gets the
site's calls (rules.classify: rebuilder, contender and default/middle; rules.base_calls for the contender calls
before the middle rule, "contend_old") from his features at t: FP overall and positional
rank, age entering t+1, NFL draft round, rookie or not, PPG rank in season t (4+ games, the site's "producing now")
and the season t / t-1 blend for 30+ tiers. The middle call combines the two (now in rules.classify):
  same call -> that call; rebuilder Sell + contender Buy -> Hold; contender Buy + rebuilder Hold/Buy -> Buy;
  rebuilder Sell + contender Hold/Use -> Use, then sell if producing, else Sell; otherwise the contender call if
  producing, else the rebuilder call.
The market, injury and rest-of-season notes are not backtestable and are left out.

A team following mode m makes equal-price trades (other side priced within x0.8-1.25 of FP rank):
  Sell (any Sell call): traded now for the other team's player m would most like and would not itself sell
    (rebuild: the youngest; contend / middle: the biggest projected lineup gain).
  Use, then sell: kept for season 1, then traded at next year's FP price (none left if he drops out of the top POOL)
    for the most valuable player in that price window (rebuild: the youngest).
  Buy (any Buy call on another team's player): up to 3, giving the own player m values least in the price window
    (rebuild: the oldest; contend / middle: the smallest projected lineup loss).
  Hold, Lottery ticket, Cut: no move (rosters are frozen and bench players score nothing).
Control ("random"): 6 equal-price swaps of random own players with random partners, to separate the calls' edge
from what any equal-price trading does in this sim. Rosters are otherwise frozen. Outcomes, against the same team making no trades (common random numbers): change in
P(title) for seasons 1..5, and dynasty value held one and three years on (sum of max(0, 200 - FP rank) over the
roster, the site's roster-strength measure, extended to POOL - FP rank when POOL > 200; players out of the top
POOL count 0).
Caveat: the rules were fit on these same 2018-25 cohorts, so this tests how the calls combine, not out of sample.
LEAGUE=sf for superflex; REPS, NSIM, SD, NOISE as in win_now.py.
"""
import json
import os
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import HERE, out_dir, rows  # noqa: E402
from long_horizon import SF  # noqa: E402
import startup as su  # noqa: E402
import win_now as wn  # noqa: E402  (sets su.NOISE from NOISE, default 0.5)

sys.path.insert(0, str(HERE.parent / "pages" / "build"))
import rules as R  # noqa: E402

REPS = int(os.environ.get("REPS", 20))
POOL = int(os.environ.get("POOL", 300))
su.CAP = {q: round(c * max(1, POOL / 192)) for q, c in su.CAP.items()}
su.ROUNDS = max(su.ROUNDS, POOL // su.TEAMS)
H = 5
F = "sf" if SF else "1qb"
MODES = ("contend", "contend_old", "rebuild", "middle", "random")
NRAND = 6
NBUY = 3
POS = ["QB", "RB", "WR", "TE"]


kind = R.kind


def pool():
    """FP dynasty rows (market_rows layout) for the top POOL of each cohort; the stored market rows stop at 200"""
    if POOL <= 200:
        return rows(["fp_dynasty"], 1, SF)
    sys.path.insert(0, str(HERE))
    import market_residuals as M
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    seasons = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    x, _ = M.prepare(M.fp_dynasty(panel), panel, M.outcomes(), seasons, 1, top_n=POOL)
    return x


def projector(pts, d):
    """win_now.projector, fit on this pool: next-season PPR from redraft ADP, isotonic by position"""
    from sklearn.isotonic import IsotonicRegression
    r = rows(["ffc_redraft"], 1, SF)[["gsis_id", "t", "mkt_rank"]]
    z = d[["gsis_id", "t", "position"]].merge(r, on=["gsis_id", "t"], how="left")
    z["nxt"] = [pts.get(t + 1, {}).get(g, 0.0) for g, t in zip(z.gsis_id, z.t)]
    fit, miss = {}, {}
    for q, g in z.groupby("position"):
        h = g[g.mkt_rank.notna()]
        fit[q] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(np.log(h.mkt_rank), h.nxt)
        miss[q] = g[g.mkt_rank.isna()].nxt.mean()

    def proj(x):
        rk = x.gsis_id.map(r[r.t == x.t.iloc[0]].set_index("gsis_id").mkt_rank)
        out = np.array([miss[q] for q in x.position], float)
        for q in fit:
            m = (x.position == q).to_numpy() & rk.notna().to_numpy()
            if m.any():
                out[m] = fit[q].predict(np.log(rk[m].to_numpy()))
        return out
    return proj


def prod_ranks(s):
    """season -> {gsis_id: PPG positional rank}, for 4+ games (current) and 6+ games (last season), as build_data.py"""
    out = {}
    for min_g in (4, 6):
        q = s[s.position.isin(POS) & (s.games >= min_g)].copy()
        q["prk"] = q.groupby(["season", "position"]).ppg.rank(ascending=False, method="min")
        out[min_g] = {y: z.set_index("gsis_id").prk.to_dict() for y, z in q.groupby("season")}
    return out


def calls(g, t, pr, look):
    """mode -> array of actions for every player in cohort t's pool"""
    x = pd.DataFrame({
        "position": g.position, "age": g.age, f"r_{F}": g.mkt_rank, f"pr_{F}": g.pos_rank_fp,
        "rd": g.draft_round_f.fillna(8), "dy": np.where(g.row_type == "rookie", t + 1, 0),
        "prk": g.gsis_id.map(pr[4].get(t, {})),
    })
    x["prod_rank"] = pd.concat([x.prk, g.gsis_id.map(pr[6].get(t - 1, {}))], axis=1).mean(axis=1)
    ctx = {"ROOKIE": t + 1, "LOOK": look}
    out = {m: [] for m in MODES if m != "random"}
    for row in x.itertuples(index=False):
        _, (ra, _), (ca, _), (ma, _) = R.classify(row, F, ctx)
        out["rebuild"].append(ra)
        out["contend"].append(ca)
        out["middle"].append(ma)
        out["contend_old"].append(R.base_calls(row, F, ctx)[2][0])
    return {m: np.array(v, dtype=object) for m, v in out.items()}


def season_titles(T, P, pos, seeds):
    """P(title) of every team per season; T is players x seasons (team ids); common random numbers via seeds"""
    out = []
    for k in range(P.shape[1]):
        L = np.array([su.lineup(P[T[:, k] == i, k], pos[T[:, k] == i]) for i in range(su.TEAMS)])
        out.append(wn.titles(L, np.random.default_rng(seeds[k]))[0])
    return np.array(out)                                     # seasons x teams


def policy(i, m, team, g, act, nxt1, rng=None):
    """per-season team assignment after team i follows mode m; also (sells now, sells later, buys)"""
    if m == "random":                                        # control: NRAND equal-price swaps with random partners
        price, cur, moved, n = g.mkt_rank.to_numpy(), team.copy(), np.zeros(len(team), bool), 0
        for j in rng.permutation(np.flatnonzero(team == i)):
            if n >= NRAND:
                break
            c = np.flatnonzero((cur >= 0) & (cur != i) & ~moved & (price >= price[j] * 0.8) & (price <= price[j] * 1.25))
            if len(c):
                get = rng.choice(c)
                cur[j], cur[get] = cur[get], i
                moved[[j, get]] = True
                n += 1
        return np.repeat(cur[:, None], H, 1), (n, 0, 0)
    pj, pos, age, price = g.proj.to_numpy(), g.position.to_numpy(), g.age.to_numpy(), g.mkt_rank.to_numpy()
    k = np.array([kind(a) for a in act[m]])
    cur, moved = team.copy(), np.zeros(len(team), bool)

    def lu(ids):
        return su.lineup(pj[ids], pos[ids])

    def window(p, pr):
        return (pr >= p * 0.8) & (pr <= p * 1.25)

    other = lambda: (cur >= 0) & (cur != i) & ~moved           # noqa: E731
    no_take = ("sell", "cut") if m in ("contend", "contend_old") else ("sell", "cut", "use")
    n_now = n_buy = n_later = 0
    for j in sorted(np.flatnonzero((cur == i) & (k == "sell")), key=lambda j: price[j]):
        c = np.flatnonzero(other() & window(price[j], price) & ~np.isin(k, no_take))
        if not len(c):
            continue
        if m == "rebuild":
            get = c[np.lexsort((price[c], age[c]))[0]]
        else:
            mine = np.flatnonzero((cur == i) & (np.arange(len(cur)) != j))
            get = c[int(np.argmax([lu(np.append(mine, x)) for x in c]))]
        cur[j], cur[get] = cur[get], i
        moved[[j, get]] = True
        n_now += 1
    tg = np.flatnonzero(other() & (k == "buy"))
    if m == "rebuild":
        tg = tg[np.argsort(age[tg])]
    else:
        mine = np.flatnonzero(cur == i)
        base = lu(mine)
        tg = tg[np.argsort([-(lu(np.append(mine, x)) - base) for x in tg])]
    for b in tg:
        if n_buy >= NBUY:
            break
        give = np.flatnonzero((cur == i) & ~moved & window(price[b], price) & (k != "buy"))
        if not len(give):
            continue
        if m == "rebuild":
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
            continue                                         # out of the top 200 a year on: nothing to sell
        c = np.flatnonzero(other() & window(p1, nxt1) & ~np.isin(k, no_take))
        if not len(c):
            continue
        get = c[np.lexsort((nxt1[c], age[c]))[0]] if m == "rebuild" else c[np.argmin(nxt1[c])]
        T[j, 1:], T[get, 1:] = cur[get], i
        moved[[j, get]] = True
        n_later += 1
    return T, (n_now, n_later, n_buy)


def value(T, i, col, rk):
    on = T[:, col] == i
    return np.nansum(np.maximum(0, max(200, POOL) - rk[on])) if rk is not None else np.nan


def main():
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    pr = prod_ranks(s)
    look = {(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"])
            for r in json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]}
    d = pool()
    d = d[d.t <= 2024]
    proj = projector(pts, d)
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    rng = np.random.default_rng(1)
    out = []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        act = calls(g, t, pr, look)
        yrs = [y for y in range(t + 1, t + 1 + H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        P = np.pad(P, ((0, 0), (0, H - P.shape[1])))           # unplayed seasons: dropped from the summaries
        pos = g.position.to_numpy()
        nxt1 = g.gsis_id.map(rank_at.get(t + 1, pd.Series(dtype=float))).to_numpy(float)
        rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
        rk3 = g.gsis_id.map(rank_at[t + 3]).to_numpy(float) if t + 3 in rank_at else None
        for rep in range(REPS):
            team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
            strength = np.array([su.lineup(g.proj.to_numpy()[team == i], pos[team == i]) for i in range(su.TEAMS)])
            rkk = (-strength).argsort().argsort() + 1
            seeds = rng.integers(1 << 31, size=H)
            T0 = np.repeat(team[:, None], H, 1)
            p0 = season_titles(T0, P, pos, seeds)
            for i in range(su.TEAMS):
                v1, v3 = value(T0, i, 1, rk1), value(T0, i, 2, rk3)
                out.append((t, rep, rkk[i], "none", *p0[:, i], v1, v3, 0, 0, 0))
                for m in MODES:
                    T, n = policy(i, m, team, g, act, nxt1, rng)
                    p = season_titles(T, P, pos, seeds)[:, i] if n != (0, 0, 0) else p0[:, i]
                    out.append((t, rep, rkk[i], m, *p, value(T, i, 1, rk1), value(T, i, 2, rk3), *n))
        print(f"cohort {t} done", flush=True)
    cols = ["t", "rep", "rank", "mode"] + [f"p{k}" for k in range(1, H + 1)] + ["val1", "val3", "sell_now",
                                                                                 "sell_later", "buys"]
    df = pd.DataFrame(out, columns=cols)
    tag = f"top{POOL}_noise{su.NOISE}_sd{wn.SD}"
    df.to_csv(out_dir(SF) / f"middle_mode_{tag}.csv", index=False)
    summarize(df)


def summarize(df):
    base = df[df["mode"] == "none"].set_index(["t", "rep", "rank"])
    z = df[df["mode"] != "none"].merge(base[[f"p{k}" for k in range(1, H + 1)] + ["val1", "val3"]],
                                       left_on=["t", "rep", "rank"], right_index=True, suffixes=("", "_0"))
    for k in range(1, H + 1):
        z[f"d{k}"] = (z[f"p{k}"] - z[f"p{k}_0"]) * 100
    z["dval1"], z["dval3"] = z.val1 - z.val1_0, z.val3 - z.val3_0
    z["grp"] = pd.cut(z["rank"], [0, 3, 8, 12], labels=["1-3", "4-8", "9-12"])
    LAST = su.LAST
    rowsout = []
    for (grp, m), q in z.groupby(["grp", "mode"], observed=True):
        r = {"ranks": grp, "mode": m, "n": len(q)}
        r["yr1"] = q.d1.mean()
        for w in (3, 5):
            qq = q[q.t + w <= LAST]
            r[f"yrs1-{w}"] = qq[[f"d{k}" for k in range(1, w + 1)]].sum(axis=1).mean()
        r["val+1y"] = q[q.t + 1 <= 2024].dval1.mean()
        r["val+3y"] = q[q.t + 3 <= 2024].dval3.mean()
        r["trades"] = (q.sell_now + q.sell_later + q.buys).mean()
        rowsout.append(r)
    tab = pd.DataFrame(rowsout).set_index(["ranks", "mode"])
    print(f"\n== {'Superflex' if SF else '1QB'}, FP top {POOL}: following each mode vs no trades. P(title) change in percentage "
          f"points summed over the window; dynasty value change (sum of {max(200, POOL)} - FP rank) held 1 and 3 years on")
    print(tab.round(1).to_string())
    by = z.groupby(["rank", "mode"]).apply(lambda q: q[q.t + 3 <= LAST][["d1", "d2", "d3"]].sum(axis=1).mean(),
                                           include_groups=False).unstack()
    print("\n-- P(title) change, seasons 1-3 summed, by projected rank")
    print(by.round(1).to_string())
    print("\n-- trades per team (sell now / sell after the season / buys), ranks 4-8")
    print(z[z.grp == "4-8"].groupby("mode")[["sell_now", "sell_later", "buys"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
