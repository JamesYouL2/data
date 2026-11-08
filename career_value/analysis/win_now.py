"""Q: how high can a contender's title odds get, this season vs next, and when is a win-now trade worth it?

Leagues: 12 teams drafted from the FP dynasty pool of cohort t (veterans + rookies, 2018-24) with the startup.py
draft (every team drafts by FP ECR with log-normal noise, NOISE env, default 0.5 so team strength spreads out like
a league a few years in). Projected strength = best lineup of projected points, where a player's projection is the
pooled position curve of next-season PPR on FantasyFootballCalculator redraft ADP (isotonic; no redraft ADP = mean
of such players). Real PPR totals score seasons t+1..t+5 with the roster frozen (no trades, waivers, rookies).

Title model ("playoffs"): weekly team score ~ Normal(season lineup total / 17, SD x league mean), SD = 0.2 by
default (SD env); 14-week random head-to-head schedule, top 6 by wins (then points) make the playoffs, seeds 1-2
get byes, single-game rounds. "Top score" = best season total (no luck), an upper bound.

Trades: for the team at projected rank r, give its most valuable young player (age <= 24.5, by FP dynasty rank)
for the player aged 27+ on another team, priced within x0.8-1.25 of the young player's FP rank
that most improves the projected starting lineup (an equal-price win-now trade). Titles over seasons 1-5 minus the no-trade titles, and each side's FP rank a year
later (how much resale value is left). Windows use only drafts with all their seasons played.
LEAGUE=sf for superflex.
"""
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import HERE, out_dir, rows
from long_horizon import SF
import startup as su

REPS = int(os.environ.get("REPS", 40))
NSIM = int(os.environ.get("NSIM", 300))
SD = float(os.environ.get("SD", 0.2))
su.NOISE = float(os.environ.get("NOISE", 0.5))
H = 5
RANKS = (1, 2, 3, 4, 6, 9)


def projector(pts):
    """next-season PPR from redraft ADP, isotonic by position; fallback = mean for players without redraft ADP"""
    r = rows(["ffc_redraft"], 1, SF)[["gsis_id", "t", "mkt_rank"]]
    d = rows(["fp_dynasty"], 1, SF)[["gsis_id", "t", "position"]]
    z = d.merge(r, on=["gsis_id", "t"], how="left")
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


def titles(L, rng):
    """P(title) per team under the playoff model, and the top-score indicator"""
    n = len(L)
    m = L / 17.0
    sd = SD * m.mean()
    win = np.zeros(n)
    wins = np.zeros((NSIM, n))
    pf = np.zeros((NSIM, n))
    for _ in range(14):
        perm = np.argsort(rng.random((NSIM, n)), 1)
        sc = m + sd * rng.standard_normal((NSIM, n))
        a, b = perm[:, 0::2], perm[:, 1::2]
        sa, sb = np.take_along_axis(sc, a, 1), np.take_along_axis(sc, b, 1)
        np.put_along_axis(wins, a, np.take_along_axis(wins, a, 1) + (sa > sb), 1)
        np.put_along_axis(wins, b, np.take_along_axis(wins, b, 1) + (sb > sa), 1)
        pf += sc
    seed = np.argsort(-(wins * 1e4 + pf), 1)[:, :6]                    # seeds 1..6
    def game(x, y):
        sx = m[x] + sd * rng.standard_normal(len(x))
        sy = m[y] + sd * rng.standard_normal(len(y))
        return np.where(sx > sy, x, y)
    q1, q2 = game(seed[:, 2], seed[:, 5]), game(seed[:, 3], seed[:, 4])
    s1, s2 = game(seed[:, 0], q2), game(seed[:, 1], q1)                # 1 seed plays the 4/5 winner
    ch = game(s1, s2)
    np.add.at(win, ch, 1)
    top = np.zeros(n)
    top[np.argmax(L)] = 1
    return win / NSIM, top


def season_titles(team, P, pos, rng):
    out = []
    for k in range(P.shape[1]):
        L = np.array([su.lineup(P[team == i, k], pos[team == i]) for i in range(su.TEAMS)])
        out.append(titles(L, rng))
    return out


def main():
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: g.set_index("gsis_id").ppr.to_dict() for y, g in s.groupby("season")}
    proj = projector(pts)
    d = rows(["fp_dynasty"], 1, SF)
    nxt_rank = d.set_index(["gsis_id", "t"]).mkt_rank
    rng = np.random.default_rng(0)
    base, trades = [], []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        yrs = [y for y in range(t + 1, t + 1 + H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        pos, age, price = g.position.to_numpy(), g.age.to_numpy(), g.mkt_rank.to_numpy()
        for rep in range(REPS):
            team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
            strength = np.array([su.lineup(g.proj.to_numpy()[team == i], pos[team == i]) for i in range(su.TEAMS)])
            rk = (-strength).argsort().argsort() + 1                   # 1 = strongest projected
            st = season_titles(team, P, pos, rng)
            for k, (pt, top) in enumerate(st, 1):
                for i in range(su.TEAMS):
                    base.append((t, rep, k, rk[i], pt[i], top[i], strength[i] / strength.mean() * 100 - 100))
            for r, kind in [(r, kind) for r in RANKS for kind in ("best young", "mid young")]:
                i = int(np.flatnonzero(rk == r)[0])
                young = np.flatnonzero((team == i) & (age <= 24.5)
                                       & ((price >= 40) & (price <= 120) if kind == "mid young" else True))
                if not len(young):
                    continue
                give = young[np.argmin(price[young])]
                cand = np.flatnonzero((team != i) & (team >= 0) & (age >= 27)
                                      & (price >= price[give] * 0.8) & (price <= price[give] * 1.25))
                if not len(cand):
                    continue
                pj, mine = g.proj.to_numpy(), np.flatnonzero((team == i) & (np.arange(len(g)) != give))
                gain = [su.lineup(pj[np.append(mine, c)], pos[np.append(mine, c)]) for c in cand]
                get = cand[int(np.argmax(gain))]              # the biggest projected lineup upgrade
                t2 = team.copy()
                t2[give], t2[get] = team[get], i
                st2 = season_titles(t2, P, pos, rng)
                keep = [nxt_rank.get((g.gsis_id[j], t + 1), np.nan) for j in (give, get)]
                for k in range(len(yrs)):
                    trades.append((t, rep, r, kind, k + 1, st2[k][0][i] - st[k][0][i], st2[k][1][i] - st[k][1][i],
                                   pos[give], pos[get], price[give], keep[0], keep[1],
                                   g.proj[get] - g.proj[give]))
    b = pd.DataFrame(base, columns=["t", "rep", "k", "rank", "p", "top", "strength"])
    tr = pd.DataFrame(trades, columns=["t", "rep", "rank", "kind", "k", "dp", "dtop", "give_pos", "get_pos", "price",
                                       "give_next", "get_next", "dproj"])
    tag = f"noise{su.NOISE}_sd{SD}"
    b.to_csv(out_dir(SF) / f"win_now_base_{tag}.csv", index=False)
    tr.to_csv(out_dir(SF) / f"win_now_trades_{tag}.csv", index=False)
    ok = lambda z: z[z.t + z.k <= su.LAST]
    print(f"== {'Superflex' if SF else '1QB'}: drafts t=2018-24, {REPS} leagues each, draft noise {su.NOISE}, "
          f"weekly SD {SD}")
    print("projected strength vs league mean %, by rank:",
          b[b.k == 1].groupby("rank").strength.mean().round(0).to_dict())
    tab = ok(b).groupby(["rank", "k"]).p.mean().unstack() * 100
    print("\n-- P(title) % by projected rank (this season = 1), roster frozen; playoff model, chance 8.3")
    print(tab.round(1).to_string())
    print("\n-- P(top regular-season score) %, no luck")
    print((ok(b).groupby(["rank", "k"]).top.mean().unstack() * 100).round(1).to_string())
    for kind, z0 in tr.groupby("kind"):
        z = z0[z0.k == 1]
        print(f"\n-- equal-FP-price trade, {kind} player (<=24.5) out, 27+ player who most improves the lineup in")
        print(f"trades: {z.groupby('rank').size().to_dict()}; median FP rank traded {z.price.median():.0f}; "
              f"projected points gained {z.dproj.median():.0f}; FP rank a year later, median: young "
              f"{z.give_next.median():.0f} (out of top 200: {z.give_next.isna().mean():.0%}), old "
              f"{z.get_next.median():.0f} (out: {z.get_next.isna().mean():.0%})")
        for col, lab in (("dp", "playoff model"), ("dtop", "top-score model")):
            out = {}
            for k in range(1, H + 1):
                out[f"yr {k}"] = z0[(z0.k == k) & (z0.t + k <= su.LAST)].groupby("rank")[col].mean() * 100
            for w in (2, 3, 5):                                  # windows on drafts with all w seasons played
                zz = z0[(z0.k <= w) & (z0.t + w <= su.LAST)]
                out[f"yrs 1-{w}"] = zz.groupby("rank")[col].sum() / zz[zz.k == 1].groupby("rank").size() * 100
            print(f"change in P(title), percentage points ({lab})")
            print(pd.DataFrame(out).round(1).to_string())


if __name__ == "__main__":
    main()
