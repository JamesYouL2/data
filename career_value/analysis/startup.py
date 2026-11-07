"""Q: in a startup draft, go all in for year 1, punt year 1, or something else?

Simulation on real MFL startup boards (keeper ADP, veterans only, top 200) for drafts after seasons t = 2010-2020:
12-team snake, 16 rounds. Eleven teams draft by ADP with noise (log-normal, sd 0.2 on rank); the focal team
(random slot) picks from the next K = 8 players on the ADP board using a strategy:
  ADP        : best ADP (control, wins 1/12 by construction up to noise)
  win-now    : highest last-season PPG among the 8
  oldest     : oldest of the 8              youngest : youngest of the 8
  playbook   : README rules as a score -- QB (SF only) and WR <=24 and producers up, RB priced RB1-3 / RB <=24
               non-starters / TEs down (partly fit on these cohorts: an upper bound, not a clean test)
  fadeRB / young+fadeTE / preferRB ... : position skips or preferences inside the window (STRATS, ';'-separated)
  adj:y=0.06,rb=0.1,te=0 : adjusted value, no window -- log ADP rank + y*(age-26) + rb*[RB] + te*[TE] (+qb, wr);
               the lowest adjusted rank on the board is taken. Best region found: y 0.03-0.10, rb 0.1-0.2, te 0, wr 0;
               qb 0 in 1QB, qb -0.1 to -0.2 in SF (QBs moved up the board).
Every team fills QB 1 (SF 2), RB 3, WR 4, TE 1 by the last round. K (window) and NOISE are env vars.
Roster caps for every team: QB 3 (SF 4), TE 3, RB 7, WR 8. No trades, waivers or rookie drafts afterwards.
Score per season t+k: best possible lineup from season PPR totals, 1QB = QB RB RB WR WR WR TE FLEX, SF adds
a QB/RB/WR/TE slot. Reported per strategy and season k: P(top score in the 12-team league) = "title" proxy,
P(top 4), and lineup points vs the league mean.
LEAGUE=sf for superflex boards and lineups.
"""
import numpy as np
import pandas as pd

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import HERE, out_dir, rows
from long_horizon import SF

import os
TEAMS, ROUNDS, REPS, YEARS = 12, 16, 25, 10
K = int(os.environ.get("K", 8))            # reach window
NOISE = float(os.environ.get("NOISE", 0.2))  # other teams' ADP noise (log sd)
CAP = {"QB": 4 if SF else 3, "TE": 3, "RB": 7, "WR": 8}
LAST = 2025


NEED = {"QB": 2 if SF else 1, "RB": 3, "WR": 4, "TE": 1}   # every team fills these by the last round
STRATS = os.environ.get("STRATS", "ADP;win-now;oldest;youngest;young+fadeRB+fadeTE;"
                        "adj:y=0.06,rb=0.1;adj:y=0.06,rb=0.1,qb=-0.15").split(";")


def score(x, strat):
    """'young+fadeRB+fadeTE' = youngest in the window, RBs/TEs only if nothing else (or a roster need) is left;
    'fadeRB' alone = best ADP in the window that is not an RB; 'preferRB' = best-ADP RB in the window."""
    a, p, pp = x.age.to_numpy(), x.position.to_numpy(), x.ppg.fillna(0).to_numpy()
    if strat.startswith("adj:"):
        # adjusted value: log ADP rank pushed down by position discounts (rb, te, qb; 0.1 = rank x1.105)
        # and by age (y per year older than 26); the best adjusted player on the board is taken
        kv = dict(z.split("=") for z in strat[4:].split(","))
        lr = np.log(x.mkt_rank.to_numpy()) + float(kv.get("y", 0)) * (a - 26)
        for q in ("RB", "TE", "QB", "WR"):
            lr = lr + float(kv.get(q.lower(), 0)) * (p == q)
        return -lr
    if "+" in strat or strat.startswith(("fade", "prefer")):
        parts = strat.split("+")
        s = -a.astype(float) if parts[0] == "young" else -x.mkt_rank.to_numpy().astype(float)
        for q in ("RB", "TE", "QB", "WR"):
            if f"fade{q}" in parts:
                s = s - 1000 * (p == q)
            if f"fadeElite{q}" in parts:                 # only the top-6-priced at the position
                s = s - 1000 * ((p == q) & (x.mkt_pos_rank.to_numpy() <= 6))
            if f"prefer{q}" in parts:
                s = s + 1000 * (p == q)
        return s
    if strat == "win-now":
        return pp
    if strat == "oldest":
        return a
    if strat == "youngest":
        return -a
    if strat == "playbook":
        pr, ppr = x.mkt_pos_rank.to_numpy(), x.ppg_pos_rank.fillna(99).to_numpy()
        s = -x.mkt_rank.to_numpy() / 10.0
        s += np.where((p == "QB") & SF, 3, 0) + np.where((p == "WR") & (a < 25), 2, 0)
        s += np.where(pr >= ppr + 6, 1, 0)                                   # priced below production
        s -= np.where((p == "RB") & (pr <= 3), 3, 0) + np.where((p == "RB") & (a < 25) & (ppr > 12), 2, 0)
        s -= np.where(p == "TE", 2, 0)
        return s
    return -x.mkt_rank.to_numpy()   # ADP


def lineup(pts, pos):
    """best season lineup from a roster's season totals"""
    by = {q: np.sort(pts[pos == q])[::-1] for q in ("QB", "RB", "WR", "TE")}
    need = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    tot, rest = 0.0, []
    for q, n in need.items():
        v = by[q]
        tot += v[:n].sum()
        rest.append(v[n:] if q != "QB" else np.array([]))
        if q == "QB":
            qb_rest = v[n:]
    flex = np.sort(np.concatenate(rest))[::-1]
    tot += flex[:1].sum()
    if SF:
        sfl = np.sort(np.concatenate([flex[1:], qb_rest]))[::-1]
        tot += sfl[:1].sum()
    return tot


def draft(x, strat, slot, rng):
    n = len(x)
    rank = x.mkt_rank.to_numpy()
    board = rank * np.exp(rng.normal(0, NOISE, n))
    sc = score(x, strat)
    st_adj = strat.startswith("adj:")
    pos = x.position.to_numpy()
    avail = np.ones(n, bool)
    team = np.full(n, -1)
    cnt = [dict.fromkeys(CAP, 0) for _ in range(TEAMS)]
    order = np.argsort(rank)
    for r in range(ROUNDS):
        for i in (range(TEAMS) if r % 2 == 0 else range(TEAMS - 1, -1, -1)):
            ok = avail & np.array([cnt[i][q] < CAP[q] for q in pos])
            left = ROUNDS - r
            short = {q for q in NEED if cnt[i][q] < NEED[q]}
            if sum(NEED[q] - cnt[i][q] for q in short) >= left:      # must fill starters now
                ok = ok & np.isin(pos, list(short))
            if not ok.any():
                ok = avail
            if i == slot:
                cand = [j for j in order if ok[j]][:(len(order) if st_adj else K)]
                j = cand[int(np.argmax(sc[cand]))]
            else:
                idx = np.flatnonzero(ok)
                j = idx[np.argmin(board[idx])]
            avail[j], team[j] = False, i
            cnt[i][pos[j]] += 1
    return team


def main():
    x = rows(["mfl_dynasty"], 1, SF)
    x = x[x.t.between(2010, 2020)]
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = s.pivot_table(index="gsis_id", columns="season", values="ppr", fill_value=0.0)
    rng = np.random.default_rng(0)
    res = []
    strats = STRATS
    for t, g in x.groupby("t"):
        g = g.reset_index(drop=True)
        yrs = [t + k for k in range(1, YEARS + 1) if t + k <= LAST]
        P = np.stack([pts.reindex(g.gsis_id)[y].fillna(0).to_numpy() if y in pts else np.zeros(len(g)) for y in yrs], 1)
        pos = g.position.to_numpy()
        for rep in range(REPS):
            slot = rep % TEAMS
            seed = rng.integers(1 << 31)
            for st in strats:
                team = draft(g, st, slot, np.random.default_rng(seed))   # same noise for every strategy
                for k, y in enumerate(yrs, 1):
                    L = np.array([lineup(P[team == i, k - 1], pos[team == i]) for i in range(TEAMS)])
                    rk = (L > L[slot]).sum() + 1
                    mine = team == slot
                    res.append((t, rep, st, k, rk == 1, rk <= 4, (L[slot] / L.mean() - 1) * 100,
                                g.age[mine].mean(), (pos[mine] == "RB").sum(), (pos[mine] == "TE").sum()))
    r = pd.DataFrame(res, columns=["t", "rep", "strat", "k", "title", "top4", "pts", "age", "nRB", "nTE"])
    if K == 8 and NOISE == 0.2 and "STRATS" not in os.environ:
        r.to_csv(out_dir(SF) / "startup_sim.csv", index=False)
    print(f"== {'Superflex' if SF else '1QB'} startup sim K={K} noise={NOISE}: drafts t={x.t.min()}-{x.t.max()}, {REPS} reps each "
          f"(seasons 6-10 from drafts through {LAST - 6}-{LAST - 10})")
    f = r[r.k == 1].groupby("strat")[["age", "nRB", "nTE"]].mean().reindex(strats).round(1)
    print("roster: mean age, RBs, TEs drafted\n" + f.to_string())
    for col, lab in (("title", "P(top score) %, chance = 8.3"), ("top4", "P(top 4) %, chance = 33"),
                     ("pts", "lineup points vs league mean %")):
        tab = r.groupby(["strat", "k"])[col].mean().unstack() * (100 if col != "pts" else 1)
        print(f"\n-- {lab}, by season after draft")
        print(tab.reindex(strats).round(1).to_string())
    w = r.assign(win=pd.cut(r.k, [0, 1, 3, 5, 10], labels=["yr1", "yr2-3", "yr4-5", "yr6-10"]))
    print("\n-- titles per draft (expected, sum of P(top) over seasons); chance = 0.083/season")
    tt = w.groupby(["strat", "win", "t", "rep"], observed=True).title.sum().groupby(["strat", "win"], observed=True).mean().unstack()
    tt["yr1-3"] = tt[["yr1", "yr2-3"]].sum(axis=1)
    tt["yr1-5"] = tt[["yr1", "yr2-3", "yr4-5"]].sum(axis=1)
    tt["top4 yr1-3 %"] = r[r.k <= 3].groupby("strat").top4.mean() * 100
    tt["top4 yr1-5 %"] = r[r.k <= 5].groupby("strat").top4.mean() * 100
    print(tt.reindex(strats).round(2).to_string())
    print("\n-- by draft year, title % in years 1-3 (is the pattern stable?)")
    print((r[r.k <= 3].groupby(["strat", "t"]).title.mean().unstack() * 100).reindex(strats).round(0).to_string())


if __name__ == "__main__":
    main()
