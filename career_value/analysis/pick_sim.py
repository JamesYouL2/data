"""Q: should a contending, a middle or a rebuilding team sell or buy next year's rookie picks?

middle_mode.py's league (12 teams draft the FP top POOL of cohort t in May of t + 1, real PPR scoring after that), with two changes:
  Picks.  Every team also holds next year's 1st-4th. After season 1 the rookie draft runs in reverse order of season 1's real lineup
          points, and each slot gets the rookie the market ranked there the next May (class t + 2, as picks.py). Rookies play from
          season 2. Without a trade this is the baseline, so every team's rookies count in it.
  Prices. Trades clear at the market's rank on the May snapshot (Dynasty Daddy; SOURCE=ktc or fantasycalc, in the league's format).
          A pick's rank is 1 + the number of listed players valued above it. Its tier is what the market could know: the owner's
          projected rank (1-4 late, 5-8 mid, 9-12 early). The slot it turns into comes from the real season, so a contender's
          "late" 1st is sometimes early.
One team makes one trade, other side priced within x0.8-1.25 of the pick's rank:
  sell R  its own round R pick for another team's player. win: the biggest projected lineup gain; young: the youngest;
          rand: a random one
  buy R   a random other team's round R pick for one of its own players. win: the smallest projected lineup loss; old: the
          oldest; rand: a random one
rand is the control: what swapping a pick for any player at its price does. Outcomes against the same league with no trade
(common random numbers): change in P(title) in season 1, seasons 1-3 and seasons 2-3, and dynasty value held at the Mays 1 and 3
years on (sum of POOL - market rank, rookies included).
Caveats: May trades only, where the site is used in season. Few cohorts: 3-year results have 2020-22 only. Picks two or more years
out aren't modelled, and neither is anything a team does with a pick besides using it. Rosters are otherwise frozen and bench
players score nothing, so a rookie only counts once he makes a lineup.
Writes outputs/[sf/]pick_sim_<source>.csv.gz. LEAGUE=sf for superflex; REPS (default 10), JOBS.
"""
import os
import pathlib
import sys
from multiprocessing import Pool

os.environ.setdefault("POOL", "340")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import HERE, out_dir  # noqa: E402
import middle_mode as mm  # noqa: E402
import startup as su  # noqa: E402
import win_now as wn  # noqa: E402,F401  (sets su.NOISE)
from long_horizon import SF  # noqa: E402
import market_residuals as M  # noqa: E402
import picks as PK  # noqa: E402

REPS = int(os.environ.get("REPS", 10))
JOBS = int(os.environ.get("JOBS", 5))
SOURCE = os.environ.get("SOURCE", "ktc")
POOL, H = mm.POOL, mm.H
ROUNDS = (1, 2, 3)                  # traded; round 4 is drafted but sits past most price lists
NR = 4 * su.TEAMS
TASTES = {"sell": ("win", "young", "rand"), "buy": ("win", "old", "rand")}
OWN = {"1-3": {"sell": "win", "buy": "win"}, "4-8": {"sell": "win", "buy": "win"}, "9-12": {"sell": "young", "buy": "old"}}
COHORTS = range(2020, 2025)


def tier(rank):
    return "late" if rank <= 4 else "mid" if rank <= 8 else "early"


def run_cohort(args):
    t, g, rk, pickrank, P, mk1, mk3, seed = args
    rng = np.random.default_rng(seed)
    n = len(g)
    pos = np.concatenate([g.position.to_numpy(), rk.position.to_numpy()])
    age, pj, price = g.age.to_numpy(), g.proj.to_numpy(), g.mrank.to_numpy(float)
    ok = ~np.isnan(price)
    out = []

    def lu(ids):
        return su.lineup(pj[ids], pos[ids])

    for rep in range(REPS):
        team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
        strength = np.array([lu(np.flatnonzero(team == k)) for k in range(su.TEAMS)])
        rkk = (-strength).argsort().argsort() + 1
        seeds = rng.integers(1 << 31, size=H)
        memo = {}

        def build(cur, owner):
            """players x seasons team ids, rookies appended; owner[r - 1, o] holds team o's round r pick"""
            pts1 = np.array([su.lineup(P[:n, 0][cur == k], pos[:n][cur == k]) for k in range(su.TEAMS)])
            slot = pts1.argsort().argsort()                              # 0 = worst team, picks first
            T = np.full((n + NR, H), -1)
            T[:n] = cur[:, None]
            for r in range(4):
                T[n + r * su.TEAMS + slot, 1:] = owner[r][:, None]
            return T

        def evaluate(T):
            key = T.tobytes()
            if key not in memo:
                memo[key] = mm.season_titles(T, P, pos, seeds)
            return memo[key]

        def val(T, col, r, i):
            return np.nansum(np.maximum(0, POOL - r[T[:, col] == i])) if r is not None else np.nan

        def row(T, i, arm, taste, done):
            return (t, rep, rkk[i], arm, taste, *evaluate(T)[:, i], val(T, 1, mk1, i), val(T, 2, mk3, i), done)

        own0 = np.tile(np.arange(su.TEAMS), (4, 1))
        T0 = build(team, own0)
        for i in range(su.TEAMS):
            out.append(row(T0, i, "none", "-", 0))
            mine = np.flatnonzero(team == i)
            for r in ROUNDS:
                # sell: team i's own pick for another team's player
                p = pickrank.get((r, tier(rkk[i])), np.nan)
                c = np.flatnonzero((team != i) & (team >= 0) & ok & (price >= p * 0.8) & (price <= p * 1.25))
                for taste in TASTES["sell"]:
                    if not len(c):
                        out.append(row(T0, i, f"sell{r}", taste, 0))
                        continue
                    get = (rng.choice(c) if taste == "rand" else c[np.lexsort((price[c], age[c]))[0]] if taste == "young"
                           else c[int(np.argmax([lu(np.append(mine, x)) for x in c]))])
                    cur, owner = team.copy(), own0.copy()
                    owner[r - 1, i], cur[get] = team[get], i
                    out.append(row(build(cur, owner), i, f"sell{r}", taste, 1))
                # buy: a random other team's pick for one of team i's players
                o = rng.choice([k for k in range(su.TEAMS) if k != i])
                p = pickrank.get((r, tier(rkk[o])), np.nan)
                c = np.flatnonzero((team == i) & ok & (price >= p * 0.8) & (price <= p * 1.25))
                for taste in TASTES["buy"]:
                    if not len(c):
                        out.append(row(T0, i, f"buy{r}", taste, 0))
                        continue
                    give = (rng.choice(c) if taste == "rand" else c[np.argmax(age[c])] if taste == "old"
                            else c[int(np.argmax([lu(mine[mine != x]) for x in c]))])
                    cur, owner = team.copy(), own0.copy()
                    owner[r - 1, o], cur[give] = i, o
                    out.append(row(build(cur, owner), i, f"buy{r}", taste, 1))
    print(f"cohort {t} done", flush=True)
    return out


def main():
    panel = pd.read_parquet(HERE / "data" / "panel.parquet")
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    ppos = panel.sort_values("t").groupby("gsis_id").position.last()
    v = M.dd_markets(panel)[f"dd_{SOURCE}"]
    v = v.assign(position=v.gsis_id.map(ppos), value=-v.mkt).dropna(subset=["position"])
    v["rank"] = v.groupby("t").mkt.rank(method="first")
    v = v.merge(panel[["gsis_id", "t", "row_type"]], on=["gsis_id", "t"], how="left")
    mrank = {t: z.set_index("gsis_id")["rank"] for t, z in v.groupby("t")}
    pk = PK.pick_prices()
    pk = pk[pk.market == f"dd_{SOURCE}"]
    d = mm.pool()
    d = d[d.t <= 2024]
    proj = mm.projector(pts, d)
    rng = np.random.default_rng(29)
    jobs, info = [], []
    for t in COHORTS:
        q = pk[(pk.t == t) & (pk["class"] == t + 2)]
        vt = v[v.t == t]
        rk = v[(v.t == t + 1) & (v.row_type == "rookie")].sort_values("mkt").head(NR)
        if q.empty or vt.empty or len(rk) < NR:
            print(f"cohort {t}: skipped ({len(q)} pick prices, {len(vt)} players priced, {len(rk)} rookies)")
            continue
        pickrank = {(int(r["round"]), r.tier): 1 + int((vt.value > r.value).sum()) for _, r in q.iterrows()}
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        g["mrank"] = g.gsis_id.map(mrank[t])
        ids = pd.concat([g.gsis_id, rk.gsis_id], ignore_index=True)
        yrs = [y for y in range(t + 1, t + 1 + H) if y <= su.LAST]
        P = np.stack([ids.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        P = np.pad(P, ((0, 0), (0, H - P.shape[1])))
        P[len(g):, 0] = 0                                                # next year's class isn't on a roster in season 1
        mk1 = ids.map(mrank[t + 1]).to_numpy(float) if t + 1 in mrank else None
        mk3 = ids.map(mrank[t + 3]).to_numpy(float) if t + 3 in mrank else None
        info.append({"t": t, "priced": int(g.mrank.notna().sum()),
                     **{f"{r}{tr[0]}": pickrank.get((r, tr)) for r in (1, 2, 3, 4) for tr in ("early", "mid", "late")}})
        jobs.append((t, g, rk[["gsis_id", "position"]].reset_index(drop=True), pickrank, P, mk1, mk3, int(rng.integers(1 << 31))))
    print(f"== {'Superflex' if SF else '1QB'}, {SOURCE}: market rank of next year's picks at each May (e / m / l = early / mid / late), "
          f"and how many of the FP top {POOL} the market prices")
    print(pd.DataFrame(info).to_string(index=False))
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "rank", "arm", "taste"] + [f"p{k}" for k in range(1, H + 1)] + ["val1", "val3", "done"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / f"pick_sim_{SOURCE}.csv.gz", index=False)
    summarize(df)


def summarize(df):
    key = ["t", "rep", "rank"]
    pc = [f"p{k}" for k in range(1, H + 1)]
    base = df[df.arm == "none"].set_index(key)[pc + ["val1", "val3"]].add_suffix("_0")
    z = df[df.arm != "none"].join(base, on=key)
    z["yr1"] = (z.p1 - z.p1_0) * 100
    z["yrs2-3"] = np.where(z.t + 3 <= su.LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in (2, 3)) * 100, np.nan)
    z["yrs1-3"] = z.yr1 + z["yrs2-3"]
    z["v1"], z["v3"] = z.val1 - z.val1_0, z.val3 - z.val3_0
    z["grp"] = pd.cut(z["rank"], [0, 3, 8, 12], labels=list(OWN)).astype(str)
    mets = ["yr1", "yrs2-3", "yrs1-3", "v1", "v3"]
    pd.set_option("display.width", 250, "display.max_rows", 400)

    def cell(q, c):
        x = q.dropna(subset=[c])
        if x.empty:
            return ""
        cl = x.groupby(["t", "rep"])[c].mean()
        by = x.groupby("t")[c].mean()
        return f"{cl.mean():+6.1f}±{1.96 * cl.std(ddof=1) / np.sqrt(len(cl)):4.1f} {int((by > 0).sum())}/{len(by)}"

    def table(title, pick):
        rows = []
        for grp in OWN:
            for arm in [f"{a}{r}" for a in ("sell", "buy") for r in ROUNDS]:
                taste = pick(grp, arm[:-1])
                q = z[(z.grp == grp) & (z.arm == arm) & (z.taste == taste)]
                rows.append({"ranks": grp, "trade": arm, "taste": taste, **{c: cell(q, c) for c in mets},
                             "made": f"{q.done.mean() * 100:.0f}%"})
        print(f"\n-- {title}")
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n== {'Superflex' if SF else '1QB'}, {SOURCE}, cohorts {sorted(z.t.unique())}, {z.rep.nunique()} reps. One pick trade against no "
          "trade: mean ± 95% CI, cohorts above zero / cohorts. Titles in points; v1, v3 = dynasty value at the May 1 and 3 years on. "
          "Teams that found no trade count as 0")
    table("each team's own taste (contenders and middle teams: lineup; rebuilders: age)", lambda grp, a: OWN[grp][a])
    table("control: a random player at the pick's price", lambda grp, a: "rand")
    print("\n-- by cohort, own taste")
    parts = []
    for grp in OWN:
        q = z[(z.grp == grp) & (z.taste == z.arm.str[:-1].map(OWN[grp]))]
        parts.append(q.groupby(["arm", "t"])[mets].mean().round(1).assign(ranks=grp))
    print(pd.concat(parts).reset_index().set_index(["ranks", "arm", "t"]).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        summarize(pd.read_csv(out_dir(SF) / f"pick_sim_{SOURCE}.csv.gz"))
    else:
        main()
