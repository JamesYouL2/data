"""Q: when should a team sell a producing older player: now, after this season, after next season, or never?

middle_mode.py's league (12 teams draft the FP top 340 in May, real PPR scoring, rosters otherwise frozen). A producing player finished
the season before inside his position's starter line by points per game (QB12, or QB24 in superflex; RB24; WR30; TE12; 4+ games).
Classes by position and age entering the season. One team sells every player it holds in a class at one of three dates, each for the
most valuable player on another team within x0.8-1.25 of his FP rank at that date:
  now      this May            after 1   next May, after playing season 1            after 2   the May after, having played two seasons
A player off the top 340 by the sale date can't be sold: that is the risk of waiting. The same partner rule at every date, so the
dates compare; against holding him throughout.
Team types by the May projection: Contending 1st-3rd, Default 4th-9th, Rebuilding 10th-12th.
Outcomes against holding: title odds in each of seasons 1-3 and summed, and value (sum of 340 - FP rank) 1 and 3 years on; mean, z with
leagues as clusters, cohorts above zero / cohorts.
Caveats: the real players in a class are few (listed), so a class is a handful of careers replayed across leagues; FP prices; the
partner is the best player in the price window at every date, which flatters every sale equally.
Writes outputs/[sf/]sell_timing_sim.csv.gz. LEAGUE=sf for superflex; REPS (default 30), JOBS. `table` reprints from the file.
"""
import os
import pathlib
import sys
from multiprocessing import Pool

os.environ.setdefault("POOL", "340")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import call_strength as cs  # noqa: E402
import middle_mode as mm  # noqa: E402
import startup as su  # noqa: E402
from common import HERE, out_dir  # noqa: E402

R, SF, F = cs.R, cs.SF, cs.F
REPS = int(os.environ.get("REPS", 30))
JOBS = int(os.environ.get("JOBS", 6))
TYPE = {**{r: "Contending (1-3)" for r in (1, 2, 3)}, **{r: "Default (4-9)" for r in range(4, 10)}, **{r: "Rebuilding (10-12)" for r in range(10, 13)}}
WHEN = {0: "now", 1: "after 1", 2: "after 2"}
BANDS = {"RB": [(0, 26, "25-"), (26, 28, "26-27"), (28, 30, "28-29"), (30, 99, "30+")], "WR": [(0, 28, "27-"), (28, 30, "28-29"), (30, 99, "30+")],
         "TE": [(0, 29, "28-"), (29, 99, "29+")], "QB": [(0, 32, "31-"), (32, 99, "32+")]}


def label(pos, age, prk):
    if not (prk <= R.START[F][pos]):
        return ""
    return next(f"{pos} {name}" for lo, hi, name in BANDS[pos] if lo <= age < hi)


def run_cohort(args):
    t, g, P, cls, prices, rk1, rk3, seed = args
    rng = np.random.default_rng(seed)
    pos = g.position.to_numpy()
    old = cls != ""
    out = []

    def val(T, i, col, rk):
        return np.nansum(np.maximum(0, mm.POOL - rk[T[:, col] == i])) if rk is not None else np.nan
    for rep in range(REPS):
        team = su.draft(g, "ADP", -1, np.random.default_rng(rng.integers(1 << 31)))
        strength = np.array([su.lineup(g.proj.to_numpy()[team == i], pos[team == i]) for i in range(su.TEAMS)])
        rkk = (-strength).argsort().argsort() + 1
        seeds = rng.integers(1 << 31, size=mm.H)
        T0 = np.repeat(team[:, None], mm.H, 1)
        p0 = mm.season_titles(T0, P, pos, seeds)
        for i in range(su.TEAMS):
            for c in sorted(set(cls[(team == i) & old])):
                J = np.flatnonzero((team == i) & (cls == c))
                out.append((t, rep, rkk[i], c, -1, *p0[:4, i], val(T0, i, 1, rk1), val(T0, i, 2, rk3), 0, len(J)))
                for k, pr in enumerate(prices):
                    if pr is None:
                        continue
                    T, moved, n = T0.copy(), np.zeros(len(team), bool), 0
                    for j in J[np.argsort(pr[J])]:
                        if np.isnan(pr[j]):
                            continue
                        c_ = np.flatnonzero((T[:, k] >= 0) & (T[:, k] != i) & ~moved & ~old & (pr >= pr[j] * 0.8) & (pr <= pr[j] * 1.25))
                        if not len(c_):
                            continue
                        x = c_[np.argmin(pr[c_])]
                        T[j, k:], T[x, k:] = T[x, k], i
                        moved[[j, x]] = True
                        n += 1
                    p = mm.season_titles(T, P, pos, seeds)[:, i] if n else p0[:, i]
                    out.append((t, rep, rkk[i], c, k, *p[:4], val(T, i, 1, rk1), val(T, i, 2, rk3), n, len(J)))
    print(f"cohort {t} done", flush=True)
    return out


def main():
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    pr4 = mm.prod_ranks(s)[4]
    d = mm.pool()
    d = d[d.t <= 2024]
    proj = mm.projector(pts, d)
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    rng = np.random.default_rng(31)
    jobs, who = [], []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        prk = g.gsis_id.map(pr4.get(t, {})).to_numpy(float)
        cls = np.array([label(p, a, r) if r == r else "" for p, a, r in zip(g.position, g.age, prk)], dtype=object)
        who += [(t, c, n) for c, n in zip(cls, g.display_name if "display_name" in g else g.gsis_id) if c]

        def at(y):
            if y not in rank_at:
                return None
            r = g.gsis_id.map(rank_at[y]).to_numpy(float)
            return np.where(r <= mm.POOL, r, np.nan)
        yrs = [y for y in range(t + 1, t + 1 + mm.H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        P = np.pad(P, ((0, 0), (0, mm.H - P.shape[1])))
        rk = lambda y: g.gsis_id.map(rank_at[y]).to_numpy(float) if y in rank_at else None  # noqa: E731
        jobs.append((t, g, P, cls, [g.mkt_rank.to_numpy(float), at(t + 1), at(t + 2)], rk(t + 1), rk(t + 3), int(rng.integers(1 << 31))))
    w = pd.DataFrame(who, columns=["t", "cls", "player"])
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 150)
    print(f"== {'Superflex' if SF else '1QB'}: producing players by class: player-seasons, distinct players")
    print(w.groupby("cls").agg(seasons=("player", "size"), players=("player", "nunique")).to_string())
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "rank", "cls", "k", "p1", "p2", "p3", "p4", "val1", "val3", "n", "held"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / "sell_timing_sim.csv.gz", index=False)
    summarize(df)


def summarize(df):
    pd.set_option("display.width", 280, "display.max_rows", 600)
    key = ["t", "rep", "rank", "cls"]
    z = df.copy()
    for k in (1, 2, 3):
        z[f"s{k}"] = np.where(z.t + k <= su.LAST, z[f"p{k}"] * 100, np.nan)
    z["yrs2"], z["yrs3"] = z.s1 + z.s2, z.s1 + z.s2 + z.s3
    z["v1"] = np.where(z.t + 1 <= 2024, z.val1, np.nan)
    z["v3"] = np.where(z.t + 3 <= 2024, z.val3, np.nan)
    mets = ["s1", "s2", "s3", "yrs2", "yrs3", "v1", "v3"]
    base = z[z.k == -1].set_index(key)[mets]
    z["team"] = z["rank"].map(TYPE)
    rows = []
    for (c, team, k), q in z[z.k >= 0].groupby(["cls", "team", "k"]):
        q = q.set_index(key)
        d = q[mets] - base.loc[q.index]
        r = {"class": c, "team": team, "sell": WHEN[k], "teams": len(q), "sold of held": f"{q.n.sum() / q.held.sum() * 100:.0f}%"}
        for m in mets:
            v = d[m].dropna()
            if v.groupby(level=["t", "rep"]).ngroups < 5:
                r[m] = ""
                continue
            lg, by = v.groupby(level=["t", "rep"]).mean(), v.groupby(level="t").mean()
            se = lg.std(ddof=1) / np.sqrt(len(lg))
            r[m] = f"{lg.mean():+6.1f} z{lg.mean() / se if se > 0 else 0:+5.1f} {int((by > 0).sum())}/{len(by)}"
        rows.append(r)
    t = pd.DataFrame(rows)
    print(f"\n== {'Superflex' if SF else '1QB'}: selling a class's producing players at each date, against holding them. s1-s3 = title odds in each season (points)")
    for c, q in t.groupby("class"):
        print(f"\n-- {c}")
        print(q.drop(columns="class").to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        summarize(pd.read_csv(out_dir(SF) / "sell_timing_sim.csv.gz"))
    else:
        main()
