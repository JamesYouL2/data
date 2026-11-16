"""Q: how strict should the bar for a call be: 2.5 standard errors, 2, or something else?

Rules. Pools of players by position x age class x price (FP rank 1-50, 51-100, 101-200, 201+), position x age class, and age class
(RB / WR / TE), scored as rule_audit.py: what the pool returned against the average player at its price, as a ratio with a standard
error (a player's seasons are one cluster) and z = (ratio - 1) / SE. Each team type reads its own lens: Rebuilding 3 seasons of VORP
plus value after, Default 1 season plus value after, Contending next season's VORP. At a bar B, a player's call comes from the
narrowest of his pools with 15+ players and |z| >= B: Buy above 1, Sell below 1. No such pool: Hold.
Training. A cohort's calls never use that cohort:
  wf    only cohorts whose outcomes were complete by then (walk-forward). Tests cohorts 2021-24.
  loco  every other cohort. Tests all of 2018-24, but a neighbouring cohort's outcome seasons overlap the tested one's.
Sim. middle_mode.py's league (12 teams draft the FP top 340 in May, equal-price trades at FP ranks, real PPR scoring): each team follows
the calls for its own type (projected 1st-3rd Contending, 4th-8th Default, 9th-12th Rebuilding). Arms: each bar under each training,
the site's calls (live), and middle_mode.py's random swaps. Against no trades: title odds in season 1 and seasons 1-3, and value
(sum of 340 - FP rank) 1 and 3 years on; mean, z with leagues as clusters, cohorts above zero.
Caveats: FP prices, and the sim takes the best player in a price window, so levels are inflated: compare arms. Few cohorts.
Writes outputs/[sf/]threshold_sim.csv.gz. LEAGUE=sf for superflex; REPS (default 20), JOBS. `table` reprints from the file.
"""
import json
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
import rule_audit as ra  # noqa: E402
import startup as su  # noqa: E402
from common import HERE, out_dir  # noqa: E402

R, F, SF = cs.R, cs.F, cs.SF
REPS = int(os.environ.get("REPS", 20))
JOBS = int(os.environ.get("JOBS", 6))
BARS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
LENS = {"rebuild": "keep3", "middle": "keep1", "contend": "prod1"}
TYPE = {**{r: "contend" for r in (1, 2, 3)}, **{r: "middle" for r in range(4, 9)}, **{r: "rebuild" for r in range(9, 13)}}
MIN_N, FIRST = 15, {"wf": 2021, "loco": 2018}


def keys(pos, a, pb):
    return [("pap", pos, a, pb), ("pa", pos, a)] + ([("a", a)] if pos != "QB" else [])


def tables(H, cohorts, lens):
    """pool key -> z, from the rows of `cohorts`"""
    x = H[ra.LENS_H[lens]]
    x = x[x.t.isin(cohorts)]
    out = {}
    groups = [(("pap", p, a, pb), q) for (p, a, pb), q in x.groupby(["position", "acls", "pb"])]
    groups += [(("pa", p, a), q) for (p, a), q in x.groupby(["position", "acls"])]
    groups += [(("a", a), q) for a, q in x[x.position != "QB"].groupby("acls")]
    for k, q in groups:
        s = ra.test(q, lens)
        if s is not None and s["n"] >= MIN_N:
            out[k] = s["z"]
    return out


def call(tab, pos, a, pb, bar):
    for k in keys(pos, a, pb):
        z = tab.get(k)
        if z is not None and abs(z) >= bar:
            return "Buy" if z > 0 else "Sell"
    return "Hold"


def run_cohort(args):
    t, g, P, act, rk1, rk3, nxt1, seed = args
    rng = np.random.default_rng(seed)
    pos = g.position.to_numpy()
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
            m = TYPE[rkk[i]]
            out.append((t, rep, rkk[i], m, "none", *p0[:, i], val(T0, i, 1, rk1), val(T0, i, 2, rk3), 0))
            for arm, a in act.items():
                T, n = mm.policy(i, "random" if arm == "random" else m, team, g, a, nxt1, rng)
                p = mm.season_titles(T, P, pos, seeds)[:, i] if sum(n) else p0[:, i]
                out.append((t, rep, rkk[i], m, arm, *p, val(T, i, 1, rk1), val(T, i, 2, rk3), sum(n)))
    print(f"cohort {t} done", flush=True)
    return out


def main():
    H = cs.horizons()
    H[1] = cs.calls(H[1])
    H[3] = H[3].join(H[1].set_index(["gsis_id", "t"])[["grp", "reb", "con", "mid", "band", "pb", "acls"]], on=["gsis_id", "t"])
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    pr = mm.prod_ranks(s)
    look = {(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"])
            for r in json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]}
    d = mm.pool()
    d = d[d.t <= 2024]
    proj = mm.projector(pts, d)
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    rng = np.random.default_rng(17)
    jobs, counts = [], []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        act = dict(mm.calls(g, t, pr, look))
        act = {"live": {m: act[m] for m in LENS}, "random": {m: act[m] for m in LENS}}
        ctx = {"ROOKIE": t + 1}
        dy = np.where(g.row_type == "rookie", t + 1, 0)
        cls = [(p, R.age_class(pd.Series({"age": a, "dy": y}), ctx), R.price_bucket(r)) for p, a, y, r in zip(g.position, g.age, dy, g.mkt_rank)]
        for how in ("wf", "loco"):
            if t < FIRST[how]:
                continue
            tabs = {}
            for m, lens in LENS.items():
                h = ra.LENS_H[lens]
                coh = [c for c in H[h].t.unique() if (c + h <= t if how == "wf" else c != t)]
                tabs[m] = tables(H, coh, lens)
            for bar in BARS:
                a = {m: np.array([call(tabs[m], *c, bar) for c in cls], dtype=object) for m in LENS}
                act[f"{how}_{bar}"] = a
                counts.append({"t": t, "how": how, "bar": bar, **{f"{m} {k}": int((a[m] == k).sum()) for m in LENS for k in ("Buy", "Sell")}})
        yrs = [y for y in range(t + 1, t + 1 + mm.H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        P = np.pad(P, ((0, 0), (0, mm.H - P.shape[1])))
        nxt1 = g.gsis_id.map(rank_at.get(t + 1, pd.Series(dtype=float))).to_numpy(float)
        rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
        rk3 = g.gsis_id.map(rank_at[t + 3]).to_numpy(float) if t + 3 in rank_at else None
        jobs.append((t, g, P, act, rk1, rk3, nxt1, int(rng.integers(1 << 31))))
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(f"== {'Superflex' if SF else '1QB'}: players called Buy / Sell of 340, mean over the tested cohorts")
    print(pd.DataFrame(counts).groupby(["how", "bar"]).mean().drop(columns="t").round(0).to_string())
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "rank", "mode", "arm"] + [f"p{k}" for k in range(1, mm.H + 1)] + ["val1", "val3", "n"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / "threshold_sim.csv.gz", index=False)
    summarize(df)


def summarize(df):
    pd.set_option("display.width", 250, "display.max_rows", 400)
    key = ["t", "rep", "rank"]
    z = df.copy()
    z["yr1"] = z.p1 * 100
    z["yrs3"] = np.where(z.t + 3 <= su.LAST, (z.p1 + z.p2 + z.p3) * 100, np.nan)
    z["v1"] = np.where(z.t + 1 <= 2024, z.val1, np.nan)
    z["v3"] = np.where(z.t + 3 <= 2024, z.val3, np.nan)
    base = z[z.arm == "none"].set_index(key)[["yr1", "yrs3", "v1", "v3"]]
    mets = ["yr1", "yrs3", "v1", "v3"]
    for how, first in FIRST.items():
        arms = ["live", "random"] + [f"{how}_{b}" for b in BARS]
        rows = []
        for m in ("contend", "middle", "rebuild"):
            for arm in arms:
                q = z[(z.arm == arm) & (z["mode"] == m) & (z.t >= first)].set_index(key)
                d = q[mets] - base.loc[q.index]
                r = {"team": m, "arm": arm.replace(how + "_", "bar "), "trades": round(q.n.mean(), 1)}
                for c in mets:
                    v = d[c].dropna()
                    if v.empty:
                        r[c] = ""
                        continue
                    lg, by = v.groupby(level=["t", "rep"]).mean(), v.groupby(level="t").mean()
                    se = lg.std(ddof=1) / np.sqrt(len(lg))
                    r[c] = f"{lg.mean():+7.1f} z{lg.mean() / se:+5.1f} {int((by > 0).sum())}/{len(by)}"
                rows.append(r)
        print(f"\n== {'Superflex' if SF else '1QB'}, training {how}, cohorts {first}-2024, against no trades: mean, z, cohorts above zero / cohorts")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        summarize(pd.read_csv(out_dir(SF) / "threshold_sim.csv.gz"))
    else:
        main()
