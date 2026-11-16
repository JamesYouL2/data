"""Q: would a rule set built only from the rules that held at 2 standard errors beat the site's calls?

Sections 31-33 tested sells, buys and sale dates one rule at a time. Here they act as sets, in middle_mode.py's league (12 teams draft the
FP top 340 in May, equal-price trades at FP ranks, real PPR scoring), each team following the calls for its own type: Contending
projected 1st-3rd, Default 4th-9th, Rebuilding 10th-12th. Classes: position x age class (rookie, 25 or under, 26-29, 30+) x FP rank 1-100
or 101-340. Everyone not named is a Hold.
  core     the class rules of sections 31 and 33:
             Contending  sell RBs 26-29 priced 101+ (1QB), QBs 30+ priced 101+ (superflex); buy TEs 30+ and RBs 30+ in the top 100
             Default     Contending's, plus sell RBs and WRs 26-29 and RBs 30+ priced 101+ (superflex: TEs 30+ too), sell RBs 26-29 in the
                         top 100 after the season, buy rookie WRs in the top 100
             Rebuilding  sell RBs 26-29 at any price, RBs 30+, WRs 26-29, WRs 30+ and TEs 30+ priced 101+ (superflex: WRs 30+ at any price,
                         QBs 30+ in the top 100); buy rookie WRs in the top 100, rookie TEs, and rookie QBs (1QB at any price, superflex
                         top 100)
  timing   core plus section 32's dates for producing players (inside the starter line by points per game the season before):
             Contending and Default  after the season: RBs 26-27, WRs 28-29. Now: RBs 28-29, WRs 30+. Default also now: RBs 30+, TEs 29+
             Rebuilding              now: RBs 26+, WRs 28+, TEs 29+ (superflex: QBs 32+)
  live     the site's calls (rules.classify)          random   middle_mode.py's random swaps
Against no trades: title odds in season 1, seasons 1-2 and seasons 1-3, value (sum of 340 - FP rank) 1 and 3 years on; then each set
less live in the same leagues. mean, z with leagues as clusters, cohorts above zero / cohorts.
Caveats: the rules come from these cohorts; FP prices and the best player in a price window, so more trades means more gain whatever is
traded (random swaps show how much); few cohorts.
Writes outputs/[sf/]ruleset_sim.csv.gz. LEAGUE=sf for superflex; REPS (default 30), JOBS. `table` reprints from the file.
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
import startup as su  # noqa: E402
from common import HERE, out_dir  # noqa: E402

R, F, SF = cs.R, cs.F, cs.SF
REPS = int(os.environ.get("REPS", 30))
JOBS = int(os.environ.get("JOBS", 6))
TYPE = {**{r: "contend" for r in (1, 2, 3)}, **{r: "middle" for r in range(4, 10)}, **{r: "rebuild" for r in range(10, 13)}}
NAME = {"contend": "Contending (1-3)", "middle": "Default (4-9)", "rebuild": "Rebuilding (10-12)"}
A, B = "1-100", "101+"
CON_SELL = {("QB", "30+", B)} if SF else {("RB", "26-29", B)}
MID_SELL = CON_SELL | {("RB", "26-29", B), ("WR", "26-29", B), ("RB", "30+", B)} | ({("TE", "30+", B)} if SF else set())
REB_SELL = MID_SELL | {("RB", "26-29", A), ("WR", "30+", B), ("TE", "30+", B)} | ({("WR", "30+", A), ("QB", "30+", A)} if SF else set())
CORE = {
    "contend": {"Sell": CON_SELL, "Buy": {("TE", "30+", A), ("RB", "30+", A)}, R.USE: set()},
    "middle": {"Sell": MID_SELL, "Buy": {("TE", "30+", A), ("RB", "30+", A), ("WR", "rookie", A)}, R.USE: {("RB", "26-29", A)}},
    "rebuild": {"Sell": REB_SELL, R.USE: set(),
                "Buy": {("WR", "rookie", A), ("TE", "rookie", A), ("TE", "rookie", B), ("QB", "rookie", A)} | (set() if SF else {("QB", "rookie", B)})},
}


def core(m, c):
    return next((act for act, cl in CORE[m].items() if c in cl), "Hold")


def timing(m, pos, age, prod, base):
    """section 32's date for a producing player, else the core call"""
    if not prod or base == "Buy":
        return base
    if m == "rebuild":
        old = (pos == "RB" and age >= 26) or (pos == "WR" and age >= 28) or (pos == "TE" and age >= 29) or (SF and pos == "QB" and age >= 32)
        return "Sell" if old else base
    if (pos == "RB" and 26 <= age < 28) or (pos == "WR" and 28 <= age < 30):
        return R.USE
    if (pos == "RB" and 28 <= age < 30) or (pos == "WR" and age >= 30):
        return "Sell"
    if m == "middle" and ((pos == "RB" and age >= 30) or (pos == "TE" and age >= 29)):
        return "Sell"
    return base


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
            out.append((t, rep, rkk[i], m, "none", *p0[:3, i], val(T0, i, 1, rk1), val(T0, i, 2, rk3), 0, 0, 0))
            for arm, a in act.items():
                T, n = mm.policy(i, "random" if arm == "random" else m, team, g, a, nxt1, rng)
                p = mm.season_titles(T, P, pos, seeds)[:, i] if sum(n) else p0[:, i]
                out.append((t, rep, rkk[i], m, arm, *p[:3], val(T, i, 1, rk1), val(T, i, 2, rk3), *n))
    print(f"cohort {t} done", flush=True)
    return out


def main():
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    pr = mm.prod_ranks(s)
    look = {(r["f"], r["pos"], r["ab"], r["tier"]): (r["v3"], r["v5"], r["y5"], r["n"])
            for r in json.loads((HERE.parent / "pages" / "build" / "old_lookup.json").read_text())["rows"]}
    d = mm.pool()
    d = d[d.t <= 2024]
    proj = mm.projector(pts, d)
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    rng = np.random.default_rng(61)
    jobs, counts = [], []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        live = dict(mm.calls(g, t, pr, look))
        act = {"live": {m: live[m] for m in NAME}, "random": {m: live[m] for m in NAME}}
        ctx = {"ROOKIE": t + 1}
        dy = np.where(g.row_type == "rookie", t + 1, 0)
        cls = [(p, R.age_class(pd.Series({"age": a, "dy": y}), ctx), A if r <= 100 else B) for p, a, y, r in zip(g.position, g.age, dy, g.mkt_rank)]
        prk = g.gsis_id.map(pr[4].get(t, {})).to_numpy(float)
        prod = np.array([r <= R.START[F][p] for r, p in zip(prk, g.position)])
        act["core"] = {m: np.array([core(m, c) for c in cls], dtype=object) for m in NAME}
        act["timing"] = {m: np.array([timing(m, p, a, pd_, b) for p, a, pd_, b in zip(g.position, g.age, prod, act["core"][m])], dtype=object) for m in NAME}
        for arm in ("live", "core", "timing"):
            counts.append({"t": t, "arm": arm, **{f"{m[:3]} {k[:4]}": int(sum(R.kind(x) == k for x in act[arm][m])) for m in NAME for k in ("buy", "sell", "use")}})
        yrs = [y for y in range(t + 1, t + 1 + mm.H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        P = np.pad(P, ((0, 0), (0, mm.H - P.shape[1])))
        nxt1 = g.gsis_id.map(rank_at.get(t + 1, pd.Series(dtype=float))).to_numpy(float)
        rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
        rk3 = g.gsis_id.map(rank_at[t + 3]).to_numpy(float) if t + 3 in rank_at else None
        jobs.append((t, g, P, act, rk1, rk3, nxt1, int(rng.integers(1 << 31))))
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(f"== {'Superflex' if SF else '1QB'}: players of 340 called Buy / Sell / Use, then sell, mean over cohorts")
    print(pd.DataFrame(counts).groupby("arm").mean().drop(columns="t").round(0).to_string())
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "rank", "mode", "arm", "p1", "p2", "p3", "val1", "val3", "sold", "later", "bought"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / "ruleset_sim.csv.gz", index=False)
    summarize(df)


def summarize(df):
    pd.set_option("display.width", 250, "display.max_rows", 400)
    key = ["t", "rep", "rank"]
    z = df.copy()
    z["yr1"] = z.p1 * 100
    z["yrs2"] = np.where(z.t + 2 <= su.LAST, (z.p1 + z.p2) * 100, np.nan)
    z["yrs3"] = np.where(z.t + 3 <= su.LAST, (z.p1 + z.p2 + z.p3) * 100, np.nan)
    z["v1"] = np.where(z.t + 1 <= 2024, z.val1, np.nan)
    z["v3"] = np.where(z.t + 3 <= 2024, z.val3, np.nan)
    mets = ["yr1", "yrs2", "yrs3", "v1", "v3"]
    base = z[z.arm == "none"].set_index(key)[mets]

    def cells(d):
        r = {}
        for c in mets:
            v = d[c].dropna()
            lg, by = v.groupby(level=["t", "rep"]).mean(), v.groupby(level="t").mean()
            se = lg.std(ddof=1) / np.sqrt(len(lg))
            r[c] = f"{lg.mean():+7.1f} z{lg.mean() / se if se > 0 else 0:+5.1f} {int((by > 0).sum())}/{len(by)}"
        return r
    rows, rows2 = [], []
    for m in NAME:
        q0 = z[z["mode"] == m]
        live = q0[q0.arm == "live"].set_index(key)
        for arm in ("live", "core", "timing", "random"):
            q = q0[q0.arm == arm].set_index(key)
            rows.append({"team": NAME[m], "arm": arm, "trades": round((q.sold + q.later + q.bought).mean(), 1), **cells(q[mets] - base.loc[q.index])})
            if arm in ("core", "timing"):
                rows2.append({"team": NAME[m], "arm": f"{arm} less live", **cells(q[mets] - live.loc[q.index, mets])})
    print(f"\n== {'Superflex' if SF else '1QB'}: each set against no trades. mean, z (leagues as clusters), cohorts above zero / cohorts")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n-- the new sets less the site's calls, same leagues")
    print(pd.DataFrame(rows2).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        summarize(pd.read_csv(out_dir(SF) / "ruleset_sim.csv.gz"))
    else:
        main()
