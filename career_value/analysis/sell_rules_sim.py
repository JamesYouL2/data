"""Q: which sells belong to which kind of team?

middle_mode.py's league (12 teams draft the FP top 340 in May, equal-price trades at FP ranks, real PPR scoring). Team types by the
projection for the coming season: Contending 1st-3rd, Default 4th-9th, Rebuilding 10th-12th. One sell rule at a time against no trades:
a team sells every player it holds in one class, position x age class (rookie, 25 or under, 26-29, 30+) x price (FP rank 1-100,
101-340), either
  now    in May, for the player in his price window who most helps the lineup (Contending, Default) or the youngest (Rebuilding)
  later  after the season, at next May's price: "Use, then sell"
Control: the same number of the team's players in that price band (up to 3), picked at random, sold the same way. Selling anyone for
the best lineup fit gains in this sim, so every number is the rule less its control in the same league.
Outcomes: title odds in season 1, seasons 1-2 and seasons 1-3, and value (sum of 340 - FP rank) 1 and 3 years on; mean, z with leagues
as clusters, cohorts above zero / cohorts.
A sell is kept for a team type if it clears that type's bar on its objective at a z of BAR and is above zero in every cohort but one:
  Contending   title odds over seasons 1-2, and no loss over seasons 1-3 at z <= -BAR
  Default      title odds over seasons 1-3, and no loss of value 1 year on at z <= -BAR
  Rebuilding   value 3 years on, and no loss of value 1 year on at z <= -BAR. (It was value 1 year on, for the extra cohorts; that hid
               buys whose price doesn't move in a year, QBs above all.)
and the objective has at least 3 cohorts. The bar is BAR (default 2) for every type. Section 30 had 1.5 a little ahead for Contending and Rebuilding within their type, but a team projected
1st-3rd finishes there 45% of the time (team_signals.py persist), and Default's calls do as well or better over three seasons for every
kind of team (middle_mode.py), so a rule that departs from Default has to clear the stricter bar.
Caveats: FP prices and the best player in a price window, so levels are inflated; leagues in a cohort share players, so z is too
large; one rule at a time, and rules interact (section 28); the bars come from section 30.
SIDE=buy runs buys the same way: every player of one class on another team is a Buy, and the team buys up to 3 of them (middle_mode.py:
the biggest lineup gain for Contending and Default, the youngest for Rebuilding, giving the player it values least in the price window).
Its control marks as many random players of that price band on other teams as Buys.
The table also lists where Default stands apart: a rule both Contending and Rebuilding keep and Default doesn't, and Default on a second
test (value 1 year on at the bar, with no title measure at z <= -2).
`windows` compares each rule's title odds over seasons 1-3, 1-4 and 1-5. The league plays five seasons on the May rosters; ROOKIES=1
adds a rookie draft before each later season (middle_mode.season_titles: 4 rounds, worst team first, FP rookie ranks with noise),
without which seasons 4-5 lean on who was young at the start.
Writes outputs/[sf/]sell_rules_sim.csv.gz (buy_rules_sim.csv.gz with SIDE=buy). LEAGUE=sf for superflex; REPS (default 20), JOBS. `table`
reprints from the file.
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

R, SF = cs.R, cs.SF
REPS = int(os.environ.get("REPS", 20))
JOBS = int(os.environ.get("JOBS", 6))
ONLY = os.environ.get("ONLY", "")                  # "QB": QB classes only, with 30+ split into 30-32 and 33+ (files get a _qb suffix)
ROOKIES = os.environ.get("ROOKIES", "") not in ("", "0")   # rookie drafts between seasons (middle_mode.ROOK); files get a _rook suffix
SIDE = os.environ.get("SIDE", "sell")              # "buy": one buy rule at a time, written to buy_rules_sim.csv.gz
TYPE = {**{r: "contend" for r in (1, 2, 3)}, **{r: "middle" for r in range(4, 10)}, **{r: "rebuild" for r in range(10, 13)}}
NAME = {"contend": "Contending (1-3)", "middle": "Default (4-9)", "rebuild": "Rebuilding (10-12)"}
BAR = float(os.environ.get("BAR", 2.0))
GOAL = {"contend": ("yrs2", "yrs3"), "middle": ("yrs3", "v1"), "rebuild": ("v3", "v1")}     # objective, guard
MIN_COHORTS = 3
METS = ["yr1", "yrs2", "yrs3", "v1", "v3"]
LONG = ["yrs4", "yrs5"]                 # title odds over seasons 1-4 and 1-5, when the file has p4 and p5 (`windows`)


def run_cohort(args):
    t, g, P, cls, rk1, rk3, nxt1, seed, mm.ROOK = args
    rng = np.random.default_rng(seed)
    pos = g.position.to_numpy()
    bands = np.array([c.split(" | ")[1] for c in cls], dtype=object)
    out = []
    acts = {}
    for c in sorted(x for x in set(cls) if not x.startswith("other")):
        for how, a in (("now", "Sell"), ("later", R.USE)):
            acts[c, how] = np.where(cls == c, a, "Hold").astype(object)

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
            v1, v3 = val(T0, i, 1, rk1), val(T0, i, 2, rk3)
            out.append((t, rep, rkk[i], m, "none", "-", *p0[:, i], v1, v3, 0, 0))
            if SIDE == "buy":
                oth = (team >= 0) & (team != i)
                for c in sorted(x for x in set(cls) if not x.startswith("other")):
                    tg = np.flatnonzero(oth & (cls == c))
                    band = c.split(" | ")[1]
                    pool_ = np.flatnonzero(oth & (bands == band))
                    for name, pick in ((c, tg), ("ctl:" + c, rng.choice(pool_, size=min(len(tg), len(pool_)), replace=False) if len(tg) else tg)):
                        if not len(pick):
                            out.append((t, rep, rkk[i], m, name, "buy", *p0[:, i], v1, v3, 0, 0))
                            continue
                        act = np.full(len(team), "Hold", dtype=object)
                        act[pick] = "Buy"
                        T, n = mm.policy(i, m, team, g, {m: act}, nxt1, rng)
                        p = mm.season_titles(T, P, pos, seeds)[:, i] if sum(n) else p0[:, i]
                        out.append((t, rep, rkk[i], m, name, "buy", *p, val(T, i, 1, rk1), val(T, i, 2, rk3), sum(n), len(pick)))
                continue
            mine = cls[team == i]
            for (c, how), a in acts.items():
                held = int((mine == c).sum())
                if not held:
                    out.append((t, rep, rkk[i], m, c, how, *p0[:, i], v1, v3, 0, 0))
                    continue
                T, n = mm.policy(i, m, team, g, {m: a}, nxt1, rng)
                p = mm.season_titles(T, P, pos, seeds)[:, i] if sum(n) else p0[:, i]
                out.append((t, rep, rkk[i], m, c, how, *p, val(T, i, 1, rk1), val(T, i, 2, rk3), sum(n), held))
            for band in ("1-100", "101+"):                     # controls: k random players of that price band
                own = np.flatnonzero((team == i) & (bands == band))
                for k in (1, 2, 3):
                    pick = rng.choice(own, size=min(k, len(own)), replace=False) if len(own) else own
                    for how, a in (("now", "Sell"), ("later", R.USE)):
                        act = np.full(len(team), "Hold", dtype=object)
                        act[pick] = a
                        T, n = mm.policy(i, m, team, g, {m: act}, nxt1, rng)
                        p = mm.season_titles(T, P, pos, seeds)[:, i] if sum(n) else p0[:, i]
                        out.append((t, rep, rkk[i], m, f"random {k} | {band}", how, *p, val(T, i, 1, rk1), val(T, i, 2, rk3), sum(n), len(pick)))
    print(f"cohort {t} done", flush=True)
    return out


def main():
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = {y: z.set_index("gsis_id").ppr.to_dict() for y, z in s.groupby("season")}
    dall = mm.pool()
    d = dall[dall.t <= 2024]
    proj = mm.projector(pts, d)
    rank_at = {t: z.set_index("gsis_id").mkt_rank for t, z in d.groupby("t")}
    rng = np.random.default_rng(23)
    jobs = []
    for t in range(2018, 2025):
        g = d[d.t == t].reset_index(drop=True)
        g["mkt_rank"] = g.mkt_rank.rank(method="first")
        g["pos_rank_fp"] = g.groupby("position").mkt_rank.rank(method="first")
        g["proj"] = proj(g)
        ctx = {"ROOKIE": t + 1}
        dy = np.where(g.row_type == "rookie", t + 1, 0)
        def ac(p, a, y):
            c = R.age_class(pd.Series({"age": a, "dy": y}), ctx)
            return ("30-32" if a < 33 else "33+") if ONLY == "QB" and c == "30+" else c
        cls = np.array([f"{p if not ONLY or p == ONLY else 'other'} {ac(p, a, y) if not ONLY or p == ONLY else 'any'} | {'1-100' if r <= 100 else '101+'}"
                        for p, a, y, r in zip(g.position, g.age, dy, g.mkt_rank)], dtype=object)
        yrs = [y for y in range(t + 1, t + 1 + mm.H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        P = np.pad(P, ((0, 0), (0, mm.H - P.shape[1])))
        nxt1 = g.gsis_id.map(rank_at.get(t + 1, pd.Series(dtype=float))).to_numpy(float)
        rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
        rk3 = g.gsis_id.map(rank_at[t + 3]).to_numpy(float) if t + 3 in rank_at else None
        jobs.append((t, g, P, cls, rk1, rk3, nxt1, int(rng.integers(1 << 31)), mm.rookie_classes(dall, pts, t) if ROOKIES else None))
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "rank", "mode", "cls", "how"] + [f"p{k}" for k in range(1, mm.H + 1)] + ["val1", "val3", "n", "held"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / f"{SIDE}_rules_sim{'_' + ONLY.lower() if ONLY else ''}{'_rook' if ROOKIES else ''}.csv.gz", index=False)
    summarize(df)


def verdicts(df, bar=None, cohort_check=True):
    """one row per team type, class and date: the rule less its control on every measure, and whether it is kept at `bar`"""
    bar = BAR if bar is None else bar
    key = ["t", "rep", "rank"]
    z = df.copy()
    z["yr1"] = z.p1 * 100
    z["yrs2"] = np.where(z.t + 2 <= su.LAST, (z.p1 + z.p2) * 100, np.nan)
    z["yrs3"] = np.where(z.t + 3 <= su.LAST, (z.p1 + z.p2 + z.p3) * 100, np.nan)
    z["v1"] = np.where(z.t + 1 <= 2024, z.val1, np.nan)
    z["v3"] = np.where(z.t + 3 <= 2024, z.val3, np.nan)
    METS = globals()["METS"] + (LONG if "p5" in z else [])
    if "p5" in z:
        z["yrs4"] = np.where(z.t + 4 <= su.LAST, (z.p1 + z.p2 + z.p3 + z.p4) * 100, np.nan)
        z["yrs5"] = np.where(z.t + 5 <= su.LAST, (z.p1 + z.p2 + z.p3 + z.p4 + z.p5) * 100, np.nan)
    base = z[z.cls == "none"].set_index(key)[METS]
    buy = (z.how == "buy").any()
    if buy:
        ctl = z[z.cls.str.startswith("ctl:")].assign(cls=lambda x: x.cls.str[4:]).set_index(key + ["cls"])[METS]
    else:
        ctl = z[z.cls.str.startswith("random")].copy()
        ctl["k"], ctl["band"] = ctl.cls.str.split().str[1].astype(int), ctl.cls.str.split(" | ", regex=False).str[1]
        ctl = ctl.set_index(key + ["how", "band", "k"])[METS]
    rows = []
    for (m, c, how), q in z[~z.cls.str.startswith(("none", "random", "ctl:"))].groupby(["mode", "cls", "how"]):
        band = c.split(" | ")[1]
        q = q.set_index(key)
        if buy:
            ref = ctl.reindex(pd.MultiIndex.from_arrays([q.index.get_level_values(0), q.index.get_level_values(1), q.index.get_level_values(2),
                                                         [c] * len(q)])).to_numpy()
        else:
            k = q.held.clip(1, 3).to_numpy()
            ref = ctl.reindex(pd.MultiIndex.from_arrays([q.index.get_level_values(0), q.index.get_level_values(1), q.index.get_level_values(2),
                                                         [how] * len(q), [band] * len(q), k])).to_numpy()
            ref = np.where((q.held.to_numpy() > 0)[:, None], ref, base.loc[q.index].to_numpy())   # holds none: the control is no trade too
        d = pd.DataFrame(q[METS].to_numpy() - ref, index=q.index, columns=METS)
        raw = q[METS] - base.loc[q.index]
        r = {"team": m, "class": c, "when": how, "sold a team": round(q.n.mean(), 2)}   # traded a team, for buys
        st = {}
        for kk in METS:
            v = d[kk].dropna()
            lg, by = v.groupby(level=["t", "rep"]).mean(), v.groupby(level="t").mean()
            se = lg.std(ddof=1) / np.sqrt(len(lg))
            st[kk] = (lg.mean(), lg.mean() / se if se > 0 else 0.0, int((by > 0).sum()), len(by))
            r[kk] = f"{st[kk][0]:+6.1f} z{st[kk][1]:+5.1f} {st[kk][2]}/{st[kk][3]}"
        g, guard = GOAL[m]
        agree = (lambda x: st[x][2] >= st[x][3] - 1) if cohort_check else (lambda x: True)
        ok = (st[g][1] >= bar and agree(g) and st[g][3] >= MIN_COHORTS and not (guard and st[guard][1] <= -bar))
        r["_st"] = {kk: st[kk] for kk in ("yr1", "yrs3") + tuple(x for x in LONG if x in st)}
        r["vs no trade"] = f"{raw[g].mean():+.1f}"
        r["keep"], r["_z"], r["_gain"] = ("KEEP" if ok else ""), st[g][1], st[g][0]
        r["_value"] = (st["v1"][1] >= bar and agree("v1") and st["v1"][3] >= MIN_COHORTS
                       and min(st[x][1] for x in ("yr1", "yrs2", "yrs3")) > -bar)
        r["_v1"] = st["v1"][0]
        rows.append(r)
    return pd.DataFrame(rows), z, base, buy


def kept(df, bar, cohort_check=True):
    """team -> {class: 'now' | 'later' | 'buy'} at `bar`; Default also takes rules that pass on value with no title cost"""
    t = verdicts(df, bar, cohort_check)[0]
    t = t[t["sold a team"] >= 0.05]
    out = {}
    for m in GOAL:
        q = t[(t.team == m) & ((t.keep == "KEEP") | ((m == "middle") & t._value))]
        q = q.assign(score=np.where(q.keep == "KEEP", q._z, 0.0)).sort_values(["score", "_v1"], ascending=False)
        out[m] = q.groupby("class").head(1).set_index("class").when.to_dict()
    return out


def windows(df):
    """does a rule's verdict on title odds change with the window: seasons 1-3, 1-4 (cohorts 2018-21), 1-5 (2018-20)?"""
    pd.set_option("display.width", 250, "display.max_rows", 600)
    t = verdicts(df)[0]
    t = t[t["sold a team"] >= 0.05]
    side = lambda s: "+" if s[1] >= BAR and s[2] >= s[3] - 1 else "-" if s[1] <= -BAR and s[2] <= 1 else "."   # noqa: E731
    rows = []
    for _, r in t.iterrows():
        st = r["_st"]
        rows.append({"team": NAME[r["team"]], "class": r["class"], "when": r["when"], **{k: f"{st[k][0]:+5.1f} z{st[k][1]:+5.1f} {st[k][2]}/{st[k][3]}" for k in st},
                     "sides": "".join(side(st[k]) for k in ("yrs3", "yrs4", "yrs5")), "_z5": st["yrs5"][1]})
    w = pd.DataFrame(rows)
    print(f"== {'Superflex' if SF else '1QB'}, {SIDE}s: one rule less its control, title odds by window. sides = seasons 1-3 / 1-4 / 1-5: "
          f"+ gains at {BAR} SE in all cohorts but one, - loses, . neither")
    for m in NAME.values():
        q = w[w.team == m]
        print(f"\n-- {m}: {len(q)} rules; same side in all three windows {int((q.sides.str[0] == q.sides.str[1]).mul(q.sides.str[1] == q.sides.str[2]).sum())}; "
              f"counts by sides {q.sides.value_counts().to_dict()}")
        ch = q[~((q.sides.str[0] == q.sides.str[1]) & (q.sides.str[1] == q.sides.str[2]))]
        print("   rules whose side changes with the window:")
        print(ch.sort_values("_z5", ascending=False).drop(columns=["team", "_z5"]).to_string(index=False) if len(ch) else "   (none)")
        print("   + in all three:"); print(q[q.sides == "+++"].sort_values("_z5", ascending=False).drop(columns=["team", "_z5"]).to_string(index=False))


def summarize(df):
    pd.set_option("display.width", 280, "display.max_rows", 600)
    key = ["t", "rep", "rank"]
    t, z, base, buy = verdicts(df)
    print(f"== {'Superflex' if SF else '1QB'}: one sell rule less its control (as many random players of that price band sold the same way), by team type. "
          "mean, z (leagues as clusters), cohorts above zero / cohorts; vs no trade = the objective against not trading")
    for m in ("contend", "middle", "rebuild"):
        g, guard = GOAL[m]
        q = t[(t.team == m) & (t["sold a team"] >= 0.05)].sort_values("_z", ascending=False)
        print(f"\n-- {NAME[m]}: objective {g} at a bar of {BAR}" + (f", guard {guard}" if guard else "") + "; sorted by the objective's z")
        print(q.drop(columns=["team", "_z", "_gain", "_value", "_v1", "_st"]).to_string(index=False))
    k = t[t.keep == "KEEP"].sort_values("_gain", ascending=False)
    best = k.groupby(["team", "class"]).head(1)
    print("\n== the sells kept, the better of now / later for each class")
    w = best.pivot(index="class", columns="team", values="when").reindex(columns=["contend", "middle", "rebuild"]).fillna("")
    print(w.rename(columns=NAME).to_string())
    both = set(k[k.team == "contend"]["class"]) & set(k[k.team == "rebuild"]["class"])
    mid = t[t.team == "middle"]
    print("\n== Default apart from its neighbours: classes both Contending and Rebuilding keep, and what Default got")
    print(mid[mid["class"].isin(both)].drop(columns=["team", "_z", "_gain", "_value", "_v1", "_st"]).to_string(index=False) if both else "(none)")
    print("\n== Default on a second test: value 1 year on at 2 standard errors, no title measure at z <= -2")
    print(mid[mid._value & (mid["sold a team"] >= 0.05)].sort_values("_gain", ascending=False).drop(columns=["team", "_z", "_gain", "_value", "_v1", "_st"]).to_string(index=False))
    if buy:
        return
    c = z[z.cls.str.startswith("random")].set_index(key)
    cd = c[METS] - base.loc[c.index]
    print("\n== the controls against no trade (what selling anyone does here)")
    print(cd.assign(team=c["mode"].to_numpy(), cls=c.cls.to_numpy(), when=c.how.to_numpy()).groupby(["team", "when", "cls"])[METS].mean().round(1).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["windows"]:
        windows(pd.read_csv(out_dir(SF) / f"{SIDE}_rules_sim{'_' + ONLY.lower() if ONLY else ''}{'_rook' if ROOKIES else ''}.csv.gz"))
    elif sys.argv[1:] == ["table"]:
        summarize(pd.read_csv(out_dir(SF) / f"{SIDE}_rules_sim{'_' + ONLY.lower() if ONLY else ''}{'_rook' if ROOKIES else ''}.csv.gz"))
    else:
        main()
