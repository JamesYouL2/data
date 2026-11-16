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
  barB     the lists read mechanically off sell_rules_sim.py's saved runs at a bar of B, a class that is both a sell and a buy left a
           Hold; zB the same without the every-cohort-but-one check; "+t" adds the timing dates
  simple   four rules, the same in both formats and nested by team type (`simple`)
  hybrid:X the same with "producing" read another way: rd = inside the starter line on this season's projection (redraft ADP), blend =
           on the average of that rank and last season's points-per-game rank, either / both = on one or on both
  hybrid   the four rules where they speak, the site's calls everywhere else, and Default kept between Contending and Rebuilding
  uB       one list for both formats (`unified`): a class rule has to clear B in the 1QB runs and in the superflex runs, sells nest
           (Contending's are Default's, Default's are Rebuilding's), QB rules are superflex only
  x:...    the same lists with each cohort's rules read from the other cohorts only, which is the fair test of a bar: a loose bar picks
           up rules that only fit the cohorts it was read on
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


RUNS = {}


def from_tests(bar, cohort_check=True, without=None):
    """mode -> {class tuple: call}, read off sell_rules_sim.py's saved runs at `bar`; a class that is both a sell and a buy is left a Hold.
    without = a cohort to leave out of the runs, so a cohort's rules never come from that cohort"""
    import sell_rules_sim as srs
    if not RUNS:
        RUNS.update({side: pd.read_csv(out_dir(SF) / f"{side}_rules_sim.csv.gz") for side in ("sell", "buy")})
    sells = srs.kept(RUNS["sell"][RUNS["sell"].t != without], bar, cohort_check)
    buys = srs.kept(RUNS["buy"][RUNS["buy"].t != without], bar, cohort_check)
    out = {}
    for m in NAME:
        d = {}
        for c, when in sells[m].items():
            d[c] = "Sell" if when == "now" else R.USE
        for c in buys[m]:
            d[c] = "Hold" if c in d else "Buy"
        out[m] = {tuple(x.strip() for x in c.replace(" | ", " ").split(" ", 2)): a for c, a in d.items()}
    return out


def simple(m, pos, age, rookie, band, prod):
    """four rules, the same in both formats; QBs are always a Hold
      1  sell RBs and WRs 26 or older and TEs 30 or older who are priced outside the top 100 (every team type)
      2  a producing veteran is sold on section 32's date: after the season for RBs 26-27 and WRs 28-29 (Rebuilding: now); now for
         RBs 28-29 and WRs 30+; now for RBs 30+ and TEs 29+ unless Contending, which holds them
      3  a top-100 RB at 26-29 who isn't producing: Contending holds, Default sells after the season, Rebuilding sells now
      4  buy top-100 rookie WRs (Default, Rebuilding), rookie TEs (Rebuilding), and top-100 TEs 30 or older (Contending, Default)"""
    if pos == "QB":
        return "Hold"
    if rookie:
        return "Buy" if (pos == "WR" and band == A and m != "contend") or (pos == "TE" and m == "rebuild") else "Hold"
    if pos == "TE" and age >= 30 and band == A and m != "rebuild":
        return "Buy"
    if prod:
        if (pos == "RB" and 26 <= age < 28) or (pos == "WR" and 28 <= age < 30):
            return "Sell" if m == "rebuild" else R.USE
        if (pos == "RB" and 28 <= age < 30) or (pos == "WR" and age >= 30):
            return "Sell"
        if (pos == "RB" and age >= 30) or (pos == "TE" and age >= 29):
            return "Hold" if m == "contend" else "Sell"
    if band == B and ((pos in ("RB", "WR") and age >= 26) or (pos == "TE" and age >= 30)):
        return "Sell"
    if band == A and pos == "RB" and 26 <= age < 30:
        return {"contend": "Hold", "middle": R.USE, "rebuild": "Sell"}[m]
    return "Hold"


# What the site's Rebuilding calls do that the four rules don't, as small rules to add to them one at a time (REB=1). age class, price
# band (A = top 100) and FP rank -> a call, or None where the rule doesn't apply.
REB_ADDS = {
    "sell RB 25- top 100": lambda p, ac, b, r: "Sell" if p == "RB" and ac == "25-" and b == A else None,
    "sell RB 25- 101+": lambda p, ac, b, r: "Sell" if p == "RB" and ac == "25-" and b == B else None,
    "sell rookie RB top 100": lambda p, ac, b, r: "Sell" if p == "RB" and ac == "rookie" and b == A else None,
    "sell TE under 30, 101+": lambda p, ac, b, r: "Sell" if p == "TE" and ac in ("25-", "26-29") and b == B else None,
    "sell WR 25- 201+": lambda p, ac, b, r: "Sell" if p == "WR" and ac == "25-" and r > 200 else None,
    "sell WR 26-29 top 100": lambda p, ac, b, r: "Sell" if p == "WR" and ac == "26-29" and b == A else None,
    "buy QB 26-29 101+": lambda p, ac, b, r: "Buy" if p == "QB" and ac == "26-29" and b == B else None,
    "buy WR 25- top 100": lambda p, ac, b, r: "Buy" if p == "WR" and ac == "25-" and b == A else None,
    "buy rookie RB top 100": lambda p, ac, b, r: "Buy" if p == "RB" and ac == "rookie" and b == A else None,
    "buy rookie QB": lambda p, ac, b, r: "Buy" if p == "QB" and ac == "rookie" else None,
    "no rookie WR buy": lambda p, ac, b, r: "Hold" if p == "WR" and ac == "rookie" and b == A else None,
    "no rookie TE buy": lambda p, ac, b, r: "Hold" if p == "TE" and ac == "rookie" else None,
}
# What the site's calls do for every team type where the four rules are silent, as candidate fifth and sixth rules (GEN=1): added to
# the four rules one at a time, the same call for every team type. (position, age class, band, FP rank, producing) -> call or None.
GEN_ADDS = {
    "sell RB 25-, not producing": lambda p, ac, b, r, pd_: "Sell" if p == "RB" and ac == "25-" and not pd_ else None,
    "sell RB 25-, not producing, top 100": lambda p, ac, b, r, pd_: "Sell" if p == "RB" and ac == "25-" and not pd_ and b == A else None,
    "sell RB 25-, not producing, 101+": lambda p, ac, b, r, pd_: "Sell" if p == "RB" and ac == "25-" and not pd_ and b == B else None,
    "sell rookie RB top 100": lambda p, ac, b, r, pd_: "Sell" if p == "RB" and ac == "rookie" and b == A else None,
    "sell TE under 30, 101+, not producing": lambda p, ac, b, r, pd_: "Sell" if p == "TE" and ac in ("25-", "26-29") and b == B and not pd_ else None,
    "sell WR 25- 201+": lambda p, ac, b, r, pd_: "Sell" if p == "WR" and ac == "25-" and r > 200 else None,
    "sell WR 26-29 top 100, not producing": lambda p, ac, b, r, pd_: "Sell" if p == "WR" and ac == "26-29" and b == A and not pd_ else None,
    "sell anyone 201+ who isn't a rookie": lambda p, ac, b, r, pd_: "Sell" if r > 200 and ac != "rookie" and p != "QB" else None,
    "buy WR 25- 101-200": lambda p, ac, b, r, pd_: "Buy" if p == "WR" and ac == "25-" and 100 < r <= 200 else None,
    "buy RB 25- 101-200": lambda p, ac, b, r, pd_: "Buy" if p == "RB" and ac == "25-" and 100 < r <= 200 else None,
    "buy WR 25- top 100": lambda p, ac, b, r, pd_: "Buy" if p == "WR" and ac == "25-" and b == A else None,
    "buy QB 26-29 101+": lambda p, ac, b, r, pd_: "Buy" if p == "QB" and ac == "26-29" and b == B else None,
    "buy rookie RB top 100": lambda p, ac, b, r, pd_: "Buy" if p == "RB" and ac == "rookie" and b == A else None,
}
def more(base, pos, ac, rank, prod, age, rules):
    """the four rules' call, with the fifth, sixth and QB rules where the four are silent (every team type gets the same call)
      5   sell anyone priced outside the top 200 who isn't a rookie or a QB
      6   sell RBs 25 or under who aren't producing
      qb  buy QBs 26-29 priced outside the top 100; sell QBs 33 or older"""
    if base != "Hold":
        return base
    if "qb" in rules and pos == "QB":
        if ac == "26-29" and rank > 100:
            return "Buy"
        if age >= 33:
            return "Sell"
    if 5 in rules and rank > 200 and ac != "rookie" and pos != "QB":
        return "Sell"
    if 6 in rules and pos == "RB" and ac == "25-" and not prod:
        return "Sell"
    return base


FINAL = {"five": (5,), "six": (5, 6), "six+qb": (5, 6, "qb"), "four+qb": ("qb",)}
PUSH = {"buy": 0, "hold": 1, "lotto": 1, "none": 1, "use": 2, "sell": 3, "cut": 3}      # how hard a call pushes a player out


def between(con, mid, reb):
    """Default's calls, moved to the nearer neighbour's wherever they fall outside Contending's and Rebuilding's"""
    out = mid.copy()
    for k, (c, m, r) in enumerate(zip(con, mid, reb)):
        pc, pm, pr_ = (PUSH[R.kind(x)] for x in (c, m, r))
        lo, hi = (c, r) if pc <= pr_ else (r, c)
        if pm < min(pc, pr_):
            out[k] = lo
        elif pm > max(pc, pr_):
            out[k] = hi
    return out


BOTH = {}


def unified(bar, without=None):
    """one rule list for both formats. A class rule is kept if it clears `bar` in the 1QB runs and in the superflex runs (the team type's
    objective, or for Default value with no title cost), with no cohort check: agreement across formats stands in for it. QB classes are
    superflex-only rules, read on superflex alone, and there are none in 1QB. Sells nest: Default also sells what Contending sells, and
    Rebuilding what Default sells. A class that ends up both a sell and a buy is a Hold."""
    import sell_rules_sim as srs
    if not BOTH:
        for side in ("sell", "buy"):
            BOTH[side] = [pd.read_csv(HERE / "outputs" / sub / f"{side}_rules_sim.csv.gz") for sub in ("", "sf")]

    def table(side):
        ts = []
        for df in BOTH[side]:
            t = srs.verdicts(df[df.t != without], bar, cohort_check=False)[0]
            t = t[t["sold a team"] >= 0.05]
            ts.append(t.assign(ok=(t.keep == "KEEP") | ((t.team == "middle") & t._value)).set_index(["team", "class", "when"])[["ok", "_z"]])
        j = ts[0].join(ts[1], lsuffix="1", rsuffix="2", how="outer").fillna({"ok1": False, "ok2": False, "_z1": 0.0, "_z2": 0.0})
        qb = j.index.get_level_values("class").str.startswith("QB")
        j["ok"] = np.where(qb, j.ok2 & SF, j.ok1 & j.ok2)
        j["score"] = j._z1 + j._z2
        j = j[j.ok].reset_index().sort_values("score", ascending=False)
        return {m: q.groupby("class").head(1).set_index("class").when.to_dict() for m, q in j.groupby("team")}
    sells, buys = table("sell"), table("buy")
    for lo, hi in (("contend", "middle"), ("middle", "rebuild")):
        sells[hi] = {**sells.get(lo, {}), **sells.get(hi, {})}
    out = {}
    for m in NAME:
        d = {c: ("Sell" if when == "now" else R.USE) for c, when in sells.get(m, {}).items()}
        for c in buys.get(m, {}):
            d[c] = "Hold" if c in d else "Buy"
        out[m] = {tuple(x.strip() for x in c.replace(" | ", " ").split(" ", 2)): a for c, a in d.items()}
    return out


def old_calls(g, t, pr, look):
    """mode -> the calls the site made before the seven rules (rules_before_seven.py), as middle_mode.calls"""
    import rules_before_seven as OLD
    x = pd.DataFrame({"position": g.position, "age": g.age, f"r_{F}": g.mkt_rank, f"pr_{F}": g.pos_rank_fp, "rd": g.draft_round_f.fillna(8),
                      "dy": np.where(g.row_type == "rookie", t + 1, 0), "prk": g.gsis_id.map(pr[4].get(t, {}))})
    x["prod_rank"] = pd.concat([x.prk, g.gsis_id.map(pr[6].get(t - 1, {}))], axis=1).mean(axis=1)
    ctx = {"ROOKIE": t + 1, "LOOK": look}
    out = {m: [] for m in NAME}
    for row in x.itertuples(index=False):
        _, (ra, _), (ca, _), (ma, _) = OLD.classify(row, F, ctx)
        out["rebuild"].append(ra)
        out["contend"].append(ca)
        out["middle"].append(ma)
    return {m: np.array(v, dtype=object) for m, v in out.items()}


def simplified(g, t, pr, look):
    """SIMPLIFY=1: the site's calls with one simplification each, arm -> mode -> calls.
      s:no Contending          a contender follows the Default calls
      s:no rookie TE buy       Rebuilding holds rookie TEs (rule 4 keeps the rookie WR and old TE buys)
      s:rule 2 after season    every producing veteran rule 2 names (RB 26+, WR 28+, TE 29+): Rebuilding sells now, everyone else
                               after the season (no sell-now tier, no contender hold)
      s:rule 2 now             the same players: Contending sells after the season, Default and Rebuilding now
      s:all three              no Contending, no rookie TE buy, rule 2 after the season
    Default is put back between Contending and Rebuilding after each change, as on the site."""
    x = pd.DataFrame({
        "position": g.position, "age": g.age, f"r_{F}": g.mkt_rank, f"pr_{F}": g.pos_rank_fp,
        "rd": g.draft_round_f.fillna(8), "dy": np.where(g.row_type == "rookie", t + 1, 0),
        "prk": g.gsis_id.map(pr[4].get(t, {})),
    })
    x["prod_rank"] = pd.concat([x.prk, g.gsis_id.map(pr[6].get(t - 1, {}))], axis=1).mean(axis=1)
    ctx = {"ROOKIE": t + 1, "LOOK": look}
    calls = {m: [] for m in NAME}
    tag2 = {m: [] for m in NAME}
    for row in x.itertuples(index=False):
        _, reb, con, mid = mm.R.classify(row, F, ctx)            # the live rules (R here is the old calls' module)
        for m, (a, why) in (("rebuild", reb), ("contend", con), ("middle", mid)):
            calls[m].append(a)
            tag2[m].append(why.startswith("Rule 2"))
    live = simplified.live = {m: np.array(v, dtype=object) for m, v in calls.items()}
    tag2 = {m: np.array(v) for m, v in tag2.items()}
    rte = ((g.position == "TE") & (g.row_type == "rookie")).to_numpy() & np.array([R.kind(a) == "buy" for a in live["rebuild"]])

    def build(no_con=False, no_te=False, r2=None):
        c, m_, r = live["contend"].copy(), live["middle"].copy(), live["rebuild"].copy()
        if no_te:
            r[rte] = "Hold"
        if r2:
            new = {"after": ("Use, then sell", "Use, then sell", "Sell"), "now": ("Use, then sell", "Sell", "Sell")}[r2]
            for arr, k, call in ((c, "contend", new[0]), (m_, "middle", new[1]), (r, "rebuild", new[2])):
                arr[tag2[k]] = R.USE if call == "Use, then sell" else call
        m_ = between(c, m_, r)
        return {"contend": m_.copy() if no_con else c, "middle": m_, "rebuild": r}
    return {"s:no Contending": build(no_con=True), "s:no rookie TE buy": build(no_te=True), "s:rule 2 after season": build(r2="after"),
            "s:rule 2 now": build(r2="now"), "s:all three": build(True, True, "after")}


GAP = os.environ.get("GAP")          # a team type ("contend"): add the old calls back to the seven rules one bucket at a time
BUCKETS = {}


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
            if GAP and m != GAP:
                continue
            out.append((t, rep, rkk[i], m, "none", *p0[:3, i], val(T0, i, 1, rk1), val(T0, i, 2, rk3), 0, 0, 0))
            for arm, a in act.items():
                if m not in a:
                    continue
                T, n = mm.policy(i, "random" if arm == "random" else m, team, g, a, nxt1, rng)
                p = mm.season_titles(T, P, pos, seeds)[:, i] if sum(n) else p0[:, i]
                out.append((t, rep, rkk[i], m, arm, *p[:3], val(T, i, 1, rk1), val(T, i, 2, rk3), *n))
    print(f"cohort {t} done", flush=True)
    return out


TESTS, ARMS = {}, ["live", "core", "timing", "simple", "hybrid", "hybrid:rd", "hybrid:blend", "hybrid:either", "hybrid:both"]
LOCO = {"x:z1.5": (1.5, False), "x:u1.5": (1.5, "u"), "x:u2": (2.0, "u")}     # rules from the other cohorts only
if os.environ.get("LOCO") == "0":
    LOCO = {}


SIMP = os.environ.get("SIMPLIFY") not in (None, "", "0")


def main():
    if SIMP:
        LOCO.clear()
        ARMS[:] = ["live", "s:no Contending", "s:no rookie TE buy", "s:rule 2 after season", "s:rule 2 now", "s:all three"]
    ARMS.extend(x for n in LOCO for x in (n, n + "+t"))
    if os.environ.get("REB"):
        ARMS.extend("r:" + n for n in REB_ADDS)
    if os.environ.get("GEN"):
        ARMS.extend("g:" + n for n in GEN_ADDS)
    if os.environ.get("FINAL"):
        ARMS.extend(FINAL)
    for n, tab in TESTS.items():
        print(n, {m: {a: sum(v == a for v in d.values()) for a in ("Sell", R.USE, "Buy")} for m, d in tab.items()})
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
        act["simple"] = {m: np.array([simple(m, p, a, c[1] == "rookie", c[2], pd_) for p, a, c, pd_ in zip(g.position, g.age, cls, prod)], dtype=object)
                         for m in NAME}
        hyb = {m: np.where(act["simple"][m] != "Hold", act["simple"][m], live[m]).astype(object) for m in NAME}
        act["hybrid"] = {**hyb, "middle": between(hyb["contend"], hyb["middle"], hyb["rebuild"])}
        # other tests of "producing": this season's projection (redraft ADP), and the two ranks averaged
        line = np.array([R.START[F][p] for p in g.position])
        rdk = g.groupby("position").proj.rank(ascending=False, method="min").to_numpy(float)
        both = np.nanmean(np.vstack([prk, rdk]), axis=0)
        for tag, pv in (("rd", rdk <= line), ("blend", both <= line), ("either", (prk <= line) | (rdk <= line)), ("both", (prk <= line) & (rdk <= line))):
            sm = {m: np.array([simple(m, p, a, c[1] == "rookie", c[2], pd_) for p, a, c, pd_ in zip(g.position, g.age, cls, pv)], dtype=object)
                  for m in NAME}
            hy = {m: np.where(sm[m] != "Hold", sm[m], live[m]).astype(object) for m in NAME}
            act[f"hybrid:{tag}"] = {**hy, "middle": between(hy["contend"], hy["middle"], hy["rebuild"])}
        if GAP:
            old = old_calls(g, t, pr, look)
            new = live[GAP]                                   # the site's calls now: the seven rules
            act["old"] = {GAP: old[GAP]}
            act["seven over old"] = {GAP: np.where(np.array([R.kind(c) for c in new]) != "hold", new, old[GAP]).astype(object)}
            b3 = np.where(g.mkt_rank <= 100, "1-100", np.where(g.mkt_rank <= 200, "101-200", "201+"))
            keys = [f"old {R.kind(o)} / seven {R.kind(n)}: {p} {c[1]} {b}{' producing' if pd_ else ''}"
                    for o, n, p, c, b, pd_ in zip(old[GAP], new, g.position, cls, b3, prod)]
            differ = np.array([R.kind(o) != R.kind(n) for o, n in zip(old[GAP], new)])
            for k in sorted(set(np.array(keys)[differ])):
                sel = differ & (np.array(keys) == k)
                BUCKETS[k] = BUCKETS.get(k, 0) + int(sel.sum())
                act["b:" + k] = {GAP: np.where(sel, old[GAP], new).astype(object)}
        if GAP and os.environ.get("EIGHT"):
            # candidate eighth rules for the gap: what to do with RBs 25 or under priced 101-200 (rule 6 sells the ones not producing),
            # WRs 25 or under at that price, and the QB buy
            new = live[GAP]
            late = g.draft_round_f.fillna(8).to_numpy() >= 4
            yrb = np.array([p == "RB" and c[1] == "25-" and 100 < r <= 200 for p, c, r in zip(g.position, cls, g.mkt_rank)])
            ywr = np.array([p == "WR" and c[1] == "25-" and 100 < r <= 200 for p, c, r in zip(g.position, cls, g.mkt_rank)])
            qbb = np.array([p == "QB" and R.kind(n) == "buy" for p, n in zip(g.position, new)])
            def put(mask, call):
                return {GAP: np.where(mask, call, new).astype(object)}
            act["e:buy RB 25- 101-200, NFL round 4+"] = put(yrb & late & ~prod, "Buy")
            act["e:hold RB 25- 101-200, NFL round 4+"] = put(yrb & late & ~prod, "Hold")
            act["e:buy RB 25- 101-200, NFL rounds 1-3"] = put(yrb & ~late & ~prod, "Buy")
            act["e:hold RB 25- 101-200, any round"] = put(yrb & ~prod, "Hold")
            act["e:buy RB 25- 101-200, any round"] = put(yrb & ~prod, "Buy")
            act["e:buy WR 25- 101-200"] = put(ywr, "Buy")
            act["e:buy WR 25- 101-200, NFL rounds 1-3"] = put(ywr & ~late, "Buy")
            act["e:no QB buy"] = put(qbb, "Hold")
            act["e:buy late RB + young WR, no QB buy"] = put(qbb, "Hold")
            act["e:buy late RB + young WR, no QB buy"][GAP][yrb & late & ~prod] = "Buy"
            act["e:buy late RB + young WR, no QB buy"][GAP][ywr] = "Buy"
            act["e:buy late RB, no QB buy"] = put(qbb, "Hold")
            act["e:buy late RB, no QB buy"][GAP][yrb & late & ~prod] = "Buy"
        if os.environ.get("FINAL"):
            for name, rules in FINAL.items():
                act[name] = {m: np.array([more(b, p, c[1], r, pd_, a, rules) for b, p, c, r, pd_, a in
                                          zip(act["simple"][m], g.position, cls, g.mkt_rank, prod, g.age)], dtype=object) for m in NAME}
        if os.environ.get("GEN"):
            for name, fn in GEN_ADDS.items():
                ov = [fn(p, c[1], c[2], r, pd_) for p, c, r, pd_ in zip(g.position, cls, g.mkt_rank, prod)]
                act["g:" + name] = {m: np.array([o or b for o, b in zip(ov, act["simple"][m])], dtype=object) for m in NAME}
        if os.environ.get("REB"):
            for name, fn in REB_ADDS.items():
                ov = [fn(p, c[1], c[2], r) for p, c, r in zip(g.position, cls, g.mkt_rank)]
                act["r:" + name] = {**act["simple"], "rebuild": np.array([o or b for o, b in zip(ov, act["simple"]["rebuild"])], dtype=object)}
        for name, (bar, chk) in LOCO.items():
            TESTS[name] = unified(bar, without=t) if chk == "u" else from_tests(bar, chk, without=t)
        for name, tab in TESTS.items():
            act[name] = {m: np.array([tab[m].get(c, "Hold") for c in cls], dtype=object) for m in NAME}
            act[name + "+t"] = {m: np.array([timing(m, p, a, pd_, b) for p, a, pd_, b in zip(g.position, g.age, prod, act[name][m])], dtype=object)
                                for m in NAME}
        if SIMP:                                              # only the site's calls and the simplified sets
            act = {"live": act["live"], **simplified(g, t, pr, look)}
            sl = simplified.live                              # its own read of the live calls has to match middle_mode's
            assert all((sl[m] == live[m]).all() for m in NAME)
        for arm in ARMS:
            counts.append({"t": t, "arm": arm, **{f"{m[:3]} {k[:4]}": int(sum(R.kind(x) == k for x in act[arm][m])) for m in NAME for k in ("buy", "sell", "use")}})
        yrs = [y for y in range(t + 1, t + 1 + mm.H) if y <= su.LAST]
        P = np.stack([g.gsis_id.map(pts.get(y, {})).fillna(0).to_numpy() for y in yrs], 1)
        P = np.pad(P, ((0, 0), (0, mm.H - P.shape[1])))
        nxt1 = g.gsis_id.map(rank_at.get(t + 1, pd.Series(dtype=float))).to_numpy(float)
        rk1 = g.gsis_id.map(rank_at[t + 1]).to_numpy(float) if t + 1 in rank_at else None
        rk3 = g.gsis_id.map(rank_at[t + 3]).to_numpy(float) if t + 3 in rank_at else None
        jobs.append((t, g, P, act, rk1, rk3, nxt1, int(rng.integers(1 << 31))))
    if GAP:
        for j in jobs:                                        # every cohort runs every bucket: one with no such player keeps the seven rules
            for k in BUCKETS:
                j[3].setdefault("b:" + k, {GAP: j[3]["live"][GAP]})
        ARMS.extend(["old", "seven over old"] + ["b:" + k for k in BUCKETS])
        if os.environ.get("EIGHT"):
            ARMS.extend(a for a in jobs[0][3] if a.startswith("e:"))
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(f"== {'Superflex' if SF else '1QB'}: players of 340 called Buy / Sell / Use, then sell, mean over cohorts")
    print(pd.DataFrame(counts).groupby("arm").mean().drop(columns="t").round(0).to_string())
    with Pool(JOBS) as pool:
        res = pool.map(run_cohort, jobs)
    cols = ["t", "rep", "rank", "mode", "arm", "p1", "p2", "p3", "val1", "val3", "sold", "later", "bought"]
    df = pd.DataFrame([r for part in res for r in part], columns=cols)
    df.round(5).to_csv(out_dir(SF) / f"ruleset_sim{os.environ.get('OUT', '')}.csv.gz", index=False)
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
        for arm in [a for a in z.arm.unique() if a != "none"]:
            q = q0[q0.arm == arm].set_index(key)
            rows.append({"team": NAME[m], "arm": arm, "trades": round((q.sold + q.later + q.bought).mean(), 1), **cells(q[mets] - base.loc[q.index])})
            if arm not in ("live", "random"):
                rows2.append({"team": NAME[m], "arm": f"{arm} less live", **cells(q[mets] - live.loc[q.index, mets])})
            if arm in FINAL and "hybrid" in set(q0.arm):
                hy = q0[q0.arm == "hybrid"].set_index(key)
                rows2.append({"team": NAME[m], "arm": f"{arm} less hybrid", **cells(q[mets] - hy.loc[q.index, mets])})
    print(f"\n== {'Superflex' if SF else '1QB'}: each set against no trades. mean, z (leagues as clusters), cohorts above zero / cohorts")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n-- the new sets less the site's calls, same leagues")
    print(pd.DataFrame(rows2).to_string(index=False))
    badd = [a for a in z.arm.unique() if a.startswith(("b:", "e:"))]
    if badd:
        m = z[z.arm.isin(badd)]["mode"].iloc[0]
        q0 = z[z["mode"] == m]
        sev = q0[q0.arm == "live"].set_index(key)
        rows5 = []
        for arm in ["old", "seven over old"] + badd:
            q = q0[q0.arm == arm].set_index(key)
            d = q[mets] - sev.loc[q.index, mets]
            rows5.append({"the old calls put back": arm[2:] if arm.startswith("b:") else f"({arm})",
                          "trades": round((q.sold + q.later + q.bought).mean() - (sev.sold + sev.later + sev.bought).mean(), 2),
                          "_s": d.yrs2.mean(), **cells(d)})
        t5 = pd.DataFrame(rows5).sort_values("_s", ascending=False).drop(columns="_s")
        print(f"\n-- {NAME[m]}: the old calls put back into the seven rules one bucket at a time, less the seven rules; sorted by seasons 1-2")
        print(t5.to_string(index=False))
    gadd = [a for a in z.arm.unique() if a.startswith("g:")]
    if gadd:
        rows4 = []
        for m in NAME:
            q0 = z[z["mode"] == m]
            sim = q0[q0.arm == "simple"].set_index(key)
            for arm in gadd + ["hybrid", "live"]:
                q = q0[q0.arm == arm].set_index(key)
                rows4.append({"team": NAME[m], "added to the four rules": arm[2:] if arm.startswith("g:") else f"({arm})",
                              "trades": round((q.sold + q.later + q.bought).mean() - (sim.sold + sim.later + sim.bought).mean(), 1),
                              **cells(q[mets] - sim.loc[q.index, mets])})
        print("\n-- one rule added to the four rules (the same call for every team type), less the four rules alone")
        print(pd.DataFrame(rows4).to_string(index=False))
    radd = [a for a in z.arm.unique() if a.startswith("r:")]
    if radd:
        q0 = z[z["mode"] == "rebuild"]
        sim = q0[q0.arm == "simple"].set_index(key)
        rows3 = []
        for arm in radd + ["live", "hybrid"]:
            q = q0[q0.arm == arm].set_index(key)
            rows3.append({"added to the four rules": arm[2:] if arm.startswith("r:") else f"({arm})",
                          "trades": round((q.sold + q.later + q.bought).mean() - (sim.sold + sim.later + sim.bought).mean(), 1),
                          **cells(q[mets] - sim.loc[q.index, mets])})
        print("\n-- Rebuilding teams: one rule added to the four rules, less the four rules alone")
        print(pd.DataFrame(rows3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        summarize(pd.read_csv(out_dir(SF) / f"ruleset_sim{os.environ.get('OUT', '')}.csv.gz"))
    else:
        main()
