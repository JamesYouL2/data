"""Q: which in-season signals hold up as simple rules, one signal for one team type with no exemptions, at 2.5 standard errors?

The league sim of market_price_sim.py at market prices. The base is the site's calls with no signals (nosig). Each arm lays one of
rules.signals' flags on that base for every flagged player, hurt or healthy, any position, the way the site applies a signal:
  sells   Hold and Use, then sell become Sell; Buy becomes Hold; Sell and Cut stay
          s_rosdown   rest-of-season rank a quarter worse than the preseason
          s_faller    dynasty rank a quarter worse than the preseason
          s_roslow    rest-of-season positional rank a quarter worse than the market's
  buys    Hold becomes Buy; Sell, Cut and Use, then sell become Hold; Buy stays
          b_gap       25 or under (not a rookie), rest-of-season positional rank a quarter better than dynasty
          b_rookie    a rookie with that gap
Every arm runs for every team type on its own teams (Contending 1st-3rd, Default 4th-8th, Rebuilding 9th-12th by projection).
Reported as the arm less the base: mean, z = mean / standard error with leagues (cohort and draft) as clusters, and cohorts above zero,
for title odds in season 1 and seasons 1-3 and value (340 - market rank) 1 and 3 years on.
Test: a rule holds if z >= 2.5 on its team type's own objective in both KTC and FantasyCalc. Objectives: Contending title odds this
season; Default title odds over seasons 1-3; Rebuilding value 3 years on. `table` reads the four saved runs and prints the verdicts.
A rule that holds alone goes into the set and is then left out of it in turn (signal_sim.py: noros, nofaller, nobuy); one whose removal
raises the team's objective is dropped. That removed ros_down and faller for 1QB Contending.
Caveats: leagues in a cohort share players, so these standard errors are too small; 3-year results rest on 2-3 cohorts.
Writes outputs/[sf/]simple_rules_sim_<source>.csv.gz. LEAGUE=sf; SOURCE=ktc or fantasycalc; REPS, WEEKS, JOBS.
"""
import os
import pathlib
import sys

os.environ["ARMS"] = "live,nosig"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import market_price_sim as mps  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rank_input_sim as rs  # noqa: E402
import signal_sim as ss  # noqa: E402
from common import HERE, out_dir  # noqa: E402

R, F, MODES = mps.R, mps.F, mps.MODES
SELL = {"s_rosdown": lambda s: "ros_down" in s, "s_faller": lambda s: "faller" in s, "s_roslow": lambda s: "ros_low" in s}
BUY = {"b_gap": lambda s: "gap" in s and not s.get("rookie"), "b_rookie": lambda s: "gap" in s and bool(s.get("rookie"))}
ARMS = ("live", "nosig", *SELL, *BUY)
GOAL = {"con": "yr1", "mid": "yrs3", "reb": "v3"}
METS = ["yr1", "yrs3", "v1", "v3"]
Z = 2.5


def snapshot_calls(q, last, w, mk, mkpre):
    c = ss.snapshot_calls(q, last, w, mk, mkpre)
    rows, _ = rs.features(q, last)
    rows, _ = mps.market_rows(q, rows, mk, mkpre)
    ctx = {"ROOKIE": int(q.season.iloc[0]), "LOOK": rs.LOOK, "EARLY": False, "WEEK": w + 1}
    sig = [R.signals(x, F, ctx) for x in rows]
    out = {}
    for m in MODES:
        out["live", m], out["nosig", m] = c["live", m], c["nosig", m]
        for arm, fn in {**SELL, **BUY}.items():
            to = {"hold": "sell", "use": "sell", "buy": "hold"} if arm in SELL else {"hold": "buy", "sell": "hold", "use": "hold", "cut": "hold"}
            k = dict(c["nosig", m])
            for g, s in zip(q.gsis_id, sig):
                if fn(s):
                    k[g] = to.get(k.get(g), k.get(g))
            out[arm, m] = k
    return out


def diffs(df):
    """arm less nosig on own-mode teams: rows (arm, mode, metric) -> mean, z, cohorts above zero / cohorts, trades"""
    key = ["t", "rep", "wk", "rank"]
    own = {m: list(r) for m, r in mps.LENS.values()}
    z = df[[r in own.get(m, ()) for r, m in zip(df["rank"], df["mode"])]].copy()
    z["yr1"] = z.p1 * 100
    z["yrs3"] = np.where(z.t + 3 <= mps.su.LAST, (z.p1 + z.p2 + z.p3) * 100, np.nan)
    z["v1"], z["v3"], z["n"] = z.mval1, z.mval3, z.sell + z.later + z.buy
    z = z.set_index(key + ["mode", "arm"])[METS + ["n"]]
    rows = []
    for arm in ARMS:
        if arm == "nosig":
            continue
        d = z.xs(f"m:{arm}", level="arm") - z.xs("m:nosig", level="arm")
        for m in MODES:
            q = d.xs(m, level="mode")
            for c in METS:
                v = q[c].dropna()
                if v.empty:
                    continue
                lg, by = v.groupby(level=["t", "rep"]).mean(), v.groupby(level="t").mean()
                se = lg.std(ddof=1) / np.sqrt(len(lg))
                rows.append({"arm": arm, "mode": m, "met": c, "mean": lg.mean(), "z": lg.mean() / se if se > 0 else 0.0,
                             "up": int((by > 0).sum()), "cohorts": len(by), "trades": q.n.mean()})
    return pd.DataFrame(rows)


def summarize(df):
    pd.set_option("display.width", 250, "display.max_rows", 500)
    d = diffs(df)
    d["cell"] = [f"{a:+7.1f} z{b:+5.1f} {c}/{e}" for a, b, c, e in zip(d["mean"], d.z, d.up, d.cohorts)]
    t = d.pivot(index=["arm", "mode"], columns="met", values="cell")[METS]
    t["trades"] = d.groupby(["arm", "mode"]).trades.first().round(1)
    print(f"\n== {'Superflex' if mps.SF else '1QB'}, {mps.SOURCE}: each arm less the calls with no signals, own-mode teams. mean, z, cohorts above zero / cohorts")
    print(t.reindex([(a, m) for a in ARMS if a != "nosig" for m in MODES]).to_string())


def table():
    """verdicts from the four saved runs. plain: z >= Z on the objective at both prices. final: Rebuilding judged on value 1 year on
    (more cohorts than 3 years on); the gain above zero in all cohorts but one at each price; and for Default and Rebuilding no value
    measure at z <= -Z. Contending is let off the value guard: it trades value for this season. A Contending rule that costs title odds
    over seasons 1-3 at z <= -Z is marked, kept or not."""
    pd.set_option("display.width", 280, "display.max_rows", 500)
    goal2 = {**GOAL, "reb": "v1"}
    for sf in (False, True):
        parts = {}
        for src in ("ktc", "fantasycalc"):
            f = HERE / "outputs" / ("sf" if sf else "") / f"simple_rules_sim_{src}.csv.gz"
            parts[src] = diffs(pd.read_csv(f)).set_index(["arm", "mode", "met"])

        def get(src, arm, m, c):
            return parts[src].loc[(arm, m, c)] if (arm, m, c) in parts[src].index else None

        def show(r):
            return "" if r is None else f"{r['mean']:+.1f} z{r.z:+.1f} {int(r.up)}/{int(r.cohorts)}"
        rows = []
        for arm in [a for a in ARMS if a not in ("nosig", "live")]:
            for m in MODES:
                pair = lambda c: [get(s_, arm, m, c) for s_ in ("ktc", "fantasycalc")]  # noqa: E731
                a, b = pair(GOAL[m])
                plain = a is not None and b is not None and min(a.z, b.z) >= Z
                a2, b2 = pair(goal2[m])
                ok = a2 is not None and b2 is not None and min(a2.z, b2.z) >= Z and all(r.up >= r.cohorts - 1 for r in (a2, b2))
                harm = [f"{c} {r.z:+.1f}" for c in ("v1", "v3") for r in pair(c) if r is not None and r.z <= -Z]
                final = ok and not (harm and m != "con")
                t3 = [r for r in pair("yrs3") if r is not None]
                oof = m == "con" and any(r.z <= -Z for r in t3)
                rows.append({"rule": arm, "team": m, "objective": goal2[m], "KTC": show(a2), "FantasyCalc": show(b2),
                             "titles 3 seasons (KTC, FC)": " | ".join(show(r) for r in t3),
                             "value 1 yr (KTC, FC)": " | ".join(show(r) for r in pair("v1")), "trades": round(a2.trades, 1) if a2 is not None else "",
                             "plain": "yes" if plain else "", "final": "KEEP" if final else "",
                             "note": ("COSTS TITLES OVER 3 SEASONS " if oof else "") + ("value: " + ", ".join(harm) if harm else "")})
        print(f"\n== {'Superflex' if sf else '1QB'}: one signal for one team type, against the calls with no signals. mean, z, cohorts above zero / cohorts")
        print(pd.DataFrame(rows).to_string(index=False))


mps.OUT = "simple_rules_sim"
mps.F_ARMS, mps.M_ARMS, mps.RANDOM, mps.KIND = (), ARMS, (), ()
mps.snapshot_calls, mps.summarize = snapshot_calls, summarize

if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        table()
    else:
        mps.main()
