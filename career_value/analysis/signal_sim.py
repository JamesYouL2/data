"""Q: at market prices, which in-season signals earn their place, for which teams?

market_price_sim.py found the signals still add titles at KTC and FantasyCalc prices, but their value gain goes flat or
negative, and rebuilders lose value the next May. Same league, cohorts and prices; every trade clears at the market rank and value
is scored at market May ranks. Only one team trades in a sim, so each mode's row is that kind of team on its own.
Arms, against live (the site's rules with their current exemptions):
  nosig          no signals
  nosell         no sell signals: buys only
  nobuy          no buy signals: sells only
  noros          no sell on a fall in the rest-of-season rank
  nofaller       no sell on a fall in the dynasty rank (1QB only: in superflex that is only a note)
  norookie       no buy of rookies (rebuilders only)
  novetbuy       no buy of veterans; rebuilders keep their rookie buys, and only they get those
  nohurtsell     no sell signals on hurt players
  nowellsell     no sell signals on healthy players
  mfsell         no sell signals where the market is already well below FP (rules.gap, the site's own test: "less")
  mfbuy          no buy signals where the market is already well above FP (rules.gap: "more")
  msig           signals read the market's rank in place of FP's dynasty rank (would need a Sept 1 market snapshot)
mfsell and mfbuy could run on the site today: it has FantasyCalc's current ranks. Caveats as market_price_sim.py: the rules were fit on
FP ranks over these cohorts, 3-year results rest on 2-3 cohorts, and the KTC and FantasyCalc runs share players and seasons.
The saved outputs ran before the site dropped healthy sells for 1QB contenders and default teams and veteran buys for superflex rebuilders;
rerun, live has those changes and those arms match it. Writes outputs/[sf/]signal_sim_<source>.csv.gz, or signal_sim_<arms>_<source>.csv.gz with ARMS set. LEAGUE=sf for superflex;
SOURCE=ktc or fantasycalc; REPS, WEEKS, JOBS; ARMS=live,novetbuy to run only those (keep live first).
"""
import os
import pathlib
import sys
from contextlib import contextmanager

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import market_price_sim as mps  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rank_input_sim as rs  # noqa: E402

R, F = mps.R, mps.F
MODES, LENS = mps.MODES, mps.LENS
SELLS = ("ros_down", "faller")
ALL_ARMS = ("live", "nosig", "nosell", "nobuy", "noros", "nofaller", "norookie", "novetbuy", "nohurtsell", "nowellsell", "mfsell", "mfbuy", "msig")
ARMS = tuple(os.environ["ARMS"].split(",")) if os.environ.get("ARMS") else ALL_ARMS


def strip(arm, x, s):
    """what fired, less what this arm turns off"""
    gp = R.gap(getattr(x, f"r_{F}"), getattr(x, f"mr_{F}"))
    off = {"nosell": SELLS, "nobuy": ("gap",), "noros": ("ros_down",), "nofaller": ("faller",),
           "norookie": ("gap",) if s.get("rookie") else (), "novetbuy": ("gap",) if not s.get("rookie") else (),
           "nohurtsell": SELLS if s.get("hurt") else (), "nowellsell": SELLS if not s.get("hurt") else (),
           "mfsell": SELLS if gp == "less" else (), "mfbuy": ("gap",) if gp == "more" else ()}.get(arm, ())
    return {k: v for k, v in s.items() if k not in off}


@contextmanager
def signal_rules(arm):
    apply0 = R.apply_signal
    R.apply_signal = lambda act, mode, x, f, s: apply0(act, mode, x, f, strip(arm, x, s))
    try:
        yield
    finally:
        R.apply_signal = apply0


def snapshot_calls(q, last, w, mk, mkpre):
    """(arm, mode) -> {gsis_id: kind}"""
    rows, inputs = rs.features(q, last)
    rows, mx = mps.market_rows(q, rows, mk, mkpre)
    base = {"ROOKIE": int(q.season.iloc[0]), "LOOK": rs.LOOK, "EARLY": False}
    live = {**base, "WEEK": w + 1}
    out = {}
    for arm in ARMS:
        si, ctx = (mx if arm == "msig" else rows), (base if arm == "nosig" else live)
        res = {m: {} for m in MODES}
        with signal_rules(arm):
            for g, xs, xc in zip(q.gsis_id, si, inputs["tilt"]):
                c = R.situational(xs, F, R.classify(xc, F, ctx), ctx)
                res["reb"][g], res["con"][g], res["mid"][g] = rs.act_kind(c[1]), rs.act_kind(c[3]), rs.act_kind(c[7])
        for m in MODES:
            out[arm, m] = res[m]
    return out


def summarize(df):
    key = ["t", "rep", "wk", "rank"]
    H = mps.H
    pc = [f"p{k}" for k in range(1, H + 1)]
    base = df[df.arm == "none"].set_index(key)[pc + ["mval1", "mval3"]].add_suffix("_0")
    z = df[df.arm != "none"].join(base, on=key)
    z["yr1"] = (z.p1 - z.p1_0) * 100
    z["yrs3"] = np.where(z.t + 3 <= mps.su.LAST, sum(z[f"p{k}"] - z[f"p{k}_0"] for k in (1, 2, 3)) * 100, np.nan)
    z["v1"], z["v3"] = z.mval1 - z.mval1_0, z.mval3 - z.mval3_0
    z["trades"] = z.sell + z.later + z.buy
    mets = ["yr1", "yrs3", "v1", "v3"]
    own = {m: list(r) for m, r in LENS.values()}
    idx = z[[r in own.get(m, ()) for r, m in zip(z["rank"], z["mode"])]].set_index(key + ["mode", "arm"])
    pd.set_option("display.width", 250, "display.max_rows", 800, "display.max_columns", 40)

    def cell(d):
        v = d.dropna()
        if v.empty:
            return ""
        c = v.groupby(level=["t", "rep", "wk"]).mean()
        by = v.groupby(level="t").mean()
        return f"{c.mean():+6.1f}±{1.96 * c.std(ddof=1) / np.sqrt(len(c)):4.1f} {int((by > 0).sum())}/{int((by != 0).sum())}"

    def table(title, a, b=None):
        d = idx.xs(a, level="arm")[mets]
        n = idx.xs(a, level="arm").trades
        if b is not None:
            d, n = d - idx.xs(b, level="arm")[mets], n - idx.xs(b, level="arm").trades
        rows = [{"mode": m, **{c: cell(d.xs(m, level="mode")[c]) for c in mets}, "trades": round(n.xs(m, level="mode").mean(), 1)}
                for m in ("con", "mid", "reb")]
        print(f"\n-- {title}\n{pd.DataFrame(rows).to_string(index=False)}")

    print(f"\n== {'Superflex' if mps.SF else '1QB'}, {mps.SOURCE}, market prices, cohorts {sorted(int(t) for t in z.t.unique())}, weeks "
          f"{mps.WEEKS}, {mps.REPS} reps. Own-mode teams: con ranks 1-3, mid 4-8, reb 9-12. mean ± 95% CI, cohorts above zero / cohorts "
          "that differ. Positive = the first arm is better")
    table("live - no trades", "m:live")
    for arm in ARMS[1:]:
        table(f"{arm} - live", f"m:{arm}", "m:live")
    print("\n=== By cohort, against live")
    for arm in [a for a in ("nosig", "nosell", "nobuy", "novetbuy", "mfsell", "mfbuy") if a in ARMS]:
        d = idx.xs(f"m:{arm}", level="arm")[mets] - idx.xs("m:live", level="arm")[mets]
        print(f"\n-- {arm} - live\n{d.groupby(level=['mode', 't']).mean().round(1).to_string()}")


mps.OUT = "signal_sim" if ARMS == ALL_ARMS else "signal_sim_" + "_".join(ARMS[1:])
mps.F_ARMS, mps.M_ARMS, mps.RANDOM, mps.KIND = (), ARMS, (), ()
mps.snapshot_calls, mps.summarize = snapshot_calls, summarize

if __name__ == "__main__":
    mps.main()
