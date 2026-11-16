"""Q: which of the site's backtested calls, and which of the pools that stand in for thin ones, hold up at 2.5 standard errors?

Rows, calls and lenses as call_strength.py: FP dynasty top 340 per cohort, the site's call for each player (rules.classify before the
strength split), and what he returned against the average player at his price:
    keep3  3 seasons of VORP plus his value after them    cohorts 2018-21   Rebuilding's buys; every mode's sells and cuts
    keep1  1 season plus his value after it               cohorts 2018-23   Default's buys
    prod1  next season's VORP                             cohorts 2018-23   Contending's buys
A lens is a ratio of sums, 1.0 = what a fair trade returns. Its standard error is the ratio estimator's, with a player's seasons as one
cluster; z = (lens - 1) / SE. A call holds if z is at least 2.5 in its direction (buy above 1; sell or cut below 1) and the lens is on
that side of 1 in every cohort but one. Holds are not tested: a hold claims nothing.
1. Every call cell the site uses: call group x mode.
2. The pools call_strength.py falls back on: position x age class x price, position x age class, age class (RB / WR / TE), every lens.
Caveats: seasons of one cohort share a league year, so the standard errors are too small; the calls were picked on these cohorts.
LEAGUE=sf for superflex. Prints only.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
import call_strength as cs  # noqa: E402

R, Z = cs.R, 2.5
LENS_H = {"keep3": 3, "keep1": 1, "prod1": 1}


def test(q, lens):
    num, den = (q.vorp1, q.v1fair) if lens == "prod1" else (q.hold, q.fair)
    if len(q) < 2 or den.sum() <= 0:
        return None
    r = num.sum() / den.sum()
    e = (num - r * den).groupby(q.gsis_id).sum()
    se = np.sqrt((e ** 2).sum()) / den.sum()
    coh = pd.DataFrame({"t": q.t, "n": num, "d": den}).groupby("t").sum()
    return {"n": len(q), "players": q.gsis_id.nunique(), "lens": r, "z": (r - 1) / se if se > 0 else 0.0,
            "above": int((coh.n > coh.d).sum()), "cohorts": len(coh)}


def holds(s, side):
    if s is None:
        return False
    if side == "buy":
        return s["z"] >= Z and s["above"] >= s["cohorts"] - 1
    return s["z"] <= -Z and s["above"] <= 1


def main():
    pd.set_option("display.width", 250, "display.max_rows", 600, "display.max_colwidth", 44)
    H = cs.horizons()
    H[1] = cs.calls(H[1])
    H[3] = H[3].join(H[1].set_index(["gsis_id", "t"])[["grp", "reb", "con", "mid", "band", "pb", "acls"]], on=["gsis_id", "t"])
    rows = []
    for mode in ("reb", "mid", "con"):
        for (g, act), _ in H[1].groupby(["grp", mode]):
            k = R.kind(act)
            if k not in ("buy", "sell", "cut"):
                continue
            lens = cs.BUY_LENS[mode] if k == "buy" else "keep3"
            x = H[LENS_H[lens]]
            s = test(x[(x.grp == g) & (x[mode] == act)], lens)
            rows.append({"mode": cs.NAME[mode], "group": g, "call": act, "lens": lens,
                         **({"n": 0} if s is None else {"n": s["n"], "ratio": round(s["lens"], 2), "z": round(s["z"], 1),
                                                         "cohorts above 1": f"{s['above']}/{s['cohorts']}"}),
                         "holds": "YES" if holds(s, "buy" if k == "buy" else "sell") else ""})
    t = pd.DataFrame(rows)
    print(f"== {'Superflex' if cs.SF else '1QB'}: 1. the site's action calls (holds = z at least {Z} in the call's direction, and the right side of 1 in every cohort but one)")
    t["kind"] = [R.kind(a) for a in t.call]
    summ = t.groupby(["mode", "kind"]).agg(cells=("holds", "size"), hold=("holds", lambda s: (s == "YES").sum()),
                                           players=("n", "sum"), players_in_holding=("n", lambda s: s[t.loc[s.index, "holds"] == "YES"].sum()))
    print(summ.to_string())
    print("\n-- the cells that hold")
    print(t[t.holds == "YES"].drop(columns=["holds", "kind"]).sort_values(["mode", "z"]).to_string(index=False))
    print("\n-- the largest cells that don't (30+ players)")
    print(t[(t.holds == "") & (t.n >= 30)].drop(columns=["holds", "kind"]).sort_values(["mode", "n"], ascending=[True, False]).to_string(index=False))

    print(f"\n== 2. the pools: lens and z for every pool of 15+ players; marked where |z| >= {Z} with the cohorts agreeing")
    prow = []
    for lens, h in LENS_H.items():
        x = H[h]
        pools = {}
        for a, q in x[x.position != "QB"].groupby("acls"):
            pools[("3 age class", f"RB/WR/TE {a}")] = q
        for (p, a), q in x.groupby(["position", "acls"]):
            pools[("2 position x age", f"{p} {a}")] = q
        for (p, a, pb), q in x.groupby(["position", "acls", "pb"]):
            pools[("1 position x age x price", f"{p} {a} | {pb}")] = q
        for (lvl, name), q in pools.items():
            s = test(q, lens)
            if s is None or s["n"] < 15:
                continue
            side = "buy" if holds(s, "buy") else "sell" if holds(s, "sell") else ""
            prow.append({"level": lvl, "pool": name, "lens": lens, "cell": f"{s['lens']:.2f} z{s['z']:+.1f} {s['above']}/{s['cohorts']}" + (f" {side.upper()}" if side else ""),
                         "n": s["n"], "side": side})
    p = pd.DataFrame(prow)
    wide = p.pivot(index=["level", "pool"], columns="lens", values="cell")[list(LENS_H)]
    wide["n (1-yr lenses)"] = p[p.lens == "keep1"].set_index(["level", "pool"]).n
    print(wide.fillna("").to_string())
    print("\npools that hold, by lens:")
    print(p[p.side != ""].groupby(["lens", "level", "side"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
