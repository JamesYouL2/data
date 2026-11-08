"""Q: the site's price bands (FP rank 1-50, 51-100, 101-150, 151-215, 216-239, 240-300, unranked) came from rookie-pick
prices. Moving to the strength buckets (1-50, 51-100, 101-200, 201-300; 300+ stays the waiver line) merges bands
that had different calls. Which call should each merged cell get?

Rows, features and lenses as call_strength.py (FP top 340, cohorts 2018-23; keep3 2018-21). Calls come from the rules
before the change (git 275513b), with the "producing now" switch off so each call is the raw one. A family is a
call group without its price suffix (", core", ", 3rd-round price", ...), plus NFL capital for young players and
rookies. A cell is family x new bucket. Where the old bands inside a cell gave different calls, each mode's lens on
the whole cell picks the call: the rebuilder by keep3, the contender by prod1 (call_strength.py), each against the
level it backs. With v = the lens, or the lens without the best player when that is under 0.5 (one hit carries the
group):
    v >= 1.15, and >= 1 without the best player -> buy;  v >= 0.85 -> hold;  v >= 0.25 -> sell;  otherwise cut
(levels: cut 0, sell 1, hold / lottery 2, buy 3). Of the calls the old bands gave in that cell, the one whose level
is nearest wins; ties, and cells under MIN_N players, go to the call most of the cell's players had. A call never
moves past what some old band gave. Only cells whose bands' usual calls differ are decided here; splits inside a
band (age, NFL round, production tier) stay as they were. Writes outputs/[sf/]rebucket.csv. LEAGUE=sf for superflex.

Findings (2026-11; per-cell numbers are in the rules.py notes marked "rebucket.py"):
- Superflex QBs 26-29 priced 101-200 and 1QB ones priced 201-300 become Buys (keep3 2.26 / 2.86); SF 201-300 backups
  stay a rebuilder Cut.
- WRs and TEs 26-29 priced 101-150 (was Hold) join 151-200 as rebuilder sells (keep3 0.44-0.74); contenders keep the
  WRs (prod1 1.31-1.36). TEs 26-29 priced 201-300 are cut (0.08 / 0.13 without the best).
- Young Day-3 RBs priced 151-239 and young WRs at 201-239 lose the "Buy with a 3rd": 201-300 is a Lottery ticket
  (keep3 0.52-0.55 for RBs). Young NFL R1-3 WRs priced 201-300 are sells (0.31 / 0.57).
- Contenders buy WRs 30+ with a 2-year window from 101 on, not 151 (prod1 1.81 / 2.45).
- 30+ fading: sell through 200; at 201-300 RBs and WRs are cut (keep3 0.00-0.34, ~0 without the best), TEs sold.
Overrides of the rule, for one call in both formats: "Young RB, NFL R1-3" 201-300 is Cut (SF: majority Sell, n=4);
"WR 26-29" 201-300 is a contender Sell (SF prod1 0.44 said Cut); "30+, fading" keeps the same call for the contender
as for the rebuilder (prod1 said Sell where keep3 said Cut); "Young QB, NFL R1-3" 201-300 in SF goes from Buy to Hold
(all six returned ~0, but the cell had one call). Rookie TEs R5+ in SF: 2 players.
League sim (middle_mode.py) checks of the result: a rebuilder Buy on "Young RB, NFL R4+/UDFA" 101-200 (the rule's
Hold: keep3 1.22 SF, 0.81 without Ekeler 2018) lifted superflex Rebuilding's 5-year titles at ranks 4-8 from 23.0 to
45.7 points and value 3 years on from 516 to 537, but cost 1QB value (641 -> 510). It stays Hold: one player's season
is not enough to move a call. Undoing the other big changes (rebuilders hold SF QBs 26-29 at 101-200; Buy SF young NFL R1-3 WRs at
101-200; hold young NFL R1-3 RBs at 101-200, young NFL R1-3 WRs at 201-300, WRs 26-29 at 101-200) did worse.
In-sample: the calls are picked on the same 2018-23 cohorts they are scored on.
"""
import pathlib
import subprocess
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import HERE, out_dir  # noqa: E402
import call_strength as cs  # noqa: E402

SF, F, MIN_N = cs.SF, cs.F, cs.MIN_N
OLD = "275513b"
SUFFIX = (", core", ", filler", ", 3rd-round price", ", 4th-round price", ", unranked")
LEVEL = {"buy": 3, "hold": 2, "lotto": 2, "sell": 1, "use": 1, "cut": 0}


def old_rules():
    path = HERE.parent / "pages" / "build" / "rules.py"
    src = subprocess.run(["git", "show", f"{OLD}:pages/build/rules.py"], capture_output=True, text=True,
                         check=True, cwd=HERE.parent).stdout
    m = types.ModuleType("rules_old")
    m.__file__ = str(path)
    exec(compile(src, str(path), "exec"), m.__dict__)
    return m


def bucket(r):
    return "U" if r is None or np.isnan(r) or r > 300 else "1-50" if r <= 50 else "51-100" if r <= 100 \
        else "101-200" if r <= 200 else "201-300"


def family(g, acls, rd):
    if g.startswith("Rookie RB") and rd >= 2:          # sell-high (by round) and lottery groups
        return "Rookie RB, NFL R2+"
    if g.startswith(("Rookie WR", "Rookie TE")):         # hold-to-February and late-rookie lottery groups
        return g.split(",")[0] + (", NFL R5+/UDFA" if rd >= 5 else ", NFL R1-4")
    for s in SUFFIX + (", backup",):
        if g.endswith(s) and not g.startswith("QB 30+"):
            g = g[: -len(s)]
    if acls in ("25-", "rookie") and "NFL" not in g and not g.startswith("Young QB priced"):
        g += ", NFL R1-3" if rd <= 3 else ", NFL R4+/UDFA"
    return g


def lens(q, col):
    num, den = (q.vorp1, q.v1fair) if col == "con" else (q.hold, q.fair)
    if len(q) == 0 or den.sum() <= 0:
        return np.nan, np.nan
    k = num.sum() / den.sum()
    best = (num - den).idxmax()
    solo = (num.sum() - num[best]) / (den.sum() - den[best]) if len(q) > 1 else np.nan
    return k, solo


def target(k, solo):
    if np.isnan(k):
        return np.nan
    v = solo if solo < 0.5 else k
    return 3 if v >= 1.15 and solo >= 1 else 2 if v >= 0.85 else 1 if v >= 0.25 else 0


def main():
    R = old_rules()
    H = cs.horizons()
    out = []
    for i, row, ctx in cs.features(H[1]):
        g, (ra, _), (ca, _) = R.base_calls(row._replace(prk=np.nan), F, ctx)
        r = getattr(row, f"r_{F}")
        ac = cs.R.age_class(row, ctx)
        out.append((i, g, ra, ca, R.band(r), bucket(r), ac, family(g, ac, row.rd)))
    c = pd.DataFrame(out, columns=["i", "grp", "reb", "con", "band", "bk", "acls", "fam"]).set_index("i")
    d1 = H[1].join(c)
    d3 = H[3].join(d1.set_index(["gsis_id", "t"])[c.columns], on=["gsis_id", "t"])
    res = []
    for (fam, bk), q1 in d1.groupby(["fam", "bk"]):
        q3 = d3[(d3.fam == fam) & (d3.bk == bk)]
        row = {"family": fam, "bucket": bk, "n": len(q1), "n3": len(q3), "bands": "/".join(sorted(q1.band.unique())),
               "vorp3": q3.vorp3.mean(), "rank_next": q1.rank_h.median()}
        for col, q in (("reb", q3), ("con", q1)):
            by = q1.groupby(col).size().sort_values(ascending=False)
            cands = list(by.index)
            per = q1.groupby("band")[col].agg(lambda s: s.mode().iloc[0])
            row[f"{col}_old"] = " ".join(f"{b}:{a}" for b, a in per.items())
            k, solo = lens(q, col)
            row[f"{col}_lens"], row[f"{col}_solo"] = k, solo
            if per.nunique() == 1:
                row[f"{col}_new"], row[f"{col}_why"] = per.iloc[0], "same"
                continue
            t = target(k, solo)
            if len(q) < MIN_N or np.isnan(t):
                row[f"{col}_new"], row[f"{col}_why"] = cands[0], f"majority (n={len(q)})"
                continue
            dist = {a: abs(LEVEL[R.kind(a)] - t) for a in cands}
            best = min(dist.values())
            pick = next(a for a in cands if dist[a] == best)   # cands are in majority order
            row[f"{col}_new"], row[f"{col}_why"] = pick, f"lens -> level {t:.0f}"
        res.append(row)
    tab = pd.DataFrame(res)
    pd.set_option("display.width", 260, "display.max_colwidth", 40, "display.max_rows", 500)
    mixed = tab[(tab.reb_why != "same") | (tab.con_why != "same")]
    print(f"== {'Superflex' if SF else '1QB'}: cells whose old bands disagreed")
    print(mixed.round(2).to_string(index=False))
    tab.to_csv(out_dir(SF) / "rebucket.csv", index=False)


if __name__ == "__main__":
    main()
