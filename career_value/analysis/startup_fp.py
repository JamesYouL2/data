"""Q: does the startup lean (ADP + youth / RB tax / injury tax / SF QB bump) still add value on top of FP dynasty ECR?

Same simulation as startup.py, on drafts after seasons t = 2018-2024 (FP dynasty coverage). Player pool: veterans in
MFL's top 200 or FP's top 200, each source re-ranked within the pool (missing = after its last ranked player).
Two leagues ("rooms"): the other eleven teams draft by MFL ADP, or by FP ECR (both with log-normal noise). The focal
team starts from a board -- ADP, FP, or Bw = ADP blended with w% FP (log-rank average) -- raw or with startup.py's
adjusted-rank lean (age x1.03 or x1.06 per year, RB x1.1, missed 8+ games x1.15, SF QB x0.86). Windows use only
drafts with all seasons played (year 1: 2018-24, years 1-2: 2018-23, years 1-3: 2018-22). Also prints the lean FP
already applies relative to ADP (regression of log FP rank / ADP rank on age and position).
Env: LEAGUE=sf, NOISE, REPS (60), BOARDS ("ADP;B10;B20;FP"; also B33, B50), FOCAL ("label=strategy;...").
Writes outputs/[sf/]startup_fp_sim.csv with the default settings.
"""
import os

import numpy as np
import pandas as pd

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import HERE, out_dir, rows
from long_horizon import SF
import startup as su

REPS = int(os.environ.get("REPS", 60))
QB = ",qb=-0.15" if SF else ""
FOCAL = {"raw": "ADP", "lean 2yr (age x1.03)": f"adj:y=0.03,rb=0.1,inj=0.15{QB}",
         "lean 3-5yr (age x1.06)": f"adj:y=0.06,rb=0.1,inj=0.15{QB}"}
BOARDS = os.environ.get("BOARDS", "ADP;B10;B20;FP").split(";")
if os.environ.get("FOCAL"):   # e.g. FOCAL="raw=ADP;inj=adj:inj=0.15"
    FOCAL = dict(z.split("=", 1) for z in os.environ["FOCAL"].split(";"))


def pool(t, m, f):
    """veterans in MFL's or FP's top 200; each source re-ranked 1..n within the pool, its unranked players after
    its last ranked one, ordered by the other source"""
    a, b = m[m.t == t], f[(f.t == t) & (f.row_type == "veteran")]
    g = pd.concat([a, b[~b.gsis_id.isin(a.gsis_id)]]).reset_index(drop=True)
    ra = g.gsis_id.map(a.set_index("gsis_id").mkt_rank)
    rb = g.gsis_id.map(b.set_index("gsis_id").mkt_rank)
    for col, own, other in (("adp", ra, rb), ("fp", rb, ra)):
        key = pd.DataFrame({"miss": own.isna(), "own": own, "other": other.fillna(999)})
        g[col] = key.sort_values(["miss", "own", "other"]).assign(r=np.arange(1, len(g) + 1)).r.reindex(g.index)
    for w in (10, 20, 33, 50):   # ADP blended with w% FP (log-rank average)
        g[f"b{w}"] = ((1 - w / 100) * np.log(g.adp) + w / 100 * np.log(g.fp)).rank(method="first")
    return g


def main():
    m, f = rows(["mfl_dynasty"], 1, SF), rows(["fp_dynasty"], 1, SF)
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pts = s.pivot_table(index="gsis_id", columns="season", values="ppr", fill_value=0.0)
    G = pd.concat([pool(t, m, f).assign(t=t) for t in range(2018, 2025)])
    G = G[(G.adp <= 150) & (G.fp <= 150)]
    terms = {"age-26": G.age - 26, **{q: (G.position == q).astype(float) for q in ("RB", "TE", "QB")},
             "inj": (G.games.fillna(0) <= 9).astype(float)}
    b = np.linalg.lstsq(np.column_stack([np.ones(len(G)), *terms.values()]), np.log(G.fp / G.adp), rcond=None)[0]
    print("lean FP applies vs ADP (rank multiplier; >1 = FP ranks him lower):",
          {k: round(float(np.exp(v)), 2) for k, v in zip(terms, b[1:])})
    rng = np.random.default_rng(0)
    res = []
    for t in range(2018, 2025):
        g = pool(t, m, f)
        yrs = [y for y in range(t + 1, t + 4) if y <= su.LAST]
        P = np.stack([pts.reindex(g.gsis_id)[y].fillna(0).to_numpy() for y in yrs], 1)
        pos = g.position.to_numpy()
        for rep in range(REPS):
            slot, seed = rep % su.TEAMS, rng.integers(1 << 31)
            for league in ("ADP", "FP"):
                opp = g[league.lower()].to_numpy()
                for board in BOARDS:
                    x = g.assign(mkt_rank=g[board.lower()])
                    for lab, st in FOCAL.items():
                        team = su.draft(x, st, slot, np.random.default_rng(seed), opp=opp)
                        for k in range(1, len(yrs) + 1):
                            L = np.array([su.lineup(P[team == i, k - 1], pos[team == i]) for i in range(su.TEAMS)])
                            res.append((t, rep, league, f"{board} {lab}", k, (L > L[slot]).sum() == 0))
    r = pd.DataFrame(res, columns=["t", "rep", "league", "focal", "k", "title"])
    if "BOARDS" not in os.environ and "FOCAL" not in os.environ and su.NOISE == 0.2 and REPS == 60:
        r.to_csv(out_dir(SF) / "startup_fp_sim.csv", index=False)
    out = {}
    for w, last in ((1, 2024), (2, 2023), (3, 2022)):
        z = r[(r.k <= w) & (r.t <= last)]
        out[f"yrs 1-{w}" if w > 1 else "yr 1"] = z.groupby(["league", "focal"]).title.mean() * w
    tab = pd.DataFrame(out).round(2)
    print(f"== {'Superflex' if SF else '1QB'}: expected titles per startup, FP-era drafts, {REPS} reps per draft, "
          f"noise {su.NOISE}")
    print(tab.to_string())
    z = r[(r.k <= 2) & (r.t <= 2023)].groupby(["league", "focal", "t"]).title.mean().unstack() * 2
    print("\nyears 1-2 titles by draft year")
    print(z.round(2).to_string())


if __name__ == "__main__":
    main()
