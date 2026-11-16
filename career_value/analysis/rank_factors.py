"""Calibrate the Rankings tab's factors (pages/app.js LEAN): how far up or down the FantasyPros list a Buy, Use-then-sell or Sell
under the seven rules belonged, judged by what he returned.

Each FP row 2018-24 gets the live site's calls (pages/build/rules.classify, base calls: no in-season signals). For a team type and a
call, the factor k is the one multiplier on rank that makes the group fairly priced: sum of fair(k x rank) = sum of what they
returned, where fair(rank) is the average return at that rank (call_strength.horizons: VORP over h seasons + the value of the rank
held after). k < 1: they belonged higher on the list. Horizon: 1 season for Contending, 3 for Default and Rebuilding. The range is k
with each cohort left out in turn.

  LEAGUE=sf uv run --with pandas --with pyarrow --with scikit-learn --with scipy --with statsmodels python analysis/rank_factors.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import call_strength as cs  # noqa: E402

spec = importlib.util.spec_from_file_location("live_rules", HERE.parents[1] / "pages" / "build" / "rules.py")
LIVE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LIVE)
F, POOL = cs.F, cs.POOL
MODES = {"con": ("Contending", 1), "mid": ("Default", 3), "reb": ("Rebuilding", 3)}


def live_calls(x):
    out = []
    for i, row, ctx in cs.features(x):
        _, (ra, _), (ca, _), (ma, _) = LIVE.classify(row, F, ctx)
        out.append((i, LIVE.kind(ra), LIVE.kind(ca), LIVE.kind(ma)))
    return x.join(pd.DataFrame(out, columns=["i", "reb", "con", "mid"]).set_index("i"))


def factor(g, fair):
    """k with sum fair(k x rank) = sum hold (fair falls with rank, so bisect)"""
    lo, hi = 0.2, 5.0
    for _ in range(40):
        k = (lo * hi) ** 0.5
        if fair.predict(np.log(np.clip(g.mkt_rank * k, 1, POOL * 1.5))).sum() > g.hold.sum():
            lo = k
        else:
            hi = k
    return (lo * hi) ** 0.5


def main():
    H = {h: live_calls(x) for h, x in cs.horizons(hs=(1, 3)).items()}
    rows = []
    for m, (name, h0) in MODES.items():
        for h in (1, 3):
            x = H[h]
            fair = cs.iso(x.mkt_rank, x.hold)
            for kd in ("buy", "use", "sell", "hold"):
                g = x[(x[m] == kd) & (x.mkt_rank <= 300)]
                if len(g) < 15:
                    continue
                loo = [factor(g[g.t != t], cs.iso(x.mkt_rank[x.t != t], x.hold[x.t != t])) for t in sorted(g.t.unique())]
                rows.append({"team": name, "seasons": h, "used": "*" if h == h0 else "", "call": kd, "n": len(g), "returned / price": round(g.hold.sum() / g.fair.sum(), 2),
                             "factor": round(factor(g, fair), 2), "range": f"{min(loo):.2f}-{max(loo):.2f}", "median rank": int(g.mkt_rank.median())})
    out = pd.DataFrame(rows)
    print(f"== {F}: factor on FP rank that prices each call fairly (* = the horizon that team type is judged on)")
    print(out.to_string(index=False))
    out.to_csv(HERE.parent / "outputs" / f"rank_factors_{F}.csv", index=False)


if __name__ == "__main__":
    main()
