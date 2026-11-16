"""A startup drafted for seasons 2-4: where does that board differ from the FantasyPros list, and from the Default calls?

FP May rows 2018-21 (the cohorts whose fourth season is known), ranks 1-300. Targets: VORP in season 1, VORP in seasons 2-4,
seasons 2-4 plus the value of the rank held after season 4 ("2-4 + after"), and VORP over seasons 1-3, 1-4 and 1-5 (1-5: 2018-20 only). For a group, the factor is the multiplier on FP rank that
prices it fairly on that target (rank_factors.factor: sum fair(k x rank) = sum of what it returned; under 1 = draft him earlier than
FP ranks him). Groups: the live Default call, and position x age class. Range = each cohort left out.

  LEAGUE=sf uv run --with pandas --with pyarrow --with scikit-learn --with scipy --with statsmodels python analysis/startup_window.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import call_strength as cs  # noqa: E402
import market_residuals as M  # noqa: E402
import rank_factors as rf  # noqa: E402

F = cs.F
TARGETS = {"season 1": "y1", "seasons 2-4": "y24", "2-4 + after": "y24a", "seasons 1-3": "y13", "seasons 1-4": "y14", "seasons 1-5": "y15"}


def frame():
    x = rf.live_calls(cs.deep(1))
    s = pd.read_parquet(HERE.parent / "data" / "player_seasons.parquet")[["gsis_id", "season", M.VALUE]]
    v = s.pivot_table(index="gsis_id", columns="season", values=M.VALUE, fill_value=0.0)
    x = x[x.t + 4 <= M.LAST].reset_index(drop=True)
    get = lambda g, y: v.at[g, y] if g in v.index and y in v.columns else 0.0  # noqa: E731
    for k in range(1, 5):
        x[f"s{k}"] = [get(g, t + k) for g, t in zip(x.gsis_id, x.t)]
    r = cs.deep(1)[["gsis_id", "t", "mkt_rank"]]
    x = x.merge(r.assign(t=r.t - 4).rename(columns={"mkt_rank": "rank4"}), on=["gsis_id", "t"], how="left")
    X3 = cs.deep(3)
    v3 = cs.iso(X3.mkt_rank, X3.vorp3)
    after = np.where(x.rank4.notna(), v3.predict(np.log(x.rank4.fillna(cs.POOL))), 0.0)
    x["y1"], x["y24"], x["y13"] = x.s1, x.s2 + x.s3 + x.s4, x.s1 + x.s2 + x.s3
    x["y24a"] = x.y24 + after
    x["y14"] = x.y13 + x.s4
    x["y15"] = np.where(x.t + 5 <= M.LAST, x.y14 + np.array([get(g, t + 5) for g, t in zip(x.gsis_id, x.t)]), np.nan)   # 2018-20 only
    x["acls"] = np.where(x.row_type == "rookie", "rookie", np.where(x.age < 26, "25-", np.where(x.age < 28, "26-27", np.where(x.age < 30, "28-29", "30+"))))
    return x[x.mkt_rank <= 300].reset_index(drop=True)


def factor(x, g, col):
    y = x[x[col].notna()].assign(hold=lambda d: d[col])
    g = g[g[col].notna()]
    x = y
    k = rf.factor(y.loc[g.index], cs.iso(y.mkt_rank, y.hold))
    loo = []
    for t in sorted(x.t.unique()):
        z = y[y.t != t]
        gi = g.index[g.t != t]
        if len(gi) >= 5:
            loo.append(rf.factor(z.loc[gi], cs.iso(z.mkt_rank, z.hold)))
    return k, (min(loo), max(loo)) if loo else (np.nan, np.nan)


def table(x, by, title):
    rows = []
    for key, g in x.groupby(by):
        if len(g) < 12:
            continue
        r = {"group": " ".join(key) if isinstance(key, tuple) else key, "n": len(g), "median FP rank": int(g.mkt_rank.median()),
             "share of 4-season VORP in season 1": f"{g.s1.clip(lower=0).sum() / max(g[['s1', 's2', 's3', 's4']].clip(lower=0).sum().sum(), 1e-9):.0%}"}
        for lab, col in TARGETS.items():
            k, (lo, hi) = factor(x, g, col)
            r[lab] = f"{k:.2f} ({lo:.2f}-{hi:.2f})"
        rows.append(r)
    print(f"\n== {F}: {title}. Factor on FP rank (under 1 = take him earlier), cohorts {sorted(x.t.unique())}")
    print(pd.DataFrame(rows).to_string(index=False))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 230)
    x = frame()
    tot = x[["s1", "s2", "s3", "s4"]].clip(lower=0).sum()
    print(f"{F}: share of the pool's 4-season VORP by season:", (tot / tot.sum()).round(2).to_dict())
    a = table(x, "mid", "by the live Default call")
    b = table(x, ["position", "acls"], "by position and age")
    c = table(x.assign(band=pd.cut(x.mkt_rank, [0, 24, 60, 120, 300], labels=["1-24", "25-60", "61-120", "121-300"]).astype(str),
                       ag=np.where(x.row_type == "rookie", "rookie", np.where(x.age < 26, "25-", np.where(x.age < 28, "26-27", "28+")))),
              ["band", "ag"], "by FP band and age")
    pd.concat([a.assign(by="call"), b.assign(by="pos x age"), c.assign(by="band x age")]).to_csv(HERE.parent / "outputs" / f"startup_window_{F}.csv", index=False)


if __name__ == "__main__":
    main()
