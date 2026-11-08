# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Regenerate build/old_lookup.json: how players aged 30+ produced over the next 3 and 5 seasons (1999+), by
position x age band x last-season PPG tier, in 1QB VORP and superflex VORP.

Needs the career_value dataset from the DynastyProcess data repo (data/player_seasons.parquet, which carries VORP
against 1QB and superflex replacement levels) and nflverse players.parquet for birth dates. Rerun once a season
completes:

  uv run --with pandas --with pyarrow python -I tools/make_lookup.py \
      ../career_value/data/player_seasons.parquet ../career_value/raw/players/players.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
from rules import ageband, tier  # noqa: E402

seasons, players = sys.argv[1], sys.argv[2]
hs = pd.read_parquet(seasons)
hs = hs[hs.position.isin(["QB", "RB", "WR", "TE"])]
b = pd.read_parquet(players, columns=["gsis_id", "birth_date"]).dropna().drop_duplicates("gsis_id").set_index("gsis_id").birth_date
LAST = int(hs.season.max())

h = hs[["gsis_id", "season", "position", "games", "ppg"]].copy()
q = h[h.games >= 6].copy()
q["r"] = q.groupby(["season", "position"]).ppg.rank(ascending=False, method="min")
h = h.merge(q[["gsis_id", "season", "r"]], on=["gsis_id", "season"], how="left")
h["age"] = (pd.to_datetime((h.season + 1).astype(str) + "-09-01") - pd.to_datetime(h.gsis_id.map(b))).dt.days / 365.25
h = h[(h.age >= 30) & h.r.notna()].copy()
h["tier"] = [tier(p, r) for p, r in zip(h.position, h.r)]
h["ab"] = h.age.map(ageband)

rows = []
for V, f in (("vorp", "1qb"), ("vorp_sf", "sf")):
    W = hs.pivot_table(index="gsis_id", columns="season", values=V, aggfunc="sum").reindex(
        columns=range(int(hs.season.min()), LAST + 1)).fillna(0)
    z = h.copy()
    for k in (3, 5):
        F = sum(W.shift(-j, axis=1) for j in range(1, k + 1)).stack().rename(f"v{k}")
        Y = (W.shift(-k, axis=1) > 0).astype(float).stack().rename(f"y{k}")
        z = z.join(pd.concat([F, Y], axis=1), on=["gsis_id", "season"])
        z.loc[z.season + k > LAST, [f"v{k}", f"y{k}"]] = np.nan
    for (pos, ab, t), g in z.groupby(["position", "ab", "tier"]):
        rows.append({"f": f, "pos": pos, "ab": ab, "tier": t, "v3": round(g.v3.mean(), 1), "v5": round(g.v5.mean(), 1),
                     "y5": round(g.y5.mean(), 3), "n": int(g.v3.notna().sum())})
out = Path(__file__).resolve().parent.parent / "build" / "old_lookup.json"
out.write_text(json.dumps({"seasons": f"{int(hs.season.min())}-{LAST}", "rows": rows}, indent=1))
print(f"{len(rows)} cells -> {out}")
