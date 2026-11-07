# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Build data/players.js: a buy / hold / sell call for every Sleeper skill player, for rebuilders and contenders,
in 1QB and superflex (rules in rules.py).

  python -I build/build_data.py RAW_DIR OUT_FILE

RAW_DIR holds the downloaded inputs (build/fetch.sh):
  values-players.csv                  DynastyProcess FP dynasty ECR, 1QB and superflex
  db_playerids.csv                    DynastyProcess id map (FantasyPros / gsis -> Sleeper), names, draft info
  players.parquet                     nflverse bios: birth dates, NFL draft round
  stats_player_reg_<season>.parquet   nflverse regular-season stats, current and last season
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rules import classify  # noqa: E402

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
POS = ["QB", "RB", "WR", "TE"]

vals = pd.read_csv(RAW / "values-players.csv")
SCRAPE = pd.Timestamp(vals.scrape_date.max())
TODAY = pd.Timestamp.today().normalize()
CUR = TODAY.year if TODAY.month >= 9 else TODAY.year - 1            # season in progress, or the one just finished
NEXT = CUR + 1                                                        # the season the calls look ahead to
ROOKIE = TODAY.year if TODAY.month >= 5 else TODAY.year - 1          # latest rookie class (post-draft)
AGE_AT = pd.Timestamp(f"{NEXT}-09-01")                                # age entering next season, as in the backtests

look = {}
for r in json.loads((HERE / "old_lookup.json").read_text())["rows"]:
    look[(r["f"], r["pos"], r["ab"], r["tier"])] = (r["v3"], r["v5"], r["y5"], r["n"])
ctx = {"ROOKIE": ROOKIE, "LOOK": look}

# ---------- players ----------
ids = pd.read_csv(RAW / "db_playerids.csv", dtype={"sleeper_id": str, "gsis_id": str})
ids = ids[ids.sleeper_id.notna() & ids.position.isin(POS)].drop_duplicates("sleeper_id")
ids["fantasypros_id"] = pd.to_numeric(ids.fantasypros_id, errors="coerce")
bio = pd.read_parquet(RAW / "players.parquet", columns=["gsis_id", "birth_date", "draft_round", "draft_year"])
bio = bio.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
p = ids[["sleeper_id", "gsis_id", "fantasypros_id", "name", "position", "team", "birthdate", "draft_year", "draft_round"]].copy()
p["birth"] = pd.to_datetime(p.gsis_id.map(bio.birth_date)).fillna(pd.to_datetime(p.birthdate, errors="coerce"))
p["dy"] = p.gsis_id.map(bio.draft_year).fillna(p.draft_year)
p["rd"] = p.gsis_id.map(bio.draft_round).fillna(p.draft_round).fillna(8)    # 8 = undrafted

v = vals[vals.pos.isin(POS)].copy()
for f, col in (("1qb", "ecr_1qb"), ("sf", "ecr_2qb")):
    v[f"r_{f}"] = v[col].rank(method="first")
    v[f"pr_{f}"] = v.groupby("pos")[col].rank(method="first")
v = v.rename(columns={"player": "vname", "team": "vteam", "pos": "vpos", "draft_year": "vdy"})
p = p.merge(v[["fp_id", "vname", "vteam", "vpos", "vdy", "r_1qb", "r_sf", "pr_1qb", "pr_sf"]],
            left_on="fantasypros_id", right_on="fp_id", how="left")
p["name"] = p.vname.fillna(p.name)
p["team"] = p.vteam.fillna(p.team)
p["position"] = p.vpos.fillna(p.position)
p["dy"] = p.vdy.fillna(p.dy)                                          # the values file knows the current rookie class
p["age"] = (AGE_AT - p.birth).dt.days / 365.25


# ---------- production: current season (4+ games) and last season (6+ games), PPR per game ----------
def ppg(season, min_g):
    f = RAW / f"stats_player_reg_{season}.parquet"
    if not f.exists():
        print(f"note: {f.name} missing, no production data for {season}")
        return pd.DataFrame(columns=["games", "ppg", "prk"], dtype=float)
    s = pd.read_parquet(f, columns=["player_id", "position", "games", "fantasy_points_ppr"])
    s = s[s.position.isin(POS) & s.games.between(1, 17)].drop_duplicates("player_id").copy()
    s["ppg"] = s.fantasy_points_ppr / s.games
    q = s[s.games >= min_g].copy()
    q["prk"] = q.groupby("position").ppg.rank(ascending=False, method="min")
    return s.merge(q[["player_id", "prk"]], on="player_id", how="left").set_index("player_id")


cs, ls = ppg(CUR, 4), ppg(CUR - 1, 6)
p["g"] = p.gsis_id.map(cs.games)
p["ppg"] = p.gsis_id.map(cs.ppg)
p["prk"] = p.gsis_id.map(cs.prk)
p["prk_last"] = p.gsis_id.map(ls.prk)
p["prod_rank"] = p[["prk", "prk_last"]].mean(axis=1)                  # blend: half a season alone is noisy

# ---------- output ----------
p = p[p.r_1qb.notna() | p.g.notna() | (p.dy >= ROOKIE - 3)].copy()   # valued, playing this season, or recent draftees
rules, rix, out = [], {}, {}


def num(z, d=1):
    return None if pd.isna(z) else round(float(z), d)


for x in p.itertuples(index=False):
    c = {}
    for f in ("1qb", "sf"):
        g, (ra, rw), (ca, cw) = classify(x, f, ctx)
        key = (g, ra, rw, ca, cw)
        if key not in rix:
            rix[key] = len(rules)
            rules.append(list(key))
        c[f] = rix[key]
    out[x.sleeper_id] = {"n": x.name, "p": x.position, "t": None if pd.isna(x.team) else x.team, "a": num(x.age),
                         "dy": None if pd.isna(x.dy) else int(x.dy), "rd": None if x.rd >= 8 else int(x.rd),
                         "r1": num(x.r_1qb, 0), "r2": num(x.r_sf, 0), "pr1": num(x.pr_1qb, 0), "pr2": num(x.pr_sf, 0),
                         "g": None if pd.isna(x.g) else int(x.g), "ppg": num(x.ppg), "prk": num(x.prk, 0),
                         "prkl": num(x.prk_last, 0), "c1": c["1qb"], "c2": c["sf"]}
# names only, for rostered players without a call (kickers, IDP, deep reserves), so the page can still list them
allids = pd.read_csv(RAW / "db_playerids.csv", dtype={"sleeper_id": str}, usecols=["sleeper_id", "name", "position", "team", "db_season"])
allids = allids[allids.sleeper_id.notna() & ~allids.sleeper_id.isin(out) & (allids.db_season >= CUR - 2)].drop_duplicates("sleeper_id")
names = {r.sleeper_id: [r.name, r.position, None if pd.isna(r.team) else r.team] for r in allids.itertuples(index=False)}
meta = {"built": str(TODAY.date()), "values_date": str(SCRAPE.date()), "season": CUR, "next": NEXT,
        "rookie": ROOKIE, "games": int(cs.games.max()) if len(cs) else 0, "last": CUR - 1}
OUT.parent.mkdir(parents=True, exist_ok=True)
js = "window.DATA = " + json.dumps({"meta": meta, "rules": rules, "players": out, "names": names}, separators=(",", ":")) + ";\n"
OUT.write_text(js)
print(f"{len(out)} players, {len(rules)} distinct calls, {len(js) / 1e6:.2f} MB; values {SCRAPE.date()}, season {CUR}")
