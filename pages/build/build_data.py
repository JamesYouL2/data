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
  fc_1qb.json, fc_sf.json             FantasyCalc dynasty trade-market values (with rest-of-season redraft values)
  sleeper_players.json                Sleeper player database: injury status, current team
  sleeper_state.json                  Sleeper's current week
  db_fpecr_latest.csv                 DynastyProcess FP rankings scrape: rest-of-season (PPR) ranks in season
  db_fpecr.parquet                    DynastyProcess FP rankings history: this season's preseason ranks (rules.signals)
  fp_latest_weekly.csv                DynastyProcess FP weekly ranks: opponent, projected PPR points
The market, injury, ROS and weekly files are optional: the build runs without them.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rules import classify, ros_tilt, situational  # noqa: E402

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
ctx["EARLY"] = bool(p.prk.isna().all())                              # before week 4: last season stands in for "producing now"

# ---------- trade market (FantasyCalc) and injuries (Sleeper) ----------
def load_json(name):
    f = RAW / name
    if not f.exists():
        print(f"note: {name} missing, skipping")
        return None
    return json.loads(f.read_text())


PICK_RE = re.compile(r"^(\d{4}) (1st|2nd|3rd|4th) \((Early|Mid|Late)\)$")
market, picks = {}, {}
for f in ("1qb", "sf"):
    fc = load_json(f"fc_{f}.json") or []
    pl = sorted([z for z in fc if z["player"]["position"] in POS and z["player"].get("sleeperId")], key=lambda z: -z["value"])
    ros = sorted([z for z in pl if z.get("redraftValue", 0) > 0], key=lambda z: -z["redraftValue"])
    rr = {z["player"]["sleeperId"]: i + 1 for i, z in enumerate(ros)}
    for i, z in enumerate(pl):
        sid = z["player"]["sleeperId"]
        v = z["value"]
        market.setdefault(sid, {})[f] = (i + 1, v, z.get("redraftValue", 0), rr.get(sid), z.get("trend30Day", 0))
    pk = [(PICK_RE.match(z["player"]["name"]), z["value"]) for z in fc if z["player"]["position"] == "PICK"]
    pk = [(mm, v) for mm, v in pk if mm]
    if pk:
        yr = min(mm.group(1) for mm, _ in pk)
        picks[f] = sorted([[f"{mm.group(3).lower()} {mm.group(2)}", v] for mm, v in pk if mm.group(1) == yr], key=lambda z: -z[1])
        picks[f + "_year"] = int(yr)
for f in ("1qb", "sf"):
    for k, i in (("mr", 0), ("mv", 1), ("rv", 2), ("rr", 3), ("mt", 4)):
        p[f"{k}_{f}"] = p.sleeper_id.map(lambda s, f=f, i=i: market.get(s, {}).get(f, (None,) * 5)[i]).astype(float)

sp = load_json("sleeper_players.json") or {}
p["inj"] = p.sleeper_id.map(lambda s: (sp.get(s) or {}).get("injury_status"))
p["inj_part"] = p.sleeper_id.map(lambda s: (sp.get(s) or {}).get("injury_body_part"))
steam = p.sleeper_id.map(lambda s: (sp.get(s) or {}).get("team"))
p["team"] = steam.where(steam.notna() | p.sleeper_id.isin(sp.keys()), p.team)   # Sleeper is current; no team = free agent

state = load_json("sleeper_state.json") or {}
WEEK = state.get("week") if state.get("season_type") == "regular" else None
ctx["WEEK"] = WEEK


# FantasyPros ranks at the last scrape before week 1 and at the latest one, from one list so the two compare (rules.signals)
def fp_moves():
    for key in ("do", "dsf", "ro"):
        p[f"{key}_pre"] = p[f"{key}_now"] = float("nan")
    f, start = RAW / "db_fpecr.parquet", state.get("season_start_date")
    if not f.exists() or not start or not WEEK:
        print("note: no FP rank history or season start, skipping the signals that need a preseason rank")
        return
    e = pd.read_parquet(f, columns=["page_type", "ecr_type", "scrape_date", "id", "pos", "ecr"],
                        filters=[("scrape_date", ">=", f"{CUR}-08-15"), ("ecr_type", "in", ["do", "dsf", "ro"])])
    e = e[e.pos.isin(POS)].assign(id=lambda d: pd.to_numeric(d.id, errors="coerce"))
    for key, pages in (("do", ["dynasty-overall", "dynasty-offense"]), ("dsf", ["dynasty-op"]),
                       ("ro", ["redraft-overall", "redraft-offense"])):
        x = e[(e.ecr_type == key) & e.page_type.isin(pages)]
        dates = x.scrape_date[x.scrape_date < start]
        if dates.empty:
            continue
        for tag, d in (("pre", dates.max()), ("now", x.scrape_date.max())):
            y = x[x.scrape_date == d].dropna(subset=["ecr"]).sort_values("page_type").drop_duplicates("id")
            p[f"{key}_{tag}"] = p.fantasypros_id.map(y.set_index("id").ecr.rank(method="first"))


fp_moves()


# FantasyPros rest-of-season (in season, the redraft pages are ROS: PPR, 1QB) and this week's ranks, via DynastyProcess
def fp_csv(name):
    f = RAW / name
    if not f.exists():
        print(f"note: {name} missing, skipping")
        return None
    return pd.read_csv(f)


ecr = fp_csv("db_fpecr_latest.csv")
ros = ecr[(ecr.page_type == "redraft-overall") & ecr.pos.isin(POS)].copy() if ecr is not None else pd.DataFrame()
ROS_ON = len(ros) > 100 and ros.fp_page.str.contains("ros-").any()   # preseason the page is a draft board, not ROS
if ROS_ON:
    ros["ros_r"] = ros.ecr.rank(method="first")
    ros["ros_p"] = ros.groupby("pos").ecr.rank(method="first")
    ros = ros.drop_duplicates("id").set_index("id")
    p["ros_r"] = p.fantasypros_id.map(ros.ros_r)
    p["ros_p"] = p.fantasypros_id.map(ros.ros_p)
else:
    p["ros_r"] = p["ros_p"] = float("nan")
wk = fp_csv("fp_latest_weekly.csv")
WK_POS = {"qb": "QB", "ppr-rb": "RB", "ppr-wr": "WR", "ppr-te": "TE"}
if wk is not None and WEEK:
    wk = wk[wk.page.isin(WK_POS.keys())].drop_duplicates("fantasypros_id").set_index("fantasypros_id")
    p["wk_p"] = p.fantasypros_id.map(wk["rank"])
    p["wk_opp"] = p.fantasypros_id.map(wk.player_opponent)
    p["wk_pts"] = pd.to_numeric(p.fantasypros_id.map(wk.r2p_pts), errors="coerce")
else:
    p["wk_p"] = p["wk_pts"] = float("nan")
    p["wk_opp"] = None
# ---------- output ----------
p = p[p.r_1qb.notna() | p.g.notna() | (p.dy >= ROOKIE - 3) | p.mr_sf.notna()].copy()   # valued, playing, recent draftees
if ROS_ON:
    p["gone"] = p.ros_r.isna()          # only matters for injured players: see rules.situational
# in season the calls price players at the FP dynasty rank tilted toward the ROS rank (rules.ros_tilt); the page still
# shows the FP rank, with the tilted one beside it
for f in ("1qb", "sf"):
    k = [ros_tilt(a, b) if ROS_ON and a == a else 1.0 for a, b in zip(p[f"pr_{f}"], p.ros_p)]
    p[f"v_{f}"], p[f"pv_{f}"] = p[f"r_{f}"] * k, p[f"pr_{f}"] * k
rules, rix, out = [], {}, {}


def num(z, d=1):
    return None if z is None or pd.isna(z) else round(float(z), d)


for x in p.itertuples(index=False):
    c = {}
    for f in ("1qb", "sf"):
        xv = x._replace(**{f"r_{f}": getattr(x, f"v_{f}"), f"pr_{f}": getattr(x, f"pv_{f}")})
        key = tuple(situational(x, f, classify(xv, f, ctx), ctx))
        if key not in rix:
            rix[key] = len(rules)
            rules.append(list(key))
        c[f] = rix[key]
    o = {"n": x.name, "p": x.position, "t": None if pd.isna(x.team) else x.team, "a": num(x.age),
         "dy": None if pd.isna(x.dy) else int(x.dy), "rd": None if x.rd >= 8 else int(x.rd),
         "r1": num(x.r_1qb, 0), "r2": num(x.r_sf, 0), "pr1": num(x.pr_1qb, 0), "pr2": num(x.pr_sf, 0),
         "g": None if pd.isna(x.g) else int(x.g), "ppg": num(x.ppg), "prk": num(x.prk, 0),
         "prkl": num(x.prk_last, 0), "c1": c["1qb"], "c2": c["sf"]}
    for f, k in (("1qb", "m1"), ("sf", "m2")):
        if not pd.isna(getattr(x, f"mr_{f}")):
            o[k] = [num(getattr(x, f"{q}_{f}"), 0) for q in ("mr", "mv", "rv", "rr", "mt")]
    if isinstance(x.inj, str):
        o["inj"] = x.inj
        if isinstance(x.inj_part, str):
            o["ib"] = x.inj_part
    if not pd.isna(x.ros_r):
        o["ros"] = [num(x.ros_r, 0), num(x.ros_p, 0)]
    if ROS_ON:
        for f, k in (("1qb", "v1"), ("sf", "v2")):
            if not pd.isna(getattr(x, f"v_{f}")):
                o[k] = num(getattr(x, f"v_{f}"), 0)
    if not pd.isna(x.wk_p):
        o["wk"] = [num(x.wk_p, 0), x.wk_opp if isinstance(x.wk_opp, str) else None, num(x.wk_pts)]
    out[x.sleeper_id] = o

# names only, for rostered players without a call (kickers, IDP, deep reserves, free agents), so the page can list them
allids = pd.read_csv(RAW / "db_playerids.csv", dtype={"sleeper_id": str}, usecols=["sleeper_id", "name", "position", "team", "db_season"])
allids = allids[allids.sleeper_id.notna() & ~allids.sleeper_id.isin(out) & (allids.db_season >= CUR - 2)].drop_duplicates("sleeper_id")
names = {r.sleeper_id: [r.name, r.position, None if pd.isna(r.team) else r.team] for r in allids.itertuples(index=False)}
for s_, z in sp.items():          # Sleeper is current: its name and team win, and it covers this year's signings
    if s_ not in out and (z.get("team") or s_ in names):
        names[s_] = [z.get("full_name") or f"{z.get('first_name', '')} {z.get('last_name', '')}".strip(), z.get("position"), z.get("team")]
meta = {"built": str(TODAY.date()), "values_date": str(SCRAPE.date()), "season": CUR, "next": NEXT,
        "rookie": ROOKIE, "games": int(cs.games.max()) if len(cs) else 0, "last": CUR - 1,
        "market": bool(market), "picks": picks, "week": WEEK, "ros": bool(ROS_ON),
        "fp_date": None if ecr is None else str(ecr.scrape_date.max())}
OUT.parent.mkdir(parents=True, exist_ok=True)
js = "window.DATA = " + json.dumps({"meta": meta, "rules": rules, "players": out, "names": names}, separators=(",", ":")) + ";\n"
OUT.write_text(js)
print(f"{len(out)} players, {len(rules)} distinct calls, {len(js) / 1e6:.2f} MB; values {SCRAPE.date()}, season {CUR}; "
      f"market {sum(1 for o in out.values() if 'm2' in o)}, injured {sum(1 for o in out.values() if 'inj' in o)}, "
      f"ROS {sum(1 for o in out.values() if 'ros' in o)}, week {WEEK} {sum(1 for o in out.values() if 'wk' in o)}")
