# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""Build a player-snapshot panel for rest-of-career (ROC) value modelling.

One row = (player, snapshot season t): the offseason after season t, before
season t+1. Features use only information available at that point; targets
use seasons t+1 onward.

Rows:
  * veteran rows  - player recorded >=1 regular-season game in season t
  * rookie rows   - every drafted QB/RB/WR/TE, snapshot t = draft_year - 1
                    (no NFL production features; avoids selecting on outcome)

Value unit: season VORP = max(0, PPR points - replacement), replacement being
the QB13 / RB31 / WR43 / TE13 season total that year (12-team, 1QB, 2RB, 3WR,
1TE, 1 FLEX). Bad seasons floor at 0: a dynasty roster can bench them.
Superflex value (vorp_sf): same lineup plus 1 SUPERFLEX (QB/RB/WR/TE). Replacement
is set per season by filling 12 teams' lineups greedily from season totals.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "raw"
OUT = HERE / "data"
POS = ["QB", "RB", "WR", "TE"]
REPL_RANK = {"QB": 13, "RB": 31, "WR": 43, "TE": 13}
LAST_COMPLETE = 2025  # 2026 season is in progress; not used


def load_stats() -> pd.DataFrame:
    files = sorted((RAW / "stats_player_reg").glob("stats_player_reg_*.parquet"))
    s = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    s = s[s.season <= LAST_COMPLETE]
    keep = [
        "player_id", "season", "recent_team", "games", "fantasy_points_ppr",
        "attempts", "passing_yards", "passing_tds", "passing_epa",
        "carries", "rushing_yards", "rushing_tds", "rushing_epa",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_air_yards", "receiving_epa",
        "target_share", "air_yards_share", "wopr",
    ]
    return s[keep].rename(columns={"player_id": "gsis_id", "fantasy_points_ppr": "ppr"})


def load_players() -> pd.DataFrame:
    p = pd.read_parquet(RAW / "players" / "players.parquet")
    p = p[p.position.isin(POS)].copy()
    p["birth_date"] = pd.to_datetime(p.birth_date, errors="coerce")
    for c in ["draft_year", "draft_round", "draft_pick", "rookie_season", "height", "weight"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    # some undrafted players carry a rookie_season but no draft_year
    p["drafted"] = p.draft_pick.notna().astype(int)
    p["entry_season"] = p.rookie_season.fillna(p.draft_year)
    return p[["gsis_id", "display_name", "position", "birth_date", "height", "weight",
              "draft_year", "draft_round", "draft_pick", "drafted", "entry_season"]]


def add_vorp(season_rows: pd.DataFrame) -> pd.DataFrame:
    d = season_rows.copy()
    d["pos_rank"] = d.groupby(["season", "position"]).ppr.rank(ascending=False, method="first")
    repl = (
        d.assign(r=d.position.map(REPL_RANK))
        .query("pos_rank == r")[["season", "position", "ppr"]]
        .rename(columns={"ppr": "repl_pts"})
    )
    d = d.merge(repl, on=["season", "position"], how="left")
    d["vorp"] = (d.ppr - d.repl_pts).clip(lower=0)
    return d


SF_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}


def add_vorp_sf(d: pd.DataFrame, teams=12) -> pd.DataFrame:
    """Greedy superflex lineup fill per season: position slots, then FLEX (RB/WR/TE),
    then SUPERFLEX (any). Replacement = best unstarted player at each position."""
    d = d.copy()
    d["repl_sf"] = np.nan
    for season, g in d.groupby("season"):
        g = g.sort_values("ppr", ascending=False)
        started = set()
        for pos, k in SF_SLOTS.items():
            started |= set(g[g.position == pos].index[: k * teams])
        rest = g[~g.index.isin(started)]
        started |= set(rest[rest.position != "QB"].index[:teams])          # FLEX
        rest = g[~g.index.isin(started)]
        started |= set(rest.index[:teams])                                  # SUPERFLEX
        bench = g[~g.index.isin(started)]
        repl = bench.groupby("position").ppr.max()
        d.loc[g.index, "repl_sf"] = g.position.map(repl).to_numpy()
    d["vorp_sf"] = (d.ppr - d.repl_sf).clip(lower=0)
    return d


def main() -> None:
    OUT.mkdir(exist_ok=True)
    players = load_players()
    stats = load_stats().merge(players[["gsis_id", "position"]], on="gsis_id", how="inner")
    seasons = add_vorp_sf(add_vorp(stats))
    seasons = seasons[seasons.games > 0]

    g = seasons["games"]
    seasons = seasons.assign(
        ppg=seasons.ppr / g,
        att_pg=seasons.attempts / g,
        pass_yds_pg=seasons.passing_yards / g,
        carries_pg=seasons.carries / g,
        rush_yds_pg=seasons.rushing_yards / g,
        targets_pg=seasons.targets / g,
        rec_yds_pg=seasons.receiving_yards / g,
        epa_pg=(seasons.passing_epa.fillna(0) + seasons.rushing_epa.fillna(0)
                + seasons.receiving_epa.fillna(0)) / g,
    )

    # ---- career-to-date features (seasons <= t) -------------------------
    seasons = seasons.sort_values(["gsis_id", "season"])
    grp = seasons.groupby("gsis_id")
    seasons["career_vorp"] = grp.vorp.cumsum()
    seasons["career_best_vorp"] = grp.vorp.cummax()
    seasons["career_best_ppg"] = grp.ppg.cummax()
    seasons["seasons_played"] = grp.cumcount() + 1
    seasons["team_prev"] = grp.recent_team.shift(1)
    seasons["changed_team"] = (seasons.team_prev.notna()
                               & (seasons.team_prev != seasons.recent_team)).astype(int)

    # prior-season (t-1) values; 0 if the player did not play in t-1
    prev = seasons[["gsis_id", "season", "vorp", "ppg", "games"]].copy()
    prev["season"] += 1
    seasons = seasons.merge(prev.rename(columns={"vorp": "vorp_prev", "ppg": "ppg_prev",
                                                 "games": "games_prev"}),
                            on=["gsis_id", "season"], how="left")
    seasons[["vorp_prev", "games_prev"]] = seasons[["vorp_prev", "games_prev"]].fillna(0)

    # ---- targets: future VORP --------------------------------------------
    fut = seasons[["gsis_id", "season", "vorp"]]
    fut_by_player = {k: v.set_index("season").vorp for k, v in fut.groupby("gsis_id")}

    def future_sums(pid, t):
        v = fut_by_player.get(pid)
        if v is None:
            return 0.0, 0.0, 0.0, 0.0
        f = v[v.index > t]
        k = f.index - t
        return (f.sum(), f[k <= 5].sum(), (f * 0.85 ** (k - 1)).sum(), f[k == 1].sum())

    vet = seasons.rename(columns={"season": "t"}).copy()
    vet["row_type"] = "veteran"

    rookies = players[players.draft_year.notna()].copy()
    rookies["t"] = rookies.draft_year - 1
    rookies["row_type"] = "rookie"
    rookies = rookies[["gsis_id", "t", "row_type"]]

    panel = pd.concat([vet, rookies], ignore_index=True)
    panel = panel.drop(columns=["position"], errors="ignore").merge(players, on="gsis_id", how="left")
    panel = panel[(panel.t >= 1999) & (panel.t <= LAST_COMPLETE)]

    sums = np.array([future_sums(p, t) for p, t in zip(panel.gsis_id, panel.t)])
    panel["roc_vorp"], panel["roc5_vorp"], panel["roc_vorp_disc"], panel["next_vorp"] = sums.T
    panel["followup_seasons"] = LAST_COMPLETE - panel.t

    # ---- snapshot-time bio features -------------------------------------
    snap = pd.to_datetime((panel.t + 1).astype(int).astype(str) + "-09-01")
    panel["age"] = (snap - panel.birth_date).dt.days / 365.25
    panel["exp"] = np.where(panel.row_type == "rookie", 0, panel.t - panel.entry_season + 1)
    panel["draft_pick_f"] = panel.draft_pick.fillna(300)  # undrafted ~ after pick 260
    panel["draft_round_f"] = panel.draft_round.fillna(8)
    panel["bmi"] = 703 * panel.weight / panel.height ** 2

    cols = [
        "gsis_id", "display_name", "position", "t", "row_type", "recent_team",
        "age", "exp", "draft_pick_f", "draft_round_f", "drafted", "height", "weight", "bmi",
        "games", "ppr", "ppg", "vorp", "vorp_sf", "pos_rank", "games_prev", "ppg_prev", "vorp_prev",
        "career_vorp", "career_best_vorp", "career_best_ppg", "seasons_played", "changed_team",
        "att_pg", "pass_yds_pg", "carries_pg", "rush_yds_pg", "targets_pg", "rec_yds_pg",
        "target_share", "air_yards_share", "wopr", "epa_pg",
        "roc_vorp", "roc5_vorp", "roc_vorp_disc", "next_vorp", "followup_seasons",
    ]
    panel = panel[cols].sort_values(["t", "position", "gsis_id"]).reset_index(drop=True)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    seasons.to_parquet(OUT / "player_seasons.parquet", index=False)

    rk = seasons.assign(r=seasons.groupby(["season", "position"]).ppr.rank(ascending=False, method="first"))
    rr = rk[(rk.ppr >= rk.repl_sf)].groupby(["season", "position"]).r.max()
    print("superflex replacement rank (median over seasons):",
          (rr.groupby(level=1).median() + 1).to_dict())
    print(panel.groupby("row_type").size())
    print(panel.groupby("t").size().to_string())
    print(panel[["age", "exp", "roc_vorp", "roc5_vorp"]].describe().round(2))


if __name__ == "__main__":
    main()
