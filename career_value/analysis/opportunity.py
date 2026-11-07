"""Q: landing spot and opportunity -- does the market price vacated targets/carries, team changes, a QB1 leaving?

Per team T and season t (player_seasons; skill positions):
  vacated  : share of T's season-t targets (WR/TE) or carries (RB) by players whose season-(t+1) team is not T
             (moved or did not play). Lookahead caveat: departures are judged by who actually played for T in t+1;
             at a May snapshot nearly all of these are known (FA, trades, cuts, retirements), injuries are the leak.
  incoming : season-t targets/carries that arriving veterans had elsewhere, as a share of T's season-t total.
  net      : vacated - incoming.
  qb1_left : T's most-attempts QB in season t is not on T in season t+1.
  rook_comp: R1-R2 NFL rookies T drafts at the same position (WR+TE for targets, RB for carries) in draft t+1
             (for a rookie row: other than himself).
The player's team for t+1 = his season-(t+1) team (else his season-t team); rookies = draft team.
Residual % from market_rows (veterans) and rookie_position_residuals.csv (rookies).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import DYN, HERE, out_dir, rows, table


def team_context():
    s = pd.read_parquet(HERE / "data" / "player_seasons.parquet")
    pl = pd.read_parquet(HERE / "raw" / "players" / "players.parquet",
                         columns=["gsis_id", "draft_year", "draft_round", "draft_team", "position"])
    nxt = s[["gsis_id", "season", "recent_team"]].assign(season=lambda d: d.season - 1).rename(
        columns={"recent_team": "team_next"})
    s = s.merge(nxt, on=["gsis_id", "season"], how="left")
    s["left"] = s.team_next != s.recent_team  # NaN team_next (did not play) counts as left
    out = []
    for vol, pos in (("targets", ["WR", "TE", "RB"]), ("carries", ["RB", "QB", "WR"])):
        tot = s.groupby(["recent_team", "season"])[vol].sum()
        vac = s[s.left].groupby(["recent_team", "season"])[vol].sum().reindex(tot.index, fill_value=0)
        arr = s[s.team_next.notna() & s.left].groupby(["team_next", "season"])[vol].sum()
        arr.index.names = ["recent_team", "season"]
        inc = arr.reindex(tot.index, fill_value=0)
        out.append(pd.DataFrame({f"vac_{vol}": vac / tot * 100, f"inc_{vol}": inc / tot * 100}))
    c = pd.concat(out, axis=1)
    qb = s[s.position == "QB"].sort_values("attempts").drop_duplicates(["recent_team", "season"], keep="last")
    c["qb1_left"] = qb.set_index(["recent_team", "season"]).left.reindex(c.index).fillna(False).astype(int)
    rk = pl[pl.draft_round.isin([1, 2])].assign(season=lambda d: d.draft_year - 1)
    for k, ps in (("rook_wrte", ["WR", "TE"]), ("rook_rb", ["RB"])):
        n = rk[rk.position.isin(ps)].groupby(["draft_team", "season"]).size()
        n.index.names = ["recent_team", "season"]
        c[k] = n.reindex(c.index, fill_value=0)
    c = c.reset_index().rename(columns={"recent_team": "team", "season": "t"})
    # player's t+1 team (vets), draft team (rookies)
    pteam = s[["gsis_id", "season", "recent_team", "team_next"]].rename(columns={"season": "t"})
    pteam["team"] = pteam.team_next.fillna(pteam.recent_team)
    pteam["moved"] = (pteam.team_next.notna() & (pteam.team_next != pteam.recent_team)).astype(int)
    return c, pteam[["gsis_id", "t", "team", "moved"]], pl[["gsis_id", "draft_team"]]


def add_opp(x):
    tgt = x.position.isin(["WR", "TE"])
    x["vac"] = np.where(tgt, x.vac_targets, x.vac_carries)
    x["inc"] = np.where(tgt, x.inc_targets, x.inc_carries)
    x["net"] = x.vac - x.inc
    x["comp"] = np.where(tgt, x.rook_wrte, x.rook_rb)
    x["vacb"] = pd.cut(x.vac, [-1, 10, 25, 1000], labels=["vac <10%", "vac 10-25%", "vac 25%+"]).astype(str)
    x["netb"] = pd.cut(x.net, [-1000, 0, 15, 1000], labels=["net <=0", "net 0-15", "net 15%+"]).astype(str)
    return x


def ols(x, label, ycol="r"):
    for pos in ["WR", "RB", "TE", "QB"]:
        d = x[x.position == pos].dropna(subset=["vac", "inc", ycol])
        if len(d) < 60:
            continue
        f = "vac + inc + qb1_left + comp" + (" + moved" if "moved" in d and d.moved.nunique() > 1 else "")
        if pos == "QB":
            f = "inc + comp + moved" if "moved" in d and d.moved.nunique() > 1 else "inc"
            d = d.assign(inc=d.inc_targets)  # receiving talent added around a QB
        m = smf.ols(f"{ycol} ~ {f} + mkt_pos_rank", d).fit(cov_type="cluster", cov_kwds={"groups": d.gsis_id})
        cells = [f"{k} {m.params[k]:+.2f} (p={m.pvalues[k]:.2f})" for k in m.params.index if k not in ("Intercept", "mkt_pos_rank")]
        print(f"   OLS {label} {pos} n={len(d)}: " + "; ".join(cells))


def main():
    ctx, pteam, dteam = team_context()
    print("team-season vacated targets share: median {:.0f}%, p90 {:.0f}%; vacated carries median {:.0f}%, p90 {:.0f}%; QB1 left {:.0f}%".format(
        ctx.vac_targets.median(), ctx.vac_targets.quantile(.9), ctx.vac_carries.median(), ctx.vac_carries.quantile(.9),
        ctx.qb1_left.mean() * 100))
    masks = {}
    for pos in ["WR", "TE", "RB"]:
        for b in ["vac <10%", "vac 10-25%", "vac 25%+"]:
            masks[f"{pos} {b}"] = (lambda pos, b: lambda d: (d.position == pos) & (d.vacb == b))(pos, b)
        for b in ["net <=0", "net 0-15", "net 15%+"]:
            masks[f"{pos} {b}"] = (lambda pos, b: lambda d: (d.position == pos) & (d.netb == b))(pos, b)
        masks[f"{pos} QB1 left"] = (lambda pos: lambda d: (d.position == pos) & (d.qb1_left == 1))(pos)
        masks[f"{pos} R1-2 rookie added at pos"] = (lambda pos: lambda d: (d.position == pos) & (d.comp > 0))(pos)
        masks[f"{pos} moved teams"] = (lambda pos: lambda d: (d.position == pos) & (d.moved == 1))(pos)
    masks["QB moved teams"] = lambda d: (d.position == "QB") & (d.moved == 1)
    masks["QB, team added 25%+ targets"] = lambda d: (d.position == "QB") & (d.inc_targets >= 25)

    for fam, mks, hs in (("Dynasty pooled 2019-24", DYN, (1, 2)), ("MFL keeper ADP 2010-24", ["mfl_dynasty"], (1, 3))):
        fr = {}
        for h in hs:
            x = rows(mks, h)
            x = x[x.row_type == "veteran"].merge(pteam, on=["gsis_id", "t"], how="left")
            x = add_opp(x.merge(ctx, on=["team", "t"], how="left"))
            fr[f"h{h}"] = x
            ols(x, f"{fam} vets h{h}")
        table(fr, masks, f"{fam} veterans: residual % by next-season opportunity")

    r = pd.read_csv(out_dir() / "rookie_position_residuals.csv").merge(dteam, on="gsis_id", how="left")
    r = add_opp(r.rename(columns={"draft_team": "team"}).merge(ctx, on=["team", "t"], how="left"))
    r["r"], r["moved"], r["mkt_pos_rank"] = r.res, 0, r.slot
    r["comp"] = r.comp - (r.draft_round_f.isin([1, 2]) & r.position.isin(["WR", "TE", "RB"])).astype(int)  # not himself
    rm = {k: v for k, v in masks.items() if "moved" not in k and "QB," not in k}
    rm["QB rookie, QB1 left"] = lambda d: (d.position == "QB") & (d.qb1_left == 1)
    rm["QB rookie, QB1 stayed"] = lambda d: (d.position == "QB") & (d.qb1_left == 0)
    for fam, mk, hs in (("MFL rookie order 2011-25", ["mfl"], (3, 5)), ("Dynasty rookie order 2020-25", DYN, (1, 2))):
        fr = {f"h{h}": r[r.market.isin(mk) & (r.h == h)] for h in hs}
        table(fr, rm, f"{fam} rookies: residual % by landing-spot opportunity (draft team)")
        for h in hs:
            ols(fr[f"h{h}"].assign(slot=fr[f"h{h}"].slot), f"{fam} rookies h{h}")


if __name__ == "__main__":
    main()
