"""Q: when a player's redraft rank, dynasty rank and last season's production disagree, which one is right?

Rows: outputs/[sf/]market_rows (market_residuals.py), the top 200 of each dynasty market (FP, KTC, FantasyCalc, DynastyProcess) in
May after season t. r = the player's VORP over the next h seasons minus what his dynasty price predicts, as a % of the mean.
Each veteran gets two gaps against his dynasty positional rank d:
  redraft     his positional rank in that summer's redraft market (FFC ADP; FP redraft ECR with REDRAFT=fp): "up" if it is a
              quarter or more better than d and at least 3 places, "down" if a quarter worse, else "level"; off the list = down
  production  his points-per-game positional rank in season t, the same way; no games = down
Age classes as the site: 25-, 26-29, 30+ (rookies have no production, so only the redraft gap).
Output per cell: mean r / median r, share beating their price, cohorts with a positive mean / cohorts, player-seasons.
Then r on both gaps as log ratios (per 1 SD, both in the model, cohort-bootstrap 95% CI), by age class.
Caveat: redraft ranks are late August and dynasty prices mostly May, so the redraft gap knows the summer's injuries and depth
charts. LEAGUE=sf for superflex. Prints only.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E402
from common import DYN, out_dir, rows  # noqa: E402

SF = os.environ.get("LEAGUE") == "sf"
REDRAFT = os.environ.get("REDRAFT", "ffc") + "_redraft"
LV = ["up", "level", "down"]


def gap(d, other):
    """up / level / down: other rank against the dynasty rank d (missing = down)"""
    ratio = (other / d).fillna(9)
    far = (other - d).abs().fillna(99) >= 3
    return np.where((ratio <= 0.75) & far, "up", np.where((ratio >= 4 / 3) & far, "down", "level"))


def load(h):
    x = rows(DYN, h, SF)
    rd = pd.read_parquet(out_dir(SF) / "market_rows" / f"{REDRAFT}_h1.parquet")[["gsis_id", "t", "mkt_pos_rank"]]
    x = x.merge(rd.rename(columns={"mkt_pos_rank": "rd_rank"}), on=["gsis_id", "t"], how="left")
    return features(x[x.t.isin(rd.t.unique())].copy())


def features(x):
    vet = x.row_type == "veteran"
    x["age_c"] = np.where(~vet, "rookie", np.where(x.age < 26, "25-", np.where(x.age < 30, "26-29", "30+")))
    x["redraft"], x["production"] = gap(x.mkt_pos_rank, x.rd_rank), gap(x.mkt_pos_rank, x.ppg_pos_rank)
    n = x.groupby(["t", "position"]).mkt_pos_rank.transform("max")
    x["lg_rd"] = np.log(x.mkt_pos_rank / x.rd_rank.fillna(2 * n))
    x["lg_pp"] = np.log(x.mkt_pos_rank / x.ppg_pos_rank.fillna(2 * n))
    return x


def load_aug(h):
    """the same-date check: FP dynasty ECR on its last scrape before Sept 5 against FP redraft ECR that week; r against that
    August price (isotonic on rank, top 200, as inseason_fp.py)"""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import market_residuals as mr
    from model import load as panel_load
    from sklearn.isotonic import IsotonicRegression

    def pick(ds, y):
        c = [d for d in ds if f"{y}-07-15" <= d <= f"{y}-09-05"]
        return c[-1] if c else None
    et, pages = ("dsf", ["dynasty-op"]) if mr.SF else ("do", ["dynasty-overall", "dynasty-offense"])
    y = f"vorp{h}"
    x = mr.fp_market(et, pages, pick).merge(panel_load(), on=["gsis_id", "t"]).merge(mr.outcomes()[["gsis_id", "t", y]], on=["gsis_id", "t"])
    x = x.dropna(subset=[y])
    x["rank"] = x.groupby("t").mkt.rank(method="first")
    x = x[x["rank"] <= 200].copy()
    x["r"] = (x[y] - IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x["rank"], x[y]).predict(x["rank"])) / x[y].mean() * 100
    x["mkt_pos_rank"] = x.groupby(["t", "position"]).mkt.rank(method="first")
    rd = mr.fp_redraft()
    rd = rd.merge(x[["gsis_id", "t", "position"]].drop_duplicates(), on=["gsis_id", "t"], how="left")
    pm = panel_load()[["gsis_id", "t", "position"]].drop_duplicates(["gsis_id", "t"])
    rd = rd.drop(columns="position").merge(pm, on=["gsis_id", "t"])
    rd["rd_rank"] = rd.groupby(["t", "position"]).mkt.rank(method="first")
    may = pd.read_parquet(out_dir(SF) / "market_rows" / "fp_dynasty_h1.parquet")[["gsis_id", "t", "ppg_pos_rank"]]
    x = x.merge(rd[["gsis_id", "t", "rd_rank"]], on=["gsis_id", "t"], how="left").merge(may, on=["gsis_id", "t"], how="left")
    x = x[x.t.isin(rd.t.unique())].copy()
    x["mk"] = "fp_dynasty (August)"
    print("   dates: " + ", ".join(f"{t + 1}: {d}" for t, d in x.groupby("t").scrape_date.first().items()))
    return features(x)


def cell(g):
    if g.drop_duplicates(["gsis_id", "t"]).shape[0] < 15:
        return ""
    coh = g.groupby("t").r.mean()
    return f"{g.r.mean():+4.0f}/{g.r.median():+4.0f} {(g.r > 0).mean() * 100:2.0f}% {(coh > 0).sum()}/{len(coh)} n{len(g.drop_duplicates(['gsis_id', 't']))}"


def table(x, by, col):
    t = {(k if isinstance(k, str) else " | ".join(k)): {c: cell(g[g[col] == c]) for c in LV} for k, g in x.groupby(by)}
    return pd.DataFrame(t).T[LV]


def main():
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 40)
    print(f"== {'Superflex' if SF else '1QB'}, {REDRAFT}: mean / median residual %, share beating price, cohorts positive, player-seasons "
          "(blank under 15). Dynasty markets pooled")
    for h, loader, name in [(h, f, n) for f, n in ((load, "May dynasty prices"), (load_aug, "same date: FP dynasty and FP redraft in late August"))
                            for h in (1, 2, 3)]:
        x = loader(h)
        v = x[x.row_type == "veteran"]
        print(f"\n==== {name}: {h}-year return, cohorts {sorted(x.t.unique())[0]}-{sorted(x.t.unique())[-1]}")
        print("-- redraft rank against dynasty rank, by age class (columns: redraft up / level / down)")
        print(table(x, "age_c", "redraft").to_string())
        print("-- last season's production against dynasty rank, by age class")
        print(table(v, "age_c", "production").to_string())
        print("-- both: rows = production gap, columns = redraft gap, veterans")
        print(table(v, "production", "redraft").reindex(LV).to_string())
        print("-- both, by age class")
        print(table(v, ["age_c", "production"], "redraft").to_string())
        print("-- by position: redraft gap, then production gap")
        print(table(v, "position", "redraft").to_string())
        print(table(v, "position", "production").to_string())
        print("-- by dynasty source: redraft gap, then production gap")
        print(table(v, "mk", "redraft").to_string())
        print(table(v, "mk", "production").to_string())
        rng, out = np.random.default_rng(2), []
        for ac, g in list(v.groupby("age_c")) + [("all veterans", v)]:
            def fit(q):
                z = [(q[c] - q[c].mean()) / q[c].std() for c in ("lg_rd", "lg_pp")]
                return np.linalg.lstsq(np.column_stack([np.ones(len(q))] + z), q.r.to_numpy(), rcond=None)[0][1:]
            b, parts, ts = fit(g), dict(tuple(g.groupby("t"))), g.t.unique()
            bs = np.array([fit(pd.concat([parts[k] for k in rng.choice(ts, len(ts))])) for _ in range(300)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            by = np.array([fit(q) for _, q in g.groupby("t")])
            out.append({"age": ac, "corr of gaps": round(g.lg_rd.corr(g.lg_pp), 2),
                        **{n: f"{b[k]:+5.0f} [{lo[k]:+.0f}, {hi[k]:+.0f}] {int((by[:, k] > 0).sum())}/{len(by)}"
                           for k, n in enumerate(("redraft gap", "production gap"))}})
        print("-- residual % per 1 SD of each gap (log ratio to the dynasty rank), both in the model [95% CI over cohorts] cohorts positive")
        print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
