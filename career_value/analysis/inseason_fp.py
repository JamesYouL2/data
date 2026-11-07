"""In-season vs offseason FP dynasty ECR. Cycle Y: snapshots from Aug Y to May Y+1, all scored on seasons Y+1..Y+h."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))  # project modules
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
import market_residuals as mr
from model import load

def first_on(md):  # md relative to cycle year Y = y-1
    def f(ds, y):
        Y = y - 1
        yy, mmdd = (Y + 1, md[1:]) if md.startswith("+") else (Y, md)
        lo = pd.Timestamp(f"{yy}-{mmdd}"); d = next((d for d in ds if d >= f"{yy}-{mmdd}"), None)
        return d if d is not None and pd.Timestamp(d) - lo <= pd.Timedelta(days=21) else None
    return f
def last_before(md):
    def f(ds, y):
        c = [d for d in ds if d <= f"{y-1}-{md}"]
        return c[-1] if c and c[-1] >= f"{y-1}-07-15" else None
    return f
DATES = {"1 Aug (pre)": last_before("09-05"), "2 mid-Oct (wk6)": first_on("10-15"), "3 mid-Nov (wk11)": first_on("11-15"),
         "4 late-Dec (wk16)": first_on("12-20"), "5 mid-Feb": first_on("+02-15"), "6 mid-May": first_on("+05-10")}
ET, PAGES = (("dsf", ["dynasty-op"]) if mr.SF else ("do", ["dynasty-overall", "dynasty-offense"]))

panel = load(); outc = mr.outcomes()
vets = panel[panel.row_type == "veteran"]
mk = {}
for k, f in DATES.items():
    m = mr.fp_market(ET, PAGES, f)
    mk[k] = m[m.t.between(2019, 2024)].merge(vets, on=["gsis_id", "t"])
print("snapshot dates used:"); print(pd.DataFrame({k: v.drop_duplicates("t").set_index("t").scrape_date for k, v in mk.items()}).to_string())

# 1) ranking accuracy on a common universe per cycle
rows = []
for h in [1, 2, 3]:
    y = f"vorp{h}"
    for t in range(2019, 2025):
        if t + h > mr.LAST: continue
        av = {k: v[v.t == t] for k, v in mk.items() if (v.t == t).sum() >= 100}
        if len(av) < len(DATES): continue
        u = set().union(*[set(v.nsmallest(150, "mkt").gsis_id) for v in av.values()])
        u = pd.DataFrame({"gsis_id": sorted(u), "t": t}).merge(outc[["gsis_id", "t", y]], on=["gsis_id", "t"])
        for k, v in av.items():
            s = u.merge(v[["gsis_id", "mkt"]], on="gsis_id", how="left").mkt
            s = s.fillna(s.max() + 1)
            rows.append({"h": h, "t": t, "snap": k, "rho": spearmanr(-s, u[y]).statistic})
r = pd.DataFrame(rows)
print("\nSpearman vs VORP in seasons Y+1..Y+h, same players, cycles with all 6 snapshots (n cycles)")
print(r.groupby(["snap", "h"]).rho.mean().round(3).unstack().to_string(), "\ncycles:", r.groupby("h").t.nunique().to_dict())
base = r[r.snap == "1 Aug (pre)"].set_index(["h", "t"]).rho
r["gain_vs_aug"] = r.rho - r.set_index(["h", "t"]).index.map(base)
print("\ngain over the August ranking (cycles better):")
print(r.groupby(["snap", "h"]).apply(lambda g: f"{g.gain_vs_aug.mean():+.3f} ({(g.gain_vs_aug>0).sum()}/{len(g)})").unstack().to_string())

# 2) residuals at each snapshot: who beats the in-season price?
def resid_frame(k, h):
    y = f"vorp{h}"
    x = mk[k].merge(outc[["gsis_id", "t", y]], on=["gsis_id", "t"]).dropna(subset=[y])
    x["rank"] = x.groupby("t").mkt.rank(method="first"); x = x[x["rank"] <= 200].copy()
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x["rank"], x[y])
    x["r"] = (x[y] - iso.predict(x["rank"])) / x[y].mean() * 100
    x["pos_rank"] = x.groupby(["t", "position"]).mkt.rank(method="first")
    return x
aug = mk["1 Aug (pre)"].copy(); aug["aug_pos"] = aug.groupby(["t", "position"]).mkt.rank(method="first")
cells = {}
for k in DATES:
    for h in [1, 2, 3]:
        x = resid_frame(k, h).merge(aug[["gsis_id", "t", "aug_pos"]], on=["gsis_id", "t"], how="left")
        x["age_b"] = pd.cut(x.age, [0, 23.5, 26.5, 29.5, 99], labels=["age <=23", "age 24-26", "age 27-29", "age 30+"]).astype(str)
        x["yr1"] = np.where(x.seasons_played <= 1, "1st-year player", "")
        mv = np.log(x.aug_pos / x.pos_rank)
        x["move"] = np.where(x.aug_pos.isna(), "unranked in Aug", np.where(mv >= np.log(1.5), "riser since Aug (>=1.5x)", np.where(mv <= -np.log(1.5), "faller since Aug (>=1.5x)", "steady since Aug")))
        x["inj"] = np.where(x.games <= 9, "<=9 g in season Y", "")
        for col in ["position", "age_b", "yr1", "move", "inj"]:
            for lev, g in x.groupby(col):
                if lev == "" or len(g) < 15: continue
                coh = g.groupby("t").r.mean()
                cells[(col, lev, k, h)] = f"{g.r.mean():+.0f} c{(coh>0).sum()}/{len(coh)}"
c = pd.Series(cells); c.index.names = ["grp", "lev", "snap", "h"]
for h in [1, 2, 3]:
    print(f"\nResidual % at each snapshot's price, h={h} (cohorts positive)")
    print(c.xs(h, level="h").unstack("snap").fillna("").to_string())
