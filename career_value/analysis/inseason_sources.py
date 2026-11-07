"""In-season accuracy: FP dynasty ECR vs Dynasty Daddy sources (KTC, FantasyCalc, DP, DD ADP, ...) at the same dates.
Cycle Y (2021-2024): Sept 1, Oct 18, Nov 17, Dec 22 (Y), Feb 17, May 15 (Y+1). Outcome: VORP in seasons Y+1..Y+h.
Universe per (cycle, snapshot): union of each available source's top 150 among veterans who played season Y."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))  # project modules
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import market_residuals as mr
from model import load

SNAPS = {"09-01": "1 Sep (pre)", "10-18": "2 Oct 18 (wk6)", "11-17": "3 Nov 17 (wk11)", "12-22": "4 Dec 22 (wk16)",
         "02-17": "5 Feb 17", "05-15": "6 May 15"}
panel = load(); outc = mr.outcomes()
vets = panel[panel.row_type == "veteran"][["gsis_id", "t"]]

# Dynasty Daddy sources, in-season + May
dd = {}
for pat in ["inseason_*.json", "snapshot_*.json"]:
    for k, v in mr.dd_markets(panel, pat).items():
        dd.setdefault(k, []).append(v)
src = []
for k, vs in dd.items():
    v = pd.concat(vs); v["snap"] = v.snap_date.str[5:].map(SNAPS); v["market"] = k
    src.append(v.dropna(subset=["snap"]))
# FP dynasty at the nearest scrape on/after each DD date (within 10 days)
ET, PAGES = (("dsf", ["dynasty-op"]) if mr.SF else ("do", ["dynasty-overall", "dynasty-offense"]))
for md, lab in SNAPS.items():
    def pick(ds, y, md=md):
        Y = y - 1; yy = Y + 1 if md < "07" else Y
        lo = pd.Timestamp(f"{yy}-{md}"); d = next((d for d in ds if d >= f"{yy}-{md}"), None)
        return d if d is not None and pd.Timestamp(d) - lo <= pd.Timedelta(days=10) else None
    f = mr.fp_market(ET, PAGES, pick).assign(snap=lab, market="fp_dynasty")
    src.append(f.rename(columns={"scrape_date": "snap_date"})[["gsis_id", "t", "mkt", "snap_date", "snap", "market"]])
S = pd.concat(src, ignore_index=True)
S = S[S.t.between(2021, 2024)].merge(vets, on=["gsis_id", "t"])
avail = S.groupby(["market", "t", "snap"]).size().unstack("t")
print("players ranked (veterans) per source / snapshot / cycle:"); print(avail.fillna(0).astype(int).to_string())

rows = []
for h in [1, 2, 3]:
    y = f"vorp{h}"
    for (t, snap), g in S.groupby(["t", "snap"]):
        if t + h > mr.LAST: continue
        g = {m: x for m, x in g.groupby("market") if len(x) >= 100}
        if "fp_dynasty" not in g or len(g) < 2: continue
        u = set().union(*[set(x.nsmallest(150, "mkt").gsis_id) for x in g.values()])
        u = pd.DataFrame({"gsis_id": sorted(u), "t": t}).merge(outc[["gsis_id", "t", y]], on=["gsis_id", "t"])
        for m, x in g.items():
            s = u.merge(x[["gsis_id", "mkt"]], on="gsis_id", how="left").mkt
            s = s.fillna(s.max() + 1)
            rows.append({"h": h, "t": t, "snap": snap, "market": m, "rho": spearmanr(-s, u[y]).statistic})
r = pd.DataFrame(rows)
r.to_csv(mr.OUT / "inseason_source_accuracy.csv", index=False)
fp = r[r.market == "fp_dynasty"].set_index(["h", "t", "snap"]).rho
r["d"] = r.rho - r.set_index(["h", "t", "snap"]).index.map(fp)
for h in [1, 2, 3]:
    x = r[(r.h == h) & (r.market != "fp_dynasty")]
    print(f"\n== h={h}: Spearman minus FP dynasty, same players & dates (cycles source beats FP / cycles)")
    print(x.groupby(["market", "snap"]).apply(lambda g: f"{g.d.mean():+.3f} ({(g.d>0).sum()}/{len(g)})").unstack("snap").fillna("").to_string())
    print("FP's own rho by snap:", r[(r.h == h) & (r.market == "fp_dynasty")].groupby("snap").rho.mean().round(3).to_dict())
# pooled in-season (Oct-Dec) vs offseason (Feb, May) vs preseason
r["phase"] = r.snap.map(lambda s: "pre (Sep)" if s.startswith("1") else "in-season (Oct-Dec)" if s[0] in "234" else "offseason (Feb-May)")
x = r[r.market != "fp_dynasty"]
print("\n== pooled: mean diff vs FP (source-snapshot-cycles beating FP / total), all horizons")
print(x.groupby(["market", "phase"]).apply(lambda g: f"{g.d.mean():+.3f} ({(g.d>0).sum()}/{len(g)})").unstack("phase").fillna("").to_string())
print("\n== rank of each source by mean rho within (h, cycle, snap) where KTC, FC, DP and FP all present")
core = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
w = r.pivot_table(index=["h", "t", "snap", "phase"], columns="market", values="rho")
w = w.dropna(subset=core)
print(w.groupby("phase")[[c for c in w.columns if w[c].notna().all()]].mean().round(3).T.to_string())
print("best-of-core count:", w[core].idxmax(axis=1).groupby(level="phase").value_counts().unstack().fillna(0).astype(int).to_string())
