"""Q: do the playbook's edges hold out of sample?

Walk-forward. For each test cohort T, a market's price curve (isotonic rank -> h-year VORP) and each rule's
average residual are learned only from cohorts t <= T - h (outcomes known by then). The rule is then scored
on cohort T against that past-only curve. Pooled over test cohorts this is what following the rule in real
time would have returned.

  vets    : dynasty markets pooled (FP, KTC, FC, DP; 2019-2024) and MFL keeper ADP (2010-2024)
  rookies : rookie_position_residuals.csv (rookie_positions.py), MFL rookie order 2011-25 and dynasty 2020-25
  generic : every g_* market group + rule: does a group's past residual predict its next-cohort residual?
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import DYN, out_dir, rows


def vet_rules(x):
    pos, vet = x.position, x.row_type == "veteran"
    hype = vet & (x.mkt_pos_rank <= x.ppg_pos_rank - 6)
    return {
        "RB 27.5+ & top-12 PPG": (pos == "RB") & (x.age >= 27.5) & (x.ppg_pos_rank <= 12),
        "RB top-6 PPG": (pos == "RB") & (x.ppg_pos_rank <= 6),
        "RB 4+ targets/g": (pos == "RB") & vet & (x.targets_pg >= 4),
        "RB priced 4-6": (pos == "RB") & x.mkt_pos_rank.between(4, 6),
        "RB priced 1-3": (pos == "RB") & (x.mkt_pos_rank <= 3),
        "RB <25 & hyped": (pos == "RB") & (x.age < 25) & hype,
        "TE all": pos == "TE",
        "TE priced 1-3": (pos == "TE") & (x.mkt_pos_rank <= 3),
        "TE <25": (pos == "TE") & (x.age < 25),
        "TE 29+": (pos == "TE") & (x.age >= 29),
        "WR 30+": (pos == "WR") & (x.age >= 30),
        "age 30+ (all)": x.age >= 30,
        "age <=23 (all)": x.age < 23.5,
        "rookies": x.row_type == "rookie",
        "injured (<=9 g), top-100 price": vet & (x.games <= 9) & (x.mkt_rank <= 100),
    }


def rookie_rules(x):
    p = x.position
    return {
        "WR NFL R1": (p == "WR") & (x.nfl == "NFL R1"),
        "WR NFL R2": (p == "WR") & (x.nfl == "NFL R2"),
        "RB NFL R1": (p == "RB") & (x.nfl == "NFL R1"),
        "RB NFL R2": (p == "RB") & (x.nfl == "NFL R2"),
        "TE all": p == "TE",
        "TE rookie R1-R2 slot": (p == "TE") & (x.slot <= 24),
        "QB rookie R3-R4 slot": (p == "QB") & (x.slot > 24),
        "QB rookie R1 slot": (p == "QB") & (x.slot <= 12),
        "WR <=21": (p == "WR") & (x.age < 21.5),
        "RB 23+": (p == "RB") & (x.age >= 22.5),
    }


def walk(x, h, ycol, rank_col, rule_fn, groups=(), min_te=3):
    """Returns per (T, rule) train/test stats and pooled test residual rows."""
    stats, test_rows = [], []
    for T in sorted(x.t.unique()):
        tr_all, te_all = x[x.t <= T - h], x[x.t == T]
        trs, tes = [], []
        for m, te in te_all.groupby("mk"):
            tr = tr_all[tr_all.mk == m]
            if tr.t.nunique() < 2:
                continue
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(tr[rank_col], tr[ycol])
            sc = tr[ycol].mean()
            trs.append(tr.assign(rr=(tr[ycol] - iso.predict(tr[rank_col])) / sc * 100))
            tes.append(te.assign(rr=(te[ycol] - iso.predict(te[rank_col])) / sc * 100))
        if not tes:
            continue
        tr, te = pd.concat(trs), pd.concat(tes)
        fr, fe = rule_fn(tr), rule_fn(te)
        for g in groups:
            for lev in te[g].dropna().unique():
                fr[f"{g}={lev}"], fe[f"{g}={lev}"] = tr[g] == lev, te[g] == lev
        for k in fe:
            a, b = tr[fr[k].fillna(False).to_numpy()], te[fe[k].fillna(False).to_numpy()]
            if len(a) >= 10 and len(b) >= min_te:
                stats.append({"T": T, "rule": k, "n_tr": len(a), "tr": a.rr.mean(), "n_te": len(b), "te": b.rr.mean()})
                test_rows.append(b.assign(rule=k)[["rule", "t", "gsis_id", "rr"]])
    return pd.DataFrame(stats), pd.concat(test_rows) if test_rows else pd.DataFrame()


def summarize(stats, test, insample, label, rules):
    out = []
    for k in rules:
        s, e = stats[stats.rule == k], test[test.rule == k]
        if s.empty:
            continue
        out.append({"rule": k, "in-sample (all cohorts)": insample.get(k, np.nan),
                    "past mean when used": (s.tr * s.n_te).sum() / s.n_te.sum(),
                    "OUT-OF-SAMPLE": e.rr.mean(), "OOS median": e.rr.median(),
                    "test cohorts +": f"{(s.te > 0).sum()}/{len(s)}", "n_test": len(e),
                    "sign held": f"{(np.sign(s.tr) == np.sign(s.te)).sum()}/{len(s)}"})
    print(f"\n== {label}")
    print(pd.DataFrame(out).set_index("rule").round(0).to_string())


def generic(stats, label):
    s = stats[(stats.n_tr >= 30) & (stats.n_te >= 10)]
    w = s.n_te
    c = np.cov(s.tr, s.te, aweights=w)
    pos, neg = s[s.tr > 10], s[s.tr < -10]
    print(f"   generic [{label}]: {len(s)} group-cohorts; weighted corr(past, next) = {c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]):+.2f}; "
          f"groups past > +10 -> next {np.average(pos.te, weights=pos.n_te):+.0f}; past < -10 -> next "
          f"{np.average(neg.te, weights=neg.n_te):+.0f}; past sign held {(np.sign(s.tr) == np.sign(s.te)).mean() * 100:.0f}%")


def main():
    for sf in (False, True):
        lg = "SF" if sf else "1QB"
        for fam, mks, hs in (("dynasty pooled", DYN, (1, 2)), ("MFL vets", ["mfl_dynasty"], (1, 2, 3, 5))):
            for h in hs:
                x = rows(mks, h, sf)
                gcols = [c for c in x.columns if c.startswith("g_")]
                rl = vet_rules(x)
                insample = {k: x[v.fillna(False)].r.mean() for k, v in rl.items()}
                st, te = walk(x, h, f"vorp{h}", "mkt_rank", vet_rules, gcols)
                summarize(st, te, insample, f"{lg} {fam} h={h}: walk-forward rule residual %", rl)
                generic(st, f"{lg} {fam} h={h}")
        f = out_dir(sf) / "rookie_position_residuals.csv"
        r = pd.read_csv(f)
        for fam, mk in (("MFL rookie order", ["mfl"]), ("dynasty rookie order", ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"])):
            for h in ((3, 5) if fam.startswith("MFL") else (1, 2)):
                x = r[r.market.isin(mk) & (r.h == h)].rename(columns={"market": "mk"})
                if fam.startswith("MFL") and sf:
                    continue  # 1QB ADP order is not a superflex price
                rl = rookie_rules(x)
                insample = {k: x[v].res.mean() for k, v in rl.items()}
                st, te = walk(x, h, "act", "slot", rookie_rules, ["position"], min_te=1)
                summarize(st, te, insample, f"{lg} {fam} h={h}: walk-forward rookie rule residual % (by class)", rl)


if __name__ == "__main__":
    main()
