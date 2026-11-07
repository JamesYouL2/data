"""Q: are there RBs that are good long-term (5-10 year) buys?

Method: long_horizon.py's translated price (MFL keeper ADP -> dynasty-style price, model trained on 2018-24 overlap),
residual % of h-year VORP against it for pre-overlap cohorts (h3/h5: 2010-17, h10: 2010-15). RB subgroups by age,
draft capital, receiving role, production vs price, experience and price tier. FP dynasty's own 5-year residuals
(cohorts 2018-20) as a direct check. LEAGUE=sf for superflex.
"""
import pandas as pd

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # noqa: E401
from common import rows
from long_horizon import FEATS, SF, dyn_target, model, prep, resid_pct, target_for


def rb_groups(x):
    rb, vet = x.position == "RB", x.row_type == "veteran"
    a, tg, pr, pp = x.age, x.targets_pg, x.mkt_pos_rank, x.ppg_pos_rank
    disc = vet & (pr >= pp + 6)   # priced 6+ spots below last year's PPG rank
    hype = vet & (pr <= pp - 6)
    g = {
        "all RBs (veterans)": vet,
        "age <=22": vet & (a < 23), "age 23": vet & a.between(23, 23.99), "age 24": vet & a.between(24, 24.99),
        "age 25": a.between(25, 25.99), "age 26": a.between(26, 26.99), "age 27": a.between(27, 27.99),
        "age 28+": a >= 28,
        "NFL R1": vet & (x.draft_round_f == 1), "NFL R2": vet & (x.draft_round_f == 2),
        "NFL R3+ / UDFA": vet & ~x.draft_round_f.isin([1, 2]),
        "4+ targets/g": vet & (tg >= 4), "3-4 targets/g": vet & tg.between(3, 3.99), "<2 targets/g": vet & (tg < 2),
        "<=24 & 3+ targets/g": vet & (a < 25) & (tg >= 3),
        "<=24 & NFL R1": vet & (a < 25) & (x.draft_round_f == 1),
        "<=24 & top-12 PPG": vet & (a < 25) & (pp <= 12),
        "<=24 & PPG 13-36 (not yet a starter)": vet & (a < 25) & pp.between(13, 36),
        "priced below PPG (6+ spots)": disc, "priced above PPG (6+ spots)": hype,
        "<=25 & priced below PPG": disc & (a < 26),
        "entering year 2": vet & (x.exp == 1), "entering year 3": vet & (x.exp == 2),
        "priced RB1-3": x.mkt_pos_rank <= 3, "priced RB4-12": x.mkt_pos_rank.between(4, 12),
        "priced RB13-24": x.mkt_pos_rank.between(13, 24), "priced RB25+": x.mkt_pos_rank >= 25,
        "<=24 & priced RB13+": vet & (a < 25) & (pr >= 13),
        "missed 8+ games last year": vet & (x.games <= 9),
    }
    return {k: (v & rb).fillna(False).to_numpy() for k, v in g.items()}


def cell(z, m, col):
    g = z[m]
    if len(g) < 8:
        return ""
    c = g.groupby("t")[col].mean()
    return f"{g[col].mean():+.0f} / {g[col].median():+.0f} ({(c > 0).sum()}/{len(c)}) n{len(g)}"


def main():
    d, live = dyn_target()
    m1 = prep(rows(["mfl_dynasty"], 1, SF))
    ov = m1[m1.t >= 2018].merge(target_for(m1[m1.t >= 2018], d, live), on=["gsis_id", "t"])
    f = model().fit(ov[FEATS], ov.y)
    out, Z = {}, {}
    for h, coh in ((3, range(2010, 2018)), (5, range(2010, 2018)), (10, range(2010, 2016))):
        z = prep(rows(["mfl_dynasty"], h, SF))
        z = z[z.t.isin(coh)].copy()
        z["pred"] = f.predict(z[FEATS])
        z["r"] = resid_pct(z, "pred", f"vorp{h}")
        Z[h] = z
        out[f"{h} yr translated"] = {k: cell(z, m, "r") for k, m in rb_groups(z).items()}
    fp = rows(["fp_dynasty"], 5, SF)
    out["5 yr FP direct (2018-20)"] = {k: cell(fp, m, "r") for k, m in rb_groups(fp).items()}
    print(f"== {'Superflex' if SF else '1QB'} RBs: residual % mean / median (cohorts positive / total) n")
    print(pd.DataFrame(out).to_string())
    # names in the most promising cells (10-year)
    z = Z[10]
    for k in ["<=24 & NFL R1", "<=24 & 3+ targets/g", "<=25 & priced below PPG", "<=24 & PPG 13-36 (not yet a starter)"]:
        g = z[rb_groups(z)[k]].sort_values("r", ascending=False)
        lab = lambda q: ", ".join(f"{n} {t + 1} {r:+.0f}" for n, t, r in zip(q.display_name, q.t, q.r))
        print(f"\n{k} (10 yr): best {lab(g.head(6))}\n   worst {lab(g.tail(4))}; "
              f"share beating price {(g.r > 0).mean():.0%}")


if __name__ == "__main__":
    main()
