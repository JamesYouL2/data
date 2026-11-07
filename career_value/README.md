# Rest-of-career value model

What drives a fantasy player's rest-of-career value? Which groups of players beat or trail the market (consensus rankings and ADP)?

## Definitions

- **Season value** is PPR points above replacement (VORP), floored at 0. Replacement is the QB13 / RB31 / WR43 / TE13 finisher that season (12-team, 1QB, 2RB, 3WR, 1TE, 1 FLEX).
- **Rest-of-career (ROC) value** is the sum of VORP over every season after the snapshot.
- **Snapshot**: the offseason after season *t*. Every feature uses data known at that point. Rookies get a pre-rookie snapshot using only draft position, age and size.
- **Market prices are offseason or preseason.** FP dynasty and Dynasty Daddy are mid-May, MFL keeper ADP is summer drafts, and the redraft sources are late August. Section 10 checks in-season prices.

## Run

```
./fetch_data.sh                 # nflverse season stats 1999-2026 + player bio/draft table
uv run fetch_markets.py         # MFL keeper ADP 2011-25, FFC PPR ADP 2013-25
uv run fetch_dynastydaddy.py    # Dynasty Daddy archive: May-15 snapshots 2021-26 + in-season Sep-Feb 2021-24 (KTC, FantasyCalc, DP, ...)
uv run build_dataset.py         # data/panel.parquet
uv run model.py                 # ROC model + primary-variable analysis  -> outputs/
uv run market_check.py          # model vs FP dynasty ECR
uv run market_residuals.py      # which groups beat/trail each market    -> outputs/
uv run joint_check.py           # joint regression of those groupings
uv run validate_horizons.py     # ranking accuracy of every market + real-time model at 1/2/3/5/10 yrs
uv run youth_premium.py         # is the KTC/FC youth premium justified at 5/10 yrs? (MFL bridge)
uv run blend_test.py            # ATC-style blend of sources + real-time bias layer
# superflex: same scripts with LEAGUE=sf (outputs -> outputs/sf/)
LEAGUE=sf uv run market_residuals.py && LEAGUE=sf uv run validate_horizons.py
LEAGUE=sf uv run joint_check.py && LEAGUE=sf uv run youth_premium.py
uv run picks.py && LEAGUE=sf uv run picks.py   # are rookie draft picks priced correctly? (needs no other step)
```

Follow-up analyses for sections 3b and 7–12 are in `analysis/`. Run them from this folder after `market_residuals.py` (they read `outputs/market_rows/`), with `uv run --with pandas --with pyarrow --with scikit-learn --with scipy --with statsmodels python analysis/<script>.py`. `old_*.py` take the outputs folder as an argument (`outputs` or `outputs/sf`). `rookie_positions.py` also needs MFL's yearly player lists in `raw/mfl_players/players_{year}.json` (`https://api.myfantasyleague.com/{year}/export?TYPE=players&JSON=1`).

| Script | Section |
|---|---|
| `old_age_bands.py`, `old_rb_te_slices.py`, `old_rb_te_buy.py`, `old_rb_top12_rule.py` | 3b, 7: old RBs, WRs and TEs, and when to buy them |
| `sf_qb_1qb_wr_studs.py` | 7 |
| `rb_value_hold.py` | 8 |
| `injured.py` | 9 |
| `inseason_fp.py`, `inseason_sources.py` | 10, 10a |
| `contender_rb_te.py` | 11 |
| `rookie_positions.py` | 12 |

## 1. Primary variables for rest-of-career value

- **Data**: trained on snapshots 2002–2010, tested on 2011–2014. Only 3.7% of the 2014 test players were still active in 2025, so careers are essentially complete.
- **Model**: gradient boosting with Poisson loss.
- **Test performance**: R² 0.24, Spearman 0.55. The baselines score lower: last-season VORP alone gets R² 0.12, and age, position and draft pick together get 0.05.

| Variable group | Test R² lost when permuted | Note |
|---|---|---|
| **Age** | 0.36 | By far the main driver. Value falls steeply after 26–27 for RB, WR and TE, and after about 29 for QB (`outputs/age_partial_dependence.csv`). |
| **Current-season production** (VORP, positional rank, PPG) | 0.16 | |
| **Draft capital** | 0.07 overall, 0.24 for rookies | The dominant signal for rookies. |
| **Experience** | 0.05 | |
| Size | 0.04 | |
| Usage (targets, carries, shares), EPA, prior-season and career production, team change | ≈ 0 | Add nothing once current production and age are known. |

Greedy forward selection picks these in order: last-season VORP → age → positional rank → prior-season VORP.

**Market check** (`market_check.py`): we compared outcomes over the next 5 seasons for the 2019 and 2020 snapshots.
- FP dynasty ECR has a Spearman of 0.65 against those outcomes; the stats model has 0.58.
- Combining both raises R² only from 0.465 to 0.474.
- The market already prices the box-score variables. Edges have to come from systematic biases, which section 2 looks for.

## 2. Markets and horizons

**Outcome.** VORP summed over seasons t+1 to t+h, for h = 1, 2, 3, 5 and 10. A cohort counts at horizon h only if t+h ≤ 2025.

| Market | Source | Snapshots (t) | Cohorts at h = 1 / 2 / 3 / 5 / 10 |
|---|---|---|---|
| fp_dynasty | FantasyPros dynasty ECR, mid-May (this repo) | 2019–2024 | 6 / 5 / 4 / 2 / 0 |
| mfl_dynasty | MFL keeper-league PPR ADP, veterans only | 2010–2024 | 15 / 14 / 13 / 11 / 6 |
| dd_ktc | KeepTradeCut 1QB value, via the Dynasty Daddy archive (May 16) | 2020–2024 | 5 / 4 / 3 / 1 / 0 |
| dd_fantasycalc | FantasyCalc 1QB value (Dynasty Daddy) | 2021–2024 | 4 / 3 / 2 / 0 / 0 |
| dd_dynastyprocess | DynastyProcess value (Dynasty Daddy) | 2022–2024 | 3 / 2 / 1 / 0 / 0 |
| dd_dynastysuperflex, dd_pfn, dd_dd_adp, dd_draftsharks | Dynasty Daddy | 2023–2024 | 1–2 at h = 1 |
| fp_redraft | FantasyPros redraft ECR, preseason (this repo) | 2019–2024 | h = 1 only, 6 |
| ffc_redraft | Fantasy Football Calculator 12-team PPR ADP, preseason | 2012–2024 | h = 1 only, 13 |

The Dynasty Daddy archive (`/api/v1/player/all/prev/{days}`) starts in May 2021 for KTC. FantasyCalc and DP values start later, in 2022–2023. Players matched to gsis ids at 99.1%. **10-year validation is possible only for MFL ADP and the stats model. 5-year validation is possible for FP dynasty (2 cohorts), KTC (1) and MFL (11).**

### 2a. Ranking accuracy by horizon (`validate_horizons.py`)

**Method.**
- **Universe**: for each cohort, the union of every available market's top 150. A player missing from a market gets that market's worst rank.
- **Metric**: Spearman correlation against h-year VORP.
- **Real-time model**: the stats model is retrained for each snapshot T, on snapshots t ≤ T−h only.
- **Head-to-head**: markets cover different years, so each market is compared with the model, KTC or MFL only on the cohorts they share. Cells show the mean Spearman difference, with cohorts won / cohorts shared in brackets.

| Comparison on shared cohorts | h=1 | h=2 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| fp_dynasty − model (all players) | +.02 (5/6) | +.05 (5/5) | +.06 (4/4) | +.12 (2/2) | |
| dd_ktc − model (all players) | −.01 (2/5) | +.01 (2/4) | +.04 (3/3) | +.11 (1/1) | |
| mfl_dynasty − model (veterans) | +.07 (15/15) | +.05 (12/14) | +.06 (12/13) | +.06 (10/11) | +.13 (6/6) |
| fp_dynasty − dd_ktc | +.03 (5/5) | +.03 (4/4) | +.01 (3/3) | −.02 (0/1) | |
| dd_fantasycalc − dd_ktc | +.01 (3/4) | −.01 (1/3) | −.01 (1/2) | | |
| fp_redraft − dd_ktc (h=1) | +.08 (5/5) | | | | |

- **Every market beats a box-score model, and the gap grows with horizon.** The model's own Spearman is about 0.44 at 1 year, 0.51 at 2–3 years, 0.48 at 5 years and 0.36 at 10 years (veterans). Markets hold about 0.50–0.62 at the same horizons.
- **FP dynasty ECR ranks a little better than KTC at 1–3 years**: it won every shared cohort, though only 3–5 of them. FantasyCalc and KTC are about even.
- **MFL keeper ADP is the best veteran ranking at 1–2 years**, ahead of every dynasty value source at h = 1. That fits a win-now drafting crowd.
- **For next season, use redraft ADP/ECR**, which is taken later (preseason): +0.07 to +0.08 over KTC.
- Full tables: `outputs/horizon_validation.csv`, `outputs/horizon_head_to_head.csv`, `outputs/horizon_validation_by_cohort.csv`.

### 2b. Who beats or trails each market (`market_residuals.py`, `joint_check.py`)

**Method.**
1. For each market and horizon, fit the expected h-year VORP from market rank with isotonic regression, pooled over years. Universe: the market's top 200.
2. Average each player's residual within groupings defined at snapshot time. Units are % of the average outcome.
3. `*` marks a robust cell: the player-bootstrap 90% CI excludes 0, there are at least 3 cohorts, and at least 70% of cohorts agree in sign.

Tables: `outputs/market_residuals_by_group.csv`, `outputs/market_residuals_summary_by_horizon.csv`, `outputs/joint_residual_regression.csv`.

**The age result depends on the horizon, and that tells you what horizon each market prices.**

| Age ≤23, residual % | h=1 | h=2 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| fp_dynasty | −22* | −6 | +1 | +14 | |
| dd_ktc | −37* | −23* | −15 | −10 | |
| dd_fantasycalc | −40* | −19* | −5 | | |
| mfl_dynasty | +15 | +24* | +30* | +50* | +72* |

| Age 30+, residual % | h=1 | h=2 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| fp_dynasty | +32* | +23* | +13 | +2 | |
| dd_ktc | +43* | +35* | +24* | | |
| dd_fantasycalc | +32* | +26* | +12 | | |
| mfl_dynasty | −14* | −21* | −26* | −36* | −32* |

The same split shows up for rookies vs year 7+ players, young vs old WRs and RBs, and in the joint regression. For example, the age 30+ coefficient is +46** for KTC and −12* for MFL at h = 1.

- **Dynasty value markets (KTC, FantasyCalc, DP, FP dynasty) pay for youth up front.** Young players and rookies trail their price for 1–2 years, and 30+ veterans beat theirs.
  - FP dynasty's age pricing is about neutral at 3 years and slightly under-prices youth at 5.
  - KTC and FantasyCalc still over-price ≤23 at 3 years. KTC also does at 5 years, but that is a single cohort.
  - **Contending teams can buy 30+ veterans from KTC/FC-priced managers.** They return about +23–43% more VORP over the next 1–2 seasons than equally valued players.
- **MFL keeper ADP is the mirror image.** It under-prices youth at every horizon (WR ≤24: +22% at 1 year, rising to +125% at 10 years) and over-prices age.

**Patterns that hold across markets and horizons:**
- **Coming off a ≤9-game season → underperforms**: −9% to −44%. The joint regression gives −12% to −38%, significant in most markets. All markets under-discount missed time.
- **Rushing QBs (≥25 rush yd/g) → outperform** in FP (+44% to +61%), MFL (+17% to +34%) and redraft (+39% to +49%). KTC is the exception (≈0).
- **TEs → underperform their dynasty price**: −10% to −19% in FP, KTC, FC and DP. They are about neutral in MFL.
- **RBs with <4 targets/game → underperform at 3+ years**: −11% to −20% in every dynasty market, and −33% at 10 years in MFL. In the dynasty value markets they are fine for 1 season.

**Mixed:**
- **Market fading last year's finish.** KTC and FC over-fade: those players beat their value by +14–25% for 1–3 years. MFL and FP redraft under-fade.

## 3. Is the KTC / FantasyCalc youth premium justified at 5–10 years? (`youth_premium.py`)

KTC has a single cohort with a 5-year outcome and none with a 10-year outcome, so it can't be tested directly. MFL keeper ADP has 5-year (11 cohorts) and 10-year (6 cohorts) history, so we use it as a bridge. The method has three parts:

1. **Justified premium**: in MFL's history, how much more or less than its MFL price each age band actually produced.
2. **Charged premium**: in the years both cover (2020–2024, veterans), how much more than MFL another market pays for that band. This is measured by mapping its ordering onto MFL's rank slots.
3. **Implied residual** = justified / charged − 1. Negative means the market overpays that band.

**Bridge check.** Where KTC, FC, DP and FP have their own 1–3-year outcomes, the implied residuals by age band track their actual residuals. Correlation is 0.83–0.97 at h = 1–2, with an average gap of 5–13 points. It weakens to 0.30–0.81 at h = 3.

| Implied residual % (KTC / FantasyCalc / FP dynasty) | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| Age ≤23 (veterans) | −18 / −14 / −11 | −5 / 0 / +2 | +9 / +14 / +17 | +32 / +36 / +40 |
| Age 30+ | +38 / +23 / +25 | +15 / +1 / +4 | −3 / −13 / −12 | +2 / −7 / −7 |
| WR ≤24 | −12 / −2 / −8 | +11 / +24 / +16 | +32 / +45 / +35 | +79 / +94 / +84 |
| RB ≤24 | −7 / −6 / −10 | −3 / −1 / −6 | +8 / +10 / +5 | −11 / −11 / −13 |
| TE ≤24 | −61 / −62 / −60 | −35 / −35 / −34 | −30 / −30 / −29 | −21 / −22 / −21 |
| QB ≤24 | −18 / −23 / +19 | −21 / −24 / +20 | −3 / −10 / +44 | |
| QB 25–27 | −38 / −36 / −8 | −29 / −29 / +3 | −34 / −33 / −6 | −37 / −36 / −14 |
| RB 28+ | +64 / +33 / +37 | +17 / −2 / 0 | −5 / −20 / −19 | −3 / −17 / −17 |

- **The youth premium breaks even at about 3 years. At 5–10 years it is too small, not too big**, but only for WRs.
  - KTC charges about +38% over MFL for WRs ≤24. Historically they produced +78% over 5 years and +134% over 10 years versus their MFL price.
- **Young RBs are priced about right, and young TEs are overpaid at every horizon.**
  - KTC, FC, DP and FP all charge +78–87% over MFL for TEs ≤24, but those players produced at most +26% over MFL even at 10 years.
- **KTC and FantasyCalc overpay QBs in 1QB value** (−29% to −38%). FP does not.
- **Old RBs and WRs are a 1–2-year buy and a 5+ year sell.**
- **Assumptions.** MFL's age pricing should be similar across eras. The ≤23 adjustment at h = 3 was +20% in 2010–15 and +37% in 2016–22, so if anything the youth edge has grown. The 10-year sum is undiscounted, so applying a discount rate shrinks the youth case. The bridge covers veterans only; rookies are excluded.
- Output: `outputs/youth_premium_bridge.csv`.

### 3b. Old RBs, WRs and TEs by finer age band

Dynasty value markets here are FP, KTC, FC and DP, 2020–2024; MFL covers 2010–2024. Cells show the mean residual %.

| | Dynasty value mkts h=1 / 2 / 3 / 5 | MFL h=1 / 3 / 5 / 10 | Bridge, KTC 28+ h=5 / 10 |
|---|---|---|---|
| RB 28–29 | +38* / +14 / −11 / −13 | −15 / −39* / −45* / −50* | −5 / −3 |
| RB 30–31 | +62 / +83 / +40 / – (n≈10) | −31* / −46* / −64* / −64* | |
| WR 28–29 | +25 / +21 / +10 / −9 | +7 / −4 / −19 / −28* | −7 / −11 |
| WR 30–31 | +50* / +27 / +33 / +2 | +1 / −22* / −40* / −26* | |
| WR 32+ | +21 / +3 / −39 / −53* | −14 / −33* / −46* / −44* | |
| TE 28–29 | −28 / −25 / −21 / – | −13 / −11 / −16 / −28* | −1 / −21 (TE 28+) |
| TE 30–31 | −3 / −2 / +7 / – | +10 / +2 / +8 / −14 | |

- **The old-RB averages are driven by a few seasons.** In KTC at h = 1, RBs 30+ average +76%, but the median is −24%. Without Henry 2023/24 and Aaron Jones 2024 the mean falls to −46%.
- **Old WRs are more broad-based.** Without the top 3, WRs 30–31 are still +25% at h = 1, and the median is +18% at h = 2.
- **Superflex (`outputs/sf/`) points the same way but is weaker**, because non-QBs are worth less.

## 4. An ATC-style blend for football (`blend_test.py`)

[ATC](https://fantasy.fangraphs.com/the-atc-projection-system/) weights each projection system per statistic by its historical accuracy. The dynasty analog weights value sources per horizon (and, with more data, per position).

**Equal-weight consensus and learned weights.** Learned weights are fit leave-one-cohort-out on 2020–2024, all players.

| Mean Spearman | h=1 | h=2 | h=3 |
|---|---|---|---|
| FP dynasty alone (best single source) | .513 | .583 | .592 |
| KTC alone | .480 | .551 | .580 |
| Equal-weight consensus of markets | .506 | .572 | .588 |
| Consensus + real-time stats model | .520 | .582 | .595 |
| Learned weights (FP, KTC, model) | .523 | .581 | .596 |

- **Gains over the best single source are small**: about +0.01 at 1 year and ≈0 at 2–3 years. The dynasty markets are highly correlated with one another.
- **The stats model gets 25–47% of the weight** even though it is the weakest source on its own. It is the only independent signal.
- **KTC gets about 0 weight at 1–2 years once FP is included.**

**Bias layer.** Real-time position × age × short-season multipliers applied to MFL ADP, learned only from cohorts complete before each snapshot. This raises Spearman by only +0.001 to +0.006, improving 9 of 12 cohorts at h = 1. The biases are real, but they shift value between groups more than they reorder players, so rank correlation barely moves. They matter more for trading at a fixed price than for ranking.

**Per-statistic ATC for redraft projections** would need archived per-stat projections from multiple sources. That kind of archive exists mainly through [Fantasy Football Analytics](https://www.r-bloggers.com/2017/03/who-has-the-best-fantasy-football-projections-2017-update/) and paid sites. Its own studies found averaging beats any single source, and that no source wins consistently.

## 5. Superflex (`LEAGUE=sf`, outputs in `outputs/sf/`)

**Value.** Superflex VORP uses the same lineup plus one SUPERFLEX slot (QB/RB/WR/TE). Replacement is set per season by filling 12 lineups greedily from season totals. The median replacement ranks are QB26, RB31, WR45 and TE14 (against QB13 in 1QB). Scoring is PPR with 4-point passing TDs. A 6-point passing TD would make QBs worth even more.

**Markets** (same names as in 1QB):

| Market | Source | Cohorts at h = 1 / 3 / 5 |
|---|---|---|
| fp_dynasty | FP superflex dynasty ECR (`dsf`, page `dynasty-op`), from Oct 2020 | 6 / 4 / 2 |
| dd_ktc, dd_fantasycalc, dd_dynastyprocess, ... | Dynasty Daddy `*_sf_trade_value`. KTC from 2021, FC 2022, DP 2023, others 2024 | KTC 5 / 3 / 1 |
| fp_redraft | FP superflex redraft ECR (`rsf`) | 5 at h = 1 |
| ffc_redraft | FFC 2QB ADP, 2014–2025 | 12 at h = 1 |
| mfl_dynasty | MFL keeper ADP, mixed formats | Bridge reference only. It badly under-prices QBs for superflex, since most of its leagues are 1QB. |

**Accuracy** (`outputs/sf/horizon_validation.csv`):
- FP superflex ECR is again the best dynasty source. It beats KTC superflex in 5/5, 4/4 and 3/3 shared cohorts at 1–3 years.
- **The stats model is about even with KTC superflex at 1–3 years**, which is better than it did in 1QB. QB production is stable from year to year, so box scores carry more signal.

**Who beats or trails superflex prices** (actual residual %, FP superflex / KTC superflex / FantasyCalc superflex):

| Group | h=1 | h=2 | h=3 | h=5 |
|---|---|---|---|---|
| All QBs | +27* / +17 / +16 | +28* / +15 / +14 | +26* / +12 / +14 | +34 / +11 / – |
| QB 28+ | +61* / +64* / +44* | +47* / +46* / +38 | +32* / +36* / +29 | +18 / – / – |
| Pocket QBs | +35* / +37* / +28* | +32* / +28* / +24 | +28* / +31* / +26 | +24 / – / – |
| QB ≤24 | −8 / −28 / −18 | 0 / −24 / −31 | +11 / −27 / – | +49 / – / – |
| Age ≤23 | −25* / −35* / −36* | −11 / −23* / −20* | −4 / −18* / −11 | +10 / −11 / – |
| Rookies | −22* / −39* / −38* | −9 / −27* / −28* | +6 / −17* / −4 | +26 / −4 / – |
| TE (all) | −16* / −20* / −22* | −17* / −20* / −22* | −15 / −17 / −20 | −12 / −12 / – |
| ≤9-game season | −10 / −9 / −7 | −15* / −18 / −17 | −11 / −7 / −8 | −13 / – / – |

**Bridge to 5–10 years** (`outputs/sf/youth_premium_bridge.csv`). The bridge check is strong here: implied vs actual correlation is 0.94–0.97 for FP at h = 1–5, and 0.88–0.95 for KTC at h = 1–3.

| Implied residual % (KTC / FC / FP) | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| QB 25–27 | +4 / +5 / +9 | +14 / +16 / +20 | +19 / +20 / +26 | +27 / +30 / +36 |
| QB 28+ | +34 / +26 / +28 | +35 / +27 / +29 | +41 / +31 / +33 | +58 / +48 / +51 |
| QB ≤24 | +3 / +3 / +18 | +4 / +3 / +19 | +19 / +16 / +37 | |
| WR ≤24 | −15 / −4 / −7 | +4 / +18 / +12 | +17 / +34 / +27 | +50 / +67 / +64 |
| RB ≤24 | −18 / −14 / −19 | −18 / −14 / −18 | −9 / −4 / −10 | −37 / −33 / −38 |
| RB 28+ | +19 / +6 / +8 | −16 / −26 / −23 | −36 / −43 / −41 | −37 / −43 / −42 |
| TE ≤24 | −65 / −64 / −61 | −47 / −45 / −41 | −45 / −44 / −40 | −42 / −41 / −36 |
| Age ≤23 | −20 / −15 / −12 | −12 / −7 / −5 | −2 / +4 / +7 | +3 / +10 / +14 |

**Takeaways.**
- **QBs are still under-priced in superflex markets, most of all veterans (28+).** They beat their price by about 30–60% at 1–3 years in every superflex market. The bridge puts it at +30–40% at 5 years and about +50% at 10 years. Superflex markets apply a skill-position age discount to QBs, but QBs keep their value much longer.
- **Pocket QBs beat their price as much as or more than rushing QBs.** The rushing-QB edge seen in 1QB mostly disappears once position is priced correctly.
- **In superflex, the youth premium is only about right at 5 years and only slightly too small at 10.** The pattern differs by position:
  - WR ≤24 is still a long-term buy (+17% to +34% at 5 years).
  - Young RBs and TEs are overpaid at every horizon.
  - Young QBs are roughly fair in KTC/FC and slightly cheap in FP.
- **KTC superflex overpays rookies through 3 years**: −17%* direct, and the joint regression gives −33%** at h = 3.
- **The caveats from 1QB still apply**: few cohorts (2020–2024), an undiscounted 10-year sum, and a bridge that covers veterans only. The superflex outcome also depends on the replacement-level assumption (QB26). If some teams start a non-QB in the superflex slot, fewer QBs start. The replacement QB is then better, and QB value somewhat lower. QB26 assumes nearly every team starts two.

## 6. Rookie draft picks (`picks.py`)

Dynasty Daddy's May snapshots list pick assets next to players (KTC 2021+, FantasyCalc 2022+, DP 2023+). Each pick's price is placed on the market's player-rank scale and compared with what it became: the rookies at those slots of a 12-team rookie draft, in the market's own rookie order (rounds 1–4). The price curve is fit on every ranked player, since 4ths sit below the top 200. A future pick scores 0 for seasons before its class arrives. Residuals use the same % scale as section 2b. Outputs: `picks_residuals.csv`, `picks_price_drift.csv`.

KTC, round level, 1QB / superflex:

| Pick | 1 yr | 2 yr | 3 yr | 5 yr (1 class) |
|---|---|---|---|---|
| Current-year 1st | −65 / −51 | −44 / −34 | −10 / −4 | +8 / +29 |
| Current-year 2nd | −48 / −42 | −33 / −27 | −16 / −20 | +16 / +2 |
| Current-year 3rd | −8 / −11 | +3 / −3 | +5 / +1 | +8 / +8 |
| Current-year 4th | +6 / 0 | +9 / +3 | +21 / +13 (about +8 / +2 without Puka Nacua) | −1 / −2 |
| Next year's 1st | −150 / −142 | −116 / −102 | −81 / −66 | +36 / +1 (2 classes) |

- **A current-year 1st–3rd is priced like the rookies it buys** (median price ratio 1.00–1.05), so it inherits the rookie pattern: behind its price for 1–2 years, about even at 3, ahead at 5 (one class only).
- **Seconds look the most overpriced**, behind in nearly every market and format through 3 years. Thirds are about fair.
- **Fourths are lottery tickets priced about right.** KTC prices a 4th like roughly the 260th player (about 5 points of 3-year VORP expected); FantasyCalc lists them at 0 and DP near 0. Only 7 of 33 rookies at 4th-round slots (2021–2023 classes) scored any VORP in 3 years, so most 4ths return nothing (13–29% beat their price at 1 year). The positive averages come from rare hits, mostly Puka Nacua (2023, rookie #46 in KTC). Price-drift ratios for 4ths are unreliable because the curve is near 0 that deep.
- **Tier splits (early/mid/late) are noise.** Each tier is 4 rookies a class; the early-1st +107% at 3 years in 1QB is one class (2023).
- **Future picks gain value as their draft approaches.** A 1st two years out cost 0.56× (1QB) / 0.67× (superflex) of the price its rookies carried once drafted; one year out, 0.87× / 0.93×. Most of the rise is in FantasyCalc (0.51–0.64×); KTC one year out is already about 1.0×.
- **For production inside 3 years, future picks lose almost every time** (0% of class-years beat their price for 1sts and 2nds). That is mostly the wait. Whether they pay off at 5–10 years can't be tested yet: only the 2022 and 2023 classes have 5-year windows.

## 7. Old vs young superflex QBs, old vs young 1QB WRs, and elite studs

Dynasty markets pooled (FP, KTC, FantasyCalc, DP; 2019–2024 snapshots). The 5-year column has 2 cohorts; "bridge" is `youth_premium.py`'s MFL-based estimate across the four markets.

| Superflex QB | 1 yr | 2 yr | 3 yr | 5 yr | Bridge 5 / 10 yr |
|---|---|---|---|---|---|
| 32+ | **+70** (6/6 cohorts +) | +44 | +29 | +29 | 28+: +31 to +41 / +48 to +58 |
| 28–31 | +44 | **+46** | **+41** | 0 | (in 28+) |
| 25–27 | +4 | +21 | +31 | **+43** | +17 to +26 / +27 to +36 |
| ≤24 | +7 | −10 | +6 | +53 (median 0) | +16 to +37 / +5 to +21 |
| Rookie | −35 | −12 | −33 | +6 | n/a |

| 1QB WR | 1 yr | 2 yr | 3 yr | 5 yr | Bridge 5 / 10 yr |
|---|---|---|---|---|---|
| 30+ | **+39** | **+23** | **+21** | −21 | 28+: −4 to −19 / −7 to −20 |
| 28–29 | +26 | +21 | +11 | −11 | (in 28+) |
| 26–27 | −13 | −18 | −11 | −8 | 25–27: +13 to +22 / +4 to +12 |
| 24–25 | +12 | +5 | −2 | −11 | (≤24 below) |
| ≤23 | −40 | +9 | +8 | +4 | ≤24: **+32 to +45 / +79 to +94** |
| Rookie | −29 | −27 | +6 | **+57** (median −13) | n/a |

Elite studs: top 3 at the position by market rank, 1QB / superflex.

| Top-3 at position | 1 yr | 2 yr | 3 yr | Who drives it |
|---|---|---|---|---|
| WR | +4 / +23 | +36 / +38 | **+42 / +48** (median +67 / +93) | Chase, Lamb, Adams, Hill hit; Kupp '21, Thomas '19 busted. MFL top-3 WRs beat price in 11 of 15 years. |
| QB | −6 / +14 | +14 / +19 | +35 / +22 | Almost all Josh Allen; without him −44 to −87. |
| RB | +9 / −5 | −11 / −26 | **−37 / −61** | Taylor, Barkley, Harris, Swift, Cook busted; Bijan, Gibbs hit. RB4–6 did better: +39 / +51 at 1–2 yr. |
| TE | **−44 / −34** | −46 / −30 | −70 / −55 | Kelce and McBride hit; Pitts, Andrews, LaPorta, Waller busted. |

Few players per elite tier (8–13), so single players swing the means. Position comparisons are more reliable than the levels.

## 8. Which RBs hold their value?

Residual % by trait. Dynasty = the four pooled markets in 1QB, 2019–2024; its 5-year column has 2 cohorts. MFL = keeper ADP 2010–2024, veterans only.

| RB trait | Dynasty 1 / 2 / 3 / 5 yr | MFL 1 / 3 / 5 / 10 yr |
|---|---|---|
| Priced ≥8 spots below last year's PPG rank ("discount") | **+22 / +18 / +16 / +8** | — |
| Priced ≥8 spots above PPG rank ("hype") | −17 / −5 / −19 / −40 | — |
| 4+ targets per game | +63 / +41 / +19 / −30 | +9 / −1 / −11 / −9 (best target band) |
| <2.5 targets per game | −7 to −20 | −16 to −23 |
| 15+ carries per game | −16 at 3 yr | −30 to −57 |
| Age ≤23 | negative | **+15 / +15 / +28 / +6** (9/11 cohorts + at 5 yr) |
| Age 28+ | — | −18 / −38 / −48 / −52 |
| 1st-round draft pick | +64 / +53 / +38 (median −18) / −16 | — |
| 2nd-round draft pick | +4 / −12 / −32 / −44 | — |
| Under 205 lb | +31 / +32 / +26 / −11 | 215+ lb is worse |
| Priced RB13–24 | +32 / +14 / +14 / −15 | −41 to −64 |
| Priced RB7–12 | collapses (−24, −42) | |

Joint OLS with player-clustered SEs, `r ~ age + targets + carries + ypc + weight + rd1 + log(price gap) + short season`:
- **Dynasty, 3 years:** only the price gap is significant: −44 per log unit of hype (p = .03).
- **MFL, 3 and 5 years:** age costs −6.5 and −9.4 points per year (p ≤ .006), and each carry per game costs −3.6 and −4.2 (p = .003).
- Targets, ypc, weight and draft round are not significant once age, workload and price gap are in the model.

Only 39% of top-24-priced RBs beat their price over 3 years in the dynasty markets, and 32% over 5 years in MFL.
- **Winners** were mostly bought before or just after a breakout, at RB10–22: Ekeler, Jacobs, Mixon, Aaron Jones, Kamara, early Henry and McCaffrey. Barkley in 2021, priced RB14 after an injury year, is another.
- **Losers** were second- and third-year backs priced in the top 12 off hype: Taylor '21, Javonte Williams, Akers, Dobbins, Edwards-Helaire, Harris. Backs aged 26+ priced top-6 also lost: Charles, Lynch, Rice, Peterson, Jones-Drew, Bell '17, D. Johnson.
- **Henry** is the one high-carry, low-target back who held value past 28.

## 9. Which injured players are worth buying?

**Definition.** "Injured" means a veteran who played ≤9 games in season t and was established: priced top-36 at the position, or ≥10 games and ≥10 PPG the year before. Pre-injury production is PPG in the short season (≥3 games), otherwise the prior season. Missed time can also be a suspension, a benching or a backup QB's few starts; the data can't tell these apart.

Residual %, dynasty pooled (1QB, 2019–2024) and MFL (2010–2024):

| Injured group | Dynasty 1 / 2 / 3 / 5 yr | MFL 1 / 3 / 5 / 10 yr |
|---|---|---|
| All injured | −12 / −30 / −17 / −11 (0 of 4–5 cohorts + at 2–3 yr) | −19 / −16 / −19 / −25 |
| All healthy (same filter) | +10 / +8 / +1 / −3 | +1 / 0 / 0 / −2 |
| Top-12 PPG at position before the injury | −23 / −10 / **+19 / +38** (64% beat at 5 yr) | −33 / −11 / −11 / −19 |
| PPG rank 13+ before the injury | −8 / −38 / −28 / −29 | −15 / −17 / −22 / −28 |
| Age ≤25 | −41 / −41 / −22 / +8 | −14 / **+7 / +15** / 0 |
| Age 29+ | +29 (median −30) / −28 / −5 / −25 | −21 / −35 / −42 / −42 |
| RB ≤25 | −82 / −54 / −35 / +21 | −46 / **+18 / +35** (57% beat at 5 yr) / +1 |
| RB top-12 PPG | +32 / +75 (n = 6) | −37 / +22 / −3 / −71 |
| QB ≤25 | −17 / −28 / −11 | +16 / +15 / +30 / +1 |
| WR (all) | −21 / −41 / −34 / −48 | −26 / −21 / −18 / −19 (healthy WRs +23 to +33) |
| TE (all) | −35 / −33 / −31 (0% beat at 1–3 yr) | −25 / −23 / −23 / −27 |
| Second short season in a row | −20 / −17 / −15 / −29 | −10 / **+2 / −1** / −16 |
| First short season after a healthy year | −8 / −37 / −18 / −7 | −24 / −26 / −29 / −31 |

Joint OLS among injured players, `r ~ age + position + log(market discount) + repeat injury + ≤4 games`, with player-clustered SEs:
- **MFL, 5 years:** age costs −6.2 points per year (p = .001), and a deeper market discount helps: +27 per log unit (p = .02).
- **Dynasty, 3 years:** the discount effect is +35 per log unit but not significant (p = .21).
- Repeat injury and the number of games missed are not significant in either market.

**Takeaways.**
- **Markets under-discount injury on average**, so most injured players are not worth buying at market price.
- **The exceptions are young, elite players, and the payoff is long term.** Players who were top-12 per game before the injury, and RBs or QBs aged ≤25, lose value for 1–2 years and then beat their price. Examples: McCaffrey '20 and '21, Mixon '20, Barkley '20, Keenan Allen '15 and '16, Julio Jones '13, Kupp '18, Dalvin Cook '17, Deebo Samuel '20.
- **Avoid injured players aged 29+, injured WRs who weren't elite, and injured TEs.** Examples: Michael Thomas '20, Kupp '22, Golladay '20, Julio '20, Gronkowski '16, Dez Bryant '15, Charles '15, Peterson '14.
- **"Injury-prone" players (a second short season) are not worse than first-time injuries.** In MFL they are about fair, because the market already marks them down. First-time injuries are the ones over-priced.
- Groups are small (6–40 players per cell). Treat the age and elite-production splits as the robust part.

## 10. In-season vs offseason prices (FP dynasty ECR)

**Method.** The repo has weekly FP dynasty scrapes from late 2019. For each season Y (2020–2024), six snapshots are scored on the same outcome: VORP in seasons Y+1 to Y+h.
- Snapshots: late August (preseason), mid-October (about week 6), mid-November (week 11), late December (week 16), mid-February and mid-May.
- Universe: veterans who played in season Y. Superflex uses FP's superflex ECR from October 2020.
- **Bias:** requiring a game in season Y flatters the August and October snapshots slightly, because players who later missed the whole season are dropped.

**Accuracy.** Ranking accuracy rises at every step in 5 of 5 cycles, in both 1QB and superflex. In 1QB, Spearman over 1 to 3 years gains versus August by:

| Snapshot | Gain vs August |
|---|---|
| October | +0.02 to +0.03 |
| November | +0.05 to +0.06 |
| December | +0.07 to +0.09 |
| February | +0.11 to +0.13 |
| May | +0.13 to +0.15 |

About half the gain comes during the season, a quarter between week 16 and February, and +0.015 from the draft and free agency.

**How biases move through the year.** Residual % at each snapshot's own price, 1QB; superflex is similar.

| Group | Aug | mid-Oct | mid-Nov | late Dec | mid-Feb | mid-May |
|---|---|---|---|---|---|---|
| Age ≤23, 3 yr | +13 | +7 | 0 | +1 | −6 | −14 |
| Age ≤23, 1 yr | −8 | −9 | −17 | −18 | −23 | −28 |
| First-year players, 2 yr | +27 | +23 | +8 | +8 | +1 | −5 |
| Age 30+, 1 yr | +4 | +5 | +9 | +13 | +21 | +23 |
| Age 30+, 3 yr | −9 | −5 | −1 | 0 | +11 | +14 |
| ≤9-game season, 1 yr | (−42) | (−37) | −29 | −24 | −20 | −21 |
| Rank rose ≥1.5× at position since Aug, 1 yr | | −23 | −22 | +2 | −7 | −15 |
| Rank rose ≥1.5× since Aug, 3 yr (superflex) | | −8 | −11 | −25 | −15 | −28 |
| Rank fell ≥1.5× since Aug, 2 yr | | −4 | −25 | −16 | +2 | −1 |
| QB / TE, 3 yr | +28 / −6 | +31 / −6 | +28 / −9 | +37 / −5 | +33 / −7 | +29 / −10 |

August and October injury cells use games that were not yet known; read them as a reference only.

**Takeaways.**
- **The youth premium is an offseason phenomenon.** Young players and rookies in their first season are fair or cheap from August to October, and most expensive in May. Old players are the reverse: about fair in August, cheapest in February to May.
  - Buy young players and rookies early in the season; sell them in the offseason.
  - Buy 30+ veterans in the offseason; their price recovers by preseason.
- **Mid-season breakouts are over-bought.** Players whose positional rank rose 1.5× or more since August trail their November price by −22% (1 yr), and are still over-priced in May.
- **Mid-season fallers keep falling.** Don't buy them in November (−17% to −25%). By February they are about fair.
- **Missed time is under-discounted at every date**, even after the season ends: −20% to −29% at 1 year.
- **Position biases are stable all year**: QBs are cheap and TEs expensive at every snapshot.

### 10a. Which source is most accurate in-season?

**Method.** `fetch_dynastydaddy.py` also saves `raw/dynastydaddy/inseason_*.json`: Sept 1, Oct 18, Nov 17 and Dec 22 of 2021–2024, and Feb 17 of the next year. `mr.dd_markets(panel, "inseason_*.json")` reads them.
- Each source is compared with the FP dynasty scrape taken within 10 days, on the same players.
- Universe: the union of every source's top 150 among veterans who played that season.
- Cells show mean Spearman minus FP, with snapshot-cycles where the source beat FP / total, pooled over h = 1–3.

| Source | 1QB preseason (Sep) | 1QB in-season (Oct–Dec) | 1QB offseason (Feb–May) | SF preseason (Sep) | SF in-season (Oct–Dec) | SF offseason (Feb–May) |
|---|---|---|---|---|---|---|
| KTC | −.005 (3/9) | **+.013 (18/27)** | −.014 (5/18) | −.008 (2/9) | +.003 (16/27) | −.007 (9/18) |
| FantasyCalc | −.026 (0/6) | +.009 (11/18) | −.008 (3/15) | +.010 (6/6) | **+.018 (14/18)** | +.001 (10/15) |
| DynastyProcess | −.005 (2/3) | +.008 (5/9) | −.001 (6/12) | +.002 (3/3) | +.008 (5/9) | .000 (5/12) |
| Dynasty Daddy ADP value | −.017 (0/3) | **−.088 (0/9)** | −.021 (1/6) | −.004 (0/3) | −.090 (0/9) | −.010 (1/6) |
| Dynasty Superflex | −.038 (0/3) | −.032 (0/9) | −.044 (0/6) | +.006 (3/3) | −.007 (4/9) | −.015 (1/6) |

- **FP is not the best in-season.**
  - In 1QB, KTC edges it at 2–3 years from October to December: +.024 at h = 2 in November (3/3 cycles), and +.031 to +.033 at h = 3 in November and December (2/2).
  - In superflex, FantasyCalc is the best in-season source: the best of FP/KTC/FC/DP in 6 of 9 comparisons.
  - The trade-based sources likely update faster than expert rankings during the season.
- **FP is still the best or tied in the offseason and preseason**, matching sections 2a and 5.
- **Dynasty Daddy's own ADP-based value goes stale in-season**: −.07 to −.18 by December. It is driven by startup-draft ADP, and most drafts happen in the offseason.
- All gaps are small (.01–.03), and cover 2–4 cycles per cell. The horizons overlap, so the win counts are not independent.

## 11. Contender playbook: buying RBs and TEs for the next 1–2 seasons

Residual % at 1 / 2 years; dynasty markets pooled (FP, KTC, FC, DP), 2019–2024, 1QB. Superflex results have the same signs but are somewhat smaller.

| RB group | 1 yr | 2 yr | Notes |
|---|---|---|---|
| PPG RB1–6 last year | **+90 / median +91** | **+72 / median +82** | 59–65% beat price; 5/6 and 4/5 cohorts |
| PPG RB7–12 last year | +70 | +33 | |
| Age 27.5+ and top-12 PPG | +45 (median +45) | +68 (median +72) | KTC: +66 / +100 (77% beat at 2 yr, 4/4 cohorts); FP: +27 / +45 |
| Age 27–28 | +57 (6/6 cohorts) | +22 | |
| 4+ targets per game | +63 | +41 (median +13) | |
| Priced RB4–6 | +39 | +51 (median +55) | |
| Priced RB1–3 | +9 (median −3) | −11 (median −33) | Elite price is not worth it |
| Under 25 and priced ≥6 spots above PPG | −28 (1/6 cohorts) | −16 | Sell these |
| Rookies | −35 (0/6 cohorts) | −3 | |

| TE group | 1 yr | 2 yr | Notes |
|---|---|---|---|
| All TEs | −18 (0/6 cohorts) | −18 (0/5) | Over-priced at every snapshot of the year (section 10) |
| Age 29+ | +9 (median −12) | +9 | Kelce, plus Kittle '22–'23; ex-best −13 / −14 |
| Age 27–28 | −31 | −30 (0/5 cohorts) | Waller '20, Andrews '21–'22, Kittle '19 |
| Under 25 and priced above PPG | −46 (0/6 cohorts) | −34 | Pitts, LaPorta, Kincaid |
| Priced TE1–3 | −44 (median −111) | −46 | |
| Priced TE13+ | −6 to −8 | −5 to −11 | Small absolute cost |
| ≤9-game season | −35 (0% beat) | −33 | |

**Playbook.**
- **RBs.** Buy last year's top-12 per-game producers, especially at 27–29, priced RB4–24 rather than RB1–3, and preferably pass-catchers.
  - Buy them from managers who use KTC or FantasyCalc, which mark these backs down most.
  - Buy in February to May: 30+ players are cheapest in the offseason, about +21% to +23% at 1 year (section 10).
  - Plan to hold for 1–2 seasons only. Old RBs collapse by year 3 (section 8; MFL is negative for old RBs at every horizon).
- **Pay with assets the market overvalues:** young hyped RBs and TEs, rookies in May, and current-year 2nd-round picks (section 6).
- **TEs.** No TE group beats its price reliably. Avoid paying for a TE1–3, a young breakout, or a 27–28-year-old.
  - The least-bad options are an established 29+ TE at a TE4–9 price (Kelce, Kittle), or a cheap TE13+ who costs little if he busts.
- **In-season:** don't buy the November breakout RB; mid-season risers trail their price (section 10).

## 12. Which positions to take with rookie picks?

Each class's rookies are ranked by market price and mapped to rookie-draft slots 1–48 (4 rounds × 12). Each rookie is compared with the average outcome at his slot (isotonic curve), as % of the average outcome of slots 1–48.
- **MFL keeper ADP** gives the rookie order for the 2011–2025 classes (5-year outcomes for 11 classes). Only the order within each class is used, so mixing startup and rookie drafts doesn't matter.
- **Dynasty markets (FP, KTC, FC, DP), pooled:** the 2020–2025 classes, 1–3 years only. n counts market rows.
- **Positions are draft-year positions** from MFL's yearly player lists (`raw/mfl_players/`). The panel uses current position, which put N'Keal Harry and Hakeem Butler among the TEs; 19 rookies were relabelled.
- **Medians are about −35 everywhere** because most rookies return nothing. Means are driven by hits.

1QB, MFL rookie order, mean residual % at 3 / 5 / 10 years (classes with a positive mean / total):

| Position | 3 yr | 5 yr | 10 yr |
|---|---|---|---|
| WR | +10 (6/13) | +10 (5/11) | +9 |
| RB | +1 (3/13) | −5 (3/11) | −16 |
| QB | −3 | +5 | −3 |
| TE | **−35 (2/13)** | **−24 (2/11)** | +17 (6 classes, late bloomers) |

| Group | 3 yr | 5 yr | Dynasty markets, 1 / 2 / 3 yr |
|---|---|---|---|
| RB, NFL 1st-rounder | **+115 (median +38)** | **+58 (median +28, 56% beat)** | −5 / **+221 / +143** |
| RB, NFL 2nd-rounder | −42 (median −60) | −30 | −46 / −48 / −68 (0/4 classes at 3 yr) |
| RB, NFL round 3+ | −3 | −8 | −21 / +1 / +7 |
| WR, NFL 1st-rounder | **+52 (8/13 classes)** | **+51** | +36 / −26 / +23 |
| WR, NFL 2nd-rounder | +12 | +20 | +14 / −8 / −30 |
| WR, NFL round 3+ | −10 | −13 | +4 / −1 / +25 |
| WR aged ≤21 at draft | +35 | +62 (median −13, 43% beat) | |
| TE in a 1st-round rookie slot | −142 (0/8 classes) | −64 | +70 / −18 / −184 |
| TE, NFL 1st-rounder | −100 | −59 | Pitts, Howard, Fant, Engram; Hockenson is the one hit |
| TE in a 4th-round rookie slot | −23 (5% beat) | −20 | |
| QB in a 1st-round rookie slot (1QB) | −27 | −47 | −28 / −14 / −88 |
| QB in a 4th-round rookie slot (1QB) | +46 | +49 | Hurts, Herbert |
| QB aged 22 / 23+ at draft | +51 / −33 | +63 / −31 | |

- **Joint OLS** (MFL; position × NFL round, rookie round and age; SE clustered by class). Relative to Day-3 WRs at 3 years:
  - WR NFL R1 +92 (p = .04), RB NFL R1 +164 (p = .09), TE NFL R1 −71 (p = .07).
  - Age: −9 per year older at 3 years (p = .09) and −14 at 5 years (p = .001).
- **Era split** (3 years, classes 2011–16 vs 2017–22):
  - TEs trail in both eras: −24 vs −47 (the other positions were −17 to −21 in the earlier era).
  - The WR edge is recent: −21 then +42. RBs went −17 to +16.

**How to use rookie picks (1QB):**
- **Default to WRs.** NFL first-rounders and young (≤21) WRs beat their slot most consistently. Day-2 WRs are fair to slightly positive.
- **Take an RB only if he has NFL first-round capital.** Those RBs are the one group with a positive median.
  - Day-2 RBs are priced like first-rounders but trail by 30–68% (Sankey, Ball, Akers, Kerryon Johnson). Chubb, Taylor and Mixon were the exceptions.
  - Older (23+) rookie RBs trail: −24 / −21.
- **Don't spend a 1st or 2nd on a TE.** First-round-slot TEs lost in every class. The TEs who paid off came cheap (Andrews, Kelce, Waller) and took years, so a late-round TE is only a long stash.
- **1QB QBs:** wait until rookie round 3–4, where QBs beat their slot (+46 to +67). First-round-slot QBs trail.
- **At equal price, take the younger player.**
- **Superflex:** the MFL order can't be used, since 1QB ADP prices QBs too low. In the dynasty superflex markets, rookie QBs are about fair overall (+1 / +23 / −35). QBs in 1st-round slots trail (−25 / −14 / −79; 4 classes). RB, WR and TE patterns match 1QB.
- **Timing:** these are prices around the rookie draft. Rookies get cheaper during their first season (section 10), and picks trail for 1–2 years (section 6), so a contender can also wait and buy these players in year one.

## Caveats

- **Position-level residuals depend on league format.** The QB results flip sign between FP (1QB) and MFL, whose keeper pool includes superflex leagues. Within-position groupings are more trustworthy.
- **Multiple comparisons.** Many groupings were tested, so prefer patterns that hold across markets and years over single starred cells.
- **Few cohorts in the dynasty value markets.** KTC, FantasyCalc and DP cover 2020–2024, so they overlap heavily. A single season (such as 30+ RBs Henry and Barkley in 2024) can drive the h=1 cells. Multi-year windows also overlap across cohorts.
- **Dynasty Daddy snapshots are a single date (May 16)**, after the NFL draft. Values before each source went live are placeholders and are dropped. 1QB runs use `*_trade_value`; `LEAGUE=sf` runs use `*_sf_trade_value`.
- **Current position.** The panel labels every player by his current position, and VORP for all his seasons uses that position's replacement level. A few converts (Waller, Logan Thomas, Cordarrelle Patterson) are mis-scored in early seasons. Section 12 groups rookies by draft-year position.
- **Unreliable rookie projections.** Rookie rows have low R² (about 0.13), so values like the 2026 rookie projections in `outputs/projections_after_2025.csv` should not be trusted.
- **The 2026 season is ignored.** It is in progress, and the current projections use the snapshot after 2025.
