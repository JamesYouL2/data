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
uv run fetch_markets.py         # MFL keeper ADP 2011-25, FFC PPR ADP 2013-25, DP values 2019-20 from git history
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
| `rookie_positions.py` | 12 (writes `outputs/rookie_position_residuals.csv`, used by 13 and 18) |
| `out_of_sample.py` | 13 |
| `te_premium.py` | 14 |
| `sell_timing.py` | 15 |
| `consolidation.py` | 16 |
| `second_year.py` | 17 |
| `opportunity.py` | 18 |
| `long_horizon.py` | 19 (also `LEAGUE=sf`) |
| `rb_long.py` | 20 (also `LEAGUE=sf`) |
| `startup.py` | 21 (also `LEAGUE=sf`; writes `outputs/startup_sim.csv`) |
| `startup_fp.py` | 22 (also `LEAGUE=sf`; writes `outputs/startup_fp_sim.csv`) |
| `team_signals.py` | 23 (also `LEAGUE=sf`; writes `outputs/team_signals.csv.gz`; `may`, `decay`, `splits`, `bye`, `persist` and `modes` arguments print only) |
| `rank_gaps.py` | 24 (also `LEAGUE=sf`; `REDRAFT=fp` for FP redraft ECR; prints only) |
| `redraft_age.py` | 25 (also `LEAGUE=sf`; prints only) |
| `gap_model_sim.py` | 26 (also `LEAGUE=sf`, `SOURCE=fantasycalc`; `check` and `ablate` print only; writes `outputs/gap_model_sim_<source>.csv.gz`) |
| `ros_low_sim.py` | 27 (also `LEAGUE=sf`, `SOURCE=fantasycalc`; `check` prints only; writes `outputs/ros_low_sim_<source>.csv.gz`) |
| `simple_rules_sim.py` | 28 (also `LEAGUE=sf`, `SOURCE=fantasycalc`; `table` prints the verdicts; writes `outputs/simple_rules_sim_<source>.csv.gz`) |
| `rule_audit.py` | 29 (also `LEAGUE=sf`; prints only) |
| `threshold_sim.py` | 30 (also `LEAGUE=sf`; `table` reprints; writes `outputs/threshold_sim.csv.gz`) |
| `sell_rules_sim.py` | 31, 33 (also `LEAGUE=sf`; `SIDE=buy` for buys; `table` reprints; writes `outputs/sell_rules_sim.csv.gz` or `buy_rules_sim.csv.gz`) |
| `sell_timing_sim.py` | 32 (also `LEAGUE=sf`; `table` reprints; writes `outputs/sell_timing_sim.csv.gz`) |
| `ruleset_sim.py` | 34, 35 (also `LEAGUE=sf`; `FINAL=1`, `GEN=1`, `REB=1` add the arms of section 35, `LOCO=0` drops the out-of-cohort ones, `OUT=` names the file; `table` reprints; writes `outputs/ruleset_sim.csv.gz`) |

`common.py` holds shared loaders. Sections 13 and 16 also run for superflex if `outputs/sf/` exists.

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
| fp_dynasty | FantasyPros dynasty ECR, mid-May (this repo; t = 2018 from `files/archives/fantasypros`) | 2018–2024 | 7 / 6 / 5 / 3 / 0 |
| mfl_dynasty | MFL keeper-league PPR ADP, veterans only | 2010–2024 | 15 / 14 / 13 / 11 / 6 |
| dd_ktc | KeepTradeCut 1QB value, via the Dynasty Daddy archive (May 16) | 2020–2024 | 5 / 4 / 3 / 1 / 0 |
| dd_fantasycalc | FantasyCalc 1QB value (Dynasty Daddy) | 2021–2024 | 4 / 3 / 2 / 0 / 0 |
| dd_dynastyprocess | DynastyProcess value (Dynasty Daddy) | 2022–2024 | 3 / 2 / 1 / 0 / 0 |
| dd_dynastysuperflex, dd_pfn, dd_dd_adp, dd_draftsharks | Dynasty Daddy | 2023–2024 | 1–2 at h = 1 |
| fp_redraft | FantasyPros redraft ECR, preseason (this repo) | 2019–2024 | h = 1 only, 6 |
| ffc_redraft | Fantasy Football Calculator 12-team PPR ADP, preseason | 2012–2024 | h = 1 only, 13 |

The Dynasty Daddy archive (`/api/v1/player/all/prev/{days}`) starts in May 2021 for KTC. FantasyCalc and DP values start later, in 2022–2023. Players matched to gsis ids at 99.1%. **10-year validation is possible only for MFL ADP and the stats model. 5-year validation is possible for FP dynasty (3 cohorts), KTC (1) and MFL (11).**

FP dynasty snapshots are the first scrape from May 10 to Aug 31; a year without one is skipped rather than filled with an in-season scrape. FP's superflex ECR starts in October 2020, so for `LEAGUE=sf` the t = 2018–19 cohorts use the FP 2QB ECR that DynastyProcess's `values-players.csv` carried (this repo's git history, May 27, 2019 and May 15, 2020). Before this rule, superflex t = 2019 used an Oct 16, 2020 scrape, five weeks into the season it was predicting.

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

### 6a. Who should sell or buy next year's picks? (`analysis/pick_sim.py`)

The league simulation behind the site's three modes (`analysis/middle_mode.py`), with next year's 1st-4th added: after season 1 each pick becomes the rookie the market ranked at that slot, in reverse order of the real season. One team trades one pick in May at its market rank, against no trade. Sellers take the player who most helps their projected lineup (teams projected 1st-8th) or the youngest (9th-12th); buyers give the player who costs their lineup least, or their oldest. 40 leagues a cohort; KTC has cohorts 2020-24 and FantasyCalc 2021-23, and the 3-year numbers rest on 3 and 2 of them. Ranges run across KTC and FantasyCalc, 1QB and superflex. Title odds in points; value is the sum of 340 − market rank over the roster.

| Team (projected rank) | Trade | Title odds, season 1 | Title odds, seasons 1-3 | Value 1 year on | Value 3 years on |
|---|---|---|---|---|---|
| Contender (1-3) | Sell 1st | +4.2 to +8.6 | +5.4 to +16.0 | −45 to −98 | −63 to −102 |
| Contender (1-3) | Sell 2nd | +1.9 to +3.7 | +1.7 to +4.3 | −43 to −94 | −52 to −108 |
| Middle (4-8) | Sell 1st | +4.4 to +9.8 | +2.5 to +10.8 | −48 to −106 | −100 to −125 |
| Middle (4-8) | Sell 2nd | +2.3 to +4.9 | +2.4 to +6.2 | −52 to −91 | −61 to −128 |
| Rebuilder (9-12) | Buy 1st | −0.9 to −1.2 | −0.8 to +1.7 | +35 to +89 | +68 to +138 |
| Rebuilder (9-12) | Buy 2nd | −0.6 to −0.7 | −0.9 to +0.2 | +57 to +131 | +84 to +114 |
| Rebuilder (9-12) | Buy 3rd | −0.3 to −0.6 | −1.0 to −0.2 | +85 to +116 | +64 to +99 |

- **A pick trade is titles against value, in every cohort.** Selling a pick raised season-1 title odds in every cohort and lowered value a year on in every cohort, for every kind of team; buying did the reverse.
- **Contenders and middle teams gain the most titles from selling a 1st.** About two thirds of that comes from choosing a player who fits the lineup: a random player at the same price added 1.5 to 3.4 points in season 1. A middle team gets about the same season-1 gain as a contender and less over three seasons.
- **Rebuilders buying picks give up almost no titles** (about 1 point in season 1 and nothing clear over three) and gain value at both dates in every cohort. Giving the oldest player at the price beat giving a random one for 2nds and 3rds.
- **A rebuilder selling its own 1st for a young player is unclear.** KTC shows +8 to +9 title points over three seasons, most of it from the 2022 cohort; FantasyCalc shows −0.7 to −1.8 and a value loss of 47 to 126.
- **3rds barely move titles** (under 2.5 points either way), but buying one still gained value.
- **No round is the one to get rid of.** Section 6 found current-year 2nds overpriced on production. For next year's picks, swapping one for a random player at its price shows no such gap between rounds. Title odds over seasons 2-3 moved +1.1 to +1.6 for a 1st and +0.5 to +0.8 for a 2nd at KTC prices, and between −1.5 and +0.8 for either at FantasyCalc prices. In the market's own value units, the roster was down a year on by 13-14% of the pick's price for a 1st, 21-25% for a 2nd and about 40% for a 3rd at KTC prices (27-30%, 31-34% and 67-80% three years on), and by 66-92% for every round at FantasyCalc prices, which lists next year's picks far below what their rookies are priced at once drafted.

Caveats: these are May trades, and the site is used in season; picks two or more years out aren't modelled; only one team trades; and value counts a rookie at his market rank the May he is drafted, when rookies are priced highest (section 10). Outputs: `pick_sim_<source>.csv.gz`.

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

## 13. Do the edges hold out of sample? (`out_of_sample.py`)

**Method: walk-forward.**
- For each test cohort T, each market's price curve is refit using only cohorts whose outcomes were known by then (t ≤ T − h). Each rule's past average is computed the same way.
- The rule is then scored on cohort T against that past-only curve. Pooled over test cohorts, this is what following the rule in real time would have returned.
- Dynasty markets give 5 test cohorts at 1 year (2020–24) and 3 at 2 years, since FP dynasty now starts at t = 2018. MFL gives 13 at 1 year and 9 at 3 years.

1QB, dynasty markets pooled, 1 year. Out-of-sample mean residual % (test cohorts positive / total):

| Rule | In-sample | Out of sample | |
|---|---|---|---|
| RB, top-6 PPG last year | +88 | **+117 (4/5)** | held |
| RB, 4+ targets per game | +59 | **+97 (4/5)** | held |
| RB priced RB4–6 | +34 | **+123 (3/3)** | held |
| RB 27.5+ and top-12 PPG | +43 | +86 (2/2) | held, few cohorts |
| Age 30+ (all positions) | +30 | **+38 (4/5)**; superflex +49 (5/5) | held |
| WR 30+ | +21 | +22 (3/5) | held, weak |
| All TEs | −17 | −17 (1/5) | held |
| TE under 25 | −36 | **−33 (0/5)** | held |
| Rookies (May) | −28 | **−28 (1/5)** | held |
| Age ≤23 | −30 | **−31 (1/5)** | held |
| RB under 25, priced 6+ spots above PPG | −27 | −22 | held |
| RB priced RB1–3 ("avoid") | +18 | +126 | **failed** (2024 Barkley, Henry, Gibbs) |
| TE 29+ | +11 | +2 | **no edge** |
| Injured (≤9 games), top-100 price | −18 | +1 at 1 yr, −52 at 2 yr | mixed |

- **Group residuals persist in general.** Across every market group, a group's past residual predicts its next cohort's (weighted correlation +0.46 at 1 year). Groups that beat their price by more than 10% went on to +24.
  - In MFL the correlation grows with horizon: +0.24 at 1 year, +0.38 / +0.49 / +0.55 at 2 / 3 / 5 years.
  - Superflex correlations are higher (+0.56 dynasty, +0.73 to +0.78 MFL).
- **MFL keeper ADP gives the opposite answer for age.** The out-of-sample signs flip:
  - Age 30+: −15 / −24 / −29 / −39 at 1 / 2 / 3 / 5 years (0/9 cohorts at 3 years).
  - Age ≤23: +16 / +33 / **+39 (9/9)** / **+57 (5/5)**.
  - RB 27.5+ and top-12 PPG: −88 at 1 year; WR 30+: −35 at 3 years (0/9).
- **This is a market difference, not an era difference.** On the same 2019–24 cohorts:
  - MFL has old top-12 RBs at −67 (1 year), while the dynasty markets have them at +45.
  - MFL has age ≤23 at +4 / +28 / +31 at 1 / 2 / 3 years, while the dynasty markets have −32 / −14 / −5.
  - Startup ADP prices old producers above the trade-value markets, and the outcomes fall between them. "Buy old, sell young" works against KTC/FC/DP/FP prices. It does not work against a startup ADP price.
- **Rookies (MFL rookie order, 3 years, by class):** WR NFL R1 +87 out of sample (5/8 classes; +225 at 5 years, 3/4). All TEs −37 (1/9). TEs in rookie R1–R2 slots −63 (2/7). RB NFL R2 +1 (−35 at 5 years).
  - The dynasty rookie order (2–4 test classes) is too short to judge.

**Takeaway.** The core within-market playbook survived out of sample:
- Buy proven RB producers and 30+ players for 1–2 years.
- Sell May rookies, under-24s and young TEs.
- Take WRs with NFL first-round capital.

The "avoid RB1–3" rule did not survive, so drop it. Use these edges only against dynasty trade values (KTC, FC, DP, FP), not startup ADP.

## 14. Does TE premium change the TE answer? (`te_premium.py`)

**Method.**
- TE seasons are rescored as PPR + 0.5 per reception, with replacement reset to TE13 under that scoring. Other positions are unchanged, and each market's price curve is refit.
- Prices are the standard 1QB prices: Dynasty Daddy stores no TEP values.
- TE premium adds about 40 points a season to a TE1–6 and 30 to a TE7–12, against about 9 for replacement.

| Group (dynasty markets, 2019–24) | PPR 1 / 2 / 3 yr | TEP 1 / 2 / 3 yr |
|---|---|---|
| All TEs | −18 / −18 / −14 | **−8 / −7 / −3** (still 0–1 positive cohorts of 4–6) |
| Priced TE1–3 | −44 / −46 / −70 | −13 / −17 / −44 |
| Priced TE4–12 | −29 / −27 / −29 | −17 / −16 / −18 |
| Priced TE13+ | −7 / −6 / +6 | −2 / 0 / +14 |
| Under 25 | −37 / −29 / −20 | −29 / −21 / −12 |
| 29+ | +9 / +21 / +11 | **+24 / +38 / +29** (5/6, 5/5, 4/4 cohorts) |
| Top-6 PPG last year | −5 / −18 / −16 | +16 / 0 / +5 |

- **MFL (2010–24) under TEP:** TEs turn slightly positive, +5 to +9 at every horizon (9–12 of 13–15 cohorts). Priced TE1–3 is +22 to +47.
- **Break-even TE price** (KTC/FC/DP raw values: scale every TE's value, re-rank, and refit):
  - In standard PPR, TE values would have to be about **20% lower** to break even (TE residual −7 to +5 at −20%).
  - In TEP the break-even is about **10% lower**. Priced TE1–12 need 10–20% lower.

**Takeaway.**
- TE premium halves the TE overpricing, but standard trade values are still slightly too high for TEs even in TEP leagues.
- If your league's calculator adds a TEP bump on top (as KTC's TEP setting does), TEs are more overpriced still.
- The groups that become buys in TEP are established producers: 29+ TEs and last year's top-6 by PPG.
- Young TEs remain overpriced.

## 15. The rebuilder's side: when to sell (`sell_timing.py`)

**Price retention.** This is the share of a top-100 player's trade value left 1 and 2 years later.
- Dynasty Daddy KTC/FC/DP values are taken as a share of each date's top-150 total.
- MFL ADP is mapped to price-implied value.
- A player who drops out of the market counts as 0.

| | KTC/FC/DP 1 yr / 2 yr (2020–24) | MFL 1 yr / 2 yr (2010–24) |
|---|---|---|
| RB ≤23 | 0.92 / 0.82 | 0.88 / 0.74 |
| RB 24–25 | 0.80 / 0.62 | 0.81–0.93 / 0.74 |
| RB 26 | 0.72 / 0.57 | 0.82 / 0.59 |
| RB 27 | 0.83 / 0.58 | **0.65 / 0.40** |
| RB 28 | **0.66 / 0.44** | 0.63 / 0.47 |
| RB 29+ | 0.75 / 0.69 | **0.55 / 0.24** (median player 0.36 / 0.00) |
| WR ≤25 | 0.95–0.98 / 0.84–0.92 | 0.92–1.13 / 0.92–1.00 |
| WR 26–27 | 0.83 / 0.60 | 0.88 / 0.76 |
| WR 28–29 | 0.75 / 0.55 | 0.80 / 0.54 |
| WR 30–31 | 0.70 / 0.42 | **0.65 / 0.39** |
| WR 32+ | 0.61 / 0.46 | 0.51 / 0.25 |
| TE 24–27 | 0.85–0.95 / 0.70–0.74 | 0.85–0.99 / 0.74–0.96 |
| TE 28–29 | 0.66 / 0.44 | 0.79 / 0.60 |
| QB 26–28 | 0.96 / 0.88 | 0.90 / 0.73 |
| QB 29–31 | 0.83 / 0.73 | 0.80 / 0.68 |

**Within-season price path** (KTC/FC/DP, cycles 2021–24). Value share at each date relative to May, top-100 players at May:

| | Sep 1 | Oct 18 | Nov 17 | Dec 22 | Feb 17 | next May |
|---|---|---|---|---|---|---|
| RB 26–28 | 1.00–1.03 | 0.88–0.97 | 0.92–0.96 | 0.82–0.94 | 0.72–0.85 | **0.64–0.77** |
| WR 28–29 | 0.98 | 0.98 | 0.92 | 0.88 | 0.84 | 0.77 |
| WR 30–31 | 0.99 | 1.05 | 1.01 | 0.92 | **0.76** | 0.70 |
| TE 28–29 | 1.00 | 0.85 | 0.87 | 0.83 | 0.82 | 0.74 |
| QB 29–31 | 0.97 | 0.84 | 0.77 | 0.80 | 0.75 | 0.73 |
| WR rookies | 0.98 | 1.01 | 1.02 | 1.05 | **1.09** | 1.06 |
| TE rookies | 1.01 | 1.11 | 1.19 | 1.24 | **1.26** | 1.25 |
| RB rookies | 1.03 | 1.06 | **1.09** | 1.06 | 1.01 | 0.94 |

**Is the market wrong about them?** Residual by age among top-100 players:
- Dynasty markets: RBs at 27 and 29+ are underpriced (+129 / +161 at 1 year), and so are WRs at 28–31 (+43 / +75). RBs at 24 are overpriced (−38 to −68).
- MFL ADP: the opposite again. RBs at 27+ are −45 to −90 at 3–5 years, and WRs at 30+ are −30 to −67.

**Takeaway for a rebuilder.**
- **Price is what a rebuilder cares about, and it falls fastest:**
  - RBs: from 26–28. About 30% is gone in a year and over 40% in two. In MFL, a 27+ RB loses a third to half a year.
  - WRs: from 28. About 25–30% a year, and over half in two years from 30.
  - TEs: at 28–29.
  - QBs hold until about 29.
- **Sell before the season (May to Sep 1).** Aging veterans keep their May price through Sep 1 and then decline.
  - For RBs and WRs the steepest part is after the season (Dec to May), when new rookies arrive and the player is a year older. A 30–31 WR goes from 0.92 in December to 0.70 the next May.
  - For 29+ QBs and 28–29 TEs most of the drop comes in the first half of the season (Sep to Nov).
  - Dynasty trade markets underprice these players, so selling an old RB or WR to a dynasty-value manager gets you a below-fair price for his production.
  - Sell where buyers pay ADP-style prices (startup-minded or win-now managers), or keep him if you are only one year away from contending.
- **Rookies rise through their first season.** WRs and TEs peak in February, at +9% and +26% over May. RBs peak in November.
  - Rebuilders should buy rookies in May and sell disappointing ones only after February.

## 16. Is consolidating 2-for-1 worth it? (`consolidation.py`)

**Method.**
- Using KTC/FC/DP raw values as a share of each cohort's top-150 total, fit a price curve (value on overall rank) and an outcome curve (h-year VORP on rank).
- For a stud at rank R, find the rank whose value is half of his (a pair) or a third (a trio), and compare the stud's expected VORP with 2× or 3× theirs.
- A ratio above 1 means the stud returns more VORP than an equal-raw-value package.

1QB, mean of the three markets:

| Stud rank | Pair rank | Stud / pair, 1 / 2 / 3 yr | Trio rank | Stud / trio, 1 / 2 / 3 yr |
|---|---|---|---|---|
| 1 | 29 | 1.14 / 1.29 / 1.30 | 50 | 1.52 / 1.67 / 1.89 |
| 3 | 35 | 1.56 / 1.50 / 1.99 | 72 | 1.72 / 1.79 / 1.98 |
| 6 | 53 | 1.41 / 1.50 / 1.88 | 93 | 2.24 / 2.26 / 2.43 |
| 12 | 70 | 1.36 / 1.72 / 1.98 | 106 | 2.32 / 2.23 / 2.16 |
| 24 | 88 | 1.53 / 1.53 / 1.69 | 136 | 2.00 / 1.93 / 2.32 |

- **It depends on the calculator.** Stud / pair at 2 years:
  - **KTC: 1.7–2.9.** FantasyCalc: 0.9–1.7, mostly 1.1–1.5. **DynastyProcess: 0.7–1.25** (pairs win from about rank 6 down).
- **VORP per unit of raw value** shows the same thing. Players ranked 100–200 return only 0.4 (KTC) or 0.6 (FC) of the VORP per value of ranks 25–48. In DP they return 2.5–3.4×, because DP's value curve is much steeper.
- **Superflex** shows the same pattern, slightly weaker: stud / pair 0.9–1.3 at 1 year and 1.1–1.7 at 3 years.

**Takeaway.**
- On KTC and FantasyCalc raw values, consolidating up is right. On KTC, two half-value players return only 35–60% of the stud's VORP, and on FantasyCalc 60–90%.
  - KTC's and FC's trade calculators add a "value adjustment" to the side getting the best player. These results say that adjustment is justified and often still too small at KTC.
  - So accept 2-for-1 deals for the stud even with the adjustment, and avoid taking the 2 side.
- On DP values, the extra consolidation premium is already in the curve. Studs ranked 6–48 are no better than two players at half their value.
- Before the VORP math, these results already favor the stud for two more reasons:
  - The second player of a pair may sit on your bench, and VORP counts every above-replacement point.
  - The 2 side also uses a roster spot.

## 17. Second-year players (`second_year.py`)

**Method.**
- Veterans one season into their career are split by rookie-year production: "hit" (PPG top-24 RB/WR, top-12 TE/QB, with 6+ games), "quiet" (6+ games, below that) or "barely played".
- They are compared with players entering year 3 and year 4+ with the same production split.

| | Dynasty markets, 1 / 2 / 3 yr | MFL, 1 / 2 / 3 / 5 yr |
|---|---|---|
| WR year 2, quiet | −35 (0/6) / −2 / +2 | **+17 / +36 / +43 / +66 (11/11)** |
| WR year 2, hit | −128 (n9) / +24 / +53 | **+46 / +155 / +171 / +231** (92% beat at 5 yr) |
| WR year 3, quiet | −1 / +9 / 0 | +11 / +12 / +25 / +44 |
| WR year 4+, quiet | +20 / +1 / −4 | +9 / +4 / +2 / −2 |
| RB year 2, quiet | −1 / +1 / −4 | −8 / +4 / +7 / +9 |
| RB year 2, hit | +22 / +32 / **−98** (median −112) | +23 / +10 / −14 / −1 (medians −60 to −94) |
| TE year 2, quiet | −26 / −23 / +11 | +22 / +19 / +34 / +35 |
| QB year 2, hit | small n | +18 / +44 / +67 / +76 |

- **By NFL draft round (dynasty markets, quiet year-2 WRs):**
  - NFL R1: −69 / −2 / −33.
  - NFL R2: −34 / −35 / −8.
  - **Day 3: +6 / +38 (5/5 cohorts) / +62.**
- **MFL, quiet year-2 WRs by round:** NFL R1 +47 / +46 / +80 at 2 / 3 / 5 years, and the other rounds +25 to +59.
- **Price move since the rookie snapshot** (dynasty markets): quiet WRs whose price fell 5+ spots are −29 / +13 / −21, against −37 / −16 / −3 for those whose price rose. There is no consistent difference, so the market isn't over-discounting the "quiet" label.
- **Names** (dynasty markets, quiet year 2, 2 years):
  - Best: Smith-Njigba '24, St. Brown '22, Chase Brown '24, Cook '23, Pittman '21.
  - Worst: Javonte Williams '22, Akers '21, Dobbins '21, Watson '23, Bateman '22, Burks '23.

**Takeaway.**
- **In 2010–24 ADP, second-year WRs were the best-value group in the data.** That held for hits and quiet ones alike, at every horizon, and mostly in year 2 rather than year 3+.
- **Today's dynasty markets price the year-2 WR leap.** Quiet year-2 WRs are fair or overpriced, especially NFL first-rounders, whom the market still pays for.
  - The remaining edge is quiet Day-3 WRs, which are cheap: +38 at 2 years in 5/5 cohorts.
- **Sell a year-2 RB after a hit rookie year** only at a 2–3 year horizon. The 1–2 year residual is positive, but by year 3 they trail by about 100% (median).
- **Quiet year-2 RBs and TEs** are about fair.

## 18. Landing spot and opportunity (`opportunity.py`)

**Measures, per team and season:**
- **Vacated:** the share of targets (WR/TE) or carries (RB) by players who are not on the team next season.
- **Incoming:** volume that arriving veterans had elsewhere.
- **Net:** vacated minus incoming.
- **QB1 left:** the team's most-attempts QB is gone next season.
- **Rookie competition:** R1–R2 rookies drafted at the same position.

The player's next-season team is used for veterans and the draft team for rookies. Median vacated share: targets 29%, carries 24%. The QB1 leaves in 27% of team-seasons.

**Lookahead caveat:** "left" is judged by who actually played for the team the next season. At a May snapshot nearly all departures are known, but a player lost to injury counts as having left.

| Group | Dynasty 1 / 2 yr | MFL 1 / 3 yr |
|---|---|---|
| RB, team's QB1 left | **−18 / −14** | **−30 (2/15) / −16** |
| RB, net vacated carries 15%+ | +36 / +25 | +10 / −2 |
| RB, net vacated ≤0 | +18 / +11 | −17 / −21 |
| TE, team's QB1 left | **−50 (0/6) / −41** | −11 / 0 |
| WR, team's QB1 left | +19 / +11 | +25 / +25 |
| WR, vacated <10% | −27 / −9 | −6 / +17 |
| WR, vacated 25%+ | +11 / +10 | +16 / +13 |
| QB moved teams | −18 / −13 | **−20 (2/15) / −16** |
| Moved teams, 1 yr (dynasty) / 3 yr (MFL) | WR +30, RB +29 | **WR −10, RB −27** |

- **Joint OLS** (residual on vacated, incoming, QB1 left, rookie competition, moved and price rank; SE clustered by player):
  - **RB, incoming carries:** −0.9 per point (p = .01, dynasty) and −0.5 (p = .01, MFL). The market under-reacts to a veteran RB being added.
  - **RB, QB1 left:** −57 (p < .01, dynasty) and −29 (p = .01, MFL).
  - **QB moved teams:** −42 (p < .01) and −20 (p = .01).
  - Moving teams at 3 years (MFL): WR −30 (p < .01), RB −23 (p = .02).
  - **Vacated targets for WRs:** +0.75 to +0.87 per point (p = .07 to .18, dynasty), and about 0 in MFL. This is mostly priced.
- **Rookies** (draft team):
  - WRs landing where net vacated targets are ≤0 trail: **−35 / −36** (MFL, 3 / 5 years) and −41 / −37 (dynasty, 1 / 2 years). WRs with 15%+ net vacated are +27 / +39 in MFL.
  - Incoming targets hurt rookie WRs: −1.0 to −1.9 per point (p = .03 to .14).
  - RB rookies with 15%+ net vacated carries: +13 / +45 in the dynasty markets; MFL is about 0.
  - QB rookies whose team's QB1 left: −8 / +32, against −24 / −20 when the QB1 stayed (dynasty markets).

**Takeaway.**
- **Fade RBs (and TEs) whose team loses its starting QB.** Also fade RBs whose team adds a veteran back. Both effects show up in both markets.
- **Fade QBs who change teams.**
- **WRs are the opposite of RBs on a QB change.** Their prices drop too far: +19 to +25 in both markets.
- Raw vacated targets are mostly priced for veterans. **For rookie WRs, avoid crowded landing spots** (net vacated ≤0), which trail by 35–40% in both markets.
- Veteran WRs or RBs who changed teams beat their price for 1–2 years in the dynasty markets but trailed at 3 years in MFL. Treat them as short-term buys.

## 19. How far back, and what holds at 5 and 10 years? (`long_horizon.py`)

**How far back the data goes.**
- **Prices are the limit, not stats.** nflverse stats go back to 1999, but the oldest market is MFL keeper ADP, and its API has no keeper drafts before 2011 (cohort t = 2010).
- **10-year outcomes:** cohorts 2010–15 only (6 cohorts, MFL only). These windows overlap heavily, and a walk-forward test is impossible because nothing can be trained before them.
- **5-year outcomes:** MFL 2010–20 (11 cohorts); FP dynasty 2018–20 (3 cohorts).
- **The dynasty markets** start in 2018 (FP, via the repo's weekly archive), 2020 (KTC), 2021 (FC) and 2022 (DP).
  - **No older FP data:** the archive's dynasty files begin in November 2018, so t = 2018 is as far back as FP goes.
  - **DP history is not a separate market:** DP values from 2019–22 in this repo's git history were computed from FP's ECR, so for 1QB they rank players the same way as FP (rank correlation 0.99 on May 2019). They are used only for superflex t = 2018–19; see section 2.
- **MFL's sample shrinks from 2020.** It drops from 3,500–7,300 drafts a year to about 1,000, so MFL before and after 2020 is not quite the same market.

**Translating MFL ADP into a dynasty-style price.**
- **Target:** on the overlap cohorts 2018–24, the dynasty consensus is the mean log overall rank across the dynasty markets live that year. A player outside a market's top 200 counts as 250.
- **Model:** a gradient-boosted model predicts it from log MFL rank, age, position, experience, last season's production and draft capital. MFL rank does most of the work; age is a distant second, then draft pick and experience.
- **Validation:** trained on 2018–21 and tested on 2022–24.
  - Rank correlation with the real dynasty price rises from 0.88–0.90 (raw MFL) to **0.94–0.95** (translated).
  - The translation reproduces the dynasty **youth premium** almost exactly: age ≤23 at 1 year is −40 translated vs −44 real; TE under 25 is −43 vs −44.
  - It captures only about **half of the dynasty discount on old players and elite RBs.** Age 30+ is +11 translated vs +21 real; RB top-6 PPG +26 vs +97; RB 27.5+ & top-12 −28 vs +37.
  - **So the translated results below are trustworthy for young players, and too pessimistic for old players and elite RBs.**

**Residual % for cohorts 2010–17 (before the overlap), against the translated price** (cohorts positive / total; raw MFL in brackets at 10 years):

| Group | 1 yr | 3 yr | 5 yr | 10 yr (2010–15) |
|---|---|---|---|---|
| Age ≤23 (all) | −4 (3/8) | +7 | +29 (4/8) | **+42 (5/6)** [raw +72] |
| Age 30+ (all) | +15 (5/8) | +3 | −4 | 0 (4/6) [raw −29] |
| WR 30+ | +27 (5/8) | +9 | −5 | −5 [raw −33] |
| RB top-6 PPG | +3 | −7 | −34 (2/8) | **−47 (2/6)** |
| RB 27.5+ & top-12 PPG | +25 (5/8) | −17 | −46 | −49 [raw −118] |
| RB priced 1–3 | −37 | −46 | **−86 (1/8)** | **−104 (0/6)** |
| RB 4+ targets/g | +33 (6/8) | +24 | +6 | −4 |
| TE under 25 | −29 (2/8) | −15 | −5 | −7 [raw +26] |
| TE 29+ | +27 (5/8) | +19 (6/8) | +7 | +7 (4/6) |
| Injured (≤9 games), top-100 price | −38 | −22 | **−29 (1/8)** | **−34 (0/6)** |

**Position by age, translated price, 1QB / superflex** (`LEAGUE=sf uv run analysis/long_horizon.py`; cohorts positive / total; FP's own 5-year residual for 2018–20 in the last column):

| Group | 5 yr (2010–17) | 10 yr (2010–15) | FP direct, 5 yr |
|---|---|---|---|
| QB ≤24 | −21 / −16 | −30 / −4 | +74 / +57 |
| QB 25–27 | −34 / +3 | −11 / **+56 (5/6)** | +33 / +11 |
| QB 28–31 | −14 / **+44 (7/8)** | +4 / **+91 (6/6)** | −18 / +6 |
| QB 32+ | −4 / **+65 (8/8)** | +11 / **+83 (6/6)** | +23 / +18 |
| WR ≤24 (veterans) | **+44 (8/8) / +31 (8/8)** | **+74 (6/6) / +48 (6/6)** | −8 / −4 |
| WR 25–27 | +11 / +9 | −4 / −12 | +6 / +1 |
| WR 30+ | −5 / −7 | −5 / −9 | **−42 / −36** |
| RB ≤24 (veterans) | +1 / −19 | −23 / **−37 (0/6)** | −18 / −25 |
| RB 28+ | −14 / −20 | −14 / −24 (0/6) | −33 / −21 |
| RB priced 1–3 | −86 / −98 | −104 / −150 | −41 / −76 |
| TE (all) | −4 / −13 (0/8) | −4 / −15 (0/6) | −8 / −5 |
| TE 25–28 | −11 / −17 | −11 / −21 (0/6) | −27 / −21 |
| TE 29+ | +7 / −4 | +7 / −4 | +44 / +33 (n = 14–15) |

- **Superflex validation:** the translation reaches 0.95 rank correlation with superflex dynasty prices (raw MFL 0.84–0.85, since MFL is mostly 1QB). It matches the real superflex residuals for QBs 28+ (+45 / +58 translated vs +39 / +60 real at 1 year), but overstates QBs 25–27 (+28 vs 0). Treat the 25–27 band as optimistic.
- **Superflex QB value assumes nearly every team starts two QBs** (replacement QB26; section 5). In 1QB, no QB group beats its price over 5–10 years.

**Direct check: FP dynasty's own 5-year residuals** (cohorts 2018–20, the first time a real dynasty market can be scored at 5 years; MFL on the same cohorts for contrast):

| Group | FP dynasty, 5 yr | MFL, same cohorts |
|---|---|---|
| Age ≤23 | **+15 (3/3)** | +41 (3/3) |
| Rookies (May) | **+36 (3/3)** | (veterans only) |
| Age 30+ | −6 (2/3) | −24 (0/3) |
| WR 30+ | **−42 (0/3)** | −46 (0/3) |
| TE 29+ (n = 14) | +44 (3/3) | +4 (3/3) |
| TE under 25 | −10 (0/3) | +6 |
| RB 4+ targets/g | −30 (0/3) | −12 (0/3) |
| RB priced 1–3 (n = 9) | −41 | +65 |
| Injured, top-100 price | −19 (0/3) | +40 |

- **These three cohorts overlap almost completely** (seasons 2019–25), so they amount to one long period, not three tests. Their direction matches the translation: young players and rookies beat FP's price over 5 years, after trailing it by about 30% at 1 year (section 13). The 30+ discount is about right at 5 years, and 30+ WRs fall short of their price.
- **MFL rookie order at 10 years** (classes 2011–16): this is too thin to use. Every group's median is negative, and the means are driven by a few hits. The only consistent signal is that the youngest rookies (≤21) beat their slot, +76 (5/6), while 23+ rookies trail, −11 (1/6).

**Takeaway.**
- **This confirms section 3's age-band bridge** with a player-level model and a direct 5-year FP check. Section 3 found age ≤23 at +32–40 and age 30+ at −7 to +2 over 10 years; here they are +42 and 0.
- **Horizon decides the youth question.** The dynasty markets' youth and rookie premium loses at 1–3 years (section 13). Over 5–10 years it is justified: young players beat dynasty prices by +15 (FP, 5 years) to +29 (translated, 5 years) and +42 (translated, 10 years), and May rookies beat FP by +36 at 5 years. Rebuilders should hold young players; contenders should sell them.
- **The edge on old players is a 1–2 year edge.** At 5–10 years the dynasty discount on 30+ players is about right; 30+ WRs fall short even of it, and MFL ADP overpriced old players by about 30%.
- **Elite RBs are short-term assets.** Even at a dynasty-style discount, RBs priced top-3 return far less than their price over 5–10 years (−86, −104). Because the translation understates the real dynasty discount, the true long-run gap is smaller, but the sign stays negative. This fits section 11: buy them to win now, not to hold.
- **Injured players priced top-100 trail at every horizon,** in the translation and in FP's own 5-year check. The discount for injury is never large enough.
- **Old TEs are a mild long-run buy and young TEs a mild fade,** shrinking from about ±27–29 at 1 year to about ±7 at 10.
- **Can MFL be regressed toward dynasty values? Yes.** The ranking part is easy (0.94 rank correlation), and the youth premium transfers well. What does not transfer is how hard dynasty traders discount age, so long-run conclusions about old players remain the least certain.

## 20. Are any RBs good long-term buys? (`rb_long.py`)

Translated MFL price (section 19) for cohorts 2010–17 (3 and 5 years) and 2010–15 (10 years), with FP dynasty's own 5-year residuals (2018–20) as a check. Residual % against the price, mean (cohorts positive / total):

| RB group | 1QB 5 yr | 1QB 10 yr | SF 10 yr | FP 5 yr (1QB) |
|---|---|---|---|---|
| All veteran RBs | −5 (3/8) | −16 (1/6) | −31 (0/6) | −19 (0/3) |
| Priced RB25+ | +2 (5/8) | −5 (1/6) | −12 (1/6) | −11 (0/3) |
| NFL R3+ / UDFA | −0 (5/8) | −4 (3/6) | −18 (0/6) | −13 (1/3) |
| 4+ targets/g | +6 (3/8) | −4 (1/6) | −40 (1/6) | −30 (0/3) |
| Priced 6+ spots below last year's PPG rank | −10 (2/8) | −10 (2/6) | −20 (1/6) | +8 (3/3) |
| ≤24, NFL R1 | +3 (3/8) | −35 (3/6) | −62 (2/6) | −24 (0/3) |
| ≤24, not yet a starter (PPG RB13–36) | −46 (2/8) | −66 (1/6) | −70 (1/6) | −18 (2/3) |
| NFL R2 | −26 (2/8) | −46 (1/6) | −66 (1/6) | −17 (1/3) |
| Priced RB1–3 | −86 (1/8) | −104 (0/6) | −150 (0/6) | −41 (1/3) |

**Takeaway.**
- **No RB group is a good long-term buy.** At 10 years every group trails its price in superflex, and the best 1QB groups only break even.
- **Long-run RB value rides on a few hits.** Medians are −15 to −90 in nearly every group; Ingram, McCoy, Bell and Murray carry the means. Only 14% of young non-starters beat their 10-year price.
- **Cheap RBs are fairly priced; expensive and high-draft-capital RBs are not.** The one hint of a buy is production-backed RBs priced below last year's output (+8 overall, +20 at ≤25, 3/3 in FP), but the translation disagrees (−10, −20) and the FP medians are negative.
- Use RBs as 1–2 year assets (sections 11 and 13) and sell before 27–28.

## 21. Startup drafts: all in, punt, or something else? (`startup.py`)

A simulation on the real MFL startup boards (keeper ADP, veterans only, top 200) for drafts after seasons 2010–20, 25 drafts each. 12-team snake, 16 rounds. Eleven teams draft by ADP with log-normal noise (sd 0.2 on rank). The focal team, in a rotating slot, follows a strategy. Every team must fill QB 1 (SF 2), RB 3, WR 4 and TE 1 by the last round. Each season after the draft scores the best lineup from real season PPR totals (QB, 2 RB, 3 WR, TE, FLEX; SF adds a QB/RB/WR/TE slot). A "title" is the top score of the 12 teams. Because the other teams draft with noise, the ADP control team wins about 20% of seasons rather than 8.3%; compare strategies with it, not with chance.

Expected titles per startup, years 1–3 / years 1–5:

| Strategy | 1QB | SF |
|---|---|---|
| Pure ADP | 0.51 / 0.85 | 0.56 / 0.90 |
| Win-now: best last-season PPG among the next 8 | 0.16 / 0.21 | 0.24 / 0.33 |
| Oldest among the next 8 | 0.04 / 0.05 | 0.09 / 0.11 |
| Youngest among the next 8 | 0.65 / 1.45 | 0.59 / 1.34 |
| Youngest, skipping RBs and TEs until forced | 0.39 / 1.04 | 0.45 / 1.24 |
| Adjusted rank: age ×1.06/yr, RB ×1.1 | **0.83 / 1.72** | 0.85 / 1.65 |
| Same, plus QB ×0.86 | 0.78 / 1.57 | 0.94 / 1.85 |
| Age ×1.06, RB ×1.1, injured ×1.16 | **0.91 / 1.88** | 0.91 / 1.80 |
| Same, plus QB ×0.86 | 0.93 / 1.79 | **1.01 / 1.95** |
| Age ×1.03, RB ×1.1, injured ×1.16 (plus QB ×0.86 in SF) | 0.80 / 1.51 | 0.89 / 1.66 |

The adjusted rank is ADP rank × 1.06 per year older than 26 (÷ 1.06 per year younger) × the position factors; the team takes the lowest adjusted rank on the board. From a grid search (`STRATS='adj:y=..,rb=..,te=..,qb=..,wr=..'`, log-scale factors):
- **Age** ×1.03–1.10 per year and **RB** ×1.1–1.2 all land close to the best. ×1.5 on RBs hurts years 1–3, and skipping RBs outright is worse still.
- **TE** and **WR** factors are neutral to negative. The age factor already captures the young-WR edge.
- **QB:** ×0.82–0.9 adds about 0.13 titles in years 1–3 in SF at every noise level, and is better than no QB factor in 8 of 11 drafts. In 1QB it is neutral to negative, and fading QBs hurts in both formats.
- **The best setting is robust:** it beats ADP at noise 0.1, 0.2 and 0.3, and for years 1–3 in 9 of 11 drafts in 1QB (SF: 8 better, 2 tied, 1 worse). Year 1 is unchanged (about 0.20 titles), so it costs nothing now.
- **Injury** (missed 8+ games last season, `inj=0.15`, rank ×1.16) helps in both formats at noise 0.1, 0.2 and 0.3; anything from ×1.1 to ×1.6 does about as well. Other flags tested were neutral (priced above production, NFL round, receiving RB, rushing QB, age 30+, new team) or hurt (priced below production, last season's top 12). A year-2 bump is weak and mostly repeats the age factor.

**Over two seasons only** (expected titles in year 1 / years 1–2 / years 1–3, noise 0.2):

| Lean | 1QB | SF |
|---|---|---|
| Pure ADP | 0.20 / 0.37 / 0.51 | 0.20 / 0.39 / 0.56 |
| Age ×1.06 only | 0.19 / 0.45 / 0.76 | 0.18 / 0.40 / 0.68 |
| RB ×1.1 only | 0.20 / 0.37 / 0.53 | 0.23 / 0.44 / 0.65 |
| Injured ×1.16 only | 0.20 / 0.39 / 0.56 | 0.22 / 0.40 / 0.61 |
| QB ×0.86 only | — | 0.22 / 0.42 / 0.61 |
| All, age ×1.06 | 0.23 / 0.55 / 0.91 | 0.26 / 0.60 / 1.01 |
| All, age ×1.03 | 0.23 / 0.49 / 0.80 | 0.29 / 0.57 / 0.89 |

Each factor alone is small over two years; together they clearly beat ADP and never lose year 1. For a 2-year window use age ×1.03, for 3–5 years ×1.06.

**Takeaway.**
- **Neither all in nor punt.** Draft close to ADP, lean young, tax RBs and last year's injured players a little. In superflex, move QBs up.
- **Win-now reaches lose even in year 1.** Last season's production is already priced and regresses, and every reach gives up ADP value.
- **This is consistent with section 19:** startup ADP overprices old players relative to trade markets. Contend by trading for veterans after the startup.

**Limits.**
- No trades, waivers or rookie drafts after the startup, so rival rosters only age. That inflates youth from about year 4 on; years 1–3 are the fair comparison.
- Rookies are not in the pool.
- "Title" means the top regular-season score.
- The factors were chosen on the same drafts.
- 2010–20 startup ADP priced youth and SF QBs less than current startups do, so the edges are probably smaller now.

## 22. Does the startup lean add anything on top of FP dynasty ECR? (`startup_fp.py`)

The same simulation on drafts after seasons 2018–24, where FP dynasty ECR exists (60 drafts each). The pool is the veterans in MFL's or FP's top 200, each board re-ranked within it. Two kinds of room: the other eleven teams draft by noisy MFL ADP, or by noisy FP ECR. The focal team uses ADP, FP, or ADP blended with 10% or 20% FP (B10, B20: average of log ranks), each raw or with the section 21 lean (age ×1.03 or ×1.06, RB ×1.1, injured ×1.16, SF QB ×0.86). Only drafts with all seasons played count: 6 drafts for years 1–2, 5 for years 1–3.

**FP already applies the lean.** Relative to ADP, FP ranks a player ×1.07 lower per year of age (SF ×1.08) and RBs ×1.45 lower (SF ×1.42). In superflex it ranks QBs ×0.29, far higher than ADP. Injured players it ranks ×0.93 (slightly higher), so that is the one piece of the lean FP does not have.

Expected titles in years 1–3, noise 0.2 (noise 0.35 in brackets):

| Board | 1QB, ADP room | 1QB, FP room | SF, ADP room | SF, FP room |
|---|---|---|---|---|
| ADP raw | 0.61 (0.94) | 0.57 (0.90) | 0.65 (1.02) | 0.40 (0.66) |
| ADP + lean | 0.87–0.89 (1.24–1.27) | 0.77–0.86 (1.06–1.09) | **1.05–1.11 (1.41–1.44)** | 0.42–0.48 (0.66–0.68) |
| B10 raw | 0.74 (1.05) | 0.59 (0.91) | 0.83 (1.25) | 0.42 (0.67) |
| B20 raw | 0.82 (1.16) | 0.64 (1.00) | 0.76 (1.22) | 0.43 (0.80) |
| B10 + lean | **0.92–0.96 (1.30–1.31)** | **0.79–0.81 (1.10–1.15)** | 0.87–0.97 (1.41–1.44) | 0.49–0.52 (0.71–0.72) |
| B20 + lean | 0.91–0.95 (1.20–1.25) | 0.76–0.77 (1.04–1.18) | 0.81–0.83 (1.25–1.27) | 0.49–0.55 (0.75–0.79) |
| FP raw | 0.27 (0.57) | 0.60 (0.98) | 0.41 (0.61) | **0.66 (1.05)** |
| FP + lean | 0.27–0.33 (0.50–0.56) | 0.45–0.54 (0.80–0.94) | 0.34–0.35 (0.60–0.63) | 0.51–0.54 (0.83–0.93) |

Single pieces on the FP board in an FP room (years 1–3, noise 0.2 / 0.35): youth ×1.03 hurts in every setting (1QB 0.52 / 0.89 vs 0.60 / 0.98 raw). RB ×1.1 and SF QB moves are neutral to negative. Injury ×1.16 is the only piece positive in all four settings, by +0.03 to +0.06. Leaning slightly *older* than FP helps year 1 and ties over years 1–3.

**How much FP to blend in** (1QB, ADP room, years 1–3, noise 0.2; B5 = 5% FP):

| Your board | ADP | B5 | B10 | B20 | B33 |
|---|---|---|---|---|---|
| raw | 0.61 | 0.66 | 0.74 | 0.82 | 0.72 |
| injury lean only | 0.60 | 0.67 | 0.78 | 0.84 | 0.77 |
| half lean (age ×1.03, RB ×1.05, injured ×1.16) | 0.87 | 0.93 | 0.93 | 0.94 | 0.79 |
| lean, age ×1.03 | 0.87 | **0.94** | 0.92 | 0.91 | 0.78 |
| lean, age ×1.06 | 0.89 | 0.93 | **0.96** | 0.95 | 0.80 |

- **In 1QB**, B5 plus a lean beats ADP plus the same lean in all 12 room × noise × lean combinations, by +0.01 to +0.07. All 12 use the same 5 drafts, so this is one weak signal, not 12.
- **In superflex**, B5 plus the lean is −0.07 to +0.06 against ADP plus the lean; 10% FP or more hurts in an ADP room.
- **33% FP or more overshoots** in both formats.
- **The blend and the lean are mostly the same information:** with a blend, a half-size lean does as well as the full lean, and B20 with only the injury lean (0.84) nearly matches ADP plus the full lean.

**What FP adds over ADP is structural.** It is almost all an age curve, an RB tax, a superflex QB bump and a softer view of injuries:
- ADP with FP's implied tilt applied (the multipliers above) beats FP itself in an FP room: 0.78–0.79 vs 0.60 in 1QB, 0.78 vs 0.66 in superflex. Removing FP's player-by-player views did not hurt.
- Once the lean is in place, those player-by-player views add about 0.05 titles at most (the B5–B10 gain).
- FP's tilt is the right direction but too strong: ×1.07 per year and ×1.45 on RBs, where the simulation does best at ×1.03–1.06 and ×1.1. That is why FP raw trails ADP plus the lean even though FP contains the lean.
- This is about veterans in a startup. Rookies and second-year players, where expert views could matter most, are not tested, and sections 10 and 13 found specific trade-market mispricings that a formula on ADP would not capture.

**Takeaway.**
- **Don't lean on top of FP ECR.** It already contains the youth and RB adjustments, more strongly than the lean. Adding them counts the same thing twice. At most, rank last year's injured players a little lower.
- **Start from the board your room drafts from.** A board far from the room's means reaching for players nobody else wants yet while the room takes the ones it likes: FP raw in an ADP room is the worst row (0.27 in 1QB).
- **Real startup rooms draft near ADP** (ADP is what they draft). There, ADP plus the lean is the rule. In 1QB a 5–10% FP blend plus the lean is a little better in every setting, by about +0.01 to +0.07 titles, within the noise. In superflex, don't blend.
- **B10 or B20 raw** (a blend without the lean) beats raw ADP and gets part of the lean's gain, because FP's ranks lean young. It still trails ADP plus the lean over years 1–3 in an ADP room, by 0.05–0.2 titles in 1QB and 0.15–0.35 in superflex. It is the simpler option, not the better one.
- **In a room drafting off FP or another dynasty consensus:** take that board raw in superflex. In 1QB, ADP plus the lean did better (0.77–0.86 vs 0.60), but most of that gap is the 2018 draft, whose FP ranks come from archives. Without 2018 it is a tie.

**Limits.**
- 5–6 drafts: treat differences under about 0.05 titles as noise.
- Veterans only, as in section 21, so FP's rookie ranks are not tested.
- MFL's superflex ADP may be a weak superflex board, which flatters FP's QB pricing in the SF columns.

## 23. Experiments: record, this-season strength or dynasty value? (`analysis/team_signals.py`)

Experiments, not used by the site. Three runs in the league simulation of section 6a (12 teams draft the FP top 340, real PPR scoring, rosters frozen, FP prices). Title odds in points; 1QB / superflex.

**In season: which measure says to buy?** After week 4, 8 or 12 each team has a record (its first weeks replayed six times with the title model's weekly noise), a projected rest-of-season lineup (from FP rest-of-season ranks) and a dynasty value (sum of 340 − FP rank). The table is the change per 1 SD of each, the other two held fixed. The win-now trade is up to three swaps, each for the player who most raises the projected lineup, priced level with or up to 20% below the one given.

| Outcome | Record | Rest-of-season strength | Dynasty value |
|---|---|---|---|
| Title odds this season | +4.0 / +4.0 | +4.7 / +5.0 | +0.6 / +0.8 |
| Title odds, next two seasons | +0.5 / +1.0 | +4.8 / +7.1 | +1.5 / +2.2 |
| Win-now trade's gain, this season | +2.9 / +3.0 | +0.1 / +0.6 | +0.4 / +1.0 |
| Win-now trade's gain, three seasons | +3.6 / +2.5 | −1.1 / +1.1 | +2.1 / +2.0 |

- **The gain from buying follows the record.** In 1QB, top-third records gained about 10 points this season at every level of rest-of-season strength; bottom-third records gained 3 to 5. Positive in every cohort.
- **Record counts for more late:** +2.8 on this season's odds after week 4, +5.4 after week 12. Strength stays at +4.3 to +5.1.
- **Record says little about later seasons;** rest-of-season strength says the most.
- **The trade here has no cost:** it also gained titles in later seasons and lost no dynasty value, as any pick of the best player in a price window does in this simulation. The run shows who gains from buying, not what buying costs.

**In May: strength this season or dynasty value, by season** (`may`). Projected lineup points from redraft ADP, and dynasty value two ways: the rank sum above, and a top-heavy sum of exp(−rank / 60), which counts the #1 player about 2.7 times the #60 and 7 times the #120 (the rank sum: 1.2 and 1.5 times). Each pair of columns is one model.

| Season | Strength | Top-heavy value | Strength | Rank-sum value |
|---|---|---|---|---|
| 1 | +4.2 / +4.0 | +1.2 / +0.8 | +4.6 / +4.1 | +0.7 / +1.0 |
| 2 | +2.7 / +2.5 | +1.7 / +1.6 | +3.2 / +2.9 | +1.3 / +1.3 |
| 3 | +2.0 / +1.8 | +1.8 / +1.9 | +2.6 / +2.4 | +1.1 / +1.5 |
| 4 | +1.8 / +1.4 | +2.4 / +2.2 | +2.6 / +2.1 | +1.4 / +1.5 |
| 5 | +1.6 / +0.7 | +2.3 / +2.6 | +2.5 / +1.7 | +0.9 / +1.3 |

- **This season's strength is the better guide through season 2.** Top-heavy value draws level at season 3 and leads from season 4 (3 of 4 cohorts in both formats). The rank sum never leads.
- **Value adds most for strong teams:** in the top third by strength, bottom-third to top-third value took title odds over seasons 2–3 from about 17 to 27.
- Frozen rosters understate value, which can't be traded or turned into rookies here. Seasons 4 and 5 have 4 and 3 cohorts.
- **The decay of 60 is fitted to titles** (`decay`). Value's correlation with title odds over seasons 2–5 peaks at a decay of 45–60 in 1QB (0.26, against 0.15 for the rank sum) and 60–90 in superflex (0.25–0.26, against 0.17), and is flat across that range. Trade markets are less top-heavy: an exponential fitted to May values gives a decay of 73–99 for FantasyCalc and 106–196 for KTC, rising each year.

**A mediocre, old team with a first-round bye** (`bye`). After week 12, teams seeded 1st or 2nd on their record, in the middle third by rest-of-season strength and the oldest third by where their value sits (1QB / superflex):

| | Title odds this season | Next two seasons | Value next May |
|---|---|---|---|
| No trade (title odds) | 15.7 / 16.0 | 20.6 / 14.6 | |
| Buy: the win-now trade | +10.6 / +11.5 | +8.3 / +9.5 | −3 / +19 |
| Sell old: the oldest players for the youngest at their price | −3.3 / −3.0 | −4.6 / −0.4 | +55 / +42 |

- **Buy.** It raised this season's title odds in every cohort; selling old players lowered them in every cohort, and in 1QB over the next two seasons as well.
- **The bye is worth about 6 points to this team** (9.5 as a 3rd to 6th seed) and buying adds more with it (+10.6) than without (+8.8). The age of the roster barely moves the buy's gain (+10 to +11 for old, middling and young).
- The buy here costs nothing, as above, so this says to use the bye, not what a real buy should cost.

**What the value is made of** (`splits`). Shares of a roster's top-heavy value, per 1 SD, with strength this season and total top-heavy value held fixed (1QB / superflex):

| Share of value in | Season 1 | Seasons 2–3 | Seasons 4–5 |
|---|---|---|---|
| Players 25 or younger | −0.1 / −0.1 | +1.7 / +1.1 | +4.9 / +4.4 |
| The best three players | −0.8 / −0.5 | −1.2 / −0.8 | −1.3 / −1.3 |
| RBs | +0.2 / +0.5 | −1.1 / −0.1 | −0.6 / −0.2 |
| WRs | +0.2 / −0.1 | +1.7 / +0.7 | +0.5 / +0.3 |
| TEs | −0.3 / 0.0 | −1.2 / +0.1 | +0.4 / +0.6 |
| QBs (superflex) | −0.2 | −0.4 | −0.1 |

- **Age is the one that matters.** Young value does nothing in season 1 and is the largest effect in the run by seasons 4–5 (3 of 3 cohorts, both formats; thirds of the share: 12 / 18 / 21 title points in 1QB, 13 / 16 / 22 in superflex).
- **Concentration adds nothing on top of top-heavy value**, and leans negative: at the same total, the roster with more of it in three players did slightly worse (clear of zero only in season 1, and in seasons 2–3 in 1QB). The measure already credits stars; this doesn't test a 2-for-1 at market prices (section 16).
- **Position is small.** In 1QB, WR-heavy value gained and RB- or TE-heavy value lost about 1 to 2 points over seasons 2–3 (4 of 5 cohorts each); in superflex nothing is clear, and nothing is by seasons 4–5.

**Does a top-heavy measure change the modes' value results?** (`modes`) The trades of section 6a's modes, value against no trades as a % of the roster, 3 years on (4 cohorts):

| Mode | Rank sum | Top-heavy |
|---|---|---|
| Rebuilding | +33% to +38% / +25% to +29% | +53% to +55% / +49% to +55% |
| Default | +24% to +26% / +22% to +26% | +32% to +34% / +33% to +37% |
| Contending | +13% to +15% / +5% to +6% | +16% to +20% / +6% to +9% |
| Random swaps | −1% to +1% | −2% to +1% |

- **No rebuilder move changes sign, and the order of the modes is the same.** The rebuilder calls gain more under the top-heavy measure, in every cohort at both dates.
- **Rebuilders buying picks (section 6a) still gain** in the market's own value units, which are top-heavy: +15% to +123% of the pick's price a year on across rounds, sources and formats.
- Superflex contending is the weak one: positive in 4 of 6 cohorts a year on and 2 to 3 of 4 at three years, top-heavy.

**A middle team with the league's best dynasty value.** Teams projected 4th–8th this season and 1st–2nd in top-heavy value: title odds with no trades, and each mode's change (1QB / superflex; seasons 4–5 have 3 cohorts).

| | Season 1 | Seasons 2–3 | Seasons 4–5 | Top-heavy value 3 years on |
|---|---|---|---|---|
| No trades (title odds) | 9.8 / 10.4 | 23.7 / 27.3 | 19.0 / 28.7 | |
| Contending | +10.6 / +7.1 | +10.5 / −2.1 | +8.3 / +1.6 | +19% / +4% |
| Default | +9.1 / +2.2 | +13.9 / +8.7 | +20.8 / +20.5 | +30% / +31% |
| Rebuilding | −3.2 / −5.2 | +12.3 / +12.5 | +36.1 / +35.7 | +48% / +42% |
| Random swaps | +0.8 / +0.2 | −0.7 / −3.8 | +0.9 / −1.9 | 0% / −3% |

- **Default is the call for this team.** In 1QB it gets most of Contending's gain this season and more in every later window. In superflex Contending's extra 5 points this season cost about 11 over seasons 2–3 and 19 over seasons 4–5.
- **These teams do become the strongest:** without trades their title odds over seasons 2–3 are 24–27 points, against 17 for mid-value teams of the same strength.
- The size of every gain is inflated by FP prices and by taking the best player in a price window (section 6a's market-price runs halve them); compare the rows, not the levels.


## 24. Experiment: redraft rank, dynasty rank or last season's production? (`analysis/rank_gaps.py`)

Not used by the site. For each veteran in a dynasty market's top 200, two gaps against his dynasty positional rank: that summer's redraft positional rank (FFC ADP), and his points-per-game positional rank last season. "Up" is a quarter or more better than the dynasty rank and at least 3 places, "down" a quarter worse. Residual % against the dynasty price, as section 2b; dynasty markets pooled (FP, KTC, FantasyCalc, DP), cohorts 2018–24.

Redraft ranks are late August and those prices mostly May, so a second run prices everyone on FP's dynasty ECR from the same week as FP's redraft ECR (2019–24; superflex 2020–24).

**Per 1 SD of each gap, both in one model, all veterans** (1 / 2 / 3 years; cohorts positive):

| | Redraft gap | Production gap |
|---|---|---|
| 1QB, May prices | +26 / +19 / +20 (7/7, 6/6, 5/5) | +7 / +7 / +5 |
| 1QB, same date | +29 / +20 / +19 (6/6, 5/5, 4/4) | +2 / +2 / −1 |
| Superflex, May prices | +21 / +16 / +18 (7/7, 6/6, 5/5) | +6 / +7 / +4 |
| Superflex, same date | +24 / +16 / +16 (5/5, 4/4, 3/3) | +2 / +3 / +1 |

**Both gaps together, 1QB, May prices** (mean residual at 1 / 3 years; rows production, columns redraft):

| Production \ redraft | Up | Level | Down |
|---|---|---|---|
| Up | +47 / +34 | +14 / 0 | −27 / −22 |
| Level | +41 / +14 | +13 / −1 | −21 / −13 |
| Down | +36 / −10 | −3 / +8 | −35 / −36 |

- **The redraft rank is the one to trust.** Its gap is worth +16 to +29 per SD at every horizon, in every cohort, in both formats and at same-date prices. The production gap adds +7 or less and is not clear of zero once redraft is in the model. FP redraft ECR in place of FFC ADP gives the same (+31 / +22 / +22).
- **Production only counts when redraft agrees.** Production up with redraft level returned +14 at 1 year and 0 at 3; with redraft down, −27. Production up and redraft up was the best cell, positive in 7 of 7 cohorts at 1 year.
- **Redraft below dynasty is the worst sign, most of all for young players.** At same-date prices, players 25 or younger with redraft down returned −54 / −44 / −35 (0 cohorts positive at any horizon; superflex −49 / −32 / −30). Rookies with redraft down: −36 at 1 year, about level by 3.
- **By age** (per SD, 1QB): for 25 and under both gaps count in year 1 (+15 redraft, +19 production) and only redraft after that. For 26–29 it is redraft in year 1 (+33) and both by years 2–3 (+20 and +10). For 30+, the production gap is negative (−12 to −14; 0 to 1 cohorts positive): an old player whose production outranks his price has not been a buy on that alone, while redraft up has (+38 / +24 / +11).
- **Means sit well above medians** (a few hits carry each cell): in the best cell 50% beat their price, against 33–38% for a level player.

Caveats: the "down" cells at May prices include players hurt over the summer, which is why the same-date run matters; the same-date run has 4–6 cohorts and prices on FP only; cells under 15 player-seasons are left out.

## 25. Experiment: does redraft rank plus age beat a dynasty rank? (`analysis/redraft_age.py`)

Not used by the site. A model of h-year VORP on the summer redraft rank, age, position and rookie or not (gradient boosting, trained in real time on cohorts from 2012), against each dynasty market's own rank over its top 200. The fair test is FP dynasty ECR against FP redraft ECR from the same week in late August (cohorts 2019–24; superflex 2020–24). Spearman with h-year VORP, mean over cohorts, 1QB / superflex:

| Years | Dynasty rank | Redraft rank alone | Redraft + age model | 25% model, 75% dynasty |
|---|---|---|---|---|
| 1 | 0.514 / 0.544 | 0.554 / 0.592 | 0.515 / 0.534 | 0.527 / 0.561 |
| 2 | 0.574 / 0.595 | 0.569 / 0.605 | 0.508 / 0.538 | 0.571 / 0.601 |
| 3 | 0.601 / 0.604 | 0.566 / 0.582 | 0.519 / 0.537 | 0.596 / 0.606 |
| 5 (2 / 1 cohorts) | 0.608 / 0.613 | 0.543 / 0.573 | 0.520 / 0.566 | 0.596 / 0.622 |

- **No. The dynasty rank is the better ranking from year 2 on**, and redraft plus age beat it in no cohort at 2, 3 or 5 years (same date, both formats). The May markets (FP, KTC, FantasyCalc) give the same order, even with the redraft rank three months newer.
- **The redraft rank alone wins year 1**, in every cohort (6/6 and 5/5), which section 2a already says: for next season, use redraft.
- **Adding age made the redraft rank worse, not better.** The model trails the raw redraft rank at every horizon, also among veterans on the redraft list. One specification; a better age adjustment may exist, but the dynasty rank is already that adjustment and more (it also ranks the rookies and stashes a redraft list leaves off).
- **A blend adds about 0.01–0.02 in year 1 and nothing after.** The site's in-season price already tilts the dynasty rank toward the rest-of-season rank by a small power (0.1).
- **For the top of the list the redraft rank helps a little longer.** The top 50 by each ranking captured, of the best possible top 50's VORP: 69% / 70% by dynasty rank and 71% / 74% by redraft rank at 1 year, 75% / 76% and 76% / 78% at 2, and 78% / 79% and 76% / 78% at 3.
- **This fits section 24.** A redraft rank that disagrees with the dynasty rank says which way a player's return will miss; it doesn't make redraft the better ranking overall.

## 26. Experiment: the redraft gap inside a model of price, position, age and draft capital (`analysis/gap_model_sim.py`)

Not used by the site. `single_model_sim.py`'s in-season panel (players in the market's top 340 after weeks 4, 8 and 12, seasons 2021–25) and model, which starts from the market price and learns where the other inputs move off it. Base = market positional rank, position, age, NFL draft round, rookie or not. Gap = log(FP rest-of-season positional rank / market positional rank). Every number is walk-forward (trained on earlier seasons only) and runs KTC 1QB / FantasyCalc 1QB / KTC superflex / FantasyCalc superflex.

**Prediction: % of the error the market price leaves that the model removes.**

| Model | Rest-of-season VORP | Value at next May's market rank |
|---|---|---|
| Base | 1.6 / −3.0 / 2.5 / −3.0 | 2.8 / 2.6 / 1.2 / 3.1 |
| Base + gap | 5.0 / 1.8 / 6.2 / 3.5 | 3.3 / 3.6 / 2.9 / 4.3 |
| Base + gap + production gap | 4.9 / 1.4 / 5.0 / 3.6 | 3.2 / 2.6 / 3.0 / 3.3 |
| Every input (`+all`) | 13.9 / 9.6 / 15.2 / 13.3 | 4.5 / 3.9 / 4.6 / 4.0 |

- **The gap helps this season in every test season of all four runs**, and next May's value by about a point (12 of 14 season-tests). The production gap adds nothing. No model predicts the one-year hold return better than the price alone.

**Ablations** (rest-of-season VORP unless said):

| Ablation | Result |
|---|---|
| Gap down only (rest-of-season rank worse than the market's) | 5.2 / 2.0 / 6.8 / 2.3: as good as both directions |
| Gap up only | 3.5 / −1.5 / 4.3 / −1.1 |
| Gap for 25 and under only | 2.5 / 0.3 / 4.5 / −0.2 |
| Gap for 26+ only | 4.0 / −0.9 / 5.8 / −0.4 |
| Off-list flag, no gap | 2.4 / −1.6 / 4.0 / −2.2 |
| By week: gain in the edge's rank correlation over base | week 4 +0.02 to +0.09, week 8 +0.06 to +0.08, week 12 +0.10 to +0.11 |
| Base + gap, no age / no draft capital / no position | no loss this season; for next May's value age matters most (2.1 / 2.4 / 2.9 / 3.2 without it) |
| `+all` less injury status / week / rest-of-season ranks / last season | −2.0 to −2.7 / −1.2 to −2.5 / +0.4 to −2.0 / −0.3 to −1.1 |
| Base + gap plus week / games played / injury status | +6.3 to +7.1 / +3.0 to +4.8 / +1.3 to +1.8 |

- **The sell side carries it:** a rest-of-season rank below the market's is the signal; a better one adds about half as much at KTC prices and nothing at FantasyCalc's.
- **Both age groups count.** Neither half matches the whole, so limiting the gap to players 25 and under leaves out about half of it.
- **It is not the off-list flag.** Players off the rest-of-season list give about a quarter of the gain; the gap among listed players gives the rest.
- **The gap is worth more late in the season**, not less. For next May's value its gain is small in every week.
- **Most of `+all`'s lead is the calendar.** The week and games played tell the model how much season is left, which sets the scale of rest-of-season VORP but ranks nobody within a snapshot. Injury status is the one input that adds real information on top of the gap (+1.3 to +1.8).

**League sim at market prices** (own-mode teams, title odds in points, value = sum of 340 − market rank).

| Contenders | Title odds, season 1 | Value 1 year on |
|---|---|---|
| Base + gap model's calls less the base model's | +2.3 / +0.9 / +3.9 / +1.5 (3/4, 1/3, 4/4, 2/3 cohorts) | −50 / −67 / −68 / −85 |
| The rules with gap-model picks in place of their signals, less the site's rules | +2.6 / −2.2 / +2.6 / −0.6 | −38 / −31 / −44 / −68 |
| The base + gap model with no rules, less the site's rules | +9.0 / +2.6 / +4.2 / +1.7 | −147 / −212 / −118 / −198 |

- **In trades the gap buys contenders titles this season and costs value**, in every run. For default and rebuilding teams it changes nothing clear (season 1: −2.2 to +1.2).
- **It doesn't beat the site's signals.** Swapping them for the gap model's picks lost value a year on in all four runs and 205 to 434 three years on, for a title gain that is positive at KTC prices and negative at FantasyCalc's.
- **A model with no rules gives up far more value** than the site's calls (520 to 1,075 three years on for contenders, on 1–2 cohorts).
- So the gap belongs where the site already uses a version of it, the contender's win-now targets, and not in the calls.

Caveats: 3–4 walk-forward seasons for prediction and 1–2 cohorts for anything three years on; the model is untuned; KTC and FantasyCalc runs share players and seasons.

## 27. A Contending sell on a rest-of-season rank well below the market's (`analysis/ros_low_sim.py`)

On the site as a Contending-only sell signal (`rules.signals`: `ros_low`), added after this run. The flag: a player on FP's rest-of-season list whose positional rank there is a quarter or more worse than his positional rank in the trade market, and at least 3 places worse. Snapshots after weeks 4, 8 and 12, market top 340, seasons 2021–25. Numbers run KTC 1QB / FantasyCalc 1QB / KTC superflex / FantasyCalc superflex.

**Who is flagged, and what they return against the market's price** (KTC 1QB; the other runs agree in sign):

| On the rest-of-season list | Rest-of-season VORP | Value at next May's rank | A snapshot |
|---|---|---|---|
| Not flagged, healthy | +2.7 | −2.4 | 203 |
| Not flagged, hurt | −2.0 | −9.0 | 24 |
| Flagged, healthy | −5.4 | +9.3 | 38 |
| Flagged, hurt | −15.7 | +7.5 | 11 |

- **Right about this season, wrong about the price.** Flagged players trailed their price in rest-of-season VORP in every season, and beat it in value next May in every season (about 64% above it).
- **They are stashes:** nearly all are rookies or 25 and under, about 49 a snapshot. Almost none is 26 or older, where the rest-of-season rank usually beats the dynasty one. The site's own sell signals fire on almost none of them.
- **The gap grows through the season** (rest-of-season VORP −1 after week 4, −14 after week 12), and a larger gap is worse this season without being worse in value.

**League sim at market prices: the flag laid on the site's calls as a sell signal**, against the site's calls (own-mode teams; title odds in points; value = sum of 340 − market rank; cohorts above zero in brackets for season 1):

| Team | Title odds, season 1 | Title odds, seasons 1–3 | Value 1 year on | Value 3 years on |
|---|---|---|---|---|
| Contending | +4.3 / +1.4 / +2.0 / +1.1 (5/5, 3/4, 4/5, 3/4) | +3.2 / −4.4 / +6.5 / −2.9 | −39 / −28 / −25 / −17 | −149 / −148 / −56 / −81 |
| Default | +1.2 / +0.6 / +1.1 / +1.4 | −4.3 / −7.0 / +0.3 / +1.1 | −66 / −44 / −61 / −31 | −138 / −84 / −59 / −12 |
| Rebuilding | +1.0 / +0.6 / +0.2 / +0.2 | +7.9 / +0.5 / +5.3 / −5.4 | −108 / −99 / −91 / −50 | +27 / +32 / +66 / +51 |
| Contending, hurt players only | +0.4 / +0.4 / +0.7 / +0.5 (5/5, 3/4, 4/5, 3/4) | +1.7 / +0.5 / +0.7 / −0.1 | +3 / −3 / +1 / −7 | −5 / −10 / −3 / −19 |
| Contending, healthy only | +3.7 / +1.0 / +1.0 / +0.5 | +1.2 / −4.9 / +4.0 / −2.7 | −38 / −20 / −25 / −12 | −139 / −124 / −63 / −72 |

- **For contenders it is titles this season for value**, about 1 to 2 extra trades a team: season-1 title odds up in 3 to 5 cohorts of every run, value a year on down in every run. Over three seasons the title gain is unclear (two runs up, two down).
- **Default and rebuilding teams lose value for little:** a point or so of title odds, and 31 to 108 of value a year on (rebuilders in every cohort of three runs).
- **Injury needs its own branch.** Hurt flagged players are few (0.2 trades a team) and selling them costs a contender no value; the title gain and the value loss both come from the healthy ones.
- **A random control** (as many Hold or Use players sold) moved season-1 title odds by −1.7 to +0.6 for contenders, so the season-1 gain is the flag's, most clearly at KTC prices (+6.0 against the control, 5 of 5 cohorts; FantasyCalc +1.0, 2 of 4).
- **A stricter gap** (half again or more) gives less of both. Limiting it to 26+ leaves almost no one.

This is not a sell by the test the site's other calls passed (titles over three seasons and value). The site uses it anyway as a Contending-only sell, for hurt and healthy players alike, as a trade of value for this season's title odds; the note on the page says so. Default and Rebuilding don't get it.

## 28. Which in-season signals hold as simple rules at 2.5 standard errors? (`analysis/simple_rules_sim.py`)

This set the site's in-season signals (`rules.SIG_RULES`). Each signal was tested as one rule for one team type, with no exemptions by position, age or injury: the flag laid on the site's calls with no signals, in the league sim at market prices (trades after weeks 4, 8 and 12, seasons 2021–25). A rule holds if its gain is at least 2.5 standard errors on the team's own objective at both KTC and FantasyCalc prices, and above zero in every cohort but one. Objectives: Contending, title odds this season; Default, title odds over three seasons; Rebuilding, value a year on. Default and Rebuilding rules also had to lose no value at 2.5 standard errors. Standard errors cluster on leagues.

Gain on the objective, KTC / FantasyCalc (z in brackets):

| Rule | Contending | Default | Rebuilding |
|---|---|---|---|
| Sell: rest-of-season rank a quarter worse than the market's, 1QB | **+6.1 / +2.4 (8.5, 4.2)** | −0.8 / −3.5 | −103 / −93 (−7.8, −6.1) |
| The same, superflex | **+6.1 / +3.9 (7.8, 5.6)** | −0.2 / +3.1 | −101 / −55 (−13.1, −4.8) |
| Sell: rest-of-season rank a quarter worse than the preseason, 1QB | +3.2 / +2.1 (5.1, 2.8): held alone, dropped below | +2.4 / +5.6 (2.4, 4.4) | −27 / −26 (−5.7, −3.8) |
| The same, superflex | **+4.0 / +3.7 (6.2, 5.3)** | +1.2 / +7.8 (1.0, 3.9) | −30 / −26 (−6.3, −3.9) |
| Sell: dynasty rank a quarter worse than the preseason, 1QB | +2.4 / +1.5 (6.1, 3.5): held alone, dropped below | +1.2 / +2.1 (2.0, 1.6) | −6 / −3 |
| The same, superflex | +1.7 / +1.3 (3.3, 2.3) | +0.4 / +2.1 | −6 / −12 |
| Buy: 25 or under, not a rookie, rest-of-season rank a quarter better than dynasty, 1QB | **+3.2 / +4.0 (5.9, 5.7)** | **+6.3 / +11.7 (5.1, 4.8)** | −30 / −33 (−4.0, −6.1) |
| The same, superflex | +2.1 / +0.6 (3.2, 0.9) | +2.1 / +0.8 | −73 / −53 (−7.8, −4.2) |
| Buy: a rookie with that gap, 1QB | +0.3 / +1.2 | +3.4 / +9.5 (3.2, 4.1), 1 of 3 cohorts at KTC | −4 / −11 |
| The same, superflex | −0.1 / +0.2 | +1.5 / +6.4 (2.4, 4.1) | −25 / −8 (−7.2, −3.1) |

- **Four rules are on the site** (bold): the market sell for Contending in both formats, the preseason-fall sell for Contending in superflex, and the young-player buy for Contending and Default in 1QB.
- **Nothing holds for Rebuilding.** Every sell and buy lost value a year on in both formats, most at 4 standard errors or more.
- **Nothing holds for Default in superflex**, and no sell holds for Default.
- **A rule that held alone also had to help the set.** With each rule left out of the new set in turn (`signal_sim.py`), the two preseason-fall sells cost 1QB Contending title odds this season once the market sell and the buy were in place: 1.1 / 1.3 points for the rest-of-season one and 0.4 / 1.5 for the dynasty one. Both were dropped there. In superflex the rest-of-season one stayed: leaving it out cost 0.7 / 1.7.
- **No Contending rule cost title odds over three seasons at 2.5 standard errors in this test.** The superflex Contending buy was +3.8 (1.5) / −2.1 (−0.7). Taken out of the old rule set it had raised three-season title odds by 4.7 (2.4) / 7.1 (4.0), so the harm was in the combination; it is gone either way, since it failed here.

**The new set against the old one** (league sim, against no trades; new less old, KTC / FantasyCalc):

| Team | Title odds, season 1 | Title odds, seasons 1–3 | Value 1 year on |
|---|---|---|---|
| 1QB Contending, before the two sells were dropped | −1.6 / −1.9 | −3.0 / −1.8 | −42 / +1 |
| 1QB Default | +0.1 / 0.0 | +0.2 / 0.0 | −20 / −13 |
| 1QB Rebuilding | −0.3 / −0.1 | −2.4 / −6.3 | +33 / +47 |
| Superflex Contending | −0.3 / +1.1 | +4.7 / +7.0 | +34 / +35 |
| Superflex Default | −2.8 / −2.8 | −2.3 / −6.9 | +36 / +3 |
| Superflex Rebuilding | −0.3 / −0.5 | −2.2 / −5.7 | +48 / +25 |

- **Superflex Default pays for the bar:** its old signals added title odds as a set (2.8 this season), but no one of them cleared 2.5 standard errors at both prices.
- **Rebuilders give up title odds over three seasons for value a year on**, which is their objective here.
- The 1QB Contending row is the set before the leave-one-out step; after it the set is the old one without its few hurt-player sells.

Caveats: leagues in a cohort share players, so the standard errors are too small; three-season results rest on 2–3 cohorts; the signals were found on these seasons.

## 29. Do the site's backtested calls and their pools hold at 2.5 standard errors? (`analysis/rule_audit.py`)

An audit, not a change to the site. The same bar on the calls behind the page: each call group's return against the average player at its price (the lenses of `call_strength.py`: three seasons plus value after for Rebuilding's buys and every sell and cut, one season plus value for Default's buys, next season's VORP for Contending's buys), with a player's seasons as one cluster. A call holds if the ratio is 2.5 standard errors from 1 in its direction and on that side of 1 in every cohort but one.

**The site's action calls that hold** (cells holding / cells; 1QB, superflex):

| | Contending | Default | Rebuilding |
|---|---|---|---|
| Buys | 0 / 8, 0 / 8 | 0 / 5, 0 / 5 | 0 / 3, 0 / 4 |
| Sells | 4 / 20, 6 / 22 | 4 / 23, 6 / 23 | 9 / 28, 8 / 29 |
| Cuts | 7 / 23, 2 / 22 | 6 / 20, 1 / 20 | 6 / 20, 1 / 20 |

- **No Buy call holds.** The largest, 1QB young NFL round 1–3 WRs priced 101–200 (57 players), returned 1.58 of their price in next season's VORP and beat it in 6 of 6 cohorts, at 1.5 standard errors: a few hits carry it.
- **About a quarter of the sells hold**, covering 125 to 226 of the 312 to 529 players sold. The ones that do: WRs 26–29 priced 101 and lower (0.22 to 0.45 of their price), young NFL round 1–3 RBs at 101–200, RBs 26–29 at 51–200 for rebuilders, WRs 30+ with a 2-year window, TEs 26–29.
- **Many cuts hold trivially:** a handful of players who all returned about nothing.

**The pools that hold** (ratio, z, cohorts above 1; 1QB, with superflex where it differs):

| Pool | 3 seasons + value | 1 season + value | Next season's VORP |
|---|---|---|---|
| Rookie RBs, WRs and TEs | 1.54, +3.4, 4/4 | 1.30, +3.5, 6/6 | 0.76, −2.3 |
| Rookie WRs | 1.88, +3.1, 3/4 | 1.36, +2.7, 6/6 | 0.86 |
| Rookie RBs | 1.28 | 1.18 | 0.64, −2.8, 0/6 |
| RBs 26–29 | 0.70, −2.7, 0/4 | 0.84, −2.4 (superflex 0.82, −2.9) | 1.18 |
| WRs 30+ | 0.44, −3.7, 0/4 | 0.70, −2.9, 1/6 | 1.12 |
| WRs 26–29 priced 201+ | 0.20, −8.0, 0/4 | 0.43, −5.2, 0/6 | 0.55 |
| WRs 26–29 priced 101–200 | 0.45, −3.9, 0/4 | 0.88 | 1.36 |
| TEs 30+ | 1.26 | 1.12 | 1.70, +3.3, 5/6 |
| QBs 30+, superflex | 1.07 | 1.04 | 1.52, +3.3, 5/6 |
| RBs 25 and under, superflex | 0.78, −2.7, 0/4 | 0.91 | 0.82, −2.5 |

- **The pools say more than the cells.** At this bar the page's price-and-draft-capital detail is mostly not separable from a fair price; what is separable is position by age.
- **A rule set that held would be about eight lines:** buy rookies for value (WRs most) and don't count on rookie RBs this season; sell RBs 26–29, WRs 26–29 priced past 100 and WRs 30+ for value; for this season, buy TEs 30+ (and QBs 30+ in superflex).
- Not done here: such a set has not been through the league sim, where calls act together and a rule can pass alone and fail in the set (section 28).

Caveats: seasons of one cohort share a league year, so the standard errors are too small; the calls were picked on these cohorts; ratios of sums with a few large returns have skewed errors, so a group that beats its price in every cohort can still miss the bar.

## 30. How strict should the bar for a call be? (`analysis/threshold_sim.py`)

An experiment, not a change to the site. Calls are built from pools alone (position x age class x price, position x age class, age class), each team type reading its own measure as in section 29: a player is a Buy or a Sell if the narrowest of his pools with 15+ players is at least B standard errors from a fair price, else a Hold. A cohort's calls never use that cohort: the table trains on every other cohort (2018–24, 30 leagues each); training only on finished seasons (2021–24) is given where it differs. Then the league sim of section 6a at FP prices, each team following the calls for its own type, against no trades. 1QB / superflex.

| Bar | Contending: title odds, season 1 | Contending: seasons 1–3 | Contending: value 3 years on | Default: title odds, seasons 1–3 | Default: value 1 year on | Rebuilding: value 1 year on | Rebuilding: value 3 years on |
|---|---|---|---|---|---|---|---|
| 0.5 | +14.4 / +13.9 | +11.2 / +16.6 | −261 / −566 | +15.0 / +18.5 | +588 / +770 | +569 / +423 | +816 / +682 |
| 1.0 | +16.1 / +12.9 | +10.5 / +9.6 | −401 / −642 | +14.1 / +21.0 | +596 / +829 | +504 / +432 | +741 / +727 |
| 1.5 | +13.1 / +10.1 | +4.7 / +7.3 | −412 / −543 | +18.7 / +21.9 | +553 / +699 | +470 / +467 | +761 / +757 |
| 2.0 | +8.8 / +8.1 | −0.3 / +5.3 | −418 / −458 | +21.7 / +21.6 | +505 / +623 | +399 / +458 | +595 / +786 |
| 2.5 | +3.2 / +4.6 | +0.6 / +1.9 | −261 / −195 | +19.7 / +19.2 | +384 / +416 | +269 / +311 | +469 / +508 |
| 3.0 | +1.2 / +1.1 | 0.0 / −3.2 | −101 / −122 | +6.1 / +12.9 | +128 / +196 | +186 / +205 | +315 / +342 |
| The site's calls | +7.7 / +4.5 | +17.1 / +7.6 | +395 / +164 | +20.0 / +13.4 | +490 / +516 | +522 / +444 | +846 / +703 |
| Random swaps | −1.4 / −0.7 | −2.9 / −1.7 | −20 / −1 | 0.0 / +0.8 | −7 / −2 | +6 / −3 | +7 / +2 |

Players called (1QB): 116 buys and 207 sells of 340 at a bar of 0.5 for rebuilders, 54 and 113 at 2, 45 and 86 at 2.5, 37 and 57 at 3.

- **2.5 is too strict, and 3 much too strict.** From 2 to 2.5 Default and Rebuilding lose a quarter to a third of their value gain in both formats, and from 2.5 to 3 another third to two thirds of what is left, for no gain in titles.
- **About 2 is the best single bar.** Default's title odds over three seasons peak at 1.5 to 2 in both formats (and at 2 on finished seasons only: +9.7 / +22.6). Below 1.5 Default trades more for fewer titles.
- **Rebuilding wants a loose bar:** value falls at every step up in 1QB and is flat to 2 in superflex. On finished seasons only it falls from 0.5 on (1QB +291 a year on at 0.5, +175 at 2).
- **For Contending the bar is not the problem, the measure is.** Rules read on next season's VORP win more this season the looser they are (+16 at a bar of 1), and at every bar they give up the title odds of seasons 2–3 and 100 to 640 of value. The site's calls, which sell on three seasons of value, win less this season and far more after.
- **Against the site's calls:** pools at a bar of 2 match them for Default in 1QB, beat them for Default in superflex (+21.6 against +13.4 over three seasons, and more value), and trail them for Rebuilding in 1QB. On finished seasons only, the site's calls lead in 1QB for every team type and trail for superflex Default.
- **For the in-season signals** (section 28) a bar of 2 would add one rule, the dynasty-fall sell for superflex Contending (+1.7 / +1.3, z 3.3 / 2.3).

**1.5 against 2, in the same leagues** (1.5 less 2; all cohorts / finished seasons only):

| Team | 1QB | Superflex |
|---|---|---|
| Contending, title odds seasons 1–2 | +9.3 / +12.1 | +1.4 / +0.1 |
| Contending, value 1 year on | +45 / +42 | −13 / −59 |
| Default, title odds seasons 1–3 | −3.0 / −3.9 | +0.3 / −6.2 |
| Default, value 1 year on | +48 / +8 | +76 / +132 |
| Rebuilding, value 1 year on | +71 / +10 | +9 / −2 |
| Rebuilding, value 3 years on | +166 / +115 | −29 / 0 |

- **Neither wins everywhere.** 1.5 is better for Contending and Rebuilding in 1QB; 2 is better for Default's titles in both formats, where 1.5 buys value with them; superflex Contending and Rebuilding are a wash.

**What should Contending be judged on?** Contending teams on pools read five ways, and the site's calls (all cohorts; 1QB / superflex; the bar is 1.5 for the two production measures and 2 for the rest, each measure's better one):

| Contending reads | Title odds, season 1 | Seasons 1–2 | Seasons 1–3 | Value 3 years on |
|---|---|---|---|---|
| Next season's VORP | +13.1 / +10.1 | +14.1 / +11.0 | +4.7 / +7.3 | −412 / −543 |
| Two seasons of VORP | +9.6 / +4.5 | +9.6 / +2.1 | +7.4 / −1.1 | −52 / −227 |
| 1 season plus value after | +1.0 / +1.2 | +7.1 / +6.6 | +14.7 / +19.9 | +687 / +735 |
| 2 seasons plus value after | +0.6 / +1.6 | +8.3 / +11.2 | +19.2 / +27.4 | +640 / +772 |
| 3 seasons plus value after | +0.3 / +0.3 | +7.9 / +10.0 | +18.3 / +27.4 | +529 / +901 |
| The site's Contending calls | +7.7 / +4.5 | +10.4 / +6.8 | +17.1 / +7.6 | +395 / +164 |

- **Production measures are short-sighted, two seasons of it included.** They win seasons 1 and 2 and give most of it back in season 3, with value lost.
- **Measures with value after win nothing in season 1** and the most by season 3.
- **The site's 1QB Contending calls already sit between the two:** most of the production rules' season-1 gain, as much over two seasons as anything but next-season VORP at a loose bar, and close to the value measures over three. They sell on three seasons of value and keep whoever is starting through the playoffs (`rules.classify`).
- **The site's superflex Contending calls are the short-sighted ones:** +7.6 over three seasons against +27 for the value measures, which also match or beat them over two (+10 to +11 against +6.8). On finished seasons only the value measures trail over two seasons (−2.5 against +9.2) and lead over three (+23.6 against +9.4), on 2 to 3 cohorts.
- **Judging Contending on seasons 1–2 picks the same calls in 1QB** and points away from production-only rules; it doesn't separate the site's calls from the value measures in superflex as clearly as three seasons does.

Caveats: FP prices, where the sim takes the best player in a price window, so levels are inflated and the standard errors (leagues as clusters) are far too small: read the order of the rows. Training on every other cohort lets a neighbouring cohort's seasons in; the finished-seasons version has 4 test cohorts, and 2 for three-season results. The bar itself is chosen on these cohorts.

## 31. Which sells belong to which kind of team? (`analysis/sell_rules_sim.py`, `team_signals.py persist`)

Experiments, not on the site. Team types by the projection for the coming season: Contending 1st–3rd, Default 4th–9th, Rebuilding 10th–12th. 1QB / superflex.

**How long does a team stay what it is?** (`persist`: the league of section 6a, rosters frozen, finish by real lineup points.)

| | That season | 1 season later | 2 later | 3 later | 4 later |
|---|---|---|---|---|---|
| Projected 1st–3rd in May: finishes 1st–3rd | 45% / 45% | 40% / 35% | 38% / 33% | 32% / 32% | 34% / 29% |
| Finished 1st–3rd: finishes 1st–3rd | | 41% / 42% | 39% / 38% | 39% / 41% | 39% / 39% |
| Finished 10th–12th: finishes 1st–3rd | | 11% / 12% | 14% / 13% | 15% / 12% | 16% / 15% |
| Gap in places, top 3 against bottom 3 finishers | 9 | 3.3 / 3.4 | 2.6 / 2.8 | 2.8 / 2.9 | 2.6 / 2.7 |
| Rank correlation with the finish | | 0.36 / 0.38 | 0.29 / 0.29 | 0.31 / 0.30 | 0.30 / 0.29 |

- **A team projected top 3 finishes there under half the time**, and a top-3 finisher repeats 41–42% of the time (34% to 50% by cohort; 25% by chance).
- **Two thirds of the gap between top-3 and bottom-3 finishers closes in one season, and then it stalls** at 2.6 to 2.9 places through four seasons. A projection's gap keeps fading (1.2 to 1.5 places five seasons on).
- Frozen rosters have no rookie draft, which would help the bottom close faster, and no difference in managers, which would hold the gap open.

**Default's calls for every kind of team** (`middle_mode.py`'s saved run; each mode's calls less Default's, for the team type the mode is named for):

| | Title odds, season 1 | Seasons 1–3 | Value 1 year on | Value 3 years on |
|---|---|---|---|---|
| Projected 1st–3rd: Contending less Default | +2.3 / +3.9 | −1.4 / −0.4 | −134 / −233 | −241 / −411 |
| Projected 10th–12th: Rebuilding less Default | −7.2 / −4.6 | −7.5 / −1.5 | +20 / −81 | +289 / +143 |

- **Default has the most title odds over three seasons for every kind of team.** Contending buys a top-3 team 2 to 4 points this season for 240 to 410 of value three years on (0 of 4 cohorts above zero); Rebuilding costs a bottom-3 team titles in every window for 140 to 290 of value three years on and none a year on.
- With the persistence above, a rule that departs from Default has to clear the stricter bar: **2 standard errors for every team type**, not the 1.5 section 30 had slightly ahead for Contending and Rebuilding within their type.

**One sell rule at a time.** A team sells every player it holds in one class (position x rookie / 25 and under / 26–29 / 30+ x FP rank 1–100 or 101–340), in May or after the season, against a control: as many of its players in that price band, picked at random, sold the same way. Selling anyone for the best lineup fit gains in this sim (three random top-100 sells gave a contender about 8 title points), so the control is what a rule has to beat. Objectives: Contending, title odds over seasons 1–2 with no loss over seasons 1–3; Default, title odds over seasons 1–3 with no loss of value a year on; Rebuilding, value a year on. A rule is kept at 2 standard errors, above zero in every cohort but one.

| Sell | Contending | Default | Rebuilding |
|---|---|---|---|
| RBs 26–29 priced 101+ | **1QB: +0.8 titles (2.0), value +70** / superflex no | +0.9 (2.0), 3 of 5 cohorts / no | **+63 (8.2) / +53 (5.8)** |
| WRs 26–29 priced 101+ | no / +0.9 (1.7) | **+1.1 (2.6) / +1.5 (3.4), value +46 / +37** | **+43 (5.1) / +31 (4.6)** |
| WRs 30+ priced 101+ | no | no | **+34 (6.0) / +19 (4.0)** |
| RBs 26–29 priced 1–100 | no | no | **+18 (4.7) / kept** |
| RBs 30+, TEs 30+ priced 101+ | no | no | **kept / kept** (+11 to +14) |
| RBs 25 and under | no | no | **1QB top 100: +25 (4.2) / superflex 101+: +22 (2.5)** |
| QBs 30+ | **1QB top 100: +1.1 (2.5), 0.2 a team / superflex 101+: +0.8 (2.6), value +36** | no | superflex top 100 kept |

(Gain on the objective less the control, z in brackets; "no" = did not clear the bar.)

- **A top-3 team keeps almost none of the sells.** Past the control, only RBs 26–29 priced outside the top 100 (1QB) and QBs 30+ (superflex, outside the top 100) clear the bar, each worth about a title point over two seasons with value gained as well. The one top-100 sell that cleared is QBs 30+ in 1QB, on 0.2 players a team. A few "sell after the season" rules on rookies also cleared, by under a point on few players; with about 30 tests a team type, one or two of those are expected by chance.
- **Selling a top-100 RB aged 26–29 costs a contender title odds this season:** −2.6 / −1.9 against the control (z −5.1 / −4.0), and −1.0 / −1.1 for Default. It gains value for all three (+12 to +18 a year on, +50 to +76 three years on), so it is a Rebuilding sell only.
- **Default keeps one sell:** WRs 26–29 priced outside the top 100, in both formats.
- **Rebuilding keeps the broad list**, all sold now: RBs 26–29 at any price, RBs 30+, WRs 26–29 and 30+ and TEs 30+ priced outside the top 100, and young RBs.
- **Timing:** nearly every sell that holds is a sell now, not a "use, then sell".
- **Class matters for value, not for titles.** Against the control no rule moved title odds by more than about 1.5 points; the Rebuilding rules moved value by 18 to 63 each.

**Default between its neighbours.** One sell is kept by both Contending and Rebuilding and not by Default: RBs 26–29 priced 101+ in 1QB, where Default gained +0.9 title points over three seasons (z 2.0, 3 of 5 cohorts, so the cohort check failed it) and +71 of value a year on (z 9.5, 6 of 6). That is the test's miss, not the rule's. Judged on value a year on at the bar with no title measure at z ≤ −2, Default also sells, now: RBs 26–29 priced 101+ (+71 / +62), WRs 26–29 priced 101+ (+46 / +37), RBs 30+ priced 101+ (+12 / +14), and in superflex QBs 30+ and TEs 30+ priced 101+ (+22, +9); and after the season, RBs 26–29 in the top 100 (+3 / +4 a year on, +51 / +33 three years on, no title cost, where selling them now cost −1.0 / −1.1 title points this season). So the lists nest: Contending's sells are in Default's, and Default's in Rebuilding's.

Caveats: FP prices and the best player in a price window; leagues in a cohort share players, so z is too large; one rule at a time, and rules interact (section 28); classes are coarser than the site's call groups.

## 32. When to sell a producing older player (`analysis/sell_timing_sim.py`)

An experiment, not on the site. Producing = inside his position's starter line by points per game the season before. One team sells every such player it holds in a class now (May), after this season, or after next season, each time for the most valuable player in his price window at that date, against holding him throughout. The league of section 6a, 40 leagues a cohort; team types as section 31. 1QB / superflex. Every sale gains about 16 to 35 of value from the partner rule alone (the "after this season" column's value a year on), so read value against that.

**Contending teams: title odds over seasons 1–3, and value 3 years on**

| Class (real players) | Sell now | After this season | After next season | Still sellable after next season |
|---|---|---|---|---|
| RBs 26–27 (29) | −0.8 / −5.6; value +105 / +61. Season 1 alone −3.5 / −3.9 | **+0.6 / +0.7; +112 / +76** | −1.5 / −1.1; +87 / +53 | 95% |
| RBs 28–29 (22) | **+0.4 / −1.2; +109 / +65**. Season 1 +0.1 / −0.8 | −4.6 / −3.4; +10 / +4 | −2.4 / −1.8; −13 / +2 | 53–63% |
| RBs 30+ (8) | +4.0 / +1.3 on 3 cohorts; seasons 1–2 −5.5 / −3.6. Season 1 −3.0 / −4.0 | seasons 1–2 −2.6 / −2.4; +30 / +56 | about 0; +19 / +10 | 31–69% |
| WRs 28–29 (28) | +0.6 / −2.6; +88 / +72. Season 1 −2.9 / −3.6 | **+1.8 / +0.7; +91 / +58** | −0.4 / −0.6; +54 / +38 | 90–95% |
| WRs 30+ (21) | **+5.0 / +3.7; +154 / +176**. Season 1 −0.3 / −1.3 | +1.9 / −0.2; +95 / +79 | +0.5 / +1.7; +57 / +66 | 76–81% |
| TEs 29+ (15) | −2.5 / −5.4; +35 / +49 | −4.7 / −2.9; +39 / +64 | −2.0 / −2.2 | 90% |
| QBs 32+, superflex (17) | −1.6; +99. Season 1 −2.6 | −1.6; +63 | +0.1; +33 | 68% |

- **"Use, then sell" is right where the price lasts a year: RBs at 26–27 and WRs at 28–29.** Selling them now costs a contender 3 to 4 title points this season; selling after it costs nothing over three seasons and keeps as much value.
- **It is wrong for RBs at 28–29.** Their price is gone a year later (value 3 years on +4 to +10 against +65 to +109 for selling now, and by the second year barely half can be sold at all), and selling late still costs the title odds of the seasons after. Selling now cost a contender nothing in season 1. For them it is now or never.
- **WRs 30+ are a sell now for everyone:** about a point of title odds this season, and the most titles over three seasons and the most value of any date.
- **A contender keeps producing TEs at 29+, superflex QBs at 32+ and RBs at 30+.** Selling cost title odds at every date, and what a late sale brings back is little more than the partner rule's own gain. The RBs are 8 careers on 3 to 4 cohorts.
- **Default** follows the same dates with smaller title costs: after this season for RBs 26–27 (+1.5 / +0.5 over three seasons against +0.5 / −0.8 now), now for RBs 28–29 (+2.0 / +0.5, value +124 / +102 against −0.9 / −1.4 and +21 / +15 a year later), now for WRs 30+, RBs 30+ and TEs 29+.
- **Rebuilding sells now in every class:** value is highest then and titles don't fall.

Caveats: few real players a class, so each row is a handful of careers replayed; FP prices; one partner rule; a player who has left the top 340 can't be sold, which is the risk of waiting and is in the numbers.

## 33. Which buys belong to which kind of team? (`analysis/sell_rules_sim.py`, `SIDE=buy`)

An experiment, not on the site. Section 31's design for buys: every player of one class on another team is a Buy and the team buys up to 3 (the biggest lineup gain for Contending and Default, the youngest for Rebuilding), against a control that marks as many random players of that price band. Objectives, team types and the bar of 2 as section 31. Gain on the objective less the control, z in brackets; 1QB / superflex.

| Buy | Contending: title odds, seasons 1–2 | Default: title odds, seasons 1–3 | Rebuilding: value 1 year on |
|---|---|---|---|
| TEs 30+ in the top 100 | **+5.8 (7.1) / +5.6 (7.3)**, 5 of 5 cohorts | **+4.0 (5.3) / +5.7 (7.4)** | +21 (3.4) in superflex, 3 of 5 cohorts |
| RBs 30+ in the top 100 | **+12.6 / +14.6**, one cohort with two seasons | +15.9 / +13.9 over two seasons, no third season yet | **+26 / +54**, one cohort |
| Rookie WRs in the top 100 | no (season 1 −0.7 in superflex) | **+8.3 (6.8) / +8.5 (5.1)**; value +109 / +120; season 1 −1.6 / −1.0 | **+102 (10.5) / +96 (11.7)**; +201 / +253 three years on |
| Rookie RBs | no | priced 101+: **+1.7 (2.8)** / +2.7 (3.3) on 2 of 5 cohorts; season 1 −1.7 / −1.8 | top 100: **+41 (3.6)** / +40, 4 of 6 cohorts |
| Rookie TEs | no | no on titles; priced 101+ +178 of value in 1QB | **+53 to +81 (7.8 to 8.8)**, title odds −1.3 to −2.6 over three seasons |
| Rookie QBs | no | no | **1QB +67 and +109; superflex top 100 +33** |
| QBs 26–29 priced 101+ | +1.3 (2.2) / +3.0 (3.5), 4 of 6 cohorts | +2.2 (3.3) / +3.1 (3.3), 3 of 5 cohorts | no |
| RBs, WRs and QBs 30+ priced 101+ | no: value −50 to −160 | no: the same | no |

- **Two families, not a nested list.** Old producers in the top 100 (TEs 30+, and RBs 30+ on one cohort) are buys for Contending and Default; rookies are buys for Rebuilding, and of the rookies only WRs in the top 100 (and RBs priced 101+ in 1QB) also clear for Default.
- **The contender's buy is the producing 30+ TE**, worth about 6 title points over two seasons beyond the control in every cohort, with no value lost. It is the mirror of section 32: the players a contender shouldn't sell are the ones it should buy.
- **Rookie WRs in the top 100 are the one buy that gains both titles over three seasons and value**, for Default and Rebuilding; they cost a point or so in season 1.
- **Old players priced outside the top 100 are not buys for anyone:** no title gain beyond the control and 50 to 160 of value lost.
- **Near misses on the cohort check:** QBs 26–29 priced 101+ (title odds up 1 to 3 points for Contending and Default in both formats, +158 of value three years on in 1QB).

**Young non-rookies and superflex QBs.**

| Buy (25 and under, not a rookie) | Contending: title odds, seasons 1–2 | Default: seasons 1–3 | Rebuilding: value 1 year on | Value 3 years on, any team |
|---|---|---|---|---|
| WRs in the top 100 | −7.6 (−5.7) / −5.8 (−4.8) | −3.0 / −1.3; season 1 −5.7 / −4.1 | +22 (2.7), 2 of 6 cohorts / **+50 (5.8)** | +28 to +193 |
| WRs priced 101+ | +1.2 (1.9) / −1.6 | −1.5 / −5.3 | −89 / −53 | −165 to +54 |
| QBs in the top 100 | −6.9 (−6.4) / −8.5 (−7.3) | −8.6 / −11.0 | +57 (5.6), 4 of 6 cohorts / −100 (−9.7) | 1QB +144 to +177; superflex −82 to +128 |
| QBs priced 101+ | −0.7 / −0.4 | +1.1 / −0.9 | −65 / −105 | — |

- **No team type buys young non-rookie WRs or QBs as a rule**, apart from superflex Rebuilding with top-100 WRs. A top-100 young WR or QB costs Contending and Default 5 to 11 title points for value three years on; the cheap ones lose value.
- **Superflex QBs.** Buying a top-100 QB of any age cost Contending and Default title odds (−7 to −14 over two to three seasons at 25 and under, 26–29 and 30+), and a top-100 QB 25 or under lost a rebuilder 100 of value a year on. The QB buy that leaned positive for every team type is 26–29 priced 101+ (Contending +3.0 over two seasons and +5.4 over three, Default +3.1, Rebuilding +98 of value three years on), short of the cohort check each time. QBs 30+ priced 101+ are a one-season rental (+1.5 title points in season 1, value −46 to −91).
- **Selling superflex QBs moves little** (section 31's run): QBs 30+ priced 101+ gain 20 to 36 of value for any team, and no sell of a QB under 30 clears. In section 32's run a producing QB under 32 is a hold for everyone (selling now cost Contending −3.0 title points over three seasons and gained no value three years on), and one 32 or older is a hold for Contending (−2.6 in the season of the sale) and a sell now for Default and Rebuilding (+49 / +47 of value a year on, +112 / +106 three years on).
- This sits beside section 5, where superflex QBs beat their price in VORP against a replacement QB26: here every team already rosters several QBs, so a top-100 QB adds less to a lineup than the top-100 player given for him.
- **The Rebuilding buy of RBs 30+ in the top 100 should not be used:** it is one cohort, and it runs against everything else here about rebuilders and old RBs.

Caveats as section 31; RBs 30+ in the top 100 exist in two cohorts.

## 34. The rules at 2 standard errors: the in-season signals, and a rule set for the calls (`simple_rules_sim.py table`, `analysis/ruleset_sim.py`)

**The in-season signals, re-read at 2** (this is what the site uses now). Section 28's runs with section 31's team types and section 30's objectives: Contending on title odds over seasons 1–2 (or season 1 and seasons 1–3), Default over seasons 1–3, Rebuilding on value a year on; 2 standard errors at both prices, every cohort but one, and for Default and Rebuilding no value lost at 2.5. KTC / FantasyCalc.

| Rule | Who | Gain |
|---|---|---|
| Sell: rest-of-season rank a quarter worse than the market's | Contending, both formats | 1QB +9.9 / +3.9 over two seasons; superflex +10.0 / +3.6 |
| Sell: rest-of-season rank a quarter worse than the preseason | Contending, superflex | +5.8 / +4.3 over two seasons |
| Buy: 25 or under, not a rookie, rest-of-season rank a quarter better than dynasty | Contending and Default, 1QB | Contending +3.2 / +4.0 in season 1, +6.6 / +10.0 over three; Default +5.8 / +10.9 over three |
| Buy: a rookie with that gap | Default, 1QB | +3.3 / +9.2 over three seasons, value +13 / +19 a year on |

- **One rule is new at 2: the rookie buy for 1QB Default.** Left out of the set it cost 2.5 / 3.2 title points over three seasons, so it stays. On today's data it changes one call.
- The two preseason-fall sells again clear alone for 1QB Contending (+4.7 / +5.5 and +2.3 / +3.3 over two seasons) and again fail when left out of the set (section 28), so they stay off. In superflex the dynasty-fall sell (z 2.5 / 1.6) and the rookie buy (value three years on down 2.8 standard errors) miss.
- Still nothing for Rebuilding, and nothing for Default in superflex.

**A rule set for the calls themselves** (`ruleset_sim.py`): sections 31–33's rules acting together, each team following its own type's list, against the site's calls in the same leagues. "core" is the class sells and buys; "timing" adds section 32's dates for producing players. 1QB / superflex, the new set less the site's calls:

| Team | Set | Title odds, season 1 | Seasons 1–2 | Seasons 1–3 | Value 1 year on | Value 3 years on | Trades a team (site's: 11 to 13) |
|---|---|---|---|---|---|---|---|
| Contending | core | −0.9 / +0.6 | −0.5 / +2.7 | −10.0 / 0.0 | −236 / −224 | −303 / −135 | 2.8 / 1.8 |
| Contending | timing | −0.3 / +1.3 | +0.2 / +3.5 | −6.1 / +2.5 | −181 / −157 | −148 / −11 | 4.4 / 3.5 |
| Default | core | +0.3 / +4.0 | +1.4 / +7.2 | −0.7 / +8.3 | −143 / −86 | −147 / −115 | 7.7 / 8.8 |
| Default | timing | +0.2 / +3.6 | +2.8 / +6.5 | +3.3 / +8.0 | −70 / −49 | −42 / −40 | 8.7 / 9.5 |
| Rebuilding | core | +0.3 / −0.3 | −1.1 / −2.7 | −5.5 / −8.5 | −121 / −30 | −275 / −88 | 9.8 / 10.9 |
| Rebuilding | timing | +0.4 / −0.4 | −0.7 / −2.7 | −4.0 / −8.4 | −97 / −32 | −236 / −82 | 10.5 / 11.3 |

- **The 2-standard-error set does not replace the site's calls.** It is worse for 1QB Contending (6 to 10 title points over three seasons, 150 to 300 of value) and for Rebuilding in both formats (4 to 9 title points over three seasons and 30 to 275 of value). The site's calls make three to five times as many trades for a contender, and the extra ones pay.
- **It is better in one place: superflex Default**, by +3.6 to +4.0 title points this season and +8.0 to +8.3 over three (z 5 to 10, 4 to 6 cohorts of 5 to 7), for 40 to 115 of value. Superflex Contending gains 2 to 3.5 title points over two seasons for 157 to 224 of value a year on; 1QB Default is level on titles and down on value.
- **Adding the sale dates helps every team type**: "timing" is ahead of "core" on titles over three seasons and on value in all six cells.
- So the bar of 2 is right for adding a rule to a set (section 30) and for the in-season signals; it is not a reason to cut the calls down to the rules that clear it one at a time. The finer calls carry real gains that single-rule tests against a control can't see.

**The same at 1.5, and what the bar really does.** At 1.5 the in-season signals add one rule to the table above: the dynasty-fall sell for superflex Contending (+2.3 / +1.5 title points over two seasons, z 2.5 / 1.6; left out of the set it cost 0.4 to 0.7 title points this season on 0.1 trades a team). It is not on the site.

For the calls, lowering the bar from 2 to 1.5 changes the class lists by one rule: what keeps rules out is the every-cohort-but-one check, not the bar. Without that check the lists are two to three times as long, and read on all cohorts such a set looks far better than the site's calls (1QB Default +19.6 title points over three seasons, superflex Contending +18.8). That is the lists fitting the cohorts they were read on. With each cohort's rules read from the other cohorts only, the sets with the timing dates, less the site's calls (1QB / superflex):

| Team | Bar, no cohort check | Title odds, season 1 | Seasons 1–3 | Value 1 year on | Value 3 years on |
|---|---|---|---|---|---|
| Contending | 1 | −2.1 / −0.5 | −10.8 / +5.6 | −95 / −103 | −230 / +49 |
| Contending | 1.5 | −2.8 / −0.4 | −13.0 / +5.1 | −214 / −130 | −281 / +10 |
| Contending | 2 | −3.9 / −1.3 | −15.3 / +0.8 | −273 / −165 | −339 / −54 |
| Default | 1 | −2.4 / −0.8 | −7.2 / +5.8 | +16 / +90 | −85 / +106 |
| Default | 1.5 | −1.5 / −1.2 | +4.5 / +5.7 | +54 / +127 | +71 / +150 |
| Default | 2 | −2.4 / −0.3 | +1.9 / +6.7 | +54 / +168 | −76 / +263 |
| Rebuilding | 1 | −0.4 / −0.8 | −1.2 / −7.6 | +11 / −31 | +67 / +93 |
| Rebuilding | 1.5 | −0.3 / −0.5 | −3.7 / −8.9 | +33 / −73 | +75 / −23 |
| Rebuilding | 2 | −0.3 / −0.4 | −5.4 / −8.9 | −12 / −103 | +11 / −53 |

- **1.5 is at least as good as 2 in most cells**, and 1 is no better than 1.5: the gain from loosening stops at about 1.5, as section 30 found.
- **No bar makes a class rule set that beats the site's calls for Contending in 1QB or for Rebuilding in superflex**, and for 1QB Default and Rebuilding the 1.5 set is about level (a few title points and 30 to 75 of value either way).
- **Superflex Default is better at every bar**, on title odds over three seasons (+5.7 to +6.7) and on value (+90 to +263), for about a point of title odds this season. It is the one place the evidence supports replacing the site's calls.
- The in-sample gain of +19.6 for 1QB Default became +4.5 out of cohort: about three quarters of what a loose bar seems to add is fit to the sample.

Caveats: the rule lists were found on these cohorts, which favours them; FP prices; in this sim more trades gain more whatever is traded, though random swaps (6 a team) gained nothing here.

## 35. Seven rules in place of the calls (`analysis/ruleset_sim.py`, `sell_rules_sim.py ONLY=QB`)

Experiments, not on the site. The aim: a short list of reasoned rules, the same in both formats, consistent across team types (Default always between Contending and Rebuilding), that can stand in for the site's backtested calls. Team types and league sim as section 34; Rebuilding is judged on value three years on (value a year on hid buys whose price doesn't move in a year, QBs above all). 1QB / superflex.

**The rules** (`simple`, `more`). "Producing" = inside his position's starter line by points per game the season before. Everyone not named is a Hold.

1. Sell RBs and WRs 26 or older and TEs 30 or older who are priced outside the top 100.
2. A producing veteran is sold on section 32's date: after the season for RBs 26–27 and WRs 28–29 (Rebuilding: now); now for RBs 28–29 and WRs 30+; now for RBs 30+ and TEs 29+, except that Contending holds them.
3. A top-100 RB at 26–29 who isn't producing: Contending holds, Default sells after the season, Rebuilding sells now.
4. Buy top-100 rookie WRs (Default, Rebuilding), rookie TEs (Rebuilding) and top-100 TEs 30 or older (Contending, Default).
5. Sell anyone priced outside the top 200 who isn't a rookie or a QB.
6. Sell RBs 25 or under who aren't producing.
7. QBs: buy at 26–29 priced outside the top 100; sell at 33 or older.

**Each set against the site's calls** (set less the site's calls; value = sum of 340 − FP rank):

| Team | Set | Title odds, season 1 | Seasons 1–3 | Value 1 year on | Value 3 years on |
|---|---|---|---|---|---|
| Contending | rules 1–4 | −1.7 / +2.3 | −3.5 / +11.2 | −21 / +106 | +5 / +164 |
| Contending | rules 1–7 | −1.8 / +1.9 | +2.3 / +22.5 | +211 / +314 | +426 / +517 |
| Contending | rules 1–4 over the site's calls | +2.8 / +2.5 | +9.8 / +11.9 | +131 / +180 | +160 / +223 |
| Default | rules 1–4 | −2.0 / +2.6 | +4.6 / +11.8 | −19 / −27 | +23 / −29 |
| Default | rules 1–7 | −2.2 / +0.1 | +11.3 / +17.8 | +323 / +215 | +486 / +379 |
| Default | rules 1–4 over the site's calls | −0.1 / +1.9 | +15.1 / +12.1 | +213 / +180 | +304 / +181 |
| Rebuilding | rules 1–4 | +0.4 / +0.1 | −5.4 / −7.3 | −116 / −71 | −222 / −92 |
| Rebuilding | rules 1–7 | −0.2 / −0.3 | −1.0 / −4.6 | +88 / +82 | +110 / +193 |
| Rebuilding | rules 1–4 over the site's calls | 0.0 / −0.4 | −3.8 / −7.1 | +29 / +39 | −48 / +56 |

("Over the site's calls": the four rules where they speak, the site's call elsewhere, Default moved between its neighbours.)

- **The seven rules alone beat the site's calls on value for every team type in both formats** (three years on: +110 to +517, above zero in 3 or 4 of 4 cohorts), and on title odds over three seasons for Default in both formats and Contending in superflex.
- **What they cost:** about 2 title points this season for 1QB Contending and Default, and 4.6 over three seasons for superflex Rebuilding. Against the four rules laid over the site's calls they give up 7.5 title points over three seasons for 1QB Contending and 3.8 for 1QB Default, and are ahead everywhere else.
- **Rule 5 does most of it.** Added alone to rules 1–4 it gained 188 to 281 of value three years on for every team type in both formats, as much as all of the site's calls laid over the four rules, and 1.7 to 5.6 title points over three seasons.
- **Rule 6 gains value everywhere** (+107 to +167 three years on alone) and costs superflex Contending and Default 2 to 3.5 title points over three seasons alone; in the set it adds 23 to 122 of value.
- **Rule 7** read on finer QB ages (rookie, 25 and under, 26–29, 30–32, 33+; one rule against random players at the price): buying 26–29 priced past 100 gained 133 to 169 of value three years on in 1QB and 57 to 104 in superflex for every team type; 30–32 did not (+3 to +35); 33+ lost 76 to 230 bought and gained 10 to 47 sold; a top-100 QB of any age cost superflex Contending and Default 3 to 12 title points. In the set it adds 11 to 130 of value three years on and 6.8 title points over three seasons for superflex Contending.
- **Calls the site makes that lose** when added to rules 1–4, in both formats: selling rookie RBs in the top 100 (value −11 to −37 three years on, every team type) and buying WRs or RBs 25 or under priced 101–200 (−2 to −73).
- **"Producing" read other ways** (this season's projection from redraft ADP, the two ranks averaged, either, both) moved the layered set by 1 to 2 title points; "both" was a little ahead and "either" a little behind in both formats.
- **Consistency:** rules 5–7 are the same call for every team type, and in rules 1–4 the push to sell never falls from Contending to Rebuilding, so Default is between its neighbours by construction.

Caveats: the rules were written after seeing these cohorts, so this is not out of sample; there are seven of them, which limits the fit but doesn't remove it. FP prices and the best player in a price window. A team makes at most three buys, so a buy rule can crowd out another. Value three years on has 4 cohorts.

## 36. Factors for a ranking by team type (`analysis/rank_factors.py`)

The site's Rankings tab re-orders the FP list by each team type's calls. The factor for a call is the one multiplier on FP rank that
prices the players given that call (live seven rules, base calls, ranks 1-300) fairly: sum of fair(k x rank) = sum of what they
returned (VORP over h seasons + the value of the rank held after). Range = each cohort left out. 1QB / superflex:

| Team type (seasons) | Buy | Use, then sell | Sell | Hold |
|---|---|---|---|---|
| Contending (1) | 0.70 / 0.82 (n 19) | 1.02 / 1.07 | 1.14 / 1.09 | 0.95 / 0.96 |
| Default (3) | 0.47 / 0.46 (n 36 / 28, range 0.32-0.52) | 1.54 / 1.50 | 1.28 / 1.23 | 0.91 / 0.93 |
| Rebuilding (3) | 0.64 / 0.65 (n 63 / 49) | none | 1.30 / 1.26 | 0.91 / 0.93 |

Holds come out under 1 only because the Sells pull the average at each rank down, so the site takes each factor over Hold's, averages
the formats (Contending 0.80 / 1.09 / 1.17, Default 0.50 / 1.65 / 1.37, Rebuilding 0.70 / 1.39) and uses the square root: 0.9 / 1.05 /
1.1, 0.7 / 1.3 / 1.15, 0.85 / 1.2. The halving is a judgment: the rules were picked on these seasons, and section 34 found most of an
in-sample gain gone out of cohort. Over three seasons a Use-then-sell (producing RB 26-27, WR 28-29) falls further than a Sell, which
is why the call is to sell after the season; over one season he is fairly priced.

## 37. Where the seven rules trail the old calls, and an exception to rule 6 (`analysis/ruleset_sim.py`, `GAP=`, `EIGHT=1`)

The seven rules alone trailed the four-rules-over-old-calls set by 7.5 title points over three seasons for 1QB Contending. `GAP=contend`
puts the old calls (`rules_before_seven.py`) back one bucket at a time wherever old and seven disagree. One bucket carries the gap: the
old calls bought RBs 25 or under priced 101-200 who weren't producing, the late-round ones (NFL round 4+ or undrafted), where rule 6
sells them: +2.6 this season, +5.5 over two, +12.9 over three (5/5 cohorts), value 3 years on -9.

`EIGHT=1` tests candidate rules for that bucket against the live seven, Contending and Default, 1QB / superflex (title odds over three
seasons, cohorts above zero; value 3 years on):

| RBs 25-, not producing, 101-200 | Contending 1QB | Contending SF | Default 1QB | Default SF |
|---|---|---|---|---|
| hold, round 4+ | +3.6 (4/5), -18 | +1.5 (4/5), +10 | +5.9 (4/5), -31 | +3.8 (5/5), +8 |
| buy, round 4+ | +12.9 (5/5), -9 | +3.4 (2/5), -44 | +7.3 (4/5), -61 | +2.6 (3/5), -54 |
| hold, any round | +0.9, -69 | -0.1, -41 | +3.0, -110 | +2.0, -67 |
| buy, any round | +5.5, -138 | -3.3, -155 | +3.9, -171 | +0.5, -157 |
| buy, rounds 1-3 | -1.7, -127 | -3.1, -122 | -0.5, -141 | -1.3, -117 |

Holding the late-round ones gains in all four cells at little cost; buying them is a 1QB result. Buying WRs 25- at 101-200 lost in
superflex (-8.7 Contending), and dropping the QB buy lost in superflex (-6.9). The site's rule 6 now holds late-round RBs at 101-200
for Contending and Default. Rebuilding wasn't tested. The title gains carry the sim's usual flattery of any trade that fills a lineup.

## 38. Would a regression order the ranking better than the factors? (`analysis/rank_model.py`)

Each cohort predicted from the others, FP ranks 1-300; target = what the player returned (1 season for Contending, 3 for Default and
Rebuilding). `fp` = the average return at his FP rank; `factors` = section 36's method; `linear` / `ridge` = least squares on the fair
return at his rank times position, age, age squared, rookie, producing and NFL round 4+; `boost` = gradient-boosted trees. Spearman
of predicted against realised, and the share of the variance fp leaves that the method explains (1QB / superflex):

| | Spearman, 1 season | skill, 1 season | Spearman, 3 seasons | skill, 3 seasons |
|---|---|---|---|---|
| fp | 0.800 / 0.801 | 0 | 0.670 / 0.696 | 0 |
| factors (site), Default's at 3 | 0.807 / 0.803 | 0.009 / 0.003 | 0.682 / 0.700 | 0.022 / 0.030 |
| linear | 0.799 / 0.798 | 0.012 / -0.002 | 0.670 / 0.691 | 0.045 / 0.047 |
| ridge | 0.799 / 0.800 | 0.011 / -0.001 | 0.664 / 0.691 | 0.052 / 0.053 |
| boost | 0.783 / 0.785 | -0.017 / -0.037 | 0.654 / 0.687 | -0.027 / -0.009 |

No. The FP order is hard to beat by any method: the factors lift the rank correlation by about 0.005-0.01 and are the only method
that orders players better than FP in most held-out cohorts in every cell. The regressions explain more of the size of the return
over three seasons (5% of what FP leaves, against 2-3%) but order players no better than FP, and in superflex their top 100 returned
less than FP's (255 against 260). The trees are worse than FP. The regression's signs agree with the rules (producing, rookie and
late-round up; age down for WRs and QBs), so it is the same information, fitted with more freedom than 4-6 cohorts support. A
logistic model of beating one's price wasn't run: it would rank by the chance of a gain, not its size, and the factors already do
that job.

## 39. The site's "Buy low (injured)" call, backtested (`analysis/injured_buy.py`)

The call: a rebuilding team buys a player who is out for the season (on IR / PUP / Out and off the FP rest-of-season list) when the
trade market has marked him well below his FP dynasty rank. Test: the in-season panel at weeks 4 / 8 / 12, 2019-25, FP top 200, each
player-season once at the first snapshot he is out. Returned / price = what he was worth from then on against the average player at
his FP rank, or at the market's rank that day (KTC, 2020 on). 1QB / superflex:

| Out for the season | n | 1 season, FP price | 1 season, market price | 3 seasons, FP price |
|---|---|---|---|---|
| healthy players (baseline) | 1204 / 1067 | 1.03 / 1.10 | 1.05 / 1.17 | 1.04 / 1.12 |
| all | 67 / 82 | 0.63 / 0.53 (0/6 cohorts) | 0.76 / 0.79 | 0.61 / 0.48 |
| **the site's call: market marked him down** | 15 / 19 | **0.22 / 0.45** | 0.38 / 0.83 | 0.19 / 0.18 |
| market not marked down | 34 / 31 | 0.87 / 0.75 | 0.87 / 0.77 | 0.83 / 0.75 |
| top 10 at position preseason (all 27 or under) | 10 / 12 | 0.83 / 0.69 | 0.85 / 0.79 | 0.96 / 0.82 |
| position rank 11-24 | 27 / 32 | 0.50 / 0.43 | 0.76 / 0.89 | 0.25 / 0.16 |
| age 25- / 28+ | 34, 15 / 36, 22 | 0.68, 0.49 / 0.58, 0.41 | | 0.66, 0.30 / 0.59, 0.16 |
| RB | 20 / 21 | 0.73 / 0.72 | 1.02 / 1.00 | 0.72 / 0.74 |
| RB 25- | 11 / 10 | 0.87 / 0.88 | 1.13 / 1.14 (3/3 cohorts, n 8 / 6) | 0.77 / 0.81 |
| WR | 24 / 28 | 0.42 / 0.33 | 0.55 / 0.51 | 0.35 / 0.26 |
| TE | 10 / 12 | 0.50 / 0.27 (0% beat) | | 0.45 / 0.29 |
| QB | 13 / 21 | 0.97 / 0.64 | 0.81 / 0.90 | 1.17 (n 5) / 0.54 |
| rookie | 10 / 7 | 1.21 / 1.03 | 1.01 / 0.85 | 1.53 (n 6) |

- **The call loses, and the market's markdown is the warning, not the bargain.** The players the market marked down returned a fifth
  to a half of their FP price and still trailed at the market's lower price. Those it left alone did better (0.75-0.87).
- **Elite and young is the best group and still not a buy** in this window: 0.69-0.96, 7-12 players (Burrow 2020 and Prescott 2020
  up; Barkley 2020, Kittle 2020, Javonte Williams 2022 down; Lance 2022 zero). Section 9's longer MFL history has young top-5 players
  ahead after 3-5 years; here they are about fair at best.
- **RBs hold up better than WRs, not worse.** Young RBs are the one cell at or above 1 at the market's price (1.13 / 1.14, n 6-8);
  injured WRs and TEs returned a third to a half at every age.
- **By position** (next season, FP price): RBs 25- 0.87 / 0.88 but 26-27 0.11 and 28+ 0.28 / 0.17; WRs 0.42 / 0.33 at every age,
  those ranked 11-24 at the position 0.16 / 0.15 (0% beat); QBs 0.97 / 0.64, top-10 QBs 1.00 / 0.70 (n 6-7).
- **A narrow version fails too.** 25 or under and a top-25 player overall preseason: 0.71 / 0.65 (n 3-4); top 50: 0.65 / 0.62
  (n 7-8, 0/4 cohorts; Breece Hall 2022 and Joe Burrow 2020 up, Barkley 2020, Javonte Williams 2022, Sutton 2020, Dobbins 2021,
  Bateman 2022, Lance 2022 down). The market almost never marks these players down, so the old call would not have fired on them.
  The site's Buy low (injured) call is dropped: every team type stashes.
- **One you own: sell in the offseason** (`own()`, 49 / 50 player-seasons with a market price, 2021-24). The market holds an injured
  player's price to May (1.03 / 1.02 of it; a healthy player's slips to 0.87 / 0.86 as rookies enter), then he returns 0.76 / 0.79
  of it. Value per player against keeping everyone: sell everyone now +31% / +27%, sell everyone in the offseason +32% / +34% (4/4
  seasons), stash RBs 25- and sell the rest +32% / +36%, stash everyone 25- and sell 26+ only +6% / +7% (young WRs keep 35 / 37 a
  player against 59-79 sold; young QBs 27 / 66 against 111 / 195). RBs 25- are level kept or sold (135 / 183 against 140 / 181); RBs
  26+ lose a third of their price by May and are worth almost nothing either way. Players hurt around week 12 lose about a fifth of
  their price by May. The site: Stash for rookies and RBs 25-, Sell in the offseason for everyone else out for the season.
- Limits: 45-82 player-seasons, 2-6 cohorts; the superflex 2019 preseason list is thin, so its position ranks that year are rough.

## 40. A startup drafted for seasons 2-4 (`analysis/startup_window.py`)

FP May rows 2018-21, ranks 1-300. Factor = the multiplier on FP rank that prices a group fairly on the target (under 1: take him
earlier than FP ranks him); range over cohorts left out is in the script's output. Season 1 holds 30% / 29% of the pool's four-season
VORP, seasons 2-4 the other 70%. 1QB / superflex:

| Group | season 1 | seasons 2-4 | share of his 4-season VORP in season 1 |
|---|---|---|---|
| Default call: buy | 0.84 / 0.72 | 0.41 / 0.36 | 21% |
| Default call: hold | 1.05 / 1.08 | 0.91 / 0.93 | 27% |
| Default call: use, then sell | 0.63 / 0.72 | 1.63 / 1.58 | 43% |
| Default call: sell | 1.06 / 0.95 | 1.35 / 1.24 | 37% |
| rookie WR | 1.10 / 1.05 | 0.55 / 0.62 | 15% |
| rookie QB | 1.21 / 1.69 | 0.55 / 0.66 | 9% |
| rookie RB | 1.36 / 1.47 | 0.80 / 0.87 | 18% |
| QB 25- | 0.67 / 0.80 | 0.64 / 0.65 | 30% |
| WR 25- | 1.01 / 0.96 | 0.91 / 0.91 | 28% |
| WR 26-27 | 0.88 / 0.90 | 1.05 / 1.13 | 33% |
| WR 28-29 | 0.77 / 0.79 | 1.49 / 1.48 | 42% |
| WR 30+ | 1.14 / 1.07 | 2.25 / 1.99 | 49% |
| RB 25- (not rookie) | 1.24 / 1.32 | 1.27 / 1.37 | 30% |
| RB 26-27 | 0.79 / 0.85 | 1.78 / 1.73 | 43% |
| FP top 24, 25- | 1.83 / 1.22 | 0.97 / 0.98 | 26% |

- **It is the Default board, pushed a little harder.** On seasons 2-4 the Default calls' factors are 0.41 / 0.91 / 1.63 / 1.35 (1QB),
  against 0.47 / 0.91 / 1.54 / 1.28 on Default's own three-season measure (section 36): same order, same sizes within the noise.
- **Where it leaves FP:** rookies up (WRs and QBs most), QBs 27 or under up, WRs 28+ and RBs 26-27 down, non-rookie RBs down at every
  age. The top 24 is FP's order: young players there are fairly priced for seasons 2-4.
- **What it gives up:** the players it passes on are the ones underpriced for season 1 (use-then-sell 0.63 / 0.72, WRs 28-29 0.77,
  RBs 26-27 0.79), and season 1 is the largest single season. Section 13's startup sim found neither all-in nor punt beat drafting
  near the list.
- **Longer windows from season 1 (1-3 / 1-4 / 1-5, the last on 2018-20 only), 1QB:** buy 0.56 / 0.53 / 0.34, hold 0.96 / 0.94 /
  0.94, sell 1.29 / 1.32 / 1.35, use-then-sell 1.02 / 1.18 / 1.42 (superflex 1.11 / 1.22 / 1.52). No call changes side; the only one
  that moves much is use-then-sell, fair over three seasons and a clear pass over five. By group the same: WRs 28-29 1.12 / 1.24 /
  1.52, RBs 26-27 1.22 / 1.29 / 1.39, rookie WRs 0.76 / 0.66 / 0.55, QBs 25- 0.69 / 0.65 / 0.54; WRs 26-27 stay fair (0.92 / 1.02 /
  1.00). This is VORP by window; the league sim keeps title odds for three seasons only.
- Four cohorts, groups chosen after seeing the rules: read the direction, halve the size.

## Caveats

- **Position-level residuals depend on league format.** The QB results flip sign between FP (1QB) and MFL, whose keeper pool includes superflex leagues. Within-position groupings are more trustworthy.
- **Multiple comparisons.** Many groupings were tested, so prefer patterns that hold across markets and years over single starred cells.
- **Sections 1–12 and 14–18 were computed before FP dynasty gained its t = 2018 cohort** (and before the superflex t = 2019 fix in section 2). Rerunning `market_residuals.py` with them moved FP's group residuals by a median 2–3 points, at most 15, and left every other market unchanged. Those sections still describe FP from 2019 onward; sections 13 and 19 use the extended data.
- **Few cohorts in the dynasty value markets.** KTC, FantasyCalc and DP cover 2020–2024, so they overlap heavily. A single season (such as 30+ RBs Henry and Barkley in 2024) can drive the h=1 cells. Multi-year windows also overlap across cohorts.
- **Dynasty Daddy snapshots are a single date (May 16)**, after the NFL draft. Values before each source went live are placeholders and are dropped. 1QB runs use `*_trade_value`; `LEAGUE=sf` runs use `*_sf_trade_value`.
- **Current position.** The panel labels every player by his current position, and VORP for all his seasons uses that position's replacement level. A few converts (Waller, Logan Thomas, Cordarrelle Patterson) are mis-scored in early seasons. Section 12 groups rookies by draft-year position.
- **Unreliable rookie projections.** Rookie rows have low R² (about 0.13), so values like the 2026 rookie projections in `outputs/projections_after_2025.csv` should not be trusted.
- **The 2026 season is ignored.** It is in progress, and the current projections use the snapshot after 2025.
