# Dynasty buy / sell

**Site:** https://jamesyoul2.github.io/data/

Enter a Sleeper league ID (or a username), pick a team, and get a buy, hold, sell or cut call for every player on the
roster, in three modes: **Default** (most teams), **Contending** and **Rebuilding**. The page also lists buy targets on other teams
and free agents worth a roster spot.

Everything runs in the browser. The page reads your league straight from the public
[Sleeper API](https://docs.sleeper.com/), and nothing is sent anywhere else.

## Where the calls come from

Every call comes from seven rules in [build/rules.py](build/rules.py), by position, age, price and whether the player is
producing. Price is the FantasyPros dynasty rank in 1QB or superflex. Age is his age next September. Producing means PPR points per
game at or inside the starter line: QB12 (QB24 in superflex), RB24, WR30, TE12; before week 4, last season's rank counts.

1. **Sell RBs and WRs 26 or older, and TEs 30 or older, priced outside the top 100.**
2. **Sell a producing veteran when his price allows:** after the season for RBs 26-27 and WRs 28-29, whose price lasts a year
   (Rebuilding sells now); now for RBs 28-29 and WRs 30+, whose price doesn't; now for RBs 30+ and TEs 29+, except that
   Contending holds them.
3. **A top-100 RB at 26-29 who isn't producing:** Contending holds, Default sells after the season, Rebuilding sells now.
4. **Buy** top-100 rookie WRs (Default, Rebuilding), rookie TEs (Rebuilding) and top-100 TEs 30 or older (Contending, Default).
5. **Sell anyone priced outside the top 200 who isn't a rookie or a QB.** Past 300 he is waiver level: cut.
6. **Sell RBs 25 or under who aren't producing.** Contending and Default hold the late-round ones (NFL round 4 or later, or undrafted) priced 101-200: holding them added title odds over three seasons in both formats.
7. **QBs:** buy at 26-29 ranked QB25-32, the fringe starters (by QB rank, so the same players in both formats); sell at 33 or older.

Everyone else is a Hold. The rules are the same in both formats.

### Three modes

**Contending** is a team projected to finish 1st to 3rd, **Rebuilding** 10th to 12th, **Default** everyone between. The rules never
push a player out harder for a contender than for a rebuilder, and Default's call is always between the other two: the build checks
it, and where an in-season adjustment moves one neighbour past Default, Default takes that neighbour's call.

Default is the right mode for most teams. In the league simulation a team projected in the top three finished there 45% of the
time, and a top-three finisher repeated 41% of the time.

### How they were tested

[career_value/analysis/ruleset_sim.py](../career_value/analysis/ruleset_sim.py) runs the rules in a 12-team league simulation: teams
draft the FantasyPros top 340 (2018-24 classes), play real PPR scoring, and each team follows the calls for its own mode with
equal-price trades. The page used to carry a much longer list of calls by price band, NFL draft capital and production tier. Against
that list, the seven rules:

- gained dynasty value for every mode in both formats (three years on: +110 to +517, in 3 or 4 of 4 classes);
- gained title odds over three seasons for Default in both formats (+11.3 and +17.8 points) and Contending in superflex (+22.5);
- cost Contending and Default about 2 title points in the first season in 1QB, and Rebuilding 4.6 over three seasons in superflex.

Each rule was first tested alone, against trading the same number of random players at the same price
([sell_rules_sim.py](../career_value/analysis/sell_rules_sim.py)), and the sale dates in rule 2 against holding the player
([sell_timing_sim.py](../career_value/analysis/sell_timing_sim.py)). Lists of rules picked by the data were also tried, with each
class's rules read from the other classes only: they kept about a quarter of what they seemed to gain, and the short reasoned list
did better. [career_value/README.md](../career_value/README.md) sections 31 to 35 have the numbers.

The limits: the rules were written after looking at these classes, so this isn't a clean test on unseen seasons; the simulation
trades at FantasyPros prices and assumes someone accepts every equal-price trade; three-year results rest on 4 or 5 classes. Two
things the page used to do lost in the same tests and are gone: selling rookie RBs in the top 100, and buying WRs and RBs 25 or
under priced 101-200.

## In-season adjustments (not backtested)

On top of the backtested call, the page adjusts for things the rankings miss during the season. These are judgment
calls, and each one shows as a note under the call:

- **Trade market.** Each player also shows his [FantasyCalc](https://fantasycalc.com) rank, which comes from real
  trades, and the rookie pick closest in value. When the market is 12+ spots (or 25%) away from FantasyPros, a note says
  whether that makes it a better time to buy or to sell.
- **Injuries** come from Sleeper. A player who is out and has dropped off the FantasyPros rest-of-season rankings is
  treated as out for the season. RBs 25 or under get **Stash** instead of Hold or Buy; everyone else gets **Sell in the offseason**: the trade market held an injured player's price until May, and he then returned about three quarters of it, so selling in the offseason was worth about a third more than keeping (`injured_buy.py`, `own()`). There is no buy-low call on injured
  players: in the backtest (`career_value/analysis/injured_buy.py`) players out for the season returned 0.63 / 0.53 of their price
  over the next season, and the ones the market had marked down, which the site used to tell rebuilders to buy, 0.22 / 0.45. Other injured players get a
  note with their rest-of-season rank.
- **Rest of season.** FantasyPros ROS and weekly rankings show each player's role this year and next week's matchup and
  projection. In Default and Contending, a Sell or Cut becomes "Use, then sell" if FantasyPros ranks him a starter for the rest of the
  season. A "Use, then sell" becomes Sell if he is ranked below twice the starter line.
- **This season section.** Default and Contending get *win-now targets*: healthy players on other teams whose FantasyCalc
  rest-of-season value is high for their dynasty value. Rebuilders get *sell to contenders*: their own players worth
  more this year than in dynasty terms, plus the teams likeliest to buy.

## In-season signals (league simulation)

From week 5 to week 14, a few rank moves change the call. Each shows as a note. Every rule is one signal for one kind of team, with
no exceptions by position, age or injury.

A rule is on the page only if it held in [simple_rules_sim.py](../career_value/analysis/simple_rules_sim.py). That is the league
simulation at KTC and FantasyCalc prices, with one team trading after week 4, 8 or 12 over the 2021-25 seasons, and the signal laid
on the page's calls with no signals. To hold, a rule had to gain at least 2 standard errors on its team's own objective at both
prices, and be above zero in every class but one. Team types are by the projection for the season: Contending 1st-3rd, Default
4th-9th, Rebuilding 10th-12th. The objectives are title odds over this season and next for Contending (or this season and all
three), title odds over three seasons for Default and dynasty value a year on for Rebuilding. Default and Rebuilding rules also had
to lose no value at 2.5 standard errors. The bar is 2 standard errors because
[calls built at each bar](../career_value/analysis/threshold_sim.py) did best in the league simulation at about 2: at 2.5 and 3 they
gave up a quarter to two thirds of the gain.

| Signal | Call | Who gets it | Gain on the objective (title points, KTC / FantasyCalc) |
|---|---|---|---|
| Rest-of-season positional rank a quarter worse than the trade market's | Sell | Contending | +9.9 / +3.9 over two seasons (1QB), +10.0 / +3.6 (superflex) |
| Rest-of-season rank a quarter worse than the preseason | Sell | Contending, superflex | +5.8 / +4.3 over two seasons |
| 25 or younger and not a rookie, positional rest-of-season rank a quarter better than dynasty | Buy | Contending and Default, 1QB | Contending +3.2 / +4.0 this season, +6.6 / +10.0 over three; Default +5.8 / +10.9 over three |

A rule that held on its own also had to help the set it joins, with each rule left out of the set in turn. Two sells passed alone
for 1QB Contending and failed that: the sell on a rest-of-season rank a quarter worse than the preseason, and the one on a dynasty
rank a quarter worse than the preseason. Once the market sell and the buy were in place, each cost title odds this season (1.1 to
1.3 and 0.4 to 1.5 points), so 1QB doesn't use them.

A fall since the preseason also has to cost at least 6 rest-of-season places, and a rank below the market's at
least 3 positional places: near the top of the list a quarter can be a single place. Players off the rest-of-season list get no
signal: the Stash and Sell in the offseason calls cover them.

The Contending sells trade dynasty value for this season. The sell on a rank below the market's cost 21 to 39 of value a year on, and
most of those players are rookies or 25 and under whose price held up the next May. The note on the page says so.

What didn't hold, and is no longer on the page:

- **Every signal for Rebuilding.** Each sell and each buy lost dynasty value a year on, most of them at 4 standard errors or more.
- **The sells for Default.** The preseason-fall sell added title odds over three seasons at FantasyCalc prices but not at KTC's, where it
  also lost value a year on. The other two sells cleared the bar at neither price.
- **The young-player buy in superflex.** It lost 37 to 73 of value a year on for every kind of team and didn't clear the bar on titles.
- **The rookie buy**, and the dynasty-fall sell in both formats. The rookie buy cleared the bar for 1QB Default alone, but neither Contending
  nor Rebuilding buys those players, and Default's call has to sit between theirs.
- **The exceptions** the earlier rules carried (1QB QBs, hurt young players, healthy players for 1QB contenders). The test sold and
  bought every flagged player, so the rules that held don't need them.

A signal becomes a **Sell** or a **Buy**. It leaves Sell and Cut calls alone. A sell signal also replaces "Use, then sell", because the simulated team sold starters too. Where a signal meets the opposite
call (a sell signal on a Buy, or a buy signal on a Sell, Cut or Use, then sell), or a sell and a buy signal fire together, the call
becomes Hold. Those Hold rules are judgment calls, as is naming what to take back in a trade.

The limits: the signals were found on these same seasons, so this is not out of sample; the standard errors treat simulated leagues
as independent when leagues from one class share players, so they are too small; anything over three seasons rests on 2 or 3
classes; and the simulation assumes someone accepts every equal-price trade.

## The Rankings tab

A second tab (`#rankings`) lists the FantasyPros top 340 for each team type and format, re-ordered by the calls. Each player's
FantasyPros rank is multiplied by a factor for his call and the list is sorted on the result (`rankList` in `app.js`):

| | Buy | Use, then sell | Sell or Cut |
|---|---|---|---|
| Contending | 0.9 | 1.05 | 1.1 |
| Default | 0.7 | 1.3 | 1.15 |
| Rebuilding | 0.85 | n/a | 1.2 |

A factor is how far down the list the call belonged: the one multiplier on rank that prices everyone the seven rules gave that call
fairly, by what they returned in 2018-24 (`career_value/analysis/rank_factors.py`; one season for Contending, three for Default and
Rebuilding), divided by the same figure for Holds. Measured: Contending 0.80 / 1.09 / 1.17, Default 0.50 / 1.65 / 1.37, Rebuilding
0.70 / 1.39, close to equal in 1QB and superflex. The site uses the square root of each, half the measured move, because the rules
were chosen on the seasons the factors are measured on. Default's Buys are 28-36 player-seasons. Because rule 5 sells most non-rookies past 200, the Holds down there (rookies,
QBs) rise about 50 places without any rule of their own.

## The 2026 test

The backtests above use seasons that helped build them. The test in
[frozen_test_2026.py](../career_value/analysis/frozen_test_2026.py) doesn't: its spec and predictions were committed
([ec155b1](https://github.com/JamesYouL2/data/commit/ec155b1)) before any 2026 result was scored. It asks whether production to
date, injury status or FantasyPros rest-of-season ranks add to the current KTC or FantasyCalc rank, in 1QB and superflex, on value
over the rest of the season (January 2027) and the market rank next May (May 2027). The
[screen](../career_value/analysis/variable_screen.py) that picked those variables used seasons through 2020 and no KTC or
FantasyCalc price. One season can't change a call: a pass goes on to 2027 and a league simulation.

The page shows `build/frozen_test.json`, which the page build reads as is because it has no scikit-learn.
[frozen-test-2026.yml](../.github/workflows/frozen-test-2026.yml) does the rest on a schedule: week 12 predictions after the Dec 4 ranks,
rest-of-season results in mid-January and next-May results in mid-May. Each run commits to `career-value` and redeploys the page. To run
one by hand: `gh workflow run frozen-test-2026.yml --repo JamesYouL2/data`. The interim table on the page is a refit on the weeks played
so far, not the saved predictions, and it isn't the test.

## Daily data

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) (at the repo root) runs every day at 11:17 UTC. It downloads the inputs,
rebuilds `data/players.js`, and deploys the site to GitHub Pages. A schedule only fires from the default branch, so `career-value` is
the default. `master` holds nothing of ours: [`sync-upstream.yml`](../.github/workflows/sync-upstream.yml) fast-forwards it to
[dynastyprocess/data](https://github.com/dynastyprocess/data) every day at 10:47 UTC. The page doesn't read `master`; merge it into
`career-value` when the analyses need newer `files/`. The page's inputs:

| Input | Source | Updated |
|---|---|---|
| `values-players.csv` (player list; dynasty ranks only for players off FantasyPros' pages) | DynastyProcess | weekly |
| `db_playerids.csv` (Sleeper / FantasyPros / NFL id map) | DynastyProcess | weekly |
| `players.parquet` (birth dates, draft round) | nflverse | daily |
| `stats_player_reg_<season>.parquet` (this season and last) | nflverse | daily in season |
| `db_fpecr_latest.csv` (FantasyPros dynasty ranks, 1QB and superflex, and rest-of-season ranks), `fp_latest_weekly.csv` | DynastyProcess | daily |
| `db_fpecr.parquet` (FantasyPros rank history, about 40 MB: preseason ranks for the in-season signals) | DynastyProcess | weekly |
| `fc_1qb.json`, `fc_sf.json` (trade market and rest-of-season values) | FantasyCalc API | live |
| `sleeper_players.json` (injuries, teams), `sleeper_state.json` | Sleeper API | live |

The last three rows are optional. If a download fails, the site builds without that data.

If upstream changes a file under `.github/workflows`, the sync fails because the workflow's token can't push workflow files. Run
`gh repo sync JamesYouL2/data --branch master` by hand then.

GitHub turns off scheduled workflows in a repository that has had no activity for 60 days. If that happens, re-enable
the workflows from the Actions tab.

## Running it locally

From this `pages/` folder:

```sh
bash build/fetch.sh raw
python -I build/build_data.py raw data/players.js     # needs pandas and pyarrow
python -m http.server                                  # then open http://localhost:8000
```

`build/old_lookup.json` is no longer read by the page: the seven rules don't use it. The analyses of the earlier calls in
`career_value` still do. To regenerate it from the career_value dataset:

```sh
python -I tools/make_lookup.py ../career_value/data/player_seasons.parquet ../career_value/raw/players/players.parquet
```
