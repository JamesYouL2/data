# Dynasty buy / sell

**Site:** https://jamesyoul2.github.io/data/

Enter a Sleeper league ID (or a username), pick a team, and get a buy, hold, sell or cut call for every player on the
roster, in three modes: **Default** (most teams), **Contending** and **Rebuilding**. The page also lists buy targets on other teams
and free agents worth a roster spot.

Everything runs in the browser. The page reads your league straight from the public
[Sleeper API](https://docs.sleeper.com/), and nothing is sent anywhere else.

## Where the calls come from

The rules in [build/rules.py](build/rules.py) come from the
[career_value backtests](../career_value), which are built on the
[DynastyProcess data repo](https://github.com/dynastyprocess/data). The backtests compare FantasyPros dynasty rankings
(2018 onward) with the value above replacement (VORP) each player went on to score. A player's call depends on:

- **Price:** overall FantasyPros dynasty rank, in 1QB or superflex, in bands of 1-50, 51-100, 101-200 and 201-300.
  Past 300 is waiver level. The bands used to follow rookie-pick prices (151-239 a 3rd, 240-300 a 4th). Where merging
  them joined bands with different calls, [rebucket.py](../career_value/analysis/rebucket.py) picked the call from
  what the whole new band returned: Rebuilding by the three-year measure below, Contending by next season's VORP. The league
  simulation then checked the biggest picks (the docstring lists them), and
  [holdout.py](../career_value/analysis/holdout.py) checks every call on 2020-23 against 2018-20.
- **Age entering next season:** rookie, 25 or younger, 26-29, or 30+.
- **NFL draft capital:** rounds 1-3, or round 4 or later and undrafted.
- **Production:** PPR points per game this season. For players 30+, it is blended with last season.

Buys and plain sells come in two strengths each, from
[career_value/analysis/call_strength.py](../career_value/analysis/call_strength.py). For each group of players with the
call, split by price (FantasyPros rank 1-50, 51-100, 101-200, 201+; top 340, 2018 on), it compares what they returned
with what the average player at the same price returned, which is what a fair trade gets you on average. It also
checks the group without its best player, since a cheap group's average often rests on one hit. A group with under 8
players at that price borrows the record of all players with the call at the same position and age class (rookie, 25
or younger, 26-29, 30+) and price, then at that position and age class at any price, then of that age class across RB,
WR and TE. QBs beat their price as a group, so they aren't pooled with the others. Past that, the note says there are
too few to tell. A leave-one-out check found no pooling order clearly better than another: a borrowed record is weak
evidence, and the note says when one is used.

- **Sells** use three seasons of VORP plus what the player was worth after them, in every mode (when each mode sells
  comes from the league simulation below; this only says how hard). **Sell: heavy** groups returned under 65% of their
  price: take any reasonable offer. **Sell: light**: sell at the market price, and hold if no one pays it.
- **Buys** are judged by what each mode plays for: Rebuilding by the three-year measure, Default by next season's VORP
  plus value a year on, Contending by next season's VORP. **Buy: strong** groups returned at least 1.5 times their
  price, at least their price without the best player, and at least 1 in 4 beat their price: pay the market price, or
  a little more. Default is a strong buy wherever Rebuilding and Contending both are. **Buy: weak**:
  market price or below. A buy that returned under 90% of its price for that mode becomes Hold. The old "Buy with a 3rd /
  4th" and "Sell for a 3rd / 4th" calls are now plain buys and sells with a strength; each row shows the rookie pick
  closest to the player's market price.

- **Hold or Lottery ticket** is set by price alone. A non-QB kept past FantasyPros rank 200 who isn't producing is a
  **Lottery ticket**: three in four players priced 201-300 returned under a quarter of their price over 3 years (about
  half at 101-150). Inside 200 it is **Hold**. QBs (backups kept as bye-week and injury cover) stay Hold at any price.
  No lottery ticket is a buy: rookie WRs priced 201-300 returned 3-4 times their price, and superflex Day-3 rookie RBs
  2.7 times, but in the league simulation buying them gained on the 2018-20 classes (McLaurin, Diontae Johnson) and
  lost on 2021-24. Two are sells for rebuilders, since selling them gained in both halves: young Day-3 RBs priced
  201-300 (about half their price over 3 years; the 2020-21 ones almost nothing) and rookie TEs priced 201-300 (54-74%,
  under a third without Dawson Knox). Contenders keep the RBs as Lottery tickets and sell the TEs too unless one is
  producing.

[qb_sells.py](../career_value/analysis/qb_sells.py) checks every QB call this way, at 1, 3 and 5 years. Five QB sells
failed and are now holds: young superflex QB1s, starting QBs 30+, 1QB QBs 26-29 priced 101-200, 1QB QBs 36+ and
superflex backup QBs 30+.

A check of every call group against two rules of thumb (young players: keep unless a sale returns more; old players:
contenders buy, rebuilders sell) added three buys, each kept only because the league simulation gained in both the
2018-20 and the 2021-24 classes:

- 1QB contenders buy young NFL R1-3 WRs priced 101-200: they returned 1.58 times their price in next season's VORP.
- Superflex rebuilders buy rookie WRs priced 51-100: they returned 1.62 times their price over 3 years.
- 1QB rebuilders buy rookie TEs priced 101-200: they returned 1.84 times their price over 3 years, and 5 of 8 beat it.
  Only 8 players, the thinnest record behind any buy. In superflex the gain came only from the 2018-20 classes, so
  superflex holds.

Three other changes gained in the simulation only through the 2018-20 classes and lost on 2021-24, so they were
dropped:

- 1QB rebuilders buying Day-2+ rookie RBs priced 101-200.
- Selling young Day-3 WRs priced 101-200.
- Buying 1QB rookie WRs priced 51-100.

Buying superflex Day-3 WRs priced 201-300 lost outright.

The exceptions to the rules of thumb stay:

- Contenders cut fading 30+ RBs and WRs priced 201-300 (nearly all returned nothing; the averages rest on one or two
  players).
- Old 1QB QBs that contenders hold aren't buys (they failed on the later classes).
- Rebuilders sell young TEs priced 201-300.

In superflex, the market prices RBs and WRs about right. Apart from rookie WRs, no RB or WR group returned more than
its price in both halves of the sample, which is why the superflex buys are mostly QBs.

Players aged 30+ are compared with how similar players produced over the next 3 and 5 seasons from 1999 on
([build/old_lookup.json](build/old_lookup.json)).

### Three modes

The backtests give two calls per player, one for a rebuilder and one for a contender. The page shows three modes,
built in `classify()`:

- **Rebuilding:** the rebuilder call. It sells anything that loses value, even a starter.
- **Default:** the contender call while a player starts for you, the rebuilder call once he doesn't. "Starting" means
  points per game at or inside the starter line: QB12 (QB24 in superflex), RB24, WR30, TE12. Before week 4, last
  season's rank counts. If the rebuilder sells and the contender holds, Default plays him through your playoffs and
  then sells ("Use, then sell"), or sells now if he isn't starting. If rebuilders sell and contenders buy, it holds. If
  the contender buys and the rebuilder holds or buys, it buys.
- **Contending:** Default's rule for players the rebuilder sells, plus the contender's buys and holds everywhere else.

[career_value/analysis/middle_mode.py](../career_value/analysis/middle_mode.py) tests the three in a 12-team league
simulation. Teams draft the FantasyPros top 340 (2018-24 classes), play real PPR scoring for the next five seasons, and
follow each mode's calls with equal-price trades. The old contender calls kept fading players who weren't starting.
Default and the new Contending won more titles over seasons 1-3 than those old calls. Contending won the most in
season 1; Default won about as many over seasons 1-3 and kept far more dynasty value. Rebuilding kept the most dynasty
value and is the call for a team playing for year four and later.

## In-season adjustments (not backtested)

On top of the backtested call, the page adjusts for things the rankings miss during the season. These are judgment
calls, and each one shows as a note under the call:

- **Trade market.** Each player also shows his [FantasyCalc](https://fantasycalc.com) rank, which comes from real
  trades, and the rookie pick closest in value. When the market is 12+ spots (or 25%) away from FantasyPros, a note says
  whether that makes it a better time to buy or to sell.
- **Injuries** come from Sleeper. A player who is out and has dropped off the FantasyPros rest-of-season rankings is
  treated as out for the season. Default and Contending get **Stash** instead of Hold or Buy. A rebuilder gets **Buy low
  (injured)** if the market has marked him down. Every mode gets **Sell when healthy** instead of Sell. Other injured players get a
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
on the page's calls with no signals. To hold, a rule had to gain at least 2.5 standard errors on its team's own objective at both
prices, and be above zero in every class but one. The objectives are title odds this season for Contending, title odds over three
seasons for Default and dynasty value a year on for Rebuilding. Default and Rebuilding rules also had to lose no value at 2.5 standard
errors.

| Signal | Call | Who gets it | Gain on the objective (title points, KTC / FantasyCalc) |
|---|---|---|---|
| Rest-of-season positional rank a quarter worse than the trade market's | Sell | Contending | +6.1 / +2.4 this season (1QB), +6.1 / +3.9 (superflex) |
| Rest-of-season rank a quarter worse than the preseason | Sell | Contending, superflex | +4.0 / +3.7 this season |
| 25 or younger and not a rookie, positional rest-of-season rank a quarter better than dynasty | Buy | Contending and Default, 1QB | +3.2 / +4.0 this season; Default +6.3 / +11.7 over three seasons |

A rule that held on its own also had to help the set it joins, with each rule left out of the set in turn. Two sells passed alone
for 1QB Contending and failed that: the sell on a rest-of-season rank a quarter worse than the preseason, and the one on a dynasty
rank a quarter worse than the preseason. Once the market sell and the buy were in place, each cost title odds this season (1.1 to
1.3 and 0.4 to 1.5 points), so 1QB doesn't use them.

A fall since the preseason also has to cost at least 6 rest-of-season places, and a rank below the market's at
least 3 positional places: near the top of the list a quarter can be a single place. Players off the rest-of-season list get no
signal: the Stash and Sell when healthy calls cover them.

The Contending sells trade dynasty value for this season. The sell on a rank below the market's cost 21 to 39 of value a year on, and
most of those players are rookies or 25 and under whose price held up the next May. The note on the page says so.

What didn't hold, and is no longer on the page:

- **Every signal for Rebuilding.** Each sell and each buy lost dynasty value a year on, most of them at 4 standard errors or more.
- **The sells for Default.** The preseason-fall sell added title odds over three seasons at FantasyCalc prices but not at KTC's, where it
  also lost value a year on. The other two sells cleared the bar at neither price.
- **The young-player buy in superflex.** It lost 37 to 73 of value a year on for every kind of team and didn't clear the bar on titles.
- **The rookie buy**, and the dynasty-fall sell in both formats.
- **The exceptions** the earlier rules carried (1QB QBs, hurt young players, healthy players for 1QB contenders). The test sold and
  bought every flagged player, so the rules that held don't need them.

A signal becomes **Sell: light** or **Buy: weak**. It leaves Sell and Cut calls alone and doesn't raise a sell to heavy or a buy to
strong. A sell signal also replaces "Use, then sell", because the simulated team sold starters too. Where a signal meets the opposite
call (a sell signal on a Buy, or a buy signal on a Sell, Cut or Use, then sell), or a sell and a buy signal fire together, the call
becomes Hold. Those Hold rules are judgment calls, as is naming what to take back in a trade.

The limits: the signals were found on these same seasons, so this is not out of sample; the standard errors treat simulated leagues
as independent when leagues from one class share players, so they are too small; anything over three seasons rests on 2 or 3
classes; and the simulation assumes someone accepts every equal-price trade.

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
| `values-players.csv` (FantasyPros dynasty rankings) | DynastyProcess | weekly |
| `db_playerids.csv` (Sleeper / FantasyPros / NFL id map) | DynastyProcess | weekly |
| `players.parquet` (birth dates, draft round) | nflverse | daily |
| `stats_player_reg_<season>.parquet` (this season and last) | nflverse | daily in season |
| `db_fpecr_latest.csv` (FantasyPros rest-of-season ranks), `fp_latest_weekly.csv` | DynastyProcess | daily |
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

`build/old_lookup.json` changes once a season. To regenerate it from the career_value dataset:

```sh
python -I tools/make_lookup.py ../career_value/data/player_seasons.parquet ../career_value/raw/players/players.parquet
```
