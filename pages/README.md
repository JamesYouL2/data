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

- **Price:** overall FantasyPros dynasty rank, in 1QB or superflex. Ranks 151-239 cost about a 3rd-round rookie pick,
  and 240-300 about a 4th.
- **Age entering next season:** rookie, 25 or younger, 26-29, or 30+.
- **NFL draft capital:** rounds 1-3, or round 4 or later and undrafted.
- **Production:** PPR points per game this season. For players 30+, it is blended with last season.

Plain sells come in two strengths, from
[career_value/analysis/sell_strength.py](../career_value/analysis/sell_strength.py). For each group of players with a Sell
call (FP top 200, 2018-21), it adds three years of VORP to what they were worth three years on and compares that with
the average player at the same price, which is what a fair trade gets you on average. **Sell: heavy** groups returned
under 65% of that: take any reasonable offer. **Sell: light** groups returned 65% or more: sell at the market price, and
hold if no one pays it. Each note also gives the share of players who beat their price: a few QB groups average over
100% only because of one or two big hits (Tom Brady's 40s), while most of their players returned less, so they stay
light sells. [qb_sells.py](../career_value/analysis/qb_sells.py) checks every QB call this way, at 1, 3 and 5 years, for
each mode and format. Two QB sells failed and are now holds: young superflex QB1s and starting QBs 30+.

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

## Daily data

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) (at the repo root) runs every day at 11:17 UTC. It downloads the inputs,
rebuilds `data/players.js`, and deploys the site to GitHub Pages:

| Input | Source | Updated |
|---|---|---|
| `values-players.csv` (FantasyPros dynasty rankings) | DynastyProcess | weekly |
| `db_playerids.csv` (Sleeper / FantasyPros / NFL id map) | DynastyProcess | weekly |
| `players.parquet` (birth dates, draft round) | nflverse | daily |
| `stats_player_reg_<season>.parquet` (this season and last) | nflverse | daily in season |
| `db_fpecr_latest.csv` (FantasyPros rest-of-season ranks), `fp_latest_weekly.csv` | DynastyProcess | daily |
| `fc_1qb.json`, `fc_sf.json` (trade market and rest-of-season values) | FantasyCalc API | live |
| `sleeper_players.json` (injuries, teams), `sleeper_state.json` | Sleeper API | live |

The last three rows are optional. If a download fails, the site builds without that data.

GitHub turns off scheduled workflows in a repository that has had no activity for 60 days. If that happens, re-enable
the workflow from the Actions tab.

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
