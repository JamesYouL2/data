# Dynasty buy / sell

**Site:** https://jamesyoul2.github.io/dynasty-buy-sell/

Enter a Sleeper league ID (or a username), pick a team, and get a buy, hold, sell or cut call for every player on the
roster. The call depends on whether you are rebuilding or contending. The page also lists buy targets on other teams
and free agents worth a roster spot.

Everything runs in the browser. The page reads your league straight from the public
[Sleeper API](https://docs.sleeper.com/), and nothing is sent anywhere else.

## Where the calls come from

The rules in [build/rules.py](build/rules.py) come from the
[career_value backtests](https://github.com/JamesYouL2/data/tree/career-value/career_value), which are built on the
[DynastyProcess data repo](https://github.com/dynastyprocess/data). The backtests compare FantasyPros dynasty rankings
(2018 onward) with the value above replacement (VORP) each player went on to score. A player's call depends on:

- **Price:** overall FantasyPros dynasty rank, in 1QB or superflex. Ranks 151-239 cost about a 3rd-round rookie pick,
  and 240-300 about a 4th.
- **Age entering next season:** rookie, 25 or younger, 26-29, or 30+.
- **NFL draft capital:** rounds 1-3, or round 4 or later and undrafted.
- **Production:** PPR points per game this season. For players 30+, it is blended with last season.

Players aged 30+ are compared with how similar players produced over the next 3 and 5 seasons from 1999 on
([build/old_lookup.json](build/old_lookup.json)).

A contender keeps anyone who is starting for them now ("Use, then sell"), even if a rebuilder should sell him.

## In-season adjustments (not backtested)

On top of the backtested call, the page adjusts for things the rankings miss during the season. These are judgment
calls, and each one shows as a note under the call:

- **Trade market.** Each player also shows his [FantasyCalc](https://fantasycalc.com) rank, which comes from real
  trades, and the rookie pick closest in value. When the market is 12+ spots (or 25%) away from FantasyPros, a note says
  whether that makes it a better time to buy or to sell.
- **Injuries** come from Sleeper. A player who is out and has dropped off the FantasyPros rest-of-season rankings is
  treated as out for the season. A contender gets **Stash** instead of Hold or Buy. A rebuilder gets **Buy low
  (injured)** if the market has marked him down, or **Sell when healthy** instead of Sell. Other injured players get a
  note with their rest-of-season rank.
- **Rest of season.** FantasyPros ROS and weekly rankings show each player's role this year and next week's matchup and
  projection. A contender's Sell or Cut becomes "Use, then sell" if FantasyPros ranks him a starter for the rest of the
  season. A "Use, then sell" becomes Sell if he is ranked below twice the starter line.
- **This season section.** Contenders get *win-now targets*: healthy players on other teams whose FantasyCalc
  rest-of-season value is high for their dynasty value. Rebuilders get *sell to contenders*: their own players worth
  more this year than in dynasty terms, plus the teams likeliest to buy.

## Daily data

[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) runs every day at 11:17 UTC. It downloads the inputs,
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

```sh
bash build/fetch.sh raw
python -I build/build_data.py raw data/players.js     # needs pandas and pyarrow
python -m http.server                                  # then open http://localhost:8000
```

`build/old_lookup.json` changes once a season. To regenerate it from the career_value dataset:

```sh
python -I tools/make_lookup.py ~/data/career_value/data/player_seasons.parquet ~/data/career_value/raw/players/players.parquet
```
