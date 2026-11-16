"""Buy / hold / sell rules for dynasty players.

classify(x, f, ctx) takes one player row and a format ("1qb" or "sf") and returns
(group, (rebuilder action, why), (contender action, why), (default action, why)). The calls are seven rules by position, age,
price and whether he is producing, the same in both formats (career_value/README.md section 35, analysis/ruleset_sim.py).
situational() then adjusts for the trade market, injuries, rest-of-season ranks and the in-season signals.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

START = {"1qb": {"QB": 12, "RB": 24, "WR": 30, "TE": 12}, "sf": {"QB": 24, "RB": 24, "WR": 30, "TE": 12}}

# ---------- rules ----------
BUY, HOLD, HOLDF, LOTTO, SELLH, SELL, USE, CUT = (
    "Buy", "Hold", "Hold to February", "Lottery ticket", "Sell high now", "Sell", "Use, then sell", "Cut")
STASH, SELLHEALTHY = "Stash", "Sell when healthy"
HEAVY, LIGHT = "Sell: heavy", "Sell: light"

STRONG, WEAK = "Buy: strong", "Buy: weak"

# In season, the price the calls use is the FP dynasty rank tilted a little toward FP's rest-of-season (ROS) rank:
# rank x (ROS positional rank / dynasty positional rank) ^ ROS_TILT, off the ROS list = ROS rank 500
# (career_value/analysis/inseason_movers.py, 2019-25: in-season dynasty ranks players by what they return better than the
# preseason one at every week; this tilt adds a little on top for the 1-year and 3-year views, fit at 0.1 in both
# halves and both formats; contenders use the same small tilt).
ROS_TILT = 0.1
ROS_OFF = 500


def ros_tilt(pr, ros_p):
    """factor on the FP overall and positional rank, in season (None / NaN ranks: no tilt)"""
    if pr is None or (isinstance(pr, float) and math.isnan(pr)) or pr <= 0:
        return 1.0
    rp = ROS_OFF if ros_p is None or (isinstance(ros_p, float) and math.isnan(ros_p)) else ros_p
    return (rp / pr) ** ROS_TILT


def price_bucket(r):
    """FP overall rank -> 1-50, 51-100, 101-200, 201+ (unranked counts as 201+)"""
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "201+"
    return "1-50" if r <= 50 else "51-100" if r <= 100 else "101-200" if r <= 200 else "201+"


def age_class(x, ctx=None):
    a = x.age
    return ("rookie" if ctx and x.dy == ctx["ROOKIE"] else "25-" if not pd.isna(a) and a < 26
            else "30+" if not pd.isna(a) and a >= 30 else "26-29")


def kind(a):
    """Buy* -> buy, Hold* / Stash -> hold, Use -> use, Sell* -> sell, Lottery -> lotto, Cut -> cut"""
    if not a:
        return "none"
    return ("buy" if a.startswith("Buy") else "hold" if a.startswith("Hold") or a == STASH else "use" if a == USE
            else "sell" if a.startswith("Sell") else "lotto" if a == LOTTO else "cut" if a == CUT else "none")


def producing(x, f, ctx):
    """(PPG positional rank at or inside the starter line, that rank). Until anyone has 4 games this season
    (ctx EARLY), last season's rank stands in."""
    rk = x.prk
    if pd.isna(rk) and ctx.get("EARLY"):
        rk = getattr(x, "prk_last", float("nan"))
    return bool(pd.notna(rk) and rk <= START[f][x.position]), rk


def band(r):
    """FP overall rank -> price band. Past 300 is waiver level."""
    if r is None or math.isnan(r) or r > 300:
        return "300+"
    return "1-50" if r <= 50 else "51-100" if r <= 100 else "101-200" if r <= 200 else "201-300"


def f0(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0f}"


# ---------- the seven rules ----------
# In the league sim (career_value/analysis/ruleset_sim.py: 12 teams, FP top 340, real scoring, 2018-24) these seven rules alone beat
# the backtested calls they replaced on dynasty value for every team type in both formats (3 years on: +110 to +517), and on title odds
# over three seasons for Default in both formats and Contending in superflex. They cost 1QB Contending and Default about 2 title points in
# the first season. Team types: Contending = projected 1st-3rd, Default 4th-9th, Rebuilding 10th-12th. "Producing" = points per game at
# or inside the starter line (START). Ages are at the September of next season. The push to sell never falls from Contending to
# Rebuilding, so Default's call is always between the other two (between() checks it).
#   1  Sell RBs and WRs 26 or older, and TEs 30 or older, priced outside the top 100.
#   2  A producing veteran is sold when his price allows (analysis/sell_timing_sim.py): after the season for RBs 26-27 and WRs 28-29
#      (Rebuilding: now), whose price lasts a year; now for RBs 28-29 and WRs 30+, whose price doesn't; now for RBs 30+ and TEs 29+,
#      except that Contending holds them: selling cost title odds at any date.
#   3  A top-100 RB at 26-29 who isn't producing: Contending holds, Default sells after the season, Rebuilding sells now.
#   4  Buy top-100 rookie WRs (Default, Rebuilding), rookie TEs (Rebuilding) and top-100 TEs 30 or older (Contending, Default).
#   5  Sell anyone priced outside the top 200 who isn't a rookie or a QB. Past 300 he is waiver level: cut.
#   6  Sell RBs 25 or under who aren't producing. Contending and Default hold the late-round ones (NFL round 4 or later, or undrafted)
#      priced 101-200: holding them added title odds over three seasons in both formats (ruleset_sim.py GAP, EIGHT: +3.6 / +1.5
#      Contending, +5.9 / +3.8 Default) at little or no cost in value. Early-round ones at that price are busts: sell.
#   7  QBs: buy at 26-29 ranked QB25-32, the fringe starters; sell at 33 or older. The band is by QB rank so that it names the same
#      players in both formats (overall, QB25-32 is about 160-265 in 1QB and 75-150 in superflex). Held three years they returned 2.5x
#      / 2.2x the average player at their price, 1.8x in both without the best one (Jared Goff 2021); QB33 and deeper returned under
#      half, most of them nothing (scratch check on call_strength.horizons, 12 player-seasons).
#      The site reads the 1QB list's QB rank for both formats: in the backtest the two lists put every QB 26-29 in the same band.
# Everyone else is a Hold.
TOP, DEEP, WAIVER = 100, 200, 300
QB_BUY = (25, 32)                                # rule 7's buy band, by QB rank
MODES3 = ("reb", "con", "mid")
PUSH = {"buy": 0, "hold": 1, "lotto": 1, "none": 1, "use": 2, "sell": 3, "cut": 3}      # how hard a call pushes a player out


def between(con, mid, reb):
    """Default's (action, why), moved to the nearer neighbour's if it falls outside Contending's and Rebuilding's"""
    pc, pm, pr = (PUSH[kind(c[0])] for c in (con, mid, reb))
    lo, hi = (con, reb) if pc <= pr else (reb, con)
    return lo if pm < min(pc, pr) else hi if pm > max(pc, pr) else mid


def classify(x, f, ctx, split=True):
    """-> (group, (reb action, why), (con action, why), (default action, why)). split is unused: it marked the old calls' strength
    split, and the analyses still pass it."""
    pos, a = x.position, x.age
    r = getattr(x, f"r_{f}")
    ranked = not (r is None or pd.isna(r) or r > WAIVER)
    rookie = x.dy == ctx["ROOKIE"]
    prod, rk = producing(x, f, ctx)
    top = ranked and r <= TOP
    out = ranked and r > TOP                      # priced outside the top 100, and on the list
    g = f"{'Rookie ' if rookie else ''}{pos}{'' if rookie else ' ' + age_class(x, ctx)}, {band(r)}"
    now = f"{pos}{f0(rk)} by points per game" if pd.notna(rk) else "no points-per-game rank yet"

    def call(m):
        if not ranked:
            if rookie:
                return HOLD, "A rookie who isn't ranked yet: keep him if you have the spot."
            return CUT, "Past 300 in the rankings is waiver level."
        if pos == "QB":
            qr = getattr(x, "qbr", None)                  # the 1QB list's QB rank in both formats, so both name the same QBs
            qr = getattr(x, f"pr_{f}", None) if qr is None or pd.isna(qr) else qr
            if not rookie and 26 <= a < 30 and qr is not None and QB_BUY[0] <= qr <= QB_BUY[1]:
                return BUY, ("Rule 7: QBs at 26-29 ranked QB25-32 returned about twice their price over three years in both formats. "
                             "They are fringe starters the market has stopped paying for, and some get a job back. Deeper backups "
                             "mostly returned nothing.")
            if a >= 33:
                return SELL, ("Rule 7: QBs at 33 or older lost value fast in both formats. Sell while someone still pays.")
            return HOLD, "QBs hold their value: no rule says to move him."
        if rookie:
            if pos == "WR" and top and m != "con":
                return BUY, ("Rule 4: top-100 rookie WRs gained both title odds over three seasons and dynasty value. They cost "
                             "about a title point in their first season.")
            if pos == "TE" and m == "reb":
                return BUY, "Rule 4: rookie TEs gained dynasty value for rebuilding teams. They don't help this season."
            return HOLD, "A rookie: no rule says to move him."
        if pos == "TE" and a >= 30 and top and m != "reb":
            return BUY, ("Rule 4: a top-100 TE at 30 or older added title odds over the next two seasons in every cohort, with no "
                         "dynasty value lost.")
        if prod:
            if (pos == "RB" and 26 <= a < 28) or (pos == "WR" and 28 <= a < 30):
                if m == "reb":
                    return SELL, f"Rule 2: producing now ({now}), and his price falls from here. Rebuilding teams sell now."
                return USE, (f"Rule 2: producing now ({now}), and his price lasts another year. Selling now cost about 3 title points "
                             "this season; selling after the season cost none and kept the value.")
            if (pos == "RB" and 28 <= a < 30) or (pos == "WR" and a >= 30):
                return SELL, (f"Rule 2: producing now ({now}), but his price doesn't last: a year later most of it was gone. Sell now, "
                              "not after the season.")
            if (pos == "RB" and a >= 30) or (pos == "TE" and a >= 29):
                if m == "con":
                    return HOLD, (f"Rule 2: producing now ({now}). Selling him cost a contender title odds at any date and brought "
                                  "little back: keep him until he stops producing.")
                return SELL, f"Rule 2: producing now ({now}), but there is little price left to wait for. Sell now."
        if out and ((pos in ("RB", "WR") and a >= 26) or (pos == "TE" and a >= 30)):
            return SELL, ("Rule 1: an RB or WR at 26 or older, or a TE at 30 or older, priced outside the top 100 gained dynasty value "
                          "sold, for every kind of team in both formats, without costing title odds.")
        if top and pos == "RB" and 26 <= a < 30:
            if m == "con":
                return HOLD, "Rule 3: a top-100 RB at 26-29. Selling one cost a contender title odds this season: keep him."
            if m == "mid":
                return USE, ("Rule 3: a top-100 RB at 26-29 who isn't producing. Selling now cost a title point this season; selling "
                             "after it cost nothing and kept most of the value.")
            return SELL, "Rule 3: a top-100 RB at 26-29 who isn't producing. Rebuilding teams sell now, while the price is there."
        if r > DEEP:
            return SELL, ("Rule 5: priced outside the top 200 and not a rookie. Selling these players gained more dynasty value than "
                          "any other rule, for every kind of team in both formats.")
        if pos == "RB" and a < 26 and not prod:
            if m != "reb" and out and getattr(x, "rd", 0) >= 4:
                return HOLD, ("Rule 6: a late-round RB at 25 or under priced 101-200. He isn't producing yet, but holding RBs like him "
                              "added title odds over three seasons in both formats. Early-round RBs at this price are the ones to sell.")
            return SELL, (f"Rule 6: an RB at 25 or under who isn't producing ({now}). RBs lose value from the start, and these gained "
                          "dynasty value sold.")
        return HOLD, "None of the seven rules applies: keep him."

    reb, con, mid = call("reb"), call("con"), call("mid")
    return g, reb, con, between(con, mid, reb)


# ---------- market, injury and rest-of-season notes (not backtested) ----------
OUT_STATUSES = ("IR", "PUP", "Out", "Sus", "DNR", "NA")


def gap(r, m):
    """'more' if the trade market prices the player well above FantasyPros, 'less' if well below, else None."""
    if r is None or m is None or (isinstance(r, float) and math.isnan(r)) or (isinstance(m, float) and math.isnan(m)):
        return None
    if min(r, m) > 300:
        return None
    t = max(12, 0.25 * r)
    return "more" if m <= r - t else "less" if m >= r + t else None


# In-season trade signals. Each is one flag for one kind of team, with no exemptions by position, age or injury. A rule is here
# only if it held in the league sim at trade-market prices (career_value/analysis/simple_rules_sim.py: the flag laid on these calls
# with no signals, trades after weeks 4, 8 and 12, seasons 2021-25): a gain of at least 2 standard errors on the team type's own
# objective at both KTC and FantasyCalc prices, above zero in every cohort but one. Team types by the May projection: Contending
# 1st-3rd, Default 4th-9th, Rebuilding 10th-12th. Objectives: Contending, title odds over this season and next (or this season and
# all three), with no title measure 2 standard errors down; Default, title odds over three seasons; Rebuilding, value a year on.
# Default and Rebuilding rules also had to lose no value at 2.5 standard errors. Gains are title points, KTC / FantasyCalc.
#   ros_low   ROS positional rank a quarter worse than the trade market's: sell. Contending, two seasons +9.9 / +3.9 (1QB),
#             +10.0 / +3.6 (superflex).
#   ros_down  ROS rank a quarter worse than the preseason: sell. Contending in superflex, two seasons +5.8 / +4.3.
#   gap       25 or younger and not a rookie, ROS positional rank a quarter better than dynasty: buy. 1QB: Contending this season
#             +3.2 / +4.0 and three seasons +6.6 / +10.0, Default three seasons +5.8 / +10.9.
#   (A rookie with that gap cleared the bar for 1QB Default alone, three seasons +3.3 / +9.2. It isn't used: neither Contending nor
#   Rebuilding buys him, and Default's call has to sit between theirs.)
# A rule that held alone also had to help the set it joins (signal_sim.py, each rule left out of the set in turn). In 1QB, ros_down
# (+4.7 / +5.5 alone) and faller, the sell on a dynasty rank a quarter worse than the preseason (+2.3 / +3.3 alone), each cost
# Contending title odds this season once ros_low and the buy were in place (-1.1 / -1.3 and -0.4 / -1.5), so neither is used there.
# The Contending sells trade value for this season: ros_low cost 21 to 39 of value a year on (2.3 to 4.4 standard errors).
# What didn't hold: every rule for Rebuilding (each sell and buy lost value a year on, most at 4 standard errors or more), the
# sells for Default, the gap buy in superflex (it lost 37 to 73 of value a year on for every team type), the rookie buy in
# superflex (value three years on down 2.8 standard errors), and faller in superflex. signals() still reports faller for the backtests.
# Players off the ROS list get no signal: mostly out for the year, and Stash / Sell when healthy cover them.
# A quarter is one place near the top of the list (#3 to #4), so a move also has to cover a floor of places.
SIG_MOVE = 4 / 3
SIG_FLOOR_ROS, SIG_FLOOR_DYN = 6, 3             # places lost since the preseason, on top of the quarter
SIG_FLOOR_LOW = 3               # places below the market's positional rank, on top of the quarter
SIG_WEEKS = (5, 14)             # Sleeper's current week: at least 4 played, and no later than the sim's last window
SIG_POOL = 340                  # the sim's tradeable pool
SIG_HURT = ("IR", "PUP", "Out")
SIG_RULES = {"ros_low": {"1qb": ("con",), "sf": ("con",)}, "ros_down": {"sf": ("con",)}, "faller": {}, "gap": {"1qb": ("con", "mid")},
             "rookie": {}}


def signals(x, f, ctx):
    """-> what fired for this player and format: ros_down / faller (preseason rank, rank now), ros_low (the market's positional
    rank, ROS positional rank), gap (dynasty positional rank, ROS positional rank), and rookie, hurt, young for the backtests"""
    wk = ctx.get("WEEK") if ctx else None
    r, ros_r = getattr(x, f"r_{f}"), getattr(x, "ros_r", None)
    if not wk or not SIG_WEEKS[0] <= wk <= SIG_WEEKS[1] or pd.isna(r) or r > SIG_POOL or ros_r is None or pd.isna(ros_r):
        return {}
    s = {}

    def fell(a, b, floor):                           # NaN compares false: no preseason rank, no signal
        return b / a >= SIG_MOVE and b - a >= floor

    if fell(x.ro_pre, x.ro_now, SIG_FLOOR_ROS):
        s["ros_down"] = (x.ro_pre, x.ro_now)
    pre, now = (x.do_pre, x.do_now) if f == "1qb" else (x.dsf_pre, x.dsf_now)
    if fell(pre, now, SIG_FLOOR_DYN):
        s["faller"] = (pre, now)
    rookie = x.dy == ctx["ROOKIE"]
    s["hurt"], s["young"] = x.inj in SIG_HURT, bool(rookie or x.age - 1 < 26)
    # the backtests' age is at the September of the season played; x.age is at next September
    if (rookie or x.age - 1 < 26) and getattr(x, f"pr_{f}") / x.ros_p >= SIG_MOVE:
        s["gap"], s["rookie"] = (getattr(x, f"pr_{f}"), x.ros_p), bool(rookie)
    mp = getattr(x, f"mp_{f}", None)                 # NaN compares false: off the market's list or the ROS list, no signal
    if mp is not None and x.ros_p / mp >= SIG_MOVE and x.ros_p - mp >= SIG_FLOOR_LOW:
        s["ros_low"] = (mp, x.ros_p)
    return s


TAKE_BACK = {"con": "Trade him for a player at his price who ranks better for the rest of the season."}


def apply_signal(act, mode, x, f, s):
    """(action, note) for one mode, from the rules in SIG_RULES. A sell or buy overrides a Hold-type call; where it meets the
    opposite backtested call, or a sell and a buy fire together, the player is held."""
    def on(k):
        return k in s and mode in SIG_RULES[k].get(f, ())
    sells = []
    if on("ros_down"):
        sells.append(f"FantasyPros' rest-of-season rank fell from #{f0(s['ros_down'][0])} in the preseason to "
                     f"#{f0(s['ros_down'][1])}, and dynasty prices lag that.")
    if on("faller"):
        sells.append(f"His dynasty rank fell from #{f0(s['faller'][0])} in the preseason to #{f0(s['faller'][1])}.")
    if on("ros_low"):
        sells.append(f"FantasyPros ranks him {x.position}{f0(s['ros_low'][1])} for the rest of the season, but the trade market "
                     f"prices him as {x.position}{f0(s['ros_low'][0])}: he won't score like his price this season.")
    gap = s["gap"] if "gap" in s and mode in SIG_RULES["rookie" if s.get("rookie") else "gap"].get(f, ()) else None
    gtxt = (f"FantasyPros ranks him {x.position}{f0(gap[1])} for the rest of the season but {x.position}{f0(gap[0])} in "
            "dynasty, and buying young players like this gained in the league sim") if gap else ""
    k = kind(act)
    if sells and gap:
        return (act if k == "cut" else HOLD), " ".join(sells) + f" Still, {gtxt}: mixed signals, so hold."
    if sells:
        cost = " Selling on this added title odds this season in the league sim and cost dynasty value. "
        if k == "buy":
            return HOLD, " ".join(sells) + f" The usual call is {act.lower()}, so hold and don't add more."
        return (act if k in ("sell", "cut") else SELL), " ".join(sells) + cost + TAKE_BACK[mode]
    if gap:
        if k == "buy":
            return act, gtxt + "."
        if k in ("sell", "cut", "use"):
            return HOLD, gtxt + f". The usual call is {act.lower()}, so hold rather than sell."
        return BUY, gtxt + ": buy at the market price, not above."
    return act, ""


def settle(g, ra, rw, ca, cw, rn, cn, ma, mw, mn):
    """the ten fields, with Default kept between Contending and Rebuilding after the adjustments too: where a signal or an injury moves
    one neighbour past it, Default takes the nearer neighbour's call, reason and note"""
    act, why, note = between((ca, cw, cn), (ma, mw, mn), (ra, rw, rn))
    return (g, ra, rw, ca, cw, rn, cn, act, why, note)


def situational(x, f, call, ctx=None):
    """Adjust the backtested calls for the trade market (FantasyCalc), injuries (Sleeper), rest-of-season value and
    the in-season signals.

    Returns (group, reb action, reb why, con action, con why, reb note, con note, default action, default why,
    default note). The notes say what moved and why; only the signals come from a backtest (the league sim). The
    default call is adjusted like the contender call: both care about this season."""
    g, (ra, rw), (ca, cw), (ma, mw) = call
    r, m = getattr(x, f"r_{f}"), getattr(x, f"mr_{f}")
    rv = getattr(x, f"rv_{f}")
    inj = x.inj if isinstance(x.inj, str) else None
    out = inj in OUT_STATUSES
    # out for the season: injured and dropped from the FantasyPros rest-of-season ranks (x.gone, set by the build);
    # without FP ranks, a FantasyCalc rest-of-season value of zero
    gone = getattr(x, "gone", None)
    if gone is None or (isinstance(gone, float) and math.isnan(gone)):
        gone = rv is not None and not (isinstance(rv, float) and math.isnan(rv)) and rv == 0
    season_over = out and bool(gone)
    gp = gap(r, m)
    where = f"FP #{f0(r)}, market #{f0(m)}"
    if season_over:
        what = f"{x.position} on {inj}" + (f" ({x.inj_part})" if isinstance(x.inj_part, str) else "") + " and off the rest-of-season rankings"

        def sell_later():
            return SELLHEALTHY, f"{what}. Selling now takes an injury discount ({where}); sell once he's playing again."
        # No "buy low" on the injured: players out for the season returned 0.63 / 0.53 of their FP price over the next season, the ones
        # the market had marked down 0.22 / 0.45, and those 25 or under ranked in the top 50 0.65 / 0.62
        # (career_value/analysis/injured_buy.py). Every team type keeps him (Stash) or waits to sell.
        def now(act):
            if kind(act) in ("hold", "buy"):
                return STASH, (f"{what}: no help this season. Keep him for next year if you have the IR slot; "
                               f"trading him now sells at an injury discount ({where}).")
            if kind(act) in ("sell", "use"):
                return sell_later()
            return act, f"{what} ({where})."
        (ra, rn), (ca, cn), (ma, mn) = now(ra), now(ca), now(ma)
        return settle(g, ra, rw, ca, cw, rn, cn, ma, mw, mn)
    # contender / default: rest-of-season role. FP's ROS positional rank decides whether he starts for you this year.
    rp = getattr(x, "ros_p", None)
    rp = None if rp is None or (isinstance(rp, float) and math.isnan(rp)) else rp
    line = START[f][x.position]

    def ros(act):
        if not out and rp is not None and rp <= line and (kind(act) == "sell" or act in (CUT, LOTTO)):
            return USE, f"FantasyPros ranks him {x.position}{f0(rp)} for the rest of the season: start him through your playoffs, then: {act.lower()}."
        if not out and act == USE and (rp is None and getattr(x, "gone", False) is True or rp is not None and rp > 2 * line):
            return SELL, (("FantasyPros no longer ranks him for the rest of the season" if rp is None else
                           f"FantasyPros ranks him only {x.position}{f0(rp)} for the rest of the season") + ": not a starter, so sell now.")
        return act, ""
    (ca, cros), (ma, mros) = ros(ca), ros(ma)
    sig = signals(x, f, ctx)
    ra, rsig = apply_signal(ra, "reb", x, f, sig)
    ca0, ma0 = ca, ma
    ca, csig = apply_signal(ca, "con", x, f, sig)
    ma, msig = apply_signal(ma, "mid", x, f, sig)
    cros, mros = ("" if ca != ca0 else cros), ("" if ma != ma0 else mros)   # the signal replaced the rest-of-season call

    def mkt_note(act):
        if gp == "more":
            if act.startswith("Sell") or act in (CUT, USE, LOTTO):
                return f"The trade market pays more than the rankings ({where}): sell at the market price."
            if act.startswith("Hold"):
                return f"The trade market pays more than the rankings ({where}): sell if someone offers that."
            return f"The trade market prices him above the rankings ({where}): buying at market price costs more."
        if gp == "less":
            if act.startswith("Buy"):
                return f"The trade market is cheaper than the rankings ({where}): buy at the market price."
            if act.startswith("Hold"):
                return (f"The trade market is cheaper than the rankings ({where}): selling gets less than the rankings "
                        "suggest, and adding more costs less.")
            if act.startswith("Sell") or act == USE:
                return f"The trade market has already marked him down ({where}): you'll get less than the rankings suggest."
            return f"Market #{f0(m)}, FP #{f0(r)}."
        return ""
    rn, cn, mn = ((a + " " + b + " " + mkt_note(c)).strip().replace("  ", " ")
                  for a, b, c in ((rsig, "", ra), (csig, cros, ca), (msig, mros, ma)))
    if out:
        rr = getattr(x, "ros_r", None)
        tag = (f"Currently {inj}" + (f" ({x.inj_part})" if isinstance(x.inj_part, str) else "")
               + (f"; FP rest-of-season #{f0(rr)}" if rr is not None and not math.isnan(rr) else "")
               + (", so little help this season" if rr is not None and not math.isnan(rr) and rr > 150 else "") + ".")
        rn, cn, mn = ((tag + " " + n).strip() for n in (rn, cn, mn))
    return settle(g, ra, rw, ca, cw, rn, cn, ma, mw, mn)
