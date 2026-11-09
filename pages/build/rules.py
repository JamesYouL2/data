"""Buy / hold / sell rules for dynasty players, from the career_value backtests (DynastyProcess data repo,
career_value/README.md sections 6 and 12, plus the deep-roster, pick-price and 30+ analyses).

classify(x, f, ctx) takes one player row and a format ("1qb" or "sf") and returns
(group, (rebuilder action, why), (contender action, why), (default action, why)). The default ("middle") call and the
contender rule that a fading player the rebuilder sells is sold unless he's starting now come from
career_value/analysis/middle_mode.py. ctx carries ROOKIE (latest rookie class) and LOOK
(the 30+ history lookup from old_lookup.json).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

START = {"1qb": {"QB": 12, "RB": 24, "WR": 30, "TE": 12}, "sf": {"QB": 24, "RB": 24, "WR": 30, "TE": 12}}

TIERS = {"QB": [0, 6, 12, 20, 999], "RB": [0, 6, 12, 24, 999], "WR": [0, 6, 12, 24, 999], "TE": [0, 3, 6, 12, 999]}
TLAB = ["elite", "top", "starter", "bench"]


def tier(pos, rank):
    if rank is None or (isinstance(rank, float) and math.isnan(rank)):
        return "bench"
    return TLAB[int(np.searchsorted(TIERS[pos], rank, side="left")) - 1]


def ageband(a):
    return "30-31" if a < 32 else "32-33" if a < 34 else "34+"


# ---------- rules ----------
BUY, HOLD, HOLDF, LOTTO, SELLH, SELL, USE, CUT = (
    "Buy", "Hold", "Hold to February", "Lottery ticket", "Sell high now", "Sell", "Use, then sell", "Cut")
STASH, BUYLOW, SELLHEALTHY = "Stash", "Buy low (injured)", "Sell when healthy"
HEAVY, LIGHT = "Sell: heavy", "Sell: light"

STRONG, WEAK = "Buy: strong", "Buy: weak"

# How hard a call pushes (career_value/analysis/call_strength.py, FP top 340, 2018-21 / 2018-23): per format, the plain
# sells, and the buys by mode, each keyed "group | price" and stored as [keep, beat, n, key used, keep without the best
# player]. keep = what players like this returned as a share of what the average player at their price did. Price is
# the FP rank: 1-50, 51-100, 101-200, 201+. A cell with too few players uses a pool: "all RB 26-29 | 101-200", then
# "all RB 26-29" (any price), then "all 26-29" (RB / WR / TE; QBs aren't pooled with them); past that there are too few players to say.
_SJ = Path(__file__).with_name("strength.json")
STRENGTH = json.loads(_SJ.read_text()) if _SJ.exists() else {}
HEAVY_BELOW = 0.65              # sells: below this, take any fair offer
STRONG_AT = 1.5                 # buys: at or above this, and still >= 1 without the best player, pay up
STRONG_BEAT = 0.25              # ... and at least this share of the players beat their price
BUY_HOLD_BELOW = 0.9            # buys: below this the buy isn't backed for that mode: hold
LENS = {"keep3": "3-year VORP plus value three years on", "keep1": "next season's VORP plus value a year on",
        "prod1": "next season's VORP"}
BUY_LENS = {"reb": "keep3", "mid": "keep1", "con": "prod1"}


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


def lookup(tab, g, pos, ac, pb):
    for k in (f"{g} | {pb}", f"all {pos} {ac} | {pb}", f"all {pos} {ac}", f"all {ac}" if pos != "QB" else None):
        if k in tab:
            return tab[k]
    return None


AGES = {"rookie": "rookies", "25-": "25 or younger", "26-29": "26-29", "30+": "30+"}


def _who(e, g, pb):
    """'Players like this (FP 101-200)' or, for a pool, which players it is"""
    at = f"FP {pb}"
    if e[3] == f"{g} | {pb}":
        return f"Players like this ({at})"
    key, _, price = e[3].partition(" | ")
    parts = key.split()[1:]
    if len(parts) == 2:
        who = f"rookie {parts[0]}s" if parts[1] == "rookie" else f"{parts[0]}s {AGES[parts[1]]}"
    else:
        who = "rookies" if parts[0] == "rookie" else f"players {AGES[parts[0]]}"
    where = f" at {at}" if price else " at any price"
    return f"Too few past players like this at his price, so this is all {who}{where} with this call ({e[2]}). They"


def strength(f, g, pos, ac, pb):
    """(Sell: heavy or Sell: light, why prefix) for a plain Sell in group g"""
    e = lookup(STRENGTH.get(f, {}).get("sell", {}), g, pos, ac, pb)
    if e is None:
        return LIGHT, "Too few past players like this to say how hard to sell: sell at the market price. "
    k, b, _, _, solo = e
    pct, beat, who = f"{100 * k:.0f}%", f"{100 * b:.0f}%", _who(e, g, pb)
    said = f"{who} returned {pct} of what the average player at their price did ({LENS['keep3']}), and {beat} beat it"
    if k < HEAVY_BELOW:
        return HEAVY, said + ": take any reasonable offer, even below market. "
    if k < 1 or solo < 1:
        hits = (f" One hit lifts the group; without it, {100 * solo:.0f}%." if k >= 1 else "")
        return LIGHT, said + f".{hits} Sell at the market price, hold if no one pays it. "
    return LIGHT, said + ", so selling is a close call: sell only at the market price. "


def buy_strength(f, mode, g, pos, ac, pb):
    """(Buy: strong, Buy: weak or Hold, why prefix) for any Buy call in mode reb / mid / con"""
    e = lookup(STRENGTH.get(f, {}).get("buy", {}).get(mode, {}), g, pos, ac, pb)
    if e is None:
        return WEAK, "Too few past players like this to say how hard to buy: buy at the market price, not above. "
    k, b, _, _, solo = e
    lens = LENS[BUY_LENS[mode]]
    pct, beat, who = f"{100 * k:.0f}%", f"{100 * b:.0f}%", _who(e, g, pb)
    if k < BUY_HOLD_BELOW:
        return HOLD, (f"{who} returned only {pct} of what the average player at their price did ({lens}): "
                      "not worth the market price. Keep him if you have him. ")
    if k >= STRONG_AT and solo >= 1 and b >= STRONG_BEAT:
        return STRONG, (f"{who} returned {pct} of what the average player at their price did ({lens}), "
                        f"{100 * solo:.0f}% even without the best one, and {beat} beat it: pay the market price, or a little more. ")
    hits = (f" Much of that is one hit; without it, {100 * solo:.0f}%." if k >= 1 and solo < 1 else
            " A few players carry it." if k >= STRONG_AT and b < STRONG_BEAT else "")
    return WEAK, (f"{who} returned {pct} of what the average player at their price did ({lens}), "
                  f"and {beat} beat it.{hits} Buy at the market price or below, not above. ")


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


def base_calls(x, f, ctx):
    """The backtested calls: (group, (reb action, why), (con action, why)), before classify's contender rule.

    Price bands are FP overall rank 1-50, 51-100, 101-200, 201-300 and 300+ (waiver level). Where the old rookie-pick
    bands (101-150, 151-215, 216-239, 240-300) merged with different calls, the call comes from
    career_value/analysis/rebucket.py: the rebuilder's by 3-year return (keep3), the contender's by next season's
    VORP (prod1), each against the average player at the same price."""
    ROOKIE, look = ctx["ROOKIE"], ctx["LOOK"]
    pos, a, r, pr, rd = x.position, x.age, getattr(x, f"r_{f}"), getattr(x, f"pr_{f}"), x.rd
    sf = f == "sf"
    bd = band(r)
    ranked = bd != "300+"
    top200 = bd in ("1-50", "51-100", "101-200")
    rookie = x.dy == ROOKIE
    young = (not rookie) and a < 26
    early = rd <= 3
    prod, prk = producing(x, f, ctx)
    prodr = x.prod_rank
    cap = "NFL R1-3" if early else "NFL R4+/UDFA"

    def G(fam):
        return f"{fam}, {bd}"

    def con_prod(act, why):                       # contenders keep whoever is starting for them now
        if prod and act in (SELL, CUT):
            return (USE, f"Producing now ({pos}{f0(prk)} by PPG). Start him through the playoffs, then: {act.lower()}. " + why)
        return (act, why)

    if math.isnan(a) and not rookie:
        a = 27.0
    # ---------------- rookies ----------------
    if rookie:
        if pos == "QB":
            if sf and pd.notna(pr) and pr <= 6:
                w = "Superflex QBs 24 or younger priced QB1-6 returned -63% / -48% of their price in years 1-2."
                return ("Rookie QB priced as a QB1", (SELL, w), (SELL, w))
            w = ("Rookie QBs ran -35% / -11% / -32% against their superflex price: hold your own, don't pay full price."
                 if sf else "Rookie QBs who slip to rookie rounds 3-4 in 1QB were the best value of the draft (Hurts, Herbert).")
            return ("Rookie QB", (HOLD, w), (HOLD, w))
        if pos == "RB":
            if rd <= 1:
                w = "NFL first-round RBs are the only rookie RBs worth their price. RB prices peak in November, so hold unless you get a big offer."
                return ("Rookie RB, NFL R1", (HOLD, w), (HOLD, w))
            if bd in ("1-50", "51-100"):
                w = "Rookie RB prices peak in November and Day-2+ RBs trail their price: sell into the peak."
                return (f"Rookie RB, NFL R{rd:.0f}", (SELLH, w), con_prod(SELLH, w) if not prod else
                        (HOLD, f"Producing now ({pos}{f0(prk)}). " + w))
            if bd == "101-200":                   # rebucket.py: keep3 1.67 / 1.14, prod1 0.44 / 0.34
                w = ("Day-2+ rookie RBs priced 101-200 kept their value over 3 years but produced little in year 1. "
                     + ("Rebuilders hold to summer; contenders sell into the November peak." if sf else
                        "Hold to summer, contenders too: the 2020-21 rookies returned more than their price over 3 years."))
                # holdout.py: the 1QB contender Sell high now failed on 2020-21 (keep3 1.61 / 1.71 without the best).
                # league sim, 1QB Contending ranks 4-8, titles yrs 1-3: Sell high now 15.6, Use then sell 15.4, Hold 15.9
                return ("Rookie RB, NFL R2+, 101-200", (LOTTO, w), (SELLH if sf else HOLD, w) if not prod else
                        (HOLD, f"Producing now ({pos}{f0(prk)}). " + w))
            w = "Day-2+ rookie RBs priced 201+: Chase Brown and Kyren Williams came from here; most return nothing."
            return (f"Rookie RB, {cap}", (LOTTO, w), (LOTTO, w))
        if rd >= 5 and not top200:                # rebucket.py: the line moved from 240 to 200
            w = ("Late rookies priced 201+ returned almost nothing in year 1, but 11-22% of those ranked 250+ got a "
                 "camp-hype price bump: hold to summer, sell into hype.")
            return (f"Rookie {pos}, NFL R5+/UDFA", (LOTTO, w), (CUT if not prod else HOLD, "Cut if you need the roster spot. " + w))
        w = ("Rookie WRs rise ~9% from now to February." if pos == "WR" else
             "Rookie TEs rise ~26% from now to February; TEs taken in rookie rounds 3-4 returned almost nothing after that." if pos == "TE"
             else "")
        if sf and pos == "WR" and bd == "51-100":  # call_strength.py: keep3 1.62 / 1.31 without the best (n=12)
            # league sim, superflex: Default value 3 years on +154-176 and 5-year titles +16-18 points, in both cohort halves
            return (f"Rookie {pos}", (BUY, "Superflex rookie WRs priced 51-100 returned 1.6 times their price over 3 years. " + w),
                    (HOLDF, w + " Sell after February if you need vets."))
        if not sf and pos == "TE" and bd == "101-200":  # call_strength.py: keep3 1.84 / 0.96 without the best (n=8)
            # league sim, 1QB ranks 4-8: Rebuilding value 3 years on +63 / +53, 2018-20 / 2021-24 cohorts; superflex
            # gained only in 2018-20 (+108 / +2), so it holds. A contender Buy added little (prod1 0.81).
            return (f"Rookie {pos}", (BUY, "1QB rookie TEs priced 101-200 returned 1.8 times their price over 3 years, and 5 "
                                           "of 8 beat it (Hockenson, Kmet, Freiermuth, McBride). " + w),
                    (HOLDF, w + " Sell after February if you need vets."))
        if pos == "TE" and bd == "201-300":    # call_strength.py: keep3 0.54 / 0.18 without the best (1QB), 0.74 / 0.29 (SF)
            # league sim, ranks 4-8: Rebuilding value 3 years on +74 / +111 (1QB) and +31 / +57 (SF), 2018-20 / 2021-24.
            # Contenders sell too unless he's producing (classify's contender rule): +19 / +28 (1QB), 0 / -8 (SF)
            return (f"Rookie {pos}", (SELL, "Rookie TEs priced 201-300 returned about 54-74% of their price over 3 years; "
                                            "under a third without Dawson Knox. " + w + " Sell then, or sooner at market price."),
                    (HOLDF, w + " Sell after February if you need vets."))
        return (f"Rookie {pos}", (HOLDF, w), (HOLDF, w + " Sell after February if you need vets."))
    # ---------------- 30+ ----------------
    if a >= 30:
        t = tier(pos, prodr)
        v3, v5, y5, n = look.get((f, pos, ageband(a), t), (0.0, 0.0, 0.0, 0))
        hist = f"30+ {pos}s like this ({ageband(a)}, {t} tier) averaged {f0(v3)} VORP over 3 yrs, {f0(v5)} over 5; {f0(100 * y5)}% still useful in year 5 (n={n})."
        if pos == "QB":
            if a >= 36:
                w = f"QBs 34+ keep producing, but at {a:.0f} retirement ends it: a 1-2 season asset. " + hist
                if not sf and ranked:              # call_strength.py: 1QB keep3 1.43, 1.13 without the best
                    w = ("1QB QBs 36+ kept returning more than the average player at their price (Brady, Rodgers): play "
                         "him out. ") + w
                return ("QB 36+, win-now", (SELL if sf and ranked else HOLD if ranked else CUT, w),
                        (HOLD if t != "bench" else CUT, w))
            if t in ("elite", "top"):
                cheap = pd.notna(pr) and pd.notna(prodr) and pr >= prodr + 6
                w = ("QBs age well: " + hist +
                     (f" Priced QB{pr:.0f} while producing like QB{prodr:.0f}." if cheap else ""))
                return ("QB 30+, long-term", (BUY if cheap else HOLD, w), (BUY, w))
            if t == "starter":               # qb_sells.py: 2/3+ beat their price in 3 years, both formats
                w = "Starting QBs 30+ beat their price: 67-78% returned more than the average player at their price. " + hist
                return ("QB 30+, mid-tier", (HOLD, w), (BUY if sf and not top200 else HOLD, w))
            if sf and top200:                # call_strength.py: SF keep3 1.45, 1.09 without the best
                hist = "Superflex backup QBs 30+ returned more than the average player at their price. " + hist
            elif sf and ranked:              # rebucket.py: SF 201-300 keep3 2.56, 0.10 without the best
                hist = "Superflex backup QBs 30+ priced 201-300: what they returned rests on one player. " + hist
            return ("QB 30+, backup", ((HOLD if sf else SELL) if top200 else CUT, hist),
                    (HOLD if sf and ranked else CUT, ("Bye-week / injury cover. " if sf else "") + hist))
        if v3 >= 40:
            w = "Value is front-loaded, a 2-year window. " + hist
            if bd not in ("1-50", "51-100"):     # rebucket.py: WRs priced 101-200 prod1 1.81 / 2.45
                w += " Priced 101+, players like this out-produced their price the next season."
            return (f"{pos} 30+, 2-year window", (SELL, w), (BUY if bd not in ("1-50", "51-100") else HOLD, w))
        if top200:
            act, w = SELL, hist + " Take any reasonable offer."
        elif ranked and pos == "TE":               # rebucket.py: TE 201-300 keep3 0.54 / 0.52
            act, w = SELL, hist + " TEs like this priced 201-300 still returned about half their price: sell if anyone pays."
        else:                                      # RB 0.00 / 0.00, WR 0.28 / 0.34 (0.03 / 0.08 without the best)
            act, w = CUT, hist + (" Priced 201-300, RBs and WRs like this returned almost nothing." if ranked else "")
        return (f"{pos} 30+, fading", (act, w), con_prod(act, w))
    # ---------------- 26-29 ----------------
    if not young:
        if pos == "QB":
            if sf:
                if bd == "1-50":
                    w = "Superflex QBs priced top 50 hold their value (median rank a year later: #8)."
                    return (G("QB 26-29"), (HOLD, w), (HOLD, w))
                if pd.notna(pr) and 13 <= pr <= 28 and pd.notna(prodr) and pr >= prodr + 6 and a >= 28:
                    w = f"28+ QBs priced QB13-28 beat their price by +75 to +123%; priced QB{pr:.0f}, producing like QB{prodr:.0f}."
                    return ("QB 28-29, underpriced", (BUY, w), (BUY, w))
                if pd.notna(pr) and 13 <= pr <= 20 and a < 28:
                    w = "QBs 25-27 priced QB13-20 returned -44% / -92% / -56% of their price."
                    return ("QB 26-27, priced QB13-20", (SELL, w), con_prod(HOLD, w))
                if bd == "51-100":
                    w = "Superflex QBs 26-29 priced 51-100 kept producing (109 VORP over 3 yrs) but fall to ~#113 next May."
                    return (G("QB 26-29"), (HOLD, w), (HOLD, w))
                if bd == "101-200":               # rebucket.py: keep3 2.26 (1.6 without the best), prod1 2.13
                    w = ("Superflex QBs 26-29 priced 101-200: 86 VORP over 3 yrs, more than twice what the average player "
                         "at their price returned (Mayfield, Darnold).")
                    return (G("QB 26-29"), (BUY, w), (BUY, w))
                if bd == "201-300":               # rebucket.py: keep3 0.61, 0.46 without the best
                    w = "Superflex QBs 26-29 priced 201-300: backups, ~8 VORP over 3 yrs. Bye-week cover for a contender."
                    return (G("QB 26-29"), (CUT, w), (HOLD, w))
                return (G("QB 26-29"), (CUT, "Waiver level."), con_prod(CUT, "Waiver level."))
            if bd in ("1-50", "51-100"):
                w = "1QB QBs 26-29 priced top 100 hold value (rank ~33-94 two years on)."
                return (G("QB 26-29"), (HOLD, w), (HOLD, w))
            if bd == "101-200":
                w = "1QB QBs 26-29 priced 101-200 returned more than the average player at their price over 3 years: keep him."
                return (G("QB 26-29"), (HOLD, w), (HOLD, w))
            if bd == "201-300":                   # rebucket.py: keep3 2.86 (1.74 without the best), prod1 2.80
                w = ("1QB QBs 26-29 priced 201-300: 23 VORP over 3 yrs, almost three times what the average player at "
                     "their price returned (Mayfield, Goff).")
                return (G("QB 26-29"), (BUY, w), (BUY, w))
            return (G("QB 26-29"), (CUT, "Waiver level."), con_prod(CUT, "Waiver level."))
        if pos == "RB":
            if bd == "1-50":
                w = "RBs 26-29 priced top 50 produce (~195-230 VORP over 3 yrs) but sit at a median ~#52-58 next May."
                return (G("RB 26-29"), (SELL, w), (HOLD, w))
            if bd == "51-100":
                w = "RBs 26-29 priced 51-100 fall to ~#117-127 next May and ~#190-205 in 2 years."
                return (G("RB 26-29"), (SELL, w), (HOLD, w))
            if bd == "101-200":
                w = f"RBs 26-29 priced 101-200: {'15' if not sf else '24'} VORP over 3 yrs, median rank next May ~#{'243' if not sf else '242'}."
                return (G("RB 26-29"), (SELL, w), (HOLD, w) if sf else con_prod(SELL, w))
            if bd == "201-300":
                w = f"RBs 26-29 priced 201-300: {'5' if not sf else '3'} VORP over 3 yrs, median rank next May ~#{'291' if not sf else '267'}."
                return (G("RB 26-29"), (SELL, w), con_prod(SELL, w))
            w = "RBs 26-29 ranked past 300: waiver level."
            return (G("RB 26-29"), (CUT, w), con_prod(CUT, w))
        if pos == "WR":
            if bd == "1-50":
                w = "WRs 26-29 priced top 50: ~220 VORP over 3 yrs; median rank next May ~#37-50."
                return (G("WR 26-29"), (HOLD if a < 28 else SELL, w), (HOLD, w))
            if bd == "51-100":
                w = f"WRs 26-29 priced 51-100: ~{'80' if not sf else '138'} VORP over 3 yrs; rank falls to ~#101-106 next May, ~#164 in 2 years."
                return (G("WR 26-29"), (SELL, w), (HOLD, w))
            if bd == "101-200":                   # rebucket.py: keep3 0.44 / 0.66, prod1 1.36 / 1.31
                w = (f"WRs 26-29 priced 101-200: {'14' if not sf else '28'} VORP over 3 yrs, well under what the average "
                     "player at their price returned, but they out-produced their price next season: contenders keep them.")
                return (G("WR 26-29"), (SELL, w), (HOLD, w))
            if bd == "201-300":
                w = "WRs 26-29 priced 201-300: 1-2 VORP over 3 yrs."
                return (G("WR 26-29"), (SELL, w), con_prod(SELL, w))
            w = "WRs 26-29 ranked past 300: waiver level."
            return (G("WR 26-29"), (CUT, w), con_prod(CUT, w))
        # TE
        if bd == "1-50":
            w = "TEs 26-29 priced top 50: ~160-170 VORP over 3 yrs, 86% had a top-6 season."
            return (G("TE 26-29"), (HOLD, w), (HOLD, w))
        if bd == "51-100":
            w = "TEs 26-29 priced 51-100 fall to ~#85-92 next May and ~#130 in 2 years."
            return (G("TE 26-29"), (SELL, w), (HOLD, w))
        if bd == "101-200":                       # rebucket.py: keep3 0.74 / 0.60
            w = "TEs 26-29 priced 101-200: ~15-17 VORP over 3 yrs, under what the average player at their price returned."
            return (G("TE 26-29"), (SELL, w), (HOLD, w))
        w = "TEs 26-29 priced 201+: 0-4 VORP over 3 yrs."     # rebucket.py: 201-300 keep3 0.08 1QB, 0.13 SF without the best
        return (G("TE 26-29"), (CUT, w), con_prod(CUT, w))
    # ---------------- young (25 or younger, not a rookie) ----------------
    if pos == "QB":
        if sf and pd.notna(pr) and pr <= 6 and a < 25:          # qb_sells.py: was Sell; 64% beat their price
            w = ("Young superflex QB1s keep their price: counting what they're still worth three years on, 64% beat the "
                 "average player at their price (Mahomes, Lamar, Herbert).")
            return ("Young QB priced as a QB1", (HOLD, w), (HOLD, w))
        if bd in ("1-50", "51-100"):
            w = "Young QBs priced top 100 hold their value."
            return (G("Young QB"), (HOLD, w), (HOLD, w))
        if early and bd == "101-200":
            w = (f"Young QBs with NFL R1-3 capital priced 101-200: {'58' if sf else '48'} VORP over 3 yrs, more than the "
                 "average player at their price returned.")
            return (G(f"Young QB, {cap}"), (HOLD, w), (HOLD, w))
        if sf and early and bd == "201-300":     # rebucket.py: was Buy (151-300); 201-300 returned ~0 (n=6)
            w = "Young superflex QBs with NFL R1-3 capital priced 201-300: a cheap bet on a starting job, but few have paid off."
            return (G(f"Young QB, {cap}"), (HOLD, w), (HOLD, w))
        w = "Young QBs priced 201+ (or without early capital) returned ~0."
        return (G(f"Young QB, {cap}"), (CUT, w), con_prod(CUT, w))
    if pos == "RB":
        if bd == "1-50":
            w = "Young RBs priced top 50 hold value: rank ~#28-34 next May, ~#51-64 in 2 years."
            return (G("Young RB"), (HOLD, w), (HOLD, w))
        if bd == "51-100":
            w = "Young RBs priced 51-100 fall to ~#101-135 next May and ~#194-200 in 2 years."
            return (G("Young RB"), (SELL, w), (HOLD, w))
        if early:
            if bd == "101-200":                   # rebucket.py: keep3 0.41 / 0.46
                w = f"Young NFL R1-3 RBs priced 101-200: {'11' if not sf else '15'} VORP over 3 yrs, under half what the average player at their price returned."
                return (G(f"Young RB, {cap}"), (SELL, w), con_prod(SELL, w))
            w = "Young NFL R1-3 RBs priced 201+: ~0-3 VORP over 3 yrs. Early picks who fall this far rarely come back."
            return (G(f"Young RB, {cap}"), (CUT, w), con_prod(CUT, w))
        if bd == "101-200":                       # rebucket.py: keep3 1.33 / 1.21; prod1 1.44 (1QB, 1.08 without the best)
            w = f"Young Day-3/UDFA RBs priced 101-200: {'30' if not sf else '36'} VORP over 3 yrs, more than the average player at their price returned."
            if not sf:                            # league sim: a rebuilder Buy cost 1QB value (641 -> 510)
                return (G(f"Young RB, {cap}"), (HOLD, w), (BUY, w + " They out-produced their price next season too."))
            # league sim: a rebuilder Buy lifted SF Rebuilding, but only through the 2018-20 cohorts (Ekeler, Pollard);
            # it lost in 2021-24 (holdout.py), so rebuilders hold
            return (G(f"Young RB, {cap}"), (HOLD, w), (HOLD, w))
        if ranked:                            # call_strength.py: keep3 0.53 / 0.29 without the best (1QB), 0.56 / 0.39 (SF)
            # league sim, ranks 4-8: Rebuilding value 3 years on +43 / +35 (1QB) and +52 / +45 (SF), Default +57-115,
            # 2018-20 / 2021-24 cohorts. Contenders keep the ticket.
            w = ("Young Day-3/UDFA RBs priced 201-300 returned about half their price over 3 years, and the 2020-21 ones "
                 "almost nothing: sell for a pick at market price.")
            return (G(f"Young RB, {cap}"), (SELL, w), (LOTTO, "Young Day-3/UDFA RBs priced 201-300: about half their price "
                                                              "over 3 years. Keep only while he might start for you."))
        w = "Young Day-3/UDFA RB: an opportunity bet. Hold if you have the roster spot."
        return (G(f"Young RB, {cap}"), (LOTTO, w), (LOTTO, w))
    if pos == "WR":
        if bd == "1-50":
            w = "The safest asset in dynasty: young WRs priced top 50 sat at median #25 and were still #28 two years later."
            return (G("Young WR"), (HOLD, w), (HOLD, w))
        if bd == "51-100":
            w = "Young WRs priced 51-100: 54-103 VORP over 3 yrs, about what the average player at their price returned."
            return (G("Young WR"), (HOLD, w), (HOLD, w))
        if bd == "101-200":
            w = ("Young NFL R1-3 WRs priced 101-200: 32-36 VORP over 3 yrs, about what the average player at their price returned." if early
                 else "Young Day-3/UDFA WRs priced 101-200: ~19 VORP over 3 yrs; 60-70% had a starter season, but few later ranked top 100.")
            if early and not sf:                  # call_strength.py: 1QB prod1 1.58 / 1.41 without the best (n=57)
                # league sim, 1QB: Contending and Default value 3 years on +65-93, year-1 titles +0.7-1.6 points; the
                # value gain held in both cohort halves
                return (G(f"Young WR, {cap}"), (HOLD, w), (BUY, w + " In 1QB they out-produced their price the next season."))
            return (G(f"Young WR, {cap}"), (HOLD, w), (HOLD, w))
        if early:
            if bd == "201-300":                   # rebucket.py: keep3 0.31 / 0.57 (0.10 / 0.20 without the best)
                w = "Young NFL R1-3 WRs priced 201-300: 3-4 VORP over 3 yrs; early picks who fall this far rarely come back."
                return (G(f"Young WR, {cap}"), (SELL, w), con_prod(SELL, w))
            w = "Young NFL R1-3 WRs ranked past 300: ~0 VORP over 3 yrs."
            return (G(f"Young WR, {cap}"), (CUT, w), con_prod(CUT, w))
        w = ("Young Day-3/UDFA WRs priced 201-300: 9-13 VORP over 3 yrs, the best price retention in the band (Renfrow, Meyers)."
             if ranked else "Young Day-3/UDFA WR: an opportunity bet. Hold if you have the roster spot.")
        return (G(f"Young WR, {cap}"), (LOTTO, w), (LOTTO, w))
    # TE
    if bd in ("1-50", "51-100"):
        w = "Young TEs priced top 100: 40-110 VORP over 3 yrs."
        return (G("Young TE"), (HOLD, w), (HOLD, w))
    if bd == "101-200":
        if early:                                 # rebucket.py: keep3 1.39 / 1.10
            w = "Young NFL R1-3 TEs priced 101-200: 31-34 VORP over 3 yrs, more than the average player at their price returned."
            return (G(f"Young TE, {cap}"), (HOLD, w), (HOLD, w))
        w = "Young Day-3/UDFA TEs priced 101-200: ~0 VORP over 3 yrs (12-13 players)."
        return (G(f"Young TE, {cap}"), (SELL, w), con_prod(SELL, w))
    if bd == "201-300":
        if early:                                 # rebucket.py: keep3 1.51 / 2.05 resting on one player; prod1 0.48 / 0.45
            w = "Young NFL R1-3 TEs priced 201-300: 9-11 VORP over 3 yrs, mostly from one player, and little next season."
            return (G(f"Young TE, {cap}"), (SELL, w), con_prod(CUT, w))
        # rebucket.py: keep3 1.32 / 1.68 but 0.06 / 0.07 without the best; prod1 3.04 / 3.30 (1.31 / 1.48 without)
        w = ("Young Day-3/UDFA TEs priced 201-300: 7-9 VORP over 3 yrs, almost all from one player (Ferguson, Schultz), "
             "but as a group they out-produced their price next season. Small sample.")
        return (G(f"Young TE, {cap}"), (SELL, w), (BUY, w))
    w = "Young TEs ranked past 300 rarely come back."
    return (G(f"Young TE, {cap}"), (CUT, w), con_prod(CUT, w))


LOTTO_PAST = 200   # FP rank. Past it, three in four players returned under a quarter of their price over 3 years


def lotto_line(act, why, x, f, prod):
    """One name for keeping a player, by price: Lottery ticket past LOTTO_PAST, Hold inside it. QBs (backups kept as
    bye-week / injury cover) and players producing now stay Hold. The league sim treats both the same."""
    if x.position == "QB":
        return act, why
    r = getattr(x, f"r_{f}")
    deep = r is None or (isinstance(r, float) and math.isnan(r)) or r > LOTTO_PAST
    if deep and act in (HOLD, HOLDF) and not prod:
        return LOTTO, why
    if not deep and act == LOTTO:
        return HOLD, why
    return act, why


def classify(x, f, ctx, split=True):
    """-> (group, (reb action, why), (con action, why), (default action, why))

    Contender: a player the rebuilder sells and the contender would hold is used through the playoffs and then sold
    if he's starting now, else sold now. Default (the middle mode): the same; Hold where rebuilders sell and
    contenders buy; the contender's Buy where rebuilders hold or buy too; otherwise the contender call if he's
    starting now, else the rebuilder call. In the middle_mode.py league sim this beat the old contender calls over
    seasons 1-3 at every projected finish and gave up little in season 1."""
    g, (ra, rw), (ca, cw) = base_calls(x, f, ctx)
    prod, rk = producing(x, f, ctx)
    pos, line = x.position, START[f][x.position]
    kr, kc = kind(ra), kind(ca)
    if kr == "sell" and kc == "hold":
        if prod:
            ca, cw = USE, f"Producing now ({pos}{f0(rk)} by PPG). Start him through the playoffs, then: {ra.lower()}. " + rw
        else:
            now = "no PPG rank yet (under 4 games)" if pd.isna(rk) else f"{pos}{f0(rk)} by PPG"
            ca, cw = ra, f"Not a starter now ({now}; starters are {pos}1-{line}), so sell before his price falls. " + rw
    elif kr == "sell" and kc == "use" and not prod:
        ca, cw = ra, rw
    (ra, rw), (ca, cw) = lotto_line(ra, rw, x, f, prod), lotto_line(ca, cw, x, f, prod)
    kr, kc = kind(ra), kind(ca)
    if ra == ca:
        mid = (ra, cw if prod else rw)
    elif kr == "sell" and kc == "buy":
        mid = (HOLD, f"Rebuilders should {ra.lower()} and contenders {ca.lower()}: in between, keep him. " + cw)
    elif kc == "buy" and kr in ("hold", "buy"):
        mid = (ca, cw)
    else:
        mid = (ca, cw) if prod else (ra, rw)

    if not split:                                   # the analyses score the calls before this split
        return g, (ra, rw), (ca, cw), mid
    ac, pb = age_class(x, ctx), price_bucket(getattr(x, f"r_{f}"))

    def push(mode, act, why):                       # how hard: Sell heavy / light, Buy strong / weak / hold
        if act == SELL:
            a, pre = strength(f, g, pos, ac, pb)
        elif kind(act) == "buy":
            a, pre = buy_strength(f, mode, g, pos, ac, pb)
        else:
            return act, why
        return a, pre + why
    r, c, m = push("reb", ra, rw), push("con", ca, cw), push("mid", *mid)
    if m[0] == WEAK and r[0] == STRONG and c[0] == STRONG:   # both sides pay up: so does Default
        m = (STRONG, "Rebuilders and contenders both have this as a strong buy: pay the market price, or a little more. "
             + mid[1])
    for side in ("buy", "sell"):                    # Default never buys or sells where neither side does
        if kind(m[0]) == side and side not in (kind(r[0]), kind(c[0])):
            m = (HOLD, f"Neither rebuilders nor contenders should {side} him here, so keep him. " + mid[1])
    return g, r, c, m


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


# In-season trade signals, from the league sim (career_value/analysis/inseason_sim.py, off-list players dropped; each
# compared with the same trades made on random players). The sim traded after weeks 4, 8 and 12 at equal price:
#   ROS rank a quarter worse than preseason: sell. Gained in both formats, every mode.
#   Dynasty rank a quarter worse than preseason: sell, 1QB only. Superflex was marginal.
#   Hurt with that dynasty fall: a note in superflex (a small gain); in 1QB he is a faller anyway.
#   25 or younger, ROS positional rank a quarter better than dynasty: buy. Best gain per trade in both formats.
#   Rookies with that gap: buy for rebuilders only. Value gained, this season's title odds didn't.
# Players off the ROS list get no signal: mostly out for the year, and the Stash / Sell when healthy calls cover them.
# Some players don't get the sells (market_price_sim.py, with trades at KTC and FantasyCalc prices, covers all of these):
#   1QB QBs: selling them lost value 3 years on in every mode (-5 to -56), for titles within noise. Contenders still sell
#   hurt ones, for this season: rest of season 0.33, 0.47 of the price (inseason_movers.csv).
#   Hurt RB / WR / TE 25 or younger, for 1QB default teams and rebuilders in both formats. 1QB default teams gained nothing
#   clear, and rebuilders lost value the next May (-9 to -18). Contenders still sell them: rest of season 0.36, 0.34.
#   Superflex has no QB exemption, and default teams sell hurt young players. Titles over 3 years: QBs +0.3 to +2.6,
#   hurt young players +1.3 to +1.4, with value 3 years on +3 to +20 and +45 to +50.
#   Healthy players, for 1QB contenders and default teams (signal_sim.py): not selling them added 0.8 to 5.6 title points over
#   3 years and 46 to 67 of value 3 years on, both sources. 1QB rebuilders keep these sells: without them titles fell 0.7 to 2.0.
# Superflex rebuilders don't get the veteran buys (signal_sim.py): without them titles over 3 years rose 1.1 to 3.5 (most of it
#   the 2022 class) and value 3 years on rose 67 to 81, both sources. They keep the rookie buys. 1QB rebuilders keep both.
# A quarter is one place near the top of the list (#3 to #4), and small moves there had no edge: top-36 players down 5
# ROS places or fewer kept their value like players with no signal (1 year on, 1.07 vs 1.04 in 1QB, 1.09 vs 1.02 in
# superflex), as did 1QB fallers down 2 dynasty places or fewer. The sim didn't have these floors.
SIG_MOVE = 4 / 3
SIG_FLOOR_ROS, SIG_FLOOR_DYN = 6, 3             # places lost, on top of the quarter
SIG_WEEKS = (5, 14)             # Sleeper's current week: at least 4 played, and no later than the sim's last window
SIG_POOL = 340                  # the sim's tradeable pool
SIG_HURT = ("IR", "PUP", "Out")


def signals(x, f, ctx):
    """-> what fired for this player and format: ros_down / faller (preseason rank, rank now), hurt, gap (dynasty
    positional rank, ROS positional rank), rookie"""
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
    return s


TAKE_BACK = {"reb": "Trade him for a younger player at his price.",
             "con": "Trade him for a player at his price who ranks better for the rest of the season.",
             "mid": "Trade him for a player at his price who is younger or ranks better for the rest of the season."}


def apply_signal(act, mode, x, f, s):
    """(action, note) for one mode. The sim's sells and buys override Hold-type calls; where one meets the opposite
    backtested call, or a sell and a buy fire together, the player is held. Some players keep their usual call on a sell
    signal, except for contenders with hurt ones: 1QB QBs, and hurt players 25 or younger, apart from superflex default
    teams, who sell them. 1QB contenders and default teams sell only hurt players. Rookie buys are for rebuilders, and
    superflex rebuilders get no others."""
    sells = []
    if x.position == "QB":
        no_sell = f == "1qb" and not (mode == "con" and s.get("hurt"))
    else:
        no_sell = bool(s.get("hurt") and s.get("young") and mode != "con" and not (f == "sf" and mode == "mid"))
    no_sell = no_sell or (f == "1qb" and mode != "reb" and not s.get("hurt"))
    if "ros_down" in s and not no_sell:
        sells.append(f"FantasyPros' rest-of-season rank fell from #{f0(s['ros_down'][0])} in the preseason to "
                     f"#{f0(s['ros_down'][1])}, and dynasty prices lag that.")
    if "faller" in s and f == "1qb" and not no_sell:
        sells.append(f"His dynasty rank fell from #{f0(s['faller'][0])} in the preseason to #{f0(s['faller'][1])}, and "
                     "selling 1QB fallers gained in the league sim.")
    buys = mode == "reb" if s.get("rookie") else not (f == "sf" and mode == "reb")
    gap = s.get("gap") if buys else None
    gtxt = (f"FantasyPros ranks him {x.position}{f0(gap[1])} for the rest of the season but {x.position}{f0(gap[0])} in "
            "dynasty, and buying young players like this gained in the league sim") if gap else ""
    k = kind(act)
    if sells and gap:
        return (act if k == "cut" else HOLD), " ".join(sells) + f" Still, {gtxt}: mixed signals, so hold."
    if sells:
        if k == "buy":
            return HOLD, " ".join(sells) + f" The usual call is {act.lower()}, so hold and don't add more."
        return (act if k in ("sell", "cut") else LIGHT), " ".join(sells) + " " + TAKE_BACK[mode]
    if gap:
        if k == "buy":
            return act, gtxt + "."
        if k in ("sell", "cut", "use"):
            return HOLD, gtxt + f". The usual call is {act.lower()}, so hold rather than sell."
        return WEAK, gtxt + ": buy at the market price, not above."
    if "faller" in s and s.get("hurt") and f == "sf" and not no_sell:
        return act, (f"Hurt, and his dynasty rank fell from #{f0(s['faller'][0])} in the preseason to "
                     f"#{f0(s['faller'][1])}: selling players like this gained a little in the league sim.")
    return act, ""


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
        if kind(ra) in ("hold", "buy") and gp == "less":
            ra, rn = BUYLOW, (f"{what}. The market has marked him down ({where}). Contenders who need help now will sell: "
                              "buy if the injury isn't career-threatening.")
        elif kind(ra) == "sell":
            ra, rn = sell_later()
        else:
            rn = f"{what} ({where})."

        def now(act):
            if kind(act) in ("hold", "buy"):
                return STASH, (f"{what}: no help this season. Keep him for next year if you have the IR slot; "
                               f"trading him now sells at an injury discount ({where}).")
            if kind(act) in ("sell", "use"):
                return sell_later()
            return act, f"{what} ({where})."
        (ca, cn), (ma, mn) = now(ca), now(ma)
        return (g, ra, rw, ca, cw, rn, cn, ma, mw, mn)
    # contender / default: rest-of-season role. FP's ROS positional rank decides whether he starts for you this year.
    rp = getattr(x, "ros_p", None)
    rp = None if rp is None or (isinstance(rp, float) and math.isnan(rp)) else rp
    line = START[f][x.position]

    def ros(act):
        if not out and rp is not None and rp <= line and (kind(act) == "sell" or act in (CUT, LOTTO)):
            return USE, f"FantasyPros ranks him {x.position}{f0(rp)} for the rest of the season: start him through your playoffs, then: {act.lower()}."
        if not out and act == USE and (rp is None and getattr(x, "gone", False) is True or rp is not None and rp > 2 * line):
            return strength(f, g, x.position, age_class(x), price_bucket(r))[0], (("FantasyPros no longer ranks him for the rest of the season" if rp is None else
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
    return (g, ra, rw, ca, cw, rn, cn, ma, mw, mn)
