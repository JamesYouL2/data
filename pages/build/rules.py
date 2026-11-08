"""Buy / hold / sell rules for dynasty players, from the career_value backtests (DynastyProcess data repo,
career_value/README.md sections 6 and 12, plus the deep-roster, pick-price and 30+ analyses).

classify(x, f, ctx) takes one player row and a format ("1qb" or "sf") and returns
(group, (rebuilder action, why), (contender action, why), (default action, why)). The default ("middle") call and the
contender rule that a fading player the rebuilder sells is sold unless he's starting now come from
career_value/analysis/middle_mode.py. ctx carries ROOKIE (latest rookie class) and LOOK
(the 30+ history lookup from old_lookup.json).
"""
import math

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
BUY, BUY3, BUY4, HOLD, HOLDF, LOTTO, SELLH, SELL, SELL3, SELL4, USE, CUT = (
    "Buy", "Buy with a 3rd", "Buy with a 4th", "Hold", "Hold to February", "Lottery ticket", "Sell high now", "Sell",
    "Sell for a 3rd", "Sell for a 4th", "Use, then sell", "Cut")
STASH, BUYLOW, SELLHEALTHY = "Stash", "Buy low (injured)", "Sell when healthy"


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
    if r is None or math.isnan(r):
        return "U"
    return "A" if r <= 50 else "B" if r <= 100 else "C" if r <= 150 else "D3" if r <= 215 else "D34" if r <= 239 \
        else "D4" if r <= 300 else "U"


def f0(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0f}"


def base_calls(x, f, ctx):
    """The backtested calls: (group, (reb action, why), (con action, why)), before classify's contender rule."""
    ROOKIE, look = ctx["ROOKIE"], ctx["LOOK"]
    pos, a, r, pr, rd = x.position, x.age, getattr(x, f"r_{f}"), getattr(x, f"pr_{f}"), x.rd
    sf = f == "sf"
    bd = band(r)
    rookie = x.dy == ROOKIE
    young = (not rookie) and a < 26
    early = rd <= 3
    prod, prk = producing(x, f, ctx)
    prodr = x.prod_rank
    cap = "NFL R1-3" if early else "NFL R4+/UDFA"

    def con_prod(act, why):                       # contenders keep whoever is starting for them now
        if prod and act in (SELL, SELL3, SELL4, CUT):
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
            if bd in ("A", "B", "C"):
                w = "Rookie RB prices peak in November and Day-2+ RBs trail their price: sell into the peak."
                return (f"Rookie RB, NFL R{rd:.0f}", (SELLH, w), con_prod(SELLH, w) if not prod else
                        (HOLD, f"Producing now ({pos}{f0(prk)}). " + w))
            w = "Day-3 rookie RBs at 4th-round prices: Chase Brown and Kyren Williams came from here; most return nothing."
            return (f"Rookie RB, {cap}", (LOTTO, w), (LOTTO, w))
        if rd >= 5 and bd in ("D4", "U"):
            w = "Late rookies ranked 250+ returned ~0 VORP, but 11-22% got a camp-hype price bump: hold to summer, sell into hype."
            return (f"Rookie {pos}, NFL R5+/UDFA", (LOTTO, w), (CUT if not prod else HOLD, "Cut if you need the roster spot. " + w))
        w = ("Rookie WRs rise ~9% from now to February." if pos == "WR" else
             "Rookie TEs rise ~26% from now to February; TEs taken in rookie rounds 3-4 returned almost nothing after that." if pos == "TE"
             else "")
        return (f"Rookie {pos}", (HOLDF, w), (HOLDF, w + " Sell after February if you need vets."))
    # ---------------- 30+ ----------------
    if a >= 30:
        t = tier(pos, prodr)
        v3, v5, y5, n = look.get((f, pos, ageband(a), t), (0.0, 0.0, 0.0, 0))
        hist = f"30+ {pos}s like this ({ageband(a)}, {t} tier) averaged {f0(v3)} VORP over 3 yrs, {f0(v5)} over 5; {f0(100 * y5)}% still useful in year 5 (n={n})."
        if pos == "QB":
            if a >= 36:
                w = f"QBs 34+ keep producing, but at {a:.0f} retirement ends it: a 1-2 season asset. " + hist
                return ("QB 36+, win-now", (SELL if bd != "U" else CUT, w),
                        (HOLD if t != "bench" else CUT, w))
            if t in ("elite", "top"):
                cheap = pd.notna(pr) and pd.notna(prodr) and pr >= prodr + 6
                w = ("QBs age well: " + hist +
                     (f" Priced QB{pr:.0f} while producing like QB{prodr:.0f}." if cheap else ""))
                return ("QB 30+, long-term", (BUY if cheap else HOLD, w), (BUY, w))
            if t == "starter":
                if sf:
                    return ("QB 30+, mid-tier", (HOLD if bd in ("C", "D3", "D34", "D4", "U") else SELL, hist),
                            (BUY if bd in ("D3", "D34", "D4", "U") else HOLD, hist))
                return ("QB 30+, mid-tier", (SELL, hist), con_prod(SELL, hist))
            return ("QB 30+, backup", (SELL if bd in ("A", "B", "C", "D3") else CUT, hist),
                    (HOLD if sf and bd != "U" else CUT, ("Bye-week / injury cover. " if sf else "") + hist))
        if v3 >= 40:
            w = "Value is front-loaded, a 2-year window. " + hist
            return (f"{pos} 30+, 2-year window", (SELL, w), (BUY if bd in ("D3", "D34", "D4", "U") else HOLD, w))
        w = hist + " Take any 3rd or 4th offered."
        return (f"{pos} 30+, fading", (SELL if bd in ("A", "B", "C", "D3") else CUT, w),
                con_prod(SELL if bd in ("A", "B", "C", "D3") else CUT, w))
    # ---------------- 26-29 ----------------
    if not young:
        if pos == "QB":
            if sf:
                if bd == "A":
                    w = "Superflex QBs priced top 50 hold their value (median rank a year later: #8)."
                    return ("QB 26-29, core", (HOLD, w), (HOLD, w))
                if pd.notna(pr) and 13 <= pr <= 28 and pd.notna(prodr) and pr >= prodr + 6 and a >= 28:
                    w = f"28+ QBs priced QB13-28 beat their price by +75 to +123%; priced QB{pr:.0f}, producing like QB{prodr:.0f}."
                    return ("QB 28-29, underpriced", (BUY, w), (BUY, w))
                if pd.notna(pr) and 13 <= pr <= 20 and a < 28:
                    w = "QBs 25-27 priced QB13-20 returned -44% / -92% / -56% of their price."
                    return ("QB 26-27, priced QB13-20", (SELL, w), con_prod(HOLD, w))
                if bd in ("B", "C"):
                    w = "QBs 26-29 priced 51-150 kept producing (170 VORP over 3 yrs at 51-100) but fall to ~#125-150 next May."
                    return ("QB 26-29", (HOLD, w), (HOLD, w))
                if bd in ("D3", "D34"):
                    w = "Superflex QBs 26-29 at a 3rd-round price: 78 VORP over 3 yrs, 60% had a starter season (Mayfield, Darnold)."
                    return ("QB 26-29, 3rd-round price", (BUY3, w), (BUY3, w))
                if bd == "D4":
                    w = "Superflex QBs 26-29 at a 4th-round price: backups, 43% had a starter season, ~7 VORP."
                    return ("QB 26-29, backup", (CUT, w), (HOLD, w))
                return ("QB 26-29, unranked", (CUT, "Waiver level."), con_prod(CUT, "Waiver level."))
            if bd in ("A", "B"):
                w = "1QB QBs 26-29 priced top 100 hold value (rank ~33-94 two years on)."
                return ("QB 26-29", (HOLD, w), (HOLD, w))
            if bd in ("C", "D3", "D34"):
                w = "1QB QBs 26-29 priced 101-215 returned 11-32 VORP over 3 yrs and fall to ~#210-220 next May."
                return ("QB 26-29", (SELL, w), con_prod(SELL, w))
            if bd == "D4":
                w = "1QB QBs 26-29 at a 4th-round price: 44 VORP over 3 yrs, 33% had a top-6 season (Mayfield, Goff)."
                return ("QB 26-29, 4th-round price", (BUY4, w), (BUY4, w))
            return ("QB 26-29, unranked", (CUT, "Waiver level."), con_prod(CUT, "Waiver level."))
        if pos == "RB":
            if bd == "A":
                w = "RBs 26-29 priced top 50 produce (~220-235 VORP over 3 yrs) but their rank falls from ~#50 to ~#71-83 within 2 years."
                return ("RB 26-29, core", (SELL, w), (HOLD, w))
            if bd == "B":
                w = "RBs priced 51-100 fall to ~#115 next May and ~#190-205 in 2 years."
                return ("RB 26-29", (SELL, w), (HOLD, w))
            if bd == "C":
                w = (f"RBs 26-29 priced 101-150: {'24' if not sf else '51'} VORP over 3 yrs, median rank next May ~#{'226' if not sf else '208'}.")
                return ("RB 26-29", (SELL3, w), (HOLD, w) if sf else con_prod(SELL3, w))
            if bd in ("D3", "D34"):
                w = "RBs 26-29 at a 3rd-round price: 8-11 VORP over 3 yrs, none became top-100. A 4th is worth more."
                return ("RB 26-29, filler", (SELL3, w), con_prod(SELL4, w))
            w = "RBs 26-29 ranked 240+: 2-6 VORP over 3 yrs; 75%+ drop out of the top 300 within a year."
            return ("RB 26-29, filler", (CUT if bd == "U" else SELL4, w), con_prod(CUT, w))
        if pos == "WR":
            if bd == "A":
                w = "WRs 26-29 priced top 50: ~214 VORP over 3 yrs; rank falls from ~#40 to ~#75-90 within 2 years."
                return ("WR 26-29, core", (HOLD if a < 28 else SELL, w), (HOLD, w))
            if bd == "B":
                w = "WRs 26-29 priced 51-100: ~72-100 VORP over 3 yrs; rank falls to ~#105 next May, ~#164 in 2 years."
                return ("WR 26-29", (SELL, w), (HOLD, w))
            if bd == "C":
                w = "WRs 26-29 priced 101-150 returned 35-51 VORP over 3 yrs: worth more than a 3rd."
                return ("WR 26-29", (HOLD, w), (HOLD, w))
            if bd in ("D3", "D34"):
                w = "WRs 26-29 at a 3rd-round price: 1-2 VORP over 3 yrs, none became top-100."
                return ("WR 26-29, filler", (SELL3, w), con_prod(SELL4, w))
            w = "WRs 26-29 ranked 240+: 0-3 VORP over 3 yrs."
            return ("WR 26-29, filler", (CUT if bd == "U" else SELL4, w), con_prod(CUT, w))
        # TE
        if bd == "A":
            w = "TEs 26-29 priced top 50: ~104 VORP over 3 yrs, 86% had a top-6 season."
            return ("TE 26-29, core", (HOLD, w), (HOLD, w))
        if bd == "B":
            w = "TEs 26-29 priced 51-100 fall to ~#100 next May and ~#130 in 2 years."
            return ("TE 26-29", (SELL, w), (HOLD, w))
        if bd in ("C", "D3"):
            w = "TEs 26-29 priced 101-215: 13-35 VORP over 3 yrs, 16-42% had a top-6 season."
            return ("TE 26-29", (HOLD if bd == "C" else SELL, w), (HOLD, w))
        w = "TEs 26-29 ranked 216+: 0-6 VORP over 3 yrs."
        return ("TE 26-29, filler", (CUT, w), con_prod(CUT, w))
    # ---------------- young (25 or younger, not a rookie) ----------------
    if pos == "QB":
        if sf and pd.notna(pr) and pr <= 6 and a < 25:
            w = "Superflex QBs 24 or younger priced QB1-6 returned -63% / -48% / -20% of their price; 28+ QBs priced QB13-28 beat theirs."
            return ("Young QB priced as a QB1", (SELL, w), (SELL, w + " Swap for an older, cheaper QB."))
        if bd in ("A", "B"):
            w = "Young QBs priced top 100 hold their value."
            return ("Young QB, core", (HOLD, w), (HOLD, w))
        if sf and early and bd in ("C", "D3", "D34", "D4"):
            if bd == "C":
                return (f"Young QB, {cap}", (HOLD, "27 VORP over 3 yrs at ranks 101-150."), (HOLD, "27 VORP over 3 yrs at ranks 101-150."))
            w = "Young superflex QBs with NFL R1-3 capital ranked 151-250: 68 VORP over 3 yrs, half later ranked top 100 (Jordan Love, Darnold)."
            return (f"Young QB, {cap}", (BUY, w), (BUY, w))
        if early and bd in ("C", "D3", "D34"):
            w = "Young 1QB QBs with NFL R1-3 capital: 23-55 VORP over 3 yrs, 14-15% later ranked top 100."
            return (f"Young QB, {cap}", (HOLD, w), (HOLD, w))
        w = "Young QBs ranked 250+ (or without early capital) returned ~0."
        return (f"Young QB, {cap}", (CUT, w), con_prod(CUT, w))
    if pos == "RB":
        if bd == "A":
            w = "Young RBs priced top 50 hold value: rank ~#30-42 next May, ~#51-64 in 2 years."
            return ("Young RB, core", (HOLD, w), (HOLD, w))
        if bd == "B":
            w = "RBs priced 51-100 fall to ~#111-122 next May and ~#194-200 in 2 years."
            return ("Young RB", (SELL, w), (HOLD, w))
        if bd == "C":
            if early:
                w = ("Young NFL R1-3 RBs priced 101-150: 50 VORP over 3 yrs, 55% had a starter season." if sf else
                     "Young NFL R1-3 RBs priced 101-150: 12 VORP over 3 yrs, no top-12 seasons, 7% later top-100.")
                return (f"Young RB, {cap}", (HOLD, w) if sf else (SELL3, w), (HOLD, w) if sf else con_prod(SELL3, w))
            w = "Young Day-3/UDFA RBs priced 101-150: 32-53 VORP over 3 yrs, 33% later top-100."
            return (f"Young RB, {cap}", (HOLD, w), (HOLD, w))
        if bd in ("D3", "D34"):
            if early:
                w = "Young NFL R1-3 RBs at a 3rd-round price: 4-7 VORP over 3 yrs. Early picks who fall this far rarely come back."
                return (f"Young RB, {cap}", (SELL4, w), con_prod(SELL4, w))
            w = "Young Day-3/UDFA RBs at a 3rd-round price: 25-31 VORP over 3 yrs, 23-24% later top-100 (beats a 3rd-round rookie)."
            return (f"Young RB, {cap}", (BUY3, w), (BUY3, w))
        if early:
            w = "Young NFL R1-3 RBs ranked 240+: ~0 VORP."
            return (f"Young RB, {cap}", (CUT, w), con_prod(CUT, w))
        w = ("Young Day-3/UDFA RBs ranked 240-300: 13-18 VORP over 3 yrs, almost all from one hit (Kyren Williams, #256)."
             if bd == "D4" else "Young Day-3/UDFA RB: an opportunity bet. Hold if you have the roster spot.")
        return (f"Young RB, {cap}", (LOTTO, w), (LOTTO, w))
    if pos == "WR":
        if bd == "A":
            w = "The safest asset in dynasty: young WRs priced top 50 sat at median #25 and were still #28 two years later."
            return ("Young WR, core", (HOLD, w), (HOLD, w))
        if bd in ("B", "C"):
            w = "Young WRs priced 51-150: 30-103 VORP over 3 yrs, 20-35% later ranked top 100."
            return ("Young WR", (HOLD, w), (HOLD, w))
        if bd in ("D3", "D34"):
            if early:
                if sf:
                    w = "Young NFL R1-3 WRs at a 3rd-round price: 38 VORP over 3 yrs, 21% later top-100 (beats a 3rd-round rookie)."
                    return (f"Young WR, {cap}", (BUY3, w), (BUY3, w))
                w = "Young NFL R1-3 WRs at a 3rd-round price: 20 VORP over 3 yrs, 13% later top-100 (about a 3rd-round rookie)."
                return (f"Young WR, {cap}", (HOLD, w), (HOLD, w))
            w = "Young Day-3/UDFA WRs at a 3rd-round price: 34-37 VORP over 3 yrs, 60-70% had a starter season, but only 7-10% later top-100."
            return (f"Young WR, {cap}", (HOLD, w), (BUY3, w))
        if early:
            w = "Young NFL R1-3 WRs fallen to 240+: 0-4 VORP over 3 yrs; 10-15% still top-300 a year later."
            return (f"Young WR, {cap}", (SELL4 if bd == "D4" else CUT, w), con_prod(SELL4 if bd == "D4" else CUT, w))
        w = ("Young Day-3/UDFA WRs ranked 240-300: 8-9 VORP over 3 yrs, the best price retention in the band (Renfrow, Meyers)."
             if bd == "D4" else "Young Day-3/UDFA WR: an opportunity bet. Hold if you have the roster spot.")
        return (f"Young WR, {cap}", (LOTTO, w), (LOTTO, w))
    # TE
    if bd in ("A", "B", "C"):
        w = "Young TEs priced top 150: 40-99 VORP over 3 yrs."
        return ("Young TE", (HOLD, w), (HOLD, w))
    if bd in ("D3", "D34"):
        w = "Young TEs at a 3rd-round price: 0-3 VORP over 3 yrs. A 4th is worth more."
        return ("Young TE", (SELL4, w), con_prod(SELL4, w))
    if not early and bd == "D4":
        w = "Young Day-3/UDFA TEs at a 4th-round price: 17-21 VORP over 3 yrs, 25% had a top-6 season (Ferguson, Schultz). Small sample."
        return (f"Young TE, {cap}", (BUY4, w), (BUY4, w))
    w = "Young TEs ranked 240+ (early-capital ones especially) rarely come back."
    return (f"Young TE, {cap}", (CUT, w), con_prod(CUT, w))


def classify(x, f, ctx):
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
    kc = kind(ca)
    if ra == ca:
        mid = (ra, cw if prod else rw)
    elif kr == "sell" and kc == "buy":
        mid = (HOLD, f"Rebuilders should {ra.lower()} and contenders {ca.lower()}: in between, keep him. " + cw)
    elif kc == "buy" and kr in ("hold", "buy"):
        mid = (ca, cw)
    else:
        mid = (ca, cw) if prod else (ra, rw)
    return g, (ra, rw), (ca, cw), mid


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


def situational(x, f, call):
    """Adjust the backtested calls for the trade market (FantasyCalc), injuries (Sleeper) and rest-of-season value.

    Returns (group, reb action, reb why, con action, con why, reb note, con note, default action, default why,
    default note). The notes say what moved and why; none of these adjustments come from the backtests. The default
    call is adjusted like the contender call: both care about this season."""
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
        if ra in (HOLD, HOLDF, BUY, BUY3, BUY4) and gp == "less":
            ra, rn = BUYLOW, (f"{what}. The market has marked him down ({where}). Contenders who need help now will sell: "
                              "buy if the injury isn't career-threatening.")
        elif kind(ra) == "sell":
            ra, rn = sell_later()
        else:
            rn = f"{what} ({where})."

        def now(act):
            if act in (HOLD, HOLDF, BUY, BUY3, BUY4):
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
        if not out and rp is not None and rp <= line and act in (SELL, SELL3, SELL4, SELLH, CUT, LOTTO):
            return USE, f"FantasyPros ranks him {x.position}{f0(rp)} for the rest of the season: start him through your playoffs, then: {act.lower()}."
        if not out and act == USE and (rp is None and getattr(x, "gone", False) is True or rp is not None and rp > 2 * line):
            return SELL, (("FantasyPros no longer ranks him for the rest of the season" if rp is None else
                           f"FantasyPros ranks him only {x.position}{f0(rp)} for the rest of the season") + ": not a starter, so sell now.")
        return act, ""
    (ca, cros), (ma, mros) = ros(ca), ros(ma)

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
    rn, cn, mn = mkt_note(ra), (cros + " " + mkt_note(ca)).strip(), (mros + " " + mkt_note(ma)).strip()
    if out:
        rr = getattr(x, "ros_r", None)
        tag = (f"Currently {inj}" + (f" ({x.inj_part})" if isinstance(x.inj_part, str) else "")
               + (f"; FP rest-of-season #{f0(rr)}" if rr is not None and not math.isnan(rr) else "")
               + (", so little help this season" if rr is not None and not math.isnan(rr) and rr > 150 else "") + ".")
        rn, cn, mn = ((tag + " " + n).strip() for n in (rn, cn, mn))
    return (g, ra, rw, ca, cw, rn, cn, ma, mw, mn)
