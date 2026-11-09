"use strict";
const D = window.DATA;
const API = "https://api.sleeper.app/v1";
const $ = (id) => document.getElementById(id);
const START = { "1qb": { QB: 12, RB: 24, WR: 30, TE: 12 }, sf: { QB: 24, RB: 24, WR: 30, TE: 12 } };

const state = { league: null, users: {}, rosters: [], roster: null, mode: null, fmt: null, autoFmt: null, suggested: {} };

// ---------- helpers ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const kind = (act) => !act ? "none" : act.startsWith("Buy") ? "buy" : act.startsWith("Hold") || act === "Stash" ? "hold" : act.startsWith("Use") ? "use"
  : act.startsWith("Sell") ? "sell" : act.startsWith("Lottery") ? "lotto" : act === "Cut" ? "cut" : "none";
const ORDER = ["buy", "hold", "use", "sell", "lotto", "cut", "none"];
const SECTION = { buy: "Buy more of", hold: "Hold", use: "Use, then sell", sell: "Sell", lotto: "Lottery tickets", cut: "Cut", none: "No call" };
const badge = (act) => `<span class="b ${kind(act)}">${esc(act || "No call")}</span>`;

async function get(path) {
  const r = await fetch(API + path);
  if (r.status === 404) throw new Error("not found");
  if (!r.ok) throw new Error(`Sleeper returned ${r.status} for ${path}`);
  const j = await r.json();
  if (j === null) throw new Error("not found");
  return j;
}

function status(msg, err = false) { const s = $("status"); s.textContent = msg; s.classList.toggle("err", err); }

function call(sid, fmt, mode) {
  const p = D.players[sid];
  if (!p) return null;
  const r = D.rules[fmt === "sf" ? p.c2 : p.c1];
  const [grp, ra, rw, ca, cw, rn, cn, ma, mw, mn] = r;
  const all = { m: [ma, mw, mn], c: [ca, cw, cn], r: [ra, rw, rn] };
  const [act, why, note] = all[mode] || all.m;
  return { grp, act, why, note, others: Object.keys(MODES).filter((k) => k !== mode).map((k) => [MODES[k], all[k][0]]) };
}
const MODES = { m: "Default", c: "Contending", r: "Rebuilding" };

// ---------- trade market (FantasyCalc), injuries, rest of season ----------
const OUT = new Set(["IR", "PUP", "Out", "Sus", "DNR", "NA"]);
const mkt = (p, fmt) => (fmt === "sf" ? p.m2 : p.m1) || null;          // [market rank, value, ROS value, ROS rank, 30-day trend]
const healthy = (p) => !OUT.has(p.inj);
function pickLabel(v, fmt) {
  const pk = D.meta.picks?.[fmt];
  if (!pk || !pk.length || v == null) return "";
  if (v > pk[0][1] * 1.25) return "more than an early 1st";
  if (v < pk[pk.length - 1][1] * 0.6) return "less than a late 4th";
  let best = pk[0];
  for (const x of pk) if (Math.abs(x[1] - v) < Math.abs(best[1] - v)) best = x;
  return `≈ ${D.meta.picks[fmt + "_year"]} ${best[0]}`;
}
function trend(m) {
  if (!m || !m[4] || m[1] - m[4] <= 0) return "";
  const pct = m[4] / (m[1] - m[4]);
  if (Math.abs(pct) < 0.1) return "";
  return ` <span class="${pct > 0 ? "up" : "dn"}" title="market value ${pct > 0 ? "up" : "down"} ${Math.round(Math.abs(pct) * 100)}% in 30 days">${pct > 0 ? "▲" : "▼"}</span>`;
}

function price(p, fmt) { return fmt === "sf" ? p.r2 : p.r1; }
function posPrice(p, fmt) { return fmt === "sf" ? p.pr2 : p.pr1; }
const rankKey = (p, fmt) => price(p, fmt) ?? 9000 + (p.prk ?? 500);

function teamName(r) {
  const u = state.users[r.owner_id];
  return (u && (u.metadata?.team_name || u.display_name)) || `Team ${r.roster_id} (no owner)`;
}
function ownerName(r) { const u = state.users[r.owner_id]; return u ? u.display_name : "orphan"; }
function record(r) {
  const s = r.settings || {};
  const w = s.wins ?? 0, l = s.losses ?? 0, t = s.ties ?? 0;
  return w + l + t ? `${w}-${l}${t ? "-" + t : ""}` : "";
}
const pf = (r) => (r.settings?.fpts ?? 0) + (r.settings?.fpts_decimal ?? 0) / 100;
const rosterPlayers = (r) => [...new Set([...(r.players || []), ...(r.reserve || []), ...(r.taxi || [])])];

// ---------- league loading ----------
async function loadInput(raw) {
  const v = raw.trim();
  if (!v) return;
  $("leagues").hidden = true;
  if (/^\d{6,}$/.test(v)) return loadLeague(v);
  status("Looking up user…");
  try {
    const u = await get(`/user/${encodeURIComponent(v)}`);
    const st = await get("/state/nfl");
    let ls = await get(`/user/${u.user_id}/leagues/nfl/${st.league_season || st.season}`);
    if (!ls.length && st.previous_season) ls = await get(`/user/${u.user_id}/leagues/nfl/${st.previous_season}`);
    if (!ls.length) return status(`${u.display_name} has no NFL leagues this season.`, true);
    const sel = $("leaguePick");
    sel.innerHTML = `<option value="">Choose a league…</option>` + ls.map((l) => `<option value="${esc(l.league_id)}">${esc(l.name)} (${l.total_rosters} teams)</option>`).join("");
    $("leagues").hidden = false;
    status(`${ls.length} league${ls.length > 1 ? "s" : ""} for ${u.display_name}.`);
    if (ls.length === 1) { sel.value = ls[0].league_id; loadLeague(ls[0].league_id); }
  } catch (e) {
    status(e.message === "not found" ? `No Sleeper user called "${v}".` : e.message, true);
  }
}

async function loadLeague(id) {
  status("Loading league…");
  try {
    const [league, users, rosters] = await Promise.all([get(`/league/${id}`), get(`/league/${id}/users`), get(`/league/${id}/rosters`)]);
    state.league = league;
    state.users = Object.fromEntries(users.map((u) => [u.user_id, u]));
    state.rosters = rosters;
    const rp = league.roster_positions || [];
    state.autoFmt = rp.includes("SUPER_FLEX") || rp.filter((x) => x === "QB").length >= 2 ? "sf" : "1qb";
    localStorage.setItem("league", id);
    $("league").value = id;
    suggestModes();
    const saved = JSON.parse(localStorage.getItem(`prefs:${id}`) || "{}");
    state.fmt = saved.fmt || state.autoFmt;
    fillTeams(saved.roster);
    $("leagueName").textContent = `${league.name} · ${league.season} · ${rosters.length} teams`;
    $("controls").hidden = false;
    status("");
    pickTeam(Number($("team").value), saved.modes2?.[$("team").value]);
  } catch (e) {
    status(e.message === "not found" ? `No Sleeper league with ID ${id}.` : `Couldn't load the league: ${e.message}`, true);
  }
}

// Where each team stands: by record and points for once games are played, otherwise by roster strength at the
// current FantasyPros prices. The top half are the likely buyers of this-season production. It no longer picks
// your mode: Default (the middle call) fits most teams.
function suggestModes() {
  const rs = state.rosters, played = rs.some((r) => pf(r) > 0);
  const fmt = state.autoFmt;
  const strength = (r) => rosterPlayers(r).reduce((s, sid) => { const p = D.players[sid]; const k = p && price(p, fmt); return s + (k ? Math.max(0, 200 - k) : 0); }, 0);
  const score = played ? (r) => (r.settings?.wins ?? 0) * 1e5 + pf(r) : strength;
  const sorted = [...rs].sort((a, b) => score(b) - score(a));
  state.suggested = {};
  sorted.forEach((r, i) => { state.suggested[r.roster_id] = { mode: i < Math.ceil(rs.length / 2) ? "c" : "r", place: i + 1, by: played ? "record and points for" : "roster value" }; });
}

function fillTeams(savedRoster) {
  const sel = $("team");
  const rs = [...state.rosters].sort((a, b) => teamName(a).localeCompare(teamName(b)));
  sel.innerHTML = rs.map((r) => {
    const rec = record(r);
    const tn = teamName(r), on = ownerName(r);
    return `<option value="${r.roster_id}">${esc(tn)}${on !== tn ? " · " + esc(on) : ""}${rec ? " · " + rec : ""}</option>`;
  }).join("");
  if (savedRoster && rs.some((r) => r.roster_id === savedRoster)) sel.value = savedRoster;
}

function savePrefs() {
  const id = state.league.league_id;
  const k = `prefs:${id}`, cur = JSON.parse(localStorage.getItem(k) || "{}");
  cur.roster = state.roster.roster_id;
  cur.fmt = state.fmt;
  delete cur.modes;                     // saved before the Default mode existed: mostly the old auto-suggestion
  cur.modes2 = { ...(cur.modes2 || {}), [state.roster.roster_id]: state.mode };
  localStorage.setItem(k, JSON.stringify(cur));
  history.replaceState(null, "", `#league=${id}&team=${state.roster.roster_id}`);
}

function pickTeam(rid, mode) {
  state.roster = state.rosters.find((r) => r.roster_id === rid) || state.rosters[0];
  state.mode = MODES[mode] ? mode : "m";
  render();
}

// ---------- rendering ----------
function setSeg(id, v) { for (const b of $(id).querySelectorAll("button")) b.classList.toggle("on", b.dataset.v === v); }

function playerCells(sid, fmt) {
  const p = D.players[sid];
  if (!p) {
    const n = D.names[sid];
    if (n) return { name: esc(n[0]), meta: `${esc(n[1])}${n[2] ? " · " + esc(n[2]) : ""}`, price: "–", prod: "–", tags: [] };
    if (/^[A-Z]{2,3}$/.test(sid)) return { name: `${esc(sid)} defense`, meta: "DEF", price: "–", prod: "–", tags: [] };
    return { name: `Player ${esc(sid)}`, meta: "not in the rankings", price: "–", prod: "–", tags: [] };
  }
  const meta = [p.p, p.t, p.a != null ? `${Math.floor(p.a)} next Sept` : null,
    p.dy === D.meta.rookie ? "rookie" : null, p.rd ? `NFL R${p.rd}` : p.dy ? "UDFA" : null].filter(Boolean).map(esc).join(" · ");
  const pr = price(p, fmt);
  const m = mkt(p, fmt);
  const tv = fmt === "sf" ? p.v2 : p.v1;
  const prc = (pr ? `FP #${pr} <span class="meta">${esc(p.p)}${posPrice(p, fmt)}</span>` : `<span class="meta">FP unranked</span>`) +
    (pr && tv != null && tv !== pr ? `<div class="meta">in-season #${tv}</div>` : "") +
    (m ? `<div class="meta">market #${m[0]}${trend(m)}<br>${esc(pickLabel(m[1], fmt))}</div>` : D.meta.market ? `<div class="meta">no market value</div>` : "");
  let prod = p.ppg != null ? `${p.ppg.toFixed(1)} <span class="meta">${p.prk ? esc(p.p) + p.prk : ""} · ${p.g}g</span>` : `<span class="meta">${p.prkl ? `${D.meta.last}: ${esc(p.p)}${p.prkl}` : "–"}</span>`;
  const now = [];
  if (p.ros) now.push(`ROS ${esc(p.p)}${p.ros[1]} (#${p.ros[0]})`);
  else if (D.meta.ros && (p.r1 || p.g)) now.push("not in ROS ranks");
  if (p.wk) now.push(`next: ${esc(p.p)}${p.wk[0]}${p.wk[1] ? " " + esc(p.wk[1]) : ""}${p.wk[2] != null ? `, ${p.wk[2]} pts` : ""}`);
  if (now.length) prod += `<div class="meta">${now.join("<br>")}</div>`;
  const tags = p.inj ? [{ t: p.inj + (p.ib ? " · " + p.ib : ""), cls: OUT.has(p.inj) ? "inj" : "q" }] : [];
  return { name: esc(p.n), meta, price: prc, prod, tags };
}

function table(rows, opts = {}) {
  if (!rows.length) return `<p class="hint">None.</p>`;
  const head = `<tr><th>Player</th>${opts.owner ? "<th>Team</th>" : ""}<th>Price</th><th>${D.meta.season} PPR/game · now</th><th>Call</th><th>Why</th></tr>`;
  const body = rows.map(({ sid, c, owner, ownerHint, tags }) => {
    const x = playerCells(sid, state.fmt);
    const other = (c?.others || []).filter(([, a]) => a !== c.act).map(([l, a]) => `<div class="other">${l}: ${esc(a)}</div>`).join("");
    const tg = [...x.tags, ...(tags || []).map((t) => ({ t }))].map((t) => `<span class="tag ${t.cls || ""}">${esc(t.t)}</span>`).join("");
    return `<tr><td><span class="nm">${x.name}</span>${tg}<div class="meta">${x.meta}</div></td>` +
      (opts.owner ? `<td class="meta">${esc(owner)}${ownerHint ? `<div>${esc(ownerHint)}</div>` : ""}</td>` : "") +
      `<td class="num">${x.price}</td><td class="num">${x.prod}</td><td>${badge(c?.act)}${other}</td>` +
      `<td class="why">${c ? `<span class="grp">${esc(c.grp)}</span>${esc(c.why)}${c.note ? `<div class="note">${esc(c.note)}</div>` : ""}` : ""}</td></tr>`;
  }).join("");
  return `<div class="scroll"><table class="tbl">${head}${body}</table></div>`;
}

function render() {
  const { fmt, mode, roster } = state;
  setSeg("mode", mode); setSeg("fmt", fmt);
  $("team").value = roster.roster_id;
  const sug = state.suggested[roster.roster_id];
  $("modeHint").textContent = (sug ? `${ordinal(sug.place)} of ${state.rosters.length} by ${sug.by}. ` : "") + {
    m: "Default fits most teams: keep your starters, sell fading players who aren't starting.",
    c: "Contending: like Default, but buys more for this season.",
    r: "Rebuilding: for teams playing for year 4 and later. Sells fading players even if they start for you.",
  }[mode];
  $("fmtHint").textContent = `League settings say ${state.autoFmt === "sf" ? "superflex" : "1QB"}.`;
  savePrefs();

  // roster, grouped by call
  const taxi = new Set(roster.taxi || []), ir = new Set(roster.reserve || []);
  const mine = rosterPlayers(roster).map((sid) => ({ sid, c: call(sid, fmt, mode), tags: [taxi.has(sid) && "taxi", ir.has(sid) && "IR"].filter(Boolean) }));
  const groups = Object.fromEntries(ORDER.map((k) => [k, []]));
  for (const r of mine) groups[kind(r.c?.act)].push(r);
  const byPrice = (a, b) => rankKey(D.players[a.sid] || {}, fmt) - rankKey(D.players[b.sid] || {}, fmt);
  const pushFirst = (r) => (r.c?.act === "Sell: heavy" || r.c?.act === "Buy: strong" ? 0 : 1);
  let html = "";
  for (const k of ORDER) {
    if (!groups[k].length) continue;
    groups[k].sort((a, b) => pushFirst(a) - pushFirst(b) || byPrice(a, b));
    html += `<h3>${SECTION[k]} (${groups[k].length})</h3>` + table(groups[k]);
  }
  $("roster").innerHTML = html || `<p class="hint">This roster is empty.</p>`;
  $("summary").innerHTML = ORDER.filter((k) => groups[k].length && k !== "none")
    .map((k) => `<span class="chip">${badge(SECTION[k] === "Buy more of" ? "Buy" : SECTION[k])} ${groups[k].length}</span>`).join("");

  // buy targets elsewhere in the league
  const owner = {};
  for (const r of state.rosters) for (const sid of rosterPlayers(r)) owner[sid] = r;
  const targets = [];
  for (const r of state.rosters) {
    if (r.roster_id === roster.roster_id) continue;
    for (const sid of rosterPlayers(r)) {
      const c = call(sid, fmt, mode);
      if (c && kind(c.act) === "buy") targets.push({ sid, c, owner: teamName(r) });
    }
  }
  targets.sort((a, b) => pushFirst(a) - pushFirst(b) || byPrice(a, b));
  $("targets").innerHTML = table(targets, { owner: true });

  // this season: contenders add production that's cheap in dynasty terms; rebuilders sell it to them.
  // Uses the FantasyCalc rest-of-season value against its dynasty value, both in this format.
  const nowRows = [], buyers = state.rosters.filter((r) => r.roster_id !== roster.roster_id && state.suggested[r.roster_id]?.mode === "c");
  const lean = (sid) => { const p = D.players[sid], m = p && mkt(p, fmt); return m && m[3] && m[3] <= 100 && healthy(p) ? m : null; };
  if (mode !== "r") {
    $("nowTitle").textContent = "Win-now targets";
    $("nowHint").textContent = "Healthy players on other teams whose rest-of-season value is high for their dynasty price: production you can add cheaply. Owners in the bottom half are the likeliest sellers.";
    for (const r of state.rosters) {
      if (r.roster_id === roster.roster_id) continue;
      const sg = state.suggested[r.roster_id];
      for (const sid of rosterPlayers(r)) {
        const m = lean(sid);
        if (m && m[2] >= 1.1 * m[1]) nowRows.push({ sid, c: call(sid, fmt, mode), owner: teamName(r), ownerHint: sg?.mode === "r" ? "bottom half" : "", k: m[2] / m[1] });
      }
    }
  } else {
    $("nowTitle").textContent = "Sell to contenders";
    $("nowHint").textContent = `Your healthy players whose rest-of-season value is well above their dynasty price: worth more to a team chasing a title this year. ` +
      (buyers.length ? `Likely buyers (top half by ${state.suggested[roster.roster_id]?.by}): ${buyers.map(teamName).join(", ")}.` : "");
    for (const sid of rosterPlayers(roster)) {
      const m = lean(sid);
      if (m && m[2] >= 1.25 * m[1]) nowRows.push({ sid, c: call(sid, fmt, mode), k: m[2] - m[1] });
    }
  }
  nowRows.sort((a, b) => b.k - a.k);
  $("now").innerHTML = D.meta.market ? table(nowRows.slice(0, 15), { owner: mode !== "r" }) : `<p class="hint">No trade-market data in this build.</p>`;

  // free agents
  const fa = [];
  for (const sid of Object.keys(D.players)) {
    if (owner[sid]) continue;
    const p = D.players[sid], c = call(sid, fmt, mode), k = kind(c.act);
    const producing = p.prk && p.prk <= START[fmt][p.p];
    const pr = price(p, fmt);
    if (["buy", "hold", "lotto"].includes(k) && ((pr != null && pr <= 300) || producing)) fa.push({ sid, c });
  }
  fa.sort(byPrice);
  $("fa").innerHTML = table(fa.slice(0, 40)) + (fa.length > 40 ? `<p class="hint">Showing the 40 highest-priced of ${fa.length}.</p>` : "");
  $("out").hidden = false;
}

const ordinal = (n) => n + (["th", "st", "nd", "rd"][(n % 100 - 20) % 10] || ["th", "st", "nd", "rd"][n % 100] || "th");

// ---------- wiring ----------
$("load").addEventListener("submit", (e) => { e.preventDefault(); loadInput($("league").value); });
$("leaguePick").addEventListener("change", (e) => e.target.value && loadLeague(e.target.value));
$("team").addEventListener("change", (e) => {
  const rid = Number(e.target.value), saved = JSON.parse(localStorage.getItem(`prefs:${state.league.league_id}`) || "{}");
  pickTeam(rid, saved.modes2?.[rid]);
});
$("mode").addEventListener("click", (e) => { const v = e.target.dataset?.v; if (v) { state.mode = v; render(); } });
$("fmt").addEventListener("click", (e) => { const v = e.target.dataset?.v; if (v) { state.fmt = v; render(); } });

const m = D.meta;
$("foot").innerHTML = `Prices: FantasyPros dynasty rankings via <a href="https://github.com/dynastyprocess/data">DynastyProcess</a>, ${esc(m.values_date)}. ` +
  `Production: <a href="https://github.com/nflverse/nflverse-data">nflverse</a> ${m.season} stats through ${m.games} games. Rebuilt daily, last ${esc(m.built)}. ` +
  `Ages are as of September ${m.next}. Price bands: FantasyPros rank 1-50, 51-100, 101-200, 201-300; past 300 is waiver level. ` +
  `The calls come from backtests of 2018-2025 rankings against later VORP; players 30+ use 1999-${m.last} history. ` +
  (m.market ? `Trade market: <a href="https://fantasycalc.com">FantasyCalc</a> dynasty values (12 teams, PPR), with "≈ pick" labels from its ${m.picks?.sf_year ?? ""} pick values. ` : "") +
  (m.ros ? `Rest of season (ROS) and next game: FantasyPros ROS PPR and weekly rankings via DynastyProcess, ${esc(m.fp_date)}; the ROS overall rank is 1QB. ` +
    `In season the calls use the "in-season" rank: the FantasyPros dynasty rank tilted slightly toward the ROS rank (a player whose ROS positional rank is half his dynasty one moves up about 7%). ` : "") +
  `Injuries: <a href="https://sleeper.com">Sleeper</a>. ` +
  `The market, injury and rest-of-season adjustments (the notes under each call, and the Stash, Buy low and Sell when healthy calls) are judgment calls, not backtested. ` +
  `The exception is the notes on rank moves since the preseason and on rest-of-season ranks well above dynasty ranks: those sells and buys, shown from week 5 to week 14, come from a league simulation on the same seasons that suggested them. ` +
  `<a href="https://github.com/JamesYouL2/data/tree/career-value/pages">Source and method</a>.`;

(function init() {
  const h = new URLSearchParams(location.hash.slice(1));
  const id = h.get("league") || localStorage.getItem("league");
  if (h.get("team") && id) {
    const k = `prefs:${id}`, cur = JSON.parse(localStorage.getItem(k) || "{}");
    cur.roster = Number(h.get("team"));
    localStorage.setItem(k, JSON.stringify(cur));
  }
  if (id) { $("league").value = id; loadLeague(id); }
})();
