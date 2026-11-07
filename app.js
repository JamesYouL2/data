"use strict";
const D = window.DATA;
const API = "https://api.sleeper.app/v1";
const $ = (id) => document.getElementById(id);
const START = { "1qb": { QB: 12, RB: 24, WR: 30, TE: 12 }, sf: { QB: 24, RB: 24, WR: 30, TE: 12 } };

const state = { league: null, users: {}, rosters: [], roster: null, mode: null, fmt: null, autoFmt: null, suggested: {} };

// ---------- helpers ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const kind = (act) => !act ? "none" : act.startsWith("Buy") ? "buy" : act.startsWith("Hold") ? "hold" : act.startsWith("Use") ? "use"
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
  const [grp, ra, rw, ca, cw] = r;
  return mode === "c" ? { grp, act: ca, why: cw, other: ra, otherLbl: "Rebuilding" } : { grp, act: ra, why: rw, other: ca, otherLbl: "Contending" };
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
    pickTeam(Number($("team").value), saved.modes?.[$("team").value]);
  } catch (e) {
    status(e.message === "not found" ? `No Sleeper league with ID ${id}.` : `Couldn't load the league: ${e.message}`, true);
  }
}

// Suggest contending for the top half of the league: by record and points for once games are played,
// otherwise by roster strength at the current FantasyPros prices.
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
  cur.modes = { ...(cur.modes || {}), [state.roster.roster_id]: state.mode };
  localStorage.setItem(k, JSON.stringify(cur));
  history.replaceState(null, "", `#league=${id}&team=${state.roster.roster_id}`);
}

function pickTeam(rid, mode) {
  state.roster = state.rosters.find((r) => r.roster_id === rid) || state.rosters[0];
  state.mode = mode || state.suggested[state.roster.roster_id]?.mode || "r";
  render();
}

// ---------- rendering ----------
function setSeg(id, v) { for (const b of $(id).querySelectorAll("button")) b.classList.toggle("on", b.dataset.v === v); }

function playerCells(sid, fmt) {
  const p = D.players[sid];
  if (!p) {
    const n = D.names[sid];
    if (n) return { name: esc(n[0]), meta: `${esc(n[1])}${n[2] ? " · " + esc(n[2]) : ""}`, price: "–", prod: "–" };
    if (/^[A-Z]{2,3}$/.test(sid)) return { name: `${esc(sid)} defense`, meta: "DEF", price: "–", prod: "–" };
    return { name: `Player ${esc(sid)}`, meta: "not in the rankings", price: "–", prod: "–" };
  }
  const meta = [p.p, p.t, p.a != null ? `${Math.floor(p.a)} next Sept` : null,
    p.dy === D.meta.rookie ? "rookie" : null, p.rd ? `NFL R${p.rd}` : p.dy ? "UDFA" : null].filter(Boolean).map(esc).join(" · ");
  const pr = price(p, fmt);
  const prc = pr ? `#${pr} <span class="meta">${esc(p.p)}${posPrice(p, fmt)}</span>` : `<span class="meta">unranked</span>`;
  const prod = p.ppg != null ? `${p.ppg.toFixed(1)} <span class="meta">${p.prk ? esc(p.p) + p.prk : ""} · ${p.g}g</span>` : `<span class="meta">${p.prkl ? `${D.meta.last}: ${esc(p.p)}${p.prkl}` : "–"}</span>`;
  return { name: esc(p.n), meta, price: prc, prod };
}

function table(rows, opts = {}) {
  if (!rows.length) return `<p class="hint">None.</p>`;
  const head = `<tr><th>Player</th>${opts.owner ? "<th>Team</th>" : ""}<th>Price (FP rank)</th><th>${D.meta.season} PPR/game</th><th>Call</th><th>Why</th></tr>`;
  const body = rows.map(({ sid, c, owner, tags }) => {
    const x = playerCells(sid, state.fmt);
    const other = c && c.other !== c.act ? `<div class="other">${c.otherLbl}: ${esc(c.other)}</div>` : "";
    return `<tr><td><span class="nm">${x.name}</span>${(tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join("")}<div class="meta">${x.meta}</div></td>` +
      (opts.owner ? `<td class="meta">${esc(owner)}</td>` : "") +
      `<td class="num">${x.price}</td><td class="num">${x.prod}</td><td>${badge(c?.act)}${other}</td>` +
      `<td class="why">${c ? `<span class="grp">${esc(c.grp)}</span>${esc(c.why)}` : ""}</td></tr>`;
  }).join("");
  return `<div class="scroll"><table class="tbl">${head}${body}</table></div>`;
}

function render() {
  const { fmt, mode, roster } = state;
  setSeg("mode", mode); setSeg("fmt", fmt);
  $("team").value = roster.roster_id;
  const sug = state.suggested[roster.roster_id];
  $("modeHint").textContent = sug ? `Suggested: ${sug.mode === "c" ? "contending" : "rebuilding"} (${ordinal(sug.place)} of ${state.rosters.length} by ${sug.by}).` : "";
  $("fmtHint").textContent = `League settings say ${state.autoFmt === "sf" ? "superflex" : "1QB"}.`;
  savePrefs();

  // roster, grouped by call
  const taxi = new Set(roster.taxi || []), ir = new Set(roster.reserve || []);
  const mine = rosterPlayers(roster).map((sid) => ({ sid, c: call(sid, fmt, mode), tags: [taxi.has(sid) && "taxi", ir.has(sid) && "IR"].filter(Boolean) }));
  const groups = Object.fromEntries(ORDER.map((k) => [k, []]));
  for (const r of mine) groups[kind(r.c?.act)].push(r);
  const byPrice = (a, b) => rankKey(D.players[a.sid] || {}, fmt) - rankKey(D.players[b.sid] || {}, fmt);
  let html = "";
  for (const k of ORDER) {
    if (!groups[k].length) continue;
    groups[k].sort(byPrice);
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
  targets.sort(byPrice);
  $("targets").innerHTML = table(targets, { owner: true });

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
  pickTeam(rid, saved.modes?.[rid]);
});
$("mode").addEventListener("click", (e) => { const v = e.target.dataset?.v; if (v) { state.mode = v; render(); } });
$("fmt").addEventListener("click", (e) => { const v = e.target.dataset?.v; if (v) { state.fmt = v; render(); } });

const m = D.meta;
$("foot").innerHTML = `Prices: FantasyPros dynasty rankings via <a href="https://github.com/dynastyprocess/data">DynastyProcess</a>, ${esc(m.values_date)}. ` +
  `Production: <a href="https://github.com/nflverse/nflverse-data">nflverse</a> ${m.season} stats through ${m.games} games. Rebuilt daily, last ${esc(m.built)}. ` +
  `Ages are as of September ${m.next}. Price bands: 151-239 is a 3rd-round rookie pick, 240-300 a 4th. ` +
  `The calls come from backtests of 2018-2025 rankings against later VORP; players 30+ use 1999-${m.last} history. ` +
  `<a href="https://github.com/JamesYouL2/dynasty-buy-sell">Source and method</a>.`;

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
