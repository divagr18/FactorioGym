"""Replay inspection for an agent run (PLAN.md 5.5).

    Create a local viewer with timeline, map view, character actions,
    inventories, throughput, errors, and interventions.

Reads a run directory and writes one self-contained HTML file beside it. Every
byte the page needs is embedded: no CDN, no fetch, no provider. That is not a
convenience, it is the acceptance criterion -- "replay can be inspected without
contacting the model provider" is a property of a file with no network calls in
it, not a promise about how it is used.

What the page shows, and why each part is there
-----------------------------------------------
*The prompt that was actually sent*, not one reconstructed from the stored
observation. The loop records both because they can disagree: a summary is a
rendering, and re-rendering it later with changed code would quietly answer a
different question than the model was asked.

*Assisted actions expanded.* One `approach_entity_1` is one decision to the
model and up to sixty to the engine. A step count cannot show that a skill spent
those steps walking out and back, which is exactly the failure that cost an
entire episode and had to be reconstructed by hand. The primitives are listed.

*Interventions marked.* A decision the model did not make -- a fallback after a
malformed reply -- is a different event from one it did, and a replay that
renders them identically is misleading in the direction that flatters the model.

Evaluator information
---------------------
`success` and `reward` are computed by the evaluator from ground truth the
policy never sees. They are shown, because a replay without them is not useful,
and they are labelled, because PLAN 5.5 requires debug overlays to be identified
as evaluator information rather than left to look like part of the observation.
The observation panel contains only what was on the wire.

Run: uv run python tools/replay.py runtime/runs/<run_id>
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --bg:#12141a; --panel:#1a1d26; --line:#2b3040; --ink:#e6e9f0; --dim:#98a0b3;
    --ok:#5ad19a; --bad:#ff7b72; --warn:#e3b341; --accent:#79b8ff; --evaluator:#c792ea;
    --addr:#f0a35e;
    /* Arguments the model supplied, distinct from a target it addressed. */
    --arg:#7fb2f0;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  header {{ padding:8px 14px; border-bottom:1px solid var(--line); display:flex;
            gap:16px; align-items:baseline; flex-wrap:wrap; }}
  header b {{ font-size:15px; }} header span {{ color:var(--dim); }}
  .key {{ border:1px solid var(--line); border-radius:3px; padding:0 4px; color:var(--dim); }}
  .wrap {{ display:grid; grid-template-columns:320px 1fr 1fr; height:calc(100vh - 42px); }}
  .col {{ overflow:auto; border-right:1px solid var(--line); padding:10px; }}
  .filters {{ display:flex; gap:4px; flex-wrap:wrap; margin-bottom:8px; }}
  .filters button {{ background:var(--panel); color:var(--dim); border:1px solid var(--line);
                     border-radius:4px; padding:2px 7px; cursor:pointer; font:inherit; }}
  .filters button.on {{ color:var(--ink); border-color:var(--accent); }}
  input[type=search] {{ width:100%; background:var(--panel); color:var(--ink);
    border:1px solid var(--line); border-radius:4px; padding:4px 7px; font:inherit;
    margin-bottom:8px; }}
  .ep {{ color:var(--dim); margin:12px 0 4px; display:flex; justify-content:space-between;
         text-transform:uppercase; letter-spacing:.07em; font-size:11px; }}
  .row {{ padding:4px 7px; border-radius:5px; cursor:pointer; display:flex;
          gap:7px; align-items:baseline; border:1px solid transparent; }}
  .row:hover {{ background:#232735; }}
  .row.sel {{ background:#2a3040; border-color:var(--accent); }}
  .row .k {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .n {{ color:var(--dim); min-width:26px; }}
  .tag {{ font-size:11px; padding:0 5px; border-radius:3px; }}
  .t-ok {{ color:var(--ok); }} .t-bad {{ color:var(--bad); }} .t-warn {{ color:var(--warn); }}
  .t-addr {{ color:var(--addr); }}
  .t-arg {{ color:var(--arg); }}
  h3 {{ margin:14px 0 6px; font-size:12px; color:var(--dim);
        text-transform:uppercase; letter-spacing:.08em; }}
  h3:first-child {{ margin-top:0; }}
  pre {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
         padding:9px; overflow:auto; white-space:pre-wrap; word-break:break-word; margin:0; }}
  .ev {{ border-color:var(--evaluator); }}
  .evlabel {{ color:var(--evaluator); }}
  table {{ border-collapse:collapse; width:100%; }}
  td {{ padding:2px 6px 2px 0; vertical-align:top; }}
  td.f {{ color:var(--dim); width:118px; }}
  canvas {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
            width:100%; height:auto; }}
  details {{ background:var(--panel); border:1px solid var(--line);
             border-radius:6px; padding:7px 9px; margin-bottom:5px; }}
  summary {{ cursor:pointer; }}
  .empty {{ color:var(--dim); }}
  .strip {{ display:flex; gap:1px; margin:6px 0 2px; }}
  .strip i {{ flex:1; height:14px; background:var(--line); border-radius:1px; cursor:pointer; }}
</style>
<header>
  <b>{title}</b>
  <span>{task}</span>
  <span>{model}</span>
  <span>{counts}</span>
  <span class="evlabel">purple = evaluator information, not visible to the agent</span>
  <span><i class="key">j</i>/<i class="key">k</i> or arrows to step</span>
</header>
<div class="wrap">
  <div class="col">
    <input type="search" id="q" placeholder="filter by action, reason or handle">
    <div class="filters" id="filters"></div>
    <div id="timeline"></div>
  </div>
  <div class="col" id="left"></div>
  <div class="col" id="right"></div>
</div>
<script>
const DECISIONS = {data};
const esc = s => String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
let current = 0;
let filter = 'all';
let query = '';

const FILTERS = [
  ['all', 'all'],
  ['addressed', 'addressed'],
  ['built', 'placements'],
  ['assisted', 'assisted'],
  ['errors', 'refused'],
  ['interventions', 'interventions'],
  ['solved', 'solved'],
];

function matches(d) {{
  const r = d.result || {{}};
  if (filter === 'addressed' && !d.target) return false;
  if (filter === 'built' && d.action_key !== 'place_at') return false;
  if (filter === 'assisted' && !r.skill) return false;
  if (filter === 'errors' && !r.action_error) return false;
  if (filter === 'interventions' && (!d.resolution || d.resolution === 'model')) return false;
  if (filter === 'solved' && !r.success) return false;
  if (query) {{
    const hay = [d.action_key, d.target, argsText(d),
                 (d.attempts || []).map(a => a.text).join(' ')]
      .join(' ').toLowerCase();
    if (!hay.includes(query)) return false;
  }}
  return true;
}}

function argsText(d) {{
  // R3.2's gate is worded about what the *replay* shows, and for a
  // construction run the substance is *where* a machine went. A bare action
  // key cannot say: `place_at` appears 31 times in one run at 31 different
  // tiles. Rendered compactly so the timeline stays readable.
  const a = d.arguments || {{}};
  const keys = Object.keys(a);
  if (!keys.length) return '';
  return keys.sort().map(function (k) {{
    const v = a[k];
    const shown = Array.isArray(v)
      ? '(' + v.map(function (n) {{ return (+n).toFixed(1); }}).join(', ') + ')'
      : String(v);
    return k + '=' + shown;
  }}).join(' ');
}}

function badge(d) {{
  const r = d.result || {{}};
  let out = '';
  if (d.target) out += `<span class="tag t-addr">&rarr;${{esc(d.target)}}</span>`;
  const args = argsText(d);
  if (args) out += `<span class="tag t-arg">${{esc(args)}}</span>`;
  if (r.success) out += '<span class="tag t-ok">solved</span>';
  else if (r.infrastructure_failure) out += '<span class="tag t-warn">infra</span>';
  else if (r.action_error) out += `<span class="tag t-bad">${{esc(r.action_error)}}</span>`;
  else if (d.resolution && d.resolution !== 'model')
    out += '<span class="tag t-warn">intervention</span>';
  return out;
}}

function episodeSummary(rows) {{
  const solved = rows.some(d => (d.result || {{}}).success);
  const refused = rows.filter(d => (d.result || {{}}).action_error).length;
  const addressed = rows.filter(d => d.target).length;
  const built = rows.filter(d => d.action_key === 'place_at'
                              && !(d.result || {{}}).action_error).length;
  return `${{rows.length}} decisions &middot; ${{addressed}} addressed &middot; `
       + `${{built}} placed &middot; `
       + `${{refused}} refused &middot; ${{solved ? 'solved' : 'unsolved'}}`;
}}

function renderFilters() {{
  document.getElementById('filters').innerHTML = FILTERS.map(([k, label]) =>
    `<button data-f="${{k}}" class="${{k === filter ? 'on' : ''}}">${{label}}</button>`).join('');
  document.querySelectorAll('#filters button').forEach(b =>
    b.onclick = () => {{ filter = b.dataset.f; renderFilters(); timeline(); }});
}}

function timeline() {{
  const host = document.getElementById('timeline');
  const byEpisode = new Map();
  DECISIONS.forEach((d, i) => {{
    if (!byEpisode.has(d.episode)) byEpisode.set(d.episode, []);
    byEpisode.get(d.episode).push(i);
  }});
  let html = '';
  for (const [episode, all] of byEpisode) {{
    const shown = all.filter(i => matches(DECISIONS[i]));
    if (!shown.length) continue;
    html += `<div class="ep"><span>episode ${{episode}}</span>`
          + `<span>${{episodeSummary(all.map(i => DECISIONS[i]))}}</span></div>`;
    // One cell per decision, coloured by outcome: the shape of an episode at a
    // glance, and a click target for jumping into it.
    html += '<div class="strip">' + all.map(i => {{
      const r = DECISIONS[i].result || {{}};
      const colour = r.success ? 'var(--ok)' : r.action_error ? 'var(--bad)'
                   : DECISIONS[i].target ? 'var(--addr)' : 'var(--line)';
      const label = esc(DECISIONS[i].action_key);
      return `<i data-i="${{i}}" style="background:${{colour}}" title="${{label}}"></i>`;
    }}).join('') + '</div>';
    html += shown.map(i => {{
      const d = DECISIONS[i];
      return `<div class="row" data-i="${{i}}"><span class="n">${{d.step}}</span>`
           + `<span class="k">${{esc(d.action_key)}}</span>${{badge(d)}}</div>`;
    }}).join('');
  }}
  host.innerHTML = html || '<div class="empty">nothing matches</div>';
  host.querySelectorAll('[data-i]').forEach(el =>
    el.onclick = () => select(parseInt(el.dataset.i, 10)));
  const sel = host.querySelector(`.row[data-i="${{current}}"]`);
  if (sel) sel.classList.add('sel');
}}

function kv(rows) {{
  return '<table>' + rows.map(([k, v]) =>
    `<tr><td class="f">${{esc(k)}}</td><td>${{v}}</td></tr>`).join('') + '</table>';
}}

// Drawn from the observation only -- same entities, same resources, same
// published objective the agent was given. Nothing from evaluator truth reaches
// this canvas, which is why it carries no overlay label.
function drawMap(index) {{
  const c = document.getElementById('map');
  if (!c) return;
  const o = DECISIONS[index].observation || {{}};
  const ctx = c.getContext('2d'), W = c.width, H = c.height, R = 34;
  const ch = (o.character || {{}}).position || [0, 0];
  const px = (x, y) => [W/2 + (x - ch[0]) / R * (W/2), H/2 + (y - ch[1]) / R * (H/2)];
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#2b3040';
  ctx.beginPath(); ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H);
  ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();

  // Where the character has already been this episode. A single frame cannot
  // show pacing; the trail is what made a policy walking nine tiles out and
  // twelve back legible as a loop rather than as progress.
  const trail = [];
  for (let i = index; i >= 0 && DECISIONS[i].episode === DECISIONS[index].episode; i--) {{
    const p = ((DECISIONS[i].observation || {{}}).character || {{}}).position;
    if (p) trail.unshift(p);
  }}
  if (trail.length > 1) {{
    ctx.strokeStyle = '#3d4658'; ctx.lineWidth = 1.5; ctx.beginPath();
    trail.forEach((p, n) => {{ const [x, y] = px(p[0], p[1]);
      n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }});
    ctx.stroke();
  }}

  ((o.resources || {{}}).tiles || []).forEach(t => {{
    const p = t.p || t.offset || [0, 0];
    const [x, y] = t.p ? px(p[0], p[1]) : px(ch[0] + p[0], ch[1] + p[1]);
    ctx.fillStyle = '#3b4a5a'; ctx.fillRect(x - 2, y - 2, 4, 4);
  }});
  const target = DECISIONS[index].target;
  (o.entities || []).forEach(e => {{
    const p = e.p || e.offset || [0, 0];
    const [x, y] = e.p ? px(p[0], p[1]) : px(ch[0] + p[0], ch[1] + p[1]);
    const handle = String(e.handle || e.h || '');
    const addressed = target && handle === target;
    ctx.fillStyle = addressed ? '#f0a35e' : e.remembered ? '#5a6478' : '#79b8ff';
    ctx.fillRect(x - 3, y - 3, 6, 6);
    if (addressed) {{
      ctx.strokeStyle = '#f0a35e'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 8, 0, 6.284); ctx.stroke();
    }}
    ctx.fillStyle = '#98a0b3'; ctx.font = '9px monospace';
    ctx.fillText(handle || String(e.name || ''), x + 6, y + 3);
  }});
  Object.entries(o.goal || {{}}).forEach(([name, p]) => {{
    const [x, y] = px(p[0], p[1]);
    ctx.strokeStyle = '#5ad19a'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.284); ctx.stroke();
    ctx.fillStyle = '#5ad19a'; ctx.fillText(name, x + 9, y + 3);
  }});
  ctx.fillStyle = '#e3b341';
  ctx.beginPath(); ctx.arc(W/2, H/2, 4, 0, 6.284); ctx.fill();
}}

function select(i) {{
  current = i;
  const d = DECISIONS[i], o = d.observation || {{}}, r = d.result || {{}};
  document.querySelectorAll('.row').forEach(el =>
    el.classList.toggle('sel', parseInt(el.dataset.i, 10) === i));

  const inv = Object.entries(o.inventory || {{}});
  const counters = Object.entries(o.counters || {{}});
  let left = '<h3>map view (observation only)</h3>'
    + '<canvas id="map" width="420" height="420"></canvas>';
  left += '<h3>character</h3>' + kv([
    ['position', esc(JSON.stringify((o.character || {{}}).position))],
    ['tick', esc(o.tick)],
    ['decision', esc(d.step) + ' of episode ' + esc(d.episode)],
  ]);
  left += '<h3>inventory</h3>' + (inv.length
    ? kv(inv.map(([k, v]) => [k, esc(v)])) : '<div class="empty">empty</div>');
  left += '<h3>throughput</h3>' + (counters.length
    ? kv(counters.map(([k, v]) => [k, esc(v)])) : '<div class="empty">none recorded</div>');
  left += '<h3>objective</h3>' + (Object.keys(o.goal || {{}}).length
    ? kv(Object.entries(o.goal).map(([k, v]) => [k, esc(JSON.stringify(v))]))
    : '<div class="empty">this task publishes no marker</div>');
  left += '<h3>prompt as sent</h3><pre>' + esc(d.prompt || '(not recorded)') + '</pre>';
  document.getElementById('left').innerHTML = left;

  let right = '<h3>action</h3>' + kv([
    ['chosen', esc(d.action_key) + ' <span class="n">#' + esc(d.action_index) + '</span>'],
    ['arguments', argsText(d)
      ? '<span class="t-arg">' + esc(argsText(d)) + '</span> (supplied by the model)'
      : '<span class="empty">none (this action takes no arguments)</span>'],
    ['target', d.target
      ? '<span class="t-addr">' + esc(d.target) + '</span> (addressed)'
      : '<span class="empty">nearest entity (catalog default)</span>'],
    ['resolution', d.resolution === 'model' ? 'model'
      : '<span class="t-warn">' + esc(d.resolution) + ' (intervention)</span>'],
    ['status', esc(r.action_status)],
    ['error', r.action_error ? '<span class="t-bad">' + esc(r.action_error) + '</span>' : 'none'],
    ['latency', esc(d.inference_ms) + ' ms'],
  ]);

  const trace = r.skill_trace || [];
  if (r.skill) {{
    right += '<h3>assisted action</h3><details open><summary>'
      + esc(r.skill) + ' &rarr; ' + esc(r.skill_outcome)
      + ' (' + esc(r.skill_steps) + ' primitive steps)</summary><table>'
      + (trace.length ? trace.map((t, n) =>
          `<tr><td class="f">${{n}}</td><td>${{esc(t.key)}}`
          + (t.error ? ' <span class="t-bad">' + esc(t.error) + '</span>' : '')
          + `</td></tr>`).join('')
        : '<tr><td class="empty">no primitive trace recorded for this run</td></tr>')
      + '</table></details>';
  }}

  right += '<h3 class="evlabel">evaluator information</h3><pre class="ev">'
    + esc(JSON.stringify({{
        success: r.success, reward: r.reward,
        terminated: r.terminated, truncated: r.truncated,
        infrastructure_failure: r.infrastructure_failure,
      }}, null, 2)) + '</pre>';

  const attempts = d.attempts || [];
  right += '<h3>model attempts (' + attempts.length + ')</h3>';
  right += attempts.map(a =>
    '<details' + (a.failure ? ' open' : '') + '><summary>attempt ' + esc(a.attempt)
    + (a.failure ? ' <span class="t-bad">' + esc(a.failure) + '</span>' : ' ok')
    + ' &middot; ' + esc(Math.round(a.latency_ms || 0)) + ' ms'
    + ' &middot; ' + esc((a.usage || {{}}).total_tokens || 0) + ' tokens</summary><pre>'
    + esc(a.text || a.error || '(no text)') + '</pre>'
    + (a.detail ? '<pre class="t-bad">' + esc(a.detail) + '</pre>' : '')
    + '</details>').join('');

  right += '<h3>legal actions offered</h3><pre>'
    + esc((d.legal_actions || []).map(a => a.index + ': ' + a.key).join('\n')) + '</pre>';
  document.getElementById('right').innerHTML = right;
  drawMap(i);
}}

function move(delta) {{
  const shown = DECISIONS.map((d, i) => i).filter(i => matches(DECISIONS[i]));
  if (!shown.length) return;
  const at = shown.indexOf(current);
  const next = at === -1 ? 0 : Math.min(shown.length - 1, Math.max(0, at + delta));
  select(shown[next]);
  const row = document.querySelector(`.row[data-i="${{shown[next]}}"]`);
  if (row) row.scrollIntoView({{block: 'nearest'}});
}}

document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'j' || e.key === 'ArrowDown') {{ e.preventDefault(); move(1); }}
  if (e.key === 'k' || e.key === 'ArrowUp') {{ e.preventDefault(); move(-1); }}
  if (e.key === 'g') select(0);
  if (e.key === 'G') select(DECISIONS.length - 1);
}});
document.getElementById('q').addEventListener('input', e => {{
  query = e.target.value.toLowerCase(); timeline();
}});

renderFilters();
timeline();
if (DECISIONS.length) select(0);
</script>
"""


def build(run_dir: Path, out: Path | None) -> Path:
    decisions_path = run_dir / "decisions.jsonl"
    if not decisions_path.exists():
        raise SystemExit(
            f"{run_dir} has no decisions.jsonl. Replay inspection is defined on agent "
            "runs; a training run records curve.csv instead."
        )
    decisions = [
        json.loads(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    task = manifest.get("task") or {}
    model = manifest.get("model") or {}
    episodes = sorted({d.get("episode") for d in decisions})
    solved = sum(
        1
        for episode in episodes
        if any(
            (d.get("result") or {}).get("success") for d in decisions if d.get("episode") == episode
        )
    )
    interventions = sum(1 for d in decisions if d.get("resolution") not in (None, "model"))

    page = PAGE.format(
        title=html.escape(manifest.get("run_id") or run_dir.name),
        task=html.escape(f"{task.get('id', '?')} v{task.get('version', '?')}"),
        # The model's identity, never its credential: `model` carries
        # `api_key_env` and `credential_present`, and the adapter redacts before
        # the manifest is written. Naming the fields explicitly keeps a future
        # adapter field from being rendered into a shareable file by accident.
        model=html.escape(str(model.get("model") or "no model recorded")),
        counts=html.escape(
            f"{len(decisions)} decisions, {len(episodes)} episodes, "
            f"{solved} solved, {interventions} interventions"
        ),
        data=json.dumps(decisions),
    )
    destination = out or (run_dir / "replay.html")
    destination.write_text(page, encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="run directory under runtime/runs/")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        candidate = ROOT / run_dir
        run_dir = candidate if candidate.exists() else run_dir
    destination = build(run_dir, Path(args.out) if args.out else None)
    size = destination.stat().st_size
    print(f"wrote {destination} ({size / 1024:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
