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
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  header {{ padding:10px 14px; border-bottom:1px solid var(--line); display:flex;
            gap:18px; align-items:baseline; flex-wrap:wrap; }}
  header b {{ font-size:15px; }} header span {{ color:var(--dim); }}
  .wrap {{ display:grid; grid-template-columns:300px 1fr 1fr; height:calc(100vh - 46px); }}
  .col {{ overflow:auto; border-right:1px solid var(--line); padding:10px; }}
  .ep {{ color:var(--dim); margin:12px 0 4px; text-transform:uppercase; letter-spacing:.08em; }}
  .row {{ padding:5px 7px; border-radius:5px; cursor:pointer; display:flex;
          gap:8px; align-items:baseline; border:1px solid transparent; }}
  .row:hover {{ background:#232735; }}
  .row.sel {{ background:#2a3040; border-color:var(--accent); }}
  .row .k {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .n {{ color:var(--dim); min-width:26px; }}
  .tag {{ font-size:11px; padding:0 5px; border-radius:3px; }}
  .t-ok {{ color:var(--ok); }} .t-bad {{ color:var(--bad); }} .t-warn {{ color:var(--warn); }}
  h3 {{ margin:14px 0 6px; font-size:12px; color:var(--dim);
        text-transform:uppercase; letter-spacing:.08em; }}
  h3:first-child {{ margin-top:0; }}
  pre {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
         padding:9px; overflow:auto; white-space:pre-wrap; word-break:break-word; margin:0; }}
  .ev {{ border-color:var(--evaluator); }}
  .evlabel {{ color:var(--evaluator); }}
  table {{ border-collapse:collapse; width:100%; }}
  td {{ padding:2px 6px 2px 0; vertical-align:top; }}
  td.f {{ color:var(--dim); width:110px; }}
  canvas {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
            width:100%; height:auto; }}
  details {{ background:var(--panel); border:1px solid var(--line);
             border-radius:6px; padding:7px 9px; }}
  summary {{ cursor:pointer; }}
  .empty {{ color:var(--dim); }}
</style>
<header>
  <b>{title}</b>
  <span>{task}</span>
  <span>{model}</span>
  <span>{counts}</span>
  <span class="evlabel">purple = evaluator information, not visible to the agent</span>
</header>
<div class="wrap">
  <div class="col" id="timeline"></div>
  <div class="col" id="left"></div>
  <div class="col" id="right"></div>
</div>
<script>
const DECISIONS = {data};
const esc = s => String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));

function badge(d) {{
  const r = d.result || {{}};
  if (r.success) return '<span class="tag t-ok">solved</span>';
  if (r.infrastructure_failure) return '<span class="tag t-warn">infra</span>';
  if (r.action_error) return '<span class="tag t-bad">' + esc(r.action_error) + '</span>';
  if (d.resolution && d.resolution !== 'model')
    return '<span class="tag t-warn">intervention</span>';
  return '';
}}

function timeline() {{
  const host = document.getElementById('timeline');
  let last = null, html = '';
  DECISIONS.forEach((d, i) => {{
    if (d.episode !== last) {{
      html += `<div class="ep">episode ${{d.episode}}</div>`;
      last = d.episode;
    }}
    html += `<div class="row" data-i="${{i}}"><span class="n">${{d.step}}</span>`
          + `<span class="k">${{esc(d.action_key)}}</span>${{badge(d)}}</div>`;
  }});
  host.innerHTML = html;
  host.querySelectorAll('.row').forEach(el =>
    el.onclick = () => select(parseInt(el.dataset.i, 10)));
}}

function kv(rows) {{
  return '<table>' + rows.map(([k, v]) =>
    `<tr><td class="f">${{esc(k)}}</td><td>${{v}}</td></tr>`).join('') + '</table>';
}}

// The map is drawn from the observation only -- same entities, same resources,
// same published objective the agent was given. Nothing from evaluator truth
// reaches this canvas, which is why it is not marked as an overlay.
function drawMap(o) {{
  const c = document.getElementById('map');
  if (!c) return;
  const ctx = c.getContext('2d'), W = c.width, H = c.height, R = 34;
  const ch = (o.character || {{}}).position || [0, 0];
  const px = (x, y) => [W/2 + (x - ch[0]) / R * (W/2), H/2 + (y - ch[1]) / R * (H/2)];
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#2b3040';
  ctx.beginPath(); ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H);
  ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();
  ((o.resources || {{}}).tiles || []).forEach(t => {{
    const [x, y] = px((t.p || [0,0])[0], (t.p || [0,0])[1]);
    ctx.fillStyle = '#3b4a5a'; ctx.fillRect(x - 2, y - 2, 4, 4);
  }});
  (o.entities || []).forEach(e => {{
    const p = e.p || e.offset || [0, 0];
    const [x, y] = e.p ? px(p[0], p[1]) : px(ch[0] + p[0], ch[1] + p[1]);
    ctx.fillStyle = e.remembered ? '#5a6478' : '#79b8ff';
    ctx.fillRect(x - 3, y - 3, 6, 6);
    ctx.fillStyle = '#98a0b3'; ctx.font = '9px monospace';
    ctx.fillText(String(e.handle || e.name || ''), x + 6, y + 3);
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
    ['decision', esc(d.step)],
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
    + esc(a.text || a.error || '(no text)') + '</pre></details>').join('');

  right += '<h3>legal actions offered</h3><pre>'
    + esc((d.legal_actions || []).map(a => a.index + ': ' + a.key).join('\\n')) + '</pre>';
  document.getElementById('right').innerHTML = right;
  drawMap(o);
}}

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
