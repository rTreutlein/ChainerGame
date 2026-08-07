"""Dependency-free local browser interface for StationOps-v2."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .game import GameConfig, GameSession


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StationOps</title>
  <style>
    :root { color-scheme: dark; --bg:#071018; --panel:#101d27; --line:#29404d;
      --text:#e8f1f4; --muted:#91a6b0; --cyan:#55d6d0; --amber:#ffbd59;
      --red:#ff6b6b; --green:#75dc9a; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 20% 0,#142c38 0,var(--bg) 45%);
      color:var(--text); font:15px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
    main { max-width:1180px; margin:auto; padding:28px; }
    header { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; margin-bottom:22px; }
    h1 { margin:0; letter-spacing:.12em; font-size:27px; }
    h1 span { color:var(--cyan); }
    .subtitle,.muted { color:var(--muted); }
    .stats { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-bottom:18px; }
    .stat,.panel,.module { background:rgba(16,29,39,.94); border:1px solid var(--line); border-radius:8px; }
    .stat { padding:12px; }
    .stat label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }
    .stat strong { font-size:20px; color:var(--cyan); }
    .layout { display:grid; grid-template-columns:minmax(0,3fr) minmax(270px,1fr); gap:18px; }
    .panel { padding:17px; }
    .panel h2 { font-size:14px; letter-spacing:.08em; text-transform:uppercase; margin:0 0 14px; }
    #modules { display:grid; gap:10px; }
    .module { display:grid; grid-template-columns:1fr auto; gap:12px; padding:14px; transition:.15s; }
    .module.alarm { border-color:#8c6535; box-shadow:inset 4px 0 var(--amber); }
    .module.selected { outline:2px solid var(--cyan); }
    .module h3 { margin:0 0 5px; font-size:16px; }
    .tags { display:flex; flex-wrap:wrap; gap:7px; color:var(--muted); font-size:12px; }
    .tag { border:1px solid var(--line); border-radius:10px; padding:2px 7px; }
    .tag.alarm { color:var(--amber); border-color:#8c6535; }
    .tag.HIGH { color:var(--red); }.tag.MEDIUM { color:var(--amber); }.tag.LOW { color:var(--green); }
    .inspection { color:var(--cyan); margin-top:8px; }
    .actions { display:flex; gap:7px; align-items:center; }
    button { background:#18303b; color:var(--text); border:1px solid #3d6371; border-radius:5px;
      padding:8px 11px; cursor:pointer; font:inherit; }
    button:hover:not(:disabled) { background:#245063; }
    button.primary { background:#176c69; border-color:var(--cyan); }
    button.danger { background:#582b31; border-color:#99505a; }
    button:disabled { opacity:.4; cursor:not-allowed; }
    .side-section { border-top:1px solid var(--line); padding-top:14px; margin-top:14px; }
    .log-row { display:flex; justify-content:space-between; margin:7px 0; }
    #message { min-height:24px; color:var(--amber); margin:12px 0; }
    #report ul { padding-left:20px; }
    #report .good { color:var(--green); } #report .bad { color:var(--red); }
    .commit { width:100%; margin-top:14px; padding:12px; }
    @media(max-width:850px) { .stats{grid-template-columns:repeat(3,1fr)} .layout{grid-template-columns:1fr} }
    @media(max-width:520px) { main{padding:14px}.stats{grid-template-columns:repeat(2,1fr)}
      .module{grid-template-columns:1fr}.actions{justify-content:flex-start} }
  </style>
</head>
<body><main>
  <header><div><h1>STATION<span>OPS</span></h1><div class="subtitle">Refinery maintenance control</div></div>
    <div id="status" class="muted"></div></header>
  <section class="stats" id="stats"></section>
  <div class="layout">
    <section class="panel"><h2>Current station state</h2><div id="modules"></div></section>
    <aside class="panel">
      <h2>Shift orders</h2>
      <div class="muted">Inspect uncertain modules, select repairs, then commit the shift. Hidden conditions are learned only through inspection, maintenance, or failure.</div>
      <div id="rules" class="muted side-section"></div>
      <div id="message"></div>
      <button id="commit" class="primary commit">Commit selected repairs</button>
      <div class="side-section"><h2>Dependency map</h2><div id="topology"></div></div>
      <div class="side-section"><h2>Confirmed maintenance log</h2><div id="log"></div></div>
      <div class="side-section"><h2>Production shortfalls</h2><div id="shortfalls"></div></div>
      <div class="side-section" id="report"></div>
    </aside>
  </div>
</main>
<script>
let state=null;
let selected=new Set();
const el=id=>document.getElementById(id);
async function api(path, body={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw new Error(data.error||'Request failed');
  return data;
}
function stat(label,value){return `<div class="stat"><label>${label}</label><strong>${value}</strong></div>`}
function render(s) {
  state=s; selected=new Set([...selected].filter(id=>s.incidents.some(x=>x.id===id)));
  el('status').textContent=s.status==='active'?`SHIFT ${s.shift} / ${s.total_shifts}`:'EPISODE COMPLETE';
  el('stats').innerHTML=stat('Credits',s.credits)+stat('Seal kits',s.seal_kits)+
    stat('Diagnostics',s.diagnostics_remaining)+stat('Repair capacity',s.repair_capacity)+
    stat('Station score',Math.round(s.station_score))+stat('Production',s.total_production);
  el('modules').innerHTML='';
  for(const item of s.incidents) {
    const card=document.createElement('article');
    card.className=`module ${item.alarm?'alarm':''} ${selected.has(item.id)?'selected':''}`;
    const info=document.createElement('div');
    info.innerHTML=`<h3>${item.module_id} · ${item.description}</h3><div class="tags">
      <span class="tag">${item.cohort}</span><span class="tag ${item.alarm?'alarm':''}">${item.sensor}</span>
      <span class="tag">${item.sensor_knowledge}</span><span class="tag ${item.criticality}">${item.criticality}</span>
      <span class="tag">own production ${item.production_value}</span>
      <span class="tag">production at risk ${item.production_at_risk}</span></div>
      <div class="muted">${item.upstream_modules.length?`Receives from ${item.upstream_modules.join(', ')}`:'No upstream dependency'}${item.downstream_modules.length?` · Feeds ${item.downstream_modules.join(', ')}`:''}</div>
      ${item.inspection?`<div class="inspection">DIAGNOSTIC: ${item.inspection}</div>`:''}`;
    const actions=document.createElement('div'); actions.className='actions';
    const inspect=document.createElement('button'); inspect.textContent='Inspect';
    if(item.known_condition) info.innerHTML+=`<div class="inspection">KNOWN: ${item.known_condition}</div>`;
    if(item.last_service) info.innerHTML+=`<div class="muted">Last service: shift ${item.last_service.shift} · ${item.last_service.status}</div>`;
    inspect.disabled=!!item.inspection||!!item.known_condition||s.diagnostics_remaining===0||s.credits<s.inspection_cost;
    inspect.onclick=async()=>{try{el('message').textContent='Running diagnostic…';const r=await api('/api/inspect',{incident_id:item.id});
      el('message').textContent=`${item.module_id}: ${r.result}`;render(r.state)}catch(e){el('message').textContent=e.message}};
    const repair=document.createElement('button'); repair.textContent=selected.has(item.id)?'Repair selected':'Select repair';
    repair.className=selected.has(item.id)?'danger':'';
    repair.disabled=!selected.has(item.id)&&selected.size>=s.repair_capacity;
    repair.onclick=()=>{selected.has(item.id)?selected.delete(item.id):selected.add(item.id);render(state)};
    actions.append(inspect,repair); card.append(info,actions); el('modules').append(card);
  }
  const rows=Object.entries(s.maintenance_log).map(([cohort,x])=>
    `<div class="log-row"><span>${cohort}</span><span>${x.confirmed_leaks} leaks / ${x.confirmed_cases} cases</span></div>`).join('');
  el('log').innerHTML=rows||'<span class="muted">No confirmed cases yet</span>';
  el('topology').innerHTML=s.dependency_graph.length?s.dependency_graph.map(x=>
    `<div class="log-row"><span>${x.upstream}</span><span>→ ${x.downstream}</span></div>`).join(''):
    '<span class="muted">Modules are independent.</span>';
  el('shortfalls').innerHTML=s.production_shortfalls.length?s.production_shortfalls.map(x=>
    `<div class="log-row"><span>Shift ${x.shift}</span><span>−${x.production_loss}</span></div>`).join(''):
    '<span class="muted">No unexplained production loss.</span>';
  el('rules').textContent=`Inspection: ${s.inspection_cost} credits · Repair: ${s.repair_cost} credits + 1 seal kit`;
  if(s.last_report){const r=s.last_report;el('report').innerHTML=`<h2>Shift ${r.shift} report</h2>
    <div>Production: ${r.production} / ${r.maximum_production}</div><div>Recovered by repairs: ${r.production_recovered}</div><div>Maintenance cost: ${r.maintenance_cost}</div>
    <div class="${r.production_loss?'bad':'good'}">Production loss: ${r.production_loss}</div>
    <ul>${r.outcomes.map(x=>`<li>${x.module_id}: ${x.result}</li>`).join('')}
    ${r.production_event?`<li>${r.production_event}</li>`:''}
    ${!r.outcomes.length&&!r.production_event?'<li>No faults discovered.</li>':''}</ul>`}
  el('commit').disabled=s.status==='active'&&selected.size>s.repair_capacity;
  el('commit').textContent=s.status==='active'?`Commit ${selected.size} repair${selected.size===1?'':'s'}`:'Restart episode';
}
el('commit').onclick=async()=>{try{el('message').textContent='Resolving shift…';
  if(state.status!=='active'){const r=await api('/api/restart');selected.clear();render(r);return}
  const r=await api('/api/commit',{repair_ids:[...selected]});selected.clear();el('message').textContent=`Shift ${r.report.shift} resolved.`;render(r.state)
}catch(e){el('message').textContent=e.message}};
fetch('/api/state').then(r=>r.json()).then(render);
</script></body></html>"""


class GameApplication:
    def __init__(self, config: GameConfig):
        self.config = config
        self.session = GameSession(config)

    def restart(self) -> dict:
        self.session = GameSession(self.config)
        return self.session.public_state()


def make_handler(application: GameApplication):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=200):
            payload = json.dumps(value, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64 * 1024:
                raise ValueError("request body is too large")
            if not length:
                return {}
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def do_GET(self):
            if self.path == "/":
                payload = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/api/state":
                self._json(application.session.public_state())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                body = self._body()
                if self.path == "/api/inspect":
                    incident_id = body.get("incident_id")
                    if not isinstance(incident_id, str):
                        raise ValueError("incident_id must be a string")
                    self._json(application.session.inspect(incident_id))
                elif self.path == "/api/commit":
                    repair_ids = body.get("repair_ids", [])
                    if not isinstance(repair_ids, list) or not all(
                        isinstance(item, str) for item in repair_ids
                    ):
                        raise ValueError("repair_ids must be a list of strings")
                    self._json(application.session.commit(repair_ids))
                elif self.path == "/api/restart":
                    self._json(application.restart())
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)

        def log_message(self, format, *args):
            return

    return Handler


def serve_game(config: GameConfig | None = None, host="127.0.0.1", port=8765):
    config = config or GameConfig()
    application = GameApplication(config)
    server = ThreadingHTTPServer((host, port), make_handler(application))
    print(f"StationOps is running at http://{host}:{server.server_port}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
