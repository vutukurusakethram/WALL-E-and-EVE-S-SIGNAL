"""Local web control panel: `python run.py` (no arguments) opens it in your browser.
Standard library only. Everything runs on your own machine (127.0.0.1)."""
import importlib.util
import json
import mimetypes
import os
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CFG = ROOT / "configs"
LOCK = threading.Lock()          # one heavy job at a time (matplotlib is not thread-safe)
JOBS = {}
SERVER = None


# ------------------------------------------------------------------ helpers
def jdump(o):
    return json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)).encode()


def start_job(fn, *args):
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"status": "running", "msg": "Starting...", "result": None, "error": None}

    def work():
        with LOCK:
            try:
                JOBS[jid]["result"] = fn(JOBS[jid], *args)
                JOBS[jid]["status"] = "done"
            except Exception:
                JOBS[jid]["status"] = "error"
                JOBS[jid]["error"] = traceback.format_exc()
    threading.Thread(target=work, daemon=True).start()
    return jid


def list_configs():
    from .config import load_config
    out = []
    for f in sorted(CFG.glob("*")):
        if f.suffix.lower() not in (".yaml", ".yml", ".json"):
            continue
        try:
            c = load_config(f)
        except Exception as e:
            out.append({"file": f.name, "name": f.stem, "desc": f"(invalid config: {e})", "facts": []})
            continue
        m, s = c["map"], c["sensor"]
        walls = len(m["obstacles"]) or (f"~{m['obstacle_density']:.0%}" if m.get("obstacle_density") else 0)
        out.append({"file": f.name, "name": c["name"], "desc": c.get("description", ""),
                    "facts": [f"{m['width']}x{m['height']} grid", f"battery {c['budget']['battery']}",
                              f"{c['budget']['max_recharges']} recharges", f"noise {s['noise']}",
                              f"spoof {s['spoof_prob']:.0%}", f"decoy clues {c['clues']['decoy_prob']:.0%}", f"walls {walls}"]})
    return out


def folder_for(stem, policy, seed):
    name = stem
    if policy == "greedy":
        name += "_greedy"
    if seed not in (None, ""):
        name += f"_seed{int(seed)}"
    return name


def _prepare(p):
    from .config import load_config
    cfg = load_config(CFG / p["config"])
    stem = Path(p["config"]).stem
    seed = p.get("seed")
    if seed not in (None, ""):
        cfg["seed"] = int(seed)
        cfg["name"] = f"{cfg['name']} (seed {int(seed)})"
    policy = p.get("policy") or cfg["agent"]["policy"]
    folder = folder_for(stem, policy if policy == "greedy" else None, seed)
    return cfg, policy, folder


def mission_payload(res, folder, cfg, policy):
    out = OUT / folder
    rows = []
    for r in res["trace"]:
        a = r["action"]
        rows.append({"step": r["step"], "decision": r["decision"], "action": a["type"] + (f" -> {tuple(a['to'])}" if a["type"] == "move" else ""),
                     "battery": r["battery"], "pos": r["pos_after"], "conf": r["belief"]["confidence"],
                     "entropy": r["belief"]["entropy"], "reasoning": r["reasoning"]})
    t = int(time.time())
    anim = out / "animation.gif"
    return {"kind": "mission", "config": cfg["name"], "policy": policy, "folder": folder, "report": res["report"], "trace": rows,
            "ascii": (out / "ascii_final.txt").read_text(), "report_md": (out / "report.md").read_text(),
            "overview_url": f"/files/{folder}/mission_overview.png?t={t}",
            "animation_url": f"/files/{folder}/animation.gif?t={t}" if anim.exists() else None}


def _with_ctx(payload, p):
    payload["cfgfile"] = p["config"]
    payload["seed"] = p.get("seed")
    return payload


def run_one(job, p):
    from .runner import run_mission
    from .anim import make_animation
    cfg, policy, folder = _prepare(p)
    job["msg"] = "Running mission..."
    res = run_mission(cfg, str(OUT / folder), policy=policy)
    if p.get("animate"):
        job["msg"] = "Rendering heartbeat animation (10-30 seconds)..."
        make_animation(cfg, str(OUT / folder / "animation.gif"), policy=policy)
    return _with_ctx(mission_payload(res, folder, cfg, policy), p)


def run_animate(job, p):
    from .runner import run_mission
    from .anim import make_animation
    cfg, policy, folder = _prepare(p)
    job["msg"] = "Rendering heartbeat animation (10-30 seconds)..."
    res = run_mission(cfg, str(OUT / folder), policy=policy)
    make_animation(cfg, str(OUT / folder / "animation.gif"), policy=policy)
    return _with_ctx(mission_payload(res, folder, cfg, policy), p)


def run_all(job, p):
    from .runner import run_mission
    rows = []
    files = [c["file"] for c in list_configs() if "invalid" not in c["desc"]]
    for i, f in enumerate(files):
        job["msg"] = f"Running {f} ({i + 1}/{len(files)})..."
        cfg, policy, folder = _prepare({"config": f, "policy": p.get("policy"), "seed": p.get("seed")})
        r = run_mission(cfg, str(OUT / folder), policy=policy)["report"]
        rows.append({"config": cfg["name"], "folder": folder, "status": r["status"], "location": r["inferred_location"],
                     "confidence": r["confidence"], "steps": r["resources"]["steps"], "scans": r["resources"]["scans"],
                     "recharges": r["resources"]["recharges_used"], "error": r["evaluation_only"]["error_cells"],
                     "correct": r["evaluation_only"]["correct"]})
    return {"kind": "all", "rows": rows, "policy": p.get("policy") or "eig"}


def run_bench(job, p):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from benchmark import run_benchmark
    n = max(2, min(int(p.get("n", 20)), 100))

    def prog(i, n_):
        job["msg"] = f"Benchmark: mission {i + 1}/{n_} (both planners)..."
    return {"kind": "benchmark", "n": n, "summary": run_benchmark(n, str(OUT / "benchmark"), prog)}


def run_tests(job, p):
    spec = importlib.util.spec_from_file_location("test_core", ROOT / "tests" / "test_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = []
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        job["msg"] = f"Running {name}..."
        t0 = time.time()
        try:
            getattr(mod, name)()
            results.append({"name": name, "ok": True, "secs": round(time.time() - t0, 2), "err": ""})
        except Exception:
            results.append({"name": name, "ok": False, "secs": round(time.time() - t0, 2), "err": traceback.format_exc()})
    return {"kind": "tests", "results": results}


# ------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/configs":
            return self._send(200, jdump(list_configs()))
        if path.startswith("/api/job/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            return self._send(200 if job else 404, jdump(job or {"error": "unknown job"}))
        if path.startswith("/files/"):
            f = (OUT / path[len("/files/"):]).resolve()
            if OUT.resolve() in f.parents and f.is_file():
                return self._send(200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        routes = {"/api/run": run_one, "/api/animate": run_animate, "/api/runall": run_all,
                  "/api/benchmark": run_bench, "/api/tests": run_tests}
        if path in routes:
            return self._send(200, jdump({"job": start_job(routes[path], data)}))
        if path == "/api/quit":
            self._send(200, jdump({"ok": True}))
            threading.Thread(target=SERVER.shutdown, daemon=True).start()
            return
        return self._send(404, b"{}")


def serve(port=8765, open_browser=True):
    global SERVER
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for p in range(port, port + 20):
        try:
            SERVER = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("Could not find a free port.")
    url = f"http://127.0.0.1:{SERVER.server_address[1]}/"
    print(f"\n  WALL-E mission control running at {url}\n  (press Ctrl+C here, or use the Quit button on the page, to stop)\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        SERVER.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Server stopped.")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>WALL-E and EVE - Mission Control</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--g:#39ff14;--g2:#1fbf2a;--dg:#0b3d0b;--bg:#020a03;--panel:#04140a;--amb:#ffb000;--red:#ff3030}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--g);font:14px/1.45 Consolas,"Courier New",monospace}
header{display:flex;align-items:center;gap:14px;padding:14px 22px;border-bottom:1px solid var(--dg);background:#010601}
header h1{margin:0;font-size:20px;letter-spacing:2px}header small{color:var(--g2)}
.ping{width:26px;height:26px;position:relative}.ping i{position:absolute;inset:0;border:2px solid var(--g);border-radius:50%;animation:pg 2s infinite;opacity:0}
.ping i:nth-child(2){animation-delay:.6s}.ping i:nth-child(3){animation-delay:1.2s}.ping b{position:absolute;left:10px;top:10px;width:6px;height:6px;border-radius:50%;background:var(--g)}
@keyframes pg{0%{transform:scale(.2);opacity:.9}100%{transform:scale(1.6);opacity:0}}
.layout{display:grid;grid-template-columns:340px 1fr;gap:18px;padding:18px 22px}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
aside,main{min-width:0}
.box{border:1px solid var(--dg);background:var(--panel);padding:12px 14px;margin-bottom:14px;border-radius:4px}
h2{margin:0 0 10px;font-size:13px;letter-spacing:2px;color:var(--g2);font-weight:normal}
label{display:block;margin:6px 0;cursor:pointer}input[type=text],input[type=number],select{background:#000;color:var(--g);border:1px solid var(--g2);padding:5px 8px;font:inherit;width:110px}
button{background:#031a06;color:var(--g);border:1px solid var(--g);padding:7px 14px;font:inherit;letter-spacing:1px;cursor:pointer;border-radius:3px}
button:hover{background:var(--g);color:#000}button:disabled{opacity:.4;cursor:wait}
.card{border:1px solid var(--dg);padding:10px;margin-bottom:10px;border-radius:4px;background:#020c04}.card h3{margin:0 0 4px;font-size:14px;color:#9dffa0}
.card p{margin:0 0 8px;color:#6fd36f;font-size:12px}.facts{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}.facts span{font-size:11px;border:1px solid var(--dg);padding:1px 6px;color:var(--g2)}
.big{width:100%;margin-top:4px}
.tabs{display:flex;gap:6px;margin:12px 0 8px;flex-wrap:wrap}.tabs button.on{background:var(--g);color:#000}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid var(--dg);padding:4px 7px;text-align:left;vertical-align:top}th{color:var(--g2);background:#031007;position:sticky;top:0}
.scroll{max-height:460px;overflow:auto;border:1px solid var(--dg)}
pre{margin:0;white-space:pre;overflow:auto;background:#000;border:1px solid var(--dg);padding:10px;font-size:12px;line-height:1.25}
img{max-width:100%;border:1px solid var(--dg);display:block;background:#000}
.badge{display:inline-block;padding:3px 12px;border:2px solid;font-weight:bold;letter-spacing:2px}
.ok{color:var(--red);border-color:var(--red)}.mid{color:var(--amb);border-color:var(--amb)}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:12px 0}.stats div{border:1px solid var(--dg);padding:6px 9px}.stats b{display:block;font-size:17px;color:#b5ffb0}
.rec{border-left:3px solid var(--g);padding:6px 12px;margin:8px 0;background:#031007}
.loader{text-align:center;padding:70px 10px}.rings{width:120px;height:120px;margin:0 auto 18px;position:relative}.rings i{position:absolute;inset:0;border:3px solid var(--g);border-radius:50%;animation:pg 1.8s infinite;opacity:0}
.rings i:nth-child(2){animation-delay:.6s}.rings i:nth-child(3){animation-delay:1.2s}.rings b{position:absolute;left:54px;top:54px;width:12px;height:12px;border-radius:50%;background:var(--g)}
.muted{color:#4fa84f}.pass{color:var(--g)}.fail{color:var(--red)}
</style></head><body>
<header><div class="ping"><i></i><i></i><i></i><b></b></div><h1>WALL-E &amp; EVE'S SIGNAL</h1><small>// MISSION CONTROL</small></header>
<div class="layout">
<aside>
  <div class="box"><h2>SETTINGS</h2>
    <label><input type="radio" name="pol" value="eig" checked> Planner: <b>EIG</b> (information per battery)</label>
    <label><input type="radio" name="pol" value="greedy"> Planner: <b>Greedy</b> baseline</label>
    <label>Seed override: <input type="number" id="seed" placeholder="(config)"> <button id="dice" title="random seed">RND</button></label>
    <label><input type="checkbox" id="anim"> Also make heartbeat animation</label>
    <div class="muted" style="font-size:12px">A new seed changes where EVE and the clues are hidden.</div>
  </div>
  <div class="box"><h2>MISSIONS</h2><div id="cards" class="muted">loading...</div>
    <button class="big" id="runall">RUN ALL MISSIONS</button></div>
  <div class="box"><h2>EXTRAS</h2>
    <label>Benchmark size: <select id="bn"><option>10</option><option selected>20</option><option>40</option></select> missions</label>
    <button class="big" id="bench">COMPARE PLANNERS (BENCHMARK)</button>
    <button class="big" id="tests">RUN UNIT TESTS</button>
    <button class="big" id="quit" style="border-color:#7a2a2a;color:#ff7777">QUIT SERVER</button></div>
</aside>
<main id="main"><div class="box"><h2>READY</h2>Pick a mission on the left and press <b>RUN</b>. Results (figure, animation, trace, report) appear here.</div></main>
</div>
<script>
const $=s=>document.querySelector(s), main=$('#main'); let busy=false, cur=null;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pol=()=>document.querySelector('input[name=pol]:checked').value;
const settings=()=>({policy:pol(),seed:$('#seed').value,animate:$('#anim').checked});
function setBusy(b){busy=b;document.querySelectorAll('button').forEach(x=>{if(x.id!=='quit')x.disabled=b})}
async function api(url,data){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})});return r.json()}
async function job(url,data){
  if(busy)return; setBusy(true);
  main.innerHTML='<div class="box loader"><div class="rings"><i></i><i></i><i></i><b></b></div><div id="msg">Sending ping...</div></div>';
  try{const {job:id}=await api(url,data);
    while(true){await new Promise(r=>setTimeout(r,500));const j=await (await fetch('/api/job/'+id)).json();
      const m=$('#msg'); if(m)m.textContent=j.msg||'';
      if(j.status==='done'){render(j.result);break}
      if(j.status==='error'){main.innerHTML='<div class="box"><h2 class="fail">ERROR</h2><pre>'+esc(j.error)+'</pre></div>';break}}
  }catch(e){main.innerHTML='<div class="box fail">Lost connection to the server: '+esc(e)+'</div>'}
  setBusy(false)}
function render(r){ if(r.kind==='mission'){cur=r;renderMission(r,'overview')} else if(r.kind==='all')renderAll(r); else if(r.kind==='benchmark')renderBench(r); else renderTests(r)}
function renderMission(r,tab){
  const rp=r.report, ev=rp.evaluation_only, rs=rp.resources, conf=rp.status==='CONFIRMED';
  const tabs=[['overview','Overview'],['animation','Animation'],['trace','Step trace'],['ascii','ASCII map'],['report','Report']];
  let body='';
  if(tab==='overview')body='<img src="'+r.overview_url+'">';
  if(tab==='animation')body=r.animation_url?'<img src="'+r.animation_url+'">':'<div class="box">No animation yet for these settings. <button id="mk">GENERATE ANIMATION</button></div>';
  if(tab==='trace')body='<div class="scroll"><table><tr><th>#</th><th>decision</th><th>action</th><th>batt</th><th>pos</th><th>conf</th><th>reasoning</th></tr>'+
    r.trace.map(t=>'<tr><td>'+t.step+'</td><td>'+esc(t.decision)+'</td><td>'+esc(t.action)+'</td><td>'+t.battery+'</td><td>'+esc(t.pos)+'</td><td>'+Math.round(t.conf*100)+'%</td><td>'+esc(t.reasoning)+'</td></tr>').join('')+'</table></div>';
  if(tab==='ascii')body='<pre>'+esc(r.ascii)+'</pre>';
  if(tab==='report')body='<pre style="white-space:pre-wrap">'+esc(r.report_md)+'</pre>';
  main.innerHTML='<div class="box"><h2>'+esc(r.config)+' &nbsp;|&nbsp; planner: '+esc(r.policy)+'</h2>'+
   '<span class="badge '+(conf?'ok':'mid')+'">'+rp.status+'</span> &nbsp; at ('+rp.inferred_location.join(', ')+') &nbsp; confidence '+Math.round(rp.confidence*100)+'%'+
   '<div class="stats"><div>steps<b>'+rs.steps+'</b></div><div>scans<b>'+rs.scans+'</b></div><div>recharges<b>'+rs.recharges_used+'</b></div><div>battery left<b>'+rs.battery_left+'/'+rs.battery_full+'</b></div><div>EVE status<b style="font-size:13px">'+esc(rp.eve_status.split(' ')[0])+'</b></div><div>error vs truth<b>'+ev.error_cells+' cells</b></div></div>'+
   '<div class="rec"><b>Recommended next step:</b> '+esc(rp.recommended_next_step)+'</div>'+
   '<div class="tabs">'+tabs.map(t=>'<button class="'+(t[0]===tab?'on':'')+'" data-t="'+t[0]+'">'+t[1]+'</button>').join('')+'</div>'+body+'</div>';
  main.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>renderMission(r,b.dataset.t));
  const mk=$('#mk'); if(mk)mk.onclick=()=>job('/api/animate',{policy:r.policy,seed:r.seed,config:r.cfgfile,animate:true})}
function renderAll(r){main.innerHTML='<div class="box"><h2>ALL MISSIONS (planner: '+esc(r.policy)+')</h2><table><tr><th>mission</th><th>outcome</th><th>location</th><th>conf</th><th>steps</th><th>scans</th><th>recharges</th><th>error</th></tr>'+
  r.rows.map(x=>'<tr><td>'+esc(x.config)+'</td><td>'+x.status+'</td><td>'+esc(x.location.join(', '))+'</td><td>'+Math.round(x.confidence*100)+'%</td><td>'+x.steps+'</td><td>'+x.scans+'</td><td>'+x.recharges+'</td><td>'+x.error+'</td></tr>').join('')+'</table><p class="muted">Files saved in outputs/&lt;mission&gt;/. Press RUN on a single mission to see its figure and trace.</p></div>'}
function renderBench(r){const e=r.summary.eig,g=r.summary.greedy,p=x=>(x*100).toFixed(1)+'%';
  main.innerHTML='<div class="box"><h2>BENCHMARK: '+r.n+' RANDOM MISSIONS</h2><table><tr><th>metric</th><th>EIG planner</th><th>Greedy baseline</th></tr>'+
  '<tr><td>EVE confirmed</td><td>'+p(e.confirmed_rate)+'</td><td>'+p(g.confirmed_rate)+'</td></tr><tr><td>Answer within 1.5 cells</td><td>'+p(e.correct_rate)+'</td><td>'+p(g.correct_rate)+'</td></tr>'+
  '<tr><td>Mean error (cells)</td><td>'+e.mean_error_cells.toFixed(2)+'</td><td>'+g.mean_error_cells.toFixed(2)+'</td></tr><tr><td>Mean steps</td><td>'+e.mean_steps.toFixed(1)+'</td><td>'+g.mean_steps.toFixed(1)+'</td></tr>'+
  '<tr><td>Mean scans</td><td>'+e.mean_scans.toFixed(2)+'</td><td>'+g.mean_scans.toFixed(2)+'</td></tr><tr><td>Mean recharges</td><td>'+e.mean_recharges.toFixed(2)+'</td><td>'+g.mean_recharges.toFixed(2)+'</td></tr></table>'+
  '<p class="muted">Small samples are noisy: treat tiny differences in success rate as ties. Raw data: outputs/benchmark/benchmark.csv</p></div>'}
function renderTests(r){main.innerHTML='<div class="box"><h2>UNIT TESTS</h2><table><tr><th>test</th><th>result</th><th>secs</th></tr>'+
  r.results.map(t=>'<tr><td>'+esc(t.name)+'</td><td class="'+(t.ok?'pass':'fail')+'">'+(t.ok?'PASS':'FAIL')+'</td><td>'+t.secs+'</td></tr>'+(t.ok?'':'<tr><td colspan=3><pre>'+esc(t.err)+'</pre></td></tr>')).join('')+'</table></div>'}
fetch('/api/configs').then(r=>r.json()).then(cs=>{
  $('#cards').innerHTML=cs.map((c,i)=>'<div class="card"><h3>'+esc(c.name)+'</h3><p>'+esc(c.desc)+'</p><div class="facts">'+c.facts.map(f=>'<span>'+esc(f)+'</span>').join('')+'</div><button data-i="'+i+'">RUN</button></div>').join('');
  $('#cards').querySelectorAll('button').forEach(b=>b.onclick=async()=>{const c=cs[b.dataset.i]; await job('/api/run',{...settings(),config:c.file});});
});
$('#runall').onclick=()=>job('/api/runall',settings());
$('#bench').onclick=()=>job('/api/benchmark',{n:+$('#bn').value});
$('#tests').onclick=()=>job('/api/tests',{});
$('#dice').onclick=()=>{$('#seed').value=Math.floor(Math.random()*9000)+1};
$('#quit').onclick=async()=>{await api('/api/quit');main.innerHTML='<div class="box"><h2>SERVER STOPPED</h2>You can close this tab. Run run.py again to reopen it.</div>'};
</script></body></html>"""
