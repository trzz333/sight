#!/usr/bin/env python
"""Sight Phase-N / C3 screen status server. Local, stdlib-only.

Run (windowless):  python tools\\launch_c3_status_detached.py
Open:              http://localhost:8766

Reports the whole pre-registered C3 seed screen without touching torch/sb3:
  - which seed is training and its iter progress (x / TARGET_ITERS)
  - dev-best and latest dev action mix (collapse check)
  - ETA for the active seed
  - per-seed held-out ledger (mean / pass) as the orchestrator evals them
  - final screen verdict when c3_screen_all.sentinel lands

State lamp:
  GREEN  training, history advancing
  AMBER  starting / evaluating on held-out / staging next seed / slow
  RED    stalled (history frozen and no workers)
  DONE   screen complete (verdict written)
"""
import glob
import json
import os
import subprocess
import time
import http.server

ROOT = r"C:\Projects\Sight"
PHASE = os.path.join(ROOT, "runs", "phase_n")
PORT = 8766
TARGET_ITERS = 60
GREEN_MAX_AGE = 420    # <7 min since last iter => running
YELLOW_MAX_AGE = 900   # 7-15 min => suspect; >15 & no workers => stalled
NEAR_MISS = 880.0
BAR = 930.27


def godot_workers():
    try:
        out = subprocess.run(
            ["tasklist", "/fi", "imagename eq godot*", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return sum(1 for ln in out.splitlines()
                   if ln.strip().lower().startswith('"godot'))
    except Exception:
        return -1


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def seed_status(seed):
    d = os.path.join(PHASE, "c3_screen_s%d" % seed)
    hp = os.path.join(d, "c3_history.json")
    rp = os.path.join(d, "c3_report.json")
    s = {"seed": seed, "exists": os.path.isdir(d), "iters": 0,
         "target": TARGET_ITERS, "best_dev": None, "last_dev": None,
         "last_dev_frac": None, "last_max_frac": None, "dev_pass": None,
         "hist_age_s": None, "report": os.path.exists(rp),
         "held_out": None, "phase": "not started"}
    if not s["exists"]:
        return s
    hist = _load_json(hp)
    if hist:
        s["iters"] = hist[-1]["iter"] + 1
        s["best_dev"] = hist[-1].get("running_best_dev")
        s["last_dev"] = hist[-1].get("dev_mean_length")
        s["last_dev_frac"] = hist[-1].get("dev_action_fractions")
        s["last_max_frac"] = hist[-1].get("dev_max_frac")
        s["dev_pass"] = hist[-1].get("dev_gate_pass")
        if os.path.exists(hp):
            s["hist_age_s"] = round(time.time() - os.path.getmtime(hp))
    # held-out gate results, once the orchestrator has run them
    ho = {}
    best = None
    any_pass = False
    for kind in ("actor", "final"):
        summ = _load_json(os.path.join(d, "eval_%s" % kind, "c1_eval_summary.json"))
        if summ:
            m = summ.get("mean_episode_length")
            ho[kind] = {"mean": round(m, 1) if m is not None else None,
                        "pass": summ.get("gate_pass"),
                        "max_frac": round(summ.get("max_action_fraction", 0), 3)}
            if m is not None and (best is None or m > best):
                best = m
            any_pass = any_pass or bool(summ.get("gate_pass"))
    if ho:
        s["held_out"] = {"detail": ho,
                         "best_mean": round(best, 1) if best is not None else None,
                         "any_pass": any_pass}
    # phase label
    if s["report"]:
        if s["held_out"] and len(s["held_out"]["detail"]) == 2:
            s["phase"] = "evaled"
        else:
            s["phase"] = "training done, gating held-out"
    elif s["iters"] > 0:
        s["phase"] = "training %d/%d" % (s["iters"], TARGET_ITERS)
    else:
        s["phase"] = "spinning up"
    return s


def compute_status():
    now_str = time.strftime("%H:%M:%S")
    workers = godot_workers()
    verdict = _load_json(os.path.join(PHASE, "c3_screen_verdict.json"))
    sentinel = os.path.join(PHASE, "c3_screen_all.sentinel")
    seeds = [seed_status(i) for i in range(3)]

    # active seed = freshest that is training (has iters, no report)
    active = None
    for s in seeds:
        if s["exists"] and s["iters"] > 0 and not s["report"]:
            if active is None or (s["hist_age_s"] or 1e9) < (active["hist_age_s"] or 1e9):
                active = s

    # per-active-seed ETA + avg iter time
    avg_iter_s = None
    eta_seed_min = None
    if active and active["iters"] > 0:
        hist = _load_json(os.path.join(PHASE, "c3_screen_s%d" % active["seed"], "c3_history.json"))
        if hist:
            last_elapsed = hist[-1].get("elapsed_seconds")
            if last_elapsed:
                avg_iter_s = last_elapsed / active["iters"]
                eta_seed_min = round((TARGET_ITERS - active["iters"]) * avg_iter_s / 60)

    base = {"now": now_str, "workers": workers, "target": TARGET_ITERS,
            "bar": BAR, "near_miss": NEAR_MISS, "seeds": seeds,
            "active_seed": active["seed"] if active else None,
            "avg_iter_s": round(avg_iter_s) if avg_iter_s else None,
            "eta_seed_min": eta_seed_min, "verdict": None,
            "screen_result": None}

    # DONE?
    if os.path.exists(sentinel) or (verdict and verdict.get("screen_result")):
        base["state"] = "DONE"
        base["screen_result"] = (verdict or {}).get("screen_result", "complete")
        base["label"] = base["screen_result"]
        base["verdict"] = verdict
        base["age_s"] = None
        # crude screen ETA = 0
        base["eta_screen_min"] = 0
        return base

    # staging awareness: how many seeds are expected to still run.
    # Before seed-0 held-out verdict: unknown (1 or 3). After: derive.
    staged_known = seeds[0]["held_out"] is not None
    remaining_seeds_after_active = 0
    if staged_known:
        s0 = seeds[0]["held_out"]
        will_stage = s0["any_pass"] or (s0["best_mean"] is not None and s0["best_mean"] >= NEAR_MISS)
        if will_stage:
            # seeds 1,2 run if not yet evaled
            for i in (1, 2):
                if seeds[i]["held_out"] is None and not (active and active["seed"] == i):
                    remaining_seeds_after_active += 1

    # screen ETA (best-effort): active seed remaining + full future seeds
    eta_screen_min = None
    if eta_seed_min is not None:
        extra = remaining_seeds_after_active * TARGET_ITERS * (avg_iter_s or 0) / 60
        eta_screen_min = round(eta_seed_min + extra)
    base["eta_screen_min"] = eta_screen_min
    base["staging_known"] = staged_known

    age = active["hist_age_s"] if active else None
    base["age_s"] = age

    # state machine
    if active is None:
        # no seed actively training: either between seeds (gating/staging) or nothing up
        any_report_pending_eval = any(
            s["report"] and (s["held_out"] is None or len(s["held_out"]["detail"]) < 2)
            for s in seeds)
        if any_report_pending_eval or (workers in (0, 1)):
            base["state"], base["label"] = "AMBER", "Gating held-out / staging next seed"
        else:
            base["state"], base["label"] = "AMBER", "Idle (waiting on orchestrator)"
        return base

    if age is not None and age < GREEN_MAX_AGE and workers >= 1:
        base["state"], base["label"] = "GREEN", "Training seed-%d" % active["seed"]
    elif age is not None and age < GREEN_MAX_AGE:
        base["state"], base["label"] = "AMBER", "Training seed-%d (workers=?)" % active["seed"]
    elif age is not None and age < YELLOW_MAX_AGE:
        base["state"], base["label"] = "AMBER", "Slow / no recent iter (seed-%d)" % active["seed"]
    elif workers == 0:
        base["state"], base["label"] = "RED", "Stalled - no workers, iter frozen"
    else:
        base["state"], base["label"] = "AMBER", "Between iters / long episodes"
    return base


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sight - C3 Screen</title>
<style>
  :root{--bg:#0c0f14;--panel:#151a22;--edge:#222b38;--text:#e6edf3;--mut:#8a97a8;
        --green:#23d160;--amber:#ffc107;--red:#ff3b4e;--blue:#2c8cff;--off:#1b212b;}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#11161f,#0c0f14);
       color:var(--text);font:15px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
       min-height:100vh;display:flex;align-items:center;justify-content:center;padding:26px}
  .wrap{width:100%;max-width:760px;display:flex;flex-direction:column;gap:18px}
  .card{background:var(--panel);border:1px solid var(--edge);border-radius:20px;
        padding:22px 24px;box-shadow:0 20px 60px #0007}
  .top{display:flex;gap:22px;align-items:center}
  .lamp{width:70px;height:70px;border-radius:50%;background:var(--off);border:2px solid #0006;flex:0 0 auto}
  .lamp.red{background:var(--red);box-shadow:0 0 30px 5px #ff3b4ecc;animation:pulse 1.4s infinite}
  .lamp.amber{background:var(--amber);box-shadow:0 0 30px 5px #ffc107cc;animation:pulse 1.8s infinite}
  .lamp.green{background:var(--green);box-shadow:0 0 34px 7px #23d160cc}
  .lamp.blue{background:var(--blue);box-shadow:0 0 34px 7px #2c8cffcc}
  @keyframes pulse{0%,100%{filter:brightness(1)}50%{filter:brightness(1.4)}}
  .state{font-size:28px;font-weight:750;margin:0}
  .label{color:var(--mut);margin:2px 0 0}
  .eta{margin-left:auto;text-align:right}
  .eta b{font-size:30px;font-variant-numeric:tabular-nums}
  .eta span{display:block;color:var(--mut);font-size:12px}
  .bar{height:12px;background:#0c1118;border:1px solid var(--edge);border-radius:7px;overflow:hidden;margin:16px 0 6px}
  .bar>i{display:block;height:100%;background:linear-gradient(90deg,#2c8cff,#23d160);width:0;transition:.6s}
  .prog-cap{display:flex;justify-content:space-between;color:var(--mut);font-size:12px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 26px;margin-top:6px}
  .row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1a2129}
  .row .k{color:var(--mut)}.row .v{font-variant-numeric:tabular-nums;font-weight:600}
  h3{margin:2px 0 10px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{padding:8px 6px;text-align:left;border-bottom:1px solid #1a2129;font-variant-numeric:tabular-nums}
  th{color:var(--mut);font-weight:600;font-size:12px}
  .pill{padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700}
  .pill.pass{background:#123d22;color:#3ee07a}.pill.fail{background:#3d1620;color:#ff7089}
  .pill.run{background:#12283d;color:#5db1ff}.pill.wait{background:#2a2f38;color:#9aa7b8}
  .banner{padding:12px 16px;border-radius:12px;font-weight:700}
  .banner.pass{background:#123d22;color:#3ee07a}.banner.fail{background:#3d1620;color:#ff9aab}
  .foot{color:var(--mut);font-size:12px;text-align:center}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2c8cff;margin-right:6px;animation:blink 2s infinite}
  @keyframes blink{50%{opacity:.25}}
</style></head><body>
<div class="wrap">
  <div class="card">
    <div class="top">
      <div id="lamp" class="lamp"></div>
      <div>
        <p id="state" class="state">...</p>
        <p id="label" class="label">connecting</p>
      </div>
      <div class="eta"><b id="eta">-</b><span>est. time left (active seed)</span></div>
    </div>
    <div class="bar"><i id="prog"></i></div>
    <div class="prog-cap"><span id="pcap">seed -</span><span id="pcap2">- / 60 iters</span></div>
    <div class="grid" style="margin-top:14px">
      <div class="row"><span class="k">Dev best (argmax)</span><span class="v" id="devbest">-</span></div>
      <div class="row"><span class="k">Latest dev mean</span><span class="v" id="devlast">-</span></div>
      <div class="row"><span class="k">Dev action mix L/S/R</span><span class="v" id="mix">-</span></div>
      <div class="row"><span class="k">Godot workers</span><span class="v" id="wk">-</span></div>
      <div class="row"><span class="k">Avg iter time</span><span class="v" id="avg">-</span></div>
      <div class="row"><span class="k">Last iter</span><span class="v" id="age">-</span></div>
      <div class="row"><span class="k">Screen ETA (best-effort)</span><span class="v" id="etascreen">-</span></div>
      <div class="row"><span class="k">Bar / near-miss</span><span class="v" id="bars">-</span></div>
    </div>
  </div>
  <div class="card">
    <h3>Pre-registered seed screen (held-out 1000-1009)</h3>
    <table><thead><tr><th>Seed</th><th>Phase</th><th>Iters</th><th>Dev best</th><th>Held-out best</th><th>Gate</th></tr></thead>
    <tbody id="ledger"></tbody></table>
    <div id="banner" style="margin-top:14px"></div>
  </div>
  <p class="foot"><span class="dot"></span>auto-refresh 8s &middot; localhost:8766 &middot; checked <span id="now">-</span></p>
</div>
<script>
const LC={GREEN:'green',AMBER:'amber',RED:'red',DONE:'blue'};
function eta(m){if(m==null)return '-';if(m<=0)return 'done';if(m<60)return m+' min';return Math.floor(m/60)+'h '+(m%60)+'m';}
function age(s){if(s==null)return '-';if(s<90)return s+'s ago';return Math.floor(s/60)+'m '+(s%60)+'s ago';}
function mix(f){if(!f)return '-';const r=x=>(100*x).toFixed(0);return r(f.left)+'/'+r(f.stay)+'/'+r(f.right)+'%';}
function gatePill(s){
  if(s.held_out&&s.held_out.best_mean!=null){
    const p=s.held_out.any_pass;return '<span class="pill '+(p?'pass':'fail')+'">'+(p?'PASS':'FAIL')+'</span>';}
  if(s.report)return '<span class="pill wait">gating</span>';
  if(s.iters>0)return '<span class="pill run">training</span>';
  return '<span class="pill wait">pending</span>';
}
async function tick(){
 try{
  const d=await(await fetch('/status.json',{cache:'no-store'})).json();
  document.getElementById('lamp').className='lamp '+(LC[d.state]||'red');
  const st=document.getElementById('state');st.textContent=d.state;
  st.style.color={green:'#23d160',amber:'#ffc107',red:'#ff3b4e',blue:'#2c8cff'}[LC[d.state]||'red'];
  document.getElementById('label').textContent=d.label||'';
  document.getElementById('eta').textContent=eta(d.eta_seed_min);
  const as=d.active_seed;
  let cur=d.seeds.find(s=>s.seed===as)||null;
  document.getElementById('pcap').textContent=as==null?'seed -':('seed '+as);
  const it=cur?cur.iters:0;
  document.getElementById('pcap2').textContent=it+' / '+d.target+' iters';
  document.getElementById('prog').style.width=(100*it/d.target)+'%';
  document.getElementById('devbest').textContent=cur&&cur.best_dev!=null?cur.best_dev:'-';
  document.getElementById('devlast').textContent=cur&&cur.last_dev!=null?cur.last_dev:'-';
  document.getElementById('mix').textContent=cur?mix(cur.last_dev_frac):'-';
  document.getElementById('wk').textContent=d.workers<0?'?':d.workers;
  document.getElementById('avg').textContent=d.avg_iter_s?d.avg_iter_s+'s':'-';
  document.getElementById('age').textContent=age(d.age_s);
  document.getElementById('etascreen').textContent=eta(d.eta_screen_min);
  document.getElementById('bars').textContent=d.bar+' / '+d.near_miss;
  document.getElementById('now').textContent=d.now;
  // ledger
  const tb=document.getElementById('ledger');tb.innerHTML='';
  d.seeds.forEach(s=>{
    const ho=s.held_out&&s.held_out.best_mean!=null?s.held_out.best_mean:'-';
    const db=s.best_dev!=null?s.best_dev:'-';
    const tr=document.createElement('tr');
    tr.innerHTML='<td>s'+s.seed+'</td><td>'+(s.exists?s.phase:'not started')+'</td>'+
      '<td>'+(s.iters||0)+'/'+d.target+'</td><td>'+db+'</td><td>'+ho+'</td><td>'+gatePill(s)+'</td>';
    tb.appendChild(tr);
  });
  const b=document.getElementById('banner');
  if(d.state==='DONE'&&d.screen_result){
    const pass=/PASS/.test(d.screen_result);
    b.className='banner '+(pass?'pass':'fail');
    b.textContent='Screen result: '+d.screen_result;
  }else{b.className='';b.textContent='';}
 }catch(e){document.getElementById('label').textContent='server unreachable';}
}
tick();setInterval(tick,8000);
</script>
</body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/status"):
            body = json.dumps(compute_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as srv:
        print("Sight C3 screen status running:  http://localhost:%d" % PORT)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
