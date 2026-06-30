#!/usr/bin/env python
"""Sight training-run status stoplight. Local, stdlib-only (no torch/sb3 needed).

Run:   python tools\\sight_status_server.py
Open:  http://localhost:8765

Auto-detects the freshest runs\\phase_n\\c1_screen_s* and reports:
  GREEN  running, checkpoint advancing
  YELLOW starting (gen 0) or slow / no recent checkpoint
  RED    stopped (no Godot workers) or stalled (checkpoint frozen)
  DONE   100/100 generations complete
"""
import json, os, glob, time, subprocess
import http.server, socketserver

ROOT = r"C:\Projects\Sight"
RUNS_GLOB = os.path.join(ROOT, "runs", "phase_n", "c1_screen_s*")
PORT = 8765
GREEN_MAX_AGE = 420    # <7 min since last checkpoint => running
YELLOW_MAX_AGE = 900   # 7-15 min => suspect; >15 => stalled
TOTAL_GENS = 100


def godot_workers():
    try:
        out = subprocess.run(
            ["tasklist", "/fi", "imagename eq godot*", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout
        return sum(1 for ln in out.splitlines() if ln.strip().lower().startswith('"godot'))
    except Exception:
        return -1


def freshest_run():
    best, best_mt = None, -1
    for d in glob.glob(RUNS_GLOB):
        sp = os.path.join(d, "es_state.pkl")
        mt = os.path.getmtime(sp) if os.path.exists(sp) else -1
        if mt < 0:
            lp = os.path.join(d, "run.log")
            mt = os.path.getmtime(lp) if os.path.exists(lp) else -1
        if mt > best_mt:
            best, best_mt = d, mt
    return best


def compute_status():
    now = time.time()
    d = freshest_run()
    base = {"total": TOTAL_GENS, "now": time.strftime("%H:%M:%S"), "workers": godot_workers()}
    if not d:
        base.update(state="RED", label="No run found", seed=None, gen=0,
                    age_s=None, running_best=None, avg_gen_s=None, eta_min=None, dir=None)
        return base
    seed = os.path.basename(d).replace("c1_screen_s", "")
    hp = os.path.join(d, "es_history.json")
    sp = os.path.join(d, "es_state.pkl")
    lp = os.path.join(d, "run.log")
    sentinel = os.path.join(d, "c1_screen.sentinel")
    workers = base["workers"]
    gen, running_best, avg_gen_s = 0, None, None
    hist = None
    if os.path.exists(hp):
        try:
            hist = json.load(open(hp))
        except Exception:
            hist = None
    if hist:
        gen = hist[-1]["gen"] + 1
        running_best = hist[-1].get("running_best_mean_length")
        last_elapsed = hist[-1].get("elapsed_seconds")
        if last_elapsed and gen:
            avg_gen_s = last_elapsed / gen
    anchor = sp if os.path.exists(sp) else (lp if os.path.exists(lp) else None)
    age = (now - os.path.getmtime(anchor)) if anchor else None

    done = gen >= TOTAL_GENS
    if not done and os.path.exists(sentinel):
        try:
            if open(sentinel).read().strip().upper().startswith("EXIT 0") and gen >= TOTAL_GENS:
                done = True
        except Exception:
            pass

    if done:
        state, label = "DONE", "Complete (100/100)"
    elif workers == 0:
        state, label = "RED", "Stopped - no Godot workers (crashed)"
    elif gen == 0 and (age is None or age < GREEN_MAX_AGE):
        state, label = "YELLOW", "Starting (gen 0 in progress)"
    elif age is None:
        state, label = "YELLOW", "Unknown (no checkpoint yet)"
    elif age < GREEN_MAX_AGE:
        state, label = "GREEN", "Running"
    elif age < YELLOW_MAX_AGE:
        state, label = "YELLOW", "Slow / no recent checkpoint"
    else:
        state, label = "RED", "Stalled (checkpoint frozen)"

    eta_min = None
    if avg_gen_s and gen < TOTAL_GENS and state in ("GREEN", "YELLOW"):
        eta_min = round((TOTAL_GENS - gen) * avg_gen_s / 60)

    base.update(state=state, label=label, seed=seed, gen=gen,
                age_s=round(age) if age is not None else None,
                running_best=round(running_best, 1) if running_best is not None else None,
                avg_gen_s=round(avg_gen_s) if avg_gen_s else None,
                eta_min=eta_min, dir=d)
    return base


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sight - Run Status</title>
<style>
  :root{--bg:#0c0f14;--panel:#151a22;--edge:#222b38;--text:#e6edf3;--mut:#8a97a8;
        --green:#23d160;--amber:#ffc107;--red:#ff3b4e;--off:#1b212b;}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#11161f,#0c0f14);
       color:var(--text);font:15px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
       min-height:100vh;display:flex;align-items:center;justify-content:center;padding:28px}
  .wrap{display:flex;gap:34px;align-items:center;flex-wrap:wrap;justify-content:center}
  .light{background:var(--panel);border:1px solid var(--edge);border-radius:26px;
         padding:24px 22px;display:flex;flex-direction:column;gap:20px;box-shadow:0 20px 60px #0008}
  .lamp{width:96px;height:96px;border-radius:50%;background:var(--off);
        border:2px solid #0006;transition:.4s;position:relative}
  .lamp.on.red{background:var(--red);box-shadow:0 0 38px 6px #ff3b4ecc;animation:pulse 1.4s infinite}
  .lamp.on.amber{background:var(--amber);box-shadow:0 0 38px 6px #ffc107cc;animation:pulse 1.8s infinite}
  .lamp.on.green{background:var(--green);box-shadow:0 0 40px 8px #23d160cc}
  @keyframes pulse{0%,100%{filter:brightness(1)}50%{filter:brightness(1.35)}}
  .panel{min-width:300px;max-width:380px}
  .state{font-size:30px;font-weight:700;letter-spacing:.3px;margin:0 0 2px}
  .sub{color:var(--mut);margin:0 0 18px}
  .bar{height:10px;background:#0c1118;border:1px solid var(--edge);border-radius:6px;overflow:hidden;margin:6px 0 16px}
  .bar>i{display:block;height:100%;background:linear-gradient(90deg,#2c8cff,#23d160);width:0;transition:.6s}
  table{width:100%;border-collapse:collapse}
  td{padding:6px 0;border-bottom:1px solid #1a2129;vertical-align:top}
  td.k{color:var(--mut);width:46%}
  td.v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
  .foot{margin-top:14px;color:var(--mut);font-size:12px}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2c8cff;margin-right:6px;animation:blink 2s infinite}
  @keyframes blink{50%{opacity:.25}}
</style></head><body>
<div class="wrap">
  <div class="light">
    <div id="lr" class="lamp red"></div>
    <div id="la" class="lamp amber"></div>
    <div id="lg" class="lamp green"></div>
  </div>
  <div class="panel">
    <p id="state" class="state">...</p>
    <p id="label" class="sub">connecting</p>
    <div class="bar"><i id="prog"></i></div>
    <table>
      <tr><td class="k">Seed</td><td class="v" id="seed">-</td></tr>
      <tr><td class="k">Generation</td><td class="v" id="gen">-</td></tr>
      <tr><td class="k">Running best (train)</td><td class="v" id="rb">-</td></tr>
      <tr><td class="k">Last checkpoint</td><td class="v" id="age">-</td></tr>
      <tr><td class="k">Godot workers</td><td class="v" id="wk">-</td></tr>
      <tr><td class="k">Avg gen time</td><td class="v" id="ag">-</td></tr>
      <tr><td class="k">Est. time left</td><td class="v" id="eta">-</td></tr>
    </table>
    <p class="foot"><span class="dot"></span>auto-refresh 8s &middot; last checked <span id="now">-</span></p>
  </div>
</div>
<script>
const C={RED:'red',YELLOW:'amber',GREEN:'green',DONE:'green'};
function fmtAge(s){if(s==null)return '-';if(s<90)return s+'s ago';return Math.floor(s/60)+'m '+(s%60)+'s ago';}
function fmtEta(m){if(m==null)return '-';if(m<60)return m+' min';return Math.floor(m/60)+'h '+(m%60)+'m';}
async function tick(){
  try{
    const r=await fetch('/status.json',{cache:'no-store'});const d=await r.json();
    document.getElementById('lr').className='lamp red';
    document.getElementById('la').className='lamp amber';
    document.getElementById('lg').className='lamp green';
    const col=C[d.state]||'red';const map={red:'lr',amber:'la',green:'lg'};
    document.getElementById(map[col]).classList.add('on');
    const st=document.getElementById('state');st.textContent=d.state;
    st.style.color={red:'#ff3b4e',amber:'#ffc107',green:'#23d160'}[col];
    document.getElementById('label').textContent=d.label;
    document.getElementById('seed').textContent=d.seed==null?'-':('s'+d.seed);
    document.getElementById('gen').textContent=d.gen+' / '+d.total;
    document.getElementById('prog').style.width=(100*d.gen/d.total)+'%';
    document.getElementById('rb').textContent=d.running_best==null?'-':d.running_best;
    document.getElementById('age').textContent=fmtAge(d.age_s);
    document.getElementById('wk').textContent=d.workers<0?'?':d.workers;
    document.getElementById('ag').textContent=d.avg_gen_s?d.avg_gen_s+'s':'-';
    document.getElementById('eta').textContent=fmtEta(d.eta_min);
    document.getElementById('now').textContent=d.now;
  }catch(e){
    document.getElementById('label').textContent='server unreachable';
  }
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
        print("Sight status stoplight running:  http://localhost:%d" % PORT)
        print("(Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
