#!/usr/bin/env python
"""sd_fast shaped-sweep status stoplight. Local, stdlib-only (no torch/sb3).

Run:   python tools\\sd_fast_status_server.py
Open:  http://localhost:8767

Reuses the sight_status_server stoplight shell; the data adapter is repointed
at the none-vs-shaped 5M sweep in runs\\sd_fast. Reads each shaped run's SB3
log for live total_timesteps + fps, the per-seed _summary.json held-out means
as they land, and shaped_sweep.sentinel for CHAIN_DONE.

  GREEN  a shaped seed is training, log advancing
  YELLOW starting, or no recent log line
  RED    stalled / trainer gone before CHAIN_DONE
  DONE   sentinel CHAIN_DONE or all 5 shaped summaries present
"""
import json, os, time
import http.server

ROOT = r"C:\Projects\Sight"
SD = os.path.join(ROOT, "runs", "sd_fast")
SENTINEL = os.path.join(SD, "shaped_sweep.sentinel")
PORT = 8767
RUN_STEPS = 5_000_000
N_SEEDS = 5
GREEN_MAX_AGE = 120     # <2 min since last log line => training
YELLOW_MAX_AGE = 600    # 2-10 min => suspect; >10 => stalled/dead
NONE = [f"sd_fast_m21_s{s}_5M" for s in range(N_SEEDS)]
SHAPED = [f"sd_fast_m21sh_s{s}_5M" for s in range(N_SEEDS)]


def summary_mean(run):
    p = os.path.join(SD, run + "_summary.json")
    if not os.path.exists(p):
        return None
    try:
        return round(json.load(open(p))["eval"]["mean_len"], 1)
    except Exception:
        return None


def tail_text(path, nbytes=8000):
    with open(path, "rb") as f:
        f.seek(0, 2)
        sz = f.tell()
        f.seek(max(0, sz - nbytes))
        return f.read().decode("utf-8", "replace")


def last_num(text, key):
    val = None
    for ln in text.splitlines():
        if key in ln and ln.lstrip().startswith("|"):
            toks = [t.strip() for t in ln.split("|")]
            for t in toks:
                tt = t.replace(".", "", 1).replace("-", "", 1)
                if tt.isdigit():
                    val = float(t)
    return val


def compute_status():
    now = time.time()
    ledger = [{"seed": s,
               "none": summary_mean(NONE[s]),
               "shaped": summary_mean(SHAPED[s])} for s in range(N_SEEDS)]
    shaped_done = sum(1 for r in ledger if r["shaped"] is not None)
    last_shaped = None
    for r in ledger:
        if r["shaped"] is not None:
            last_shaped = r["shaped"]

    sentinel_done = False
    if os.path.exists(SENTINEL):
        try:
            sentinel_done = "CHAIN_DONE" in open(SENTINEL).read().upper()
        except Exception:
            sentinel_done = False
    done = sentinel_done or shaped_done >= N_SEEDS

    # current seed = smallest shaped seed without a summary
    cur_seed = next((s for s in range(N_SEEDS) if ledger[s]["shaped"] is None), None)
    cur_ts, fps, age = 0, None, None
    if cur_seed is not None:
        log = os.path.join(SD, SHAPED[cur_seed] + ".log")
        if os.path.exists(log):
            age = now - os.path.getmtime(log)
            try:
                txt = tail_text(log)
                ts = last_num(txt, "total_timesteps")
                fp = last_num(txt, "fps")
                cur_ts = int(ts) if ts else 0
                fps = int(fp) if fp else None
            except Exception:
                pass

    overall = (shaped_done * RUN_STEPS + min(cur_ts, RUN_STEPS))
    overall_frac = overall / (N_SEEDS * RUN_STEPS)

    if done:
        state, label = "DONE", f"Sweep complete ({shaped_done}/{N_SEEDS} shaped + all none)"
    elif cur_seed is None:
        state, label = "YELLOW", "Waiting"
    elif age is None:
        state, label = "YELLOW", f"Starting shaped s{cur_seed}"
    elif age < GREEN_MAX_AGE:
        state, label = "GREEN", f"Training shaped s{cur_seed}"
    elif age < YELLOW_MAX_AGE:
        state, label = "YELLOW", f"No recent log line (s{cur_seed})"
    else:
        state, label = "RED", f"Stalled or trainer gone (s{cur_seed})"

    eta_min = None
    if fps and not done:
        remaining = N_SEEDS * RUN_STEPS - overall
        eta_min = round(remaining / fps / 60)

    return {"state": state, "label": label, "now": time.strftime("%H:%M:%S"),
            "cur_seed": cur_seed, "cur_ts": cur_ts, "run_steps": RUN_STEPS,
            "shaped_done": shaped_done, "overall_frac": overall_frac,
            "age_s": round(age) if age is not None else None,
            "fps": fps, "eta_min": eta_min, "last_shaped": last_shaped,
            "ledger": ledger}


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sight - sd_fast shaped sweep</title>
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
  .panel{min-width:320px;max-width:400px}
  .state{font-size:30px;font-weight:700;letter-spacing:.3px;margin:0 0 2px}
  .sub{color:var(--mut);margin:0 0 18px}
  .bar{height:10px;background:#0c1118;border:1px solid var(--edge);border-radius:6px;overflow:hidden;margin:6px 0 16px}
  .bar>i{display:block;height:100%;background:linear-gradient(90deg,#2c8cff,#23d160);width:0;transition:.6s}
  table{width:100%;border-collapse:collapse}
  td{padding:6px 0;border-bottom:1px solid #1a2129;vertical-align:top}
  td.k{color:var(--mut);width:52%}
  td.v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
  .led{margin-top:18px}
  .led h4{margin:0 0 6px;color:var(--mut);font-weight:600;font-size:12px;letter-spacing:.4px;text-transform:uppercase}
  .led table{font-size:13px}
  .led td,.led th{padding:4px 0;text-align:right;font-variant-numeric:tabular-nums}
  .led th{color:var(--mut);font-weight:600;border-bottom:1px solid #1a2129}
  .led td:first-child,.led th:first-child{text-align:left;color:var(--mut)}
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
      <tr><td class="k">Current seed</td><td class="v" id="cs">-</td></tr>
      <tr><td class="k">Seed timesteps</td><td class="v" id="ts">-</td></tr>
      <tr><td class="k">Shaped complete</td><td class="v" id="sd">-</td></tr>
      <tr><td class="k">Last log line</td><td class="v" id="age">-</td></tr>
      <tr><td class="k">Throughput</td><td class="v" id="fps">-</td></tr>
      <tr><td class="k">Est. time to CHAIN_DONE</td><td class="v" id="eta">-</td></tr>
    </table>
    <div class="led">
      <h4>held-out mean (seeds 5000-5029)</h4>
      <table>
        <thead><tr><th>seed</th><th>none</th><th>shaped</th></tr></thead>
        <tbody id="ledbody"></tbody>
      </table>
    </div>
    <p class="foot"><span class="dot"></span>auto-refresh 8s &middot; last checked <span id="now">-</span></p>
  </div>
</div>
<script>
const C={RED:'red',YELLOW:'amber',GREEN:'green',DONE:'green'};
function fmtAge(s){if(s==null)return '-';if(s<90)return s+'s ago';return Math.floor(s/60)+'m '+(s%60)+'s ago';}
function fmtEta(m){if(m==null)return '-';if(m<60)return m+' min';return Math.floor(m/60)+'h '+(m%60)+'m';}
function fmtM(n){return (n/1e6).toFixed(2)+'M';}
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
    document.getElementById('cs').textContent=d.cur_seed==null?'-':('s'+d.cur_seed);
    document.getElementById('ts').textContent=fmtM(d.cur_ts)+' / '+fmtM(d.run_steps);
    document.getElementById('sd').textContent=d.shaped_done+' / 5';
    document.getElementById('prog').style.width=(100*d.overall_frac)+'%';
    document.getElementById('age').textContent=fmtAge(d.age_s);
    document.getElementById('fps').textContent=d.fps?d.fps+' steps/s':'-';
    document.getElementById('eta').textContent=fmtEta(d.eta_min);
    let rows='';
    for(const r of d.ledger){
      rows+='<tr><td>s'+r.seed+'</td><td>'+(r.none==null?'-':r.none)
           +'</td><td>'+(r.shaped==null?'<span style=\"color:#8a97a8\">pending</span>':r.shaped)+'</td></tr>';
    }
    document.getElementById('ledbody').innerHTML=rows;
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
        print("sd_fast sweep stoplight running:  http://localhost:%d" % PORT)
        print("(Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
