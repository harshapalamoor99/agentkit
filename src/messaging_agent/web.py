"""FastAPI web UI for testing the context-aware messaging agent.

Run:
    PYTHONPATH=src python -m messaging_agent.web
    # or
    PYTHONPATH=src uvicorn messaging_agent.web:api --reload --port 8000

Then open http://127.0.0.1:8000 — paste/edit a JSONL record and click "Run agent"
to see the decision, the produced message, every acceptance-criterion pass/fail,
and the compliance-repair audit warnings.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .graph import app as agent_app
from .llm_client import LLMClient

api = FastAPI(title="Context-Aware Messaging Agent")


@api.on_event("startup")
async def _warmup_llm() -> None:
    """Prime the LLM gateway connection so the first request isn't cold, then keep it
    warm for the whole demo session (KEEPWARM_ENABLED) so latency stays at the warm
    ~0.8s even across idle gaps between clicks."""
    from . import config
    from .nodes import llm as _llmnode
    if _llmnode._client.available:
        await _llmnode._client.warmup()
        if config.KEEPWARM_ENABLED:
            _llmnode._client.start_keepwarm()


@api.on_event("shutdown")
async def _stop_keepwarm() -> None:
    from .nodes import llm as _llmnode
    await _llmnode._client.stop_keepwarm()

# Load the bundled sample dataset (used as few-shot context + UI presets).
_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "evals" / "sample_8613.jsonl"
_ADVERSARIAL_PATH = Path(__file__).resolve().parents[2] / "data" / "evals" / "adversarial.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_raw_line": line})
    return out


SAMPLE = _load_jsonl(_SAMPLE_PATH)
ADVERSARIAL = _load_jsonl(_ADVERSARIAL_PATH)
DATASET = SAMPLE + ADVERSARIAL


class RunRequest(BaseModel):
    record: str  # raw JSON text for one record


class BatchRequest(BaseModel):
    content: str  # raw JSONL text (one record per line)


@api.get("/api/provider")
def provider():
    p = LLMClient().provider
    return {"provider": p or "NONE — LLM-only: records abort without an API key",
            "llm_active": bool(p)}


@api.get("/api/observability")
def observability_status():
    """Tracing + LangSmith config so a UI can show whether spans/cost are flowing."""
    from .observability import status as otel_status
    from .prod.tracing import status as langsmith_status
    from .evals.langsmith_eval import status as langsmith_eval_status

    return {"otel": otel_status(), "langsmith": langsmith_status(),
            "langsmith_eval": langsmith_eval_status()}


class WebhookRequest(BaseModel):
    prospect_id: str
    event_type: str            # tour_booked | link_clicked | stop | ...
    context: dict | None = None
    record: dict | None = None
    produced_output: dict | None = None


@api.post("/api/webhook")
def webhook(req: WebhookRequest):
    """Real-time consumer event ingestion (AC-6 state cancellation + AC-15 telemetry).

    Transitions the prospect's workflow state and cancels/rewrites obsolete scheduled
    messages, then emits a closed-loop telemetry record pairing input features + copy
    + outcome for the next training cycle.
    """
    from . import telemetry, workflow

    wf = workflow.engine.handle_event(req.prospect_id, req.event_type, req.context or {})
    tele = None
    if req.record is not None and req.produced_output is not None:
        tele = telemetry.store.emit_outcome(
            record=req.record, produced_output=req.produced_output,
            outcome=req.event_type, metadata=req.context or {})
    return {"workflow": wf, "telemetry_emitted": tele is not None,
            "telemetry": tele}


@api.get("/api/presets")
def presets():
    def label(r, kind):
        return {"task_id": r.get("task_id", "(raw line)"), "kind": kind,
                "json": json.dumps(r, ensure_ascii=False, indent=2)}
    return {"presets": [label(r, "sample") for r in SAMPLE]
            + [label(r, "adversarial") for r in ADVERSARIAL]}


@api.get("/api/agents")
def agents():
    """Discover the registered domain agents (multi-agent surface)."""
    from .domain import available_domains
    from .multiagent import AgentService
    out = []
    for name in available_domains():
        tool = AgentService(name).as_tool()
        out.append({"domain": name, "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"]})
    return {"agents": out}


class DispatchRequest(BaseModel):
    record: dict                      # the input record (may carry a "domain" field)
    domain: str | None = None         # explicit domain override
    dataset: list[dict] | None = None


@api.post("/api/dispatch")
async def dispatch(req: DispatchRequest):
    """Route a record to the right domain agent and return its decision."""
    from .multiagent import AgentRouter
    record = dict(req.record)
    if req.domain and "domain" not in record:
        record["domain"] = req.domain
    router = AgentRouter()
    try:
        return JSONResponse(await router.dispatch(record, dataset=req.dataset))
    except Exception as exc:
        return JSONResponse({"error": str(exc), "type": type(exc).__name__}, status_code=500)


@api.post("/api/run")
async def run(req: RunRequest):
    text = req.record.strip()
    try:
        record = json.loads(text)
        init = {"record": record, "raw_line": json.dumps(record, ensure_ascii=False),
                "dataset": DATASET, "task_id": record.get("task_id", "ui-test")}
    except json.JSONDecodeError as e:
        # Feed the malformed line straight in so AC-17 handling is exercised.
        init = {"raw_line": text, "dataset": DATASET}
        _ = e
    try:
        result = await agent_app.ainvoke(init)
        return JSONResponse(result.get("final_output",
                            {"error": "no output", "state_keys": list(result.keys())}))
    except Exception as exc:  # never let the UI crash; surface the error
        return JSONResponse({"error": str(exc), "type": type(exc).__name__}, status_code=500)


@api.post("/api/batch")
async def batch(req: BatchRequest):
    lines = [ln.strip() for ln in req.content.splitlines() if ln.strip()]

    # Run records concurrently with bounded concurrency (mirrors prod/runner.py). Each
    # record is one independent LLM round-trip, so a batch of N is dominated by N
    # sequential calls unless we overlap them; the semaphore caps in-flight calls so we
    # don't overrun the gateway. Order is preserved by gathering by index.
    concurrency = max(1, int(os.getenv("WEB_BATCH_CONCURRENCY", "16")))
    sem = asyncio.Semaphore(concurrency)

    async def _run_line(i: int, ln: str) -> dict:
        try:
            record = json.loads(ln)
            init = {"record": record, "raw_line": json.dumps(record, ensure_ascii=False),
                    "dataset": DATASET, "task_id": record.get("task_id", f"line-{i}")}
        except json.JSONDecodeError:
            init = {"raw_line": ln, "dataset": DATASET}
        async with sem:
            try:
                result = await agent_app.ainvoke(init)
                return {"line": i, "out": result.get("final_output", {})}
            except Exception as exc:
                return {"line": i, "error": str(exc)}

    settled = await asyncio.gather(*(_run_line(i, ln) for i, ln in enumerate(lines, 1)))

    rows = []
    outputs = []
    agg = {"records": 0, "send": 0, "no_send": 0, "ac_pass": 0, "ac_total": 0,
           "critical_fail_records": 0, "errors": 0, "max_latency_ms": 0}
    for res in settled:
        i = res["line"]
        if "error" in res:
            agg["errors"] += 1
            rows.append({"line": i, "task_id": f"line-{i}", "error": res["error"]})
            outputs.append({"line": i, "error": res["error"]})
            continue
        out = res["out"]
        outputs.append(out)
        ev = out.get("evaluation", {})
        crit = ev.get("critical_fails", []) or []
        p, t = ev.get("passed", 0), ev.get("total", 0)
        lat = out.get("latency_ms", 0)
        agg["records"] += 1
        agg["send" if out.get("should_send") else "no_send"] += 1
        agg["ac_pass"] += p
        agg["ac_total"] += t
        agg["max_latency_ms"] = max(agg["max_latency_ms"], lat)
        if crit:
            agg["critical_fail_records"] += 1
        rows.append({
            "line": i,
            "task_id": out.get("task_id", f"line-{i}"),
            "should_send": out.get("should_send"),
            "channel": (out.get("next_message") or {}).get("channel"),
            "ac_score": ev.get("score", "?"),
            "critical_fails": [c["id"] for c in crit],
            "warnings": len(out.get("warnings", [])),
            "latency_ms": lat,
            "abort_reason": out.get("abort_reason"),
        })
    agg["all_pass"] = agg["critical_fail_records"] == 0 and agg["errors"] == 0
    return {"summary": agg, "rows": rows, "outputs": outputs}


@api.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Messaging Agent — Test Console</title>
<style>
  :root { --bg:#0f1419; --panel:#1a2129; --border:#2c3742; --txt:#d7e0ea;
          --muted:#8b97a4; --green:#3fb950; --red:#f85149; --amber:#d29922;
          --blue:#58a6ff; --crit:#f85149; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--txt); }
  header { padding:14px 20px; border-bottom:1px solid var(--border);
           display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; }
  #provider { font-size:12px; color:var(--muted); }
  .badge { padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .badge.live { background:#1f3a26; color:var(--green); }
  .badge.fallback { background:#3a2f1f; color:var(--amber); }
  main { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:16px;
         align-items:start; }
  @media (max-width:900px){ main{ grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--border);
          border-radius:10px; padding:14px; }
  label { font-size:12px; color:var(--muted); display:block; margin-bottom:6px; }
  select, textarea, button { font-family:inherit; }
  select { background:var(--bg); color:var(--txt); border:1px solid var(--border);
           border-radius:6px; padding:6px 8px; width:100%; margin-bottom:10px; }
  textarea { width:100%; min-height:340px; background:#0b0f14; color:var(--txt);
             border:1px solid var(--border); border-radius:6px; padding:10px;
             font:12px/1.5 ui-monospace,Menlo,monospace; resize:vertical; }
  button { background:var(--blue); color:#04101f; border:0; border-radius:6px;
           padding:9px 16px; font-weight:600; cursor:pointer; margin-top:10px; }
  button:disabled { opacity:.5; cursor:default; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .pill { display:inline-block; padding:3px 9px; border-radius:6px; font-size:12px;
          font-weight:600; }
  .send-yes { background:#1f3a26; color:var(--green); }
  .send-no  { background:#3a1f22; color:var(--red); }
  .msg { background:#0b0f14; border:1px solid var(--border); border-radius:8px;
         padding:12px; margin-top:10px; white-space:pre-wrap; word-break:break-word; }
  .kv { font-size:12px; color:var(--muted); margin:2px 0; }
  .kv b { color:var(--txt); font-weight:600; }
  h3 { font-size:13px; margin:16px 0 8px; color:var(--muted);
       text-transform:uppercase; letter-spacing:.04em; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td { padding:5px 6px; border-bottom:1px solid var(--border); vertical-align:top; }
  td.s { width:46px; text-align:center; font-weight:700; }
  .ok { color:var(--green); } .fail { color:var(--red); }
  .sev { font-size:10px; padding:1px 5px; border-radius:4px; }
  .sev.critical { background:#3a1f22; color:var(--red); }
  .sev.high { background:#3a2f1f; color:var(--amber); }
  .sev.medium { background:#22303a; color:var(--blue); }
  .warn { color:var(--amber); font-size:12px; margin:3px 0;
          font-family:ui-monospace,monospace; }
  .empty { color:var(--muted); font-style:italic; }
  .score { font-size:18px; font-weight:700; }
  .reason { font-style:italic; color:var(--muted); margin-top:8px; }
  nav.tabs { display:flex; gap:6px; margin-left:auto; }
  nav.tabs button { background:transparent; color:var(--muted); border:1px solid var(--border);
                    padding:5px 14px; border-radius:6px; margin:0; font-weight:600; }
  nav.tabs button.active { background:var(--blue); color:#04101f; border-color:var(--blue); }
  .hidden { display:none; }
  .drop { border:1.5px dashed var(--border); border-radius:8px; padding:18px; text-align:center;
          color:var(--muted); cursor:pointer; }
  .drop.over { border-color:var(--blue); color:var(--blue); }
  .stat { display:inline-block; background:#0b0f14; border:1px solid var(--border);
          border-radius:8px; padding:8px 12px; margin:4px 6px 4px 0; }
  .stat b { display:block; font-size:18px; }
  .stat span { font-size:11px; color:var(--muted); text-transform:uppercase; }
  tr.crit td { background:#2a1518; }
  td.ta { font-family:ui-monospace,monospace; font-size:11px; }
  #batchInput { min-height:120px; }
</style>
</head>
<body>
<header>
  <h1>🛰️ Context-Aware Messaging Agent</h1>
  <span id="provider">checking provider…</span>
  <nav class="tabs">
    <button id="tab-single" class="active">Single record</button>
    <button id="tab-batch">Batch (JSONL)</button>
  </nav>
</header>
<main id="view-single">
  <section class="card">
    <label for="preset">Load a preset record</label>
    <select id="preset"><option value="">— choose a sample / adversarial record —</option></select>
    <label for="input">Record (JSON)</label>
    <textarea id="input" spellcheck="false"></textarea>
    <button id="run">Run agent ▶</button>
  </section>
  <section class="card" id="out">
    <p class="empty">Run a record to see the agent's decision, message, AC results and audit warnings.</p>
  </section>
</main>
<main id="view-batch" class="hidden" style="grid-template-columns:1fr;">
  <section class="card">
    <label>Upload or paste a JSONL file (one record per line)</label>
    <div class="drop" id="drop">⬆ Drop a <b>.jsonl</b> file here, or click to choose
      <input type="file" id="file" accept=".jsonl,.json,.txt" class="hidden"/></div>
    <div class="row" style="margin:10px 0;">
      <button id="loadSample" style="background:var(--border);color:var(--txt)">Load bundled sample+adversarial</button>
    </div>
    <textarea id="batchInput" spellcheck="false" placeholder='{"task_id":"...", ...}\n{"task_id":"...", ...}'></textarea>
    <button id="runBatch">Run batch ▶</button>
  </section>
  <section class="card" id="batchOut">
    <p class="empty">Upload a JSONL file and run it to get a per-record pass/fail summary.</p>
  </section>
</main>
<script>
const $ = s => document.querySelector(s);
let PRESETS = [];

async function loadProvider(){
  const p = await (await fetch('/api/provider')).json();
  const el = $('#provider');
  el.innerHTML = 'LLM: <span class="badge '+(p.llm_active?'live':'fallback')+'">'
    + p.provider + '</span>';
}
async function loadPresets(){
  const r = await (await fetch('/api/presets')).json();
  PRESETS = r.presets;
  const sel = $('#preset');
  PRESETS.forEach((p,i)=>{
    const o=document.createElement('option'); o.value=i;
    o.textContent='['+p.kind+'] '+p.task_id; sel.appendChild(o);
  });
  if(PRESETS.length){ sel.value=0; $('#input').value=PRESETS[0].json; }
}
$('#preset').addEventListener('change', e=>{
  const i=e.target.value; if(i!=='') $('#input').value=PRESETS[i].json;
});

function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function render(d){
  if(d.error){ return '<p class="fail">Error ('+esc(d.type||'')+'): '+esc(d.error)+'</p>'; }
  const send = d.should_send;
  const ev = d.evaluation||{};
  const acs = d.ac_results||[];
  const fails = acs.filter(a=>!a.pass);
  const msg = d.next_message;
  let h = '';
  h += '<div class="row"><span class="pill '+(send?'send-yes':'send-no')+'">'
     + (send?'WILL SEND':'WILL NOT SEND')+'</span>';
  h += '<span class="score '+(ev.critical_fails&&ev.critical_fails.length?'fail':'ok')
     + '">AC '+(ev.score||'?')+'</span>';
  h += '<span class="kv">latency <b>'+(d.latency_ms||0)+'ms</b></span></div>';
  if(d.abort_reason) h+='<div class="kv">abort_reason: <b>'+esc(d.abort_reason)+'</b></div>';

  if(msg){
    h+='<h3>Message</h3><div class="msg">';
    h+='<div class="kv">channel: <b>'+esc(msg.channel)+'</b></div>';
    h+='<div class="kv">send_at: <b>'+esc(msg.send_at)+'</b></div>';
    if(msg.subject) h+='<div class="kv">subject: <b>'+esc(msg.subject)+'</b></div>';
    h+='<div style="margin-top:8px">'+esc(msg.body||'')+'</div>';
    if(msg.cta) h+='<div class="kv" style="margin-top:8px">cta: '+esc(JSON.stringify(msg.cta))+'</div>';
    h+='</div>';
  }
  h+='<h3>Next action</h3><div class="kv">'+esc(JSON.stringify(d.next_action))+'</div>';
  if(d.reasoning) h+='<div class="reason">"'+esc(d.reasoning)+'"</div>';

  h+='<h3>Compliance audit warnings</h3>';
  if(d.warnings&&d.warnings.length) h+=d.warnings.map(w=>'<div class="warn">⚠ '+esc(w)+'</div>').join('');
  else h+='<p class="empty">none — LLM output passed compliance unchanged</p>';

  if(d.safety_violations&&d.safety_violations.length){
    h+='<h3>Safety violations detected</h3>';
    h+=d.safety_violations.map(v=>'<div class="warn">⚠ '+esc(JSON.stringify(v))+'</div>').join('');
  }

  h+='<h3>Acceptance criteria ('+(acs.length-fails.length)+'/'+acs.length+')</h3><table>';
  acs.forEach(a=>{
    h+='<tr><td class="s '+(a.pass?'ok':'fail')+'">'+(a.pass?'✓':'✗')+'</td>'
     + '<td><b>'+esc(a.id)+'</b> <span class="sev '+esc(a.severity)+'">'+esc(a.severity)+'</span><br>'
     + esc(a.title)+(a.detail?'<br><span class="empty">'+esc(a.detail)+'</span>':'')+'</td></tr>';
  });
  h+='</table>';
  return h;
}

$('#run').addEventListener('click', async ()=>{
  const btn=$('#run'); btn.disabled=true; btn.textContent='Running…';
  $('#out').innerHTML='<p class="empty">running…</p>';
  try{
    const r = await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({record:$('#input').value})});
    const d = await r.json();
    $('#out').innerHTML = render(d);
  }catch(e){ $('#out').innerHTML='<p class="fail">Request failed: '+esc(e)+'</p>'; }
  btn.disabled=false; btn.textContent='Run agent ▶';
});

loadProvider(); loadPresets();

// ---- tabs ----
function showTab(which){
  $('#tab-single').classList.toggle('active', which==='single');
  $('#tab-batch').classList.toggle('active', which==='batch');
  $('#view-single').classList.toggle('hidden', which!=='single');
  $('#view-batch').classList.toggle('hidden', which!=='batch');
}
$('#tab-single').addEventListener('click', ()=>showTab('single'));
$('#tab-batch').addEventListener('click', ()=>showTab('batch'));

// ---- batch upload ----
const drop=$('#drop'), fileInput=$('#file');
drop.addEventListener('click', ()=>fileInput.click());
fileInput.addEventListener('change', e=>{ if(e.target.files[0]) readFile(e.target.files[0]); });
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop', e=>{ const f=e.dataTransfer.files[0]; if(f) readFile(f); });
function readFile(f){ const r=new FileReader(); r.onload=()=>{$('#batchInput').value=r.result;}; r.readAsText(f); }

$('#loadSample').addEventListener('click', ()=>{
  $('#batchInput').value = PRESETS.map(p=>JSON.stringify(JSON.parse(p.json))).join('\\n');
});

function renderBatch(d){
  if(d.error) return '<p class="fail">'+esc(d.error)+'</p>';
  const s=d.summary;
  let h='<div class="row"><span class="pill '+(s.all_pass?'send-yes':'send-no')+'">'
      +(s.all_pass?'ALL PASS':'CRITICAL FAILS')+'</span>'
      +'<button id="download" style="margin:0;background:var(--border);color:var(--txt)">⬇ Download results (.jsonl)</button></div>';
  h+='<div style="margin:10px 0">'
    +'<span class="stat"><b>'+s.records+'</b><span>records</span></span>'
    +'<span class="stat"><b>'+s.ac_pass+'/'+s.ac_total+'</b><span>AC passed</span></span>'
    +'<span class="stat"><b class="'+(s.critical_fail_records?'fail':'ok')+'">'+s.critical_fail_records+'</b><span>crit-fail recs</span></span>'
    +'<span class="stat"><b>'+s.send+'</b><span>will send</span></span>'
    +'<span class="stat"><b>'+s.no_send+'</b><span>no send</span></span>'
    +'<span class="stat"><b>'+s.max_latency_ms+'ms</b><span>max latency</span></span>'
    +(s.errors?'<span class="stat"><b class="fail">'+s.errors+'</b><span>errors</span></span>':'')
    +'</div>';
  h+='<table><tr><td>#</td><td>task_id</td><td>send</td><td>channel</td><td>AC</td>'
    +'<td>crit fails</td><td>warn</td><td>ms</td></tr>';
  d.rows.forEach(r=>{
    if(r.error){ h+='<tr class="crit"><td>'+r.line+'</td><td class="ta">'+esc(r.task_id)
      +'</td><td colspan="6" class="fail">'+esc(r.error)+'</td></tr>'; return; }
    const crit=r.critical_fails&&r.critical_fails.length;
    h+='<tr'+(crit?' class="crit"':'')+'><td>'+r.line+'</td><td class="ta">'+esc(r.task_id)+'</td>'
      +'<td class="'+(r.should_send?'ok':'')+'">'+(r.should_send?'yes':'no')+(r.abort_reason?' <span class="empty">('+esc(r.abort_reason)+')</span>':'')+'</td>'
      +'<td>'+esc(r.channel||'—')+'</td>'
      +'<td class="'+(crit?'fail':'ok')+'">'+esc(r.ac_score)+'</td>'
      +'<td class="fail ta">'+(crit?esc(r.critical_fails.join(',')):'')+'</td>'
      +'<td>'+(r.warnings||'')+'</td><td>'+r.latency_ms+'</td></tr>';
  });
  h+='</table>';
  return h;
}

let LAST_BATCH=null;

$('#runBatch').addEventListener('click', async ()=>{
  const btn=$('#runBatch'); btn.disabled=true; btn.textContent='Running…';
  $('#batchOut').innerHTML='<p class="empty">running batch…</p>';
  try{
    const r=await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({content:$('#batchInput').value})});
    LAST_BATCH=await r.json();
    $('#batchOut').innerHTML=renderBatch(LAST_BATCH);
    const dl=$('#download');
    if(dl) dl.addEventListener('click', downloadResults);
  }catch(e){ $('#batchOut').innerHTML='<p class="fail">Request failed: '+esc(e)+'</p>'; }
  btn.disabled=false; btn.textContent='Run batch ▶';
});

function downloadResults(){
  if(!LAST_BATCH||!LAST_BATCH.outputs) return;
  const jsonl=LAST_BATCH.outputs.map(o=>JSON.stringify(o)).join('\\n')+'\\n';
  const blob=new Blob([jsonl],{type:'application/x-ndjson'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='agent_results_'+new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')+'.jsonl';
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
}
</script>
</body>
</html>
"""


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Messaging-agent web test console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(api, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
