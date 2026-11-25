"""Self-contained zero-dependency Web Control Console (/_console) for basalt-colab."""
from __future__ import annotations

CONSOLE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>basalt-colab · Control Console</title>
<style>
  :root {
    --bg: #0b0e14; --card: #131822; --panel: #0e131c; --border: #232b3a; --text: #e6edf3; --muted: #8b949e;
    --accent: #3fb950; --cyan: #58a6ff; --amber: #d29922; --red: #f85149;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 20px; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
  .brand { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .brand h1 { margin: 0; font-size: 18px; letter-spacing: -0.02em; }
  .pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; background: #1d2433; border: 1px solid var(--border); }
  .pill.ready { color: var(--accent); border-color: #264d32; background: #112418; }
  .pill.starting, .pill.stopping { color: var(--cyan); border-color: #1f3c63; background: #101d30; }
  .pill.off { color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
  .actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  button, a.btn { cursor: pointer; font: inherit; font-size: 13px; font-weight: 600; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border); background: #1a2130; color: var(--text); text-decoration: none; display: inline-flex; align-items: center; gap: 6px; transition: background 0.12s, border-color 0.12s; }
  button:hover, a.btn:hover { background: #232d42; }
  button.primary, a.btn.primary { background: #1e6b34; border-color: #2ea043; color: #fff; }
  button.primary:hover, a.btn.primary:hover { background: #238636; }
  button.warn-btn { background: #3b2d12; border-color: #7c5b17; color: #f2cc60; }
  button.warn-btn:hover { background: #4e3b17; }
  button.danger { background: #3b191d; border-color: #6e282e; color: #ffa198; }
  button.danger:hover { background: #541f25; }
  button.sm { padding: 4px 9px; font-size: 12px; border-radius: 6px; }
  button.active { background: #17344f; border-color: var(--cyan); color: #a5d6ff; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 16px; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 14px; margin-top: 14px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .card h2 { margin: 0 0 12px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
  .kpi { font-size: 24px; font-weight: 700; font-family: var(--mono); }
  .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .badge { font-size: 10px; font-weight: 700; letter-spacing: 0.05em; padding: 2px 6px; border-radius: 999px; text-transform: uppercase; }
  .badge.hot { color: var(--accent); background: #112418; border: 1px solid #264d32; }
  .badge.cold { color: var(--amber); background: #2c220f; border: 1px solid #6e5218; }
  .steps { list-style: none; margin: 0; padding: 0; font-family: var(--mono); font-size: 12px; }
  .steps li { padding: 4px 0; display: flex; gap: 10px; border-bottom: 1px solid #1b2230; }
  .steps li:last-child { border-bottom: 0; }
  .ok { color: var(--accent); } .dim { color: var(--muted); } .err { color: var(--red); } .warn { color: var(--amber); }
  textarea, input[type=text], input[type=password], input[type=number], select {
    width: 100%; background: #0c1018; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 7px 9px; font-family: var(--mono); font-size: 12px;
  }
  input[type=range] { width: 100%; accent-color: var(--cyan); }
  textarea { min-height: 220px; resize: vertical; line-height: 1.45; }
  pre { margin: 0; background: #0c1018; border: 1px solid var(--border); border-radius: 8px; padding: 10px; max-height: 310px; overflow: auto; font: 12px/1.45 var(--mono); white-space: pre-wrap; word-break: break-word; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--mono); margin-top: 10px; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid #1b2230; }
  th { color: var(--muted); font-weight: 600; }
  .bar-chart { width: 100%; height: 110px; display: block; margin-top: 8px; }
  .auth-box { max-width: 440px; margin: 80px auto; }
  .hidden { display: none !important; }
  .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .between { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
  label.chk { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); cursor: pointer; }
  .toast { margin-top: 10px; font-size: 12px; min-height: 18px; }
  .sec-box { background: var(--panel); border: 1px solid #1d2433; border-radius: 10px; padding: 12px; margin-bottom: 12px; }
  .sec-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; font-size: 12px; font-weight: 600; color: var(--text); }
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .field label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .pills { display: flex; gap: 6px; flex-wrap: wrap; }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
  .kv-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin-bottom: 12px; }
  .kv-item { background: var(--panel); border: 1px solid #1d2433; border-radius: 8px; padding: 8px 10px; }
  .kv-item .k { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
  .kv-item .v { font-family: var(--mono); font-size: 12px; margin-top: 2px; word-break: break-all; display: flex; align-items: center; justify-content: space-between; gap: 6px; }
  details { margin-top: 12px; }
  details summary { cursor: pointer; font-size: 12px; color: var(--muted); user-select: none; padding: 4px 0; }
  details[open] summary { margin-bottom: 8px; color: var(--text); }
  code { font-family: var(--mono); font-size: 12px; background: #0c1018; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<div id="authView" class="wrap auth-box hidden">
  <div class="card">
    <h2>basalt-colab · Authentication</h2>
    <p class="sub">Enter your local API key (from <code>basalt-colab status</code> or <code>basalt-colab console</code>). Stored only in browser <code>localStorage</code>.</p>
    <input id="keyInput" type="password" placeholder="sk-ds41-..." autocomplete="off">
    <div class="row" style="margin-top:12px">
      <button class="primary" onclick="saveKey()">Connect</button>
    </div>
    <div id="authErr" class="toast err"></div>
  </div>
</div>

<div id="appView" class="wrap hidden">
  <header>
    <div class="brand">
      <h1>basalt-colab</h1>
      <span id="statePill" class="pill off"><span class="dot"></span><span id="stateText">OFF</span></span>
      <span id="modelMeta" class="sub"></span>
    </div>
    <div class="actions">
      <label class="chk"><input id="waitGpuChk" type="checkbox"> --wait-gpu</label>
      <button id="btnUp" class="primary" onclick="startGpu()">Start G4</button>
      <button id="btnRestart" class="warn-btn" onclick="restartGpu()" title="Stop G4, reload saved config.toml, and boot fresh">Restart G4</button>
      <button id="btnDown" class="danger" onclick="stopGpu()">Stop G4</button>
      <a id="btnWebUi" class="btn hidden" href="/" target="_blank" rel="noopener">Open Basalt Web UI ↗</a>
      <button onclick="logout()" title="Clear saved API key">Sign out</button>
    </div>
  </header>

  <div id="errBanner" class="card hidden" style="margin-top:14px;border-color:#6e282e;background:#241216;">
    <span class="err" id="errText"></span>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Decode Speed</h2>
      <div class="kpi" id="kpiSpeed">—</div>
      <div class="sub" id="kpiHit">VRAM hit — · PCIe —</div>
    </div>
    <div class="card">
      <h2>GPU Activity</h2>
      <div class="kpi" id="kpiActivity">Idle</div>
      <div class="sub" id="kpiIdle">Auto-stop —</div>
    </div>
    <div class="card">
      <h2>Session Cost</h2>
      <div class="kpi" id="kpiSessionCu">0.00 CU</div>
      <div class="sub" id="kpiRan">Not running</div>
    </div>
    <div class="card">
      <h2>Billing Ledger</h2>
      <div class="kpi" id="kpiMonthCu">0.00 CU</div>
      <div class="sub" id="kpiTotals">Today 0.00 · All-time 0.00 CU</div>
    </div>
  </div>

  <div id="bootCard" class="card hidden" style="margin-top:14px;">
    <h2>Boot Timeline <span id="bootStage" class="dim"></span></h2>
    <ul id="bootSteps" class="steps"></ul>
  </div>

  <div class="card" style="margin-top:14px;">
    <h2>
      <span>Client Integration (Live Config Generator)</span>
      <span class="row">
        <button class="sm" onclick="loadClients()">Refresh</button>
      </span>
    </h2>
    <div class="kv-strip">
      <div class="kv-item">
        <div class="k">OpenAI Base URL</div>
        <div class="v"><span id="epBaseUrl">—</span><button class="sm" onclick="copyText('epBaseUrl', this)">Copy</button></div>
      </div>
      <div class="kv-item">
        <div class="k">Active Model Alias</div>
        <div class="v"><span id="epModel">—</span><button class="sm" onclick="copyText('epModel', this)">Copy</button></div>
      </div>
      <div class="kv-item">
        <div class="k">Local API Key</div>
        <div class="v">
          <span id="epKeyMask">sk-ds41-••••••••</span>
          <span class="row">
            <button class="sm" id="btnRevealKey" onclick="toggleRevealKey()">Show</button>
            <button class="sm" onclick="copyRaw(apiKey, this)">Copy</button>
          </span>
        </div>
      </div>
    </div>
    <div id="clientTabs" class="tabs"></div>
    <div id="clientPane">
      <div class="between" style="margin-bottom:8px;">
        <div>
          <strong id="clTitle">—</strong>
          <div id="clSub" class="sub" style="margin-top:2px;"></div>
        </div>
        <div class="row">
          <span id="clFileHint" class="sub"></span>
          <button class="sm primary" onclick="copyRaw(activeClient ? activeClient.snippet : '', this)">Copy Config</button>
        </div>
      </div>
      <pre id="clSnippet">Loading...</pre>
      <div class="between" style="margin-top:10px;margin-bottom:6px;">
        <span class="sub">Shell Environment</span>
        <span class="row">
          <button id="btnEnvSh" class="sm active" onclick="setEnvShell('sh')">bash / zsh</button>
          <button id="btnEnvPs" class="sm" onclick="setEnvShell('ps')">PowerShell</button>
          <button class="sm" onclick="copyRaw(activeEnvText(), this)">Copy Env</button>
        </span>
      </div>
      <pre id="clEnv" style="max-height:130px;"></pre>
      <div id="clNotes" class="sub" style="margin-top:8px;"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>
        <span>Visual Configuration</span>
        <span class="row">
          <button class="sm" onclick="loadConfig()">Refresh</button>
          <button class="sm primary" onclick="saveVisualConfig(false)">Save &amp; Apply</button>
          <button class="sm warn-btn" onclick="saveVisualConfig(true)" title="Save config.toml and restart G4 to apply engine changes">Save &amp; Restart G4</button>
        </span>
      </h2>

      <div class="sec-box">
        <div class="sec-title">
          <span>Performance Profile</span>
          <span class="badge cold">Engine Preset</span>
        </div>
        <div class="pills">
          <button id="prof-balanced" class="sm" onclick="applyProfilePreset('balanced')">Balanced (2-slot · default)</button>
          <button id="prof-solo-fast" class="sm" onclick="applyProfilePreset('solo-fast')">Solo-Fast (1-slot · max tok/s)</button>
          <button id="prof-exact" class="sm" onclick="applyProfilePreset('exact')">Exact (2-slot · lossless)</button>
        </div>
      </div>

      <div class="sec-box">
        <div class="sec-title">
          <span>Runtime Controls</span>
          <span class="badge hot">Live Hot-Reload</span>
        </div>
        <div class="field" style="margin-bottom:10px;">
          <label>Reasoning Effort (default injected when request omits effort)</label>
          <div class="row">
            <div class="pills">
              <button class="sm" data-effort="low" onclick="pickEffort('low')">low</button>
              <button class="sm" data-effort="50" onclick="pickEffort('50')">50</button>
              <button class="sm" data-effort="high" onclick="pickEffort('high')">high</button>
              <button class="sm" data-effort="max" onclick="pickEffort('max')">max</button>
            </div>
            <input id="fEffort" type="text" style="width:90px;" placeholder="1-100" oninput="syncEffortPills()">
          </div>
        </div>
        <div class="form-grid">
          <div class="field">
            <label for="fIdleMin">Idle Auto-Stop (min)</label>
            <input id="fIdleMin" type="number" step="1" min="1" max="120">
          </div>
          <div class="field">
            <label for="fMaxHours">Hard Stop Cap (hours)</label>
            <input id="fMaxHours" type="number" step="0.5" min="0.5" max="24">
          </div>
          <div class="field">
            <label for="fWaitGpuMin">Wait-GPU Limit (min, 0=∞)</label>
            <input id="fWaitGpuMin" type="number" step="5" min="0" max="1440">
          </div>
        </div>
        <div class="row" style="margin-top:10px;">
          <label class="chk"><input id="fWaitGpu" type="checkbox"> Default --wait-gpu on wake</label>
          <label class="chk"><input id="fStrictModel" type="checkbox"> Strict model name matching</label>
        </div>
      </div>

      <div class="sec-box">
        <div class="sec-title">
          <span>Engine &amp; Model Settings</span>
          <span class="badge cold">Applies on Next G4 Boot</span>
        </div>
        <div class="form-grid">
          <div class="field">
            <label for="fQuant">Quantization Pack</label>
            <select id="fQuant" onchange="togglePackUrl()">
              <option value="ud-iq1_m">ud-iq1_m (100% RAM · fastest boot)</option>
              <option value="ud-iq2_xxs">ud-iq2_xxs (-13% KLD · hybrid)</option>
              <option value="ud-q2_k_xl">ud-q2_k_xl (custom URL)</option>
              <option value="custom">custom (.basalt URL)</option>
            </select>
          </div>
          <div class="field">
            <label for="fMaxContext">Context / Slot</label>
            <select id="fMaxContext">
              <option value="32768">32K</option>
              <option value="65536">64K</option>
              <option value="131072">128K</option>
              <option value="262144">256K (default)</option>
              <option value="524288">512K</option>
              <option value="1048576">1M</option>
            </select>
          </div>
          <div class="field">
            <label for="fParallel">Parallel Slots</label>
            <select id="fParallel" onchange="syncProfileHighlight()">
              <option value="1">1 (Solo)</option>
              <option value="2">2 (Default)</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>
          </div>
          <div class="field">
            <label for="fRouting">Routing Mode</label>
            <select id="fRouting" onchange="syncProfileHighlight()">
              <option value="balanced">balanced (+20-37% speed)</option>
              <option value="fast">fast (max speed)</option>
              <option value="exact">exact (lossless)</option>
            </select>
          </div>
          <div class="field">
            <label for="fKv">KV Precision</label>
            <select id="fKv">
              <option value="fp8">fp8 (default)</option>
              <option value="fp16">fp16</option>
            </select>
          </div>
          <div class="field">
            <label for="fVision">Vision</label>
            <select id="fVision">
              <option value="dynamic">dynamic (images on)</option>
              <option value="off">off</option>
            </select>
          </div>
          <div class="field">
            <label for="fAnswerRoom">Answer Room (anti-truncation)</label>
            <input id="fAnswerRoom" type="text" placeholder="auto or 0..65536">
          </div>
          <div class="field">
            <label for="fVramReserve">VRAM Reserve (MiB)</label>
            <input id="fVramReserve" type="number" step="50" min="256" max="16384" onchange="syncProfileHighlight()">
          </div>
        </div>
        <div id="packUrlRow" class="field hidden" style="margin-top:10px;">
          <label for="fPackUrl">Custom .basalt Pack URL</label>
          <input id="fPackUrl" type="text" placeholder="https://.../model.basalt">
        </div>
        <div class="between" style="margin-top:10px;">
          <label class="chk"><input id="fUncensor" type="checkbox" onchange="syncUncensorUi()"> Abliteration (Uncensored)</label>
          <span class="sub" id="uncensorReadout">1.00×</span>
        </div>
        <input id="fUncensorScale" type="range" min="0" max="2" step="0.05" value="1" oninput="syncUncensorUi()">
      </div>

      <div id="cfgToast" class="toast"></div>

      <details>
        <summary>Advanced: Raw <code>config.toml</code> Editor</summary>
        <textarea id="cfgText" spellcheck="false"></textarea>
        <div class="row" style="margin-top:8px;">
          <button class="sm" onclick="saveRawToml(false)">Save Raw TOML</button>
          <button class="sm warn-btn" onclick="saveRawToml(true)">Save Raw &amp; Restart G4</button>
        </div>
      </details>
    </div>

    <div class="card">
      <h2>CU Cost Curve (Last 14 Days) <span id="ledgerCount" class="dim"></span></h2>
      <svg id="cuChart" class="bar-chart" viewBox="0 0 420 110" preserveAspectRatio="none"></svg>
      <table>
        <thead><tr><th>Started</th><th>Duration</th><th>Est. CU</th><th>Reason</th><th>Req</th></tr></thead>
        <tbody id="recentBody"></tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:14px;">
    <h2>
      <span>Logs</span>
      <span class="row">
        <label class="chk"><input id="vmLogChk" type="checkbox" onchange="loadLogs()"> Include VM logs</label>
        <button class="sm" onclick="loadLogs()">Refresh</button>
      </span>
    </h2>
    <pre id="logOut">Loading logs...</pre>
  </div>
</div>

<script>
const STORAGE_KEY = "basalt_colab_api_key";
let apiKey = "";
let pollTimer = null;
let loadedConfig = null;
let clientsData = null;
let activeClientId = "dsh";
let activeClient = null;
let envShell = "sh";
let keyRevealed = false;

function initAuth() {
  const hash = location.hash || "";
  const m = hash.match(/key=([^&]+)/);
  if (m) {
    localStorage.setItem(STORAGE_KEY, decodeURIComponent(m[1]));
    history.replaceState(null, "", location.pathname + location.search);
  }
  apiKey = localStorage.getItem(STORAGE_KEY) || "";
  if (!apiKey) { showAuth(); return; }
  verifyAndBoot();
}

function showAuth(msg) {
  document.getElementById("authView").classList.remove("hidden");
  document.getElementById("appView").classList.add("hidden");
  document.getElementById("authErr").textContent = msg || "";
}

async function saveKey() {
  apiKey = document.getElementById("keyInput").value.trim();
  if (!apiKey) return;
  localStorage.setItem(STORAGE_KEY, apiKey);
  await verifyAndBoot();
}

function logout() {
  localStorage.removeItem(STORAGE_KEY);
  apiKey = "";
  if (pollTimer) clearInterval(pollTimer);
  showAuth();
}

async function api(path, opts = {}) {
  const headers = Object.assign({ "Authorization": "Bearer " + apiKey }, opts.headers || {});
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (r.status === 401) { logout(); showAuth("Invalid API key."); throw new Error("unauthorized"); }
  return r;
}

async function verifyAndBoot() {
  try {
    const r = await api("/_control/status");
    const st = await r.json();
    document.getElementById("authView").classList.add("hidden");
    document.getElementById("appView").classList.remove("hidden");
    renderStatus(st);
    await Promise.all([loadConfig(), loadClients(), loadUsage(), loadLogs()]);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const sr = await api("/_control/status");
        renderStatus(await sr.json());
      } catch (_) {}
    }, 1500);
  } catch (e) {
    if (e.message !== "unauthorized") showAuth("Cannot reach proxy: " + e.message);
  }
}

function mmss(s) {
  s = Math.max(0, Math.floor(s || 0));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

function renderStatus(st) {
  const base = st.state.startsWith("ready") ? "ready" : st.state;
  const pill = document.getElementById("statePill");
  pill.className = "pill " + base;
  document.getElementById("stateText").textContent = st.state.toUpperCase();
  document.getElementById("modelMeta").textContent = st.model + " · " + (st.engine_summary || "") + " · effort " + st.effort;
  document.getElementById("btnUp").disabled = (base === "ready" || base === "starting" || base === "stopping");
  document.getElementById("btnRestart").disabled = (base === "starting" || base === "stopping");
  document.getElementById("btnDown").disabled = (base === "off" || base === "stopping");
  document.getElementById("btnWebUi").classList.toggle("hidden", base !== "ready");

  const errB = document.getElementById("errBanner");
  if (st.last_error && base === "off") {
    errB.classList.remove("hidden");
    document.getElementById("errText").textContent = "Last error: " + st.last_error;
  } else {
    errB.classList.add("hidden");
  }

  const b = st.basalt || {};
  const spd = (b.busy && b.live_tok_s != null) ? b.live_tok_s : b.last_tok_s;
  document.getElementById("kpiSpeed").textContent = spd != null ? spd.toFixed(1) + " tok/s" : "—";
  const hit = b.hit_rate != null ? (b.hit_rate * 100).toFixed(1) + "%" : "—";
  const pcie = b.pcie_share != null ? (b.pcie_share * 100).toFixed(1) + "%" : "—";
  document.getElementById("kpiHit").textContent = "VRAM hit " + hit + " · PCIe " + pcie;

  if (base === "ready") {
    const left = Math.min(st.idle_stop_min * 60 - st.idle_s, st.max_hours * 3600 - (st.ran || 0));
    document.getElementById("kpiActivity").textContent = st.active ? (st.active + " active") : ("Idle " + mmss(st.idle_s));
    document.getElementById("kpiIdle").textContent = "Stops in " + mmss(left) + " · " + st.served + " served";
  } else if (base === "starting") {
    document.getElementById("kpiActivity").textContent = "Starting";
    document.getElementById("kpiIdle").textContent = st.stage || "booting...";
  } else {
    document.getElementById("kpiActivity").textContent = "Off";
    document.getElementById("kpiIdle").textContent = "Starts on request or Start G4";
  }

  document.getElementById("kpiSessionCu").textContent = st.cu != null ? st.cu.toFixed(2) + " CU" : "0.00 CU";
  document.getElementById("kpiRan").textContent = st.ran != null ? ("Running " + mmss(st.ran) + " · ref " + st.basalt_ref) : "0 CU billed while off";

  const bootCard = document.getElementById("bootCard");
  if (base === "starting" || (st.steps && st.steps.length && base !== "off")) {
    bootCard.classList.remove("hidden");
    document.getElementById("bootStage").textContent = base === "starting" ? ("· " + (st.stage || "")) : "· Ready";
    document.getElementById("bootSteps").innerHTML = (st.steps || []).map(x =>
      `<li><span class="dim">${mmss(x.elapsed_s)}</span><span class="ok">✓</span><span>${esc(x.label)}</span></li>`
    ).join("");
  } else {
    bootCard.classList.add("hidden");
  }
}

async function _drainControl(path, body) {
  const r = await api(path, { method: "POST", body: JSON.stringify(body) });
  const reader = r.body.getReader();
  while (true) {
    const { done } = await reader.read();
    if (done) break;
  }
}

async function startGpu() {
  const wait_gpu = document.getElementById("waitGpuChk").checked;
  document.getElementById("btnUp").disabled = true;
  try {
    await _drainControl("/_control/up", { wait_gpu });
  } finally {
    await Promise.all([loadClients(), loadUsage(), loadLogs()]);
  }
}

async function restartGpu() {
  const wait_gpu = document.getElementById("waitGpuChk").checked;
  document.getElementById("btnRestart").disabled = true;
  document.getElementById("btnUp").disabled = true;
  try {
    await _drainControl("/_control/restart", { wait_gpu });
  } finally {
    await Promise.all([loadClients(), loadUsage(), loadLogs()]);
  }
}

async function stopGpu() {
  document.getElementById("btnDown").disabled = true;
  try {
    await _drainControl("/_control/down", { exit: false });
  } finally {
    await Promise.all([loadUsage(), loadLogs()]);
  }
}

/* ---------- Visual Config Form ---------- */
function populateForm(cfg) {
  if (!cfg) return;
  const c = cfg.colab || {}, p = cfg.proxy || {}, b = cfg.basalt || {};
  document.getElementById("fEffort").value = b.effort != null ? String(b.effort) : "50";
  document.getElementById("fIdleMin").value = c.idle_min != null ? c.idle_min : 10;
  document.getElementById("fMaxHours").value = c.max_hours != null ? c.max_hours : 6;
  document.getElementById("fWaitGpu").checked = !!c.wait_gpu;
  document.getElementById("waitGpuChk").checked = !!c.wait_gpu;
  document.getElementById("fWaitGpuMin").value = c.wait_gpu_min != null ? c.wait_gpu_min : 0;
  document.getElementById("fStrictModel").checked = !!p.strict_model;
  document.getElementById("fQuant").value = b.quant || "ud-iq1_m";
  document.getElementById("fPackUrl").value = b.pack_url || "";
  document.getElementById("fMaxContext").value = String(b.max_context || 262144);
  document.getElementById("fParallel").value = String(b.parallel || 2);
  document.getElementById("fRouting").value = b.routing_mode || "balanced";
  document.getElementById("fKv").value = b.kv || "fp8";
  document.getElementById("fVision").value = b.vision || "dynamic";
  document.getElementById("fAnswerRoom").value = b.answer_room != null ? String(b.answer_room) : "auto";
  document.getElementById("fVramReserve").value = b.vram_reserve_mib != null ? b.vram_reserve_mib : 450;
  document.getElementById("fUncensor").checked = b.uncensor !== false;
  document.getElementById("fUncensorScale").value = b.uncensor_scale != null ? b.uncensor_scale : 1.0;
  togglePackUrl();
  syncEffortPills();
  syncUncensorUi();
  syncProfileHighlight();
}

function collectConfigFromForm() {
  const base = loadedConfig ? JSON.parse(JSON.stringify(loadedConfig)) : { colab: {}, proxy: {}, basalt: { sampling: {} } };
  base.colab = Object.assign(base.colab || {}, {
    idle_min: Number(document.getElementById("fIdleMin").value),
    max_hours: Number(document.getElementById("fMaxHours").value),
    wait_gpu: document.getElementById("fWaitGpu").checked,
    wait_gpu_min: Number(document.getElementById("fWaitGpuMin").value || 0),
  });
  base.proxy = Object.assign(base.proxy || {}, {
    strict_model: document.getElementById("fStrictModel").checked,
  });
  const uncOn = document.getElementById("fUncensor").checked;
  const uncScale = Number(document.getElementById("fUncensorScale").value);
  base.basalt = Object.assign(base.basalt || {}, {
    quant: document.getElementById("fQuant").value,
    pack_url: document.getElementById("fPackUrl").value.trim(),
    max_context: Number(document.getElementById("fMaxContext").value),
    parallel: Number(document.getElementById("fParallel").value),
    routing_mode: document.getElementById("fRouting").value,
    kv: document.getElementById("fKv").value,
    vision: document.getElementById("fVision").value,
    answer_room: document.getElementById("fAnswerRoom").value.trim() || "auto",
    vram_reserve_mib: Number(document.getElementById("fVramReserve").value),
    uncensor: uncOn,
    uncensor_scale: uncOn ? uncScale : 0.0,
    effort: document.getElementById("fEffort").value.trim() || "50",
  });
  return base;
}

function togglePackUrl() {
  const q = document.getElementById("fQuant").value;
  document.getElementById("packUrlRow").classList.toggle("hidden", !(q === "ud-q2_k_xl" || q === "custom"));
}

function pickEffort(v) {
  document.getElementById("fEffort").value = v;
  syncEffortPills();
}

function syncEffortPills() {
  const cur = document.getElementById("fEffort").value.trim().toLowerCase();
  document.querySelectorAll("[data-effort]").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-effort") === cur);
  });
}

function syncUncensorUi() {
  const on = document.getElementById("fUncensor").checked;
  const sc = Number(document.getElementById("fUncensorScale").value);
  document.getElementById("fUncensorScale").disabled = !on;
  document.getElementById("uncensorReadout").textContent = on ? (sc.toFixed(2) + "×") : "off (original)";
}

function applyProfilePreset(name) {
  const presets = {
    "balanced":  { parallel: "2", routing: "balanced", reserve: 450 },
    "solo-fast": { parallel: "1", routing: "fast",     reserve: 400 },
    "exact":     { parallel: "2", routing: "exact",    reserve: 450 },
  };
  const p = presets[name];
  if (!p) return;
  document.getElementById("fParallel").value = p.parallel;
  document.getElementById("fRouting").value = p.routing;
  document.getElementById("fVramReserve").value = p.reserve;
  syncProfileHighlight();
}

function syncProfileHighlight() {
  const par = Number(document.getElementById("fParallel").value);
  const rt = document.getElementById("fRouting").value;
  const res = Number(document.getElementById("fVramReserve").value);
  const match = (par === 2 && rt === "balanced" && res === 450) ? "balanced"
              : (par === 1 && rt === "fast" && res === 400) ? "solo-fast"
              : (par === 2 && rt === "exact" && res === 450) ? "exact" : "";
  for (const id of ["balanced", "solo-fast", "exact"]) {
    const el = document.getElementById("prof-" + id);
    if (el) el.classList.toggle("active", id === match);
  }
}

async function loadConfig() {
  const r = await api("/_control/config");
  const d = await r.json();
  loadedConfig = d.config || null;
  document.getElementById("cfgText").value = d.toml || "";
  populateForm(loadedConfig);
  const toast = document.getElementById("cfgToast");
  toast.textContent = d.path || "";
  toast.className = "toast dim";
}

async function _postConfig(payload, doRestart) {
  const toast = document.getElementById("cfgToast");
  try {
    const r = await api("/_control/config", { method: "POST", body: JSON.stringify(payload) });
    const d = await r.json();
    if (!r.ok) { toast.textContent = "Error: " + (d.error?.message || r.status); toast.className = "toast err"; return; }
    if (d.config) { loadedConfig = d.config; populateForm(loadedConfig); }
    if (d.toml) document.getElementById("cfgText").value = d.toml;
    await loadClients();
    if (doRestart) {
      toast.textContent = "Saved. Restarting G4 with updated settings...";
      toast.className = "toast warn";
      await restartGpu();
      toast.textContent = "Saved and G4 restart triggered.";
      toast.className = "toast ok";
      return;
    }
    const msg = d.restart_required
      ? "Saved. Live settings applied; cold engine changes (" + d.cold_changed.join(", ") + ") need Restart G4."
      : "Saved and applied live.";
    toast.textContent = msg;
    toast.className = d.restart_required ? "toast warn" : "toast ok";
  } catch (e) {
    toast.textContent = "Save failed: " + e.message; toast.className = "toast err";
  }
}

async function saveVisualConfig(doRestart) {
  await _postConfig({ config: collectConfigFromForm() }, doRestart);
}

async function saveRawToml(doRestart) {
  await _postConfig({ toml: document.getElementById("cfgText").value }, doRestart);
}

/* ---------- Client Integration Tabs ---------- */
async function loadClients() {
  try {
    const r = await api("/_control/clients");
    clientsData = await r.json();
    const ep = clientsData.endpoint || {};
    document.getElementById("epBaseUrl").textContent = ep.openai_base_url || "—";
    document.getElementById("epModel").textContent = ep.model || "—";
    renderKeyMask();
    renderClientTabs();
  } catch (_) {}
}

function renderKeyMask() {
  const el = document.getElementById("epKeyMask");
  const btn = document.getElementById("btnRevealKey");
  if (keyRevealed) {
    el.textContent = apiKey || "—";
    btn.textContent = "Hide";
  } else {
    el.textContent = apiKey ? (apiKey.slice(0, 10) + "••••••••") : "—";
    btn.textContent = "Show";
  }
}

function toggleRevealKey() {
  keyRevealed = !keyRevealed;
  renderKeyMask();
}

function renderClientTabs() {
  const list = (clientsData && clientsData.clients) || [];
  const tabsEl = document.getElementById("clientTabs");
  tabsEl.innerHTML = list.map(c =>
    `<button class="sm ${c.id === activeClientId ? 'active' : ''}" onclick="selectClient('${esc(c.id)}')">${esc(c.label)}</button>`
  ).join("");
  activeClient = list.find(c => c.id === activeClientId) || list[0] || null;
  if (!activeClient) return;
  document.getElementById("clTitle").textContent = activeClient.label;
  document.getElementById("clSub").textContent = activeClient.subtitle || "";
  document.getElementById("clFileHint").textContent = activeClient.file_hint ? ("Target: " + activeClient.file_hint) : "";
  document.getElementById("clSnippet").textContent = activeClient.snippet || "";
  document.getElementById("clEnv").textContent = activeEnvText();
  document.getElementById("clNotes").textContent = activeClient.notes || "";
}

function selectClient(id) {
  activeClientId = id;
  renderClientTabs();
}

function setEnvShell(mode) {
  envShell = mode;
  document.getElementById("btnEnvSh").classList.toggle("active", mode === "sh");
  document.getElementById("btnEnvPs").classList.toggle("active", mode === "ps");
  document.getElementById("clEnv").textContent = activeEnvText();
}

function activeEnvText() {
  if (!activeClient) return "";
  return envShell === "ps" ? (activeClient.env_ps || "") : (activeClient.env_sh || "");
}

async function copyRaw(text, btn) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); document.body.removeChild(ta);
  }
  if (btn) {
    const orig = btn.textContent;
    btn.textContent = "Copied ✓";
    setTimeout(() => { btn.textContent = orig; }, 1400);
  }
}

function copyText(id, btn) {
  const el = document.getElementById(id);
  copyRaw(el ? el.textContent : "", btn);
}

/* ---------- Usage & Logs ---------- */
async function loadUsage() {
  const r = await api("/_control/usage?days=14");
  const u = await r.json();
  document.getElementById("kpiMonthCu").textContent = u.month_cu.toFixed(2) + " CU";
  document.getElementById("kpiTotals").textContent = `Today ${u.today_cu.toFixed(2)} · All-time ${u.total_cu.toFixed(2)} CU (${u.total_hours.toFixed(1)}h)`;
  document.getElementById("ledgerCount").textContent = `${u.sessions_count} sessions`;
  const days = u.daily || [];
  const maxCu = Math.max(0.5, ...days.map(d => d.cu));
  const w = 420, h = 110, pad = 14, bw = Math.max(8, Math.floor((w - pad * 2) / Math.max(1, days.length)) - 4);
  const bars = days.map((d, i) => {
    const bh = Math.max(2, Math.round((d.cu / maxCu) * (h - 30)));
    const x = pad + i * ((w - pad * 2) / Math.max(1, days.length));
    const y = h - 18 - bh;
    const lbl = d.date.slice(5);
    return `<g><title>${d.date}: ${d.cu.toFixed(2)} CU (${d.sessions} sessions)</title>`
         + `<rect x="${x.toFixed(1)}" y="${y}" width="${bw}" height="${bh}" rx="3" fill="${d.cu > 0 ? '#3fb950' : '#1f2737'}"/>`
         + (i % 3 === 0 ? `<text x="${(x + bw / 2).toFixed(1)}" y="${h - 4}" text-anchor="middle" fill="#8b949e" font-size="9">${lbl}</text>` : "")
         + `</g>`;
  }).join("");
  document.getElementById("cuChart").innerHTML = bars;
  const rows = (u.recent || []).slice(0, 8).map(e =>
    `<tr><td>${esc((e.started_at || "").replace("T", " ").slice(5, 16))}</td><td>${mmss(e.ran_s)}</td><td>${Number(e.cu_est || 0).toFixed(2)}</td><td>${esc(e.reason || "")}</td><td>${e.served || 0}</td></tr>`
  ).join("");
  document.getElementById("recentBody").innerHTML = rows || `<tr><td colspan="5" class="dim">No sessions recorded yet</td></tr>`;
}

async function loadLogs() {
  const vm = document.getElementById("vmLogChk").checked ? "1" : "0";
  const r = await api("/_control/logs?vm=" + vm);
  const d = await r.json();
  document.getElementById("logOut").textContent = (d.local || "(no local log)") + (d.vm ? ("\n\n--- VM ---\n" + d.vm) : "");
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

initAuth();
</script>
</body>
</html>
"""


def render_console_html() -> bytes:
    return CONSOLE_HTML.encode("utf-8")
