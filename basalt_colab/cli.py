"""CLI entrypoint for basalt-colab."""
from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import re
import secrets
import select
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from typing import Any, Callable

from basalt_colab import __version__
from basalt_colab.clients import (
    VALID_CLIENTS,
    ClientEndpoint,
    all_client_bundles,
    client_bundle,
    env_vars,
    format_env_lines,
    resolve_client,
)
from basalt_colab.colab import (
    Logger,
    alive,
    child_env,
    colab_cmd,
    colab_present,
    colab_usage,
    find_tool,
    http_call,
    kill_tree,
    load_session,
    missing_vm_files,
    read_api_key,
    read_bytes,
    redact,
    release_lock,
    run_cmd,
    serve_running,
    session_alive,
    spawn_detached,
    take_lock,
    vmx,
)
from basalt_colab.config import (
    HOT_FIELDS,
    Config,
    ConfigError,
    answer_room_value,
    context_value,
    dump_toml,
    effort_value,
    get_config_value,
    load_config,
    profile_value,
    quant_value,
    routing_mode_value,
    save_config,
    set_config_value,
    short_context,
    validate_config,
)
from basalt_colab.lifecycle import UPSTREAM_TIMEOUT, mmss, run_watchdog, stop_cloud
from basalt_colab.manifest import Manifest, load_manifest
from basalt_colab.notify import notify
from basalt_colab.paths import IS_WIN, Paths, get_paths, migrate_legacy_state
from basalt_colab.proxy import serve, write_opencode_config
from basalt_colab.usage import UsageLedger

USAGE = f"""basalt-colab {__version__} — DeepSeek-V4.1-Flash on Google Colab G4 via Basalt v0.5.7

usage: basalt-colab COMMAND [options]
  up | start [opts]                    start G4 (~1.5-2 min), print endpoints & live status
     -d, --detach, --no-watch          return immediately after G4 is ready (skip live watch view)
     --quant Q                         ud-iq1_m (default, 100% RAM) | ud-iq2_xxs (-13% KLD) | ud-q2_k_xl | custom
     --pack-url URL                    override .basalt URL (required for ud-q2_k_xl / custom)
     --profile P                       balanced (default: 2 slots, route balanced) | solo-fast (1 slot, route fast) | exact
     --routing-mode M                  balanced (default, KLD +0.0005, +20-37% speed) | exact (lossless) | fast
     --parallel N                      concurrent slots (1-4; default 2, use 1 for lowest solo latency)
     --uncensor-scale S                refusal-removal strength multiplier (0.0=off, 1.0=default, 1.15-1.25=stronger)
     --wait-gpu [MIN]                  retry G4 allocation until ready (0/omitted = unlimited, or cap at MIN minutes)
     --no-notify                       disable terminal bell and OS desktop notification on ready/failure
     --basalt-ref REF                  build an upstream Basalt tag/branch/SHA from source
  restart [up opts]                    stop current G4, reload saved config.toml, and boot fresh (keeps proxy)
  down | stop [--all]                  stop G4, wipe VM secrets, record CU ledger, stop detached proxy
  status                               Control Console & Web UI URLs, Basalt telemetry, billing ledger, Colab balance
  watch                                live terminal status bar (tok/s, VRAM hit %, idle countdown, CU cost)
  console [--no-open]                  ensure local proxy is running and open the Unified Control Console (/_console)
  config [subcmd]                      inspect or edit config.toml (hot-reloads live!)
     show | path | wizard              show TOML, print path, or run interactive wizard
     get <key>                         print value (e.g. quant, effort, idle_min, basalt.answer_room)
     set <key> <value>                 update key, save config.toml, and hot-reload running proxy
     profile <name>                    apply balanced | solo-fast | exact and save
  connect [CLIENT] [opts]              generate ready-to-paste config for dsh | claude | codex | opencode | openai
     --raw | --json | --sh | --powershell | --write (for opencode)
  dsh | claude | codex | opencode      shortcuts for `basalt-colab connect <client>` (pure config generation)
  env [--sh|--powershell|--json]       print live env exports (OPENAI_*, ANTHROPIC_*, BASALT_API_KEY)
  ask [opts] TEXT                      stream an answer (--effort VAL, --max, --no-think, --hide-think, --max-tokens N, --system S, --image PATH)
  chat [opts]                          interactive multi-turn terminal chat REPL (/effort, /clear, /image, /system, /exit)
  logs [-f|--follow] [--local|--vm]    local proxy log plus VM boot timeline, Basalt server & engine logs
  doctor                               diagnose Python, OpenSSH, Colab CLI, OAuth balance, config.toml, VM payload & port
  setup                                install Colab CLI, complete Google OAuth, create keys & migrate legacy state/
  serve [--detached]                   run the local proxy in the foreground (Ctrl+C stops the GPU too)

env: BASALT_COLAB_HOME, BASALT_COLAB_PORT, BASALT_COLAB_BIND, BASALT_COLAB_QUANT, BASALT_COLAB_PACK_URL,
     BASALT_COLAB_PROFILE (balanced|solo-fast|exact), BASALT_COLAB_ROUTING_MODE (balanced|exact|fast),
     BASALT_COLAB_ANSWER_ROOM (auto|0..65536), BASALT_COLAB_UNCENSOR, BASALT_COLAB_UNCENSOR_SCALE,
     BASALT_COLAB_IDLE_MIN, BASALT_COLAB_MAX_HOURS, BASALT_COLAB_WAIT_GPU, BASALT_COLAB_WAIT_GPU_MIN,
     BASALT_COLAB_NO_NOTIFY, BASALT_COLAB_REF, BASALT_COLAB_MAX_CONTEXT (32K..1M), BASALT_COLAB_PARALLEL (1-4),
     BASALT_COLAB_KV (fp8|fp16), BASALT_COLAB_EFFORT, BASALT_COLAB_STRICT_MODEL, HF_TOKEN, BASALT_COLAB_YES=1"""

TTY, COLOR = [False], [False]


def init_console() -> None:
    TTY[0] = bool(sys.stdout and sys.stdout.isatty())
    if not TTY[0] or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return
    if IS_WIN:
        try:
            import ctypes

            k32, mode = ctypes.windll.kernel32, ctypes.c_uint32()
            h = k32.GetStdHandle(-11)
            if not (k32.GetConsoleMode(h, ctypes.byref(mode)) and k32.SetConsoleMode(h, mode.value | 4)):
                return
        except (AttributeError, OSError):
            return
    COLOR[0] = True


def paint(text: str, *styles: str) -> str:
    codes = {"bold": "1", "dim": "2", "red": "31", "green": "32", "cyan": "36", "yellow": "33"}
    return f"\x1b[{';'.join(codes[s] for s in styles)}m{text}\x1b[0m" if COLOR[0] else text


def glyph(fancy: str, plain: str) -> str:
    utf = "utf" in (getattr(sys.stdout, "encoding", None) or "").lower()
    return fancy if utf and (not IS_WIN or os.environ.get("WT_SESSION")) else plain


def units(cu: float) -> str:
    return f"about {cu:.2f} compute units" if cu < 1 else f"about {cu:.1f} compute units"


def minutes(n: float) -> str:
    return f"{n:g} minute{'' if n == 1 else 's'}"


def cli_name() -> str:
    return "basalt-colab" if shutil.which("basalt-colab") else "python3 -m basalt_colab"


class Live:
    def __init__(self) -> None:
        self.width, self.n = 0, 0

    def clear(self) -> None:
        if self.width:
            sys.stdout.write("\r" + " " * self.width + "\r")
            self.width = 0

    def put(self, parts: list[tuple[str, tuple[str, ...]]]) -> None:
        if not TTY[0]:
            return
        cols = shutil.get_terminal_size((80, 24)).columns - 1
        plain = "".join(t for t, _ in parts)
        self.clear()
        sys.stdout.write(plain[:cols] if len(plain) > cols else "".join(paint(t, *st) for t, st in parts))
        sys.stdout.flush()
        self.width = min(len(plain), cols)

    def show(self, when: str, text: str) -> None:
        spin = glyph("⣾⣽⣻⢿⡿⣟⣯⣷", "|/-\\")
        self.n += 1
        self.put([("  ", ()), (f"{when:>5}", ("dim",)), ("  ", ()), (spin[self.n % len(spin)], ("cyan",)), ("  " + text, ())])

    def line(self, text: str) -> None:
        self.clear()
        print(text, flush=True)


def control_call(
    paths: Paths, cfg: Config, port: int, path: str, body: dict[str, Any] | None, timeout: float
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    data = None if body is None else json.dumps(body).encode()
    return http_call(
        port,
        "GET" if body is None else "POST",
        path,
        data,
        {"Authorization": "Bearer " + read_api_key(paths), "Content-Type": "application/json"},
        timeout,
        cfg.proxy.host,
    )


def fetch_live_status(paths: Paths, cfg: Config, port: int, timeout: float = 2.5) -> dict[str, Any] | None:
    try:
        conn, r = control_call(paths, cfg, port, "/_control/status", None, timeout)
        data = json.loads(r.read()) if r.status == 200 else None
        conn.close()
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, http.client.HTTPException):
        return None


def fetch_live_endpoint(paths: Paths, cfg: Config) -> tuple[ClientEndpoint, bool]:
    """Builds ClientEndpoint synced with the running proxy when active, else falls back to config.toml."""
    if not paths.key_file.exists():
        sys.exit(f"missing {paths.key_file}; run: {cli_name()} setup")
    held = serve_running(paths)
    port = int(held["port"]) if held else cfg.proxy.port
    live = fetch_live_status(paths, cfg, port) if held else None
    ep = ClientEndpoint.from_config(cfg, read_api_key(paths), port=port, live_status=live, paths=paths)
    return ep, bool(held)


def follow(resp: http.client.HTTPResponse, header: str | None = None) -> tuple[str, str]:
    live = Live()
    for raw in resp:
        kind, _, rest = raw.decode("utf-8", "replace").strip().partition(" ")
        if kind in ("step", "wait"):
            if header:
                print(header + "\n", flush=True)
                header = None
            when, _, text = rest.partition(" ")
            if kind == "step":
                live.line(f"  {paint(f'{mmss(float(when)):>5}', 'dim')}  {paint(glyph('✓', '+'), 'green')}  {text}")
            else:
                live.show(mmss(float(when)), text)
        elif kind:
            live.clear()
            return kind, rest
    live.clear()
    return "", ""


def confirm(question: str) -> bool:
    if os.environ.get("BASALT_COLAB_YES") == "1" or os.environ.get("LLM_YES") == "1":
        return True
    if not (sys.stdin and sys.stdin.isatty()):
        return False
    try:
        return input(question + " [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def install_colab(manifest: Manifest) -> bool:
    home_bin = os.path.join(os.path.expanduser("~"), ".local", "bin")
    uv = shutil.which("uv") or next((p for p in (os.path.join(home_bin, "uv.exe" if IS_WIN else "uv"),) if os.path.exists(p)), None)
    pipx = shutil.which("pipx")
    tool = "uv" if uv or not pipx else "pipx"
    print("Google's Colab CLI is not installed yet. basalt-colab uses it to start and stop the G4 GPU.", flush=True)
    if not confirm(f"Install {manifest.colab_cli} with {tool}{' (installs uv first)' if tool == 'uv' and not uv else ''}?"):
        return False
    print("Installing...", end=" ", flush=True)
    if tool == "uv" and not uv:
        rc, out = run_cmd(
            ["powershell", "-ExecutionPolicy", "ByPass", "-c", "irm https://astral.sh/uv/install.ps1 | iex"]
            if IS_WIN
            else ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            900,
        )
        if rc != 0:
            print("failed:\n" + out[-2000:], flush=True)
            return False
        uv = shutil.which("uv") or os.path.join(home_bin, "uv.exe" if IS_WIN else "uv")
    assert uv or pipx
    rc, out = run_cmd([uv, "tool", "install", manifest.colab_cli] if tool == "uv" and uv else [pipx or "pipx", "install", manifest.colab_cli], 900)
    print("done." if rc == 0 else "failed:\n" + out[-2000:], flush=True)
    return rc == 0


def colab_login() -> None:
    hide = (b"Current balance", b"Usage rate", b"Active assignments")
    p = subprocess.Popen(colab_cmd() + ["usage"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env({"PYTHONUNBUFFERED": "1"}))
    assert p.stdout is not None
    buf = b""
    while True:
        chunk = os.read(p.stdout.fileno(), 4096)
        if not chunk:
            break
        *lines, buf = (buf + chunk).split(b"\n")
        keep = [l + b"\n" for l in lines if not l.lstrip().startswith(hide)]
        if buf and not any(h.startswith(buf.lstrip()) for h in hide):
            keep, buf = keep + [buf], b""
        sys.stdout.flush()
        sys.stdout.buffer.write(b"".join(keep))
        sys.stdout.buffer.flush()
    p.wait()


def ensure_setup(paths: Paths, cfg: Config, manifest: Manifest) -> float:
    paths.ensure_dirs()
    migrated = migrate_legacy_state(paths)
    if migrated:
        print(paint(f"  Migrated legacy credentials into {paths.state_dir}: {', '.join(migrated)}", "dim"), flush=True)
    missing = [t for t in ("ssh", "ssh-keygen") if not (shutil.which(t) or os.path.exists(find_tool(t)))]
    if missing:
        sys.exit(f"Missing {' and '.join(missing)}. Install an OpenSSH client and run again.")
    if not colab_present():
        if not install_colab(manifest):
            sys.exit(f'The Colab CLI is needed. Install it with "uv tool install {manifest.colab_cli}" (or pipx), then run again.')
        if not colab_present():
            sys.exit('Installed, but "colab" is not on PATH yet. Open a new terminal and run again.')
    bal, _ = colab_usage()
    if bal is None:
        if not (sys.stdin and sys.stdin.isatty()):
            sys.exit(f"The Colab CLI is installed but not logged in yet. Run in a terminal: {cli_name()} setup")
        print(f"\n{paint('Google login (only once)', 'bold')}: open the link below, sign in and paste the code here.", flush=True)
        colab_login()
        bal, _ = colab_usage()
        if bal is None:
            sys.exit(f"The login did not work. Try again: {cli_name()} setup")
        print(f"{paint('Logged in.', 'bold', 'green')} Colab balance: {bal:.1f} compute units.\n", flush=True)
    if not paths.key_file.exists():
        fd = os.open(str(paths.key_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(("sk-ds41-" + secrets.token_hex(24) + "\n").encode())
    if not paths.ssh_key.exists():
        rc, out = run_cmd([find_tool("ssh-keygen"), "-q", "-t", "ed25519", "-N", "", "-C", "basalt-colab", "-f", str(paths.ssh_key)], 30)
        if rc:
            sys.exit("ssh-keygen failed: " + out)
    if not paths.config_file.exists():
        save_config(cfg, paths)
    write_opencode_config(paths, cfg, cfg.proxy.port)
    return bal


def ensure_detached_proxy(paths: Paths) -> dict[str, Any]:
    held = serve_running(paths)
    if held:
        return held
    paths.ensure_dirs()
    extra = {"BASALT_COLAB_HOME": str(paths.config_dir)} if os.environ.get("BASALT_COLAB_HOME") else None
    with paths.serve_out_file.open("wb") as out:
        p = spawn_detached([sys.executable, "-m", "basalt_colab", "serve", "--detached"], out, extra_env=extra)
    t0 = time.time()
    while not held:
        if p.poll() is not None or time.time() - t0 > 30:
            raw = (read_bytes(paths.serve_out_file) or b"").decode("utf-8", "replace")
            sys.exit(f"The local endpoint did not start: {raw}")
        time.sleep(0.3)
        held = serve_running(paths)
    return held


def format_telemetry(b: dict[str, Any] | None) -> str:
    if not isinstance(b, dict):
        return ""
    parts: list[str] = []
    if b.get("busy") and b.get("live_tok_s") is not None:
        parts.append(f"{b['live_tok_s']:.1f} tok/s")
    elif b.get("last_tok_s") is not None:
        parts.append(f"{b['last_tok_s']:.1f} tok/s")
    if b.get("hit_rate") is not None:
        parts.append(f"hit {b['hit_rate'] * 100:.0f}%")
    if b.get("pcie_share") is not None:
        parts.append(f"pcie {b['pcie_share'] * 100:.1f}%")
    return " · ".join(parts)


def endpoint_info(paths: Paths, cfg: Config, manifest: Manifest, port: int, bal: float | None = None) -> None:
    me_cmd = cli_name()
    host = cfg.proxy.host
    rows = (
        ("Console", f"http://{host}:{port}/_console"),
        ("Web UI", f"http://{host}:{port}/"),
        ("Base URL", f"http://{host}:{port}/v1"),
        ("API key", read_api_key(paths)),
        ("Model", f"{cfg.basalt.effective_model_alias} (effort {cfg.basalt.effort}, {cfg.describe_uncensor()})"),
        ("Engine", f"{cfg.describe_engine()} · ~{(manifest.pack(cfg.basalt.quant).estimated_speed_tok_s if manifest.pack(cfg.basalt.quant) else manifest.estimated_speed_tok_s):.0f} tok/s"),
    )
    print()
    for name, value in rows:
        print(f"  {name:<10} {paint(value, 'bold')}")
    print()
    for name, value in (
        ("Try it", f'{me_cmd} ask "Hello!"   /   {me_cmd} chat'),
        ("Console", f"{me_cmd} console"),
        ("Agents", f"{me_cmd} connect [dsh|claude|codex|opencode|openai]"),
        ("Shell env", f'eval "$({me_cmd} env)"'),
        ("Stop", f"{me_cmd} down"),
    ):
        print(f"  {name:<10} {paint(value, 'cyan')}")
    bal_str = f" · balance {bal:.1f}" if bal is not None else ""
    print("\n" + paint(f"  About {manifest.g4_cu_per_hour:.0f} compute units/hr{bal_str} · stops after {minutes(cfg.colab.idle_min)} idle", "dim"))


def stopped_note(kind: str, rest: str) -> None:
    if kind != "STOPPED":
        sys.exit(f"The GPU may still be running. Run this again: {cli_name()} down")
    cu, ran = (rest.split() + ["None", "None"])[:2]
    done = paint(glyph("✓", "+") + " Stopped", "bold", "green")
    if cu != "None" and ran != "None":
        print(f"{done} · the GPU ran {mmss(float(ran))} · {units(float(cu))}")
    else:
        print(f"{done} · nothing is running or billed now")


def watch_loop(paths: Paths, cfg: Config, port: int) -> None:
    if not TTY[0]:
        return
    print(paint("\n  Ctrl+C closes this view; the endpoint keeps running.\n", "dim"))
    live, prev, dots = Live(), None, (glyph("●", "*"), glyph("◌", "~"), glyph("○", "o"))
    try:
        while True:
            try:
                conn, r = control_call(paths, cfg, port, "/_control/status", None, 5)
                st = json.loads(r.read())
                conn.close()
            except (OSError, ValueError, http.client.HTTPException):
                live.line(f"  {paint(time.strftime('%H:%M'), 'dim')}  the endpoint has stopped")
                return
            state = "ready" if st["state"].startswith("ready") else st["state"]
            if prev and state != prev:
                live.line(f"  {paint(time.strftime('%H:%M'), 'dim')}  {watch_event(prev, state, st)}")
            prev = state
            spent = f"ran {mmss(st['ran'])} · {units(st['cu'])}" if st.get("ran") is not None else ""
            if state == "ready":
                left = min(st["idle_stop_min"] * 60 - st["idle_s"], st["max_hours"] * 3600 - (st["ran"] or 0))
                doing = (
                    f"answering {st['active']} request{'' if st['active'] == 1 else 's'}"
                    if st["active"]
                    else f"idle {mmss(st['idle_s'])} · stops in {mmss(left)}"
                )
                perf = format_telemetry(st.get("basalt"))
                tail = " · ".join(p for p in (perf, spent) if p)
                live.put([("  ", ()), (dots[0] + " GPU running", ("green", "bold")), ("   " + doing + "   ", ()), (tail, ("dim",))])
            elif state in ("starting", "stopping"):
                what = "starting · " + st["stage"] if state == "starting" else "stopping"
                live.put([("  ", ()), (dots[1] + " GPU " + what, ("cyan",)), ("   " + spent, ("dim",))])
            else:
                live.put([("  ", ()), (dots[2] + " GPU off", ("bold",)), ("   starts on the next request (~2-3 min)", ("dim",))])
            time.sleep(1)
    except KeyboardInterrupt:
        live.clear()
        me_cmd = cli_name()
        print(
            f"\n  Closed live view; endpoint keeps running.\n"
            f"  {paint('Live view', 'dim')} {paint(me_cmd + ' watch', 'cyan')}   "
            f"{paint('Console', 'dim')} {paint(me_cmd + ' console', 'cyan')}   "
            f"{paint('Stop', 'dim')} {paint(me_cmd + ' down', 'cyan')}"
        )


def watch_event(prev: str, state: str, st: dict[str, Any]) -> str:
    stop = st.get("last_stop") or {}
    if state in ("stopping", "off") and prev in ("ready", "starting"):
        if st.get("last_error") and prev == "starting":
            return paint("start failed: " + st["last_error"], "red")
        cost = f": ran {mmss(stop['ran'])}, {units(stop['cu'])}" if stop.get("ran") is not None else ""
        why = (
            f"after {minutes(st['idle_stop_min'])} without requests"
            if stop.get("reason", "").startswith("idle")
            else stop.get("reason", "")
        )
        return f"GPU stopped {why}{cost}"
    if state == "starting":
        return "GPU starting for a new request"
    if state == "ready":
        return paint("GPU ready", "green")
    return "GPU " + state


def pick(label: str, hint: str, default: Any, parse: Callable[[str], Any]) -> Any:
    d_str = f"[{default}]"
    row = (
        f"  {paint(f'{label:<14}', 'bold')}{paint(f'{hint:<30}', 'dim')}"
        f"{paint(f'{d_str:>10}', 'cyan')} {paint(glyph('›', '>'), 'cyan')} "
    )
    while True:
        try:
            text = input(row).strip()
        except EOFError:
            text = ""
        val = parse(text or str(default))
        if val is not None:
            return val
        print(f"  {' ' * 14}{paint('not valid, try again', 'red')}")


def _parse_num(text: str, low: float, high: float) -> int | float | None:
    try:
        v = float(text)
    except ValueError:
        return None
    return (int(v) if v == int(v) else v) if low <= v <= high else None


def _parse_yes_no(text: str) -> bool | None:
    return {"y": True, "yes": True, "true": True, "n": False, "no": False, "false": False}.get(text.lower())


def interactive_config_wizard(paths: Paths, cfg: Config) -> Config | None:
    if not (sys.stdin and sys.stdin.isatty()):
        return None
    print(f"{paint('Configuration', 'bold')}  {paint('Enter keeps the value in brackets', 'dim')}\n")
    new = cfg.clone()
    new.basalt.quant = pick("Quant pack", "ud-iq1_m | ud-iq2_xxs | ud-q2_k_xl", cfg.basalt.quant, quant_value)
    if new.basalt.quant in ("ud-q2_k_xl", "custom"):
        new.basalt.pack_url = pick(
            "Pack URL",
            "https://.../model.basalt",
            cfg.basalt.pack_url or "",
            lambda t: t.strip() if re.match(r"^https?://", t.strip()) and "'" not in t else None,
        )
    else:
        new.basalt.pack_url = ""
    new.basalt.max_context = pick("Context/slot", "32K..1M (slot-full-context)", short_context(cfg.basalt.max_context), context_value)
    new.basalt.parallel = int(pick("Parallel", "1-4 concurrent slots", cfg.basalt.parallel, lambda t: int(t) if t.isdigit() and 1 <= int(t) <= 4 else None))
    new.basalt.routing_mode = pick("Routing", "balanced | fast | exact", cfg.basalt.routing_mode, routing_mode_value)
    new.basalt.kv = pick("KV precision", "fp8 (0.97G@1M) or fp16", cfg.basalt.kv, lambda t: t.lower() if t.lower() in ("fp8", "fp16") else None)
    new.basalt.answer_room = pick("Answer room", "auto (anti-trunc) or 0..65536", cfg.basalt.answer_room, answer_room_value)
    new.basalt.effort = pick("Effort", "1-100, low, high, max (live)", cfg.basalt.effort, effort_value)
    new.colab.idle_min = float(pick("Idle stop", "minutes 1-120 (live)", f"{cfg.colab.idle_min:g}", lambda t: _parse_num(t, 1, 120)))
    new.colab.max_hours = float(pick("Hard stop", "hours 1-24 (live)", f"{cfg.colab.max_hours:g}", lambda t: _parse_num(t, 1, 24)))
    new.basalt.uncensor = bool(pick("Uncensored", "43KB mrfakename rank-2, y/n", "y" if cfg.basalt.uncensor else "n", _parse_yes_no))
    if new.basalt.uncensor:
        new.basalt.uncensor_scale = float(pick("Uncensor scale", "0.5-2.0 (1.0=std, 1.2=stronger)", f"{cfg.basalt.uncensor_scale:g}", lambda t: _parse_num(t, 0.0, 2.0)))
    new.basalt.ref = pick("Basalt ref", "blank=pinned, or tag/branch", cfg.basalt.ref or "-", lambda t: "" if t in ("-", "pinned", "") else t)
    save_config(new, paths)
    print(f"\n  {paint(glyph('✓', '+'), 'green')} Saved {paint(str(paths.config_file), 'dim')}\n")
    return new


def cmd_setup(paths: Paths, cfg: Config, manifest: Manifest) -> None:
    bal = ensure_setup(paths, cfg, manifest)
    print(f"Setup done. Config: {paths.config_file}")
    print(f"Colab balance: {bal:.1f} compute units (~{bal / manifest.g4_cu_per_hour:.0f} GPU hours).")
    print(f"Next: {cli_name()} up   (or {cli_name()} console)")


def _parse_wait_gpu_args(args: list[str]) -> tuple[bool, float | None]:
    wait_gpu = False
    wait_gpu_min: float | None = None
    for i, a in enumerate(args):
        if a == "--wait-gpu":
            wait_gpu = True
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                try:
                    wait_gpu_min = float(args[i + 1])
                except ValueError:
                    sys.exit(f"invalid --wait-gpu minutes {args[i + 1]!r} (expected number >= 0)")
                if wait_gpu_min < 0 or wait_gpu_min > 1440:
                    sys.exit("--wait-gpu minutes must be 0 (unlimited) or 0.1..1440")
        elif a.startswith("--wait-gpu="):
            wait_gpu = True
            raw = a.split("=", 1)[1]
            try:
                wait_gpu_min = float(raw)
            except ValueError:
                sys.exit(f"invalid --wait-gpu minutes {raw!r} (expected number >= 0)")
            if wait_gpu_min < 0 or wait_gpu_min > 1440:
                sys.exit("--wait-gpu minutes must be 0 (unlimited) or 0.1..1440")
    return wait_gpu, wait_gpu_min


def _build_up_opts(
    cfg: Config,
    manifest: Manifest,
    *,
    wait_gpu: bool,
    wait_gpu_min: float | None,
    basalt_ref: str | None,
    quant: str | None,
    pack_url: str | None,
    uncensor_scale: float | None,
    profile: str | None,
    routing_mode: str | None,
    parallel: int | None,
) -> tuple[Config, dict[str, Any]]:
    want_cfg = cfg.clone()
    if profile is not None:
        p = profile_value(profile)
        if not p:
            sys.exit(f"invalid --profile {profile!r} (expected balanced, solo-fast, or exact)")
        want_cfg.apply_profile(p)
        profile = p
    if basalt_ref is not None:
        want_cfg.basalt.ref = basalt_ref
    if quant is not None:
        q = quant_value(quant)
        if not q:
            sys.exit(f"invalid --quant {quant!r} (expected ud-iq1_m, ud-iq2_xxs, ud-q2_k_xl, or custom)")
        want_cfg.basalt.quant = q
    if pack_url is not None:
        want_cfg.basalt.pack_url = pack_url
    if uncensor_scale is not None:
        want_cfg.basalt.uncensor_scale = uncensor_scale
        want_cfg.basalt.uncensor = uncensor_scale > 0.0
    if routing_mode is not None:
        rt = routing_mode_value(routing_mode)
        if not rt:
            sys.exit(f"invalid --routing-mode {routing_mode!r} (expected balanced, exact, or fast)")
        want_cfg.basalt.routing_mode = rt
    if parallel is not None:
        want_cfg.basalt.parallel = parallel
    try:
        validate_config(want_cfg)
        want_cfg.boot_vars(manifest=manifest)
    except ConfigError as e:
        sys.exit(f"config error: {e}")

    up_opts: dict[str, Any] = {}
    if wait_gpu:
        up_opts["wait_gpu"] = True
    if wait_gpu_min is not None:
        up_opts["wait_gpu_min"] = wait_gpu_min
    if basalt_ref is not None:
        up_opts["basalt_ref"] = basalt_ref
    if profile is not None:
        up_opts["profile"] = profile
    if quant is not None:
        up_opts["quant"] = want_cfg.basalt.quant
    if pack_url is not None:
        up_opts["pack_url"] = want_cfg.basalt.pack_url
    if uncensor_scale is not None:
        up_opts["uncensor_scale"] = want_cfg.basalt.uncensor_scale
    if routing_mode is not None:
        up_opts["routing_mode"] = want_cfg.basalt.routing_mode
    if parallel is not None:
        up_opts["parallel"] = want_cfg.basalt.parallel
    return want_cfg, up_opts


def cmd_up(
    paths: Paths,
    cfg: Config,
    manifest: Manifest,
    wait_gpu: bool,
    basalt_ref: str | None,
    quant: str | None = None,
    pack_url: str | None = None,
    uncensor_scale: float | None = None,
    profile: str | None = None,
    routing_mode: str | None = None,
    parallel: int | None = None,
    wait_gpu_min: float | None = None,
    detach: bool = False,
    do_notify: bool = True,
    restart: bool = False,
) -> None:
    want_cfg, up_opts = _build_up_opts(
        cfg,
        manifest,
        wait_gpu=wait_gpu,
        wait_gpu_min=wait_gpu_min,
        basalt_ref=basalt_ref,
        quant=quant,
        pack_url=pack_url,
        uncensor_scale=uncensor_scale,
        profile=profile,
        routing_mode=routing_mode,
        parallel=parallel,
    )
    print(paint(f"basalt-colab {__version__} · {want_cfg.basalt.effective_model_alias} (Basalt {want_cfg.basalt.ref or manifest.basalt_tag})", "bold") + "\n")
    bal = ensure_setup(paths, want_cfg, manifest)
    held = serve_running(paths)
    if not restart and held and held.get("cold_signature") and held["cold_signature"] != want_cfg.cold_signature():
        have = held.get("engine_summary") or "different engine settings"
        sys.exit(f"The endpoint runs with {have}. To apply new engine settings, run: {cli_name()} restart")
    if bal < want_cfg.colab.min_balance and not (held and load_session(paths)):
        sys.exit(f"Colab balance {bal:.2f} compute units is below min_balance {want_cfg.colab.min_balance:.0f}")
    if not held:
        held = ensure_detached_proxy(paths)
    endpoint_path = "/_control/restart" if restart else "/_control/up"
    header = (
        paint("Restarting Colab G4 GPU", "bold")
        if restart
        else paint("Starting a Colab G4 GPU", "bold")
    ) + paint(" · ~1.5-2 minutes (prebuilt binary)", "dim")
    conn, r = control_call(paths, want_cfg, held["port"], endpoint_path, up_opts, 120)
    kind, rest = follow(r, header)
    conn.close()
    if kind != "READY":
        err_msg = rest or "the local endpoint closed the connection"
        notify("basalt-colab: G4 Start Failed", err_msg[:140], enabled=do_notify)
        print(paint("Start failed: " + err_msg, "red"))
        sys.exit(f"Details: {cli_name()} logs")
    secs = int(rest or 0)
    ready_line = f"{glyph('●', '*')} The GPU is already running" if secs < 2 else f"\n{glyph('✓', '+')} Ready after {mmss(secs)}"
    print(paint(ready_line, "bold", "green"))
    if secs >= 2:
        notify("basalt-colab: G4 Ready", f"{want_cfg.basalt.effective_model_alias} ready after {mmss(secs)}", enabled=do_notify)
    endpoint_info(paths, want_cfg, manifest, held["port"], bal)
    if not detach:
        watch_loop(paths, want_cfg, held["port"])


def cmd_down(paths: Paths, cfg: Config, manifest: Manifest, everything: bool = False) -> None:
    held = serve_running(paths)
    if held:
        leave = everything or bool(held.get("detached"))
        try:
            conn, r = control_call(paths, cfg, held["port"], "/_control/down", {"exit": leave}, 120)
            kind, rest = follow(r)
            conn.close()
            t0 = time.time()
            while leave and alive(held) and time.time() - t0 < 30:
                time.sleep(0.3)
            if leave and alive(held):
                kill_tree(int(held["pid"]), hard=True)
            return stopped_note(kind, rest)
        except (OSError, http.client.HTTPException) as e:
            Logger(paths).log(f"endpoint pid {held['pid']} did not answer ({e.__class__.__name__}); killing and stopping directly")
            kill_tree(int(held["pid"]), hard=True)
            time.sleep(1)
    held = take_lock(paths, {"port": 0})
    if held:
        sys.exit(f"Another basalt-colab command holds the lock (pid {held.get('pid')}).")
    try:
        still, cu, ran = stop_cloud(
            paths, cfg, manifest, UsageLedger(paths.usage_file), Logger(paths), load_session(paths), "down"
        )
    finally:
        release_lock(paths)
    stopped_note("STILL" if still else "STOPPED", f"{cu} {ran}")


def cmd_status(paths: Paths, cfg: Config) -> None:
    held = serve_running(paths)
    rows: list[tuple[str, str]] = []
    if held:
        try:
            conn, r = control_call(paths, cfg, held["port"], "/_control/status", None, 10)
            st = json.loads(r.read())
            conn.close()
            gpu = {"off": "off, starts on the next request", "starting": "starting: " + st["stage"]}.get(st["state"], st["state"])
            if st["state"].startswith("ready"):
                gpu = (
                    f"{st['state']}, idle {mmss(st['idle_s'])} of {minutes(st['idle_stop_min'])}, "
                    f"{st['served']} request{'' if st['served'] == 1 else 's'} served, "
                    f"ran {mmss(st['ran'] or 0)}, {units(st['cu'] or 0)}"
                )
            rows += [
                ("Console", f"http://{cfg.proxy.host}:{held['port']}/_console"),
                ("Web UI", f"http://{cfg.proxy.host}:{held['port']}/"),
                ("Endpoint", f"http://{cfg.proxy.host}:{held['port']}/v1 · {st.get('engine_summary') or ''}"),
                ("Model", f"{st.get('model') or cfg.basalt.effective_model_alias} (effort {st.get('effort', cfg.basalt.effort)}, {st.get('uncensor', cfg.describe_uncensor())}, ref {st.get('basalt_ref', 'pinned')})"),
                ("GPU", gpu),
            ]
            perf = format_telemetry(st.get("basalt"))
            if perf:
                rows.append(("Telemetry", perf))
            if st.get("last_error"):
                rows.append(("Last error", st["last_error"]))
        except (OSError, ValueError, KeyError, http.client.HTTPException) as e:
            rows.append(("Endpoint", f"pid {held['pid']} not answering ({e.__class__.__name__})"))
    else:
        rows.append(("Endpoint", f"not running (start it: {cli_name()} up)"))
    ledger = UsageLedger(paths.usage_file).summary(days=30)
    if ledger["sessions_count"]:
        rows.append(("Ledger", f"today {ledger['today_cu']:.2f} CU · month {ledger['month_cu']:.2f} CU · {ledger['sessions_count']} sessions"))
    bal, _ = colab_usage()
    rows.append((
        "Colab",
        f"session {cfg.colab.session} {'running' if session_alive(cfg.colab.session) else 'not running'}, "
        f"balance {f'{bal:.1f}' if bal is not None else 'unknown'} compute units",
    ))
    if held and paths.key_file.exists():
        rows.append(("API key", read_api_key(paths)))
    for name, value in rows:
        print(f"  {name:<10} {value}")


def cmd_console(paths: Paths, cfg: Config, manifest: Manifest, no_open: bool = False) -> None:
    if not paths.key_file.exists():
        ensure_setup(paths, cfg, manifest)
    held = ensure_detached_proxy(paths)
    url = f"http://{cfg.proxy.host}:{held['port']}/_console#key={urllib.parse.quote(read_api_key(paths))}"
    clean_url = f"http://{cfg.proxy.host}:{held['port']}/_console"
    print(f"  {paint('Console', 'bold')}  {clean_url}")
    if not no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass


def _reload_proxy_if_running(paths: Paths, cfg: Config) -> None:
    held = serve_running(paths)
    if not held:
        return
    try:
        conn, r = control_call(paths, cfg, held["port"], "/_control/reload", {}, 10)
        res = json.loads(r.read())
        conn.close()
        ap = res.get("applied") or {}
        print(f"  {paint(glyph('✓', '+'), 'green')} Applied live (effort {ap.get('effort')} · idle {ap.get('idle_min')} min · cap {ap.get('max_hours')} h)")
        if res.get("restart_required"):
            print(paint(f"  Cold engine changes ({', '.join(res.get('cold_changed', []))}) apply on restart: {cli_name()} restart", "dim"))
    except (OSError, ValueError, http.client.HTTPException) as e:
        print(paint(f"  Saved to disk, but live reload failed ({e}).", "red"))


def cmd_config(paths: Paths, cfg: Config, args: list[str]) -> None:
    sub = args[0] if args else ("wizard" if (sys.stdin and sys.stdin.isatty()) else "show")
    if sub == "path":
        print(paths.config_file)
        return
    if sub == "show":
        print(dump_toml(cfg), end="")
        return
    if sub == "get":
        if len(args) < 2:
            sys.exit(f"usage: {cli_name()} config get <key>")
        try:
            canonical, val = get_config_value(cfg, args[1])
        except ConfigError as e:
            sys.exit(f"config error: {e}")
        print(f"{canonical} = {val}")
        return
    if sub == "set":
        if len(args) < 3:
            sys.exit(f"usage: {cli_name()} config set <key> <value>")
        disk_cfg = load_config(paths, apply_env=False)
        try:
            updated, canonical, val = set_config_value(disk_cfg, args[1], " ".join(args[2:]))
            save_config(updated, paths)
        except ConfigError as e:
            sys.exit(f"config error: {e}")
        tag = "live" if canonical in HOT_FIELDS else "restart required if G4 running"
        print(f"  {paint(glyph('✓', '+'), 'green')} {canonical} = {val}  {paint(f'({tag})', 'dim')}")
        _reload_proxy_if_running(paths, updated)
        return
    if sub == "profile":
        if len(args) < 2:
            sys.exit(f"usage: {cli_name()} config profile <balanced|solo-fast|exact>")
        disk_cfg = load_config(paths, apply_env=False)
        try:
            disk_cfg.apply_profile(args[1])
            save_config(disk_cfg, paths)
        except ConfigError as e:
            sys.exit(f"config error: {e}")
        print(f"  {paint(glyph('✓', '+'), 'green')} Applied profile {args[1]} -> {disk_cfg.describe_engine()}")
        _reload_proxy_if_running(paths, disk_cfg)
        return
    if sub != "wizard":
        sys.exit(f"unknown config subcommand {sub!r} (use: show, path, wizard, get, set, profile)")
    if not (sys.stdin and sys.stdin.isatty()):
        sys.exit(f"Config: {cfg.summary()}\nEdit {paths.config_file} or run '{cli_name()} config' in a terminal.")
    updated = interactive_config_wizard(paths, cfg)
    if updated:
        _reload_proxy_if_running(paths, updated)


def cmd_logs(paths: Paths, cfg: Config, args: list[str] | None = None) -> None:
    opts = args or []
    follow_mode = "-f" in opts or "--follow" in opts
    only_local = "--local" in opts
    only_vm = "--vm" in opts
    key = read_api_key(paths) if paths.key_file.exists() else ""

    if not only_vm:
        raw = read_bytes(paths.log_file)
        print(redact(raw.decode("utf-8", "replace")[-4000:], key) if raw is not None else "(no local log)")
    if not only_local and not follow_mode and session_alive(cfg.colab.session):
        _, out = vmx(
            cfg.colab.session,
            "cat /content/res/timeline.txt 2>/dev/null; tail -n 25 /content/res/server.log 2>/dev/null; "
            "tail -n 15 /root/.cache/basalt/*/engine.log 2>/dev/null || true",
            40,
        )
        print(redact(out, key))
    if follow_mode:
        print(paint("--- following local log (Ctrl+C to stop) ---", "dim"), flush=True)
        pos = paths.log_file.stat().st_size if paths.log_file.exists() else 0
        try:
            while True:
                if paths.log_file.exists():
                    size = paths.log_file.stat().st_size
                    if size < pos:
                        pos = 0
                    if size > pos:
                        with paths.log_file.open("rb") as f:
                            f.seek(pos)
                            chunk = f.read(size - pos)
                            pos = size
                        sys.stdout.write(redact(chunk.decode("utf-8", "replace"), key))
                        sys.stdout.flush()
                time.sleep(0.4)
        except KeyboardInterrupt:
            print()


def cmd_watch(paths: Paths, cfg: Config) -> None:
    held = serve_running(paths)
    if not held:
        sys.exit(f"The endpoint is not running. Start it first: {cli_name()} up")
    if not TTY[0]:
        sys.exit(f'watch needs a terminal; use "{cli_name()} status" for a snapshot')
    watch_loop(paths, cfg, held["port"])


def parse_ask_options(words: list[str]) -> dict[str, Any]:
    effort: str | None = None
    images: list[str] = []
    rest: list[str] = []
    max_tokens: int | None = None
    system_prompt: str | None = None
    no_think = False
    hide_think = False
    i = 0
    while i < len(words):
        w = words[i]
        if w == "--max":
            effort = "max"
        elif w == "--no-think":
            no_think = True
            effort = "none"
        elif w == "--hide-think":
            hide_think = True
        elif w == "--effort" and i + 1 < len(words):
            i += 1
            raw_e = words[i].strip().lower()
            if raw_e in ("none", "off"):
                no_think = True
                effort = "none"
            else:
                effort = effort_value(raw_e) or sys.exit(f"invalid --effort {words[i]!r} (use 1-100, low, high, max, none)")
        elif w == "--max-tokens" and i + 1 < len(words):
            i += 1
            if not words[i].isdigit() or int(words[i]) <= 0:
                sys.exit(f"invalid --max-tokens {words[i]!r} (expected positive integer)")
            max_tokens = int(words[i])
        elif w == "--system" and i + 1 < len(words):
            i += 1
            system_prompt = words[i]
        elif w == "--image" and i + 1 < len(words):
            i += 1
            images.append(words[i])
        else:
            rest.append(w)
        i += 1
    return {
        "effort": effort,
        "images": images,
        "prompt": " ".join(rest).strip(),
        "max_tokens": max_tokens,
        "system": system_prompt,
        "no_think": no_think,
        "hide_think": hide_think,
    }


def parse_ask_args(words: list[str]) -> tuple[str | None, list[str], str]:
    opts = parse_ask_options(words)
    return opts["effort"], opts["images"], opts["prompt"]


def encode_image_url(path: str) -> str:
    if not os.path.isfile(path):
        sys.exit(f"image file not found: {path}")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def read_piped_stdin(has_prompt: bool) -> str:
    if not sys.stdin or sys.stdin.isatty():
        return ""
    if has_prompt:
        if IS_WIN:
            return ""
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not rlist:
                return ""
        except (OSError, ValueError):
            return ""
    return sys.stdin.read().strip()


def stream_chat_turn(
    paths: Paths,
    cfg: Config,
    port: int,
    model: str,
    messages: list[dict[str, Any]],
    *,
    effort: str | None = None,
    max_tokens: int | None = None,
    hide_think: bool = False,
) -> tuple[str, str]:
    """Streams one chat completion turn, returning (assistant_content, reasoning_content)."""
    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": messages,
    }
    if effort:
        body["reasoning_effort"] = effort
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    t_req = time.time()
    conn, r = http_call(
        port,
        "POST",
        "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": "Bearer " + read_api_key(paths), "Content-Type": "application/json"},
        UPSTREAM_TIMEOUT,
        cfg.proxy.host,
    )
    if r.status != 200:
        err_body = r.read()[:500].decode("utf-8", "replace")
        conn.close()
        sys.exit(f"HTTP {r.status}: {err_body}")

    live, part, timings, usage_obj, t_first = Live(), None, None, None, None
    out_text: list[str] = []
    think_text: list[str] = []
    show_think = TTY[0] and not hide_think

    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        wait = re.match(r": starting gpu vm \((\d+) s\)(?: - (.*))?", line)
        if wait:
            live.show(mmss(float(wait.group(1))), wait.group(2) or "starting the GPU (~1.5-2 min)")
        if not line.startswith("data: {"):
            continue
        event = json.loads(line[6:])
        if event.get("error"):
            live.clear()
            conn.close()
            sys.exit(f"\nerror: {event['error'].get('message', event['error'])}")
        timings = event.get("timings") or timings
        if event.get("usage"):
            usage_obj = event["usage"]
        for ch in event.get("choices", []):
            delta = ch.get("delta") or {}
            for kind, text in (("think", delta.get("reasoning_content")), ("say", delta.get("content"))):
                if not text:
                    continue
                if t_first is None:
                    t_first = time.time()
                if kind == "think":
                    think_text.append(text)
                    if not show_think:
                        continue
                else:
                    out_text.append(text)
                live.clear()
                if kind == "think" and part is None:
                    sys.stdout.write(paint(f"{glyph('╭─', '---')} thinking\n", "dim"))
                elif kind == "say" and part == "think":
                    sys.stdout.write(paint(f"\n{glyph('╰─', '---')}\n\n", "dim"))
                part = kind
                sys.stdout.write(paint(text, "dim") if kind == "think" else text)
                sys.stdout.flush()
    t_end = time.time()
    conn.close()
    live.clear()
    if part == "think" and show_think:
        sys.stdout.write(paint(f"\n{glyph('╰─', '---')}", "dim"))
    print()
    answer = "".join(out_text)
    reasoning = "".join(think_text)
    if not answer and reasoning and TTY[0]:
        print(paint("  (thinking finished without final answer — try a larger --max-tokens or --no-think)", "yellow"))
    if TTY[0]:
        if timings and timings.get("predicted_n"):
            print(paint(f"  {model}: {timings['predicted_n']} tokens at {timings.get('predicted_per_second', 0):.1f} tok/s", "dim"))
        elif usage_obj and usage_obj.get("completion_tokens") and t_first is not None:
            n = int(usage_obj["completion_tokens"])
            dec_s = max(0.001, t_end - t_first)
            ttft = max(0.0, t_first - t_req)
            print(paint(f"  {model}: {n} tokens · TTFT {ttft:.2f}s · {n / dec_s:.1f} tok/s", "dim"))
    return answer, reasoning


def _ensure_proxy_for_chat(paths: Paths, cfg: Config) -> tuple[int, str]:
    if not paths.key_file.exists():
        sys.exit(f"missing {paths.key_file}; run: {cli_name()} setup")
    held = ensure_detached_proxy(paths)
    port = int(held["port"])
    live = fetch_live_status(paths, cfg, port)
    model = str((live or {}).get("model") or cfg.basalt.effective_model_alias)
    return port, model


def cmd_ask(paths: Paths, cfg: Config, words: list[str]) -> None:
    opts = parse_ask_options(words)
    port, model = _ensure_proxy_for_chat(paths, cfg)
    prompt = opts["prompt"]
    images = opts["images"]
    piped = read_piped_stdin(bool(prompt or images))
    if piped and prompt:
        prompt = piped + "\n\n" + prompt
    elif piped:
        prompt = piped
    elif not prompt:
        prompt = "Describe this image." if images else "Hello!"
    content: Any = (
        [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": encode_image_url(p)}} for p in images]
        if images
        else prompt
    )
    messages: list[dict[str, Any]] = []
    if opts["system"]:
        messages.append({"role": "system", "content": opts["system"]})
    messages.append({"role": "user", "content": content})
    stream_chat_turn(
        paths,
        cfg,
        port,
        model,
        messages,
        effort=opts["effort"],
        max_tokens=opts["max_tokens"],
        hide_think=opts["hide_think"],
    )


def cmd_chat(paths: Paths, cfg: Config, words: list[str]) -> None:
    opts = parse_ask_options(words)
    port, model = _ensure_proxy_for_chat(paths, cfg)
    effort: str | None = opts["effort"]
    max_tokens: int | None = opts["max_tokens"]
    hide_think: bool = opts["hide_think"]
    system_prompt: str | None = opts["system"]
    pending_images: list[str] = list(opts["images"])
    history: list[dict[str, Any]] = []

    print(paint(f"basalt-colab chat · {model}", "bold") + paint("  (/help for commands, Ctrl+D or /exit to quit)", "dim"))
    while True:
        try:
            user_text = input(f"\n{paint('you ›', 'cyan', 'bold')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text.startswith("/"):
            parts = user_text.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/exit", "/quit", "/q"):
                return
            if cmd == "/help":
                print("  /clear            reset conversation history")
                print("  /effort <val>     set effort (low | 50 | high | max | none | default)")
                print("  /think            toggle hiding thinking tokens")
                print("  /system <text>    set or clear system prompt")
                print("  /image <path>     attach image to the next message")
                print("  /history          show turn count")
                print("  /exit             quit chat")
                continue
            if cmd == "/clear":
                history.clear()
                pending_images.clear()
                print(paint("  Conversation cleared.", "dim"))
                continue
            if cmd == "/think":
                hide_think = not hide_think
                print(paint(f"  Thinking display: {'hidden' if hide_think else 'visible'}", "dim"))
                continue
            if cmd == "/effort":
                if not arg or arg in ("default", "reset"):
                    effort = None
                    print(paint("  Effort reset to server default.", "dim"))
                elif arg.lower() in ("none", "off"):
                    effort = "none"
                    print(paint("  Thinking disabled (effort=none).", "dim"))
                else:
                    parsed = effort_value(arg)
                    if not parsed:
                        print(paint("  Invalid effort (use 1-100, low, high, max, none).", "red"))
                    else:
                        effort = parsed
                        print(paint(f"  Effort set to {effort}.", "dim"))
                continue
            if cmd == "/system":
                system_prompt = arg or None
                print(paint("  System prompt updated." if system_prompt else "  System prompt cleared.", "dim"))
                continue
            if cmd == "/image":
                if not arg or not os.path.isfile(arg):
                    print(paint(f"  Image not found: {arg or '(empty)'}", "red"))
                else:
                    pending_images.append(arg)
                    print(paint(f"  Attached {arg} (will send with next message).", "dim"))
                continue
            if cmd == "/history":
                print(paint(f"  {len(history) // 2} turns in context.", "dim"))
                continue
            print(paint(f"  Unknown command {cmd!r}. Type /help.", "red"))
            continue

        content: Any = (
            [{"type": "text", "text": user_text}] + [{"type": "image_url", "image_url": {"url": encode_image_url(p)}} for p in pending_images]
            if pending_images
            else user_text
        )
        pending_images.clear()
        msgs: list[dict[str, Any]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(history)
        msgs.append({"role": "user", "content": content})
        print()
        answer, _ = stream_chat_turn(
            paths,
            cfg,
            port,
            model,
            msgs,
            effort=effort,
            max_tokens=max_tokens,
            hide_think=hide_think,
        )
        history.append({"role": "user", "content": content})
        if answer:
            history.append({"role": "assistant", "content": answer})


def client_env_vars(paths: Paths, cfg: Config, port: int) -> tuple[tuple[str, str], ...]:
    """Compatibility helper returning env var tuples for a given port."""
    ep = ClientEndpoint.from_config(cfg, read_api_key(paths), port=port, paths=paths)
    return env_vars(ep)


def cmd_connect(paths: Paths, cfg: Config, target: str | None, args: list[str]) -> None:
    ep, running = fetch_live_endpoint(paths, cfg)
    me_cmd = cli_name()
    as_json = "--json" in args
    raw_only = "--raw" in args
    want_ps = "--powershell" in args or (IS_WIN and "--sh" not in args)
    only_env = "--sh" in args or "--powershell" in args
    do_write = "--write" in args

    if not target:
        if as_json:
            print(json.dumps(all_client_bundles(ep), indent=2))
            return
        state_note = "live proxy" if running else "from config.toml (proxy not running)"
        print(paint(f"basalt-colab Client Integration ({state_note})", "bold"))
        print(f"  Base URL   {paint(ep.openai_base_url, 'cyan')}")
        print(f"  Model      {paint(ep.model, 'bold')}")
        print(f"  Effort     {ep.effort}  ·  Vision: {'dynamic' if ep.vision_enabled else 'off'}\n")
        for cid in VALID_CLIENTS:
            b = client_bundle(ep, cid)
            print(f"  {paint(f'{me_cmd} connect {cid:<9}', 'cyan')}  {b['label']:<17} {paint(b['file_hint'], 'dim')}")
        print(f"\n  {paint('Tip:', 'dim')} add {paint('--raw', 'cyan')} to print only the snippet for piping, or open {paint(f'{me_cmd} console', 'cyan')} for 1-click Copy.")
        return

    cid = resolve_client(target)
    if cid is None:
        sys.exit(f"unknown client {target!r} (supported: {', '.join(VALID_CLIENTS)})")
    bundle = client_bundle(ep, cid)

    if do_write:
        if cid == "opencode":
            write_opencode_config(paths, cfg, ep.port)
            print(f"  {paint(glyph('✓', '+'), 'green')} Wrote {paths.opencode_file}")
            print(f"  Run: {paint(f'OPENCODE_CONFIG={shlex.quote(str(paths.opencode_file))} opencode', 'cyan')}")
            return
        sys.exit(f"--write is currently supported for opencode; for {cid}, copy or pipe `{me_cmd} connect {cid} --raw` into {bundle['file_hint']}")

    if as_json:
        print(json.dumps(bundle, indent=2))
        return
    if raw_only:
        sys.stdout.write(bundle["snippet"])
        return
    if only_env:
        print(bundle["env_ps"] if want_ps else bundle["env_sh"])
        return

    print(paint(f"{bundle['label']} — {bundle['subtitle']}", "bold"))
    print(f"  Target: {paint(bundle['file_hint'], 'cyan')}")
    print(f"  Model:  {ep.model}  ({ep.openai_base_url})\n")
    print(bundle["snippet"].rstrip("\n"))
    print(paint("\n# Environment:", "dim"))
    print(bundle["env_ps"] if want_ps else bundle["env_sh"])
    if bundle.get("run_hint"):
        print(paint("\n# Quick start:", "dim"))
        print("  " + paint(bundle["run_hint"], "cyan"))
    if bundle.get("notes"):
        print("\n" + paint(bundle["notes"], "dim"))


def cmd_env(paths: Paths, cfg: Config, args: list[str]) -> None:
    ep, _ = fetch_live_endpoint(paths, cfg)
    if "--json" in args:
        print(json.dumps(dict(env_vars(ep)), indent=2))
        return
    ps = "--powershell" in args or (IS_WIN and "--sh" not in args)
    for flag, cid in (("--dsh", "dsh"), ("--claude", "claude"), ("--codex", "codex"), ("--opencode", "opencode")):
        if flag in args:
            b = client_bundle(ep, cid)
            print(b["env_ps"] if ps else b["env_sh"])
            return
    print(format_env_lines(env_vars(ep), powershell=ps))


def cmd_doctor(paths: Paths, cfg: Config, manifest: Manifest) -> None:
    ok_all = True

    def check(label: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        if not ok:
            ok_all = False
        mark = paint(glyph("✓", "+"), "green") if ok else paint(glyph("✗", "!"), "red")
        print(f"  {mark}  {label:<18} {detail}")

    print(paint(f"basalt-colab {__version__} doctor", "bold") + "\n")
    check("Python", sys.version_info >= (3, 11), f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    ssh_ok = all(shutil.which(t) or os.path.exists(find_tool(t)) for t in ("ssh", "ssh-keygen"))
    check("OpenSSH", ssh_ok, "ssh + ssh-keygen found" if ssh_ok else "missing ssh or ssh-keygen")
    c_ok = colab_present()
    check("Colab CLI", c_ok, colab_cmd()[0] if c_ok else f"not installed (run {cli_name()} setup)")
    if c_ok:
        bal, _ = colab_usage()
        check("Colab OAuth", bal is not None, f"balance {bal:.1f} CU" if bal is not None else f"not logged in (run {cli_name()} setup)")
    missing_vm = missing_vm_files(paths.vm_dir)
    check("VM payload", not missing_vm, f"{paths.vm_dir} (Basalt {manifest.basalt_tag})" if not missing_vm else f"missing: {', '.join(missing_vm)}")
    check("Config", True, f"{paths.config_file} ({'saved' if paths.config_file.exists() else 'defaults'}) · {cfg.describe_engine()}")
    check("Credentials", paths.key_file.exists() and paths.ssh_key.exists(), str(paths.state_dir))
    held = serve_running(paths)
    if held:
        check("Local port", True, f"proxy already running on {cfg.proxy.host}:{held['port']} (pid {held['pid']})")
    else:
        s = socket.socket()
        try:
            s.bind((cfg.proxy.bind, cfg.proxy.port))
            port_free = True
        except OSError:
            port_free = False
        finally:
            s.close()
        check("Local port", port_free, f"{cfg.proxy.bind}:{cfg.proxy.port} available" if port_free else f"{cfg.proxy.bind}:{cfg.proxy.port} in use")
    if not ok_all:
        sys.exit(1)


def _extract_flag_val(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def main(argv: list[str] | None = None) -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"basalt-colab needs Python 3.11+ (this is {sys.version_info.major}.{sys.version_info.minor})")
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "help"
    init_console()

    paths = get_paths()
    if cmd == "_watchdog" and len(args) == 6:
        run_watchdog(paths, args[1], float(args[2]), float(args[3]), float(args[4]), float(args[5]))
        return
    if cmd in ("help", "-h", "--help") or (
        cmd not in ("ask", "chat")
        and any(a in ("-h", "--help") for a in args[1:])
    ):
        print(USAGE)
        return
    if cmd in ("-V", "--version"):
        print(f"basalt-colab {__version__}")
        return

    try:
        manifest = load_manifest(paths.vm_dir)
        cfg = load_config(paths)
    except (OSError, ConfigError, KeyError, ValueError) as e:
        sys.exit(f"config/manifest error: {e}")

    try:
        if cmd == "serve":
            serve(paths, cfg, manifest, detached="--detached" in args[1:])
        elif cmd == "setup":
            cmd_setup(paths, cfg, manifest)
        elif cmd in ("up", "start", "restart"):
            sub_args = args[1:]
            raw_scale = _extract_flag_val(sub_args, "--uncensor-scale")
            try:
                parsed_scale = float(raw_scale) if raw_scale is not None else None
            except ValueError:
                sys.exit(f"invalid --uncensor-scale {raw_scale!r} (expected float 0.0-2.0)")
            raw_par = _extract_flag_val(sub_args, "--parallel")
            try:
                parsed_par = int(raw_par) if raw_par is not None else None
            except ValueError:
                sys.exit(f"invalid --parallel {raw_par!r} (expected integer 1-4)")
            wait_gpu, wait_gpu_min = _parse_wait_gpu_args(sub_args)
            detach = any(f in sub_args for f in ("-d", "--detach", "--no-watch"))
            do_notify = "--no-notify" not in sub_args
            cmd_up(
                paths,
                cfg,
                manifest,
                wait_gpu=wait_gpu,
                wait_gpu_min=wait_gpu_min,
                basalt_ref=_extract_flag_val(sub_args, "--basalt-ref"),
                quant=_extract_flag_val(sub_args, "--quant"),
                pack_url=_extract_flag_val(sub_args, "--pack-url"),
                uncensor_scale=parsed_scale,
                profile=_extract_flag_val(sub_args, "--profile"),
                routing_mode=_extract_flag_val(sub_args, "--routing-mode"),
                parallel=parsed_par,
                detach=detach,
                do_notify=do_notify,
                restart=(cmd == "restart"),
            )
        elif cmd in ("down", "stop"):
            cmd_down(paths, cfg, manifest, everything="--all" in args[1:])
        elif cmd == "status":
            cmd_status(paths, cfg)
        elif cmd == "watch":
            cmd_watch(paths, cfg)
        elif cmd == "console":
            cmd_console(paths, cfg, manifest, no_open="--no-open" in args[1:])
        elif cmd == "config":
            cmd_config(paths, cfg, args[1:])
        elif cmd == "logs":
            cmd_logs(paths, cfg, args[1:])
        elif cmd == "ask":
            cmd_ask(paths, cfg, args[1:])
        elif cmd == "chat":
            cmd_chat(paths, cfg, args[1:])
        elif cmd == "connect":
            rest = [a for a in args[1:] if not a.startswith("-")]
            flags = [a for a in args[1:] if a.startswith("-")]
            cmd_connect(paths, cfg, rest[0] if rest else None, flags)
        elif cmd in ("dsh", "harness", "claude", "codex", "opencode"):
            cmd_connect(paths, cfg, cmd, args[1:])
        elif cmd == "env":
            cmd_env(paths, cfg, args[1:])
        elif cmd == "doctor":
            cmd_doctor(paths, cfg, manifest)
        else:
            sys.exit(USAGE)
    except KeyboardInterrupt:
        print()
        if cmd in ("up", "start", "restart") and serve_running(paths):
            print(f"Interrupted. The GPU may keep starting in the background. Stop it with: {cli_name()} down")
        sys.exit(130)
