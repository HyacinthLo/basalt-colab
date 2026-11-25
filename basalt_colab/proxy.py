"""Local HTTP gateway: OpenAI/Anthropic proxy, Basalt Web UI, Unified Control Console, and /_control/* API."""
from __future__ import annotations

import hmac
import http.client
import json
import selectors
import signal
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from basalt_colab.clients import ClientEndpoint, all_client_bundles, opencode_config_dict
from basalt_colab.colab import (
    Logger,
    read_api_key,
    read_bytes,
    redact,
    release_lock,
    session_alive,
    take_lock,
    update_lock,
    vm_logs,
)
from basalt_colab.config import (
    BootOverrides,
    Config,
    ConfigError,
    config_meta,
    dump_toml,
    load_config,
    parse_toml,
    save_config,
)
from basalt_colab.console import render_console_html
from basalt_colab.lifecycle import UPSTREAM_TIMEOUT, Manager, VmState
from basalt_colab.manifest import Manifest
from basalt_colab.paths import IS_WIN, Paths
from basalt_colab.usage import UsageLedger

MAX_BODY = 64 << 20
HOP = {
    "connection", "keep-alive", "proxy-connection", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    "host", "accept-encoding", "expect", "server", "date",
}


def write_opencode_config(paths: Paths, cfg: Config, port: int | None = None) -> None:
    key = read_api_key(paths) if paths.key_file.exists() else ""
    ep = ClientEndpoint.from_config(cfg, key, port=port or cfg.proxy.port, paths=paths)
    doc = opencode_config_dict(ep, key_mode="file")
    paths.ensure_dirs()
    paths.opencode_file.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# Official DeepSeek API compatibility aliases (api-docs.deepseek.com/api/create-message):
# DeepSeek-V4.1 natively accepts none | low (50) | high (75, default) | max (100) | 1..100.
DEEPSEEK_EFFORT_ALIASES: dict[str, str] = {
    "minimal": "low",
    "medium": "high",
    "xhigh": "high",
    "ultra": "max",
}


def _normalize_effort(val: Any) -> Any:
    if isinstance(val, str):
        mapped = DEEPSEEK_EFFORT_ALIASES.get(val.strip().lower())
        if mapped is not None:
            return mapped
    return val


def rewrite_request(body: bytes, cfg: Config, path: str = "/v1/chat/completions") -> bytes:
    """Single-model routing (unless strict_model), DeepSeek effort alias normalization, and default effort injection."""
    try:
        req = json.loads(body)
    except ValueError:
        return body
    if not isinstance(req, dict):
        return body
    changed = False
    target_model = cfg.basalt.effective_model_alias
    if not cfg.proxy.strict_model and req.get("model") != target_model:
        req["model"] = target_model
        changed = True

    # Normalize compatibility effort aliases (e.g. Codex default "medium" -> "high") so Basalt DS4.1 never 500s.
    if path == "/v1/chat/completions":
        if "reasoning_effort" in req:
            norm = _normalize_effort(req["reasoning_effort"])
            if norm != req["reasoning_effort"]:
                req["reasoning_effort"] = norm
                changed = True
        ctk = req.get("chat_template_kwargs")
        if isinstance(ctk, dict) and "reasoning_effort" in ctk:
            norm = _normalize_effort(ctk["reasoning_effort"])
            if norm != ctk["reasoning_effort"]:
                req["chat_template_kwargs"] = dict(ctk, reasoning_effort=norm)
                changed = True
    elif path == "/v1/responses":
        reasoning = req.get("reasoning")
        if isinstance(reasoning, dict) and "effort" in reasoning:
            norm = _normalize_effort(reasoning["effort"])
            if norm != reasoning["effort"]:
                req["reasoning"] = dict(reasoning, effort=norm)
                changed = True
    elif path in ("/v1/messages", "/v1/messages/count_tokens"):
        oc = req.get("output_config")
        if isinstance(oc, dict) and "effort" in oc:
            norm = _normalize_effort(oc["effort"])
            if norm != oc["effort"]:
                req["output_config"] = dict(oc, effort=norm)
                changed = True

    effort = str(cfg.basalt.effort)
    if effort != "50":
        if path == "/v1/chat/completions" and req.get("reasoning_effort") is None:
            req["reasoning_effort"] = effort
            changed = True
        elif path == "/v1/responses":
            reasoning = req.get("reasoning")
            if reasoning is None:
                req["reasoning"] = {"effort": effort}
                changed = True
            elif isinstance(reasoning, dict) and reasoning.get("effort") is None:
                req["reasoning"] = dict(reasoning, effort=effort)
                changed = True
    return json.dumps(req).encode() if changed else body


def abort_upstream(conn: http.client.HTTPConnection | None) -> None:
    if conn is None:
        return
    try:
        if conn.sock:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        conn.close()
    except OSError:
        pass


def error_obj(code: int, msg: str, kind: str = "unavailable_error") -> dict[str, Any]:
    return {"error": {"code": code, "message": msg, "type": kind}}


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = not IS_WIN
    mgr: Manager
    paths: Paths

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), int(port)

    def handle_error(self, request: Any, client_address: Any) -> None:
        e = sys.exc_info()[1]
        if not isinstance(e, (OSError, http.client.HTTPException)):
            self.mgr.logger.log(f"handler error: {e!r}", echo=False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 900
    disable_nagle_algorithm = True
    server_version, sys_version = "basalt-colab", ""
    server: ProxyServer

    def setup(self) -> None:
        self.request.settimeout(30)
        super().setup()

    def log_message(self, *args: Any) -> None:
        pass

    def chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

    def send_cors(self, preflight: bool = False) -> None:
        if not preflight and not self.headers.get("Origin"):
            return
        self.send_header("Access-Control-Allow-Origin", "*")
        if preflight:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header(
                "Access-Control-Allow-Headers",
                asked or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta",
            )
            self.send_header("Access-Control-Max-Age", "86400")

    def send_bytes(self, code: int, ctype: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, code: int, obj: Any) -> None:
        self.send_bytes(code, "application/json", json.dumps(obj).encode())

    def start_chunked(self, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_cors()
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def fail(self, code: int, msg: str, sse: bool = False, obj: dict[str, Any] | None = None) -> None:
        obj = obj or error_obj(code, msg)
        if not sse:
            self.send_json(code, obj)
            return
        path = self.path.split("?")[0]
        raw_err = obj.get("error") if isinstance(obj, dict) else None
        err: dict[str, Any] = (
            raw_err
            if isinstance(raw_err, dict)
            else {"code": code, "message": msg or str(obj), "type": "unavailable_error"}
        )
        if path == "/v1/messages":
            payload = b"event: error\ndata: " + json.dumps({"type": "error", "error": err}).encode() + b"\n\n"
        elif path == "/v1/responses":
            payload = (
                b"event: error\ndata: "
                + json.dumps({"type": "error", "code": err.get("code", code), "message": err.get("message", msg)}).encode()
                + b"\n\n"
            )
        else:
            payload = b"data: " + json.dumps({"error": err}).encode() + b"\n\ndata: [DONE]\n\n"
        try:
            self.chunk(payload)
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass
        self.close_connection = True

    def authed(self) -> bool:
        key = read_api_key(self.server.paths).encode()
        got = self.headers.get("Authorization", "").encode()
        xkey = self.headers.get("x-api-key", "").encode()
        if hmac.compare_digest(got, b"Bearer " + key) or (xkey and hmac.compare_digest(xkey, key)):
            return True
        self.send_json(401, error_obj(401, "Invalid API Key", "authentication_error"))
        return False

    def read_body(self) -> bytes | None:
        try:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                parts: list[bytes] = []
                total = 0
                while True:
                    n = int(self.rfile.readline(1024).split(b";")[0].strip(), 16)
                    if n == 0:
                        while self.rfile.readline(1024) not in (b"\r\n", b"\n", b""):
                            pass
                        return b"".join(parts)
                    total += n
                    if total > MAX_BODY:
                        break
                    parts.append(self.rfile.read(n))
                    self.rfile.readline(8)
            else:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= MAX_BODY:
                    return self.rfile.read(n) if n > 0 else b""
            self.close_connection = True
            self.fail(413, "request body too large", obj=error_obj(413, "request body too large", "invalid_request_error"))
        except ValueError:
            self.close_connection = True
            self.fail(400, "bad request body", obj=error_obj(400, "bad request body", "invalid_request_error"))
        return None

    def host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        return not self.server.mgr.cfg.proxy.loopback or name in ("", "127.0.0.1", "localhost", "[::1]")

    def client_gone(self) -> bool:
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(self.connection, selectors.EVENT_READ)
                ready = sel.select(timeout=0)
            return bool(ready) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except (ConnectionError, TimeoutError):
            return True
        except (OSError, ValueError):
            return False

    def do_OPTIONS(self) -> None:
        if not self.host_ok():
            return self.fail(403, "unexpected Host header", obj=error_obj(403, "unexpected Host header", "permission_error"))
        self.send_response(204)
        self.send_cors(preflight=True)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self.route(None)

    def do_POST(self) -> None:
        body = self.read_body()
        if body is not None:
            self.route(body)

    def models_payload(self) -> dict[str, Any]:
        cfg = self.server.mgr.cfg
        entry = {
            "id": cfg.basalt.effective_model_alias,
            "object": "model",
            "owned_by": "basalt",
            "created": 0,
            "quant": cfg.basalt.quant,
            "reasoning_effort": str(cfg.basalt.effort),
            "context_length": cfg.basalt.max_context,
            "max_model_len": cfg.basalt.max_context,
            "image_input": cfg.basalt.vision != "off",
            "reasoning": {
                "default_enabled": True,
                "default_effort": str(cfg.basalt.effort),
                "supported_efforts": ["max", "high", "low", "none"],
            },
        }
        return {"object": "list", "data": [entry], "models": [entry]}

    def route(self, body: bytes | None) -> None:
        mgr = self.server.mgr
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if not self.host_ok():
            return self.fail(403, "unexpected Host header", obj=error_obj(403, "unexpected Host header", "permission_error"))
        if path in ("/_console", "/_console/") and body is None:
            return self.send_bytes(200, "text/html; charset=utf-8", render_console_html())
        if path.startswith("/_control/"):
            return self.control(path, parsed.query, body)
        if path in ("/v1/models", "/models") and body is None:
            if self.authed():
                self.send_json(200, self.models_payload())
            return
        if body and path in ("/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses"):
            body = rewrite_request(body, mgr.cfg, path)
        if body is not None and path.startswith("/v1/"):
            if self.authed():
                t0 = time.time()
                self.held(body)
                mgr.logger.log(f"{self.command} {path} done in {time.time() - t0:.1f} s", echo=False)
            return
        if body is None and mgr.ready():
            return self.forward(None)
        if body is not None and mgr.enter():
            try:
                return self.forward(body)
            finally:
                mgr.leave()
        if path == "/" and body is None:
            # GPU is off or booting: serve the unified Control Console directly at /
            return self.send_bytes(200, "text/html; charset=utf-8", render_console_html())
        if path in ("/health", "/v1/health"):
            return self.send_json(503, dict(error_obj(503, f"gpu vm {mgr.state}"), status=mgr.state))
        self.fail(503, f"gpu vm {mgr.state}; a POST to /v1/... or /_console starts it")

    def held(self, body: bytes) -> None:
        mgr = self.server.mgr
        if mgr.enter():
            try:
                return self.forward(body)
            finally:
                mgr.leave()
        stream = False
        try:
            stream = bool(json.loads(body).get("stream")) if body else False
        except (ValueError, AttributeError):
            pass
        if not stream:
            err = mgr.acquire(is_gone=self.client_gone)
            if err == "client disconnected":
                self.close_connection = True
                return
            if err:
                return self.fail(503, err)
        else:
            self.start_chunked("text/event-stream")

            def tick(n: int) -> None:
                stage = mgr.activity()
                msg = f": starting gpu vm ({n} s) - {stage}\n\n" if stage else f": starting gpu vm ({n} s)\n\n"
                self.chunk(msg.encode("utf-8"))

            try:
                tick(0)
                err = mgr.acquire(tick, is_gone=self.client_gone)
            except OSError:
                self.close_connection = True
                return
            if err == "client disconnected":
                self.close_connection = True
                return
            if err:
                return self.fail(503, err, sse=True)
        try:
            self.forward(body, sse=stream)
        finally:
            mgr.leave()

    def forward(self, body: bytes | None, sse: bool = False) -> None:
        mgr = self.server.mgr
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        done = threading.Event()
        cur_conn: list[http.client.HTTPConnection | None] = [None]
        aborted = [False]
        wake_r, wake_w = socket.socketpair()

        def watch_client() -> None:
            sock = self.connection
            try:
                with selectors.DefaultSelector() as sel:
                    sel.register(sock, selectors.EVENT_READ, "client")
                    sel.register(wake_r, selectors.EVENT_READ, "done")
                    while not done.is_set():
                        for key, _ in sel.select():
                            if key.data == "done":
                                return
                            try:
                                if sock.recv(1, socket.MSG_PEEK) == b"":
                                    aborted[0] = True
                                    abort_upstream(cur_conn[0])
                                    return
                                return
                            except (ConnectionError, TimeoutError):
                                aborted[0] = True
                                abort_upstream(cur_conn[0])
                                return
            except (OSError, ValueError):
                return
            finally:
                try:
                    wake_r.close()
                except OSError:
                    pass

        threading.Thread(target=watch_client, daemon=True).start()
        conn: http.client.HTTPConnection | None = None
        try:
            for attempt in range(2):
                if aborted[0]:
                    self.close_connection = True
                    return
                assert mgr.port is not None
                c = http.client.HTTPConnection("127.0.0.1", mgr.port, timeout=UPSTREAM_TIMEOUT)
                cur_conn[0] = c
                try:
                    c.request(self.command, self.path, body=body, headers=headers)
                    resp = c.getresponse()
                    conn = c
                    break
                except (OSError, http.client.HTTPException) as e:
                    c.close()
                    if aborted[0]:
                        self.close_connection = True
                        return
                    if attempt == 0 and mgr.tunnel_failed():
                        continue
                    return self.fail(502, f"upstream unreachable ({e.__class__.__name__})", sse)
            if sse:
                if resp.status != 200:
                    raw = resp.read()
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        obj = None
                    if not isinstance(obj, dict) or "error" not in obj:
                        obj = error_obj(resp.status, raw.decode("utf-8", "replace")[:500], "upstream_error")
                    return self.fail(resp.status, "", True, obj)
                self.pipe(resp, True)
                return
            self.send_response(resp.status, resp.reason)
            has_acao = False
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    if k.lower() == "access-control-allow-origin":
                        has_acao = True
                    self.send_header(k, v)
            if not has_acao:
                self.send_cors()
            n = resp.getheader("Content-Length")
            if n is not None and not resp.chunked:
                self.send_header("Content-Length", n)
                self.end_headers()
                self.pipe(resp, False)
            else:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.pipe(resp, True)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            done.set()
            try:
                wake_w.send(b"\x01")
            except OSError:
                pass
            try:
                wake_w.close()
            except OSError:
                pass
            if conn is not None:
                conn.close()

    def pipe(self, resp: http.client.HTTPResponse, chunked: bool) -> None:
        while True:
            data = resp.read1(65536) if chunked else resp.read(65536)
            if not data:
                break
            if chunked:
                self.chunk(data)
            else:
                self.wfile.write(data)
        if chunked:
            self.wfile.write(b"0\r\n\r\n")

    def _sync_lock_and_opencode(self) -> None:
        mgr, paths = self.server.mgr, self.server.paths
        update_lock(paths, {"cold_signature": mgr.cfg.cold_signature(), "engine_summary": mgr.cfg.describe_engine()})
        try:
            write_opencode_config(paths, mgr.cfg, self.server.server_port)
        except OSError:
            pass

    def control(self, path: str, query: str, body: bytes | None) -> None:
        if not self.authed():
            return
        qs = urllib.parse.parse_qs(query)
        if path == "/_control/status":
            return self.send_json(200, self.server.mgr.status())
        if path == "/_control/events" and body is None:
            return self._ctrl_events()
        if path == "/_control/clients" and body is None:
            return self._ctrl_clients()
        if path == "/_control/usage":
            days = int((qs.get("days") or ["30"])[0])
            return self.send_json(200, self.server.mgr.usage_summary(days=days))
        if path == "/_control/logs":
            return self._ctrl_logs(include_vm=(qs.get("vm") or ["0"])[0] == "1")
        if path == "/_control/config":
            return self._ctrl_config(body)
        if path == "/_control/reload" and body is not None:
            return self._ctrl_reload(body)
        if path in ("/_control/up", "/_control/down", "/_control/restart") and body is not None:
            return self._ctrl_lifecycle(path, body)
        return self.fail(404, "unknown control request", obj=error_obj(404, "unknown control request", "not_found_error"))

    def _ctrl_events(self) -> None:
        mgr = self.server.mgr
        self.start_chunked("text/event-stream")
        self.close_connection = True
        try:
            rev = mgr.event_rev()
            st = mgr.status()
            self.chunk(b"data: " + json.dumps(st).encode("utf-8") + b"\n\n")
            while not mgr.closing:
                if self.client_gone():
                    break
                active_mode = st["state"] != VmState.OFF
                new_rev = mgr.wait_event(rev, timeout=1.0 if active_mode else 15.0)
                if mgr.closing or self.client_gone():
                    break
                if new_rev == rev and not active_mode:
                    self.chunk(b": ping\n\n")
                    continue
                rev = new_rev
                st = mgr.status()
                self.chunk(b"data: " + json.dumps(st).encode("utf-8") + b"\n\n")
        except OSError:
            pass
        try:
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass

    def _ctrl_clients(self) -> None:
        ep = ClientEndpoint.from_config(
            self.server.mgr.cfg,
            read_api_key(self.server.paths),
            port=self.server.server_port,
            paths=self.server.paths,
        )
        self.send_json(200, all_client_bundles(ep))

    def _ctrl_logs(self, *, include_vm: bool) -> None:
        mgr, paths = self.server.mgr, self.server.paths
        raw = read_bytes(paths.log_file)
        key = read_api_key(paths)
        local_txt = redact(raw.decode("utf-8", "replace")[-6000:], key) if raw is not None else ""
        vm_txt = ""
        if include_vm and session_alive(mgr.cfg.colab.session):
            _, out = vm_logs(mgr.cfg.colab.session)
            vm_txt = redact(out, key)
        self.send_json(200, {"local": local_txt, "vm": vm_txt})

    def _ctrl_config(self, body: bytes | None) -> None:
        mgr, paths = self.server.mgr, self.server.paths
        if body is None:
            disk_cfg = load_config(paths, apply_env=False)
            return self.send_json(200, {
                "config": disk_cfg.to_dict(),
                "toml": dump_toml(disk_cfg),
                "path": str(paths.config_file),
                "meta": config_meta(),
            })
        try:
            prev_disk = load_config(paths, apply_env=False)
            payload = json.loads(body or b"{}")
            if "toml" in payload:
                new_cfg = parse_toml(str(payload["toml"]))
            elif "config" in payload:
                new_cfg = Config.from_dict(payload["config"])
            else:
                raise ConfigError("expected 'toml' or 'config' in request body")
            save_config(new_cfg, paths)
            hot, cold = mgr.reload_config(new_cfg, prev_cfg=prev_disk)
            self._sync_lock_and_opencode()
            mgr.logger.log(f"config saved: hot={hot} cold={cold}", echo=False)
            self.send_json(200, {
                "applied": mgr.cfg.hot_snapshot(),
                "toml": dump_toml(new_cfg),
                "config": new_cfg.to_dict(),
                "hot_changed": hot,
                "cold_changed": cold,
                "restart_required": bool(cold),
            })
        except (ValueError, ConfigError) as e:
            self.fail(400, str(e), obj=error_obj(400, str(e), "invalid_request_error"))

    def _ctrl_reload(self, body: bytes) -> None:
        mgr, paths = self.server.mgr, self.server.paths
        try:
            payload = json.loads(body or b"{}")
            new_cfg = load_config(paths, apply_env=False)
            if isinstance(payload, dict) and payload:
                new_cfg = new_cfg.overlay_hot_dict(payload)
            hot, cold = mgr.reload_config(new_cfg)
            self._sync_lock_and_opencode()
            self.send_json(200, {
                "applied": mgr.cfg.hot_snapshot(),
                "hot_changed": hot,
                "cold_changed": cold,
                "restart_required": bool(cold),
            })
        except (ValueError, ConfigError) as e:
            self.fail(400, str(e), obj=error_obj(400, str(e), "invalid_request_error"))

    def _ctrl_lifecycle(self, path: str, body: bytes) -> None:
        mgr, paths = self.server.mgr, self.server.paths
        try:
            opts = json.loads(body or b"{}")
        except ValueError:
            opts = {}
        self.start_chunked("text/plain; charset=utf-8")

        def say(text: str) -> None:
            self.chunk((text + "\n").encode())

        try:
            if path == "/_control/restart":
                if mgr.state is not VmState.OFF or paths.session_file.exists():
                    mgr.pop_last_stopped()
                    mgr.request_stop("restart")
                    still = mgr.wait_off(lambda n: say(f"wait {n} stopping current G4 for restart"), every=1)
                    if still:
                        say("FAILED could not stop running G4 session before restart")
                        self.wfile.write(b"0\r\n\r\n")
                        return
                try:
                    fresh_cfg = load_config(paths, apply_env=True)
                    mgr.reload_config(fresh_cfg)
                    self._sync_lock_and_opencode()
                except (OSError, ConfigError) as e:
                    say(f"FAILED config reload error: {e}")
                    self.wfile.write(b"0\r\n\r\n")
                    return
                path = "/_control/up"

            if path == "/_control/up":
                try:
                    overrides = BootOverrides.from_dict(opts)
                except ConfigError as e:
                    say(f"FAILED {e}")
                    self.wfile.write(b"0\r\n\r\n")
                    return
                with mgr.cond:
                    joined = mgr.state is VmState.STARTING
                    t0, sent = (mgr.begun if joined else time.time()), [0 if joined else len(mgr.steps)]

                def emit_steps() -> None:
                    done = mgr.steps
                    sent[0] = 0 if sent[0] > len(done) else sent[0]
                    for _, at, label in done[sent[0]:]:
                        if label:
                            say(f"step {max(0, int(at - t0))} {label}")
                    sent[0] = len(done)

                def tick(n: int) -> None:
                    emit_steps()
                    say(f"wait {n} {mgr.activity()}")

                err = mgr.acquire(tick, every=1, explicit=True, overrides=overrides)
                if not err:
                    mgr.leave(served=0)
                    emit_steps()
                    self._sync_lock_and_opencode()
                say(f"FAILED {err}" if err else f"READY {int(time.time() - t0)}")
                self.wfile.write(b"0\r\n\r\n")
                return

            mgr.pop_last_stopped()
            mgr.request_stop("down")
            still = mgr.wait_off(lambda n: say(f"wait {n} stopping the GPU"), every=1)
            cu, ran = mgr.pop_last_stopped()
            say("STILL" if still else f"STOPPED {cu} {ran}")
            self.wfile.write(b"0\r\n\r\n")
            if opts.get("exit"):
                self.close_connection = True
                threading.Thread(target=self.server.shutdown, daemon=True).start()
        except OSError:
            self.close_connection = True


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def serve(paths: Paths, cfg: Config, manifest: Manifest, detached: bool = False) -> None:
    logger = Logger(paths, echo=not detached)
    if not paths.key_file.exists():
        sys.exit(f"missing {paths.key_file}; run: basalt-colab setup")
    held = take_lock(paths, {
        "port": cfg.proxy.port,
        "detached": detached,
        "cold_signature": cfg.cold_signature(),
        "engine_summary": cfg.describe_engine(),
    })
    if held:
        sys.exit(f"basalt-colab serve is already running (pid {held.get('pid')}, port {held.get('port')})")
    try:
        srv = ProxyServer((cfg.proxy.bind, cfg.proxy.port), Handler)
    except OSError as e:
        release_lock(paths)
        sys.exit(f"cannot serve on {cfg.proxy.bind}:{cfg.proxy.port}: {e} (set BASALT_COLAB_PORT)")
    ledger = UsageLedger(paths.usage_file)
    srv.paths = paths
    srv.mgr = Manager(paths, cfg, manifest, ledger, logger)
    for sig in ("SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _interrupt)
    if paths.session_file.exists():
        srv.mgr.adopt()
    threading.Thread(target=srv.mgr.monitor, daemon=True).start()
    logger.log(
        f"serve: http://{cfg.proxy.host}:{cfg.proxy.port}/v1 (console /_console), "
        f"{cfg.basalt.effective_model_alias} ({cfg.describe_engine()}, {cfg.describe_uncensor()}), "
        f"session {cfg.colab.session} (idle {cfg.colab.idle_min:.1f} min, cap {cfg.colab.max_hours:.1f} h)"
    )
    try:
        srv.serve_forever(0.5)
    except KeyboardInterrupt:
        logger.log("serve: interrupted")
    finally:
        srv.server_close()
        try:
            srv.mgr.shutdown()
        except KeyboardInterrupt:
            logger.log("serve: second interrupt, leaving the VM to the watchdog")
        release_lock(paths)
        logger.log("serve: exit")
