"""Zero-fork Keep-Alive HTTP Range fetcher with direct os.pwrite() and optional RAM backpressure.

Usage:
  python3 pfetch.py JOBS.json [--workers 64] [--chunk-mb 64] [--min-avail-mib 0] [--log FILE] [--tag NAME]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import queue
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request

try:
    import urllib3  # type: ignore
except ImportError:
    urllib3 = None


READ_BUF = 2 << 20  # 2 MiB streaming buffer (releases GIL in OpenSSL & kernel pwrite64)


def _read_headers_file(path: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                k, _, v = line.strip().partition(":")
                if k:
                    headers[k.strip()] = v.strip()
    return headers


def resolve(url: str, hdr: str | None) -> str:
    headers = _read_headers_file(hdr)
    headers["Range"] = "bytes=0-0"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl() if "huggingface.co" in url else url


def _pwrite_all(fd: int, data: bytes | bytearray | memoryview, off: int) -> int:
    mv = memoryview(data)
    written = 0
    while written < len(mv):
        n = os.pwrite(fd, mv[written:], off + written)
        if n <= 0:
            raise OSError("short pwrite")
        written += n
    return written


def _mem_avail_mib() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


class _Backpressure:
    """Caches /proc/meminfo MemAvailable every 100ms so workers throttle cleanly without syscall spam."""

    def __init__(self, min_avail_mib: int) -> None:
        self.min_avail_mib = max(0, int(min_avail_mib))
        self._lock = threading.Lock()
        self._ts = 0.0
        self._val = 1 << 30

    def wait_room(self) -> None:
        if self.min_avail_mib <= 0:
            return
        while True:
            now = time.monotonic()
            with self._lock:
                if now - self._ts >= 0.1:
                    self._val = _mem_avail_mib()
                    self._ts = now
                avail = self._val
            if avail >= self.min_avail_mib:
                return
            time.sleep(0.15)


class _StdlibKeepAlive:
    """Thread-local HTTP/HTTPS Keep-Alive fallback when urllib3 is absent."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._ssl_ctx = ssl.create_default_context()

    def _conn(self, scheme: str, host: str, port: int):
        cur = getattr(self._local, "conn", None)
        key = (scheme, host, port)
        if cur is not None and getattr(self._local, "key", None) == key:
            return cur
        self.close_local()
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=30, context=self._ssl_ctx)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=30)
        self._local.conn = conn
        self._local.key = key
        return conn

    def close_local(self) -> None:
        cur = getattr(self._local, "conn", None)
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
            self._local.conn = None
            self._local.key = None

    def fetch_range(self, url: str, headers: dict[str, str], fd: int, off: int, expected: int, bp: _Backpressure, on_bytes) -> int:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        req_headers = dict(headers)
        req_headers.setdefault("Connection", "keep-alive")
        conn = self._conn(scheme, host, port)
        try:
            conn.request("GET", target, headers=req_headers)
            resp = conn.getresponse()
            status = resp.status
            if status not in (200, 206):
                resp.read()
                return status
            written = 0
            while True:
                bp.wait_room()
                chunk = resp.read(READ_BUF)
                if not chunk:
                    break
                _pwrite_all(fd, chunk, off + written)
                n = len(chunk)
                written += n
                on_bytes(n)
            if written != expected:
                raise IOError(f"truncated range read: got {written}, expected {expected}")
            return status
        except Exception:
            self.close_local()
            raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--min-avail-mib", type=int, default=0, help="Pause workers while MemAvailable < N MiB")
    ap.add_argument("--log")
    ap.add_argument("--tag", default="pfetch")
    a = ap.parse_args()
    log = open(a.log, "a", encoding="utf-8") if a.log else sys.stdout
    lk = threading.Lock()

    def say(s: str) -> None:
        with lk:
            log.write(time.strftime("%H:%M:%S ") + a.tag + " " + s + "\n")
            log.flush()

    jobs = json.load(open(a.jobs, encoding="utf-8"))
    for j in jobs:
        if not j.get("hdr") and os.environ.get("HF_HDR") and "huggingface.co" in j["url"]:
            j["hdr"] = os.environ["HF_HDR"]

    t0 = time.time()
    fds: dict[str, int] = {}
    hdr_cache: dict[str, dict[str, str]] = {}
    try:
        for j in jobs:
            dest = j["dest"]
            if dest not in fds:
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT, 0o644)
                if j.get("dest_size") is not None and os.fstat(fd).st_size != j["dest_size"]:
                    os.ftruncate(fd, j["dest_size"])
                fds[dest] = fd
            hpath = j.get("hdr")
            if hpath and hpath not in hdr_cache:
                hdr_cache[hpath] = _read_headers_file(hpath)

        q: queue.Queue[tuple[int, int, int]] = queue.Queue()
        chunk_sz = a.chunk_mb << 20
        total = 0
        for i, j in enumerate(jobs):
            s0 = j.get("src_off", 0)
            for s in range(s0, s0 + j["size"], chunk_sz):
                q.put((i, s, min(s + chunk_sz, s0 + j["size"])))
            total += j["size"]

        done = [0]
        errors: list[tuple[int, int, int, str]] = []
        final: dict[str, str | None] = {}
        resolve_locks: dict[str, threading.Lock] = {}
        bp = _Backpressure(a.min_avail_mib)

        pool = None
        stdlib_ka = None
        if urllib3 is not None:
            pool = urllib3.PoolManager(
                num_pools=16,
                maxsize=max(16, a.workers + 8),
                block=False,
                retries=False,
                timeout=urllib3.Timeout(connect=20.0, read=30.0),
            )
            mode_desc = "urllib3-keepalive"
        else:
            stdlib_ka = _StdlibKeepAlive()
            mode_desc = "stdlib-keepalive"

        def add_done(n: int) -> None:
            with lk:
                done[0] += n

        def url_of(i: int, refresh: bool = False) -> str:
            u = jobs[i]["url"]
            with lk:
                rl = resolve_locks.setdefault(u, threading.Lock())
                cached = None if refresh else final.get(u)
            if cached:
                return cached
            with rl:
                with lk:
                    if not refresh and final.get(u):
                        return final[u] or u
                for attempt in range(5):
                    try:
                        r = resolve(u, jobs[i].get("hdr"))
                        with lk:
                            final[u] = r
                        return r
                    except Exception as e:
                        say("resolve retry %s: %r" % (os.path.basename(u.split("?")[0]), e))
                        time.sleep(min(2 ** attempt, 16))
            return u

        def worker() -> None:
            while True:
                try:
                    i, s, e = q.get_nowait()
                except queue.Empty:
                    return
                j = jobs[i]
                fd = fds[j["dest"]]
                off = j.get("dest_off", 0) + s - j.get("src_off", 0)
                expected = e - s
                base_hdrs = hdr_cache.get(j.get("hdr") or "", {})
                last_err = ""
                for attempt in range(8):
                    u = url_of(i, refresh=(attempt > 0 and last_err in ("401", "403", "410")))
                    req_hdrs = {"Range": f"bytes={s}-{e - 1}", "Connection": "keep-alive"}
                    if "huggingface.co" in u and base_hdrs:
                        req_hdrs.update(base_hdrs)
                    chunk_written = [0]

                    def on_b(n: int) -> None:
                        chunk_written[0] += n
                        add_done(n)

                    try:
                        if pool is not None:
                            resp = pool.request("GET", u, headers=req_hdrs, preload_content=False, redirect=True)
                            try:
                                status = resp.status
                                if status not in (200, 206):
                                    resp.drain_conn()
                                else:
                                    written = 0
                                    for piece in resp.stream(READ_BUF, decode_content=False):
                                        if not piece:
                                            continue
                                        bp.wait_room()
                                        _pwrite_all(fd, piece, off + written)
                                        n = len(piece)
                                        written += n
                                        on_b(n)
                                    if written != expected:
                                        raise IOError(f"truncated range read: got {written}, expected {expected}")
                            finally:
                                resp.release_conn()
                        else:
                            assert stdlib_ka is not None
                            status = stdlib_ka.fetch_range(u, req_hdrs, fd, off, expected, bp, on_b)
                        if status in (200, 206):
                            break
                        last_err = str(status)
                        if status in (401, 403, 410):
                            url_of(i, refresh=True)
                    except Exception as exc:
                        last_err = repr(exc)
                        if chunk_written[0]:
                            add_done(-chunk_written[0])
                    time.sleep(min(2 ** attempt, 30))
                else:
                    with lk:
                        errors.append((i, s, e, last_err))
            if stdlib_ka is not None:
                stdlib_ka.close_local()

        nw = max(1, min(a.workers, q.qsize() or 1))
        ths = [threading.Thread(target=worker, daemon=True) for _ in range(nw)]
        for t in ths:
            t.start()

        last, last_done = time.time(), 0
        while any(t.is_alive() for t in ths):
            time.sleep(0.5)
            now = time.time()
            if now - last >= 5:
                with lk:
                    cur = done[0]
                say(
                    "%.1f/%.1f GB, last 5 s %.0f MB/s, avg %.0f MB/s"
                    % (cur / 1e9, total / 1e9, (cur - last_done) / 1e6 / (now - last), cur / 1e6 / max(now - t0, 1e-3))
                )
                last, last_done = now, cur

        dt = max(time.time() - t0, 1e-3)
        if errors:
            say("FAILED %d chunks (first: %s)" % (len(errors), errors[0][3]))
            sys.exit(1)
        say("DONE %.2f GB in %.1f s = %.0f MB/s (%s, w=%d)" % (done[0] / 1e9, dt, done[0] / 1e6 / dt, mode_desc, nw))
        if pool is not None:
            pool.clear()
    finally:
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        if a.log:
            log.close()


if __name__ == "__main__":
    main()
