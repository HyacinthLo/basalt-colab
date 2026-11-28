"""Parallel HTTP range fetcher: multi-process native C curl|dd (zero Python GIL), chunks land in place.

Usage:
  python3 pfetch.py JOBS.json [--workers 64] [--chunk-mb 128] [--min-avail-mib 0] [--log FILE] [--tag NAME]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import queue
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request


def _load_headers(hdr: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if hdr and os.path.exists(hdr):
        with open(hdr, encoding="utf-8") as f:
            for line in f:
                k, _, v = line.strip().partition(":")
                if k:
                    headers[k.strip()] = v.strip()
    return headers


def resolve(url: str, hdr: str | None) -> str:
    headers = {"Range": "bytes=0-0"}
    headers.update(_load_headers(hdr))
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl() if "huggingface.co" in url else url


def _mem_avail_mib() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--min-avail-mib", type=int, default=0, help="Pause before starting new chunks if MemAvailable < N MiB")
    ap.add_argument("--keepalive", action="store_true", help="Use persistent HTTP/1.1 Keep-Alive sockets + os.pwrite (zero fork, ideal for thousands of spans)")
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
    final: dict[str, str | None] = {}
    dest_fds: dict[str, int] = {}
    seen_dest: set[tuple[str, int | None]] = set()
    for j in jobs:
        key = (j["dest"], j.get("dest_size"))
        if key not in seen_dest:
            seen_dest.add(key)
            fd = os.open(j["dest"], os.O_WRONLY | os.O_CREAT, 0o644)
            if j.get("dest_size") is not None and os.fstat(fd).st_size != j["dest_size"]:
                os.ftruncate(fd, j["dest_size"])
            if a.keepalive and j["dest"] not in dest_fds:
                dest_fds[j["dest"]] = fd
            else:
                os.close(fd)
        elif a.keepalive and j["dest"] not in dest_fds:
            dest_fds[j["dest"]] = os.open(j["dest"], os.O_WRONLY | os.O_CREAT, 0o644)

    q: queue.Queue[tuple[int, int, int]] = queue.Queue()
    chunk = a.chunk_mb << 20
    total = 0
    for i, j in enumerate(jobs):
        s0 = j.get("src_off", 0)
        for s in range(s0, s0 + j["size"], chunk):
            q.put((i, s, min(s + chunk, s0 + j["size"])))
        total += j["size"]
    done, errors = [0], []
    ssl_ctx = ssl.create_default_context() if a.keepalive else None
    hdr_cache: dict[str, dict[str, str]] = {}

    def url_of(i: int, refresh: bool = False) -> str:
        u = jobs[i]["url"]
        with lk:
            if refresh or u not in final:
                final[u] = None
        if final.get(u) is None:
            for attempt in range(5):
                try:
                    r = resolve(u, jobs[i].get("hdr"))
                    with lk:
                        final[u] = r
                    break
                except Exception as e:
                    say("resolve retry %s: %r" % (os.path.basename(u.split("?")[0]), e))
                    time.sleep(2 ** attempt)
        return final.get(u) or u

    def worker_curl() -> None:
        while True:
            try:
                i, s, e = q.get_nowait()
            except queue.Empty:
                return
            if a.min_avail_mib > 0:
                while _mem_avail_mib() < a.min_avail_mib:
                    time.sleep(0.2)
            j = jobs[i]
            off = j.get("dest_off", 0) + s - j.get("src_off", 0)
            for attempt in range(8):
                u = url_of(i)
                h = " -H @'%s'" % j["hdr"] if j.get("hdr") else ""
                cmd = (
                    "set -o pipefail; curl -sf --connect-timeout 20 --speed-limit 1000000 --speed-time 30%s -r %d-%d '%s' | "
                    "dd of='%s' bs=4M seek=%d oflag=seek_bytes conv=notrunc status=none iflag=fullblock"
                ) % (h, s, e - 1, u, j["dest"], off)
                rc = subprocess.run(["bash", "-c", cmd]).returncode
                if rc == 0:
                    with lk:
                        done[0] += e - s
                    break
                if rc == 22:
                    url_of(i, refresh=True)
                time.sleep(min(2 ** attempt, 30))
            else:
                errors.append((i, s, e))

    def worker_keepalive() -> None:
        conn: http.client.HTTPConnection | None = None
        conn_key: tuple[str, str, int] | None = None
        buf = bytearray(4 << 20)
        mv = memoryview(buf)

        def close_conn() -> None:
            nonlocal conn, conn_key
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                conn_key = None

        try:
            while True:
                try:
                    i, s, e = q.get_nowait()
                except queue.Empty:
                    return
                if a.min_avail_mib > 0:
                    while _mem_avail_mib() < a.min_avail_mib:
                        time.sleep(0.2)
                j = jobs[i]
                want = e - s
                base_off = j.get("dest_off", 0) + s - j.get("src_off", 0)
                fd = dest_fds[j["dest"]]
                orig_host = urllib.parse.urlsplit(j["url"]).netloc
                for attempt in range(8):
                    u = url_of(i)
                    sp = urllib.parse.urlsplit(u)
                    scheme = sp.scheme.lower()
                    host = sp.hostname or ""
                    port = sp.port or (443 if scheme == "https" else 80)
                    path = sp.path + (f"?{sp.query}" if sp.query else "")
                    key = (scheme, host, port)
                    try:
                        if conn is None or conn_key != key:
                            close_conn()
                            if scheme == "https":
                                conn = http.client.HTTPSConnection(host, port, timeout=60, context=ssl_ctx)
                            else:
                                conn = http.client.HTTPConnection(host, port, timeout=60)
                            conn_key = key
                        req_hdrs = {"Range": f"bytes={s}-{e - 1}", "Connection": "keep-alive"}
                        hdr_file = j.get("hdr")
                        if hdr_file and sp.netloc == orig_host:
                            if hdr_file not in hdr_cache:
                                with lk:
                                    hdr_cache.setdefault(hdr_file, _load_headers(hdr_file))
                            req_hdrs.update(hdr_cache[hdr_file])
                        conn.request("GET", path, headers=req_hdrs)
                        resp = conn.getresponse()
                        status = resp.status
                        ok_status = (status == 206) or (status == 200 and s == 0)
                        if ok_status and (resp.length is None or resp.length == want):
                            got = 0
                            while got < want:
                                take = min(len(buf), want - got)
                                n = resp.readinto(mv[:take])
                                if not n:
                                    break
                                written = 0
                                while written < n:
                                    w = os.pwrite(fd, mv[written:n], base_off + got + written)
                                    if w <= 0:
                                        raise OSError("short pwrite")
                                    written += w
                                got += n
                            if got == want:
                                if resp.will_close:
                                    close_conn()
                                with lk:
                                    done[0] += want
                                break
                            # Short body -> drop connection and retry
                            close_conn()
                        else:
                            if resp.length is not None and resp.length <= 65536:
                                resp.read()
                            else:
                                close_conn()
                            if resp.will_close or status >= 500 or status in (401, 403, 410, 429):
                                close_conn()
                            if status in (301, 302, 303, 307, 308, 401, 403, 410):
                                url_of(i, refresh=True)
                    except Exception:
                        close_conn()
                    time.sleep(min(2 ** attempt, 30))
                else:
                    errors.append((i, s, e))
        finally:
            close_conn()

    nw = max(1, min(a.workers, q.qsize() or 1))
    worker_fn = worker_keepalive if a.keepalive else worker_curl
    ths = [threading.Thread(target=worker_fn, daemon=True) for _ in range(nw)]
    for t in ths:
        t.start()
    last, last_done = time.time(), 0
    while any(t.is_alive() for t in ths):
        time.sleep(0.5)
        now = time.time()
        if now - last >= 5:
            say(
                "%.1f/%.1f GB, last 5 s %.0f MB/s, avg %.0f MB/s"
                % (done[0] / 1e9, total / 1e9, (done[0] - last_done) / 1e6 / (now - last), done[0] / 1e6 / max(now - t0, 1e-3))
            )
            last, last_done = now, done[0]
    for fd in dest_fds.values():
        os.close(fd)
    dt = max(time.time() - t0, 1e-3)
    if errors:
        say("FAILED %d chunks" % len(errors))
        sys.exit(1)
    mode = "keepalive" if a.keepalive else "curl"
    say("DONE %.2f GB in %.1f s = %.0f MB/s (%s, w=%d, chunk=%dM)" % (done[0] / 1e9, dt, done[0] / 1e6 / dt, mode, nw, a.chunk_mb))
    if a.log:
        log.close()


if __name__ == "__main__":
    main()
