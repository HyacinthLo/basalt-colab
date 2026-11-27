"""Unit tests for basalt_colab config, paths, legacy migration, manifest, usage ledger, colab helpers, and VM payload."""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basalt_colab import colab, config, manifest, paths, proxy, usage  # noqa: E402
from basalt_colab.cli import parse_ask_args  # noqa: E402


class ConfigAndUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="bcolab-unit-"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.p = paths.get_paths(home_override=self.tmp, vm_override=ROOT / "basalt_colab" / "vm")
        self.p.ensure_dirs()
        self.p.key_file.write_text("sk-ds41-" + "a" * 48 + "\n", encoding="utf-8")

    def test_manifest_and_vm_files_in_sync(self) -> None:
        m = manifest.load_manifest(self.p.vm_dir)
        self.assertEqual(m.basalt_tag, "v0.5.7")
        self.assertEqual(m.pack_model_id, config.DEFAULT_MODEL_ALIAS)
        self.assertIsNotNone(m.pack("ud-iq1_m"))
        self.assertIsNotNone(m.pack("ud-iq2_xxs"))
        self.assertIsNotNone(m.pack("ud-q2_k_xl"))
        self.assertIn("UD-IQ2_XXS.basalt", m.pack("ud-iq2_xxs").url)
        self.assertEqual(colab.missing_vm_files(self.p.vm_dir), [])

    def test_config_toml_roundtrip_validation_and_hot_cold_diff(self) -> None:
        cfg = config.Config()
        cfg.basalt.quant = "ud-iq2_xxs"
        cfg.basalt.max_context = 1048576
        cfg.basalt.parallel = 3
        cfg.basalt.kv = "fp16"
        cfg.basalt.effort = "high"
        cfg.basalt.uncensor_scale = 1.2
        cfg.basalt.ref = "main"
        cfg.basalt.sampling.temperature = 0.7
        cfg.basalt.sampling.top_p = 0.95
        config.save_config(cfg, self.p)

        loaded = config.load_config(self.p, apply_env=False)
        self.assertEqual(loaded.to_dict(), cfg.to_dict())
        self.assertEqual(loaded.basalt.effective_model_alias, "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS")
        self.assertEqual(loaded.basalt.max_context, 1048576)
        self.assertEqual(loaded.basalt.sampling.to_basalt_dict(), {"temperature": 0.7, "top-p": 0.95})
        self.assertEqual(loaded.describe_uncensor(), "uncensored ×1.2")

        # Hot vs cold diff
        hot_only = loaded.clone()
        hot_only.basalt.effort = "max"
        hot_only.colab.idle_min = 25
        hot, cold = config.diff_hot_cold(loaded, hot_only)
        self.assertEqual(sorted(hot), ["basalt.effort", "colab.idle_min"])
        self.assertEqual(cold, [])

        cold_change = loaded.clone()
        cold_change.basalt.kv = "fp8"
        cold_change.basalt.uncensor_scale = 1.0
        _, cold2 = config.diff_hot_cold(loaded, cold_change)
        self.assertEqual(sorted(cold2), ["basalt.kv", "basalt.uncensor_scale"])

        # Context & quant parsers
        self.assertEqual(
            [config.context_value(v) for v in ("32k", "262K", "512k", "1M", "5000", "4095", "2000000", "abc")],
            [32768, 262144, 524288, 1048576, 5120, None, None, None],
        )
        self.assertEqual(
            [config.quant_value(v) for v in ("iq1_m", "UD-IQ2_XXS", "q2_k_xl", "custom", "nvfp4")],
            ["ud-iq1_m", "ud-iq2_xxs", "ud-q2_k_xl", "custom", None],
        )

        # Validation rejects out-of-range values
        bad = loaded.clone()
        bad.basalt.parallel = 8
        with self.assertRaises(config.ConfigError):
            config.validate_config(bad)

        bad_kv = loaded.clone()
        bad_kv.basalt.kv = "int4"
        with self.assertRaises(config.ConfigError):
            config.validate_config(bad_kv)

        bad_scale = loaded.clone()
        bad_scale.basalt.uncensor_scale = 2.5
        with self.assertRaises(config.ConfigError):
            config.validate_config(bad_scale)

    def test_boot_vars_and_cmd(self) -> None:
        m = manifest.load_manifest(self.p.vm_dir)
        cfg = config.Config()
        cfg.basalt.quant = "ud-iq2_xxs"
        cfg.basalt.max_context = 524288
        cfg.basalt.uncensor_scale = 1.25
        cfg.basalt.ref = "v0.5.8"
        cfg.basalt.sampling.temperature = 0.8
        bv = cfg.boot_vars(manifest=m)
        self.assertEqual(bv["MODEL_QUANT"], "ud-iq2_xxs")
        self.assertEqual(bv["MODEL_ALIAS"], "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS")
        self.assertIn("UD-IQ2_XXS.basalt", bv["MODEL_URL"])
        self.assertEqual(bv["CTX"], "524288")
        self.assertEqual(bv["RAM_BUDGET"], "0")
        self.assertEqual(bv["UNCENSOR"], "1")
        self.assertEqual(bv["UNCENSOR_SCALE"], "1.25")
        self.assertEqual(bv["BASALT_REF"], "v0.5.8")
        self.assertEqual(bv["VISION_MODE"], "dynamic")
        self.assertEqual(bv["BASALT_SAMPLING_JSON"], '{"temperature":0.8}')
        cmd = colab.boot_cmd(bv)
        self.assertIn("setsid nohup bash vm/boot.sh", cmd)
        self.assertNotIn("libpin.so", cmd)

        # Experimental ud-q2_k_xl requires explicit self-hosted pack_url
        q2 = config.Config()
        q2.basalt.quant = "ud-q2_k_xl"
        with self.assertRaises(config.ConfigError):
            q2.boot_vars(manifest=m)
        q2.basalt.pack_url = "https://huggingface.co/me/ds41/resolve/main/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL.basalt"
        bv2 = q2.boot_vars(manifest=m)
        self.assertEqual(bv2["MODEL_ALIAS"], "DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL")
        self.assertEqual(bv2["MODEL_URL"], q2.basalt.pack_url)

    def test_legacy_state_migration(self) -> None:
        legacy = self.tmp / "old_state"
        legacy.mkdir()
        (legacy / "api_key").write_text("sk-ds41-legacy\n", encoding="utf-8")
        (legacy / "id_ed25519").write_text("PRIVATE-KEY\n", encoding="utf-8")
        new_p = paths.get_paths(home_override=self.tmp / "fresh_home", vm_override=self.p.vm_dir)
        migrated = paths.migrate_legacy_state(new_p, legacy_dir=legacy)
        self.assertEqual(sorted(migrated), ["api_key", "id_ed25519"])
        self.assertEqual(new_p.key_file.read_text(encoding="utf-8"), "sk-ds41-legacy\n")
        if not paths.IS_WIN:
            self.assertEqual(os.stat(new_p.key_file).st_mode & 0o777, 0o600)
        # Idempotent second run migrates nothing
        self.assertEqual(paths.migrate_legacy_state(new_p, legacy_dir=legacy), [])

    def test_usage_ledger_aggregation(self) -> None:
        ledger = usage.UsageLedger(self.p.usage_file)
        now = time.time()
        ledger.record(session="basalt-ds41", started_ts=now - 1800, stopped_ts=now - 600, reason="idle 10.0 min", ran_s=1200, cu_est=2.97, served=14)
        ledger.record(session="basalt-ds41", started_ts=now - 300, stopped_ts=now, reason="down", ran_s=300, cu_est=0.74, served=3)
        s = ledger.summary(days=7, active_ran_s=180, active_cu=0.45)
        self.assertEqual(s["sessions_count"], 2)
        self.assertAlmostEqual(s["today_cu"], 4.16, places=2)
        self.assertEqual(len(s["daily"]), 7)
        self.assertEqual(s["recent"][0]["reason"], "down")

    def test_rewrite_request_and_parse_ask(self) -> None:
        cfg = config.Config()
        self.assertEqual(cfg.basalt.effort, "high")
        # Default "high" matches Basalt/DeepSeek native default (omits redundant injection)
        chat_def = json.loads(proxy.rewrite_request(json.dumps({"model": "claude-haiku-4-5", "messages": []}).encode(), cfg))
        self.assertEqual(chat_def["model"], config.DEFAULT_MODEL_ALIAS)
        self.assertNotIn("reasoning_effort", chat_def)

        # Non-default configured effort ("low", "50", "max") injects into OpenAI chat & responses
        cfg.basalt.effort = "low"
        chat = json.loads(proxy.rewrite_request(json.dumps({"model": "claude-haiku-4-5", "messages": []}).encode(), cfg))
        self.assertEqual((chat["model"], chat["reasoning_effort"]), (config.DEFAULT_MODEL_ALIAS, "low"))
        resp = json.loads(proxy.rewrite_request(json.dumps({"model": "gpt-5", "input": "hi"}).encode(), cfg, "/v1/responses"))
        self.assertEqual((resp["model"], resp["reasoning"]), (config.DEFAULT_MODEL_ALIAS, {"effort": "low"}))
        self.assertNotIn("reasoning_effort", resp)
        anth = json.loads(proxy.rewrite_request(json.dumps({"model": "claude-sonnet-4-5", "messages": []}).encode(), cfg, "/v1/messages"))
        self.assertEqual(anth["model"], config.DEFAULT_MODEL_ALIAS)
        self.assertNotIn("reasoning_effort", anth)
        cfg.proxy.strict_model = True
        strict = json.loads(proxy.rewrite_request(json.dumps({"model": "other", "messages": []}).encode(), cfg))
        self.assertEqual(strict["model"], "other")

        # Official DeepSeek compatibility aliases across all 3 APIs (minimal->low, medium/xhigh->high, ultra->max)
        cfg.proxy.strict_model = False
        cfg.basalt.effort = "high"
        c_med = json.loads(proxy.rewrite_request(
            json.dumps({"model": "x", "reasoning_effort": "medium", "chat_template_kwargs": {"reasoning_effort": "minimal"}, "messages": []}).encode(),
            cfg, "/v1/chat/completions",
        ))
        self.assertEqual(c_med["reasoning_effort"], "high")
        self.assertEqual(c_med["chat_template_kwargs"]["reasoning_effort"], "low")
        r_med = json.loads(proxy.rewrite_request(
            json.dumps({"model": "gpt-5", "reasoning": {"effort": "medium"}, "input": "hi"}).encode(),
            cfg, "/v1/responses",
        ))
        self.assertEqual(r_med["reasoning"]["effort"], "high")
        r_xhi = json.loads(proxy.rewrite_request(
            json.dumps({"model": "gpt-5", "reasoning": {"effort": "xhigh"}, "input": "hi"}).encode(),
            cfg, "/v1/responses",
        ))
        self.assertEqual(r_xhi["reasoning"]["effort"], "high")
        m_ult = json.loads(proxy.rewrite_request(
            json.dumps({"model": "claude", "output_config": {"effort": "ultra"}, "messages": []}).encode(),
            cfg, "/v1/messages",
        ))
        self.assertEqual(m_ult["output_config"]["effort"], "max")

        self.assertEqual(parse_ask_args(["--max", "--image", "a.png", "hi"]), ("max", ["a.png"], "hi"))
        self.assertEqual(parse_ask_args(["--effort", "75", "hello", "world"]), ("75", [], "hello world"))

    def test_bundle_lf_modes_and_log_redaction(self) -> None:
        vm_crlf = self.tmp / "vmcrlf"
        files = colab.vm_files(self.p.vm_dir)
        for f in files:
            dst = vm_crlf / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((self.p.vm_dir / f).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        p_crlf = paths.Paths(self.p.config_dir, self.p.state_dir, vm_crlf)
        out = self.tmp / "b.tgz"
        colab.build_bundle(p_crlf, out)
        with tarfile.open(out) as t:
            names = {m.name: m.mode for m in t.getmembers()}
            body = {m.name: t.extractfile(m).read() for m in t.getmembers()}
        self.assertIn("vm/manifest.json", names)
        self.assertFalse([n for n, b in body.items() if b"\r" in b])
        self.assertEqual(names["secrets/key"], 0o600)

        logger = colab.Logger(self.p, echo=False)
        key = colab.read_api_key(self.p)
        logger.log(f"secret {key} and Authorization: Bearer tok123")
        txt = self.p.log_file.read_text(encoding="utf-8")
        self.assertNotIn(key, txt)
        self.assertNotIn("tok123", txt)

    def test_serve_sh_v2_schema_jobs_and_abliterate(self) -> None:
        serve_sh = (self.p.vm_dir / "serve.sh").read_text(encoding="utf-8")
        m = re.search(r"python3 - \"\$R/basalt_config\.json\" <<'EOF'\n(.*?)\nEOF", serve_sh, re.S)
        self.assertIsNotNone(m)
        cfg_path = self.tmp / "basalt_config.json"
        env = dict(
            os.environ,
            PORT="18081", CTX="1048576", PARALLEL="2", KV_TYPE="fp8", VISION_MODE="dynamic",
            SPEC_MAX="6", DRAFT_EXPERTS="32", VRAM_RESERVE="450", RAM_BUDGET="55", ANSWER_ROOM="0",
            ROUTING_MODE="exact", POOL_WORKERS="0", ENGRAM_CACHE_MIB="1024",
            MODEL_ALIAS=config.DEFAULT_MODEL_ALIAS, CONV_CACHE_MIB="32768", CONV_CACHE_DIR="/content/conv_cache",
            MODEL_PATH="/mnt/ram/model.basalt", ENGRAM_PATH="/mnt/ram/engram.bin",
            BASALT_SAMPLING_JSON='{"temperature":0.7,"top-p":0.95}',
        )
        r = subprocess.run([sys.executable, "-c", m.group(1), str(cfg_path)], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], 2)
        m0 = doc["models"][0]
        self.assertEqual((m0["max-context"], m0["slot-full-context"], m0["kv"], m0["vision"], m0["temperature"], m0["top-p"]),
                         (1048576, True, "fp8", "dynamic", 0.7, 0.95))
        self.assertEqual((m0["answer-room"], m0["idle-unload"]), (0, 0))
        self.assertEqual(doc["engine"]["conversation-cache-disk-mib"], 32768)
        self.assertEqual(doc["engine"]["vram-reserve-mib"], 450)
        self.assertNotIn("route-prior", doc["engine"])
        self.assertNotIn("route-tail-skip", doc["engine"])
        self.assertNotIn("pool-workers", doc["engine"])

        # Opt-in balanced routing sets route-prior=0.1, route-tail-skip=5, and explicit pool-workers
        env_bal = dict(env, ROUTING_MODE="balanced", POOL_WORKERS="12")
        self.assertEqual(subprocess.run([sys.executable, "-c", m.group(1), str(cfg_path)], env=env_bal).returncode, 0)
        doc_bal = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual((doc_bal["engine"]["route-prior"], doc_bal["engine"]["route-tail-skip"]), (0.1, 5))
        self.assertEqual(doc_bal["engine"]["pool-workers"], 12)

        # Solo lossless keeps parallel=1, vram-reserve=400, and zero lossy routing flags
        env_solo = dict(env, ROUTING_MODE="exact", PARALLEL="1", VRAM_RESERVE="400")
        self.assertEqual(subprocess.run([sys.executable, "-c", m.group(1), str(cfg_path)], env=env_solo).returncode, 0)
        doc_s = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual((doc_s["models"][0]["parallel"], doc_s["engine"]["vram-reserve-mib"]), (1, 400))
        self.assertNotIn("route-prior", doc_s["engine"])
        self.assertNotIn("route-tail-skip", doc_s["engine"])

        # Synthetic BASALTMF v2 pack + punch_sections + abliterate_basalt
        sys.path.insert(0, str(self.p.vm_dir))
        try:
            bj = importlib.import_module("basalt_jobs")
            ab = importlib.import_module("abliterate_basalt")
            hdr = bytearray(65536)
            sections = [
                ("meta", 65536, 4096),
                ("gguf", 131072, 1048576),
                ("dspark/gguf", 1179648, 262144),
                ("vision/mmproj", 1441792, 65536),
                ("experts", 2097152, 4194304),
                ("engram", 8388608, 16777216),
            ]
            file_bytes = 8388608 + 16777216
            struct.pack_into("<8sIIQ", hdr, 0, b"BASALTMF", 2, len(sections), file_bytes)
            for i, (name, soff, sbytes) in enumerate(sections):
                struct.pack_into("<64sQQ", hdr, 64 + i * 128, name.encode(), soff, sbytes)
            fb, parsed = bj.parse_header(bytes(hdr))
            bb_jobs, eng_jobs, tail_jobs = bj.build_jobs("https://example.test/m.basalt", fb, parsed, "/mnt/ram/model.basalt", "/mnt/ram/engram.bin")
            self.assertEqual((len(eng_jobs), len(tail_jobs)), (1, 0))
            self.assertEqual(sum(j["size"] for j in bb_jobs), 65536 + 4096 + 1048576 + 262144 + 65536 + 4194304)

            # Split Engram build_jobs produces contiguous head + tail summing to total engram
            _, head_j, split_tail_j = bj.build_jobs(
                "https://example.test/m.basalt", fb, parsed,
                "/mnt/ram/model.basalt", "/mnt/ram/engram.bin",
                engram_tail_dest="/content/engram_tail.bin", engram_head_bytes=12582912,
            )
            self.assertEqual((len(head_j), len(split_tail_j)), (1, 1))
            self.assertEqual(head_j[0]["src_off"], 8388608)
            self.assertEqual(head_j[0]["size"], 12582912)
            self.assertEqual(split_tail_j[0]["src_off"], 8388608 + 12582912)
            self.assertEqual(head_j[0]["size"] + split_tail_j[0]["size"], 16777216)

            if sys.platform.startswith("linux"):
                pf = self.tmp / "punch.basalt"
                with pf.open("wb") as f:
                    f.write(hdr)
                    f.seek(1179648)
                    f.write(b"\xAA" * 262144)
                self.assertEqual(bj.punch_sections(str(pf), {"dspark/gguf"}), 262144)
                ok_p, before_b, after_b, rc_p, _ = bj.test_punch_support(str(self.tmp))
                self.assertTrue(ok_p, f"punch probe failed: before={before_b} after={after_b} rc={rc_p}")

            # Storage planner across real Colab G4 specs (95.6 GiB VRAM, 176.9 GiB RAM, 236 GiB disk)
            GIB = 1024 ** 3
            p_iq1 = bj.plan_storage(int(139.1 * GIB), int(103.0 * GIB), ram_total_gib=176.9, disk_free_gib=236.0, vram_gib=95.6)
            self.assertEqual((p_iq1["stage_mode"], p_iq1["model_path"], p_iq1["engram_path"], p_iq1["engram_tail_path"]),
                             ("ram+ram", "/mnt/ram/model.basalt", "/mnt/ram/engram.bin", ""))
            self.assertEqual(p_iq1["engram_tail_bytes"], 0)

            p_iq2 = bj.plan_storage(int(162.85 * GIB), int(103.0 * GIB), ram_total_gib=176.9, disk_free_gib=236.0, vram_gib=95.6)
            self.assertEqual((p_iq2["stage_mode"], p_iq2["model_path"], p_iq2["engram_path"], p_iq2["engram_tail_path"], p_iq2["parallel_tail"]),
                             ("ram+hybrid", "/mnt/ram/model.basalt", "/mnt/ram/engram.bin", "/content/engram_tail.bin", True))
            self.assertEqual(p_iq2["engram_head_bytes"] % 4096, 0)
            self.assertGreaterEqual(p_iq2["engram_head_bytes"], int(75 * GIB))
            self.assertLessEqual(p_iq2["engram_tail_bytes"], int(28 * GIB))
            self.assertEqual(p_iq2["engram_head_bytes"] + p_iq2["engram_tail_bytes"], int(103.0 * GIB))

            p_q2xl = bj.plan_storage(int(205.0 * GIB), int(103.0 * GIB), ram_total_gib=176.9, disk_free_gib=236.0, vram_gib=95.6)
            self.assertEqual((p_q2xl["stage_mode"], p_q2xl["model_path"], p_q2xl["engram_path"], p_q2xl["engram_tail_path"]),
                             ("ssd+hybrid", "/content/model.basalt", "/mnt/ram/engram.bin", "/content/engram_tail.bin"))
            self.assertGreaterEqual(p_q2xl["engram_head_bytes"], int(30 * GIB))
            self.assertEqual(p_q2xl["engram_head_bytes"] + p_q2xl["engram_tail_bytes"], int(103.0 * GIB))

            with self.assertRaises(RuntimeError):
                bj.plan_storage(int(294.2 * GIB), int(104.4 * GIB), ram_total_gib=176.9, disk_free_gib=236.0, vram_gib=95.6)

            dim = 4
            def npy(shape, fortran, vals):
                h = (f"{{'descr': '<f4', 'fortran_order': {fortran}, 'shape': {shape!r}, }}").encode()
                h = h + b" " * ((64 - (10 + len(h) + 1) % 64) % 64) + b"\n"
                return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(h)) + h + struct.pack(f"<{len(vals)}f", *vals)

            npz = self.tmp / "edit.npz"
            with zipfile.ZipFile(npz, "w") as zf:
                zf.writestr("Q.npy", npy((dim, 2), True, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
                zf.writestr("scales.npy", npy((2, 2), False, [0.5, 0.0, 0.0, 0.75]))
            _, secs = ab.parse_basalt_header(bytes(hdr))
            meta = {"tensors": {
                "blk.0.attn_output_b.weight": {"section": "gguf", "offset": 0, "bytes": 210 * dim, "shape": [256, dim], "type_id": 14},
                "blk.1.ffn_down_shexp.weight": {"section": "gguf", "offset": 4096, "bytes": 176 * dim, "shape": [256, dim], "type_id": 13},
            }}
            _, _, _, plan = ab.build_plan_from_meta("/mnt/ram/model.basalt", secs, meta, str(npz), scale_mult=1.2)
            self.assertEqual([(p[1], p[5], round(p[7], 6)) for p in plan],
                             [("blk.0.attn_output_b.weight", 131072, 0.6),
                              ("blk.1.ffn_down_shexp.weight", 131072 + 4096, 0.9)])
        finally:
            sys.path.remove(str(self.p.vm_dir))

    def test_profiles_and_pfetch_keepalive(self) -> None:
        cfg = config.Config()
        self.assertEqual((cfg.basalt.parallel, cfg.basalt.routing_mode, cfg.basalt.vram_reserve_mib, cfg.basalt.engram_cache_mib),
                         (2, "exact", 450, 1024))
        self.assertIn("route exact", cfg.describe_engine())
        partial = config.parse_toml("[basalt]\nquant = 'ud-iq2_xxs'\n")
        self.assertEqual((partial.basalt.quant, partial.basalt.routing_mode, partial.basalt.vram_reserve_mib, partial.basalt.pool_workers),
                         ("ud-iq2_xxs", "exact", 450, 0))
        cfg.apply_profile("solo")
        self.assertEqual((cfg.basalt.parallel, cfg.basalt.routing_mode, cfg.basalt.vram_reserve_mib), (1, "exact", 400))
        # Legacy alias solo-fast maps to lossless solo
        cfg.apply_profile("balanced")
        self.assertEqual((cfg.basalt.parallel, cfg.basalt.routing_mode, cfg.basalt.vram_reserve_mib), (2, "balanced", 450))
        self.assertIn("route balanced (lossy)", cfg.describe_engine())
        cfg.apply_profile("solo-fast")
        self.assertEqual((cfg.basalt.parallel, cfg.basalt.routing_mode, cfg.basalt.vram_reserve_mib), (1, "exact", 400))
        cfg.apply_profile("exact")
        self.assertEqual((cfg.basalt.parallel, cfg.basalt.routing_mode, cfg.basalt.vram_reserve_mib), (2, "exact", 450))
        with self.assertRaises(config.ConfigError):
            cfg.apply_profile("unknown")

        # pfetch.py runs on the remote Ubuntu Colab VM and uses GNU dd (oflag=seek_bytes)
        if not sys.platform.startswith("linux"):
            return

        # Live local HTTP Range server test for zero-fork Keep-Alive pfetch.py (with retry on transient 503)
        import hashlib
        import http.server
        import threading

        payload = bytes((i * 37 + 11) & 0xFF for i in range(3 * 1024 * 1024 + 12345))
        injected_503 = [True]

        class RangeHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if injected_503[0]:
                    injected_503[0] = False
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                rng = self.headers.get("Range", "")
                m = re.match(r"bytes=(\d+)-(\d+)", rng)
                if not m:
                    self.send_response(400)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                s, e = int(m.group(1)), int(m.group(2))
                chunk = payload[s:e + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {s}-{e}/{len(payload)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            dest = self.tmp / "fetched.bin"
            jobs_file = self.tmp / "jobs.json"
            log_file = self.tmp / "pfetch.log"
            jobs = [
                {"url": f"http://127.0.0.1:{port}/f", "dest": str(dest), "dest_size": len(payload),
                 "src_off": 0, "dest_off": 0, "size": 1024 * 1024},
                {"url": f"http://127.0.0.1:{port}/f", "dest": str(dest), "dest_size": len(payload),
                 "src_off": 1024 * 1024, "dest_off": 1024 * 1024, "size": len(payload) - 1024 * 1024},
            ]
            jobs_file.write_text(json.dumps(jobs), encoding="utf-8")
            r = subprocess.run(
                [sys.executable, str(self.p.vm_dir / "pfetch.py"), str(jobs_file),
                 "--workers", "4", "--chunk-mb", "1", "--min-avail-mib", "1", "--tag", "unit", "--log", str(log_file)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(r.returncode, 0, r.stderr + log_file.read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(), hashlib.sha256(payload).hexdigest())
            self.assertIn("DONE", log_file.read_text(encoding="utf-8"))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_v03_clients_config_get_set_answer_room_and_wait_gpu(self) -> None:
        from basalt_colab import clients
        from basalt_colab.cli import _parse_wait_gpu_args, parse_ask_options

        # Default answer_room is 'auto' (matching Basalt upstream); quant remains ud-iq1_m; idle_min=30m; wait_gpu_min=120m
        cfg = config.Config()
        self.assertEqual(cfg.basalt.quant, "ud-iq1_m")
        self.assertEqual(cfg.basalt.answer_room, "auto")
        self.assertEqual(cfg.colab.idle_min, 30.0)
        self.assertEqual(cfg.colab.wait_gpu_min, 120.0)
        self.assertEqual(cfg.boot_vars(manifest=manifest.load_manifest(self.p.vm_dir))["ANSWER_ROOM"], "auto")

        # config get/set with auto-reflected leaf names & dotted keys
        k, v = config.get_config_value(cfg, "answer_room")
        self.assertEqual((k, v), ("basalt.answer_room", "auto"))
        cfg2, k2, v2 = config.set_config_value(cfg, "answer_room", "2048")
        self.assertEqual((k2, v2), ("basalt.answer_room", "2048"))
        cfg3, k3, v3 = config.set_config_value(cfg2, "wait_gpu_min", "45")
        self.assertEqual((k3, v3), ("colab.wait_gpu_min", 45.0))
        cfg4, k4, v4 = config.set_config_value(cfg3, "temperature", "0.7")
        self.assertEqual((k4, v4), ("basalt.sampling.temperature", 0.7))
        hot, cold = config.diff_hot_cold(cfg2, cfg3)
        self.assertEqual((hot, cold), (["colab.wait_gpu_min"], []))
        with self.assertRaises(config.ConfigError):
            config.set_config_value(cfg, "answer_room", "badval")
        with self.assertRaises(config.ConfigError):
            config.get_config_value(cfg, "nonexistent.key")

        # --wait-gpu optional minutes parser
        self.assertEqual(_parse_wait_gpu_args([]), (False, None))
        self.assertEqual(_parse_wait_gpu_args(["--wait-gpu"]), (True, None))
        self.assertEqual(_parse_wait_gpu_args(["--wait-gpu", "--detach"]), (True, None))
        self.assertEqual(_parse_wait_gpu_args(["--wait-gpu", "60"]), (True, 60.0))
        self.assertEqual(_parse_wait_gpu_args(["--wait-gpu=0"]), (True, 0.0))

        # parse_ask_options extended flags
        opts = parse_ask_options(["--no-think", "--hide-think", "--max-tokens", "512", "--system", "be brief", "hi"])
        self.assertEqual(opts["effort"], "none")
        self.assertTrue(opts["no_think"])
        self.assertTrue(opts["hide_think"])
        self.assertEqual(opts["max_tokens"], 512)
        self.assertEqual(opts["system"], "be brief")
        self.assertEqual(opts["prompt"], "hi")

        # Client config generation (DeepSeek Harness + Claude + Codex + OpenCode + OpenAI)
        cfg_xxs, _, _ = config.set_config_value(cfg, "quant", "ud-iq2_xxs")
        ep = clients.ClientEndpoint.from_config(cfg_xxs, "sk-ds41-testkey", port=18080, paths=self.p)
        all_b = clients.all_client_bundles(ep)
        self.assertEqual([c["id"] for c in all_b["clients"]], ["dsh", "claude", "codex", "opencode", "openai"])
        self.assertEqual(all_b["endpoint"]["model"], "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS")

        dsh_b = clients.client_bundle(ep, "harness")
        self.assertEqual(dsh_b["id"], "dsh")
        self.assertIn("- id: llm-pi-ai", dsh_b["snippet"])
        self.assertIn("apiKeyEnv: BASALT_API_KEY", dsh_b["snippet"])
        self.assertIn("supportsDeveloperRole: false", dsh_b["snippet"])
        self.assertIn("maxTokensField: max_tokens", dsh_b["snippet"])
        self.assertIn("off: none", dsh_b["snippet"])
        self.assertIn("reasoning: high", dsh_b["snippet"])
        # Regression check: Basalt DeepSeek-V4.1 rejects 'medium'
        self.assertNotIn("medium", dsh_b["snippet"])
        self.assertIn("input: [text, image]", dsh_b["snippet"])
        self.assertIn("id: DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", dsh_b["snippet"])

        # Vision off omits image modality in dsh YAML
        cfg_novis, _, _ = config.set_config_value(cfg_xxs, "vision", "off")
        ep_nv = clients.ClientEndpoint.from_config(cfg_novis, "sk-ds41-testkey", port=18080, paths=self.p)
        self.assertIn("input: [text]", clients.dsh_patch_yaml(ep_nv))
        self.assertNotIn("image", clients.dsh_patch_yaml(ep_nv))

        # Claude, Codex, OpenCode snippets (OpenCode snippet must NOT embed plaintext key)
        self.assertIn("ANTHROPIC_BASE_URL", clients.client_bundle(ep, "claude")["snippet"])
        self.assertIn('wire_api = "responses"', clients.client_bundle(ep, "codex")["snippet"])
        oc = json.loads(clients.client_bundle(ep, "opencode")["snippet"])
        self.assertEqual(oc["model"], "basalt/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS")
        self.assertEqual(oc["provider"]["basalt"]["options"]["apiKey"], "{env:OPENAI_API_KEY}")
        self.assertEqual(clients.dsh_effort_default("50"), "low")
        self.assertEqual(clients.dsh_effort_default("high"), "high")
        self.assertEqual(clients.dsh_effort_default("95"), "max")
        self.assertEqual(clients.dsh_effort_default("20"), "low")

    def test_kernel_lock_argparse_and_static_console(self) -> None:
        from basalt_colab import console
        from basalt_colab.cli import _WAIT_GPU_DEFAULT, build_parser

        # Static web/console.html loaded cleanly via importlib.resources
        html = console.render_console_html()
        self.assertIn(b"<!doctype html>", html.lower())
        self.assertIn(b"/_control/events", html)
        self.assertIn(b"basalt-colab", html)

        # Strict argparse subparsers: valid combos & typo rejection
        parser = build_parser()
        ns1 = parser.parse_args(["up", "--quant", "ud-iq2_xxs", "--wait-gpu", "--detach"])
        self.assertEqual((ns1.cmd, ns1.quant, ns1.detach), ("up", "ud-iq2_xxs", True))
        self.assertIs(ns1.wait_gpu, _WAIT_GPU_DEFAULT)
        ns2 = parser.parse_args(["up", "--wait-gpu", "45"])
        self.assertEqual(ns2.wait_gpu, 45.0)
        ns3 = parser.parse_args(["up", "--wait-gpu=0"])
        self.assertEqual(ns3.wait_gpu, 0.0)
        ns4 = parser.parse_args(["ask", "--effort", "high", "--max-tokens", "256", "hello", "world"])
        self.assertEqual((ns4.effort, ns4.max_tokens, ns4.prompt), ("high", 256, ["hello", "world"]))

        env = dict(os.environ, PYTHONPATH=str(ROOT))
        r_typo = subprocess.run(
            [sys.executable, "-m", "basalt_colab", "up", "--qunat", "ud-iq2_xxs"],
            env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(r_typo.returncode, 2)
        self.assertIn("unrecognized arguments", r_typo.stderr)
        r_help = subprocess.run(
            [sys.executable, "-m", "basalt_colab", "up", "--help"],
            env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(r_help.returncode, 0)
        self.assertIn("--wait-gpu", r_help.stdout)

        # Cross-platform OS kernel file lock (fcntl.flock / msvcrt.locking):
        # survives concurrent contention and releases instantly on hard kill (SIGKILL / TerminateProcess)
        lock_home = self.tmp / "lock_home"
        lp = paths.get_paths(home_override=lock_home, vm_override=self.p.vm_dir)
        lp.ensure_dirs()
        holder_code = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "from basalt_colab import colab, paths\n"
            "p = paths.get_paths(home_override=Path(sys.argv[1]), vm_override=Path(sys.argv[2]))\n"
            "assert colab.take_lock(p, {'port': 18999}) is None\n"
            "print('LOCKED', flush=True)\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", holder_code, str(lock_home), str(self.p.vm_dir)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            assert proc.stdout is not None
            line = proc.stdout.readline().strip()
            if line != "LOCKED":
                err = proc.stderr.read() if proc.stderr else ""
                self.fail(f"lock holder did not print LOCKED (got {line!r}): {err}")
            self.assertTrue(colab.is_lock_held(lp))
            running = colab.serve_running(lp)
            self.assertIsNotNone(running)
            self.assertEqual(running["port"], 18999)
            # Second caller cannot steal the lock and sees holder metadata
            contended = colab.take_lock(lp, {"port": 19000})
            self.assertIsNotNone(contended)
            self.assertEqual(contended["port"], 18999)
        finally:
            proc.kill()
            proc.wait(timeout=10)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

        # Even though serve.lock JSON remains on disk after hard kill, kernel released serve.lck immediately
        self.assertTrue(lp.lock_file.exists())
        self.assertFalse(colab.is_lock_held(lp))
        self.assertIsNone(colab.serve_running(lp))
        # Parent can immediately take and cleanly release the lock
        self.assertIsNone(colab.take_lock(lp, {"port": 19001}))
        self.assertTrue(colab.is_lock_held(lp))
        self.assertEqual(colab.serve_running(lp)["port"], 19001)
        colab.release_lock(lp)
        self.assertFalse(colab.is_lock_held(lp))

    def test_boot_overrides_hot_fields_meta_and_vm_state(self) -> None:
        from basalt_colab.lifecycle import VmState

        # VmState StrEnum serializes cleanly as plain JSON strings
        self.assertEqual(json.dumps({"state": VmState.READY}), '{"state": "ready"}')
        self.assertEqual(VmState.OFF, "off")

        # BootOverrides single-source validation + serialization + apply_to
        m = manifest.load_manifest(self.p.vm_dir)
        cfg = config.Config()
        ov = config.BootOverrides.from_dict({
            "wait_gpu": True,
            "wait_gpu_min": 45,
            "profile": "solo",
            "quant": "ud-iq2_xxs",
            "uncensor_scale": 1.2,
            "basalt_ref": "main",
        })
        applied = ov.apply_to(cfg, manifest=m)
        self.assertEqual((applied.basalt.parallel, applied.basalt.routing_mode, applied.basalt.vram_reserve_mib), (1, "exact", 400))
        self.assertEqual((applied.basalt.quant, applied.basalt.uncensor_scale, applied.basalt.ref), ("ud-iq2_xxs", 1.2, "main"))
        self.assertEqual(ov.resolve_wait_gpu(cfg), (True, 45.0))
        self.assertEqual(ov.to_dict()["quant"], "ud-iq2_xxs")
        # Original cfg is untouched (immutable apply_to)
        self.assertEqual(cfg.basalt.quant, "ud-iq1_m")
        with self.assertRaises(config.ConfigError):
            config.BootOverrides(quant="bogus").apply_to(cfg, manifest=m)

        # HOT_FIELDS-driven apply_hot_from, hot_snapshot, and overlay_hot_dict
        overlaid = cfg.overlay_hot_dict({"effort": "max", "idle_min": 45, "orphan_boot_s": 60, "strict_model": True})
        dst = config.Config()
        dst.apply_hot_from(overlaid)
        snap = dst.hot_snapshot()
        self.assertEqual((snap["effort"], snap["idle_min"], snap["orphan_boot_s"], snap["strict_model"]), ("max", 45.0, 60.0, True))
        self.assertEqual(set(snap.keys()), {k.split(".", 1)[1] for k in config.HOT_FIELDS})

        # Shared Console schema metadata
        meta = config.config_meta()
        self.assertEqual(meta["quants"], list(config.VALID_QUANTS))
        self.assertEqual(set(meta["profiles"].keys()), set(config.PERFORMANCE_PROFILES.keys()))
        self.assertEqual(meta["hot_fields"], sorted(config.HOT_FIELDS))

    def test_colab_cli_supported_list_and_doctor_version_check(self) -> None:
        import importlib.util
        import stat
        from io import StringIO
        from unittest import mock

        from basalt_colab import cli

        m = manifest.load_manifest(self.p.vm_dir)
        pinned = m.colab_cli_pinned_version()
        self.assertEqual(pinned, "0.8.0")
        self.assertIn(pinned, m.colab_cli_supported)
        self.assertTrue(m.is_colab_cli_supported("0.7.5"))
        self.assertTrue(m.is_colab_cli_supported("0.7.6"))
        self.assertTrue(m.is_colab_cli_supported("0.8.0"))
        self.assertFalse(m.is_colab_cli_supported("0.9.0"))
        self.assertFalse(m.is_colab_cli_supported(None))

        self.assertEqual(colab.parse_colab_version("Version: 0.7.5\n"), "0.7.5")
        self.assertEqual(colab.parse_colab_version("Version: 0.8.0"), "0.8.0")
        self.assertEqual(colab.parse_colab_version("1.0.0rc1"), "1.0.0rc1")
        self.assertIsNone(colab.parse_colab_version("no version here"))

        # Prepare minimal credentials so doctor checks pass aside from what we test
        self.p.ensure_dirs()
        self.p.key_file.write_text("sk-test\n", encoding="utf-8")
        self.p.ssh_key.write_text("KEY\n", encoding="utf-8")
        cfg = config.Config()
        cfg.proxy.port = 0

        def run_doctor(detected: str | None, man: manifest.Manifest) -> str:
            buf = StringIO()
            with (
                mock.patch.object(cli.shutil, "which", return_value="/usr/bin/ssh"),
                mock.patch.object(cli, "colab_present", return_value=True),
                mock.patch.object(cli, "colab_cmd", return_value=["/usr/bin/colab", "--auth", "oauth2"]),
                mock.patch.object(cli, "colab_version", return_value=detected),
                mock.patch.object(cli, "colab_usage", return_value=(100.0, 0.0)),
                mock.patch.object(cli, "serve_running", return_value={"port": 18080, "pid": 123}),
                mock.patch("sys.stdout", buf),
            ):
                cli.cmd_doctor(self.p, cfg, man)
            return buf.getvalue()

        out_ok = run_doctor(pinned, m)
        self.assertIn(f"(v{pinned}, verified)", out_ok)

        out_older = run_doctor("0.7.5", m)
        self.assertIn(f"(v0.7.5, supported; latest verified: v{pinned})", out_older)

        out_unverified = run_doctor("0.9.0", m)
        self.assertIn("v0.9.0, unverified", out_unverified)
        self.assertIn(f"uv tool install --force {m.colab_cli}", out_unverified)

        # colab_env() injects Linux headless keyring shim unless user overrides PYTHON_KEYRING_BACKEND
        with mock.patch.object(colab.sys, "platform", "linux"), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHON_KEYRING_BACKEND", None)
            os.environ.pop("BASALT_COLAB_KEYRING_FILE", None)
            env_auto = colab.colab_env()
            self.assertEqual(env_auto["PYTHON_KEYRING_BACKEND"], "basalt_colab_keyring.FallbackFileKeyring")
            self.assertIn(str(colab.KEYRING_SHIM_DIR), env_auto["PYTHONPATH"].split(os.pathsep))
            self.assertTrue(env_auto["BASALT_COLAB_KEYRING_FILE"].endswith("colab_keyring.json"))

            env_custom = colab.colab_env({"PYTHON_KEYRING_BACKEND": "custom.Backend"})
            self.assertEqual(env_custom["PYTHON_KEYRING_BACKEND"], "custom.Backend")

        # FallbackFileKeyring: 0600 file store + native delegation + error fallback
        shim_file = colab.KEYRING_SHIM_DIR / "basalt_colab_keyring.py"
        spec = importlib.util.spec_from_file_location("basalt_colab_keyring_test", shim_file)
        assert spec is not None and spec.loader is not None
        kr_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kr_mod)
        kr = kr_mod.FallbackFileKeyring()

        store_file = self.tmp / "kr_store" / "colab_keyring.json"
        with mock.patch.dict(os.environ, {"BASALT_COLAB_KEYRING_FILE": str(store_file)}, clear=False):
            os.environ.pop("DBUS_SESSION_BUS_ADDRESS", None)
            self.assertIsNone(kr.get_password("google-colab-cli", "oauth_token"))
            kr.set_password("google-colab-cli", "oauth_token", '{"token":"abc"}')
            self.assertEqual(kr.get_password("google-colab-cli", "oauth_token"), '{"token":"abc"}')
            if not paths.IS_WIN:
                self.assertEqual(stat.S_IMODE(store_file.stat().st_mode), 0o600)
            kr.delete_password("google-colab-cli", "oauth_token")
            self.assertIsNone(kr.get_password("google-colab-cli", "oauth_token"))

            # When a native backend works, it is preferred over the file store
            native_mem: dict[tuple[str, str], str] = {}

            class WorkingNative:
                def get_password(self, s: str, u: str) -> str | None:
                    return native_mem.get((s, u))

                def set_password(self, s: str, u: str, p: str) -> None:
                    native_mem[(s, u)] = p

                def delete_password(self, s: str, u: str) -> None:
                    native_mem.pop((s, u), None)

            with mock.patch.object(kr_mod, "_native_backends", return_value=[WorkingNative()]):
                kr.set_password("google-colab-cli", "oauth_token", "native-tok")
                self.assertEqual(kr.get_password("google-colab-cli", "oauth_token"), "native-tok")
                self.assertEqual(native_mem[("google-colab-cli", "oauth_token")], "native-tok")

            # When a native backend raises, seamlessly falls back to file store
            class BrokenNative:
                def get_password(self, s: str, u: str) -> str | None:
                    raise RuntimeError("locked")

                def set_password(self, s: str, u: str, p: str) -> None:
                    raise RuntimeError("locked")

            with mock.patch.object(kr_mod, "_native_backends", return_value=[BrokenNative()]):
                kr.set_password("google-colab-cli", "oauth_token", "fallback-tok")
                self.assertEqual(kr.get_password("google-colab-cli", "oauth_token"), "fallback-tok")


if __name__ == "__main__":
    unittest.main()
