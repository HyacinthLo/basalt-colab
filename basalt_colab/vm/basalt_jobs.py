#!/usr/bin/env python3
"""Generate pfetch.py HTTP Range job JSONs and storage plans for Basalt DeepSeek-V4.1-Flash packs.

Reads the 64 KiB BASALTMF v2 header (and small `meta` section for hybrid packs) via HTTP Range
and plans zero-SSD-expert staging (honoring Stage 0 `--ssd-dir` when fast local NVMe is present):
  - ram+ram (e.g. UD-IQ1_M, 139.1 + 103.0 GiB):
      Backbone -> /mnt/ram/model.basalt (punch-on-read) + 100% Engram -> /mnt/ram/engram.bin
  - ram+hybrid (e.g. UD-IQ2_XXS, 162.85 + 103.0 GiB):
      Backbone -> /mnt/ram/model.basalt (punch-on-read)
      Engram RAM windows (~83-86 GiB: Layer 1 all + Layer 14 3/4-gram) -> /mnt/ram/engram_*.bin
      Engram sparse SSD (~17-20 GiB Layer 14 2-gram, +4096 B boundary overlap) -> $SSD_DIR/engram_tail.bin (parallel at t=2s!)
  - ssd+hybrid / ssd+ssd (e.g. UD-Q2_K_XL, ~194.3 + 125.9 GiB Q5_K):
      Backbone -> $SSD_DIR/model.basalt (punch-on-read)
      Engram RAM window (~42 GiB Layer 1 3/4-gram) -> /mnt/ram/engram_0.bin + sparse SSD (~84 GiB) -> $SSD_DIR/engram_tail.bin
  - Fail-fast guard rejects packs whose backbone exceeds zero-SSD-expert VRAM+RAM capacity (e.g. NVFP4).
"""
import argparse, ctypes, json, os, shlex, struct, sys, tempfile, urllib.request

PACKS = {
    'iq1_m': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt',
    'ud-iq1_m': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt',
    'iq2_xxs': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.basalt',
    'ud-iq2_xxs': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.basalt',
}

HEADER_BYTES = 65536
GIB = 1024 ** 3
SPLIT_ALIGN = 64 * 1024 * 1024  # 64 MiB (page-aligned + pfetch chunk-aligned)
PAGE_BYTES = 4096


def _range_get(url, start, end, hdr_file=None, timeout=60):
    headers = {'Range': f'bytes={int(start)}-{int(end)}'}
    if hdr_file and os.path.exists(hdr_file):
        with open(hdr_file) as f:
            for line in f:
                k, _, v = line.strip().partition(':')
                if k:
                    headers[k.strip()] = v.strip()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_header(data):
    assert len(data) == HEADER_BYTES, f'expected {HEADER_BYTES} header bytes, got {len(data)}'
    magic, ver, nsec, file_bytes = struct.unpack('<8sIIQ', data[:24])
    assert magic == b'BASALTMF' and ver == 2, f'bad basalt header: magic={magic!r} ver={ver}'
    sections = []
    for i in range(nsec):
        off = 64 + i * 128
        name_b, soff, sbytes, sha = struct.unpack('<64sQQ32s', data[off:off + 112])
        name = name_b.split(b'\x00', 1)[0].decode('utf-8')
        sections.append({'name': name, 'offset': soff, 'bytes': sbytes, 'sha256': sha.hex()})
    return file_bytes, sections


def read_header(url, hdr_file=None):
    return parse_header(_range_get(url, 0, HEADER_BYTES - 1, hdr_file=hdr_file))


def read_meta(url, sections, hdr_file=None):
    """Fetches and parses the small BASALTMF v2 `meta` JSON section via HTTP Range."""
    meta_sec = next((s for s in sections if s['name'] == 'meta' and s['bytes'] > 0), None)
    if meta_sec is None:
        return {}
    try:
        raw = _range_get(url, meta_sec['offset'], meta_sec['offset'] + meta_sec['bytes'] - 1, hdr_file=hdr_file)
        return json.loads(raw.decode('utf-8'))
    except Exception:
        return {}


def extract_engram_subtables(meta, engram_bytes):
    """Extracts per-layer, per-ngram-order Engram byte/row spans from BASALTMF v2 `meta`."""
    eng = (meta or {}).get('engram')
    if not isinstance(eng, dict):
        return None
    layers = eng.get('ple.layers')
    ngram_size = int(eng.get('ple.ngram_size') or 0)
    heads_per_ngram = int(eng.get('ple.heads_per_ngram') or 0)
    head_offsets = eng.get('ple.head_offsets')
    head_vocabs = eng.get('ple.head_vocab_sizes')
    row_bytes = int(eng.get('row_bytes') or 0)
    orders_per_layer = ngram_size - 1
    if (
        not isinstance(layers, list)
        or not layers
        or orders_per_layer < 1
        or heads_per_ngram < 1
        or row_bytes <= 0
        or not isinstance(head_offsets, list)
        or not isinstance(head_vocabs, list)
    ):
        return None
    expected_heads = len(layers) * orders_per_layer * heads_per_ngram
    if len(head_offsets) != expected_heads or len(head_vocabs) != expected_heads:
        return None
    total_rows = int(eng.get('rows') or (int(engram_bytes) // row_bytes))
    if total_rows * row_bytes != int(engram_bytes):
        return None

    subtables = []
    idx = 0
    for li, layer in enumerate(layers):
        for oi in range(orders_per_layer):
            first_h = idx
            last_h = idx + heads_per_ngram - 1
            row_lo = int(head_offsets[first_h])
            row_hi = int(head_offsets[last_h]) + int(head_vocabs[last_h])
            subtables.append({
                'layer': int(layer),
                'layer_idx': li,
                'ngram': 2 + oi,
                'row_lo': row_lo,
                'row_hi': row_hi,
                'byte_lo': row_lo * row_bytes,
                'byte_hi': row_hi * row_bytes,
            })
            idx += heads_per_ngram
    return {
        'row_bytes': row_bytes,
        'total_rows': total_rows,
        'subtables': subtables,
    }


def _merge_ranges(ranges):
    pts = sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a))
    if not pts:
        return []
    out = [list(pts[0])]
    for a, b in pts[1:]:
        if a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract_ranges(want, have):
    """Returns sorted disjoint ranges in `want` not already covered by `have`."""
    want_m = _merge_ranges(want)
    have_m = _merge_ranges(have)
    out = []
    hi = 0
    for a, b in want_m:
        while hi < len(have_m) and have_m[hi][1] <= a:
            hi += 1
        cur = a
        j = hi
        while j < len(have_m) and have_m[j][0] < b:
            if have_m[j][0] > cur:
                out.append((cur, have_m[j][0]))
            cur = max(cur, have_m[j][1])
            if cur >= b:
                break
            j += 1
        if cur < b:
            out.append((cur, b))
    return out


def _ssd_complement(engram_bytes, ram_windows):
    """Complement of `ram_windows` in [0, engram_bytes) with +4096 B interior boundary overlap."""
    eng = int(engram_bytes)
    wins = _merge_ranges(ram_windows)
    raw = []
    cur = 0
    for lo, hi in wins:
        if lo > cur:
            raw.append((cur, lo))
        cur = max(cur, hi)
    if cur < eng:
        raw.append((cur, eng))
    expanded = []
    for a, b in raw:
        ea = max(0, a - PAGE_BYTES) if a > 0 else 0
        eb = min(eng, b + PAGE_BYTES) if b < eng else eng
        expanded.append((ea, eb))
    return _merge_ranges(expanded)


def plan_engram_windows(engram_bytes, avail_ram_gib, engram_info=None, reserve_gib=5.0, min_ram_gib=16.0):
    """Plans 64 MiB-aligned RAM mmap windows, +4096 B-protected sparse SSD ranges, and 4-gram RowCache skip rows.

    Priority order: (layer_idx, -ngram) — Layer 1 4-gram/3-gram/2-gram first (zero layer-hiding before Layer 1),
    then Layer 14 4-gram/3-gram/2-gram (2-gram is hottest in 4 GiB RowCache and last to place in RAM).
    """
    eng = int(engram_bytes)
    usable_bytes = int(max(0.0, float(avail_ram_gib) - float(reserve_gib)) * GIB)
    subtables = (engram_info or {}).get('subtables') or []

    if usable_bytes >= eng:
        return {
            'ram_windows': [(0, eng)],
            'ssd_ranges': [],
            'cache_skip_rows': [],
            'ram_bytes': eng,
            'ssd_bytes': 0,
        }

    budget_bytes = (usable_bytes // SPLIT_ALIGN) * SPLIT_ALIGN
    if budget_bytes < int(min_ram_gib * GIB):
        ram_windows = []
    elif subtables:
        ordered = sorted(subtables, key=lambda st: (st['layer_idx'], -st['ngram']))
        selected = []
        for st in ordered:
            cand_lo = (int(st['byte_lo']) // SPLIT_ALIGN) * SPLIT_ALIGN
            cand_hi = min(eng, ((int(st['byte_hi']) + SPLIT_ALIGN - 1) // SPLIT_ALIGN) * SPLIT_ALIGN)
            merged = _merge_ranges(selected + [(cand_lo, cand_hi)])
            if sum(h - l for l, h in merged) <= budget_bytes:
                selected = merged
                continue
            # Extend an adjacent RAM window into cold 3/4-gram subtables without fragmenting or chewing 2-gram headroom.
            if int(st['ngram']) >= 3:
                cur_bytes = sum(h - l for l, h in selected)
                rem = ((budget_bytes - cur_bytes) // SPLIT_ALIGN) * SPLIT_ALIGN
                if rem >= SPLIT_ALIGN:
                    touch_hi = any(l <= cand_hi <= h for l, h in selected)
                    touch_lo = any(l <= cand_lo <= h for l, h in selected)
                    if touch_hi or touch_lo:
                        if touch_hi:
                            p_lo = max(cand_lo, ((cand_hi - rem) // SPLIT_ALIGN) * SPLIT_ALIGN)
                            p_hi = cand_hi
                        else:
                            p_lo = cand_lo
                            p_hi = min(cand_hi, cand_lo + rem)
                        trial = _merge_ranges(selected + [(p_lo, p_hi)])
                        if sum(h - l for l, h in trial) <= budget_bytes:
                            selected = trial
        ram_windows = selected
    else:
        head = min(budget_bytes, ((eng - GIB) // SPLIT_ALIGN) * SPLIT_ALIGN)
        ram_windows = [(0, head)] if head >= int(min_ram_gib * GIB) else []

    ssd_ranges = _ssd_complement(eng, ram_windows)

    cache_skip_rows = []
    for st in subtables:
        if int(st['ngram']) >= 4:
            blo, bhi = int(st['byte_lo']), int(st['byte_hi'])
            fully_in_ram = any(wlo <= blo and bhi <= whi for wlo, whi in ram_windows)
            if not fully_in_ram:
                cache_skip_rows.append((int(st['row_lo']), int(st['row_hi'])))

    return {
        'ram_windows': ram_windows,
        'ssd_ranges': ssd_ranges,
        'cache_skip_rows': cache_skip_rows,
        'ram_bytes': sum(h - l for l, h in ram_windows),
        'ssd_bytes': sum(h - l for l, h in ssd_ranges),
    }


def build_engram_window_jobs(url, eng_offset, eng_bytes, ram_windows, ssd_ranges, ram_dir='/mnt/ram', ssd_file=''):
    """Builds compact RAM window jobs (`engram_0.bin`, ...) and logical full-size sparse SSD jobs."""
    ram_base = (ram_dir or '/mnt/ram').rstrip('/') or '/mnt/ram'
    ram_jobs = []
    ram_layout = []
    for idx, (lo, hi) in enumerate(ram_windows):
        dest = f'{ram_base}/engram_{idx}.bin'
        ram_jobs.append({
            'url': url,
            'dest': dest,
            'dest_size': int(hi) - int(lo),
            'src_off': int(eng_offset) + int(lo),
            'dest_off': 0,
            'size': int(hi) - int(lo),
        })
        ram_layout.append((dest, int(lo), int(hi)))

    ssd_jobs = []
    if ssd_file:
        for lo, hi in ssd_ranges:
            ssd_jobs.append({
                'url': url,
                'dest': ssd_file,
                'dest_size': int(eng_bytes),
                'src_off': int(eng_offset) + int(lo),
                'dest_off': int(lo),
                'size': int(hi) - int(lo),
            })
    return ram_jobs, ssd_jobs, ram_layout


def format_layout_text(ram_layout, ssd_file='', cache_skip_rows=()):
    lines = ['# basalt-colab multi-window Engram layout (PleReader BASALT_ENGRAM_LAYOUT)']
    if ssd_file:
        lines.append(f'ssd {ssd_file}')
    for path, lo, hi in ram_layout:
        lines.append(f'ram {path} {int(lo)} {int(hi)}')
    for rlo, rhi in cache_skip_rows:
        lines.append(f'cache_skip {int(rlo)} {int(rhi)}')
    return '\n'.join(lines) + '\n'


def finalize_engram_layout(plan_doc, mem_avail_gib, ram_dir, ssd_file, reserve_gib=5.0):
    """Post-/load finalizer: sizes RAM windows from real MemAvailable and computes any extra SSD delta."""
    url = plan_doc['url']
    eng_offset = int(plan_doc['eng_offset'])
    eng_bytes = int(plan_doc['eng_bytes'])
    engram_info = plan_doc.get('engram_info')
    initial_ssd = [tuple(r) for r in (plan_doc.get('initial_ssd_ranges') or [])]

    win_plan = plan_engram_windows(eng_bytes, mem_avail_gib, engram_info=engram_info, reserve_gib=reserve_gib)
    final_ssd = win_plan['ssd_ranges']
    ssd_extra = subtract_ranges(final_ssd, initial_ssd) if final_ssd else []

    ram_jobs, ssd_extra_jobs, ram_layout = build_engram_window_jobs(
        url, eng_offset, eng_bytes, win_plan['ram_windows'], ssd_extra, ram_dir=ram_dir, ssd_file=ssd_file,
    )
    layout_text = format_layout_text(
        ram_layout,
        ssd_file=ssd_file if final_ssd else '',
        cache_skip_rows=win_plan['cache_skip_rows'],
    )
    return {
        'plan': win_plan,
        'ram_jobs': ram_jobs,
        'ssd_extra_jobs': ssd_extra_jobs,
        'ssd_extra_ranges': ssd_extra,
        'has_ssd': bool(final_ssd),
        'layout_text': layout_text,
    }


def summarize_sections(sections):
    bb_bytes = HEADER_BYTES
    eng_bytes = 0
    for s in sections:
        if s['bytes'] == 0:
            continue
        if s['name'] == 'engram':
            eng_bytes += s['bytes']
        else:
            bb_bytes += s['bytes']
    return bb_bytes, eng_bytes


def _split_engram_bytes(engram_bytes, ram_total_gib, ram_budget_gib, reserve_gib=14.0, min_head_gib=16.0):
    """Returns (head_ram_bytes, tail_ssd_bytes) with head aligned to 64 MiB (0 if not worth splitting)."""
    avail_gib = float(ram_total_gib) - float(ram_budget_gib) - float(reserve_gib)
    if avail_gib < min_head_gib:
        return 0, int(engram_bytes)
    head = (int(avail_gib * GIB) // SPLIT_ALIGN) * SPLIT_ALIGN
    # Keep at least 1 GiB tail if splitting; otherwise full RAM should already have been selected.
    max_head = ((int(engram_bytes) - GIB) // SPLIT_ALIGN) * SPLIT_ALIGN
    head = min(head, max(0, max_head))
    if head < int(min_head_gib * GIB):
        return 0, int(engram_bytes)
    return head, int(engram_bytes) - head


def plan_storage(backbone_bytes, engram_bytes, ram_total_gib, disk_free_gib, vram_gib=96.0, explicit_budget_gib=0, ssd_dir='/content'):
    """Computes zero-SSD-expert staging plan (ram+ram, ram+hybrid/ram+ssd, or ssd+hybrid/ssd+ssd)."""
    ssd_dir = (ssd_dir or '/content').rstrip('/') or '/content'
    bb_gib = backbone_bytes / GIB
    eng_gib = engram_bytes / GIB
    # Conservative usable VRAM for weights after drafter (2.4 GiB), KV, workspace, and reserve.
    vram_weights_gib = max(0.0, float(vram_gib) - 10.0)
    ram_needed_gib = max(0.0, bb_gib - vram_weights_gib)
    max_ram_experts_gib = max(0.0, float(ram_total_gib) - 22.0)

    if ram_needed_gib > max_ram_experts_gib:
        max_bb = vram_weights_gib + max_ram_experts_gib
        raise RuntimeError(
            f'pack backbone ({bb_gib:.1f} GiB) exceeds zero-SSD-expert capacity on this VM '
            f'(VRAM {vram_gib:.0f} GiB + RAM {ram_total_gib:.0f} GiB -> max ~{max_bb:.0f} GiB backbone; '
            f'spilling experts to Colab SSD degrades decode to ~2-5 tok/s)'
        )

    bb_fits_tmpfs = bb_gib <= (float(ram_total_gib) - 12.0)
    both_fit_ram = bb_fits_tmpfs and ((ram_needed_gib + eng_gib) <= (float(ram_total_gib) - 8.0))

    engram_tail_path = ''
    engram_head_bytes = int(engram_bytes)
    engram_tail_bytes = 0
    parallel_tail = False

    if both_fit_ram:
        stage_mode = 'ram+ram'
        model_path = '/mnt/ram/model.basalt'
        engram_path = '/mnt/ram/engram.bin'
        punch_file = '/mnt/ram/model.basalt'
        auto_budget = max(55, min(160, int(ram_needed_gib + 4)))
        ram_budget = int(explicit_budget_gib) if int(explicit_budget_gib) > 0 else int(auto_budget)
    elif bb_fits_tmpfs:
        model_path = '/mnt/ram/model.basalt'
        punch_file = '/mnt/ram/model.basalt'
        auto_budget = min(max(55, int(ram_total_gib - 30)), max(55, int(ram_needed_gib + 8)))
        ram_budget = int(explicit_budget_gib) if int(explicit_budget_gib) > 0 else int(auto_budget)
        head_b, tail_b = _split_engram_bytes(engram_bytes, ram_total_gib, ram_budget)
        if head_b > 0:
            if tail_b / GIB > float(disk_free_gib) - 6.0:
                raise RuntimeError(
                    f'Engram tail ({tail_b / GIB:.1f} GiB) does not fit in {ssd_dir} disk ({disk_free_gib:.1f} GiB free)'
                )
            stage_mode = 'ram+hybrid'
            engram_path = '/mnt/ram/engram.bin'
            engram_tail_path = f'{ssd_dir}/engram_tail.bin'
            engram_head_bytes = head_b
            engram_tail_bytes = tail_b
            parallel_tail = True
        else:
            if eng_gib > float(disk_free_gib) - 6.0:
                raise RuntimeError(
                    f'Engram ({eng_gib:.1f} GiB) does not fit in {ssd_dir} disk ({disk_free_gib:.1f} GiB free)'
                )
            stage_mode = 'ram+ssd'
            engram_path = f'{ssd_dir}/engram.bin'
            engram_head_bytes = 0
            engram_tail_bytes = int(engram_bytes)
            parallel_tail = True
    else:
        if bb_gib > float(disk_free_gib) - 6.0:
            raise RuntimeError(
                f'pack backbone ({bb_gib:.1f} GiB) exceeds both /mnt/ram ({ram_total_gib:.0f} GiB) '
                f'and {ssd_dir} disk ({disk_free_gib:.1f} GiB free)'
            )
        model_path = f'{ssd_dir}/model.basalt'
        punch_file = f'{ssd_dir}/model.basalt'
        auto_budget = min(max(55, int(ram_total_gib - 24)), max(100, int(ram_needed_gib + 8)))
        ram_budget = int(explicit_budget_gib) if int(explicit_budget_gib) > 0 else int(auto_budget)
        head_b, tail_b = _split_engram_bytes(engram_bytes, ram_total_gib, ram_budget)
        if head_b > 0:
            if tail_b / GIB > float(disk_free_gib) - 8.0:
                raise RuntimeError(
                    f'Engram tail ({tail_b / GIB:.1f} GiB) does not fit in {ssd_dir} disk ({disk_free_gib:.1f} GiB free)'
                )
            stage_mode = 'ssd+hybrid'
            engram_path = '/mnt/ram/engram.bin'
            engram_tail_path = f'{ssd_dir}/engram_tail.bin'
            engram_head_bytes = head_b
            engram_tail_bytes = tail_b
            # Backbone already saturates /content's ~288 MiB/s write cap; download tail + RAM head together after /load punches backbone.
            parallel_tail = False
        else:
            if eng_gib > float(disk_free_gib) - 8.0:
                raise RuntimeError(
                    f'Engram ({eng_gib:.1f} GiB) does not fit in {ssd_dir} disk ({disk_free_gib:.1f} GiB free)'
                )
            stage_mode = 'ssd+ssd'
            engram_path = f'{ssd_dir}/engram.bin'
            engram_head_bytes = 0
            engram_tail_bytes = int(engram_bytes)
            parallel_tail = False

    return {
        'stage_mode': stage_mode,
        'model_path': model_path,
        'engram_path': engram_path,
        'engram_tail_path': engram_tail_path,
        'engram_head_bytes': engram_head_bytes,
        'engram_tail_bytes': engram_tail_bytes,
        'parallel_tail': parallel_tail,
        'punch_file': punch_file,
        'ram_budget': ram_budget,
        'backbone_gib': round(bb_gib, 2),
        'engram_gib': round(eng_gib, 2),
        'ram_needed_gib': round(ram_needed_gib, 2),
    }


def build_jobs(url, file_bytes, sections, backbone_dest, engram_dest, engram_tail_dest='', engram_head_bytes=0):
    backbone_jobs = [{
        'url': url,
        'dest': backbone_dest,
        'dest_size': file_bytes,
        'src_off': 0,
        'dest_off': 0,
        'size': HEADER_BYTES,
    }]
    eng_sec = None
    for s in sections:
        if s['bytes'] == 0:
            continue
        if s['name'] == 'engram':
            assert eng_sec is None, 'multiple engram sections found'
            eng_sec = s
        else:
            backbone_jobs.append({
                'url': url,
                'dest': backbone_dest,
                'dest_size': file_bytes,
                'src_off': s['offset'],
                'dest_off': s['offset'],
                'size': s['bytes'],
            })
    assert eng_sec is not None, 'expected 1 engram section, found 0'

    eng_total = eng_sec['bytes']
    if engram_tail_dest and 0 < int(engram_head_bytes) < eng_total:
        head_b = int(engram_head_bytes)
        tail_b = eng_total - head_b
        engram_jobs = [{
            'url': url,
            'dest': engram_dest,
            'dest_size': head_b,
            'src_off': eng_sec['offset'],
            'dest_off': 0,
            'size': head_b,
        }]
        engram_tail_jobs = [{
            'url': url,
            'dest': engram_tail_dest,
            'dest_size': tail_b,
            'src_off': eng_sec['offset'] + head_b,
            'dest_off': 0,
            'size': tail_b,
        }]
        return backbone_jobs, engram_jobs, engram_tail_jobs

    engram_jobs = [{
        'url': url,
        'dest': engram_dest,
        'dest_size': eng_total,
        'src_off': eng_sec['offset'],
        'dest_off': 0,
        'size': eng_total,
    }]
    return backbone_jobs, engram_jobs, []


def _fallocate_punch(fd, offset, length):
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    libc.fallocate.restype = ctypes.c_int
    FALLOC_FL_KEEP_SIZE = 0x01
    FALLOC_FL_PUNCH_HOLE = 0x02
    rc = libc.fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, offset, length)
    return rc, ctypes.get_errno()


def test_punch_support(directory):
    """Probes whether `directory` actually frees disk blocks on FALLOC_FL_PUNCH_HOLE."""
    os.makedirs(directory, exist_ok=True)
    size = 4 * 1024 * 1024
    fd, path = tempfile.mkstemp(prefix='.basalt_punch_probe_', dir=directory)
    try:
        os.write(fd, b'\x5a' * size)
        os.fsync(fd)
        before_blocks = os.fstat(fd).st_blocks
        rc, err = _fallocate_punch(fd, 0, size)
        os.fsync(fd)
        after_blocks = os.fstat(fd).st_blocks
        ok = (rc == 0 and before_blocks > 0 and after_blocks < before_blocks // 2)
        return ok, before_blocks, after_blocks, rc, err
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


def punch_sections(pack_path, names):
    """Inward-4K-aligned FALLOC_FL_PUNCH_HOLE on named BASALTMF sections (e.g. dspark/gguf, dense)."""
    with open(pack_path, 'r+b') as f:
        _, sections = parse_header(f.read(HEADER_BYTES))
        fd = f.fileno()
        punched = 0
        for s in sections:
            if s['name'] not in names or s['bytes'] == 0:
                continue
            pa = (s['offset'] + 4095) & ~4095
            pb = (s['offset'] + s['bytes']) & ~4095
            if pb > pa:
                rc, _ = _fallocate_punch(fd, pa, pb - pa)
                if rc == 0:
                    punched += pb - pa
    return punched


def _write_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quant', default='ud-iq1_m')
    ap.add_argument('--url', default=None)
    ap.add_argument('--backbone-dest', default='/mnt/ram/model.basalt')
    ap.add_argument('--engram-dest', default='/mnt/ram/engram.bin')
    ap.add_argument('--engram-tail-dest', default='')
    ap.add_argument('--engram-head-bytes', type=int, default=0)
    ap.add_argument('--backbone-jobs', default=None)
    ap.add_argument('--engram-jobs', default=None)
    ap.add_argument('--engram-tail-jobs', default=None)
    ap.add_argument('--engram-size-out', default=None, help='Optional path to write exact Engram byte size')
    ap.add_argument('--engram-plan-out', default=None,
                    help='Emit multi-window Engram plan JSON + sparse SSD tail jobs for lazy post-/load finalize')
    ap.add_argument('--est-ram-avail-gib', type=float, default=0.0,
                    help='Estimated post-/load MemAvailable (GiB) used for initial parallel SSD range planning')
    ap.add_argument('--no-initial-ssd', action='store_true',
                    help='Defer all SSD range generation to --finalize-engram (used when PARALLEL_TAIL=0)')
    ap.add_argument('--finalize-engram', action='store_true',
                    help='Finalize RAM windows, layout file, and any extra SSD jobs from real post-/load MemAvailable')
    ap.add_argument('--plan-json', default=None)
    ap.add_argument('--mem-avail-gib', type=float, default=0.0)
    ap.add_argument('--ram-dir', default='/mnt/ram')
    ap.add_argument('--ssd-file', default='')
    ap.add_argument('--layout-out', default=None)
    ap.add_argument('--ram-jobs-out', default=None)
    ap.add_argument('--ssd-extra-jobs-out', default=None)
    ap.add_argument('--reserve-gib', type=float, default=5.0)
    ap.add_argument('--punch-sections', nargs='+', default=None,
                    help='Punch named sections (e.g. dspark/gguf dense) in --backbone-dest after /load')
    ap.add_argument('--test-punch', default=None, help='Probe FALLOC_FL_PUNCH_HOLE support in directory')
    ap.add_argument('--plan', action='store_true',
                    help='Read remote 64 KiB header and emit shell exports for optimal storage staging')
    ap.add_argument('--ssd-dir', default='/content')
    ap.add_argument('--ram-total-gib', type=float, default=176.0)
    ap.add_argument('--disk-free-gib', type=float, default=220.0)
    ap.add_argument('--vram-gib', type=float, default=96.0)
    ap.add_argument('--ram-budget', type=int, default=0)
    args = ap.parse_args()

    if args.test_punch:
        ok, before_b, after_b, rc, err = test_punch_support(args.test_punch)
        print(f'punch probe ({args.test_punch}): ok={ok} blocks={before_b}->{after_b} rc={rc} errno={err}')
        sys.exit(0 if ok else 1)

    if args.punch_sections:
        freed = punch_sections(args.backbone_dest, set(args.punch_sections))
        print(f'punched {args.punch_sections} in {args.backbone_dest}: freed {freed / GIB:.2f} GiB')
        return

    if args.finalize_engram:
        assert args.plan_json and args.layout_out and args.ram_jobs_out, \
            '--plan-json, --layout-out, and --ram-jobs-out are required with --finalize-engram'
        with open(args.plan_json) as f:
            plan_doc = json.load(f)
        res = finalize_engram_layout(
            plan_doc,
            mem_avail_gib=args.mem_avail_gib,
            ram_dir=args.ram_dir,
            ssd_file=args.ssd_file,
            reserve_gib=args.reserve_gib,
        )
        parent = os.path.dirname(os.path.abspath(args.layout_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.layout_out, 'w') as f:
            f.write(res['layout_text'])
        _write_json(args.ram_jobs_out, res['ram_jobs'])
        if args.ssd_extra_jobs_out:
            _write_json(args.ssd_extra_jobs_out, res['ssd_extra_jobs'])
        wp = res['plan']
        extra_gib = sum(b - a for a, b in res['ssd_extra_ranges']) / GIB
        print(
            f"engram finalize (avail={args.mem_avail_gib:.1f} GiB): "
            f"ram={wp['ram_bytes'] / GIB:.2f} GiB ({len(wp['ram_windows'])} win) | "
            f"ssd={wp['ssd_bytes'] / GIB:.2f} GiB (extra={extra_gib:.2f} GiB) | "
            f"cache_skip={len(wp['cache_skip_rows'])}"
        )
        return

    url = args.url or PACKS.get(args.quant.lower())
    if not url:
        sys.exit(f'unknown quant {args.quant!r} without --url')

    hf_hdr = os.environ.get('HF_HDR')
    file_bytes, sections = read_header(url, hf_hdr)

    if args.plan:
        bb_bytes, eng_bytes = summarize_sections(sections)
        try:
            plan = plan_storage(
                bb_bytes,
                eng_bytes,
                ram_total_gib=args.ram_total_gib,
                disk_free_gib=args.disk_free_gib,
                vram_gib=args.vram_gib,
                explicit_budget_gib=args.ram_budget,
                ssd_dir=args.ssd_dir,
            )
        except RuntimeError as e:
            sys.stderr.write(f'PLAN ERROR: {e}\n')
            sys.exit(2)
        for k, v in (
            ('STAGE_MODE', plan['stage_mode']),
            ('MODEL_PATH', plan['model_path']),
            ('ENGRAM_PATH', plan['engram_path']),
            ('ENGRAM_TAIL_PATH', plan['engram_tail_path']),
            ('ENGRAM_HEAD_BYTES', str(plan['engram_head_bytes'])),
            ('ENGRAM_TAIL_BYTES', str(plan['engram_tail_bytes'])),
            ('PARALLEL_TAIL', '1' if plan['parallel_tail'] else '0'),
            ('BASALT_PUNCH_FILE', plan['punch_file']),
            ('RAM_BUDGET', str(plan['ram_budget'])),
            ('BACKBONE_GIB', str(plan['backbone_gib'])),
            ('ENGRAM_GIB', str(plan['engram_gib'])),
            ('RAM_NEEDED_GIB', str(plan['ram_needed_gib'])),
            ('MODEL_URL', url),
        ):
            print(f'export {k}={shlex.quote(v)}')
        return

    assert args.backbone_jobs and args.engram_jobs, '--backbone-jobs and --engram-jobs are required'
    bb_jobs, eng_jobs, tail_jobs = build_jobs(
        url,
        file_bytes,
        sections,
        args.backbone_dest,
        args.engram_dest,
        engram_tail_dest=args.engram_tail_dest,
        engram_head_bytes=args.engram_head_bytes,
    )

    eng_sec = next(s for s in sections if s['name'] == 'engram' and s['bytes'] > 0)
    eng_total = eng_sec['bytes']

    if args.engram_plan_out and args.engram_tail_dest:
        meta = read_meta(url, sections, hf_hdr)
        engram_info = extract_engram_subtables(meta, eng_total)
        if args.no_initial_ssd:
            initial_ssd = []
            tail_jobs = []
        else:
            est_avail = args.est_ram_avail_gib
            if est_avail <= 0.0:
                est_avail = max(0.0, args.engram_head_bytes / GIB + args.reserve_gib)
            init_plan = plan_engram_windows(eng_total, est_avail, engram_info=engram_info, reserve_gib=args.reserve_gib)
            initial_ssd = init_plan['ssd_ranges']
            _, tail_jobs, _ = build_engram_window_jobs(
                url, eng_sec['offset'], eng_total, [], initial_ssd, ssd_file=args.engram_tail_dest,
            )
        plan_doc = {
            'url': url,
            'eng_offset': eng_sec['offset'],
            'eng_bytes': eng_total,
            'engram_info': engram_info,
            'initial_ssd_ranges': [list(r) for r in initial_ssd],
        }
        _write_json(args.engram_plan_out, plan_doc)
        eng_jobs = []  # populated by --finalize-engram after POST /load

    _write_json(args.backbone_jobs, bb_jobs)
    _write_json(args.engram_jobs, eng_jobs)
    if args.engram_tail_jobs and tail_jobs:
        _write_json(args.engram_tail_jobs, tail_jobs)
    if args.engram_size_out:
        with open(args.engram_size_out, 'w') as f:
            f.write(str(eng_total if args.engram_plan_out else eng_jobs[0]['size']) + '\n')

    bb_gb = sum(j['size'] for j in bb_jobs) / 1e9
    eng_gb = sum(j['size'] for j in eng_jobs) / 1e9
    tail_gb = sum(j['size'] for j in tail_jobs) / 1e9
    tail_desc = f' + tail={tail_gb:.2f} GB -> {args.engram_tail_dest}' if tail_jobs else ''
    print(f'basalt_jobs ({args.quant}): logical={file_bytes/1e9:.2f} GB | '
          f'backbone={bb_gb:.2f} GB ({len(bb_jobs)} ranges -> {args.backbone_dest}) | '
          f'engram={eng_gb:.2f} GB (-> {args.engram_dest}){tail_desc}')


if __name__ == '__main__':
    main()
