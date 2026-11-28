#!/usr/bin/env python3
"""Surgical Colab G4 patches for jesdga95/basalt v0.6.0 (BASALT_COLAB_PATCH).

Note: Basalt v0.6.0 upstreamed O_DIRECT->EINVAL tmpfs fallback (#835) and alloc_ram
MADV_HUGEPAGE (#834). Remaining patches enable zero-SSD-expert ~95s boot, Stage 1
Micro-Wave streaming (wave+hybrid), and multi-window lazy Engram on Colab G4:
  1. src/models/deepseek41/model.cpp (BASALT_COLAB_ENGRAM_ENV):
     BASALT_ENGRAM_FILE separate path at offset 0, BASALT_ENGRAM_CACHE_MIB (0..4096 MiB),
     and BASALT_ENGRAM_INFLIGHT (1..1024, default 256).
  2. src/platform/direct_file.cpp (BASALT_COLAB_ENGRAM_SPLIT):
     Legacy split head+tail file support when BASALT_ENGRAM_TAIL_FILE is set.
  3a. src/core/expert_tier.cpp (BASALT_COLAB_PUNCH_ON_READ):
     Credits punchable tmpfs backbone blocks (st_blocks*512 minus 2 GiB) into avail.
  3b. src/core/loader.cpp (BASALT_COLAB_PUNCH_ON_READ + BASALT_COLAB_WAVE_FETCH):
     Two-wave barriered execution + inward-4K FALLOC_FL_PUNCH_HOLE on BASALT_PUNCH_FILE.
     When BASALT_WAVE_FETCH_CMD + BASALT_EXPERTS_RANGE are set (Stage 1 wave+hybrid),
     clips spans strictly to [experts_lo, experts_hi) (never touching abliterated dense),
     fetches Wave 1 VRAM experts on demand, then sorts Wave 2 RAM experts by ascending
     file offset and streams in ~20 GiB micro-waves (fetch -> load -> punch, peak <=141 GiB).
  4. serve/service.py (BASALT_COLAB_PUNCH_GUARD):
     Refuses unload and engine respawn once BASALT_PUNCH_FILE has punched tmpfs weights.
  5. src/ngram/ple_reader.cpp (BASALT_COLAB_ENGRAM_MMAP):
     Zero-copy mmap(), BASALT_ENGRAM_LAZY_OPEN=1, multi-window BASALT_ENGRAM_LAYOUT
     (ram + logical full-size sparse ssd), and unified RowCache::insert() cache_skip filter.
  6. src/core/ram_load.cpp (BASALT_COLAB_RAM_PARALLEL_PIN):
     4-thread parallel cudaHostRegister across layer spans when unlimited.

Usage: python3 patch_basalt.py BASALT_SRC_DIR
"""
import os, sys


def patch_file(src_dir, rel, marker, edits):
    path = os.path.join(src_dir, rel)
    if not os.path.isfile(path):
        sys.exit(f'[PATCH FAIL] {marker}: target file missing in Basalt tree: {rel}')
    text = open(path, 'r', encoding='utf-8').read()
    if marker in text:
        print(f'already patched {rel} ({marker})')
        return
    for idx, (old, new) in enumerate(edits, 1):
        count = text.count(old)
        if count != 1:
            first_line = old.strip().splitlines()[0][:120] if old.strip() else ''
            sys.exit(
                f'[PATCH FAIL] {marker} ({rel} edit #{idx}): expected 1 anchor match, got {count} '
                f'(first line: {first_line!r}). Upstream Basalt may have moved this code; '
                f'omit --basalt-ref to use the pinned manifest or update patch_basalt.py.'
            )
        text = text.replace(old, new, 1)
    open(path, 'w', encoding='utf-8').write(text)
    print(f'patched {rel} ({marker})')


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python3 patch_basalt.py BASALT_SRC_DIR')
    src = sys.argv[1]

    # 1. Separate Engram file + configurable RowCache MiB (0..4096) & inflight depth (1..1024)
    patch_file(
        src,
        'src/models/deepseek41/model.cpp',
        'BASALT_COLAB_ENGRAM_ENV',
        [
            (
                '#include <cstdio>',
                '#include <cstdio>\n#include <cstdlib>',
            ),
            (
                '''    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        // Map now so tier_spec() reserves its RAM; lock/touch in attach() after the expert tier's reads finish so
        // buffered expert reads cannot evict unlocked Engram pages from the OS page cache.
        if (ple_ram) {
            if (!engram_->open_ram(path, e.file_offset, e.rows, err, e.row_bytes, /*lock=*/false)) return false;
        } else if (!engram_->open(path, e.file_offset, e.rows, 256, 1 << 20, err, e.row_bytes)) {
            return false;
        }
    }''',
                '''    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        // BASALT_COLAB_ENGRAM_ENV + BASALT_COLAB_ENGRAM_TAIL_CACHE:
        // separate Engram path at offset 0, RowCache size in MiB (0..4096), and inflight depth (1..1024).
        const char* engram_env = std::getenv("BASALT_ENGRAM_FILE");
        const std::string engram_path = (engram_env && *engram_env) ? std::string(engram_env) : path;
        const uint64_t engram_off = (engram_env && *engram_env) ? 0ull : e.file_offset;
        if (ple_ram) {
            if (!engram_->open_ram(engram_path, engram_off, e.rows, err, e.row_bytes, /*lock=*/false)) return false;
        } else {
            uint64_t cache_rows = 1ull << 20;
            if (const char* cm = std::getenv("BASALT_ENGRAM_CACHE_MIB"); cm && *cm) {
                char* endp = nullptr;
                const long long mib = std::strtoll(cm, &endp, 10);
                if (endp != cm && mib >= 0) {
                    if (mib == 0) {
                        cache_rows = 0;
                    } else {
                        const uint64_t clamped_mib = std::min<uint64_t>((uint64_t) mib, 4096ull);
                        const uint64_t rb = std::max<uint32_t>(e.row_bytes, 1u);
                        cache_rows = std::min<uint64_t>(e.rows, (clamped_mib << 20) / rb) & ~7ull;
                    }
                }
            }
            uint32_t inflight = 256u;
            if (const char* inf = std::getenv("BASALT_ENGRAM_INFLIGHT"); inf && *inf) {
                char* endp = nullptr;
                const long long v = std::strtoll(inf, &endp, 10);
                if (endp != inf && v >= 1) inflight = (uint32_t) std::min<long long>(v, 1024ll);
            }
            if (!engram_->open(engram_path, engram_off, e.rows, inflight, cache_rows, err, e.row_bytes)) return false;
        }
    }''',
            ),
        ],
    )

    # 2. DirectFile legacy split Engram tail support (BASALT_COLAB_ENGRAM_SPLIT)
    #    (O_DIRECT -> EINVAL tmpfs fallback is native in Basalt v0.6.0 via #835)
    patch_file(
        src,
        'src/platform/direct_file.cpp',
        'BASALT_COLAB_ENGRAM_SPLIT',
        [
            (
                '''struct DirectFile::Impl {
    int fd = -1;
    bool no_buffering = false;
    uint64_t size = 0;''',
                '''struct DirectFile::Impl {
    int fd = -1;
    int fd2 = -1; // BASALT_COLAB_ENGRAM_SPLIT: optional legacy tail file for split Engram
    uint64_t split_bytes = 0;
    bool no_buffering = false;
    uint64_t size = 0;''',
            ),
            (
                '''                const ssize_t got = pread(fd, dst + total, p.length - total, (off_t) (p.offset + total));''',
                '''                const uint64_t cur_off = p.offset + total;
                const bool in_tail = (fd2 >= 0 && cur_off >= split_bytes);
                const ssize_t got = pread(in_tail ? fd2 : fd, dst + total, p.length - total,
                                          (off_t) (in_tail ? cur_off - split_bytes : cur_off));''',
            ),
            (
                '''    if (impl_->fd < 0) {
        impl_->no_buffering = false;
        err = "DirectFile: cannot open " + path;
        return false;
    }
    return finish_open(path, err, workers);''',
                '''    if (impl_->fd < 0) {
        impl_->no_buffering = false;
        err = "DirectFile: cannot open " + path;
        return false;
    }
    const char* eng_head = std::getenv("BASALT_ENGRAM_FILE");
    const char* eng_tail = std::getenv("BASALT_ENGRAM_TAIL_FILE");
    if (eng_head && *eng_head && eng_tail && *eng_tail && path == eng_head) {
        impl_->fd2 = ::open(eng_tail, O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (impl_->fd2 < 0 && errno == EINVAL) {
            impl_->fd2 = ::open(eng_tail, O_RDONLY | O_CLOEXEC);
        }
        if (impl_->fd2 < 0) {
            err = "DirectFile: cannot open split tail " + std::string(eng_tail);
            close();
            return false;
        }
    }
    return finish_open(path, err, workers);''',
            ),
            (
                '''    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;''',
                '''    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
    if (impl_->fd2 >= 0) {
        struct stat st2{};
        if (fstat(impl_->fd2, &st2) != 0) { err = "DirectFile: cannot size split tail"; close(); return false; }
        if (impl_->size % alignment() != 0) { err = "DirectFile: split head size must be page-aligned"; close(); return false; }
        impl_->split_bytes = impl_->size;
        impl_->size += (uint64_t) st2.st_size;
    }''',
            ),
            (
                '''    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->no_buffering = false;
    impl_->size = 0;''',
                '''    if (impl_->fd >= 0) ::close(impl_->fd);
    if (impl_->fd2 >= 0) ::close(impl_->fd2);
    impl_->fd = -1;
    impl_->fd2 = -1;
    impl_->split_bytes = 0;
    impl_->no_buffering = false;
    impl_->size = 0;''',
            ),
        ],
    )

    # 3a. ExpertTier: credit punchable tmpfs backbone bytes into avail when BASALT_PUNCH_FILE is active
    patch_file(
        src,
        'src/core/expert_tier.cpp',
        'BASALT_COLAB_PUNCH_ON_READ',
        [
            (
                '#include <cstring>',
                '#include <cstring>\n#ifdef __linux__\n#include <sys/stat.h>\n#endif',
            ),
            (
                '''    uint64_t avail = available();
    // places per layer above the reserve, less the spare each layer keeps, in `bytes` of RAM''',
                '''    uint64_t avail = available();
    // BASALT_COLAB_PUNCH_ON_READ: credit tmpfs backbone blocks that Loader::run will punch before RAM reads
#ifdef __linux__
    uint64_t punch_credit = 0;
    if (const char* pf = std::getenv("BASALT_PUNCH_FILE"); pf && *pf) {
        struct stat st{};
        if (::stat(pf, &st) == 0 && st.st_blocks > 0) {
            const uint64_t phys = (uint64_t) st.st_blocks * 512ull;
            constexpr uint64_t kKeepBytes = 2ull << 30;
            if (phys > kKeepBytes) punch_credit = phys - kKeepBytes;
        }
    }
    const uint64_t phys_cap = std::min(platform::total_physical_memory(), hm.cgroup_limit);
    if (punch_credit > 0) {
        avail = std::min(avail + punch_credit, phys_cap > host_bytes_ ? phys_cap - host_bytes_ : avail);
    }
#endif
    // places per layer above the reserve, less the spare each layer keeps, in `bytes` of RAM''',
            ),
        ],
    )

    # 3b. Loader: two-wave barriered execution + inward-4K fallocate(PUNCH_HOLE) + Stage 1 Micro-Wave
    #     (BASALT_COLAB_PUNCH_ON_READ + BASALT_COLAB_WAVE_FETCH)
    patch_file(
        src,
        'src/core/loader.cpp',
        'BASALT_COLAB_PUNCH_ON_READ',
        [
            (
                '#include <cstring>',
                '#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <fstream>\n#ifdef __linux__\n#include <fcntl.h>\n#include <sys/stat.h>\n#include <sys/wait.h>\n#include <unistd.h>\n#endif',
            ),
            (
                '''    const size_t device_tasks = rounds[0].size();
    rounds[0].insert(rounds[0].end(), host_reads.begin(), host_reads.end());
    const int n_workers = (int) std::min<size_t>((size_t) m.threads, std::max(rounds[0].size(), rounds[1].size()));''',
                '''    // BASALT_COLAB_PUNCH_ON_READ + BASALT_COLAB_WAVE_FETCH:
    // When punching a tmpfs model file on read, run Wave 1 (VRAM) -> punch, then Wave 2 (RAM) -> punch.
    // When BASALT_WAVE_FETCH_CMD + BASALT_EXPERTS_RANGE are also set (Stage 1 wave+hybrid), stream ONLY
    // [experts_lo, experts_hi) on demand (never touching abliterated dense), sorting Wave 2 by ascending
    // file offset and executing in ~20 GiB micro-waves (peak RAM <= 141 GiB, 0 bytes backbone on SSD).
    int punch_fd = -1;
    std::vector<bool> file_is_punch(handles.size(), false);
    std::string wave_fetch_cmd;
    uint64_t experts_lo = 0, experts_hi = 0;
    uint64_t wave_chunk_bytes = 20ull << 30;
#ifdef __linux__
    if (const char* pf = std::getenv("BASALT_PUNCH_FILE"); pf && *pf) {
        struct stat target_st{};
        if (::stat(pf, &target_st) == 0) {
            bool any_match = false;
            for (size_t i = 0; i < handles.size(); ++i) {
                struct stat f_st{};
                const int fd_chk = handles[i]->buffered >= 0 ? handles[i]->buffered : handles[i]->direct;
                if (fd_chk >= 0 && ::fstat(fd_chk, &f_st) == 0 &&
                    f_st.st_dev == target_st.st_dev && f_st.st_ino == target_st.st_ino) {
                    file_is_punch[i] = true;
                    any_match = true;
                }
            }
            if (any_match) {
                punch_fd = ::open(pf, O_RDWR | O_CLOEXEC);
                if (punch_fd >= 0) early_ram = 0;
            }
        }
    }
    if (punch_fd >= 0) {
        const char* wf = std::getenv("BASALT_WAVE_FETCH_CMD");
        const char* er = std::getenv("BASALT_EXPERTS_RANGE");
        if (wf && *wf && er && *er) {
            unsigned long long elo = 0, ehi = 0;
            if (std::sscanf(er, "%llu:%llu", &elo, &ehi) == 2 && ehi > elo) {
                wave_fetch_cmd = wf;
                experts_lo = (uint64_t) elo;
                experts_hi = (uint64_t) ehi;
            }
        }
        if (const char* cg = std::getenv("BASALT_WAVE_CHUNK_GIB"); cg && *cg) {
            char* endp = nullptr;
            const long long v = std::strtoll(cg, &endp, 10);
            if (endp != cg && v >= 4) {
                wave_chunk_bytes = (uint64_t) std::min<long long>(v, 64ll) << 30;
            }
        }
    }
#endif
    const size_t device_tasks = rounds[0].size();
    std::vector<Task> punch_host_wave;
    if (punch_fd >= 0 && device_tasks > 0 && !host_reads.empty()) {
        punch_host_wave = std::move(host_reads);
    } else {
        rounds[0].insert(rounds[0].end(), host_reads.begin(), host_reads.end());
    }
    const size_t max_wave_tasks = std::max(std::max(rounds[0].size(), rounds[1].size()), punch_host_wave.size());
    const int n_workers = (int) std::min<size_t>((size_t) m.threads, max_wave_tasks);''',
            ),
            (
                '        if (device_tasks < rounds[0].size()) stages.push_back("ram");',
                '        if (device_tasks < rounds[0].size() || !punch_host_wave.empty()) stages.push_back("ram");',
            ),
            (
                '''    for (const std::vector<Task>& tasks : rounds) {   // each round's copies are complete before the next round reads them
        if (&tasks == &rounds[1] && !failed.load())   // the reads have landed: the plan's steps, before the copies
            for (const auto& step : plan.steps()) {
                std::string e;
                if (!step(e)) { fail(e); break; }
            }
        if (tasks.empty() || failed.load()) continue;
        std::atomic<size_t> next{0};
        const int n = (int) std::min<size_t>((size_t) n_workers, tasks.size());
        std::vector<std::thread> pool;
        for (int i = 1; i < n; ++i) pool.emplace_back(work, i, std::cref(tasks), std::ref(next));
        work(0, tasks, next);
        for (auto& t : pool) t.join();
    }
    if (any_dev) (void) cudaSetDevice(prev_device);''',
                '''    auto run_wave = [&](const std::vector<Task>& wave) {
        if (wave.empty() || failed.load()) return;
        std::atomic<size_t> next{0};
        const int n = (int) std::min<size_t>((size_t) n_workers, wave.size());
        std::vector<std::thread> pool;
        for (int i = 1; i < n; ++i) pool.emplace_back(work, i, std::cref(wave), std::ref(next));
        work(0, wave, next);
        for (auto& t : pool) t.join();
    };
#ifdef __linux__
    using Span = std::pair<uint64_t, uint64_t>;
    auto merge_spans = [](std::vector<Span>& sp) {
        if (sp.empty()) return;
        std::sort(sp.begin(), sp.end());
        size_t w = 0;
        for (size_t i = 1; i < sp.size(); ++i) {
            if (sp[i].first <= sp[w].second) sp[w].second = std::max(sp[w].second, sp[i].second);
            else sp[++w] = sp[i];
        }
        sp.resize(w + 1);
    };
    auto collect_spans = [&](const std::vector<Task>& wave, size_t from = 0, size_t to = SIZE_MAX) {
        std::vector<Span> sp;
        const size_t end = std::min(to, wave.size());
        for (size_t i = from; i < end; ++i) {
            const Task& t = wave[i];
            const LoadItem& it = items[t.item];
            if (it.from == nullptr && it.file >= 0 && file_is_punch[(size_t) it.file] && t.len > 0)
                sp.emplace_back(it.off + t.off, it.off + t.off + t.len);
        }
        merge_spans(sp);
        return sp;
    };
    auto clip_spans = [&](const std::vector<Span>& sp, uint64_t lo, uint64_t hi) {
        std::vector<Span> out;
        for (auto [a, b] : sp) {
            const uint64_t ca = std::max(a, lo);
            const uint64_t cb = std::min(b, hi);
            if (cb > ca) out.emplace_back(ca, cb);
        }
        merge_spans(out);
        return out;
    };
    auto invoke_wave_fetch = [&](const std::vector<Span>& sp, const std::string& tag) {
        if (wave_fetch_cmd.empty() || sp.empty() || failed.load()) return;
        const std::string jpath = "/tmp/basalt_wave_" + std::to_string((long long) ::getpid()) + "_" + tag + ".json";
        {
            std::ofstream out(jpath);
            if (!out.is_open()) {
                fail("BASALT_WAVE_FETCH_CMD: cannot write spans JSON " + jpath);
                return;
            }
            out << "[";
            for (size_t i = 0; i < sp.size(); ++i) {
                if (i > 0) out << ",";
                out << "[" << sp[i].first << "," << sp[i].second << "]";
            }
            out << "]\\n";
        }
        uint64_t total_b = 0;
        for (auto [a, b] : sp) total_b += (b - a);
        basalt::log::info("basalt loader", "wave fetch (%s): %.2f GiB across %zu span(s)",
                          tag.c_str(), (double) total_b / 1073741824.0, sp.size());
        const pid_t pid = ::fork();
        if (pid < 0) {
            (void) ::unlink(jpath.c_str());
            fail("BASALT_WAVE_FETCH_CMD: fork failed");
            return;
        }
        if (pid == 0) {
            if (::access(wave_fetch_cmd.c_str(), X_OK) == 0) {
                char* argv[] = {
                    const_cast<char*>(wave_fetch_cmd.c_str()),
                    const_cast<char*>(jpath.c_str()),
                    const_cast<char*>(tag.c_str()),
                    nullptr,
                };
                ::execv(wave_fetch_cmd.c_str(), argv);
            } else {
                char bash_bin[] = "/bin/bash";
                char* argv[] = {
                    bash_bin,
                    const_cast<char*>(wave_fetch_cmd.c_str()),
                    const_cast<char*>(jpath.c_str()),
                    const_cast<char*>(tag.c_str()),
                    nullptr,
                };
                ::execv(bash_bin, argv);
            }
            ::_exit(127);
        }
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0) {
            if (errno != EINTR) { status = 1; break; }
        }
        (void) ::unlink(jpath.c_str());
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fail("BASALT_WAVE_FETCH_CMD failed (" + tag + ", rc=" +
                 std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + ")");
        }
    };
    auto punch_spans = [&](const std::vector<Span>& sp, const std::vector<Span>& keep, const char* tag) {
        if (punch_fd < 0 || sp.empty()) return;
        uint64_t punched = 0;
        size_t ki = 0;
        auto do_punch = [&](uint64_t a, uint64_t b) {
            const uint64_t pa = (a + 4095ull) & ~4095ull;   // inward 4K alignment: never touches adjacent tensors
            const uint64_t pb = b & ~4095ull;
            if (pb > pa && ::fallocate(punch_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                                       (off_t) pa, (off_t) (pb - pa)) == 0)
                punched += (pb - pa);
        };
        for (auto [a, b] : sp) {
            while (ki < keep.size() && keep[ki].second <= a) ++ki;
            size_t j = ki;
            uint64_t cur = a;
            while (j < keep.size() && keep[j].first < b) {
                if (keep[j].first > cur) do_punch(cur, keep[j].first);
                cur = std::max(cur, keep[j].second);
                if (cur >= b) break;
                ++j;
            }
            if (cur < b) do_punch(cur, b);
        }
        if (punched > 0)
            basalt::log::info("basalt loader", "punch-on-read (%s): freed %.2f GiB tmpfs",
                              tag, (double) punched / 1073741824.0);
    };
#endif

#ifdef __linux__
    if (punch_fd >= 0 && !wave_fetch_cmd.empty() && !failed.load()) {
        const auto dev_fetch = clip_spans(collect_spans(rounds[0]), experts_lo, experts_hi);
        invoke_wave_fetch(dev_fetch, "wave1");
    }
#endif
    run_wave(rounds[0]);
#ifdef __linux__
    if (punch_fd >= 0 && !failed.load()) {
        if (!punch_host_wave.empty()) {
            const auto dev_sp  = collect_spans(rounds[0]);
            const auto host_sp = collect_spans(punch_host_wave);
            punch_spans(dev_sp, host_sp, "after-vram");
            LoadProgress::into("ram");
            basalt::log::info("basalt loader", "the cards' share is read (%.2f GiB in %.1f s); RAM next",
                              (double) device_bytes / 1073741824.0,
                              std::chrono::duration<double>(std::chrono::steady_clock::now() - run_t0).count());
            if (!wave_fetch_cmd.empty()) {
                // Sort RAM expert reads by ascending file byte offset so micro-waves coalesce HTTP ranges
                // and keep punched byte intervals strictly monotonic (peak RAM <= ~141 GiB).
                std::stable_sort(punch_host_wave.begin(), punch_host_wave.end(), [&](const Task& a, const Task& b) {
                    const uint64_t oa = items[a.item].off + a.off;
                    const uint64_t ob = items[b.item].off + b.off;
                    return oa < ob;
                });
                size_t pos = 0;
                size_t wave_idx = 0;
                while (pos < punch_host_wave.size() && !failed.load()) {
                    size_t end = pos;
                    uint64_t acc = 0;
                    while (end < punch_host_wave.size()) {
                        const uint64_t len = punch_host_wave[end].len;
                        if (end > pos && acc + len > wave_chunk_bytes) break;
                        acc += len;
                        ++end;
                    }
                    std::vector<Task> sub_wave(punch_host_wave.begin() + (ptrdiff_t) pos,
                                               punch_host_wave.begin() + (ptrdiff_t) end);
                    const auto sub_sp = collect_spans(sub_wave);
                    const auto sub_fetch = clip_spans(sub_sp, experts_lo, experts_hi);
                    invoke_wave_fetch(sub_fetch, "wave2_" + std::to_string(wave_idx));
                    run_wave(sub_wave);
                    if (!failed.load()) {
                        const auto future_sp = collect_spans(punch_host_wave, end);
                        punch_spans(sub_sp, future_sp, "after-ram-wave");
                    }
                    pos = end;
                    ++wave_idx;
                }
            } else {
                run_wave(punch_host_wave);
                if (!failed.load()) punch_spans(host_sp, {}, "after-ram");
            }
        } else {
            const auto all_sp = collect_spans(rounds[0]);
            punch_spans(all_sp, {}, "after-reads");
        }
    }
    if (punch_fd >= 0) ::close(punch_fd);
#endif

    if (!failed.load())   // the reads have landed: the plan's steps, before the copies
        for (const auto& step : plan.steps()) {
            std::string e;
            if (!step(e)) { fail(e); break; }
        }
    run_wave(rounds[1]);
    if (any_dev) (void) cudaSetDevice(prev_device);''',
            ),
        ],
    )

    # 4. Serve guard: refuse unload and refuse engine respawn after BASALT_PUNCH_FILE punched the tmpfs pack
    patch_file(
        src,
        'serve/service.py',
        'BASALT_COLAB_PUNCH_GUARD',
        [
            (
                'import subprocess\n',
                'import os\nimport subprocess\n',
            ),
            (
                '''        try:
            self._ensure_loaded()
            if self.on_loaded is not None:
                self.on_loaded()''',
                '''        try:
            self._ensure_loaded()
            if os.environ.get("BASALT_PUNCH_FILE"):
                self._punched_once = True
            if self.on_loaded is not None:
                self.on_loaded()''',
            ),
            (
                '''        if self.loaded():
            if self._vision_down():
                print("[server] restarting the vision encoder", flush=True)
                self.vision.restart()
            return''',
                '''        if self.loaded():
            if self._vision_down():
                print("[server] restarting the vision encoder", flush=True)
                self.vision.restart()
            return
        # BASALT_COLAB_PUNCH_GUARD: punched tmpfs weights cannot be re-read after an engine exit
        if os.environ.get("BASALT_PUNCH_FILE") and getattr(self, "_punched_once", False):
            raise EngineDied("model file was punched on read; restart the VM to reload")''',
            ),
            (
                '''        if not hasattr(self.engine, "unload"):
            return "unsupported"''',
                '''        if not hasattr(self.engine, "unload") or os.environ.get("BASALT_PUNCH_FILE"):
            return "unsupported"''',
            ),
        ],
    )

    # 5. PleReader RAM zero-copy mmap(), multi-window RAM+sparse-SSD layout, lazy open, and RowCache skip filter
    patch_file(
        src,
        'src/ngram/ple_reader.cpp',
        'BASALT_COLAB_ENGRAM_MMAP',
        [
            (
                '#include <cstring>',
                '#include <cstdlib>\n#include <cstring>\n#include <fstream>\n#include <sstream>\n#ifdef __linux__\n#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n#endif',
            ),
            (
                '''    std::vector<uint8_t> next;      // per-set replacement pointer
    uint64_t used = 0;''',
                '''    std::vector<uint8_t> next;      // per-set replacement pointer
    uint64_t used = 0;
    std::vector<std::pair<uint32_t, uint32_t>> skip_ranges; // BASALT_COLAB_ENGRAM_MMAP: skip cold 4-gram admission''',
            ),
            (
                '''    void insert(uint32_t row, const uint8_t* bytes) {
        if (sets == 0 || find(row) != nullptr) return;''',
                '''    void insert(uint32_t row, const uint8_t* bytes) {
        if (sets == 0) return;
        for (const auto& r : skip_ranges) {
            if (row >= r.first && row < r.second) return;
        }
        if (find(row) != nullptr) return;''',
            ),
            (
                '''struct PleReader::Impl {
    DirectFile file;''',
                '''struct PleReader::Impl {
    struct RamWindow {
        const uint8_t* ptr = nullptr;
        uint64_t lo = 0;
        uint64_t hi = 0;
        uint64_t len = 0;
    };
    DirectFile file;
    const uint8_t* mmap_head = nullptr;   // BASALT_COLAB_ENGRAM_MMAP: legacy prefix tmpfs Engram head
    uint64_t mmap_head_bytes = 0;
    std::vector<RamWindow> ram_windows;   // multi-window RAM mmap regions [lo, hi)
    bool lazy_pending = false;            // BASALT_ENGRAM_LAZY_OPEN: defer file/mmap until first issue()
    std::string open_path;
    std::string layout_path;

    void unmap_windows() {
#ifdef __linux__
        if (mmap_head != nullptr) {
            (void) ::munmap((void*) mmap_head, (size_t) mmap_head_bytes);
            mmap_head = nullptr;
            mmap_head_bytes = 0;
        }
        for (const auto& w : ram_windows) {
            if (w.ptr != nullptr && w.len > 0) (void) ::munmap((void*) w.ptr, (size_t) w.len);
        }
#endif
        ram_windows.clear();
    }

    bool ensure_opened(std::string& err) {
        if (!lazy_pending) return true;
        unmap_windows();
        cache.skip_ranges.clear();
        std::string ssd_path;
        struct PendingRam { std::string path; uint64_t lo; uint64_t hi; };
        std::vector<PendingRam> pending_ram;
        if (!layout_path.empty()) {
            std::ifstream in(layout_path);
            if (!in.is_open()) {
                err = "PleReader: cannot open BASALT_ENGRAM_LAYOUT " + layout_path;
                return false;
            }
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                std::string kind;
                if (!(iss >> kind)) continue;
                if (kind == "ssd") {
                    iss >> ssd_path;
                } else if (kind == "ram") {
                    PendingRam pr{};
                    if (iss >> pr.path >> pr.lo >> pr.hi && pr.hi > pr.lo) pending_ram.push_back(std::move(pr));
                } else if (kind == "cache_skip") {
                    uint32_t rlo = 0, rhi = 0;
                    if (iss >> rlo >> rhi && rhi > rlo) cache.skip_ranges.emplace_back(rlo, rhi);
                }
            }
        }
        const std::string backing = !layout_path.empty() ? ssd_path : open_path;
        if (!backing.empty()) {
            if (!file.open(backing, err, 0, /*mapped_elsewhere=*/true)) return false;
            if (table_offset + n_rows * (uint64_t) row_bytes > file.size()) {
                err = "PleReader: the table extends past the end of " + backing;
                file.close();
                return false;
            }
        }
#ifdef __linux__
        for (const auto& pr : pending_ram) {
            const uint64_t want = pr.hi - pr.lo;
            const int mfd = ::open(pr.path.c_str(), O_RDONLY | O_CLOEXEC);
            struct stat mst{};
            if (mfd < 0 || ::fstat(mfd, &mst) != 0 || (uint64_t) mst.st_size < want) {
                if (mfd >= 0) ::close(mfd);
                unmap_windows();
                file.close();
                err = "PleReader: cannot map RAM window " + pr.path;
                return false;
            }
            void* mp = ::mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, mfd, 0);
            ::close(mfd);
            if (mp == MAP_FAILED) {
                unmap_windows();
                file.close();
                err = "PleReader: mmap failed for " + pr.path;
                return false;
            }
            (void) ::madvise(mp, (size_t) want, MADV_RANDOM);
            ram_windows.push_back(RamWindow{(const uint8_t*) mp, pr.lo, pr.hi, want});
        }
#endif
        if (!file.is_open() && ram_windows.empty()) {
            err = "PleReader: lazy layout configured neither ssd nor ram windows";
            return false;
        }
        lazy_pending = false;
        return true;
    }''',
            ),
            (
                '''    double keepalive_due(double now) const {
        if (keep_us <= 0 || !error.empty() || last_issue_us <= 0 || now - last_issue_us > keep_window_us) return -1;''',
                '''    double keepalive_due(double now) const {
        if (lazy_pending || !file.is_open() || keep_us <= 0 || !error.empty() || last_issue_us <= 0 || now - last_issue_us > keep_window_us) return -1;''',
            ),
            (
                '''    if (!impl_->file.open(path, err, 0, /*mapped_elsewhere=*/true)) return false;
    impl_->row_bytes = row_bytes;
    if (table_offset + n_rows * (uint64_t) row_bytes > impl_->file.size()) {
        err = "PleReader: the table extends past the end of " + path;
        close();
        return false;
    }''',
                '''    impl_->row_bytes = row_bytes;
    impl_->open_path = path;
    impl_->layout_path.clear();
    impl_->lazy_pending = false;
    const char* lazy_env = std::getenv("BASALT_ENGRAM_LAZY_OPEN");
    const char* layout_env = std::getenv("BASALT_ENGRAM_LAYOUT");
    const bool want_lazy = (lazy_env && std::strcmp(lazy_env, "1") == 0);
    if (want_lazy) {
        if (layout_env && *layout_env) impl_->layout_path = layout_env;
        impl_->lazy_pending = true;
    } else {
        if (!impl_->file.open(path, err, 0, /*mapped_elsewhere=*/true)) return false;
        if (table_offset + n_rows * (uint64_t) row_bytes > impl_->file.size()) {
            err = "PleReader: the table extends past the end of " + path;
            close();
            return false;
        }
#ifdef __linux__
        {
            const char* mm_env = std::getenv("BASALT_ENGRAM_MMAP");
            const bool want_mm = (table_offset == 0) &&
                ((mm_env && std::strcmp(mm_env, "1") == 0) ||
                 ((!mm_env || std::strcmp(mm_env, "0") != 0) && path.rfind("/mnt/ram/", 0) == 0));
            if (want_mm) {
                const int mfd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                struct stat mst{};
                if (mfd >= 0 && ::fstat(mfd, &mst) == 0 && mst.st_size > 0) {
                    void* mp = ::mmap(nullptr, (size_t) mst.st_size, PROT_READ, MAP_SHARED, mfd, 0);
                    if (mp != MAP_FAILED) {
                        impl_->mmap_head = (const uint8_t*) mp;
                        impl_->mmap_head_bytes = (uint64_t) mst.st_size;
                        (void) ::madvise(mp, (size_t) mst.st_size, MADV_RANDOM);
                    }
                }
                if (mfd >= 0) ::close(mfd);
            }
        }
#endif
    }''',
            ),
            (
                '''    m.file.close();
    m.mapped.reset();''',
                '''    m.unmap_windows();
    m.lazy_pending = false;
    m.cache.skip_ranges.clear();
    m.file.close();
    m.mapped.reset();''',
            ),
            (
                '''bool PleReader::is_open() const { return impl_->ram_mode || impl_->file.is_open(); }''',
                '''bool PleReader::is_open() const { return impl_->ram_mode || impl_->file.is_open() || impl_->lazy_pending || !impl_->ram_windows.empty(); }''',
            ),
            (
                '''    const uint32_t rb = m.row_bytes;
    for (size_t i = 0; i < n; ++i) {''',
                '''    if (m.lazy_pending) {
        std::string lerr;
        if (!m.ensure_opened(lerr)) {
            if (m.error.empty()) m.error = lerr.empty() ? "PleReader: lazy open failed" : lerr;
            m.tickets[id].pending = 0;
            lk.unlock();
            m.cv_done.notify_all();
            return Ticket{id};
        }
    }
    const uint32_t rb = m.row_bytes;
    for (size_t i = 0; i < n; ++i) {''',
            ),
            (
                '''        if (const uint8_t* hit = m.cache.find(rows[i])) {
            std::memcpy(dst, hit, rb);
            ++m.stats.cache_hits;
            continue;
        }
        const uint64_t at = m.table_offset + (uint64_t) rows[i] * rb;''',
                '''        const uint64_t at = m.table_offset + (uint64_t) rows[i] * rb;
        if (m.mmap_head != nullptr && at + rb <= m.mmap_head_bytes) {
            std::memcpy(dst, m.mmap_head + at, rb);
            ++m.stats.cache_hits;
            continue;
        }
        bool win_hit = false;
        for (const auto& w : m.ram_windows) {
            if (at >= w.lo && at + rb <= w.hi) {
                std::memcpy(dst, w.ptr + (at - w.lo), rb);
                ++m.stats.cache_hits;
                win_hit = true;
                break;
            }
        }
        if (win_hit) continue;
        if (const uint8_t* hit = m.cache.find(rows[i])) {
            std::memcpy(dst, hit, rb);
            ++m.stats.cache_hits;
            continue;
        }''',
            ),
            (
                '''    // the ticket is known before its reads are planned; it completes once every page it counts is read
    m.tickets[id].pending = 1;
    lk.unlock();''',
                '''    if (miss.empty()) {
        m.tickets[id].pending = 0;
        lk.unlock();
        if (rearm) m.cv_work.notify_one();
        return Ticket{id};
    }
    if (!m.file.is_open()) {
        if (m.error.empty()) m.error = "PleReader: row missed RAM windows with no SSD backing";
        m.tickets[id].pending = 0;
        lk.unlock();
        m.cv_done.notify_all();
        return Ticket{id};
    }
    // the ticket is known before its reads are planned; it completes once every page it counts is read
    m.tickets[id].pending = 1;
    lk.unlock();''',
            ),
        ],
    )

    # 6. Parallel cudaHostRegister across layer spans (BASALT_COLAB_RAM_PARALLEL_PIN)
    #    (MADV_HUGEPAGE on alloc_ram is native in Basalt v0.6.0 via #834)
    patch_file(
        src,
        'src/core/ram_load.cpp',
        'BASALT_COLAB_RAM_PARALLEL_PIN',
        [
            (
                '''    std::vector<RamPin> pins(spans.size());
    uint64_t pinned = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i] == nullptr) continue;
        const uint64_t padded = ram_padded(bytes[i]);
        const bool room = !limited || (pinned <= budget && padded <= budget - pinned);
        if constexpr (platform::kBoundHostRegistration) {
            // DXGI usage can lag registration. Also charge each span against the initial allowance.
            const uint64_t remaining = limited ? (budget > pinned ? budget - pinned : 0) : ~uint64_t(0);
            pins[i] = pin_ram_bounded(spans[i], padded, bytes[i], {}, mapped, nullptr, remaining);
        } else {
            pins[i] = pin_ram(spans[i], room ? padded : 0, bytes[i], {}, mapped);
        }
        pinned += pins[i].registered_bytes;
    }
    return pins;''',
                '''    // BASALT_COLAB_RAM_PARALLEL_PIN: parallelize cudaHostRegister across 4 threads when unlimited
    std::vector<RamPin> pins(spans.size());
    if constexpr (!platform::kBoundHostRegistration) {
        if (!limited && spans.size() >= 4) {
            int cur_dev = 0;
            if (cudaGetDevice(&cur_dev) != cudaSuccess) { (void) cudaGetLastError(); cur_dev = 0; }
            std::atomic<size_t> next{0};
            const size_t nw = std::min<size_t>(4, spans.size());
            auto worker = [&] {
                (void) cudaSetDevice(cur_dev);
                for (;;) {
                    const size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= spans.size()) break;
                    if (spans[i] == nullptr) continue;
                    pins[i] = pin_ram(spans[i], ram_padded(bytes[i]), bytes[i], {}, mapped);
                }
            };
            std::vector<std::thread> threads;
            threads.reserve(nw > 0 ? nw - 1 : 0);
            for (size_t w = 1; w < nw; ++w) threads.emplace_back(worker);
            worker();
            for (auto& t : threads) t.join();
            return pins;
        }
    }
    uint64_t pinned = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i] == nullptr) continue;
        const uint64_t padded = ram_padded(bytes[i]);
        const bool room = !limited || (pinned <= budget && padded <= budget - pinned);
        if constexpr (platform::kBoundHostRegistration) {
            // DXGI usage can lag registration. Also charge each span against the initial allowance.
            const uint64_t remaining = limited ? (budget > pinned ? budget - pinned : 0) : ~uint64_t(0);
            pins[i] = pin_ram_bounded(spans[i], padded, bytes[i], {}, mapped, nullptr, remaining);
        } else {
            pins[i] = pin_ram(spans[i], room ? padded : 0, bytes[i], {}, mapped);
        }
        pinned += pins[i].registered_bytes;
    }
    return pins;''',
            ),
        ],
    )


if __name__ == '__main__':
    main()
