#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "error.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
    // False when initialisation failed. Callers must check before using the
    // handles: backend_init no longer exits the process, so a failed pair is
    // a value they have to look at.
    bool           ok;
};

// Cached backend state (shared across all modules in the same binary)
static BackendPair g_backend_cache = {};
static int         g_backend_refs  = 0;

// Explicit thread count, 0 = use the heuristic below. Set once before the
// first backend_init; later changes do not affect an already-created backend.
static int g_backend_n_threads = 0;

// Override the CPU thread count.
//
// The heuristic is wrong on big.LITTLE. It assumes logical/2 means "physical
// cores", which holds for SMT desktops and not for a phone: on a 2 big + 6
// little part it returns 4 and lands three quarters of the work on cores a
// third the speed, where the fast cores then wait for them at every barrier.
// Hosts that know the topology should set the big-core count here.
static void backend_set_n_threads(int n) {
    g_backend_n_threads = n > 0 ? n : 0;
}

// Physical core count heuristic (logical / 2 for HT/SMT).
// Used for GGML CPU thread count: GEMM shares SIMD units across hyperthreads,
// so one thread per physical core is optimal.
static int backend_cpu_n_threads(void) {
    if (g_backend_n_threads > 0) {
        return g_backend_n_threads;
    }
    // Env override, for benchmarking a device without rebuilding. The API
    // setter wins over it; both win over the heuristic.
    if (const char * e = std::getenv("GGML_N_THREADS")) {
        int n = atoi(e);
        if (n > 0) {
            return n;
        }
    }
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
        }
    }
    return cpu;
}

// Collapse exact consecutive duplicate ggml log lines and report the total
// count when the run ends (tames the CUDA graph capture "reused" flood).
static void acestep_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    static char last[256] = { 0 };
    static int  count     = 0;

    if (count > 0 && strcmp(text, last) == 0) {
        count++;
        return;
    }

    if (count > 1) {
        fprintf(stderr, "[Dedup] Previous line repeated %d times total\n", count);
    }

    fputs(text, stderr);
    strncpy(last, text, sizeof(last) - 1);
    last[sizeof(last) - 1] = 0;
    count                  = 1;
    fflush(stderr);
}

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool).
static BackendPair backend_init(const char * label) {
    static bool log_installed = false;
    if (!log_installed) {
        ggml_log_set(acestep_ggml_log, nullptr);
        log_installed = true;
    }

    if (g_backend_refs > 0) {
        g_backend_refs++;
        fprintf(stderr, "[Load] %s backend: %s (shared)\n", label, ggml_backend_name(g_backend_cache.backend));
        return g_backend_cache;
    }

    ggml_backend_load_all();
    BackendPair bp = {};

    // GGML_BACKEND env var: force a specific device instead of auto-best.
    // Device names: CUDA0, Vulkan0, CPU, BLAS (see ggml_backend_dev_name).
    const char * force_backend = std::getenv("GGML_BACKEND");
    if (force_backend) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            char avail[256] = { 0 };
            size_t used     = 0;
            for (size_t i = 0; i < ggml_backend_dev_count() && used < sizeof(avail) - 1; i++) {
                int n = snprintf(avail + used, sizeof(avail) - used, " %s",
                                 ggml_backend_dev_name(ggml_backend_dev_get(i)));
                if (n > 0) {
                    used += (size_t) n;
                }
            }
            ace_set_error(ACE_ERR_NO_BACKEND, "[Load] GGML_BACKEND=%s not found. Available:%s", force_backend, avail);
            return bp;  // ok == false
        }
    } else {
        bp.backend = ggml_backend_init_best();
    }
    if (!bp.backend) {
        ace_set_error(ACE_ERR_NO_BACKEND, "[Load] no backend available");
        return bp;  // ok == false
    }
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        ace_set_error(ACE_ERR_NO_BACKEND, "[Load] failed to init CPU backend");
        if (bp.backend && bp.backend != bp.cpu_backend) {
            ggml_backend_free(bp.backend);
        }
        bp.backend = NULL;
        return bp;  // ok == false
    }
    bp.has_gpu = !best_is_cpu;
    bp.ok      = true;
    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\n", label, ggml_backend_name(bp.backend), n_threads);

    g_backend_cache = bp;
    g_backend_refs  = 1;
    return bp;
}

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    if (g_backend_refs <= 0) {
        return;
    }
    g_backend_refs--;
    if (g_backend_refs == 0) {
        if (backend && backend != cpu_backend) {
            ggml_backend_free(backend);
        }
        if (cpu_backend) {
            ggml_backend_free(cpu_backend);
        }
        g_backend_cache = {};
    }
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { NULL, NULL };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : NULL;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        ace_set_error(ACE_ERR_OOM, "[Load] failed to create scheduler (max_nodes=%d)", max_nodes);
        return NULL;
    }
    return sched;
}

// Combined init: backend pair + scheduler, with both failure paths handled.
// Every module loader opens with exactly this sequence, so it lives here once
// instead of eight times. On failure the error is already recorded and the
// backend reference (if any) has been released.
// Returns false and leaves the out params untouched on failure.
[[nodiscard]] static bool backend_init_sched(const char *           label,
                               int                    max_nodes,
                               ggml_backend_t *       backend_out,
                               ggml_backend_t *       cpu_out,
                               ggml_backend_sched_t * sched_out,
                               bool *                 has_gpu_out) {
    BackendPair bp = backend_init(label);
    if (!bp.ok) {
        return false;
    }
    ggml_backend_sched_t sched = backend_sched_new(bp, max_nodes);
    if (!sched) {
        backend_release(bp.backend, bp.cpu_backend);
        return false;
    }
    *backend_out = bp.backend;
    *cpu_out     = bp.cpu_backend;
    *sched_out   = sched;
    if (has_gpu_out) {
        *has_gpu_out = bp.has_gpu;
    }
    return true;
}
