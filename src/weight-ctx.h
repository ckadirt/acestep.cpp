#pragma once
// weight-ctx.h: format-independent weight loading context for ggml backends
//
// Manages a ggml_context for weight tensors + their backend buffer.
// Used by gguf-weights.h for all model loaders.
//
// Usage:
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = <loader>_load_tensor(&wctx, source, "name");
//   wctx_alloc(&wctx, backend);

#include "error.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    // Sticky failure flag. A loader that cannot find or convert a tensor sets
    // this and returns NULL. Model loaders issue ~100 gf_load_tensor calls in
    // a row; checking every one at the call site would bury the load code, so
    // instead they check this once before wctx_alloc, which also refuses to
    // allocate when it is set. Once true it never clears: the first failure is
    // the one worth reporting.
    bool failed;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void *         src;
        size_t               nbytes;
        size_t               offset;  // byte offset into dst tensor (0 for regular loads)
    };

    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data, kept alive until wctx_alloc.
    // unique_ptr keeps the data address stable even when the outer vector grows,
    // so src pointers stored in pending stay valid across staging.push_back().
    std::vector<std::unique_ptr<float[]>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    wctx->ctx    = ggml_init(params);
    wctx->buffer = NULL;
    wctx->failed = false;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
    if (!wctx->ctx) {
        ace_set_error(ACE_ERR_OOM, "[WeightCtx] ggml_init failed for %d tensors (%zu bytes)", n_tensors, ctx_size);
        wctx->failed = true;
    }
}

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    // A tensor load failed earlier; the error is already recorded. Refuse
    // rather than allocating a buffer for a model with holes in it.
    if (wctx->failed) {
        return false;
    }
    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        ace_set_error(ACE_ERR_OOM, "[WeightCtx] failed to allocate backend buffer");
        return false;
    }
    // Mark as weight buffer so ggml_backend_sched assigns ops to the correct
    // backend based on weight location (avoids fallback through expansion).
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t total = 0;
    for (auto & pc : wctx->pending) {
        ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        total += pc.nbytes;
    }
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend\n", wctx->pending.size(),
            (float) total / (1024 * 1024));
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
    }
    wctx->buffer = NULL;
    wctx->ctx    = NULL;
}
