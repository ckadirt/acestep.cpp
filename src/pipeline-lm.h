#pragma once
// pipeline-lm.h: ACE-Step LM pipeline
//
// A lightweight context bound to a ModelStore. On each generate call the
// pipeline acquires the Qwen3 LM, BPE tokenizer and FSM template from the
// store and releases the LM through RAII before returning.

#include "request.h"
#include "task-types.h"

#include <vector>

struct AceLm;
struct ModelStore;

struct AceLmParams {
    const char * model_path;     // LM GGUF (required)
    int          max_seq;        // KV cache length
    int          max_batch;      // max lm_batch_size for generate
    bool         use_fsm;        // constrained decoding
    bool         use_fa;         // flash attention
    bool         use_batch_cfg;  // batch cond+uncond in one forward
    bool         clamp_fp16;     // clamp hidden states to FP16 range
};

// Progress sink, called from inside the decode loops with tokens produced
// and the cap. Set on the context for the same reason the synth one is: the
// callback belongs to whoever is watching, not to a particular call.
typedef void (*AceLmProgressFn)(int i, int n, void * userdata);

// Token-level checkpoint.
//
// The KV cache is the obvious thing to save and the wrong one: it is
// n_kv_sets x layers x 2 x D x max_seq x Nkv in F16, hundreds of megabytes at
// max_seq 8192. Writing that to phone storage costs more than recomputing it.
//
// So the checkpoint is the generated TOKEN IDS. On resume the prompt and
// those tokens are re-prefilled in a single batched forward, which is one
// compute-bound pass where generating them was that many sequential
// memory-bound decode steps - typically ten to fifty times faster than
// regenerating. Sampling is not bit-identical afterwards (the RNG stream is
// not saved), but every token already paid for is preserved, which is the
// part the user cares about.
//
// Only lm_batch_size == 1 is supported: with N > 1 each sequence has its own
// token stream and FSM state, and resuming a partially-finished batch is a
// different problem. ace_lm_generate refuses rather than dropping sequences.
struct AceLmCheckpoint {
    int              phase;   // 1 = CoT/metadata, 2 = audio codes. 0 = nothing saved.
    std::vector<int> tokens;  // generated so far, excluding the prompt
};

void ace_lm_set_progress(AceLm * ctx, AceLmProgressFn fn, void * userdata);

void ace_lm_default_params(AceLmParams * p);

// Build a lightweight LM context bound to a ModelStore. The GPU model is
// acquired per request, never owned by the context. NULL on invalid input.
AceLm * ace_lm_load(ModelStore * store, const AceLmParams * params);

// Enrich request with metadata, lyrics, audio codes.
// out[lm_batch_size] allocated by caller, filled with enriched copies of req.
// mode: LM_MODE_GENERATE (full), LM_MODE_INSPIRE (no codes), LM_MODE_FORMAT (no codes).
// dump_logits/dump_tokens: debug output paths (NULL to disable).
// cancel/cancel_data: abort callback, polled between tokens. NULL = never cancel.
// ckpt: token-level checkpoint, in and out. When non-NULL and carrying
//   tokens, generation resumes from them. On a cancel it is filled with what
//   was produced and the call returns 1 (paused) instead of -1 (error).
// Returns 0 on success, 1 when paused with ckpt filled, -1 on error.
int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * req,
                    int                lm_batch_size,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens,
                    bool (*cancel)(void *) = nullptr,
                    void * cancel_data     = nullptr,
                    int    mode            = LM_MODE_GENERATE,
                    AceLmCheckpoint * ckpt = nullptr);

void ace_lm_free(AceLm * ctx);

// Read the LM ModelKey the context builds for store_require_lm. Used by
// test-model-store to verify both ace_lm and ace_understand resolve to the
// same LM instance when their params are propagated consistently.
struct ModelKey;
const ModelKey * ace_lm_lm_key(const AceLm * ctx);
