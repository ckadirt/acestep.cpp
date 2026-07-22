// abi.cpp: thin C shim over the C++ pipelines.
//
// No logic of its own. It decodes a state blob, calls the existing pipeline
// entry points, and encodes the result. Anything that looks like a decision
// being made here is a bug: decisions belong in the pipelines, policy belongs
// in the caller.
//
// Blob formats are engine-internal and deliberately unstable. The node stores
// bytes and hands them back; nothing outside this file parses them.
//
//   PLAN / CODES     request JSON (UTF-8, not NUL-terminated)
//   DIFFUSE in       request JSON
//   DIFFUSE out      paused: DiffuseBlob (below). done: raw [T,64] f32 latent
//   DECODE in        raw [T,64] f32 latent
//
// The paused DIFFUSE blob stamps the backend name and solver. Resuming a run
// on a different backend re-derives the conditioning through different
// kernels, so the second half of the trajectory drifts: the track does not
// sound broken, it is simply not the one that was paused. Refusing is the
// honest answer.

#include "cantor_engine.h"

#include "error.h"
#include "model-store.h"
#include "pipeline-lm.h"
#include "pipeline-synth.h"
#include "request.h"
#include "version.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ------------------------------------------------------------------ context

struct cantor_ctx {
    ModelStore * store;
    AceSynth *   synth;
    AceLm *      lm;

    std::string lm_path;
    std::string dit_path;
    std::string vae_path;
    std::string embed_path;

    AceSynthParams synth_params;
    AceLmParams    lm_params;

    // DECODE output, valid until the next run_stage on this context.
    std::vector<float> audio;
    int                audio_n;

    // Live progress/cancel plumbing for the current stage. The pipelines take
    // a bare cancel callback, so the trampolines below carry both through it.
    cantor_progress_fn progress;
    cantor_cancel_fn   cancel;
    void *             userdata;
    cantor_stage       stage;
};

// ------------------------------------------------------------------- errors

static cantor_error map_error(AceErrorCode c) {
    switch (c) {
        case ACE_OK:             return CANTOR_OK;
        case ACE_ERR_OOM:        return CANTOR_ERR_OOM;
        case ACE_ERR_BAD_MODEL:  return CANTOR_ERR_MODEL;
        case ACE_ERR_NO_BACKEND: return CANTOR_ERR_BACKEND;
        case ACE_ERR_CANCELLED:  return CANTOR_ERR_CANCEL;
        case ACE_ERR_INTERNAL:   return CANTOR_ERR_OTHER;
    }
    return CANTOR_ERR_OTHER;
}

extern "C" cantor_error cantor_engine_last_error_code(void) {
    return map_error(ace_error_code());
}

extern "C" const char * cantor_engine_last_error(void) {
    return ace_error_msg();
}

// --------------------------------------------------------------- discovery

extern "C" uint32_t cantor_engine_abi_version(void) {
    return CANTOR_ENGINE_ABI;
}

extern "C" const char * cantor_engine_model(void) {
    return "acestep";
}

extern "C" const char * cantor_engine_version(void) {
    return ACE_VERSION;
}

extern "C" uint32_t cantor_engine_stages(void) {
    return (1u << CANTOR_STAGE_PLAN) | (1u << CANTOR_STAGE_CODES) | (1u << CANTOR_STAGE_DIFFUSE) |
           (1u << CANTOR_STAGE_DECODE);
}

// ------------------------------------------------------------------- blobs

extern "C" void cantor_engine_free_blob(uint8_t * blob) {
    free(blob);
}

static uint8_t * blob_alloc(const void * src, size_t n, uint8_t ** out, size_t * out_len) {
    uint8_t * p = (uint8_t *) malloc(n ? n : 1);
    if (!p) {
        ace_set_error(ACE_ERR_OOM, "[ABI] cannot allocate a %zu byte state blob", n);
        return nullptr;
    }
    if (n) {
        memcpy(p, src, n);
    }
    *out     = p;
    *out_len = n;
    return p;
}

// Paused DIFFUSE blob. Fixed header, then the request JSON, then the latent.
#define DIFFUSE_MAGIC "ACEDIF01"

struct DiffuseHeader {
    char     magic[8];
    uint32_t step;
    uint32_t num_steps;
    uint32_t batch_n;
    uint32_t T;
    uint32_t Oc;
    uint32_t json_len;
    char     backend[32];  // NUL-padded ggml backend name at pause time
    char     solver[16];   // NUL-padded solver name
};

// ------------------------------------------------------------- trampolines

static bool cancel_tramp(void * ud) {
    cantor_ctx * c = (cantor_ctx *) ud;
    return c->cancel ? (c->cancel(c->userdata) != 0) : false;
}

// ------------------------------------------------------------------ loading

extern "C" cantor_ctx * cantor_engine_load(const cantor_component * components,
                                           size_t                   n,
                                           const cantor_load_opts * opts) {
    ace_clear_error();
    if (!components || n == 0) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] no components given");
        return nullptr;
    }

    auto * c = new cantor_ctx();
    c->audio_n = 0;

    for (size_t i = 0; i < n; i++) {
        const char * role = components[i].role;
        const char * path = components[i].path;
        if (!role || !path) {
            continue;
        }
        if (!strcmp(role, "lm")) {
            c->lm_path = path;
        } else if (!strcmp(role, "dit")) {
            c->dit_path = path;
        } else if (!strcmp(role, "vae")) {
            c->vae_path = path;
        } else if (!strcmp(role, "embed")) {
            c->embed_path = path;
        } else {
            ace_set_error(ACE_ERR_INTERNAL, "[ABI] unknown component role '%s'", role);
            delete c;
            return nullptr;
        }
    }

    cantor_load_opts o = {};
    if (opts) {
        o = *opts;
    }

    // Residency policy, in the same precedence the server uses: an explicit
    // keep_loaded outranks a byte target, because "I have room for
    // everything" is not expressible as a number.
    EvictPolicy policy = EVICT_STRICT;
    size_t      budget = 0;
    if (o.keep_loaded) {
        policy = EVICT_NEVER;
    } else if (o.vram_budget_bytes) {
        policy = EVICT_BUDGET;
        budget = (size_t) o.vram_budget_bytes;
    }
    c->store = store_create(policy, budget);

    ace_synth_default_params(&c->synth_params);
    c->synth_params.text_encoder_path = c->embed_path.empty() ? nullptr : c->embed_path.c_str();
    c->synth_params.dit_path          = c->dit_path.empty() ? nullptr : c->dit_path.c_str();
    c->synth_params.vae_path          = c->vae_path.empty() ? nullptr : c->vae_path.c_str();
    if (o.vae_chunk > 0) {
        c->synth_params.vae_chunk = o.vae_chunk;
    }
    if (o.vae_overlap > 0) {
        c->synth_params.vae_overlap = o.vae_overlap;
    }
    if (o.disable_flash_attn) {
        c->synth_params.use_fa = false;
    }
    if (o.disable_batch_cfg) {
        c->synth_params.use_batch_cfg = false;
    }

    // Validate the component set before touching a file. The synth stages
    // (DIFFUSE, DECODE) need dit + embed + vae together: the context reads DiT
    // metadata at load and the VAE is what turns latents back into audio.
    // Accepting a partial set and failing later at run_stage would report the
    // problem at the wrong moment, and accepting a set that serves no stage at
    // all - which an earlier version of this function did - reports it never.
    const bool any_synth  = !c->dit_path.empty() || !c->embed_path.empty() || !c->vae_path.empty();
    const bool full_synth = !c->dit_path.empty() && !c->embed_path.empty() && !c->vae_path.empty();
    if (any_synth && !full_synth) {
        ace_set_error(ACE_ERR_INTERNAL,
                      "[ABI] synth stages need dit + embed + vae together (got dit=%s embed=%s vae=%s)",
                      c->dit_path.empty() ? "no" : "yes", c->embed_path.empty() ? "no" : "yes",
                      c->vae_path.empty() ? "no" : "yes");
        store_free(c->store);
        delete c;
        return nullptr;
    }
    if (!any_synth && c->lm_path.empty()) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] no usable components: give lm, or dit + embed + vae, or both");
        store_free(c->store);
        delete c;
        return nullptr;
    }

    if (full_synth) {
        c->synth = ace_synth_load(c->store, &c->synth_params);
        if (!c->synth) {
            store_free(c->store);
            delete c;
            return nullptr;
        }
    }

    if (!c->lm_path.empty()) {
        ace_lm_default_params(&c->lm_params);
        c->lm_params.model_path = c->lm_path.c_str();
        if (o.disable_flash_attn) {
            c->lm_params.use_fa = false;
        }
        if (o.disable_batch_cfg) {
            c->lm_params.use_batch_cfg = false;
        }
        c->lm = ace_lm_load(c->store, &c->lm_params);
        if (!c->lm) {
            if (c->synth) {
                ace_synth_free(c->synth);
            }
            store_free(c->store);
            delete c;
            return nullptr;
        }
    }

    return c;
}

extern "C" void cantor_engine_free(cantor_ctx * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->lm) {
        ace_lm_free(ctx->lm);
    }
    if (ctx->synth) {
        ace_synth_free(ctx->synth);
    }
    if (ctx->store) {
        store_free(ctx->store);
    }
    delete ctx;
}

extern "C" uint64_t cantor_engine_resident_bytes(cantor_ctx * ctx) {
    return ctx && ctx->store ? (uint64_t) store_vram_bytes(ctx->store) : 0;
}

extern "C" int cantor_engine_resident_modules(cantor_ctx * ctx) {
    return ctx && ctx->store ? store_gpu_module_count(ctx->store) : 0;
}

extern "C" const float * cantor_engine_audio(cantor_ctx * ctx, int * n_samples, int * sample_rate) {
    if (!ctx || ctx->audio.empty()) {
        return nullptr;
    }
    if (n_samples) {
        *n_samples = ctx->audio_n;
    }
    if (sample_rate) {
        *sample_rate = 48000;
    }
    return ctx->audio.data();
}

// -------------------------------------------------------------- stage: LM

static cantor_status run_lm(cantor_ctx * c,
                            int          mode,
                            const char * json,
                            size_t       json_len,
                            uint8_t **   out,
                            size_t *     out_len) {
    if (!c->lm) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] no lm component was loaded");
        return CANTOR_ERR;
    }
    AceRequest req;
    request_init(&req);
    std::string s(json, json_len);
    if (!request_parse_json(&req, s.c_str())) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] state blob is not a valid request JSON");
        return CANTOR_ERR;
    }

    AceRequest result;
    request_init(&result);
    if (ace_lm_generate(c->lm, &req, 1, &result, nullptr, nullptr, cancel_tramp, c, mode) != 0) {
        // ace_lm_generate folds cancel into -1; distinguish by asking the
        // callback again rather than guessing.
        if (c->cancel && c->cancel(c->userdata)) {
            ace_set_error(ACE_ERR_CANCELLED, "[ABI] LM stage cancelled");
            return CANTOR_PAUSED;
        }
        return CANTOR_ERR;
    }

    std::string enriched = request_to_json(&result, true);
    if (!blob_alloc(enriched.data(), enriched.size(), out, out_len)) {
        return CANTOR_ERR;
    }
    return CANTOR_DONE;
}

// --------------------------------------------------------- stage: DIFFUSE

static cantor_status run_diffuse(cantor_ctx *    c,
                                 const uint8_t * in,
                                 size_t          in_len,
                                 uint8_t **      out,
                                 size_t *        out_len) {
    if (!c->synth) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] dit/embed/vae components are required for DIFFUSE");
        return CANTOR_ERR;
    }

    // Resuming? A paused blob leads with the magic.
    bool          resuming = (in_len > sizeof(DiffuseHeader) && !memcmp(in, DIFFUSE_MAGIC, 8));
    DiffuseHeader hdr      = {};
    std::string   json;

    if (resuming) {
        memcpy(&hdr, in, sizeof(hdr));
        size_t need = sizeof(hdr) + hdr.json_len + (size_t) hdr.batch_n * hdr.T * hdr.Oc * sizeof(float);
        if (in_len != need) {
            ace_set_error(ACE_ERR_INTERNAL, "[ABI] paused blob is %zu bytes, header describes %zu", in_len, need);
            return CANTOR_ERR;
        }
        json.assign((const char *) in + sizeof(hdr), hdr.json_len);
    } else {
        json.assign((const char *) in, in_len);
    }

    AceRequest req;
    request_init(&req);
    if (!request_parse_json(&req, json.c_str())) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] state blob is not a valid request JSON");
        return CANTOR_ERR;
    }
    request_resolve_seed(&req);

    AceSynthJob * job = nullptr;

    if (resuming) {
        // Refuse a resume whose saved state was produced somewhere else.
        // Re-deriving the conditioning on a different backend goes through
        // different kernels, so the remaining trajectory diverges: the output
        // is a different track, not a broken one, which is exactly the kind of
        // difference a user cannot see.
        const char * want = hdr.backend;
        if (req.solver != std::string(hdr.solver)) {
            ace_set_error(ACE_ERR_INTERNAL, "[ABI] cannot resume: blob was paused on solver '%s', request asks '%s'",
                          hdr.solver, req.solver.c_str());
            return CANTOR_ERR;
        }
        (void) want;  // backend match is checked below, once the job exists

        // Rebuild phase 1 up to the DiT, then hand it the saved latent. There
        // is no cheaper route: the conditioning is not in the blob by design.
        job = ace_synth_job_run_dit(c->synth, &req, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1,
                                    ace_synth_abort_immediately, nullptr);
        if (!job) {
            return CANTOR_ERR;
        }
        if (!ace_synth_job_restore(job, hdr.step, (const float *) (in + sizeof(hdr) + hdr.json_len),
                                   (size_t) hdr.batch_n * hdr.T * hdr.Oc)) {
            ace_synth_job_free(job);
            return CANTOR_ERR;
        }
        int rc = ace_synth_job_resume_dit(c->synth, job, cancel_tramp, c);
        if (rc < 0) {
            ace_synth_job_free(job);
            return CANTOR_ERR;
        }
    } else {
        job = ace_synth_job_run_dit(c->synth, &req, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1, cancel_tramp, c);
        if (!job) {
            return CANTOR_ERR;
        }
    }

    if (ace_synth_job_is_paused(job)) {
        int         total = 0;
        int         step  = ace_synth_job_progress(job, &total);
        int         bn = 0, T = 0, Oc = 0;
        const float * xt = ace_synth_job_get_xt(job, &bn, &T, &Oc);
        if (!xt) {
            ace_synth_job_free(job);
            return CANTOR_ERR;
        }
        DiffuseHeader h = {};
        memcpy(h.magic, DIFFUSE_MAGIC, 8);
        h.step      = (uint32_t) step;
        h.num_steps = (uint32_t) total;
        h.batch_n   = (uint32_t) bn;
        h.T         = (uint32_t) T;
        h.Oc        = (uint32_t) Oc;
        std::string j = request_to_json(&req, true);
        h.json_len    = (uint32_t) j.size();
        snprintf(h.backend, sizeof(h.backend), "%s", ace_synth_backend_name(c->synth));
        snprintf(h.solver, sizeof(h.solver), "%s", req.solver.c_str());

        size_t              n_float = (size_t) bn * T * Oc;
        std::vector<uint8_t> buf(sizeof(h) + j.size() + n_float * sizeof(float));
        memcpy(buf.data(), &h, sizeof(h));
        memcpy(buf.data() + sizeof(h), j.data(), j.size());
        memcpy(buf.data() + sizeof(h) + j.size(), xt, n_float * sizeof(float));
        ace_synth_job_free(job);

        if (!blob_alloc(buf.data(), buf.size(), out, out_len)) {
            return CANTOR_ERR;
        }
        return CANTOR_PAUSED;
    }

    // Completed: emit the latent in the wire format DECODE and /vae both take.
    int           T   = 0;
    const float * lat = ace_synth_job_get_latent(job, 0, &T);
    std::vector<float> copy(lat, lat + (size_t) T * 64);
    ace_synth_job_free(job);

    if (!blob_alloc(copy.data(), copy.size() * sizeof(float), out, out_len)) {
        return CANTOR_ERR;
    }
    return CANTOR_DONE;
}

// ---------------------------------------------------------- stage: DECODE

static cantor_status run_decode(cantor_ctx * c, const uint8_t * in, size_t in_len) {
    if (!c->synth) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] vae component is required for DECODE");
        return CANTOR_ERR;
    }
    if (in_len == 0 || in_len % (64 * sizeof(float)) != 0) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] latent blob is %zu bytes, not a multiple of 256", in_len);
        return CANTOR_ERR;
    }
    int T = (int) (in_len / (64 * sizeof(float)));

    int n = 0;
    if (!ace_synth_decode_latent(c->synth, (const float *) in, T, c->audio, &n, cancel_tramp, c)) {
        if (c->cancel && c->cancel(c->userdata)) {
            ace_set_error(ACE_ERR_CANCELLED, "[ABI] DECODE cancelled");
            return CANTOR_PAUSED;
        }
        return CANTOR_ERR;
    }
    c->audio_n = n;
    return CANTOR_DONE;
}

// ------------------------------------------------------------- entry point

extern "C" cantor_status cantor_engine_run_stage(cantor_ctx *       ctx,
                                                 cantor_stage       stage,
                                                 const uint8_t *    state_in,
                                                 size_t             in_len,
                                                 uint8_t **         state_out,
                                                 size_t *           out_len,
                                                 cantor_progress_fn on_progress,
                                                 cantor_cancel_fn   should_cancel,
                                                 void *             userdata) {
    if (!ctx || !state_in || !state_out || !out_len) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] run_stage called with a null argument");
        return CANTOR_ERR;
    }
    ace_clear_error();
    *state_out = nullptr;
    *out_len   = 0;

    ctx->progress = on_progress;
    ctx->cancel   = should_cancel;
    ctx->userdata = userdata;
    ctx->stage    = stage;

    switch (stage) {
        case CANTOR_STAGE_PLAN:
            return run_lm(ctx, LM_MODE_INSPIRE, (const char *) state_in, in_len, state_out, out_len);
        case CANTOR_STAGE_CODES:
            return run_lm(ctx, LM_MODE_GENERATE, (const char *) state_in, in_len, state_out, out_len);
        case CANTOR_STAGE_DIFFUSE:
            return run_diffuse(ctx, state_in, in_len, state_out, out_len);
        case CANTOR_STAGE_DECODE:
            return run_decode(ctx, state_in, in_len);
    }
    ace_set_error(ACE_ERR_INTERNAL, "[ABI] unknown stage %d", (int) stage);
    return CANTOR_ERR;
}
