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
//   ENCODE in        request JSON naming its audio
//   ENCODE out       done: EncodedBlob (below). paused: the request JSON again
//   LISTEN in        an EncodedBlob (ENCODE's output)
//   LISTEN out       done: the EncodedBlob, its caption filled if it was empty.
//                    paused: the input blob again
//   DIFFUSE in       request JSON, or an EncodedBlob
//   DIFFUSE out      paused: DiffuseBlob (below). done: raw [T,64] f32 latent
//   DECODE in        raw [T,64] f32 latent
//
// Audio in comes by path, inside the request (src_audio_path, ref_audio_path):
// the host hands over a request, never buffers, and a paused DIFFUSE blob
// carries the request, so a resume re-reads the same files. A finished DIFFUSE
// emits the latent with the repaint/lego splice already applied, because
// DECODE decodes a bare latent and has no source to splice from.
//
// The paused DIFFUSE blob stamps the backend name and solver. Resuming a run
// on a different backend re-derives the conditioning through different
// kernels, so the second half of the trajectory drifts: the track does not
// sound broken, it is simply not the one that was paused. Refusing is the
// honest answer.

#include "cantor_engine.h"

#include "audio-io.h"
#include "error.h"
#include "model-store.h"
#include "pipeline-lm.h"
#include "pipeline-synth.h"
#include "pipeline-understand.h"
#include "request.h"
#include "task-types.h"
#include "version.h"

#include <cstdlib>
#include <cstring>
#include <exception>
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

    // The understanding pipeline LISTEN runs, loaded on first use: it shares
    // the LM with PLAN/CODES through the store, so it costs no second copy.
    AceUnderstand * understand = nullptr;

    // DECODE output, valid until the next run_stage on this context.
    std::vector<float> audio;
    int                audio_n;

    // LISTEN output: what it heard, as JSON. Empty when the last stage was
    // not a LISTEN that finished.
    std::string heard;

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
           (1u << CANTOR_STAGE_DECODE) | (1u << CANTOR_STAGE_ENCODE) | (1u << CANTOR_STAGE_LISTEN);
}

// What this build does with audio. Kept in step with run_diffuse below: cover
// and cover-nofsq take src_audio_path, repaint inside the source is inpaint and
// outside it is extend, and ref_audio_path is the timbre reference. lego,
// extract and complete need the base DiT, which a catalog variant declares
// separately, and are not claimed here until a host has run them.
extern "C" const char * cantor_engine_capabilities(void) {
    return "cover,extend,inpaint,timbre";
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

// Paused LM blob. Fixed header, then the request JSON, then the token ids.
// Small: a few hundred tokens, so a few kilobytes against the DiT blob's 77.
#define LM_MAGIC "ACELM001"

struct LmHeader {
    char     magic[8];
    uint32_t phase;     // 1 = CoT/metadata, 2 = audio codes
    uint32_t n_tokens;
    uint32_t json_len;
};

// Paused DIFFUSE blob. Fixed header, then the request JSON, then the latent.
// ACEDIF02, for a run that started from ENCODE's latents, carries the whole
// EncodedBlob where 01 carries the JSON, so a resume never re-encodes.
#define DIFFUSE_MAGIC "ACEDIF01"
#define DIFFUSE_MAGIC_ENCODED "ACEDIF02"

// ENCODE's output: a request and the latents of the audio it names. The
// reference is omitted when it is the source (a kept voice): same latents.
#define ENCODED_MAGIC "ACEENC01"

struct EncodedHeader {
    char     magic[8];
    uint32_t json_len;
    uint32_t T_src;       // 0: no source
    uint32_t T_ref;       // 0: no reference, or the source is the reference
    uint32_t ref_is_src;  // 1: the reference is the source
};

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

// ---------------------------------------------------------- exception guard
//
// ggml backends throw. ggml-vulkan in particular throws std::runtime_error
// straight out of device init when a GPU lacks a required feature - an
// Adreno 619 without 16-bit storage terminates the process with
// "libc++abi: terminating due to uncaught exception".
//
// Part 5 removed every exit() and abort() from this repository's own code,
// but an exception crossing the ABI boundary kills the host just as dead, and
// crossing a C boundary at all is undefined behaviour. So every entry point
// that can reach ggml runs inside this.
template <typename F> static auto abi_guard(F && fn, decltype(fn()) on_throw) -> decltype(fn()) {
    try {
        return fn();
    } catch (const std::exception & e) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] backend threw: %s", e.what());
        return on_throw;
    } catch (...) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] backend threw an unknown exception");
        return on_throw;
    }
}

// ------------------------------------------------------------- trampolines

static bool cancel_tramp(void * ud) {
    cantor_ctx * c = (cantor_ctx *) ud;
    return c->cancel ? (c->cancel(c->userdata) != 0) : false;
}

// The pipelines report progress as (i, n); the ABI adds which stage it was,
// which only this side knows.
static void progress_tramp(int i, int n, void * ud) {
    cantor_ctx * c = (cantor_ctx *) ud;
    if (c->progress) {
        c->progress(c->stage, i, n, c->userdata);
    }
}

// ------------------------------------------------------------------ loading

static cantor_ctx * engine_load_inner(const cantor_component * components,
                                      size_t                   n,
                                      const cantor_load_opts * opts) {
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

    // Before anything creates a backend: the thread count is baked in when the
    // shared backend is first initialised, so setting it later is silently
    // ignored.
    if (o.n_threads > 0) {
        ace_engine_set_n_threads(o.n_threads);
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

extern "C" cantor_ctx * cantor_engine_load(const cantor_component * components,
                                           size_t                   n,
                                           const cantor_load_opts * opts) {
    ace_clear_error();
    cantor_ctx * c = abi_guard([&] { return engine_load_inner(components, n, opts); }, (cantor_ctx *) nullptr);
    // Contract: a NULL return always comes with a reason. Some internal load
    // paths still only log to stderr, so backstop rather than let a caller
    // read "no error" off a failure - which is what the T4 run did.
    if (!c && ace_error_code() == ACE_OK) {
        ace_set_error(ACE_ERR_BAD_MODEL, "[ABI] engine load failed (see stderr for the failing component)");
    }
    return c;
}

extern "C" void cantor_engine_free(cantor_ctx * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->understand) {
        ace_understand_free(ctx->understand);
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

extern "C" const char * cantor_engine_heard(cantor_ctx * ctx) {
    return ctx && !ctx->heard.empty() ? ctx->heard.c_str() : nullptr;
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

static cantor_status run_lm(cantor_ctx *    c,
                            int             mode,
                            const uint8_t * in,
                            size_t          in_len,
                            uint8_t **      out,
                            size_t *        out_len) {
    if (!c->lm) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] no lm component was loaded");
        return CANTOR_ERR;
    }

    // Resuming? A paused LM blob leads with the magic.
    AceLmCheckpoint ckpt;
    ckpt.phase       = 0;
    bool        resuming = (in_len > sizeof(LmHeader) && !memcmp(in, LM_MAGIC, 8));
    std::string json;

    if (resuming) {
        LmHeader h;
        memcpy(&h, in, sizeof(h));
        size_t need = sizeof(h) + h.json_len + (size_t) h.n_tokens * sizeof(int32_t);
        if (in_len != need) {
            ace_set_error(ACE_ERR_INTERNAL, "[ABI] paused LM blob is %zu bytes, header describes %zu", in_len, need);
            return CANTOR_ERR;
        }
        json.assign((const char *) in + sizeof(h), h.json_len);
        const int32_t * toks = (const int32_t *) (in + sizeof(h) + h.json_len);
        ckpt.phase           = (int) h.phase;
        ckpt.tokens.assign(toks, toks + h.n_tokens);
        fprintf(stderr, "[ABI] resuming LM phase %u with %u tokens\n", h.phase, h.n_tokens);
    } else {
        json.assign((const char *) in, in_len);
    }

    AceRequest req;
    request_init(&req);
    if (!request_parse_json(&req, json.c_str())) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] state blob is not a valid request JSON");
        return CANTOR_ERR;
    }

    AceRequest result;
    request_init(&result);
    int rc = ace_lm_generate(c->lm, &req, 1, &result, nullptr, nullptr, cancel_tramp, c, mode, &ckpt);
    if (rc < 0) {
        return CANTOR_ERR;
    }
    if (rc == 1) {
        // Paused: the tokens already paid for, plus the request they belong to.
        LmHeader h = {};
        memcpy(h.magic, LM_MAGIC, 8);
        h.phase    = (uint32_t) ckpt.phase;
        h.n_tokens = (uint32_t) ckpt.tokens.size();
        std::string j = request_to_json(&req, true);
        h.json_len    = (uint32_t) j.size();

        std::vector<uint8_t> buf(sizeof(h) + j.size() + ckpt.tokens.size() * sizeof(int32_t));
        memcpy(buf.data(), &h, sizeof(h));
        memcpy(buf.data() + sizeof(h), j.data(), j.size());
        std::vector<int32_t> t32(ckpt.tokens.begin(), ckpt.tokens.end());
        memcpy(buf.data() + sizeof(h) + j.size(), t32.data(), t32.size() * sizeof(int32_t));
        if (!blob_alloc(buf.data(), buf.size(), out, out_len)) {
            return CANTOR_ERR;
        }
        return CANTOR_PAUSED;
    }

    std::string enriched = request_to_json(&result, true);
    if (!blob_alloc(enriched.data(), enriched.size(), out, out_len)) {
        return CANTOR_ERR;
    }
    return CANTOR_DONE;
}

// --------------------------------------------------------- stage: DIFFUSE

// Tasks that read a source track. Without one the pipeline would have nothing
// to cover or repaint, so the request is refused here, by name, rather than
// failing deep in the DiT or quietly becoming text2music.
static bool task_needs_source(const std::string & task) {
    return task == TASK_COVER || task == TASK_COVER_NOFSQ || task == TASK_REPAINT || task == TASK_LEGO ||
           task == TASK_EXTRACT || task == TASK_COMPLETE;
}

// Read an audio file named by a request field into interleaved stereo 48 kHz,
// the layout ace_synth_job_run_dit takes. An empty path is "none".
static bool read_request_audio(const std::string & path, const char * field, std::vector<float> & out, int * len) {
    out.clear();
    *len = 0;
    if (path.empty()) {
        return true;
    }
    int     T      = 0;
    float * planar = audio_read_48k(path.c_str(), &T);
    if (!planar || T <= 0) {
        free(planar);
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] cannot read %s '%s' (WAV or MP3)", field, path.c_str());
        return false;
    }
    float * inter = audio_planar_to_interleaved(planar, T);
    free(planar);
    if (!inter) {
        ace_set_error(ACE_ERR_OOM, "[ABI] cannot allocate %s (%d samples)", field, T);
        return false;
    }
    out.assign(inter, inter + (size_t) 2 * T);
    free(inter);
    *len = T;
    return true;
}

// ----------------------------------------------------------- stage: ENCODE

// A request with its audio already encoded: what ENCODE produces and DIFFUSE
// takes in place of the files.
struct Encoded {
    std::string        json;
    std::vector<float> src;  // [T_src * 64]
    std::vector<float> ref;  // [T_ref * 64], empty when ref_is_src
    int                T_src      = 0;
    int                T_ref      = 0;
    bool               ref_is_src = false;
};

static std::vector<uint8_t> encoded_bytes(const Encoded & e) {
    EncodedHeader h = {};
    memcpy(h.magic, ENCODED_MAGIC, 8);
    h.json_len   = (uint32_t) e.json.size();
    h.T_src      = (uint32_t) e.T_src;
    // A kept voice's reference is the source: 0, as ENCODE writes it (parsing
    // fills T_ref with T_src for convenience, which must not round-trip).
    h.T_ref      = e.ref_is_src ? 0u : (uint32_t) e.T_ref;
    h.ref_is_src = e.ref_is_src ? 1u : 0u;
    std::vector<uint8_t> buf(sizeof(h) + e.json.size() + (e.src.size() + e.ref.size()) * sizeof(float));
    uint8_t * p = buf.data();
    memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    memcpy(p, e.json.data(), e.json.size());
    p += e.json.size();
    memcpy(p, e.src.data(), e.src.size() * sizeof(float));
    p += e.src.size() * sizeof(float);
    memcpy(p, e.ref.data(), e.ref.size() * sizeof(float));
    return buf;
}

static bool is_encoded(const uint8_t * in, size_t in_len) {
    return in_len >= sizeof(EncodedHeader) && !memcmp(in, ENCODED_MAGIC, 8);
}

// Parses an EncodedBlob that must fill exactly in_len bytes.
static bool parse_encoded(const uint8_t * in, size_t in_len, Encoded & e) {
    EncodedHeader h;
    memcpy(&h, in, sizeof(h));
    const size_t src_n = (size_t) h.T_src * 64;
    const size_t ref_n = h.ref_is_src ? 0 : (size_t) h.T_ref * 64;
    const size_t need  = sizeof(h) + h.json_len + (src_n + ref_n) * sizeof(float);
    if (in_len != need) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] encoded blob is %zu bytes, header describes %zu", in_len, need);
        return false;
    }
    const uint8_t * p = in + sizeof(h);
    e.json.assign((const char *) p, h.json_len);
    p += h.json_len;
    e.src.assign((const float *) p, (const float *) p + src_n);
    p += src_n * sizeof(float);
    e.ref.assign((const float *) p, (const float *) p + ref_n);
    e.T_src      = (int) h.T_src;
    e.T_ref      = h.ref_is_src ? (int) h.T_src : (int) h.T_ref;
    e.ref_is_src = h.ref_is_src != 0;
    return true;
}

static cantor_status run_encode(cantor_ctx *    c,
                                const uint8_t * in,
                                size_t          in_len,
                                uint8_t **      out,
                                size_t *        out_len) {
    if (!c->synth) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] dit/embed/vae components are required for ENCODE");
        return CANTOR_ERR;
    }
    Encoded e;
    e.json.assign((const char *) in, in_len);
    AceRequest req;
    request_init(&req);
    if (!request_parse_json(&req, e.json.c_str())) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] state blob is not a valid request JSON");
        return CANTOR_ERR;
    }
    if (req.src_audio_path.empty() && req.ref_audio_path.empty()) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] ENCODE needs src_audio_path or ref_audio_path");
        return CANTOR_ERR;
    }
    // A kept voice is the source itself: the same file, the same latents.
    e.ref_is_src = !req.ref_audio_path.empty() && req.ref_audio_path == req.src_audio_path;

    auto encode = [&](const std::string & path, const char * field, std::vector<float> & latent, int * T) {
        std::vector<float> audio;
        int                n = 0;
        if (!read_request_audio(path, field, audio, &n)) {
            return CANTOR_ERR;
        }
        if (!ace_synth_encode_audio(c->synth, audio.data(), n, latent, T, cancel_tramp, c)) {
            // Stopped between tiles: nothing worth keeping, so the resume
            // blob is the request itself and ENCODE starts over.
            if (ace_error_code() == ACE_ERR_CANCELLED) {
                return blob_alloc(in, in_len, out, out_len) ? CANTOR_PAUSED : CANTOR_ERR;
            }
            return CANTOR_ERR;
        }
        return CANTOR_DONE;
    };

    if (!req.src_audio_path.empty()) {
        cantor_status st = encode(req.src_audio_path, "src_audio_path", e.src, &e.T_src);
        if (st != CANTOR_DONE) {
            return st;
        }
    }
    if (!req.ref_audio_path.empty() && !e.ref_is_src) {
        cantor_status st = encode(req.ref_audio_path, "ref_audio_path", e.ref, &e.T_ref);
        if (st != CANTOR_DONE) {
            return st;
        }
    }
    fprintf(stderr, "[ABI] ENCODE: src %d frames, ref %s\n", e.T_src,
            e.ref_is_src ? "is the source" : (e.T_ref > 0 ? "encoded" : "none"));
    std::vector<uint8_t> buf = encoded_bytes(e);
    if (!blob_alloc(buf.data(), buf.size(), out, out_len)) {
        return CANTOR_ERR;
    }
    return CANTOR_DONE;
}

// ----------------------------------------------------------- stage: LISTEN

// The understanding pipeline returns the same failure for a stop as for an
// error, so LISTEN remembers whether the host asked it to stop.
struct ListenCancel {
    cantor_ctx * c;
    bool         fired;
};

static bool listen_cancel(void * ud) {
    auto * l = (ListenCancel *) ud;
    if (cancel_tramp(l->c)) {
        l->fired = true;
    }
    return l->fired;
}

// ENCODE's blob in, the same blob out with an empty caption filled from what
// the LM hears in the source's latents. The understanding pipeline stops
// before lyrics (caption_only): a small LM's lyrics of a song it only heard
// are rarely the song's, and the caption is what a version conditions on.
static cantor_status run_listen(cantor_ctx *    c,
                                const uint8_t * in,
                                size_t          in_len,
                                uint8_t **      out,
                                size_t *        out_len) {
    if (!c->lm || !c->synth) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] LISTEN needs the lm and dit/embed/vae components");
        return CANTOR_ERR;
    }
    if (!is_encoded(in, in_len)) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] LISTEN takes ENCODE's output, not a request");
        return CANTOR_ERR;
    }
    Encoded e;
    if (!parse_encoded(in, in_len, e)) {
        return CANTOR_ERR;
    }
    if (e.T_src <= 0) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] LISTEN needs a source: src_audio_path was not encoded");
        return CANTOR_ERR;
    }
    AceRequest req;
    request_init(&req);
    if (!request_parse_json(&req, e.json.c_str())) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] encoded blob's request is not valid JSON");
        return CANTOR_ERR;
    }
    if (!c->understand) {
        AceUnderstandParams p;
        ace_understand_default_params(&p);
        // The same LM key as PLAN/CODES (path, max_seq, batch), so the store
        // hands both pipelines one instance.
        p.model_path   = c->lm_path.c_str();
        p.dit_path     = c->dit_path.c_str();
        p.vae_path     = c->vae_path.c_str();
        p.max_seq      = c->lm_params.max_seq;
        p.max_batch    = c->lm_params.max_batch;
        p.use_fa       = c->lm_params.use_fa;
        p.caption_only = true;
        c->understand  = ace_understand_load(c->store, &p);
        if (!c->understand) {
            if (ace_error_code() == ACE_OK) {
                ace_set_error(ACE_ERR_BAD_MODEL, "[ABI] the understanding pipeline did not load");
            }
            return CANTOR_ERR;
        }
    }
    AceRequest heard;
    request_init(&heard);
    ListenCancel stop = { c, false };
    int          rc   = ace_understand_generate(c->understand, nullptr, 0, e.src.data(), e.T_src, &req, &heard,
                                                nullptr, nullptr, listen_cancel, &stop);
    if (rc != 0) {
        // Stopped mid-sentence: nothing worth keeping, LISTEN starts over.
        if (stop.fired) {
            return blob_alloc(in, in_len, out, out_len) ? CANTOR_PAUSED : CANTOR_ERR;
        }
        if (ace_error_code() == ACE_OK) {
            ace_set_error(ACE_ERR_INTERNAL, "[ABI] the LM could not describe the source");
        }
        return CANTOR_ERR;
    }
    AceRequest said;
    request_init(&said);
    said.caption        = heard.caption;
    said.bpm            = heard.bpm;
    said.keyscale       = heard.keyscale;
    said.timesignature  = heard.timesignature;
    said.vocal_language = heard.vocal_language;
    c->heard            = request_to_json(&said, true);
    fprintf(stderr, "[ABI] LISTEN: %zu-char caption\n", heard.caption.size());
    // A caption someone wrote is theirs: LISTEN only fills an empty one, and
    // otherwise hands its input on byte for byte.
    if (!req.caption.empty() || heard.caption.empty()) {
        return blob_alloc(in, in_len, out, out_len) ? CANTOR_DONE : CANTOR_ERR;
    }
    req.caption                = heard.caption;
    e.json                     = request_to_json(&req, true);
    std::vector<uint8_t> buf = encoded_bytes(e);
    if (!blob_alloc(buf.data(), buf.size(), out, out_len)) {
        return CANTOR_ERR;
    }
    return CANTOR_DONE;
}

static cantor_status run_diffuse(cantor_ctx *    c,
                                 const uint8_t * in,
                                 size_t          in_len,
                                 uint8_t **      out,
                                 size_t *        out_len) {
    if (!c->synth) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] dit/embed/vae components are required for DIFFUSE");
        return CANTOR_ERR;
    }

    // Resuming? A paused blob leads with the magic. Starting from ENCODE?
    // Its blob does. Otherwise the input is a request naming its files.
    const bool resuming = in_len > sizeof(DiffuseHeader) &&
                          (!memcmp(in, DIFFUSE_MAGIC, 8) || !memcmp(in, DIFFUSE_MAGIC_ENCODED, 8));
    DiffuseHeader hdr = {};
    std::string   json;
    Encoded       enc;
    bool          have_enc = false;

    if (resuming) {
        memcpy(&hdr, in, sizeof(hdr));
        size_t need = sizeof(hdr) + hdr.json_len + (size_t) hdr.batch_n * hdr.T * hdr.Oc * sizeof(float);
        if (in_len != need) {
            ace_set_error(ACE_ERR_INTERNAL, "[ABI] paused blob is %zu bytes, header describes %zu", in_len, need);
            return CANTOR_ERR;
        }
        const uint8_t * body = in + sizeof(hdr);
        if (!memcmp(in, DIFFUSE_MAGIC_ENCODED, 8)) {
            if (!is_encoded(body, hdr.json_len) || !parse_encoded(body, hdr.json_len, enc)) {
                return CANTOR_ERR;
            }
            have_enc = true;
            json     = enc.json;
        } else {
            json.assign((const char *) body, hdr.json_len);
        }
    } else if (is_encoded(in, in_len)) {
        if (!parse_encoded(in, in_len, enc)) {
            return CANTOR_ERR;
        }
        have_enc = true;
        json     = enc.json;
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

    if (task_needs_source(req.task_type) && (have_enc ? enc.T_src == 0 : req.src_audio_path.empty())) {
        ace_set_error(ACE_ERR_INTERNAL, "[ABI] task '%s' needs src_audio_path", req.task_type.c_str());
        return CANTOR_ERR;
    }
    // Latents from ENCODE, or the files themselves, encoded inside phase 1.
    std::vector<float> src, ref;
    int                src_len = 0, ref_len = 0;
    if (!have_enc && (!read_request_audio(req.src_audio_path, "src_audio_path", src, &src_len) ||
                      !read_request_audio(req.ref_audio_path, "ref_audio_path", ref, &ref_len))) {
        return CANTOR_ERR;
    }
    const float * src_p   = src.empty() ? nullptr : src.data();
    const float * ref_p   = ref.empty() ? nullptr : ref.data();
    const float * src_lat = have_enc && enc.T_src > 0 ? enc.src.data() : nullptr;
    const float * ref_lat = !have_enc || enc.T_ref == 0 ? nullptr
                            : enc.ref_is_src          ? enc.src.data()
                                                      : enc.ref.data();
    const int     src_T   = have_enc ? enc.T_src : 0;
    const int     ref_T   = have_enc ? enc.T_ref : 0;

    AceSynthJob * job = nullptr;

    if (resuming) {
        // Refuse a resume whose saved state was produced somewhere else.
        // Re-deriving the conditioning on a different backend goes through
        // different kernels, so the remaining trajectory diverges: the output
        // is a different track, not a broken one, which is exactly the kind of
        // difference a user cannot see.
        if (req.solver != std::string(hdr.solver)) {
            ace_set_error(ACE_ERR_INTERNAL, "[ABI] cannot resume: blob was paused on solver '%s', request asks '%s'",
                          hdr.solver, req.solver.c_str());
            return CANTOR_ERR;
        }
        // Backend match. Resuming re-derives the conditioning, and a different
        // backend runs that through different kernels, so the remaining
        // trajectory diverges. The output is a different track rather than a
        // broken one - the kind of wrongness nobody can see - so refuse.
        // Checked before any work is done, not after.
        const char * have = ace_synth_backend_name(c->synth);
        if (strncmp(hdr.backend, have, sizeof(hdr.backend)) != 0) {
            ace_set_error(ACE_ERR_INTERNAL,
                          "[ABI] cannot resume: blob was paused on backend '%s', this engine runs on '%s'. "
                          "Resume on the device the job started on.",
                          hdr.backend, have);
            return CANTOR_ERR;
        }

        // Rebuild phase 1 up to the DiT, then hand it the saved latent. There
        // is no cheaper route: the conditioning is not in the blob by design.
        job = ace_synth_job_run_dit(c->synth, &req, src_p, src_len, src_lat, src_T, ref_p, ref_len, ref_lat, ref_T,
                                    1, ace_synth_abort_immediately, nullptr);
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
        job = ace_synth_job_run_dit(c->synth, &req, src_p, src_len, src_lat, src_T, ref_p, ref_len, ref_lat, ref_T,
                                    1, cancel_tramp, c);
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
        memcpy(h.magic, have_enc ? DIFFUSE_MAGIC_ENCODED : DIFFUSE_MAGIC, 8);
        h.step      = (uint32_t) step;
        h.num_steps = (uint32_t) total;
        h.batch_n   = (uint32_t) bn;
        h.T         = (uint32_t) T;
        h.Oc        = (uint32_t) Oc;
        // The request with its seed resolved, and, after ENCODE, the latents
        // it was conditioned on: a resume must not encode again.
        std::string j = request_to_json(&req, true);
        if (have_enc) {
            enc.json                     = j;
            std::vector<uint8_t> encoded = encoded_bytes(enc);
            j.assign((const char *) encoded.data(), encoded.size());
        }
        h.json_len = (uint32_t) j.size();
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

    // Completed: emit the latent in the wire format DECODE and /vae both take,
    // spliced, so a repaint keeps its source outside the region.
    int                T = 0;
    std::vector<float> copy;
    bool               ok = ace_synth_job_get_spliced_latent(job, 0, copy, &T);
    ace_synth_job_free(job);
    if (!ok) {
        return CANTOR_ERR;
    }

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
    ctx->heard.clear();
    if (ctx->synth) {
        ace_synth_set_progress(ctx->synth, on_progress ? progress_tramp : nullptr, ctx);
    }
    if (ctx->lm) {
        ace_lm_set_progress(ctx->lm, on_progress ? progress_tramp : nullptr, ctx);
    }

    switch (stage) {
        case CANTOR_STAGE_PLAN:
            return abi_guard([&] { return run_lm(ctx, LM_MODE_INSPIRE, state_in, in_len, state_out, out_len); },
                             CANTOR_ERR);
        case CANTOR_STAGE_CODES:
            return abi_guard([&] { return run_lm(ctx, LM_MODE_GENERATE, state_in, in_len, state_out, out_len); },
                             CANTOR_ERR);
        case CANTOR_STAGE_DIFFUSE:
            return abi_guard([&] { return run_diffuse(ctx, state_in, in_len, state_out, out_len); }, CANTOR_ERR);
        case CANTOR_STAGE_DECODE:
            return abi_guard([&] { return run_decode(ctx, state_in, in_len); }, CANTOR_ERR);
        case CANTOR_STAGE_ENCODE:
            return abi_guard([&] { return run_encode(ctx, state_in, in_len, state_out, out_len); }, CANTOR_ERR);
        case CANTOR_STAGE_LISTEN:
            return abi_guard([&] { return run_listen(ctx, state_in, in_len, state_out, out_len); }, CANTOR_ERR);
    }
    ace_set_error(ACE_ERR_INTERNAL, "[ABI] unknown stage %d", (int) stage);
    return CANTOR_ERR;
}
