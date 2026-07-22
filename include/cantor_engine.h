// cantor_engine.h — the only surface the node depends on.
//
// Everything else in this repository is free to change without the node
// caring. Bump CANTOR_ENGINE_ABI on any breaking change to what is declared
// here; the node refuses a version it does not understand, which is what
// stops "update the fork, cut a release, point at it" from turning into a
// silent segfault on someone's machine.
//
// The generation model is a chain of stages, not one blocking call. Each
// stage consumes a state blob and produces the next one, so work can stop
// between stages, resume inside one, and be scheduled by which model it
// needs rather than by which song it belongs to.
//
// State blobs are OPAQUE to the caller. The node stores bytes and hands them
// back; it never parses them. That is deliberate: the engine can change what
// is in a blob without touching the node and without bumping this ABI. Only
// the signatures below are the contract.

#ifndef CANTOR_ENGINE_H
#define CANTOR_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Export marker. The shared library is built with hidden visibility so that
// only these symbols are reachable from the host - acestep-core's internals
// and ggml's stay private, which keeps them from colliding with whatever the
// host has already loaded.
#if defined(_WIN32)
#  if defined(CANTOR_ENGINE_BUILD)
#    define CANTOR_API __declspec(dllexport)
#  else
#    define CANTOR_API __declspec(dllimport)
#  endif
#else
#  define CANTOR_API __attribute__((visibility("default")))
#endif

#define CANTOR_ENGINE_ABI 1

// ---------------------------------------------------------------- discovery

// Called first. The node refuses a value it does not know.
CANTOR_API uint32_t cantor_engine_abi_version(void);

// Which model family this build understands, e.g. "acestep". Static string.
CANTOR_API const char * cantor_engine_model(void);

// Build identifier (git describe), for logs and bug reports. Static string.
CANTOR_API const char * cantor_engine_version(void);

// ------------------------------------------------------------------- stages

typedef enum {
    CANTOR_STAGE_PLAN    = 1,  // LM: caption -> metadata + lyrics, no audio codes
    CANTOR_STAGE_CODES   = 2,  // LM: -> audio codes
    CANTOR_STAGE_DIFFUSE = 3,  // DiT: flow matching, re-enterable
    CANTOR_STAGE_DECODE  = 4,  // VAE: latents -> PCM
} cantor_stage;

// Bitmask of the stages this build can run, so a cut-down or CPU-only engine
// can advertise honestly instead of failing at call time.
// Bit position is the stage value: (1u << CANTOR_STAGE_DIFFUSE) etc.
CANTOR_API uint32_t cantor_engine_stages(void);

typedef enum {
    CANTOR_DONE   = 0,   // stage complete; state_out feeds the next stage
    CANTOR_PAUSED = 1,   // stopped on the cancel callback; state_out resumes THIS stage
    CANTOR_ERR    = -1,  // see cantor_engine_last_error()
} cantor_status;

// Error classes a caller can act on differently. Anything finer belongs in
// the message, not here.
typedef enum {
    CANTOR_OK          = 0,
    CANTOR_ERR_OOM     = 1,  // retry smaller: lower vae_chunk, shorter duration, lighter variant
    CANTOR_ERR_MODEL   = 2,  // the file is wrong for this build; re-pull the blob
    CANTOR_ERR_BACKEND = 3,  // requested backend unavailable; fall back
    CANTOR_ERR_CANCEL  = 4,  // normal outcome, not a failure
    CANTOR_ERR_OTHER   = 5,
} cantor_error;

CANTOR_API cantor_error cantor_engine_last_error_code(void);
CANTOR_API const char * cantor_engine_last_error(void);

// ----------------------------------------------------------------- contexts

typedef struct cantor_ctx cantor_ctx;

// One model file, by role. Paths point into the node's blob store; the roles
// match the catalog's component roles exactly.
typedef struct {
    const char * role;  // "lm" | "dit" | "vae" | "embed"
    const char * path;
} cantor_component;

// Residency and device knobs. Zero-initialise and set what you care about.
typedef struct {
    // 0 keeps at most one module resident (lowest peak, most reloading).
    // Non-zero keeps as much as fits under this many bytes, evicting the
    // least recently used first. Peak may exceed it by roughly one module:
    // a module's size is not known until it is loaded.
    uint64_t vram_budget_bytes;

    // Keep everything resident regardless of budget. Overrides the above.
    int      keep_loaded;

    // VAE tiling. The VAE, not the DiT, is the memory peak. Smaller tiles
    // lower it and give finer cancel and progress granularity, at the cost
    // of more tiles. 0 uses the engine default (1024 / 64).
    int      vae_chunk;
    int      vae_overlap;

    // 0 uses the engine's physical-core heuristic, which is wrong on
    // big.LITTLE: set it to the big-core count there.
    int      n_threads;

    // Per-backend quirks, wired from the loader's driver checks.
    int      disable_flash_attn;
    int      disable_batch_cfg;
} cantor_load_opts;

// Load an engine context. Returns NULL on failure; see the error accessors.
CANTOR_API cantor_ctx * cantor_engine_load(const cantor_component * components, size_t n, const cantor_load_opts * opts);

CANTOR_API void cantor_engine_free(cantor_ctx * ctx);

// ---------------------------------------------------------------- callbacks

// Progress, called from inside a stage. stage is the running stage, i the
// units completed, n the total. On a phone a diffuse stage runs for minutes;
// a UI with no counter reads as a hang and users cancel work that was about
// to finish.
typedef void (*cantor_progress_fn)(cantor_stage stage, int i, int n, void * userdata);

// Return non-zero to stop. Polled between DiT steps, VAE tiles and LM tokens.
// A stage that stops this way returns CANTOR_PAUSED, not an error.
typedef int (*cantor_cancel_fn)(void * userdata);

// -------------------------------------------------------------- run a stage

// state_in / state_out are opaque bytes:
//   PLAN, CODES   the request JSON
//   DIFFUSE       in: request JSON. out: request + step + latent when paused,
//                 or the raw [T,64] f32 latent when done
//   DECODE        the raw [T,64] f32 latent; output is written to the sink
//
// On CANTOR_DONE the blob feeds the next stage. On CANTOR_PAUSED it resumes
// this same stage — pass it straight back as state_in.
//
// *state_out is allocated by the engine; release it with
// cantor_engine_free_blob. It is NULL when a stage produces no blob.
CANTOR_API cantor_status cantor_engine_run_stage(cantor_ctx *       ctx,
                                      cantor_stage       stage,
                                      const uint8_t *    state_in,
                                      size_t             in_len,
                                      uint8_t **         state_out,
                                      size_t *           out_len,
                                      cantor_progress_fn on_progress,
                                      cantor_cancel_fn   should_cancel,
                                      void *             userdata);

CANTOR_API void cantor_engine_free_blob(uint8_t * blob);

// DECODE writes audio here rather than into a state blob: it is the one
// stage whose output is not another stage's input.
// samples is planar stereo [L0..Ln, R0..Rn], n_samples per channel.
// Valid until the next run_stage call on the same context.
CANTOR_API const float * cantor_engine_audio(cantor_ctx * ctx, int * n_samples, int * sample_rate);

// Observability: bytes currently resident, and how many modules.
CANTOR_API uint64_t cantor_engine_resident_bytes(cantor_ctx * ctx);
CANTOR_API int      cantor_engine_resident_modules(cantor_ctx * ctx);

#ifdef __cplusplus
}
#endif

#endif  // CANTOR_ENGINE_H
