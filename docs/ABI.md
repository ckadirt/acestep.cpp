# The engine ABI

`include/cantor_engine.h` is the only surface a host depends on. Everything
else in this repository can change without the host caring.

The generation model is **a chain of stages, not one blocking call**. That is
what lets work stop between stages, resume inside one, and be scheduled by
which model it needs rather than by which song it belongs to.

---

## Contents

- [The shape of it](#the-shape-of-it)
- [Loading](#loading)
- [Running a stage](#running-a-stage)
- [State blobs](#state-blobs)
- [Pause and resume](#pause-and-resume)
- [Progress and cancellation](#progress-and-cancellation)
- [Errors](#errors)
- [Residency](#residency)
- [Threading](#threading)
- [Worked example](#worked-example)
- [Building against it](#building-against-it)

---

## The shape of it

```
     request JSON
          │
    ┌─────▼─────┐
    │   PLAN    │  LM    → metadata + lyrics, no audio codes
    └─────┬─────┘
    ┌─────▼─────┐
    │   CODES   │  LM    → audio codes
    └─────┬─────┘
    ┌─────▼─────┐
    │  DIFFUSE  │  DiT   → latents          (re-enterable)
    └─────┬─────┘
    ┌─────▼─────┐
    │  DECODE   │  VAE   → PCM 48 kHz stereo
    └───────────┘
```

Every stage takes an opaque byte blob and produces the next one. You do not
have to run all four: `CODES` alone is the normal path when you already have
lyrics, and `DECODE` alone turns a latent you saved last week back into audio.

Check the version first and refuse what you do not know:

```c
if (cantor_engine_abi_version() != CANTOR_ENGINE_ABI) { /* refuse */ }
```

`cantor_engine_stages()` returns a bitmask of what this build can actually
run, so a cut-down engine can advertise honestly rather than failing at call
time. Bit position is the stage value:

```c
if (cantor_engine_stages() & (1u << CANTOR_STAGE_DIFFUSE)) { /* available */ }
```

---

## Loading

Components are named by role. The roles are exactly the catalog's component
roles, so a variant record maps onto this without translation.

```c
cantor_component comps[] = {
    { "lm",    "/blobs/sha256/bdaf…" },
    { "dit",   "/blobs/sha256/a241…" },
    { "vae",   "/blobs/sha256/0599…" },
    { "embed", "/blobs/sha256/972f…" },
};

cantor_load_opts opts = {0};       /* zero-init, then set what you care about */
opts.vram_budget_bytes = 4ull << 30;
opts.vae_chunk         = 256;       /* mobile: lower peak, finer progress */

cantor_ctx * ctx = cantor_engine_load(comps, 4, &opts);
if (!ctx) {
    fprintf(stderr, "%s\n", cantor_engine_last_error());
}
```

**Which roles you need depends on which stages you want.** `PLAN` and `CODES`
need `lm`. `DIFFUSE` and `DECODE` need `dit`, `embed` and `vae` **together** —
the context reads DiT metadata at load and the VAE is what turns latents back
into audio. Supplying a partial synth set is rejected at load rather than
failing later at `run_stage`, because a problem reported at the wrong moment
is nearly as bad as one not reported at all.

`cantor_engine_free(ctx)` releases everything.

---

## Running a stage

```c
uint8_t * out     = NULL;
size_t    out_len = 0;

cantor_status st = cantor_engine_run_stage(
    ctx, CANTOR_STAGE_DIFFUSE,
    state_in, in_len,
    &out, &out_len,
    on_progress, should_cancel, userdata);
```

Three outcomes:

| Status | Meaning | What to do with `out` |
|---|---|---|
| `CANTOR_DONE` | stage finished | feed it to the **next** stage |
| `CANTOR_PAUSED` | stopped on your cancel callback | feed it back to **this same** stage to continue |
| `CANTOR_ERR` | see the error accessors | nothing; `out` is NULL |

`*state_out` is allocated by the engine. Release it with
`cantor_engine_free_blob()`, never `free()` — that is the same call today but
the guarantee is the function, not the implementation.

`DECODE` is the one stage whose output is not another stage's input, so it
writes audio rather than a blob:

```c
int n_samples, rate;
const float * pcm = cantor_engine_audio(ctx, &n_samples, &rate);
/* planar stereo: [L0..Ln, R0..Rn], n_samples per channel, rate is 48000 */
```

That buffer is valid until the next `run_stage` call on the same context.
Copy it if you need it longer.

---

## State blobs

**Blobs are opaque. Do not parse them.** Store the bytes, hand them back.

That is not politeness, it is the versioning strategy: the engine can change
what is inside a blob — add a field to the request, repack the latent — without
touching your code and without bumping `CANTOR_ENGINE_ABI`. Only the function
signatures are the contract. If you parse a blob you have opted out of that,
and a future engine release will break you.

What you can rely on:

| Stage | `state_in` | `state_out` |
|---|---|---|
| `PLAN` | request JSON | enriched request JSON |
| `CODES` | request JSON | request JSON with `audio_codes` |
| `DIFFUSE` | request JSON | latent (done) / resume blob (paused) |
| `DECODE` | latent | none; audio via `cantor_engine_audio` |

One exception worth knowing: a **completed** `DIFFUSE` blob is a raw
`[T, 64]` little-endian f32 latent with no header — `T = bytes / 256`. That is
the same byte layout `neural-codec` writes as `.vae` and the HTTP API
exchanges. So you can hand a `DIFFUSE` output to any of them, and hand any of
theirs to `DECODE`. It is a documented interchange format, unlike the paused
blobs, which are genuinely internal.

Sizes, measured on a 12-second track:

| Blob | Size |
|---|---|
| Completed latent | 76 800 B (≈ 1.1 MB for 3 min) |
| Paused `DIFFUSE` | 77 273 B |
| Paused `PLAN` / `CODES` | ~372 B |

---

## Pause and resume

Return `1` from your cancel callback and the stage stops cleanly. What comes
back is `CANTOR_PAUSED` with a blob that resumes **that same stage**:

```c
st = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, req, req_len,
                             &blob, &blob_len, prog, my_cancel, ud);

if (st == CANTOR_PAUSED) {
    write_to_disk(blob, blob_len);          /* atomically: temp file, rename */
    cantor_engine_free_blob(blob);
}

/* … later, possibly a different process … */
st = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, saved, saved_len,
                             &out, &out_len, prog, NULL, NULL);
```

You pass the paused blob back as `state_in` to the same stage. The engine
recognises it and continues; you do not need to tell it you are resuming.

**Write the blob atomically.** Temp file then rename. A process killed
mid-write otherwise leaves a truncated latent that resumes into noise.

### What it costs

The two stages resume for different reasons and at different prices.

**`DIFFUSE`** saves the diffusion latent and the step number. Everything else
— schedule, seeds, context, encoder states — is a deterministic function of
the request, so it is re-derived rather than stored. That re-derivation is one
text-encoder plus one cond-encoder pass; the DiT itself re-enters in about
1.6 ms. This is what keeps the blob at 77 KB instead of ~6 MB.

**`PLAN` / `CODES`** save the generated **token ids**. The obvious thing to
save is the KV cache and it is the wrong one: hundreds of megabytes at
`max_seq 8192`, more expensive to write to flash than to recompute. On resume
the prompt and the saved tokens are re-prefilled in one batched forward —
one compute-bound pass where generating them was that many sequential
memory-bound decode steps.

The general rule, if you are extending this: **autoregressive stages
checkpoint their output tokens; iterative-refinement stages checkpoint their
working tensor.**

### When resume is refused

Refused loudly, rather than approximated:

- **`solver: "dpm3m"` or `"stork4"`.** Multistep solvers keep a velocity
  history that is not in the blob. A resumed run would restart its bootstrap
  and follow a different trajectory. Use `euler` or `sde` for resumable jobs.
- **`guidance_scale > 1.0`.** CFG carries an APG running average across steps.
  All three shipped catalog tiers run `cfg: 1.0`, so this is inactive for them.
- **`lm_batch_size > 1`** for LM stages. Each sequence has its own token
  stream and FSM state; a partially-finished batch is a different problem.
- **A different solver than the blob was paused on.** Stamped and checked.

The reason these refuse instead of degrading gracefully: a drifted resume does
not sound broken. It quietly produces a *different song*. That is the kind of
wrongness a user has no way to detect, so failing is the honest answer.

- **A different backend than the blob was paused on.** Stamped and checked
  before any work is done:

  ```
  [ABI] cannot resume: blob was paused on backend 'CPU', this engine runs on
  'Vulkan0'. Resume on the device the job started on.
  ```

  Resume determinism is guaranteed *within* a backend — verified byte-exact on
  both CPU and Vulkan/RADV — but not across them, because re-deriving the
  conditioning through different kernels makes the trajectory drift.

---

## Progress and cancellation

```c
void on_progress(cantor_stage stage, int i, int n, void * ud) {
    /* i of n units done */
}

int should_cancel(void * ud) {
    return atomic_load(&job->cancelled);   /* non-zero stops the stage */
}
```

Both are optional; pass `NULL` to skip. Cancellation is polled between DiT
steps, VAE tiles and LM tokens — fine enough that a user-visible stop feels
immediate, and it is the only granularity available, since a ggml graph cannot
be aborted mid-flight.

Progress units are per stage: DiT steps for `DIFFUSE`, VAE tiles for `DECODE`,
tokens for `PLAN` and `CODES`.

> **A gap worth planning around.** When a track is shorter than one VAE chunk
> the decoder short-circuits to a single un-tiled decode, which cannot report
> from inside. You get `0/1` then `1/1` and nothing between — on a 12-second
> track that is 11 seconds of apparent silence. Setting `vae_chunk` to 256
> restores granularity, and is what you want on mobile anyway for the memory
> peak.

---

## Errors

```c
if (st == CANTOR_ERR) {
    switch (cantor_engine_last_error_code()) {
        case CANTOR_ERR_OOM:     /* retry smaller */          break;
        case CANTOR_ERR_MODEL:   /* re-pull the blob */       break;
        case CANTOR_ERR_BACKEND: /* fall back to another */   break;
        case CANTOR_ERR_CANCEL:  /* normal, not a failure */  break;
        case CANTOR_ERR_OTHER:   /* report it */              break;
    }
    fprintf(stderr, "%s\n", cantor_engine_last_error());
}
```

Five codes, because five is what a caller can act on differently. Anything
finer is in the message.

| Code | Cause | Response |
|---|---|---|
| `OOM` | device or host allocation failed | lower `vae_chunk`, shorter duration, lighter variant |
| `MODEL` | missing tensor, unreadable GGUF, config mismatch | re-pull the blob; its digest verified but the file is wrong for this build |
| `BACKEND` | requested device unavailable | fall back to the next backend |
| `CANCEL` | your callback returned non-zero | nothing; this is a normal outcome |
| `OTHER` | everything else | surface and report |

**No engine call kills the host process.** There is no `exit()` or `abort()`
reachable below this ABI — `tools/check-no-fatal.sh` enforces it in CI. An
out-of-memory allocating a VAE tile is an error return, not a process death.

The error record is thread-local and valid until the next engine call on the
same thread. Read it before doing anything else.

---

## Residency

Three policies, selected through `cantor_load_opts`:

| Setting | Behaviour | For |
|---|---|---|
| all zero | at most one model resident | lowest peak, most reloading |
| `vram_budget_bytes = N` | keep what fits under N, evict least-recently-used | the middle, and what the catalog's `vram_bytes` was measured for |
| `keep_loaded = 1` | never evict | workstations; overrides the budget |

Budget caveat, stated plainly: **peak can exceed the budget by roughly one
module.** A model's size is not known until it is loaded, so eviction settles
after an install rather than predicting one. Size the budget with that
headroom, or treat it as a steady-state target rather than a cap.

Observability:

```c
uint64_t bytes   = cantor_engine_resident_bytes(ctx);
int      modules = cantor_engine_resident_modules(ctx);
```

### Scheduling for residency

The strongest reason for stages, and the one easiest to miss: because stages
are independently invocable and their state persists, **you can sort a queue
by which model each piece of work needs**. Five queued songs run stage-major
become one LM load, one DiT load and one VAE load — instead of five of each.
On a phone that is the difference between a feature and a demo. A monolithic
`generate()` cannot do this at all.

---

## Threading

The engine is **not** internally synchronised. One context, one thread at a
time. Two threads in `run_stage` on the same context will corrupt the model
store.

Separate contexts on separate threads are also not supported today: the ggml
backend is process-global and reference-counted across all of them.

If you need concurrency, do what `ace-server` does — a single worker thread
consuming a job queue, with cancellation flags set from the request-handling
threads. That is what the cancel callback is designed for.

---

## Worked example

Caption in, audio out, with a pause in the middle:

```c
#include "cantor_engine.h"

static int cancelled(void * ud) { return *(int *) ud; }

int generate(cantor_ctx * ctx, const char * caption_json, int * stop_flag) {
    uint8_t *e = NULL, *codes = NULL, *latent = NULL;
    size_t   el = 0, cl = 0, ll = 0;

    /* 1. lyrics and metadata */
    if (cantor_engine_run_stage(ctx, CANTOR_STAGE_PLAN,
                                (const uint8_t *) caption_json, strlen(caption_json),
                                &e, &el, NULL, cancelled, stop_flag) != CANTOR_DONE) {
        return -1;
    }

    /* 2. audio codes */
    if (cantor_engine_run_stage(ctx, CANTOR_STAGE_CODES, e, el,
                                &codes, &cl, NULL, cancelled, stop_flag) != CANTOR_DONE) {
        cantor_engine_free_blob(e);
        return -1;
    }
    cantor_engine_free_blob(e);

    /* 3. diffusion, resuming for as long as it keeps getting interrupted */
    uint8_t * in  = codes;
    size_t    inl = cl;
    for (;;) {
        cantor_status st = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, in, inl,
                                                   &latent, &ll, NULL, cancelled, stop_flag);
        if (in != codes) {
            cantor_engine_free_blob(in);
        }
        if (st == CANTOR_DONE)  { break; }
        if (st == CANTOR_ERR)   { cantor_engine_free_blob(codes); return -1; }

        /* paused: persist and wait for the user to come back */
        save_to_disk(latent, ll);
        wait_until_resumed(stop_flag);
        in = latent; inl = ll; latent = NULL;
    }
    cantor_engine_free_blob(codes);

    /* 4. audio */
    uint8_t * unused = NULL; size_t ul = 0;
    if (cantor_engine_run_stage(ctx, CANTOR_STAGE_DECODE, latent, ll,
                                &unused, &ul, NULL, NULL, NULL) != CANTOR_DONE) {
        cantor_engine_free_blob(latent);
        return -1;
    }
    cantor_engine_free_blob(latent);

    int n, rate;
    const float * pcm = cantor_engine_audio(ctx, &n, &rate);
    write_wav(pcm, n, rate);
    return 0;
}
```

`tests/test-abi.c` is a fuller worked example and is kept passing.

---

## Building against it

The ABI lives in `acestep-core`, which is a static library today:

```cmake
target_include_directories(your_target PRIVATE path/to/acestep.cpp/include)
target_link_libraries(your_target PRIVATE acestep-core)
```

`cantor_engine.h` is C-compatible and `extern "C"`-guarded. The test driver is
deliberately written in C rather than C++, because a C++ test would not catch
a header that only compiles as C++. If you link from C, force C++ linkage on
the final binary:

```cmake
set_target_properties(your_target PROPERTIES LINKER_LANGUAGE CXX)
```

Run the ABI test:

```bash
./build/test-abi --models /path/to/models
```

Shared-library builds and the `libcantor_acestep_<backend>_<arch>.so` naming
from the engine plan are not wired up yet; `acestep-core` is static. That is
Part 3 of the models-and-engine plan, not this ABI.
