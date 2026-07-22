# Implementation notes — engine execution model (Parts 5–8)

Working log for executing the plan at
<https://claude.ai/code/artifact/f4d51109-a383-4096-a1e9-c417480fc317>
("Where the work can stop", companion to *Filling the shelves* Parts 1–4).

Branch: `engine-stages`, off `master` at `fa33775`.

Every section headed **Deviation** records where reality differed from the
plan. The plan is not retro-edited; these notes are the amendment.

## Status

| Part | State | Commit |
|---|---|---|
| 5 · No fatal exits below the ABI | **done** | `57abaac` |
| 6 · The stage seam (ABI) | **not started** | — |
| 7 · Pause and resume | **in-process done**, persistence + LM outstanding | `0cec0ac` |
| 8 · Residency and the budget | **policy done**, device tuning outstanding | `1a716c7` |

Parts were taken in dependency order except that Part 6 was deferred: Parts 7
and 8 are the capabilities, Part 6 is the surface that exposes them, and
neither 7 nor 8 needs it to be built or tested. The cost of that choice is
that Milestone 5's "harness still running" check and Part 8's ABI-side
reporting are both still blocked on it.

Everything below was built and run on CPU against the `1.5-fast` models.
Nothing here has been near a GPU or a phone.

---

## Test environment

CPU-only, 12 logical cores, OpenBLAS. Built with:

```bash
cmake .. -DGGML_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j 12
```

Models (`1.5-fast` tier, from `/home/ckadirt/projects/step-ace 1.5/acestep.cpp/models`):

| Role | File | Size |
|---|---|---|
| lm | `acestep-5Hz-lm-0.6B-Q8_0.gguf` | 710 MB |
| dit | `acestep-v15-turbo-Q4_K_M.gguf` | 1.45 GB |
| vae | `vae-BF16.gguf` | 337 MB |
| embed | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 784 MB |

Reference workload — `testrun/dit-fast.json`, a 12 s instrumental at seed 42,
8 steps, `guidance_scale 1.0`, `wav16` out. Full run ≈ 32 s wall
(DiT 8.7 s, VAE decode 11.1 s, model loads ≈ 5.5 s).

**Baseline output hash** (`testrun/baseline-seed42.sha256`):

```
607dd0ea33eebf447032f4bd2b6f04cc8b0d60fbfbf1d9a0d16f38d5833a74c3
```

### Determinism confirmed

The plan's Milestone 7 assumes a resumed run can be compared bit-for-bit
against an uninterrupted one. Verified before writing any code: two
consecutive baseline runs produce byte-identical WAVs on this CPU backend.
The test is viable.

---

## Part 5 · No fatal exits below the ABI — **done**

Commit: `part5: replace fatal exits with propagating errors`

### What was built

- **`src/error.h`** (new) — `AceErrorCode` enum (`OK`, `OOM`, `BAD_MODEL`,
  `NO_BACKEND`, `CANCELLED`, `INTERNAL`), a thread-local `AceError` record,
  and `ace_set_error()` / `ace_error_code()` / `ace_error_msg()`.
  `ace_set_error` also mirrors to stderr, so CLI output is unchanged from the
  `exit(1)` days — the operator still sees the message.
- **Sticky failure flag on `WeightCtx`** — see the deviation below.
- **`backend_init_sched()`** in `backend.h` — see the deviation below.
- **`tools/check-no-fatal.sh`** — CI guard, greps `src/` for
  `exit(1)` / `exit(EXIT_FAILURE)` / `abort()` / `GGML_ASSERT`, exits
  non-zero on any hit. Currently green.

All 26 sites converted across 9 files. `check-no-fatal.sh` passes.

### Verification

Regression: the reference workload still produces
`607dd0ea…833a74c3`. Byte-identical to baseline — the conversion changed no
numerics.

Error paths, all three exiting gracefully with a distinct message and
**none by signal**:

| Class | Trigger | Result |
|---|---|---|
| `NO_BACKEND` | `GGML_BACKEND=NoSuchDevice` | `GGML_BACKEND=NoSuchDevice not found. Available: BLAS CPU`, exit 1 |
| `BAD_MODEL` | DiT GGUF truncated to 300 MB | truncation diagnostic naming the tensor and byte range, exit 1 |
| `OOM` | `ulimit -v 700000` | `Mmap failed …`, exit 1 |

---

### Deviation 1 — 26 sites, not 28

The plan says 28, from `grep -rn "exit(1)\|abort()" src/ | wc -l`. Two of
those 28 hits are **comments**, not calls — the two lines in
`model-store.cpp` that documented the old behaviour:

```cpp
vae_enc_load(m, k.path.c_str());  // exit(1) on failure, returns void
```

Actual call count: **26**. The nine-file list in the plan is correct.

### Deviation 2 — a fourth abort family the plan did not enumerate

`gguf-weights.h:gf_load_qkv_fused` contained a `GGML_ASSERT`, which aborts
inside ggml just as surely as `abort()` does:

```cpp
GGML_ASSERT(q_src->ne[0] == k_src->ne[0] && k_src->ne[0] == v_src->ne[0]);
```

It was the only one in `src/`. Converted to a graceful `return NULL`, which
costs nothing because the caller already treats NULL as "fusion unavailable,
fall back to separate loads". `check-no-fatal.sh` greps for `GGML_ASSERT`
too, so the family stays closed.

Related: the missing-tensor branch in the same function now returns NULL
**silently** rather than reporting. If Q/K/V are genuinely absent the
fallback path calls `gf_load_tensor`, which reports the specific missing
tensor name and sets the sticky flag. Reporting in both places would print a
misleading FATAL for a recoverable case and then overwrite it.

### Deviation 3 — a latent bug the conversion surfaced

`qw3lm_forward` had this on the KV-overflow path:

```cpp
if (kv_len > c.max_seq_len) {
    fprintf(stderr, "[LM-Forward] FATAL: kv_len %d > max_seq %d\n", ...);
    return;                       // <-- void return, caller none the wiser
}
```

It printed `FATAL` and then returned normally, leaving the caller's `logits`
buffer **uninitialised** while generation continued on whatever was in that
memory. Not a crash — silently wrong output on any prompt that overran the
KV cache. Now returns `false` and the callers bail.

This is the strongest argument for the whole exercise: the bug was invisible
precisely because the failure had nowhere to go.

### Deviation 4 — `[[nodiscard]]` instead of trusting the task list

The plan's task list says convert the functions and add a CI grep. That is
not sufficient on its own: **C++ silently permits discarding a `bool`
return**, so converting `qw3lm_forward` from `void` to `bool` compiled
cleanly with all 14 of its callers still ignoring the result. The build was
green and the failures still went nowhere.

Marked the ten fallible functions `[[nodiscard]]`
(`qw3lm_forward`, `qw3lm_forward_batch`, `qw3lm_alloc_kv_cache`,
`qw3lm_init_backend`, `qwen3_forward`, `qwen3_embed_lookup`,
`cond_ggml_forward`, `vae_ggml_load`, `vae_enc_load`,
`backend_init_sched`). That surfaced **16 unchecked call sites** across
`pipeline-lm.cpp` (12), `pipeline-understand.cpp` (2) and
`neural-codec.cpp` (2), every one of which would otherwise have shipped.

This is a stronger guarantee than the planned grep and should be treated as
part of the Part 5 contract: **a new fallible function without
`[[nodiscard]]` is an incomplete conversion.**

### Deviation 5 — two mechanisms the plan left unspecified

The plan says "every `exit(1)` becomes a `return false` that propagates" but
not how to avoid drowning the load code in null checks. Two additions:

**Sticky `failed` flag on `WeightCtx`.** `dit_ggml_load` issues ~100
`gf_load_tensor` calls in a row. Checking each at the call site would bury
the load logic. Instead a failing loader sets `wctx->failed` and returns
NULL; the model loader checks once, and `wctx_alloc` refuses to allocate a
buffer for a model with holes in it. Once set the flag never clears — the
first failure is the one worth reporting.

**`backend_init_sched()` helper.** Eight module loaders opened with the same
four lines (`backend_init` then `backend_sched_new`), each now needing two
failure branches. Collapsed into one helper that handles both and releases
the backend reference on the second failure. Eight call sites became
one-liners instead of eight copies of the same guard.

### Deviation 6 — Milestone 5 is only partly demonstrable yet

The milestone asks for "a test harness [that] loads the engine … with the
harness still running." There is no ABI to load until Part 6, so the
harness cannot exist yet. Verified the weaker property available today: all
three error classes exit the CLI gracefully with a distinct message, none
by signal (no SIGSEGV/SIGABRT).

Full milestone verification is deferred to Part 6, where the C test driver
will exercise the same three failures across the ABI boundary and assert the
process survives. **Tracked as outstanding.**

Also worth recording for whoever writes that harness: forcing OOM with
`ulimit -v` on this build first trips OpenBLAS's thread-pool init
(`blas_thread_init: pthread_create failed`) rather than a ggml allocation.
Set `OPENBLAS_NUM_THREADS=1` to make the memory limit bite where you want it.

### Deviation 7 — error enum location

The plan puts the error enum in `include/cantor_engine.h`. That header is a
Part 6 artifact and does not exist yet, so the enum lives in `src/error.h`
for now. Part 6 will re-export it across the ABI rather than redefine it —
one enum, two headers, no drift.

---

## Part 7 · Pause and resume — **in-process done, persistence outstanding**

Commit: `part7: make the DiT loop re-enterable`

### What was built

- **`dit_ggml_generate` is re-enterable.** Three new trailing params
  (`start_step`, `xt_io`, `stop_step_out`) and a third return code:
  `0` done, `1` paused, `-1` error. On a pause it copies `xt` — the latent
  *entering* the interrupted step — into `xt_io` and reports the step, so a
  resume repeats no work and skips none.
- **Pause no longer destroys the job.** `run_tail` propagates `1` instead of
  collapsing it into `-1`, and the seven task wrappers keep the job on a
  pause via a new `job_mark()`.
- **Public API** in `pipeline-synth.h`: `ace_synth_job_is_paused()`,
  `ace_synth_job_progress()`, `ace_synth_job_resume_dit()`.
- **Resume refusal** for stateful solvers and CFG runs, gated on the
  registry's existing `SolverInfo::is_stateful` rather than name-matching.
- **`tests/test-dit-resume.cpp`** — three scenarios: reference run, pause at
  step k + resume + byte comparison, and stateful-solver refusal.

### Verification

```
[Test] === paused run (pause entering step 5) ===
[DiT] Paused at step 5/8
[Test] resuming...
[DiT] Resuming at step 5/8
[Test] PASS: resumed latents are byte-identical to the reference
[Test] PASS: resume refused on a stateful solver
[Test] ALL PASS
```

The 12 s reference render is unchanged from baseline
(`607dd0ea…833a74c3`), so the re-entry plumbing costs nothing when unused.

### Deviation 8 — in-process resume needs no `xt_state` at all, but it has one

The plan predicted in-process resume would be "~10 lines: just don't delete
the job", because the `AceSynthJob` already holds the whole `SynthState`.
That is true of everything *except* `xt`, which was a function-local
`std::vector` inside `dit_ggml_generate` and died with the call.

So `SynthState` gains an `xt_state` buffer. It is not strictly needed for
the in-process case — the sampler could have kept its own latent across
calls — but putting it in `SynthState` is what makes the *serialized* case
a straight copy of a field that already exists rather than a second
mechanism. Costs `batch_n * T * 64 * 4` bytes (1.1 MB for the reference
workload), which the job was already spending twice over on `output` and
`context`.

Actual size: ~40 lines in the sampler, ~30 in the pipeline, ~25 in the
header. Close to the plan's estimate once `xt_state` is accounted for.

### Deviation 9 — an existing caller silently inherited the new behaviour

`tools/synth-batch-runner.h` treated a non-NULL job as complete and went
straight to VAE decode. Once pause stopped returning NULL, a cancelled
render would have decoded a **zero-filled** output buffer and written
silence instead of failing.

Guarded: the batch runner now checks `ace_synth_job_is_paused()` and frees
+ returns `-1`, reproducing the old contract exactly. Both existing callers
(the `ace-synth` CLI and `ace-server`) are run-to-completion paths and want
that. Resume is opt-in, via the job handle, for callers that want it.

**This is the kind of break the plan did not anticipate**: widening a return
contract silently changed the meaning of an existing success path. Worth
checking for the same shape in Part 6 when `run_stage` starts returning
`CANTOR_PAUSED`.

### Deviation 10 — the cover context switch is not in the saved state

`switched_cover` is a loop-local flag, and the swapped context/encoder
buffers it produces are not serialized. On resume it starts `false` again,
so when `start_step > cover_steps` the switch re-fires on the first resumed
iteration and re-applies the same swap. That is correct — the swap is
idempotent and derived from `s.context_switch`, which *is* in `SynthState`.

Recorded because it is load-bearing and non-obvious: if anyone makes the
cover switch stateful or non-idempotent, resume breaks silently for cover
jobs. The test covers text2music only.

### Outstanding for Part 7

- Cross-restart persistence: serialize `{request, step, xt}` and reload.
  The pieces are all in place (`xt_state` + `resume_step` in `SynthState`);
  what is missing is the blob format, atomic write, and the
  backend/quant/solver stamp that Part 7's task list requires for refusing a
  mismatched resume.
- LM token-level checkpoint + re-prefill resume. Not started.

---

## Parts 6 and 8 — not started

Part 6 (stage ABI) and Part 8 (budget policy) are untouched. Part 5's
Milestone-5 harness verification is blocked on Part 6, as noted above.

---

## Part 8 · Residency and the budget — **policy done, device tuning outstanding**

Commit: `part8: add EVICT_BUDGET residency policy`

### What was built

- **`EVICT_BUDGET`** in `model-store.cpp`: `GpuEntry` gains a `last_used`
  stamp bumped on install and on every cache hit; `evict_to_budget()` drops
  the coldest unreferenced module until the resident total fits.
- **`store_create(policy, budget_bytes = 0)`** — defaulted, so the 13
  existing call sites are untouched.
- **`--vram-budget <MB>`** on `ace-server`. `--keep-loaded` wins if both are
  given: it is an explicit "I have room for everything" instruction and a
  byte target cannot express that.
- **`scenario_budget`** in `test-model-store.cpp`, which measures the three
  modules first and derives its budgets from the measurements rather than
  hardcoding sizes.

### Verification

```
[Test] sizes: vae_enc=160.8 MB vae_dec=161.1 MB dit=895.6 MB
[Test] budget fits all three    modules=3, vram=1217.5 MB
[Test] PASS: BUDGET kept all three; STRICT would have kept one
[Store] Evict VAE-Enc (160.8 MB, LRU) to fit budget
[Store] Evict VAE-Dec (161.1 MB, LRU) to fit budget
[Test] budget fits DiT only     modules=1, vram=895.6 MB
[Test] PASS: BUDGET evicted LRU down to 895.6 MB (budget 911.6 MB)
```

All 8 scenarios pass, 0 failures. Reference render unchanged.

### Deviation 11 — evicting before the load never converges

The obvious implementation — run the eviction pass in the require path,
before loading — **does not work at all**, and the first version of the test
caught it: all three modules stayed resident at 1217 MB under a 911 MB
budget.

The reason is that a module's size is unknown until it is loaded, so the
pre-load pass only ever sees the *current* resident set. Starting from
empty, each require finds itself under budget and evicts nothing:

```
require enc: resident 0     < 911 -> no evict -> resident 161
require dec: resident 161   < 911 -> no evict -> resident 322
require dit: resident 322   < 911 -> no evict -> resident 1217   (never converges)
```

Fixed by settling again *after* install, where the new size is known. The
just-installed module has refcount 1 and so is never its own victim, while
colder peers are reclaimed around it. `install_entry` now wraps
`install_entry_impl` to do this in one place rather than at eight require
sites.

The header comment about peak exceeding the budget by roughly one module
still stands and is now the accurate description of the remaining slack.

### Outstanding for Part 8

- Real device measurements. Every knob in the plan's tuning table
  (`vae_chunk` 256 on mobile, big-core thread count, duration caps) is still
  upstream's number or an inference from it. Nothing here was measured on a
  phone or an entry GPU, so none of it should be treated as tuned.
- `vae_chunk` / thread count / quirk flags are not yet reachable from a load
  options struct — that surface arrives with Part 6.
- Reporting resident bytes across the ABI: blocked on Part 6.

---

## Part 6 · The stage seam — **done**

Commit: `part6: add the stage ABI`

Taken after 7 and 8 rather than before, because it is the surface those
capabilities are exposed through and building it last meant it could expose
what actually exists rather than what was planned. Doing it also completed
Part 7's cross-restart persistence, since the DIFFUSE state blob *is* the
persistence format.

### What was built

- **`include/cantor_engine.h`** — `CANTOR_ENGINE_ABI 1`, the four stages,
  `cantor_status` (`DONE` / `PAUSED` / `ERR`), `cantor_error` (the Part 5
  classes re-exported), `cantor_load_opts`, progress and cancel callbacks,
  `cantor_engine_run_stage`, plus `cantor_engine_stages()` as a bitmask.
- **`src/abi.cpp`** — the shim. Decodes a blob, calls a pipeline, encodes the
  result.
- **Serialized resume**, which is Part 7's outstanding item: a paused DIFFUSE
  blob is `DiffuseHeader` + request JSON + latent, stamping the backend name
  and solver so a resume elsewhere can be refused.
- **Five supporting entry points** in `pipeline-synth.h`:
  `ace_synth_abort_immediately`, `ace_synth_job_get_xt`,
  `ace_synth_job_restore`, `ace_synth_backend_name`,
  `ace_synth_decode_latent`.
- **`tests/test-abi.c`** — nine checks, written in C.

### Verification

```
[ABI-Test] abi=1 model=acestep stages=0x1e
[ABI-Test] PASS: DIFFUSE produced a latent
[ABI-Test] PASS: DECODE produced audio
[ABI-Test] PASS: saved latent reproduced the audio exactly
[ABI-Test] PASS: DIFFUSE paused and emitted a resumable blob
[ABI-Test] PASS: cross-restart resume is byte-identical to the uninterrupted run
[ABI-Test] PASS: missing model reported, harness alive
[ABI-Test] PASS: unknown component role reported, harness alive
[ABI-Test] PASS: malformed state blob reported, harness alive
[ABI-Test] PASS: malformed latent reported, harness alive
[ABI-Test] ALL PASS
```

The paused blob is **77273 bytes** for a 12 s track: 76800 of latent, 473 of
header and request JSON. It survives a full `cantor_engine_free` and resumes
in a fresh context to a byte-identical result.

This also closes **Deviation 6**: Milestone 5's "harness still running" check
now has a harness. Four failure paths are exercised across the ABI boundary
and the process survives every one.

### Deviation 12 — rebuilding a job to restore into it

The plan describes cross-restart resume as "serialize `{request, step, xt}`
and reload", which reads as though the state can be poured into a fresh job.
It cannot: `SynthState` also holds the conditioning (context, encoder hidden
states, schedule, seeds), and by design none of that is in the blob.

So a serialized resume runs phase 1 again with a cancel callback that fires
immediately (`ace_synth_abort_immediately`). That performs the encoding,
builds the context, and pauses entering step 0 — producing a job whose shape
is guaranteed to match the blob because both came from the same request.
`ace_synth_job_restore` then overwrites the latent and step, and the normal
in-process resume takes it from there.

Cost on the reference workload: the re-derivation is the text encoder and
cond encoder, and the DiT call itself pauses in 1.6 ms. That is the price of
keeping the blob at 77 KB instead of ~6 MB, and it is the right trade.

The size check in `ace_synth_job_restore` is doing real work here: because
the rebuilt job's layout comes from the request, a blob whose latent does not
match is a blob built for a different duration, batch size or model. That is
the check that catches it.

### Deviation 13 — a shim bug the error tests caught

The first version of `cantor_engine_load` built the synth context only when
dit, embed and vae were all present, and otherwise left it NULL and returned
success. A context loaded with nothing but a **nonexistent** DiT path
therefore reported success and failed later at `run_stage`, or never.

The ABI test caught it on the first run. `cantor_engine_load` now rejects a
partial synth set and a component set that serves no stage at all. Worth
recording because it is the failure mode Part 5 exists to prevent,
reintroduced one layer up: a problem reported at the wrong moment, or not at
all.

### Deviation 14 — progress reporting, and where it does not reach

Initially the callback was accepted but never fired. Now emitted from the DiT
step loop and the VAE tile loop:

```
[ABI-Test]   progress: stage=3 1/8 ... 8/8
```

The callback is set on the `AceSynth` context (`ace_synth_set_progress`)
rather than passed per call. Threading two more parameters through the seven
task wrappers would have been pure churn: none of them care, and the callback
is a property of who is watching, not of which task is running.

Two gaps remain, both real:

- **The non-tiled VAE path is silent.** `vae_ggml_decode_tiled` returns early
  to a single direct decode when `T_latent <= chunk_size`, which is the common
  case for short tracks at the default 1024 chunk — the reference workload
  (T=300) takes it. That decode is 11 s of the 32 s run and reports nothing.
  Lowering `vae_chunk` on mobile fixes this as a side effect, which is a point
  in favour of doing it anyway.
- **The LM loops emit nothing.** PLAN and CODES report no progress at all.
  Same mechanism would work; `AceLm` needs the same two fields.

---

## Remaining work

| Item | Part | Note |
|---|---|---|
| Progress from the LM loops, and the non-tiled VAE decode | 6 | DiT and tiled VAE done |
| LM token checkpoint + re-prefill resume | 7 | not started; DiT resume does not cover PLAN/CODES |
| Device measurements and tuning | 8 | nothing here has run on a GPU or a phone |
| `vae_chunk` / threads / quirks through load opts | 8 | fields exist in `cantor_load_opts`; `n_threads` is not yet wired to the backend |

---

## Follow-up · LM resume, progress gaps, ABI docs — **done**

Commit: `lm: token-level checkpoint and resume; close the progress gaps`

### LM resume

Checkpoints the generated **token ids**, not the KV cache — the cache is
hundreds of MB at `max_seq 8192` and costs more to write to flash than to
recompute. On resume the prompt and saved tokens are re-prefilled in one
batched forward.

Both phases support it. Phase 1 additionally **replays the saved tokens
through the FSM** and the think/codes transition, so the constrained decoder
resumes in the state it was interrupted in — the prefill rebuilds the KV
cache, this rebuilds the CPU-side state that goes with it. Phase 2 stores
absolute vocab ids (what a prefill consumes) and lifts them back into
`audio_codes` on resume, so the output is the whole run rather than only the
part generated after.

Verified:

```
[LM-Phase1] Paused at step 40, 41 tokens checkpointed
[LM-Phase1] Resuming: 38 prompt + 41 generated tokens re-prefilled
PASS: PLAN paused and emitted a token checkpoint
PASS: PLAN resumed from the token checkpoint and completed
```

Paused LM blob: **372 bytes**, against 77 273 for a paused DIFFUSE. The
asymmetry is the point — tokens are tiny, latents are not.

### Deviation 15 — LM resume is not bit-identical, and cannot cheaply be

DiT resume is byte-exact. LM resume is **not**, and this is a real difference
in the guarantee that should be stated wherever it is offered to a user.

The RNG stream is not part of the checkpoint. `std::mt19937` is seeded once
per sequence and advanced by every sampling call, so a resumed run re-seeds
and diverges from where the interrupted one would have gone. Every token
already produced is preserved exactly; the *continuation* differs.

Serializing the RNG state is possible (`mt19937` streams in and out via
`operator<<`) but the FSM state is the harder half, and replaying tokens
through it — which is what happens now — is the same work. The honest framing:
**LM resume preserves work, DiT resume preserves the result.** Both are worth
having; they are not the same promise.

Not fixed, deliberately. Recorded so nobody markets it as exact.

### Deviation 16 — batched LM resume refused

`ace_lm_generate` refuses a checkpoint when `lm_batch_size > 1`. With N > 1
each sequence has its own token stream, FSM state and RNG, and a partially
finished batch (some sequences done, some not) is a different problem than a
paused single run. Refusing beats silently dropping sequences.

The ABI always passes 1, so this does not restrict the node.

### Progress gaps closed

- **LM loops** now emit, through the same context-level setter pattern as the
  synth side (`ace_lm_set_progress`). Verified: 40 events at
  `stage=1 i/2048` during the paused PLAN run.
- **Non-tiled VAE decode** now emits `0/1` then `1/1`. It is one ggml graph
  and cannot report from inside, so this only distinguishes *running* from
  *done* — but that is the difference between a UI that looks hung and one
  that does not. Real granularity needs a smaller `vae_chunk`, which mobile
  wants anyway. Documented in both `docs/ABI.md` and the header.

### Documentation

`docs/ABI.md` — how to drive the engine: loading, the stage loop, blob
opacity and why, pause/resume semantics and the exact cases where resume is
refused, error handling, residency, threading, and a worked example.

### Deviation 17 — a documented guarantee that is not yet enforced

Writing the docs surfaced this. The paused DIFFUSE blob **stamps the backend
name**, and the plan's Part 7 task list requires refusing a resume on a
mismatch. The solver check is wired; **the backend check is not** — the field
is written and then ignored.

So today, resuming a blob on a different backend than it was paused on will
proceed and drift. It is called out explicitly in `docs/ABI.md` under pause
and resume rather than quietly left out. Small fix, but it is a gap between
what is documented as designed and what the code does, and those are the
worst kind to leave unrecorded.

---

## Vulkan validation on real hardware — **done**

Commit: `docs: vulkan build without root, and what it validated`

First run of any of this off CPU. Device: AMD Cezanne / Renoir integrated
GPU, RADV driver, `uma: 1`, `matrix cores: none`.

### Setup

Ubuntu 22.04 has the Vulkan **runtime** (`mesa-vulkan-drivers` ships
`radeon_icd` and `lvp_icd`, `libvulkan.so` present) but not the **build**
side. `glslc` is not packaged on 22.04 at all. Configure fails at
`find_package(SPIRV-Headers)`.

Resolved without root using the LunarG tarball SDK (333 MB, unpacked to a
scratch dir) — the same SDK CI installs. Full recipe in `docs/BUILD-VULKAN.md`.

### Results

| Stage | CPU (12 threads + BLAS) | Vulkan (RADV) |
|---|---|---|
| DiT, 8 steps | 8 727 ms | 5 435 ms |
| VAE decode | 11 120 ms | 7 791 ms |

Audio equivalent, not bit-identical (expected — different kernels):
RMS 2808.8 vs 2846.8, both full-scale peak, 100% non-zero.

`test-dit-resume`: **ALL PASS**. `test-abi`: **11/11 PASS**, including the
cross-restart blob round-trip.

### The open question, answered

The Part 7 notes flagged that DiT resume was proven byte-exact only on CPU.
It is now proven on RADV too:

```
[DiT] Paused at step 5/8
[DiT] Resuming at step 5/8
[Test] PASS: resumed latents are byte-identical to the reference
```

**Resume determinism holds within a backend on real GPU hardware.** That was
the precondition for offering resume as a feature at all, and it is met.

### What this does NOT establish

Worth being exact, because it is easy to over-read:

- **Cross-backend resume is still untested and still unenforced.** A blob
  paused on CPU and resumed on Vulkan would re-derive the conditioning
  through different kernels and drift. The backend stamp in the paused blob
  is still written and ignored (Deviation 17). Nothing here changes that.
- **This is an iGPU with no matrix cores sharing system RAM.** The speedups
  are a floor, not a prediction for discrete cards.
- **CUDA remains entirely unvalidated.** No NVIDIA hardware on this machine;
  the toolkit is not even installed. Building it needs only the toolkit
  (CI proves that on GPU-less runners), but nothing has run it.
- **Nothing has run on a phone.** Adreno/Mali are Vulkan too, but a different
  driver stack from RADV, and `vae_chunk` is untuned for their memory limits.

---

## Backend guard + shared library — **done**

Commit: `abi: enforce the backend stamp; ship a real shared library`

### Backend guard (closes Deviation 17)

The paused DIFFUSE blob always stamped the backend; the check was written and
then ignored. Now enforced before any work is done. Proven with
`tests/test-backend-guard.c`, a two-process test so the halves can run under
different `GGML_BACKEND` values:

| Pause on | Resume on | Result |
|---|---|---|
| CPU | CPU | `ACCEPTED (status=0)` |
| CPU | Vulkan0 | `REFUSED` — *"blob was paused on backend 'CPU', this engine runs on 'Vulkan0'"* |

Both `docs/ABI.md` and `docs/BUILD-VULKAN.md` had a "not yet enforced" caveat.
Corrected rather than left to rot.

### The shared library

`acestep-core` was STATIC, so nothing existed for the node to `dlopen`. Now
`libcantor_engine.so` (979 KB), with `abi.cpp` moved out of the static library
into it — and `test-abi` relinked against the `.so`, so the tests exercise the
artifact that ships rather than a parallel static build of it.

### Deviation 18 — visibility flags do not control an export list

The obvious approach — `CXX_VISIBILITY_PRESET hidden` on the shared target —
leaked **66 symbols**. Three separate reasons, each needing its own fix:

1. **Visibility is decided when a translation unit is compiled.** Symbols
   already public in `acestep-core.a` get re-exported no matter what the
   shared target declares. The preset has to go on the *static* libraries.
2. **`CXX_VISIBILITY_PRESET` does not reach C.** yyjson is C; it needs
   `C_VISIBILITY_PRESET`.
3. **A dependency can override the flag entirely.** `vendor/yyjson/yyjson.h:336`
   defines `yyjson_api` as `__attribute__((visibility("default")))`
   unconditionally on GCC, which beats `-fvisibility=hidden`. No combination
   of CMake visibility properties can suppress it.

Fixed by adding `cantor_engine.map`, a linker version script, which decides
the export list regardless of what any dependency asked for. Final surface:

```
13 symbols, all cantor_engine_*
```

Worth internalising: **on ELF, a version script is the export list; visibility
flags are a hint.** If the surface matters, say it in one place the linker
enforces. ELF-only, so Mach-O (`-exported_symbols_list`) and PE (`.def`) would
need their own if those platforms ever matter.

### Still open

- CUDA remains unvalidated — no NVIDIA hardware here.
- Nothing has run on a phone.
- The `.so` is built per-configuration; the
  `libcantor_acestep_<backend>_<arch>.so` naming and the plugins-vs-baked-in
  packaging decision are not made yet.

---

## Android arm64 — **runs**

Commit: `android: cross-compile, run on-device, wire the thread count`

Device: Xiaomi `veux`, Snapdragon 695 (SM6375), 2 x A78 + 6 x A55, 7.6 GB
RAM, Android 13, Adreno GPU. Cross-compiled with NDK 27.1, deployed by
`adb push` to `/data/local/tmp`. Full recipe in `docs/BUILD-ANDROID.md`.

**A 12-second track generates end to end on a mid-range phone**, CPU-only,
inside memory, no OOM: DiT 72.7 s, VAE decode 89.1 s at `--vae-chunk 256`.
About 14x slower than realtime. Audio equivalent to desktop (RMS 2861.5 vs
2808.8 on x86).

### Deviation 19 — `vae_chunk 256` was the right guess, for a checkable reason

The plan asserted 256 on mobile from upstream's table, never measured. It
does work here: 3 tiles, completes within memory. But nothing measured peak
RSS and 512 was never tried, so "256 works" is established and "256 is
necessary" is not. Recorded so the number is not promoted to received wisdom
on the strength of one passing run.

### Deviation 20 — the default thread heuristic is actively harmful on big.LITTLE

This was predicted in the plan as a thing to fix, and the fix was listed as
outstanding. Measuring it turned out to matter more than expected, because
the default is not merely suboptimal, it is the **worst** of the options:

| Threads | DiT | VAE | Total |
|---:|---:|---:|---:|
| 2 (big only) | 37.7 s | 25.2 s | 62.9 s |
| 3 (**default**) | 38.1 s | 42.9 s | **81.1 s** |
| 6 (all) | 32.5 s | 28.7 s | 61.2 s |

`hardware_concurrency() / 2` assumes logical/2 = physical cores, which is an
SMT-desktop assumption. `nproc` reports 6 here, so it picks 3 — two big cores
plus one little one. The little core is the straggler at every barrier and
the big cores wait on it: the VAE takes **70% longer** than at 2 threads.
Two avoids the little cores; six has enough parallelism to hide them.

`n_threads` existed in `cantor_load_opts` but was never wired to the backend
(noted as outstanding in Part 8). Now wired:
`backend_set_n_threads()` -> `ace_engine_set_n_threads()` -> the ABI load
option, plus a `GGML_N_THREADS` env override for benchmarking without a
rebuild. Order of precedence: API setter, then env, then heuristic.

Caveat on the numbers: single runs, no thermal soak. The 3-thread VAE gap is
far too large to be noise; the 2-vs-6 difference is within it.

### Still open

- Vulkan on Adreno: driver present, build not attempted.
- CUDA/T4: still unvalidated.
- Peak RSS on device never measured; `vae_chunk` 512 never tried.
- `libcantor_engine.so` was pushed but nothing on-device loaded it — the
  test used the `ace-synth` CLI, not the ABI.
