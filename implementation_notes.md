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
