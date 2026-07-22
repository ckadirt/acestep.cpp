# Implementation notes — engine execution model (Parts 5–8)

Working log for executing the plan at
<https://claude.ai/code/artifact/f4d51109-a383-4096-a1e9-c417480fc317>
("Where the work can stop", companion to *Filling the shelves* Parts 1–4).

Branch: `engine-stages`, off `master` at `fa33775`.

Every section headed **Deviation** records where reality differed from the
plan. The plan is not retro-edited; these notes are the amendment.

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
