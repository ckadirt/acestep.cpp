# Running on Android

Cross-compiled from x86_64 with the NDK. No Termux, no on-device toolchain,
no root — `adb push` into `/data/local/tmp`, which permits exec.

## Build

```bash
NDK=$HOME/Android/Sdk/ndk/27.1.12297006

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16

cmake --build build-android --target ace-synth cantor-engine -j "$(nproc)"
```

`GGML_NATIVE=OFF` matters: without it the build detects the *host* CPU and
emits x86 feature flags into an ARM build.

### Picking the ARM arch flags

Read them off the device rather than guessing:

```bash
adb shell "grep -m1 Features /proc/cpuinfo" | tr ' ' '\n' \
  | grep -E "asimddp|i8mm|sve|asimdhp|bf16"
```

| Feature | Means | Flag |
|---|---|---|
| `asimddp` | dot product | `+dotprod` |
| `asimdhp` | fp16 arithmetic | `+fp16` |
| `i8mm` | int8 matmul | `+i8mm` |
| `sve` | scalable vectors | `+sve` |

These map onto ggml's own variant list (`ggml_add_cpu_backend_variant(android_armv8.2_2 DOTPROD FP16_VECTOR_ARITHMETIC)` and friends), so a device usually matches an existing one exactly.

## Deploy

```bash
adb shell "mkdir -p /data/local/tmp/ace/models"
adb push build-android/ace-synth build-android/*.so /data/local/tmp/ace/
adb shell "chmod +x /data/local/tmp/ace/ace-synth"
adb push <models>/*.gguf /data/local/tmp/ace/models/     # 2.57 GB, ~76 s over USB

adb shell "cd /data/local/tmp/ace && \
  LD_LIBRARY_PATH=/data/local/tmp/ace GGML_N_THREADS=6 \
  ./ace-synth --models models --request req.json --vae-chunk 256"
```

Binaries must live on `/data/local/tmp` — `/sdcard` is mounted `noexec`.
Models can live anywhere readable.

## Measured: Snapdragon 695 (SM6375), 7.6 GB RAM, Android 13

2 x Cortex-A78 @ 2.2 GHz + 6 x Cortex-A55 @ 1.8 GHz. CPU backend only.

12-second track, `1.5-fast` models, `--vae-chunk 256`, 2 threads:

| Stage | Time |
|---|---|
| DiT, 8 steps | 72.7 s |
| VAE decode (3 tiles) | 89.1 s |

It completes, within memory, no OOM. It is roughly **14x slower than
realtime** on this class of device, so treat mid-range CPU-only as a
correctness target rather than a product experience. The Adreno GPU is
present with `vulkan.adreno.so` and is the obvious next step.

Audio is equivalent to desktop: RMS 2861.5 against 2808.8 on x86_64 CPU and
2846.8 on Vulkan, all at full-scale peak.

### Thread count: the default heuristic is the worst choice here

6-second track, sweeping `GGML_N_THREADS`:

| Threads | DiT | VAE | Total |
|---:|---:|---:|---:|
| 2 (big cores only) | 37.7 s | 25.2 s | 62.9 s |
| 3 (**the old default**) | 38.1 s | 42.9 s | **81.1 s** |
| 6 | 32.5 s | 28.7 s | 61.2 s |
| 8 = all cores (**the current default**) | **28.2 s** | **22.0 s** | **50.2 s** |

The old default was `hardware_concurrency() / 2`, which assumes logical/2
means "physical cores". That holds for an SMT desktop. This SoC has no SMT,
so halving simply discarded cores — and it landed on 3, which is both big
cores plus one little one. The little core strands every barrier while the
fast cores wait: the VAE took 42.9 s against 25.2 s at two threads.

The current default checks `/sys/devices/system/cpu/smt/active` and only
halves when SMT is really present. Here that yields all cores, which is
**38% faster end to end** than the old default and is the best of every count
tried, on both stages.

Note the core count is not stable: `nproc` reported 6 during one session and
`/proc/cpuinfo` lists 8, because Android parks cores. The old heuristic
therefore picked 3 or 4 depending on when it was asked.

Explicit control remains available and is still worth using on mobile —
`cantor_load_opts.n_threads`, `ace_engine_set_n_threads()`, `--threads N` on
the CLI, or `GGML_N_THREADS` for benchmarking. Two threads is 25% slower
here but uses a quarter of the cores, which may be the trade you want on
battery.

Single runs, no thermal soak, device at 32.8 degC throughout — so these are
cold-device numbers and sustained generation will likely be slower.

## Not done yet

- **Vulkan on Adreno.** The driver is there; the build is not.
- **Memory headroom.** Nothing measured peak RSS, and no larger variant or
  longer track was tried. `vae_chunk 256` was chosen and worked; whether 512
  also fits is unknown.
- **The engine `.so` was pushed but nothing loaded it.** `ace-synth` is a
  CLI; a real host would `dlopen libcantor_engine.so`.
