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
| 2 (big cores only) | 37.7 s | **25.2 s** | 62.9 s |
| 3 (**the built-in default**) | 38.1 s | 42.9 s | **81.1 s** |
| 6 (all cores) | **32.5 s** | 28.7 s | **61.2 s** |

The default is `hardware_concurrency() / 2`, which assumes logical/2 means
"physical cores". That holds for an SMT desktop. Here `nproc` reports 6, so
it picks 3 — and 3 is the **worst** of the three, 32% slower overall than
either neighbour.

The VAE is where it hurts: 42.9 s at 3 threads against 25.2 s at 2. Three
threads is two big cores plus one little one, and the little core becomes the
straggler at every barrier; the fast cores spend their time waiting. Two
threads avoids the little cores entirely; six has enough parallelism to
absorb them.

**Set the thread count explicitly on mobile.** Either
`cantor_load_opts.n_threads`, `ace_engine_set_n_threads()`, or the
`GGML_N_THREADS` environment variable for benchmarking. Six was marginally
best here, but two is within 3% and draws far less power — worth measuring
per device rather than assuming.

Single runs, no thermal soak. The 3-thread VAE gap is large enough to be
signal; the 2-vs-6 difference is within run-to-run noise.

## Not done yet

- **Vulkan on Adreno.** The driver is there; the build is not.
- **Memory headroom.** Nothing measured peak RSS, and no larger variant or
  longer track was tried. `vae_chunk 256` was chosen and worked; whether 512
  also fits is unknown.
- **The engine `.so` was pushed but nothing loaded it.** `ace-synth` is a
  CLI; a real host would `dlopen libcantor_engine.so`.
