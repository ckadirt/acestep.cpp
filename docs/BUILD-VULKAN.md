# Building the Vulkan backend without root

The LunarG SDK is the supported route and is what CI uses. On a machine
where you cannot `sudo apt install vulkan-sdk`, the tarball SDK works
unpacked anywhere — nothing needs installing system-wide.

What Ubuntu 22.04 already gives you is the **runtime**: `mesa-vulkan-drivers`
ships `radeon_icd` (RADV, for AMD) and `lvp_icd` (lavapipe, software), and
`libvulkan.so` is present. What is missing is the **build** side: `glslc` and
SPIRV-Headers. `glslc` is not packaged on 22.04 at all, which is the usual
reason a Vulkan configure fails here:

```
CMake Error at ggml/src/ggml-vulkan/CMakeLists.txt:14 (find_package):
  Could not find a package configuration file provided by "SPIRV-Headers"
```

## Setup

```bash
curl -L -o vulkan-sdk.tar.xz \
  "https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.xz"
tar xf vulkan-sdk.tar.xz            # ~333 MB download
export VULKAN_SDK=$PWD/<version>/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
```

## Build

```bash
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$VULKAN_SDK"
cmake --build build-vulkan --config Release -j "$(nproc)"
```

Expect the shader generation phase to dominate — ggml compiles thousands of
variants, so 20-30 minutes on 12 cores is normal. Zero compiler errors in a
clean tree.

## Running

```bash
LD_LIBRARY_PATH=$VULKAN_SDK/lib GGML_BACKEND=Vulkan0 \
  ./build-vulkan/ace-synth --models <dir> --request req.json
```

**Pin the device explicitly.** `vulkaninfo --summary` on this machine
enumerates two: RADV RENOIR (the real iGPU) and llvmpipe (software). Without
`GGML_BACKEND=Vulkan0` a run that reports "Vulkan" may have quietly selected
llvmpipe, which tells you nothing about hardware behaviour. The log line to
check names the device:

```
ggml_vulkan: 0 = AMD Unknown (RADV RENOIR) (radv) | uma: 1 | fp16: 1 | ...
```

## Measured on AMD Cezanne (Renoir iGPU, unified memory)

12-second reference render, `1.5-fast` models, seed 42:

| Stage | CPU (12 threads + BLAS) | Vulkan (RADV) |
|---|---|---|
| DiT, 8 steps | 8 727 ms | **5 435 ms** |
| VAE decode | 11 120 ms | **7 791 ms** |

Roughly 1.6x and 1.4x. Note `matrix cores: none` and `uma: 1` — this is an
integrated GPU sharing system RAM with no tensor-core equivalent, so treat
these as a floor rather than an indication of what a discrete card does.

Audio is equivalent but **not bit-identical** to CPU, which is expected:
different kernels, different float ordering.

| | peak | rms | non-zero |
|---|---|---|---|
| CPU | 32767 | 2808.8 | 100% |
| Vulkan | 32767 | 2846.8 | 100% |

1.4% RMS difference, full-scale peak on both.

## What this validated

`test-dit-resume` and `test-abi` both pass on RADV. In particular
**pausing at step 5 of 8 and resuming reaches byte-identical latents on
Vulkan**, the same guarantee as CPU. Resume determinism therefore holds
within a backend on real GPU hardware, not only on CPU.

It does **not** say anything about resuming a blob produced on one backend
and continued on another — that remains unenforced and untested, and the
paused blob's backend stamp is still written and ignored. See `docs/ABI.md`.
