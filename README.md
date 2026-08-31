# NVFP4 SafeTensors Dequantizer

A CPU NVFP4 dequantizer written in Zig 0.16. It reads NVIDIA ModelOpt-style NVFP4 weights from a SafeTensors file, converts them to F16 in bounded chunks, and writes a new SafeTensors file. Non-quantized tensors are copied byte-for-byte.

## Features

- SafeTensors header parsing, conversion planning, and output generation
- NVFP4 decoding from packed E2M1 values, E4M3FN block scales, and an F32 global scale
- SIMD dequantization of complete 16-value NVFP4 blocks using Zig vector types
- Configurable multicore execution with deterministic output
- Bounded-memory processing instead of loading complete tensors into memory
- I/O built around `std.Io.Reader` and `std.Io.Writer`
- No third-party runtime dependencies

## Supported formats

NVFP4 tensors are detected using the following ModelOpt naming and dtype convention:

- `<name>.weight`: `U8`, containing two packed E2M1 values per byte
- `<name>.weight_scale`: `F8_E4M3`, containing one scale per 16-value block
- `<name>.weight_scale_2`: `F32`, containing one global scale

Recognized SafeTensors dtypes are `U8`, `F8_E4M3`, `F16`, `BF16`, and `F32`.

Output formats:

- `F16` — currently the only dequantized output format

Converted weight tensors retain their dimensions except for the final dimension, which is doubled to account for the two values packed in each input byte. Their scale tensors are omitted from the output. Other recognized tensors are preserved unchanged.

## Requirements

- Zig 0.16.0
- Python 3 for integration tests and fixture utilities

## Build

Build and install the debug executable:

```sh
zig build
```

The resulting binary is `zig-out/bin/nvfp4-dequant`.

For an optimized build:

```sh
zig build -Doptimize=ReleaseFast
```

## Usage

```sh
./zig-out/bin/nvfp4-dequant INPUT OUTPUT \
    [--output-format FORMAT] [--threads N] [--chunk-blocks N]
```

For example:

```sh
./zig-out/bin/nvfp4-dequant model_nvfp4.safetensors model_f16.safetensors \
    --output-format f16 --threads 8 --chunk-blocks 4096
```

`--output-format` defaults to `f16`, which is currently the only supported output format. The numeric options must be greater than zero: `--threads` defaults to `1`, while `--chunk-blocks` defaults to `4096` and controls the maximum number of NVFP4 blocks held in each processing chunk. The executable can also be built and run in one command:

```sh
zig build run -- model_nvfp4.safetensors model_f16.safetensors \
    --output-format f16 --threads 8 --chunk-blocks 4096
```

## Design

The converter first parses the input header and constructs an output plan. Quantized tensors are then processed in chunks of up to 4096 blocks by default. Each chunk reads only its packed weights and block scales, applies the global scale, and writes the resulting F16 values. Passthrough tensors are streamed directly to the output. Memory use is therefore bounded by the chunk size rather than the model size.

The SIMD kernel handles one NVFP4 block at a time: eight packed bytes are unpacked into sixteen E2M1 values, scaled, and converted to F16 using Zig `@Vector` operations.

Multithreading is performed within each chunk. The main thread reads a chunk once, partitions it into disjoint block ranges, waits for workers to dequantize those ranges, and writes the completed output chunk sequentially. Workers never share a seekable reader or writer, which avoids I/O cursor races and out-of-order writes while keeping output deterministic. Since a chunk can contain thousands of blocks, each task still provides meaningful multicore work. A future producer/worker/ordered-writer pipeline could overlap I/O and computation, at the cost of additional synchronization and in-flight buffers.

### Specification coverage

| Requirement | Implementation |
| --- | --- |
| SafeTensors input | Native header parser and conversion planner |
| On-the-fly dequantization | Bounded chunk processing and streamed passthrough copies |
| SIMD | Vectorized 16-value NVFP4 block kernel |
| `std.Io.Reader` / `std.Io.Writer` | Reader/writer interfaces throughout parsing and conversion |
| Multicore | Configurable intra-chunk parallel dequantization |

## Project structure

```text
.
├── build.zig                         Build, run, and test steps
├── src/
│   ├── main.zig                      Command-line interface
│   ├── convert.zig                   Conversion planning, chunking, and threading
│   ├── nvfp4.zig                     Scalar and SIMD NVFP4 kernels
│   └── safetensors.zig               SafeTensors header reader and writer
├── tests/
│   ├── tests.zig                     Codec, SIMD, chunk, and header tests
│   ├── convert_tests.zig             File conversion and determinism tests
│   ├── test_integration.py           End-to-end CLI tests
│   └── fixtures/                     Small known-good SafeTensors fixtures
└── utils/
    ├── inspect_safetensors.py         Dependency-free file inspector
    ├── make_tiny_nvfp4_fixture.py     Small fixture generator
    ├── make_ultimate_nvfp4.py         Large edge-case fixture generator
    └── validate_ultimate_nvfp4.py     Exact large-fixture validator
```

## Testing

Run the complete test suite:

```sh
zig build test
```

The suite includes scalar decoding checks, exhaustive SIMD/scalar equivalence cases, SafeTensors parsing and writing, conversion error handling, chunk-boundary coverage, thread-count and chunk-size invariance, repeated-run determinism, and end-to-end CLI validation.

Individual groups can be run with:

```sh
zig build test-unit
zig build test-integration
```

The larger generated fixture exercises multiple tensor shapes, boundary sizes, physical tensor ordering, passthrough dtypes, and more than one million NVFP4 blocks:

```sh
python3 utils/make_ultimate_nvfp4.py /tmp/ultimate_nvfp4_input.safetensors
./zig-out/bin/nvfp4-dequant \
    /tmp/ultimate_nvfp4_input.safetensors \
    /tmp/ultimate_f16_output.safetensors \
    --threads 8
python3 utils/validate_ultimate_nvfp4.py \
    /tmp/ultimate_nvfp4_input.safetensors \
    /tmp/ultimate_f16_output.safetensors
```

SafeTensors headers and values can be inspected with:

```sh
python3 utils/inspect_safetensors.py FILE [--decode-nvfp4]
```
