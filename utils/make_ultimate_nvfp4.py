#!/usr/bin/env python3
"""
Generate a comprehensive, valid NVIDIA ModelOpt-style NVFP4 SafeTensors fixture.

It is designed to stress:
  - complex dotted tensor names;
  - multiple independent NVFP4 weight/scale/scale_2 trios;
  - scale tensors appearing before/after their weights in header order;
  - header order differing from physical data order;
  - 1, 3, 17, 4095, 4096, 4097, 8193, and 1,048,593 NVFP4 blocks;
  - exact chunk boundaries and irregular final chunks for chunk_blocks=4096;
  - uneven multicore partitions;
  - all 256 packed byte values over the repeating weight pattern;
  - a broad positive E4M3FN scale pattern, including subnormal and max finite values;
  - different F32 global scales per quantized tensor;
  - passthrough U8, F8_E4M3, F16, BF16, F32, scalar, and empty tensors;
  - a tensor whose name ends in ".weight" but is not NVFP4;
  - a tensor containing "weight_scale" in its name without using the reserved suffix.

The file is generated in bounded memory.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import OrderedDict
from pathlib import Path

CHUNK_BLOCKS = 4096
WRITE_BUFFER_BYTES = 1024 * 1024

# Positive, finite E4M3FN codes. This covers subnormals, small/medium normals,
# large values, and max finite (0x7e = 448).
SCALE_PATTERN = bytes([
    0x01, 0x07, 0x10, 0x20,
    0x28, 0x30, 0x38, 0x40,
    0x48, 0x50, 0x58, 0x60,
    0x68, 0x70, 0x78, 0x7E,
])

NVFP4_SPECS = [
    # name, blocks, packed shape, global scale, deterministic seed
    ("model.layers.0.self_attn.q_proj.weight",       1,       [1, 8],       1.0,   3),
    ("model.layers.0.self_attn.k_proj.weight",       3,       [3, 8],       0.25,  7),
    ("model.layers.0.self_attn.v_proj.weight",      17,       [1, 136],     2.0,  11),
    ("model.layers.1.mlp.gate_proj.weight",       4095,       [5, 6552],    0.5,  13),
    ("model.layers.1.mlp.up_proj.weight",         4096,       [64, 512],    1.5,  17),
    ("model.layers.1.mlp.down_proj.weight",       4097,       [4097, 8],    0.125,19),
    ("model.layers.12.self_attn.o_proj.weight",   8193,       [3, 21848],   0.75, 23),
    ("model.layers.31.mlp.down_proj.weight",   1048593,       [273, 30728], 1.25, 29),
]

# Header order is intentionally not physical data order.
HEADER_ORDER = [
    "__metadata__",

    "model.layers.1.mlp.up_proj.weight_scale",
    "model.embed_tokens.weight",
    "model.layers.12.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_proj.weight_scale_2",
    "model.layers.0.self_attn.q_proj.weight",

    "model.layers.31.mlp.down_proj.weight_scale_2",
    "model.layers.0.input_layernorm.weight",
    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.0.self_attn.k_proj.weight_scale",

    "tokenizer.lookup.weight",
    "model.layers.1.mlp.down_proj.weight_scale_2",
    "model.layers.0.self_attn.v_proj.weight",
    "aux.fp8_buffer",
    "model.layers.0.self_attn.k_proj.weight",

    "model.layers.12.self_attn.o_proj.weight_scale_2",
    "model.layers.1.mlp.gate_proj.weight_scale",
    "diagnostics.weight_scale_statistics",
    "model.layers.31.mlp.down_proj.weight",
    "model.layers.0.self_attn.v_proj.weight_scale",

    "scalar.temperature",
    "model.layers.1.mlp.up_proj.weight",
    "model.layers.0.self_attn.q_proj.weight_scale",
    "unquantized.weight",
    "empty.buffer",

    "model.layers.1.mlp.down_proj.weight",
    "model.layers.12.self_attn.o_proj.weight_scale",
    "model.layers.0.self_attn.k_proj.weight_scale_2",
    "model.layers.31.mlp.down_proj.weight_scale",
    "model.norm.weight",

    "model.layers.0.self_attn.v_proj.weight_scale_2",
    "model.layers.1.mlp.gate_proj.weight_scale_2",
    "model.layers.1.mlp.up_proj.weight_scale_2",
    "model.layers.1.mlp.down_proj.weight_scale",
]

# Physical data order is deliberately different again.
PHYSICAL_ORDER = [
    "model.layers.31.mlp.down_proj.weight",
    "model.layers.31.mlp.down_proj.weight_scale",
    "model.layers.31.mlp.down_proj.weight_scale_2",

    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.weight_scale",
    "model.layers.0.self_attn.q_proj.weight_scale_2",

    "model.embed_tokens.weight",

    "model.layers.1.mlp.down_proj.weight",
    "model.layers.1.mlp.down_proj.weight_scale",
    "model.layers.1.mlp.down_proj.weight_scale_2",

    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.k_proj.weight_scale",
    "model.layers.0.self_attn.k_proj.weight_scale_2",

    "model.layers.12.self_attn.o_proj.weight",
    "model.layers.12.self_attn.o_proj.weight_scale",
    "model.layers.12.self_attn.o_proj.weight_scale_2",

    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.1.mlp.gate_proj.weight_scale",
    "model.layers.1.mlp.gate_proj.weight_scale_2",

    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.v_proj.weight_scale",
    "model.layers.0.self_attn.v_proj.weight_scale_2",

    "model.layers.1.mlp.up_proj.weight",
    "model.layers.1.mlp.up_proj.weight_scale",
    "model.layers.1.mlp.up_proj.weight_scale_2",

    "model.layers.0.input_layernorm.weight",
    "tokenizer.lookup.weight",
    "aux.fp8_buffer",
    "diagnostics.weight_scale_statistics",
    "scalar.temperature",
    "unquantized.weight",
    "empty.buffer",
    "model.norm.weight",
]


def bf16_bytes(values):
    out = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        # These chosen values are exactly representable enough for the fixture;
        # keep the high 16 bits as BF16 storage.
        out += struct.pack("<H", bits >> 16)
    return bytes(out)


def f16_bytes(values):
    return b"".join(struct.pack("<e", float(v)) for v in values)


def make_passthrough():
    return {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [128, 64],
            "raw": bf16_bytes(((i % 29) - 14) / 8.0 for i in range(128 * 64)),
        },
        "model.layers.0.input_layernorm.weight": {
            "dtype": "F16",
            "shape": [64],
            "raw": f16_bytes(1.0 + (i % 7) / 16.0 for i in range(64)),
        },
        # Ends in .weight but has no NVFP4 scales: must remain a passthrough tensor.
        "tokenizer.lookup.weight": {
            "dtype": "U8",
            "shape": [257],
            "raw": bytes((i * 37 + 11) & 0xFF for i in range(257)),
        },
        "aux.fp8_buffer": {
            "dtype": "F8_E4M3",
            "shape": [32],
            "raw": (SCALE_PATTERN * 2),
        },
        # Contains "weight_scale" but does not end with ".weight_scale".
        "diagnostics.weight_scale_statistics": {
            "dtype": "F32",
            "shape": [16],
            "raw": b"".join(struct.pack("<f", i / 8.0) for i in range(16)),
        },
        "scalar.temperature": {
            "dtype": "F32",
            "shape": [],
            "raw": struct.pack("<f", 0.75),
        },
        "unquantized.weight": {
            "dtype": "F16",
            "shape": [32],
            "raw": f16_bytes((i - 16) / 4.0 for i in range(32)),
        },
        "empty.buffer": {
            "dtype": "F16",
            "shape": [0],
            "raw": b"",
        },
        "model.norm.weight": {
            "dtype": "F32",
            "shape": [64],
            "raw": b"".join(struct.pack("<f", 1.0 + i / 128.0) for i in range(64)),
        },
    }


def packed_period(seed: int) -> bytes:
    # 32 blocks * 8 bytes = 256 bytes. Every possible packed byte occurs
    # exactly once per period, in a seed-dependent rotation.
    return bytes((seed * 37 + i) & 0xFF for i in range(256))


def write_repeated(f, pattern: bytes, total_bytes: int) -> None:
    if total_bytes == 0:
        return
    repeats = max(1, WRITE_BUFFER_BYTES // len(pattern))
    chunk = pattern * repeats
    remaining = total_bytes
    while remaining:
        n = min(remaining, len(chunk))
        f.write(chunk[:n])
        remaining -= n


def build_entries():
    entries = {}

    for name, blocks, shape, global_scale, seed in NVFP4_SPECS:
        assert math_prod(shape) == blocks * 8
        entries[name] = {
            "dtype": "U8",
            "shape": shape,
            "kind": "packed",
            "blocks": blocks,
            "seed": seed,
        }
        entries[name + "_scale"] = {
            "dtype": "F8_E4M3",
            "shape": [blocks],
            "kind": "scale",
            "blocks": blocks,
            "seed": seed,
        }
        entries[name + "_scale_2"] = {
            "dtype": "F32",
            "shape": [],
            "kind": "global",
            "global_scale": global_scale,
        }

    for name, spec in make_passthrough().items():
        entries[name] = {
            **spec,
            "kind": "raw",
        }

    return entries


def math_prod(xs):
    result = 1
    for x in xs:
        result *= x
    return result


def byte_size(spec):
    kind = spec["kind"]
    if kind == "packed":
        return spec["blocks"] * 8
    if kind == "scale":
        return spec["blocks"]
    if kind == "global":
        return 4
    return len(spec["raw"])


def write_data(f, spec):
    kind = spec["kind"]

    if kind == "packed":
        write_repeated(f, packed_period(spec["seed"]), spec["blocks"] * 8)
    elif kind == "scale":
        seed = spec["seed"]
        # Rotate the 16-code pattern without changing its coverage.
        k = seed % len(SCALE_PATTERN)
        pattern = SCALE_PATTERN[k:] + SCALE_PATTERN[:k]
        write_repeated(f, pattern, spec["blocks"])
    elif kind == "global":
        f.write(struct.pack("<f", spec["global_scale"]))
    elif kind == "raw":
        f.write(spec["raw"])
    else:
        raise AssertionError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    entries = build_entries()
    assert set(PHYSICAL_ORDER) == set(entries), (set(entries) - set(PHYSICAL_ORDER), set(PHYSICAL_ORDER) - set(entries))
    assert set(HEADER_ORDER[1:]) == set(entries), (set(entries) - set(HEADER_ORDER[1:]), set(HEADER_ORDER[1:]) - set(entries))

    # Assign offsets according to PHYSICAL_ORDER.
    offsets = {}
    cursor = 0
    for name in PHYSICAL_ORDER:
        size = byte_size(entries[name])
        offsets[name] = [cursor, cursor + size]
        cursor += size

    header = OrderedDict()
    for name in HEADER_ORDER:
        if name == "__metadata__":
            header[name] = {
                "format": "ultimate-nvfp4-stress-v1",
                "purpose": "streaming SIMD multicore structural stress test",
                "chunk_blocks_reference": str(CHUNK_BLOCKS),
            }
            continue

        spec = entries[name]
        header[name] = {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "data_offsets": offsets[name],
        }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padded_len = (len(header_bytes) + 7) & ~7
    header_bytes += b" " * (padded_len - len(header_bytes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name in PHYSICAL_ORDER:
            write_data(f, entries[name])

    total_blocks = sum(spec[1] for spec in NVFP4_SPECS)
    print(f"Wrote: {args.output}")
    print(f"Input size: {args.output.stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"NVFP4 tensors: {len(NVFP4_SPECS)}")
    print(f"Total NVFP4 blocks: {total_blocks:,}")
    print("Block counts:", ", ".join(f"{x[1]:,}" for x in NVFP4_SPECS))
    print(f"Largest tensor final chunk at 4096 blocks: {NVFP4_SPECS[-1][1] % CHUNK_BLOCKS} blocks")


if __name__ == "__main__":
    main()
