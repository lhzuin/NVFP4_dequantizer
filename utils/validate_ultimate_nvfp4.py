#!/usr/bin/env python3
"""
Validate the F16 output produced from ultimate_nvfp4_input.safetensors.

Usage:
    python3 validate_ultimate_nvfp4.py \
        ultimate_nvfp4_input.safetensors \
        converted_output.safetensors

The validator checks:
  - output tensor order;
  - removal of NVFP4 scale tensors;
  - output dtype and doubled final dimension for every NVFP4 weight;
  - recomputed contiguous output offsets;
  - exact passthrough bytes for all copied tensors;
  - every dequantized F16 byte for every NVFP4 tensor.

Expected NVFP4 output is generated from a 32-block repeating period, so even
the million-block tensor can be checked efficiently without materializing the
full expected result.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

E2M1 = (
    0.0, 0.5, 1.0, 1.5,
    2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
)

SCALE_PATTERN = bytes([
    0x01, 0x07, 0x10, 0x20,
    0x28, 0x30, 0x38, 0x40,
    0x48, 0x50, 0x58, 0x60,
    0x68, 0x70, 0x78, 0x7E,
])

NVFP4_INFO = {
    "model.layers.0.self_attn.q_proj.weight":     (1,       1.0,   3),
    "model.layers.0.self_attn.k_proj.weight":     (3,       0.25,  7),
    "model.layers.0.self_attn.v_proj.weight":     (17,      2.0,  11),
    "model.layers.1.mlp.gate_proj.weight":        (4095,    0.5,  13),
    "model.layers.1.mlp.up_proj.weight":          (4096,    1.5,  17),
    "model.layers.1.mlp.down_proj.weight":        (4097,    0.125,19),
    "model.layers.12.self_attn.o_proj.weight":    (8193,    0.75, 23),
    "model.layers.31.mlp.down_proj.weight":       (1048593, 1.25, 29),
}


def read_header(path: Path):
    f = path.open("rb")
    length_raw = f.read(8)
    if len(length_raw) != 8:
        raise AssertionError(f"{path}: truncated header prefix")
    header_len = struct.unpack("<Q", length_raw)[0]
    header_raw = f.read(header_len)
    if len(header_raw) != header_len:
        raise AssertionError(f"{path}: truncated JSON header")
    header = json.loads(header_raw.decode("utf-8"))
    return f, header, 8 + header_len


def is_nvfp4_scale(name: str) -> bool:
    return name.endswith(".weight_scale") or name.endswith(".weight_scale_2")


def decode_e4m3(bits: int) -> float:
    sign = -1.0 if (bits & 0x80) else 1.0
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07

    if exponent == 0:
        value = (mantissa / 8.0) * (2.0 ** -6)
    elif exponent == 0x0F:
        if mantissa == 0x07:
            return float("nan")
        value = (1.0 + mantissa / 8.0) * (2.0 ** 8)
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))

    return sign * value


def packed_period(seed: int) -> bytes:
    return bytes((seed * 37 + i) & 0xFF for i in range(256))


def scale_period(seed: int) -> bytes:
    k = seed % len(SCALE_PATTERN)
    return SCALE_PATTERN[k:] + SCALE_PATTERN[:k]


def expected_f16_period(global_scale: float, seed: int) -> bytes:
    packed = packed_period(seed)
    scales = scale_period(seed)
    out = bytearray()

    # packed period = 32 blocks; scale period = 16 blocks.
    for block in range(32):
        scale = decode_e4m3(scales[block % 16]) * global_scale
        block_bytes = packed[block * 8:(block + 1) * 8]
        for byte in block_bytes:
            low = byte & 0x0F
            high = byte >> 4
            out += struct.pack("<e", E2M1[low] * scale)
            out += struct.pack("<e", E2M1[high] * scale)

    return bytes(out)


def read_tensor(f, data_start: int, info):
    begin, end = info["data_offsets"]
    f.seek(data_start + begin)
    raw = f.read(end - begin)
    if len(raw) != end - begin:
        raise AssertionError("truncated tensor data")
    return raw


def compare_repeated(actual_f, actual_start, actual_info, period: bytes, total_bytes: int, name: str):
    begin, end = actual_info["data_offsets"]
    if end - begin != total_bytes:
        raise AssertionError(f"{name}: output byte size {end-begin} != expected {total_bytes}")

    actual_f.seek(actual_start + begin)
    compare_chunk = period * max(1, (1024 * 1024) // len(period))

    remaining = total_bytes
    byte_offset = 0
    while remaining:
        n = min(remaining, len(compare_chunk))
        actual = actual_f.read(n)
        if len(actual) != n:
            raise AssertionError(f"{name}: truncated output")

        # n may cut the repeating period.
        expected = (period * ((n + len(period) - 1) // len(period)))[:n]
        if actual != expected:
            for i, (a, e) in enumerate(zip(actual, expected)):
                if a != e:
                    raise AssertionError(
                        f"{name}: first mismatch at output byte {byte_offset+i}: "
                        f"actual=0x{a:02x}, expected=0x{e:02x}"
                    )
            raise AssertionError(f"{name}: output mismatch")

        byte_offset += n
        remaining -= n


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: validate_ultimate_nvfp4.py INPUT_NVFP4 CONVERTED_F16"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    inf, ih, ids = read_header(input_path)
    outf, oh, ods = read_header(output_path)

    try:
        input_names = [n for n in ih if n != "__metadata__"]
        expected_output_names = [n for n in input_names if not is_nvfp4_scale(n)]
        actual_output_names = [n for n in oh if n != "__metadata__"]

        if actual_output_names != expected_output_names:
            raise AssertionError(
                "output tensor order differs\n"
                f"expected: {expected_output_names}\n"
                f"actual:   {actual_output_names}"
            )

        cursor = 0

        for name in expected_output_names:
            input_info = ih[name]
            actual_info = oh[name]

            if name in NVFP4_INFO:
                blocks, global_scale, seed = NVFP4_INFO[name]

                expected_shape = list(input_info["shape"])
                expected_shape[-1] *= 2
                expected_bytes = blocks * 16 * 2

                if actual_info["dtype"] != "F16":
                    raise AssertionError(f"{name}: dtype != F16")
                if actual_info["shape"] != expected_shape:
                    raise AssertionError(
                        f"{name}: shape {actual_info['shape']} != {expected_shape}"
                    )

                if actual_info["data_offsets"] != [cursor, cursor + expected_bytes]:
                    raise AssertionError(
                        f"{name}: offsets {actual_info['data_offsets']} "
                        f"!= {[cursor, cursor + expected_bytes]}"
                    )

                period = expected_f16_period(global_scale, seed)
                compare_repeated(
                    outf, ods, actual_info, period, expected_bytes, name
                )
                cursor += expected_bytes
                print(f"OK NVFP4: {name} ({blocks:,} blocks)")

            else:
                raw = read_tensor(inf, ids, input_info)
                actual = read_tensor(outf, ods, actual_info)

                if actual_info["dtype"] != input_info["dtype"]:
                    raise AssertionError(f"{name}: passthrough dtype changed")
                if actual_info["shape"] != input_info["shape"]:
                    raise AssertionError(f"{name}: passthrough shape changed")
                if actual_info["data_offsets"] != [cursor, cursor + len(raw)]:
                    raise AssertionError(f"{name}: passthrough offsets wrong")
                if actual != raw:
                    raise AssertionError(f"{name}: passthrough bytes differ")

                cursor += len(raw)
                print(f"OK copy:  {name} ({len(raw):,} bytes)")

        print("\nULTIMATE VALIDATION PASSED")
        print(f"Validated {len(NVFP4_INFO)} NVFP4 tensors and "
              f"{len(expected_output_names) - len(NVFP4_INFO)} passthrough tensors.")
    finally:
        inf.close()
        outf.close()


if __name__ == "__main__":
    main()
