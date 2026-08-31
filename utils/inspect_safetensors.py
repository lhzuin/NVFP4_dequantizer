#!/usr/bin/env python3
"""
inspect_safetensors.py

Tiny dependency-free SafeTensors inspector, intended for debugging the NVFP4
fixture used in the Zig dequantizer project.

Usage:
    python inspect_safetensors.py tiny_nvfp4.safetensors
    python inspect_safetensors.py tiny_nvfp4.safetensors --decode-nvfp4
"""

import argparse
import json
import math
import struct
from pathlib import Path


E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5,
    2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
]


def decode_e4m3fn(bits: int) -> float:
    """Decode one NVIDIA-style E4M3FN byte to Python float."""
    sign = -1.0 if (bits & 0x80) else 1.0
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    bias = 7

    # E4M3FN NaN encoding: exponent=1111, mantissa=111.
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan

    if exponent == 0:
        value = (2.0 ** (1 - bias)) * (mantissa / 8.0)
    else:
        value = (2.0 ** (exponent - bias)) * (1.0 + mantissa / 8.0)

    return sign * value


def read_safetensors(path: Path):
    data = path.read_bytes()

    if len(data) < 8:
        raise ValueError("File is too small to be a SafeTensors file")

    # SafeTensors stores the JSON header length as u64 LITTLE-ENDIAN.
    header_len = struct.unpack_from("<Q", data, 0)[0]

    header_begin = 8
    header_end = header_begin + header_len
    if header_end > len(data):
        raise ValueError("Header extends beyond end of file")

    header_bytes = data[header_begin:header_end]
    header_text = header_bytes.decode("utf-8")
    header = json.loads(header_text)

    data_section = data[header_end:]

    return header_len, header_text, header, data_section


def tensor_bytes(meta, data_section: bytes) -> bytes:
    begin, end = meta["data_offsets"]
    return data_section[begin:end]


def decode_known_dtype(dtype: str, raw: bytes):
    if dtype == "U8":
        return list(raw)

    if dtype == "F32":
        if len(raw) % 4 != 0:
            return "<invalid F32 byte count>"
        return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))

    if dtype == "F16":
        if len(raw) % 2 != 0:
            return "<invalid F16 byte count>"
        return list(struct.unpack("<" + "e" * (len(raw) // 2), raw))

    if dtype in ("F8_E4M3", "F8_E4M3FN"):
        return [decode_e4m3fn(x) for x in raw]

    return "<not decoded by this script>"


def inspect(path: Path, decode_nvfp4: bool):
    header_len, header_text, header, data_section = read_safetensors(path)

    print(f"File: {path}")
    print(f"Total file size: {path.stat().st_size} bytes")
    print(f"Header length: {header_len} bytes")
    print(f"Data section starts at absolute byte: {8 + header_len}")
    print(f"Data section size: {len(data_section)} bytes")

    print("\n=== RAW JSON HEADER ===")
    print(header_text)

    print("\n=== PRETTY JSON HEADER ===")
    print(json.dumps(header, indent=2))

    print("\n=== TENSORS ===")
    for name, meta in header.items():
        if name == "__metadata__":
            print("\n__metadata__:")
            print(meta)
            continue

        raw = tensor_bytes(meta, data_section)
        begin, end = meta["data_offsets"]

        print(f"\n{name}")
        print(f"  dtype:        {meta['dtype']}")
        print(f"  shape:        {meta['shape']}")
        print(f"  data_offsets: [{begin}, {end}]   (relative to data section)")
        print(f"  byte size:    {len(raw)}")
        print(f"  raw hex:      {raw.hex(' ')}")
        print(f"  decoded:      {decode_known_dtype(meta['dtype'], raw)}")

        if name.endswith(".weight") and meta["dtype"] == "U8":
            codes = []
            for byte in raw:
                codes.append(byte & 0x0F)       # low nibble first
                codes.append((byte >> 4) & 0x0F)
            print(f"  FP4 codes:    {codes}")
            print(f"  E2M1 values:  {[E2M1_VALUES[c] for c in codes]}")

    if decode_nvfp4:
        print("\n=== NVFP4 LOGICAL WEIGHTS ===")

        for name, meta in header.items():
            if name == "__metadata__" or not name.endswith(".weight"):
                continue
            if meta.get("dtype") != "U8":
                continue

            scale_name = name + "_scale"
            scale2_name = name + "_scale_2"

            if scale_name not in header or scale2_name not in header:
                print(f"{name}: skipped (matching scale tensors not found)")
                continue

            packed = tensor_bytes(meta, data_section)
            block_scale_raw = tensor_bytes(header[scale_name], data_section)
            global_scale_raw = tensor_bytes(header[scale2_name], data_section)

            if len(global_scale_raw) != 4:
                raise ValueError(f"{scale2_name}: expected exactly one F32")
            global_scale = struct.unpack("<f", global_scale_raw)[0]

            fp4_codes = []
            for byte in packed:
                fp4_codes.append(byte & 0x0F)
                fp4_codes.append((byte >> 4) & 0x0F)

            if len(fp4_codes) % 16 != 0:
                raise ValueError(f"{name}: number of logical FP4 values is not divisible by 16")

            expected_scales = len(fp4_codes) // 16
            if len(block_scale_raw) != expected_scales:
                raise ValueError(
                    f"{scale_name}: expected {expected_scales} block scales, "
                    f"got {len(block_scale_raw)}"
                )

            values = []
            for block_idx in range(expected_scales):
                scale = decode_e4m3fn(block_scale_raw[block_idx]) * global_scale
                block_codes = fp4_codes[block_idx * 16:(block_idx + 1) * 16]
                values.extend(E2M1_VALUES[c] * scale for c in block_codes)

            packed_shape = meta["shape"]
            logical_shape = list(packed_shape)
            if logical_shape:
                logical_shape[-1] *= 2

            print(f"\n{name}")
            print(f"  packed shape:  {packed_shape}")
            print(f"  logical shape: {logical_shape}")
            print(f"  global scale:  {global_scale}")
            print(
                "  block scales:  "
                + str([decode_e4m3fn(x) for x in block_scale_raw])
            )
            print(f"  dequantized:   {values}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--decode-nvfp4",
        action="store_true",
        help="also reconstruct logical NVFP4 weight values",
    )
    args = parser.parse_args()

    inspect(args.path, args.decode_nvfp4)


if __name__ == "__main__":
    main()
