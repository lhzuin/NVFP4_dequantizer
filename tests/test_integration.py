#!/usr/bin/env python3
"""
End-to-end tests for the NVFP4 SafeTensors dequantizer (AI generated).

Run from the repository root:

    python3 tests/test_integration.py

The script:
  1. runs `zig build` when invoked directly;
  2. converts the tiny fixture through the real CLI;
  3. compares output metadata and raw tensor bytes against the known-good file;
  4. generates a 4097-block NVFP4 fixture so the converter must cross a
     4096-block chunk boundary, converts it, and validates every block;
  5. verifies invalid block-scale counts and global-scale sizes are rejected.

Only the Python standard library is used.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "zig-out" / "bin" / "nvfp4-dequant"
FIXTURES = ROOT / "tests" / "fixtures"

E2M1 = [
    0.0, 0.5, 1.0, 1.5,
    2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
]


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def run_expect_failure(*args: str, expected_error: str) -> None:
    print("+", " ".join(args), f"(expecting {expected_error})")
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "command unexpectedly succeeded"
    diagnostics = result.stdout + result.stderr
    assert expected_error in diagnostics, (
        f"expected {expected_error!r} in diagnostics:\n{diagnostics}"
    )


def read_safetensors(path: Path):
    raw = path.read_bytes()

    if len(raw) < 8:
        raise AssertionError(f"{path}: file shorter than 8-byte header prefix")

    header_len = struct.unpack_from("<Q", raw, 0)[0]
    header_end = 8 + header_len

    if header_end > len(raw):
        raise AssertionError(f"{path}: declared header exceeds file size")

    header = json.loads(raw[8:header_end].decode("utf-8"))
    data = raw[header_end:]
    return header, data


def tensor_bytes(header, data, name: str) -> bytes:
    begin, end = header[name]["data_offsets"]
    return data[begin:end]


def compare_safetensors(actual_path: Path, expected_path: Path) -> None:
    ah, ad = read_safetensors(actual_path)
    eh, ed = read_safetensors(expected_path)

    actual_names = [name for name in ah if name != "__metadata__"]
    expected_names = [name for name in eh if name != "__metadata__"]

    assert actual_names == expected_names, (
        f"tensor order/names differ:\n"
        f"actual:   {actual_names}\n"
        f"expected: {expected_names}"
    )

    for name in expected_names:
        for field in ("dtype", "shape", "data_offsets"):
            assert ah[name][field] == eh[name][field], (
                f"{name}.{field}: "
                f"{ah[name][field]!r} != {eh[name][field]!r}"
            )

        actual_bytes = tensor_bytes(ah, ad, name)
        expected_bytes = tensor_bytes(eh, ed, name)

        assert actual_bytes == expected_bytes, (
            f"{name}: raw bytes differ"
        )


def write_safetensors(path: Path, entries, data_parts) -> None:
    header = {}
    offset = 0

    for name, dtype, shape, raw in entries:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)

    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
    ).encode("utf-8")

    # Match common SafeTensors writers: pad header with spaces to 8 bytes.
    padded_len = (len(header_bytes) + 7) & ~7
    header_bytes += b" " * (padded_len - len(header_bytes))

    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for raw in data_parts:
            f.write(raw)


def make_chunk_boundary_fixture(path: Path, blocks: int = 4097) -> None:
    # Every encoded byte is 0x21:
    # low nibble = 1 -> +0.5
    # high nibble = 2 -> +1.0
    weight_raw = bytes([0x21]) * (blocks * 8)

    # Every block scale is 1.0 in E4M3FN.
    scale_raw = bytes([0x38]) * blocks

    # Global scale = 1.0 F32.
    global_raw = struct.pack("<f", 1.0)

    entries = [
        ("linear.weight", "U8", [1, blocks * 8], weight_raw),
        ("linear.weight_scale", "F8_E4M3", [blocks], scale_raw),
        ("linear.weight_scale_2", "F32", [], global_raw),
    ]

    write_safetensors(
        path,
        entries,
        [weight_raw, scale_raw, global_raw],
    )


def validate_chunk_boundary_output(path: Path, blocks: int = 4097) -> None:
    header, data = read_safetensors(path)

    names = [name for name in header if name != "__metadata__"]
    assert names == ["linear.weight"], names

    info = header["linear.weight"]
    assert info["dtype"] == "F16"
    assert info["shape"] == [1, blocks * 16]
    assert info["data_offsets"] == [0, blocks * 16 * 2]

    raw = tensor_bytes(header, data, "linear.weight")
    assert len(raw) == blocks * 16 * 2

    values = struct.unpack("<" + "e" * (len(raw) // 2), raw)

    # 0x21 repeats -> [0.5, 1.0] repeated eight times per block.
    expected_block = [0.5, 1.0] * 8
    for block_index in range(blocks):
        begin = block_index * 16
        assert list(values[begin:begin + 16]) == expected_block, (
            f"block {block_index} differs"
        )


def make_invalid_fixture(
    path: Path,
    *,
    weight_blocks: int,
    scale_count: int,
    global_scale_raw: bytes,
) -> None:
    weight_raw = bytes([0x21]) * (weight_blocks * 8)
    scale_raw = bytes([0x38]) * scale_count

    entries = [
        ("linear.weight", "U8", [1, weight_blocks * 8], weight_raw),
        ("linear.weight_scale", "F8_E4M3", [scale_count], scale_raw),
        ("linear.weight_scale_2", "F32", [], global_scale_raw),
    ]
    write_safetensors(
        path,
        entries,
        [weight_raw, scale_raw, global_scale_raw],
    )


def main() -> None:
    if os.environ.get("NVFP4_SKIP_BUILD") != "1":
        run("zig", "build")

    tiny_input = FIXTURES / "tiny_nvfp4.safetensors"
    tiny_expected = FIXTURES / "tiny_expected_f16.safetensors"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        tiny_output = tmp / "tiny_output.safetensors"

        run(
            str(BIN),
            str(tiny_input),
            str(tiny_output),
        )

        compare_safetensors(tiny_output, tiny_expected)
        print("OK: tiny end-to-end fixture")

        large_input = tmp / "chunk_boundary_input.safetensors"
        large_output = tmp / "chunk_boundary_output.safetensors"

        make_chunk_boundary_fixture(large_input, blocks=4097)

        run(
            str(BIN),
            str(large_input),
            str(large_output),
        )

        validate_chunk_boundary_output(large_output, blocks=4097)
        print("OK: 4097-block chunk-boundary fixture")

        invalid_scales_input = tmp / "invalid_scales_input.safetensors"
        invalid_scales_output = tmp / "invalid_scales_output.safetensors"
        make_invalid_fixture(
            invalid_scales_input,
            weight_blocks=2,
            scale_count=1,
            global_scale_raw=struct.pack("<f", 1.0),
        )
        run_expect_failure(
            str(BIN),
            str(invalid_scales_input),
            str(invalid_scales_output),
            expected_error="InvalidBlockScaleCount",
        )
        print("OK: invalid block-scale count is rejected")

        invalid_global_input = tmp / "invalid_global_input.safetensors"
        invalid_global_output = tmp / "invalid_global_output.safetensors"
        make_invalid_fixture(
            invalid_global_input,
            weight_blocks=1,
            scale_count=1,
            global_scale_raw=b"\x00\x00",
        )
        run_expect_failure(
            str(BIN),
            str(invalid_global_input),
            str(invalid_global_output),
            expected_error="InvalidGlobalScale",
        )
        print("OK: invalid global-scale size is rejected")

    print("\nAll integration tests passed.")


if __name__ == "__main__":
    main()
