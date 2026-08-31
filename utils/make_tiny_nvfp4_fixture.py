from pathlib import Path
import json
import struct

# Tiny NVIDIA ModelOpt-style NVFP4 fixture:
#   linear.weight         : packed E2M1 nibbles, U8, shape [2, 8]
#   linear.weight_scale   : E4M3FN block scales, shape [2, 1]
#   linear.weight_scale_2 : global F32 scale, scalar
#   linear.bias           : ordinary F16 tensor to test passthrough
#
# Each row represents 16 original weights, i.e. one NVFP4 block.

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def pack_codes(codes):
    assert len(codes) % 2 == 0
    return bytes((codes[i] & 0x0F) | ((codes[i + 1] & 0x0F) << 4)
                 for i in range(0, len(codes), 2))

def write_safetensors(path, tensors):
    """
    tensors: list[(name, dtype_string, shape, raw_bytes)]
    Offsets are relative to the start of the data section.
    """
    header = {}
    offset = 0
    for name, dtype, shape, raw in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # SafeTensors permits trailing ASCII-space padding in the JSON header.
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _, _, _, raw in tensors:
            f.write(raw)

def main():
    here = Path(__file__).resolve().parent

    row0_codes = list(range(16))
    row1_codes = list(reversed(range(16)))
    packed_weight = pack_codes(row0_codes) + pack_codes(row1_codes)

    # E4M3FN: 0x38 = 1.0, 0x30 = 0.5.
    block_scales = bytes([0x38, 0x30])
    global_scale = struct.pack("<f", 1.0)
    bias = struct.pack("<ee", 0.25, -0.25)

    write_safetensors(
        here / "tiny_nvfp4.safetensors",
        [
            ("linear.weight", "U8", [2, 8], packed_weight),
            ("linear.weight_scale", "F8_E4M3", [2, 1], block_scales),
            ("linear.weight_scale_2", "F32", [], global_scale),
            ("linear.bias", "F16", [2], bias),
        ],
    )

    expected_row0 = E2M1
    expected_row1 = [E2M1[c] * 0.5 for c in row1_codes]
    expected_weight = b"".join(struct.pack("<e", x) for x in expected_row0 + expected_row1)

    write_safetensors(
        here / "tiny_expected_f16.safetensors",
        [
            ("linear.weight", "F16", [2, 16], expected_weight),
            ("linear.bias", "F16", [2], bias),
        ],
    )

    print("Wrote:")
    print(" ", here / "tiny_nvfp4.safetensors")
    print(" ", here / "tiny_expected_f16.safetensors")

if __name__ == "__main__":
    main()
