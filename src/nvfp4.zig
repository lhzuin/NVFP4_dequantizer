// Implements the nvfp4 specific logic, by receiving the packed bytes and scales and returning the values in the desired precision
// Will include SIMD logic as a future improvement

const std = @import("std");

pub const block_size = 16;
pub const packed_block_size = 8;

const e2m1_values = [_]f32{
    0.0, 0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

pub fn decodeE2M1(code: u4) f32 {
    // Receives single unsigned integer (4 bits) and returns float associated to it
    return e2m1_values[code];
}

pub fn decodeE4M3(bits: u8) f32 {
    // Decodes a byte in the format s eeee mmm
    // (-1)^S * 2^(E-bias) * (1 + M/8) for E <> 0
    // (-1)^S * 2^(1-bias) * (M / 8) for E == 0

    const exponent_bias: i8 = 7;
    const mantissa: u3 = @truncate(bits);
    const exponent: u4 = @truncate(bits >> 3);
    const negative: bool = (bits >> 7) != 0;

    // E4M3FN NaN: s 1111 111
    if (exponent == 0b1111 and mantissa == 0b111) {
        return std.math.nan(f32);
    }

    const mantissa_float: f32 = @floatFromInt(mantissa);
    var value: f32 = undefined;

    if (exponent == 0) {
        const unbiased_exponent: i8 = 1 - exponent_bias;
        value = @exp2(@as(f32, @floatFromInt(unbiased_exponent))) * (mantissa_float / 8.0);
    } else {
        const unbiased_exponent: i8 = @as(i8, exponent) - exponent_bias;
        value = @exp2(@as(f32, @floatFromInt(unbiased_exponent))) * (1.0 + mantissa_float / 8.0);
    }

    if (negative) {
        value = -value;
    }
    return value;
}

pub fn dequantizeBlockF16(
    packed_bytes: *const [8]u8,
    block_scale_bits: u8,
    global_scale: f32,
    out: *[16]f16,
) void {
    // Dequantizes an entire block by decoding its scale, and then unpacking and dequantizing all the 16 values inside it.
    const local_scale = decodeE4M3(block_scale_bits) * global_scale;
    for (0..packed_block_size) |i| {
        // Unpacks 2 nibbles inside a byte and dequantizes them
        const low: u4 = @truncate(packed_bytes[i]);
        const high: u4 = @truncate(packed_bytes[i] >> 4);
        out[2 * i] = @floatCast(decodeE2M1(low) * local_scale);
        out[2 * i + 1] = @floatCast(decodeE2M1(high) * local_scale);
    }
}

pub fn dequantizeBlocksF16(
    packed_bytes: []const u8,
    scales: []const u8,
    global_scale: f32,
    output: []f16,
) !void {
    // Dequantizes a stream of blocks by reading the packed bytes and scales, and writing the dequantized values to the output slice.
    if (packed_bytes.len % packed_block_size != 0) {
        return error.InvalidPackedLength;
    }
    const num_blocks = packed_bytes.len / packed_block_size;
    if (scales.len != num_blocks) {
        return error.ScaleCountMismatch;
    }
    if (output.len != num_blocks * block_size) {
        return error.OutputLengthMismatch;
    }

    for (0..num_blocks) |block_index| {
        const packed_block = packed_bytes[block_index * packed_block_size .. (block_index + 1) * packed_block_size];
        const block_scale_bits = scales[block_index];
        var dequantized_block: [16]f16 = undefined;
        dequantizeBlockF16(&packed_block, block_scale_bits, global_scale, &dequantized_block);
        std.mem.copy(f16, output[block_index * block_size .. (block_index + 1) * block_size], dequantized_block);
    }
}

pub fn dequantizeStreamF16(
    weights: *std.Io.Reader,
    scales: *std.Io.Reader,
    global_scale: f32,
    writer: *std.Io.Writer,
    num_values: usize,
) !void {
    // Dequantizes the entire stream of weights, reading the packed bytes and scales, and writing the dequantized values to the output writer.
    if (num_values == 0) {
        return error.EmptyStream;
    }
    if (num_values % block_size != 0) {
        return error.InvalidBlockCount;
    }
    const num_blocks = num_values / block_size;
    for (0..num_blocks) |_| {
        // Read 8 weight bytes
        var packed_block: [8]u8 = undefined;
        try weights.readSliceAll(&packed_block);

        // Read 1 scale byte
        var block_scale_bits: [1]u8 = undefined;
        try scales.readSliceAll(&block_scale_bits);

        // Dequantize block
        var dequantized_block: [16]f16 = undefined;
        dequantizeBlockF16(&packed_block, block_scale_bits[0], global_scale, &dequantized_block);

        // Write dequantized block
        const write_bits = std.mem.asBytes(&dequantized_block);
        try writer.writeAll(write_bits);
    }
}
