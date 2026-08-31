// AI generated tests for the project
const std = @import("std");
const testing = std.testing;

const nvfp4 = @import("nvfp4");
const st = @import("safetensors");

const tiny_input = @embedFile("fixtures/tiny_nvfp4.safetensors");

fn expectSimdMatchesScalar(
    packed_bytes: *const [nvfp4.packed_block_size]u8,
    block_scale_bits: u8,
    global_scale: f32,
) !void {
    // Different initial values ensure the SIMD implementation writes every lane.
    var scalar = [_]f16{123.0} ** nvfp4.block_size;
    var simd = [_]f16{-456.0} ** nvfp4.block_size;

    nvfp4.dequantizeBlockF16(packed_bytes, block_scale_bits, global_scale, &scalar);
    nvfp4.dequantizeBlockF16Simd(packed_bytes, block_scale_bits, global_scale, &simd);

    for (scalar, simd, 0..) |expected, actual, lane| {
        const matches = if (std.math.isNan(expected))
            std.math.isNan(actual)
        else
            @as(u16, @bitCast(expected)) == @as(u16, @bitCast(actual));

        if (!matches) {
            std.debug.print(
                "SIMD mismatch: packed={any}, block_scale=0x{x:0>2}, global_scale={}, lane={}, scalar={} (0x{x:0>4}), simd={} (0x{x:0>4})\n",
                .{
                    packed_bytes.*,
                    block_scale_bits,
                    global_scale,
                    lane,
                    expected,
                    @as(u16, @bitCast(expected)),
                    actual,
                    @as(u16, @bitCast(actual)),
                },
            );
            return error.TestExpectedEqual;
        }
    }
}

test "E2M1 decodes all 16 codes" {
    const expected = [_]f32{
        0.0,  0.5,  1.0,  1.5,
        2.0,  3.0,  4.0,  6.0,
        0.0,  -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    };

    inline for (0..16) |i| {
        try testing.expectEqual(
            expected[i],
            nvfp4.decodeE2M1(@intCast(i)),
        );
    }
}

test "E4M3FN known values" {
    try testing.expectEqual(@as(f32, 0.0), nvfp4.decodeE4M3(0x00));
    try testing.expectEqual(@as(f32, 0.5), nvfp4.decodeE4M3(0x30));
    try testing.expectEqual(@as(f32, 1.0), nvfp4.decodeE4M3(0x38));
    try testing.expectEqual(@as(f32, 2.0), nvfp4.decodeE4M3(0x40));
    try testing.expectEqual(@as(f32, 448.0), nvfp4.decodeE4M3(0x7e));

    try testing.expect(std.math.isNan(nvfp4.decodeE4M3(0x7f)));
    try testing.expect(std.math.isNan(nvfp4.decodeE4M3(0xff)));
}

test "E4M3FN sign is decoded" {
    try testing.expectEqual(@as(f32, -1.0), nvfp4.decodeE4M3(0xb8));
    try testing.expectEqual(@as(f32, -0.5), nvfp4.decodeE4M3(0xb0));
}

test "one NVFP4 block dequantizes correctly" {
    // Low nibble is decoded first.
    // These bytes therefore represent codes 0,1,2,...,15.
    const encoded = [_]u8{
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xba, 0xdc, 0xfe,
    };

    var actual: [16]f16 = undefined;

    // 0x38 is E4M3FN 1.0; global scale is also 1.0.
    nvfp4.dequantizeBlockF16(
        &encoded,
        0x38,
        1.0,
        &actual,
    );

    const expected = [_]f16{
        0.0,  0.5,  1.0,  1.5,
        2.0,  3.0,  4.0,  6.0,
        0.0,  -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    };

    try testing.expectEqualSlices(f16, &expected, &actual);
}

test "SIMD block dequantizer preserves canonical lane ordering" {
    // Low nibbles are decoded before high nibbles for every packed byte.
    const encoded = [_]u8{
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xba, 0xdc, 0xfe,
    };
    var actual: [nvfp4.block_size]f16 = undefined;

    nvfp4.dequantizeBlockF16Simd(&encoded, 0x38, 1.0, &actual);

    const expected = [_]f16{
        0.0,  0.5,  1.0,  1.5,
        2.0,  3.0,  4.0,  6.0,
        0.0,  -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    };
    try testing.expectEqualSlices(f16, &expected, &actual);
}

test "SIMD block matches scalar for every byte value in every packed position" {
    for (0..nvfp4.packed_block_size) |position| {
        for (0..256) |byte_value| {
            var packed_bytes = [_]u8{0xa5} ** nvfp4.packed_block_size;
            packed_bytes[position] = @intCast(byte_value);

            try expectSimdMatchesScalar(&packed_bytes, 0x38, 1.0);
        }
    }
}

test "SIMD block matches scalar for every E4M3 block scale" {
    const patterns = [_][nvfp4.packed_block_size]u8{
        [_]u8{0x00} ** nvfp4.packed_block_size,
        [_]u8{0xff} ** nvfp4.packed_block_size,
        .{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe },
        .{ 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01 },
        .{ 0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0xf0 },
        .{ 0x87, 0x96, 0xa5, 0xb4, 0xc3, 0xd2, 0xe1, 0x0f },
    };
    const global_scales = [_]f32{ 0.0, -0.0, 0.125, 1.0, -0.75, 16.0 };

    for (patterns) |packed_bytes| {
        for (global_scales) |global_scale| {
            for (0..256) |scale_bits| {
                try expectSimdMatchesScalar(
                    &packed_bytes,
                    @intCast(scale_bits),
                    global_scale,
                );
            }
        }
    }
}

test "SIMD block matches scalar for global scale edge cases" {
    const packed_bytes = [_]u8{ 0x10, 0x8f, 0x27, 0x9e, 0x34, 0xab, 0x56, 0xcd };
    const block_scales = [_]u8{
        0x00, // +0
        0x01, // smallest positive E4M3 subnormal
        0x30, // +0.5
        0x38, // +1
        0x7e, // largest positive finite E4M3 value
        0x80, // -0
        0x81, // smallest negative E4M3 subnormal
        0xb8, // -1
        0xfe, // largest negative finite E4M3 value
        0x7f, // +NaN
        0xff, // -NaN
    };
    const global_scales = [_]f32{
        0.0,
        -0.0,
        0x1p-149,
        0x1p-126,
        0.125,
        1.0,
        -1.0,
        65504.0,
        -65504.0,
        std.math.inf(f32),
        -std.math.inf(f32),
        std.math.nan(f32),
    };

    for (block_scales) |block_scale_bits| {
        for (global_scales) |global_scale| {
            try expectSimdMatchesScalar(&packed_bytes, block_scale_bits, global_scale);
        }
    }
}

test "SIMD block matches scalar for deterministic mixed blocks" {
    const global_scales = [_]f32{ 0.03125, 0.5, 1.0, -0.25, -2.0, 32.0 };
    var state: u64 = 0x4d595df4d0f33173;

    for (0..4096) |case_index| {
        var packed_bytes: [nvfp4.packed_block_size]u8 = undefined;
        for (&packed_bytes) |*byte| {
            state = state *% 6364136223846793005 +% 1442695040888963407;
            byte.* = @truncate(state >> 32);
        }

        state = state *% 6364136223846793005 +% 1442695040888963407;
        const block_scale_bits: u8 = @truncate(state >> 40);
        const global_scale = global_scales[case_index % global_scales.len];

        try expectSimdMatchesScalar(&packed_bytes, block_scale_bits, global_scale);
    }
}

test "chunk kernel dequantizes multiple blocks" {
    const encoded = [_]u8{
        // codes 0..15
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xba, 0xdc, 0xfe,

        // codes 15..0
        0xef, 0xcd, 0xab, 0x89,
        0x67, 0x45, 0x23, 0x01,
    };

    // block scales: 1.0 and 0.5
    const scales = [_]u8{ 0x38, 0x30 };

    var actual: [32]f16 = undefined;

    try nvfp4.dequantizeBlocksF16(
        &encoded,
        &scales,
        1.0,
        &actual,
    );

    const base = [_]f16{
        0.0,  0.5,  1.0,  1.5,
        2.0,  3.0,  4.0,  6.0,
        0.0,  -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    };

    var expected: [32]f16 = undefined;

    for (0..16) |i| {
        expected[i] = base[i];
        expected[16 + i] = base[15 - i] * @as(f16, 0.5);
    }

    try testing.expectEqualSlices(f16, &expected, &actual);
}

test "chunk kernel validates lengths" {
    const encoded_bad = [_]u8{0} ** 9;
    const one_scale = [_]u8{0x38};
    var output: [16]f16 = undefined;

    try testing.expectError(
        error.InvalidPackedLength,
        nvfp4.dequantizeBlocksF16(
            &encoded_bad,
            &one_scale,
            1.0,
            &output,
        ),
    );

    const encoded_one_block = [_]u8{0} ** 8;
    const two_scales = [_]u8{ 0x38, 0x38 };

    try testing.expectError(
        error.ScaleCountMismatch,
        nvfp4.dequantizeBlocksF16(
            &encoded_one_block,
            &two_scales,
            1.0,
            &output,
        ),
    );

    var output_bad: [15]f16 = undefined;

    try testing.expectError(
        error.OutputLengthMismatch,
        nvfp4.dequantizeBlocksF16(
            &encoded_one_block,
            &one_scale,
            1.0,
            &output_bad,
        ),
    );
}

test "parse tiny SafeTensors header" {
    var reader = std.Io.Reader.fixed(tiny_input);

    var header = try st.Header.parse(
        testing.allocator,
        &reader,
    );
    defer header.deinit();

    try testing.expectEqual(@as(u64, 296), header.data_start);
    try testing.expectEqual(@as(usize, 4), header.tensors.len);

    const weight =
        header.find("linear.weight") orelse return error.TestExpectedWeight;

    try testing.expectEqual(st.DType.u8, weight.dtype);
    try testing.expectEqualSlices(
        usize,
        &[_]usize{ 2, 8 },
        weight.shape,
    );
    try testing.expectEqual(@as(u64, 0), weight.begin);
    try testing.expectEqual(@as(u64, 16), weight.end);

    const block_scale =
        header.find("linear.weight_scale") orelse return error.TestExpectedBlockScale;

    try testing.expectEqual(st.DType.f8_e4m3, block_scale.dtype);
    try testing.expectEqualSlices(
        usize,
        &[_]usize{ 2, 1 },
        block_scale.shape,
    );
    try testing.expectEqual(@as(u64, 16), block_scale.begin);
    try testing.expectEqual(@as(u64, 18), block_scale.end);

    const global_scale =
        header.find("linear.weight_scale_2") orelse return error.TestExpectedGlobalScale;

    try testing.expectEqual(st.DType.f32, global_scale.dtype);
    try testing.expectEqual(@as(u64, 18), global_scale.begin);
    try testing.expectEqual(@as(u64, 22), global_scale.end);

    const bias =
        header.find("linear.bias") orelse return error.TestExpectedBias;

    try testing.expectEqual(st.DType.f16, bias.dtype);
    try testing.expectEqualSlices(
        usize,
        &[_]usize{2},
        bias.shape,
    );
    try testing.expectEqual(@as(u64, 22), bias.begin);
    try testing.expectEqual(@as(u64, 26), bias.end);

    try testing.expect(header.find("does.not.exist") == null);
}

test "SafeTensors header write is parseable and preserves metadata" {
    const weight_shape = [_]usize{ 2, 16 };
    const bias_shape = [_]usize{2};

    const infos = [_]st.TensorInfo{
        .{
            .name = "linear.weight",
            .dtype = .f16,
            .shape = &weight_shape,
            .begin = 0,
            .end = 64,
        },
        .{
            .name = "linear.bias",
            .dtype = .f16,
            .shape = &bias_shape,
            .begin = 64,
            .end = 68,
        },
    };

    var header = try st.Header.fromInfos(
        testing.allocator,
        &infos,
    );
    defer header.deinit();

    var output: std.Io.Writer.Allocating =
        .init(testing.allocator);
    defer output.deinit();

    // Recommended final signature:
    // Header.write(temporary_allocator, writer)
    try header.write(
        testing.allocator,
        &output.writer,
    );

    var reader = std.Io.Reader.fixed(output.written());

    var reparsed = try st.Header.parse(
        testing.allocator,
        &reader,
    );
    defer reparsed.deinit();

    try testing.expectEqual(@as(usize, 2), reparsed.tensors.len);

    const weight =
        reparsed.find("linear.weight") orelse return error.TestExpectedWeight;

    try testing.expectEqual(st.DType.f16, weight.dtype);
    try testing.expectEqualSlices(usize, &weight_shape, weight.shape);
    try testing.expectEqual(@as(u64, 0), weight.begin);
    try testing.expectEqual(@as(u64, 64), weight.end);

    const bias =
        reparsed.find("linear.bias") orelse return error.TestExpectedBias;

    try testing.expectEqual(st.DType.f16, bias.dtype);
    try testing.expectEqualSlices(usize, &bias_shape, bias.shape);
    try testing.expectEqual(@as(u64, 64), bias.begin);
    try testing.expectEqual(@as(u64, 68), bias.end);
}

test "dtype names, sizes, and tensor byte size" {
    const dtypes = [_]st.DType{ .u8, .f8_e4m3, .f16, .f32, .bf16 };
    const names = [_][]const u8{ "U8", "F8_E4M3", "F16", "F32", "BF16" };
    const sizes = [_]usize{ 1, 1, 2, 4, 2 };

    for (dtypes, names, sizes) |dtype, name, size| {
        try testing.expectEqualStrings(name, dtype.toString());
        try testing.expectEqual(size, dtype.size());
    }

    const info = st.TensorInfo{
        .name = "tensor",
        .dtype = .f16,
        .shape = &[_]usize{4},
        .begin = 12,
        .end = 20,
    };
    try testing.expectEqual(@as(u64, 8), info.byteSize());
}

test "E4M3FN decodes subnormals and signed zero" {
    try testing.expectEqual(@as(f32, 0.001953125), nvfp4.decodeE4M3(0x01));
    try testing.expectEqual(@as(f32, -0.001953125), nvfp4.decodeE4M3(0x81));
    try testing.expect(!std.math.signbit(nvfp4.decodeE4M3(0x00)));
    try testing.expect(std.math.signbit(nvfp4.decodeE4M3(0x80)));
}

test "one NVFP4 block applies block and global scales" {
    const encoded = [_]u8{0x21} ** nvfp4.packed_block_size;
    var actual: [nvfp4.block_size]f16 = undefined;

    // E4M3 0x30 is 0.5, so the combined scale is 0.5 * 0.25 = 0.125.
    nvfp4.dequantizeBlockF16(&encoded, 0x30, 0.25, &actual);

    const expected_pair = [_]f16{ 0.0625, 0.125 };
    for (0..nvfp4.packed_block_size) |i| {
        try testing.expectEqual(expected_pair[0], actual[2 * i]);
        try testing.expectEqual(expected_pair[1], actual[2 * i + 1]);
    }
}

test "chunk kernel accepts empty input" {
    const encoded = [_]u8{};
    const scales = [_]u8{};
    var output: [0]f16 = .{};

    try nvfp4.dequantizeBlocksF16(&encoded, &scales, 1.0, &output);
}

test "stream kernel writes a dequantized block" {
    const encoded = [_]u8{
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xba, 0xdc, 0xfe,
    };
    const scales = [_]u8{0x38};
    var weights_reader = std.Io.Reader.fixed(&encoded);
    var scales_reader = std.Io.Reader.fixed(&scales);
    var output: std.Io.Writer.Allocating = .init(testing.allocator);
    defer output.deinit();

    try nvfp4.dequantizeStreamF16(
        &weights_reader,
        &scales_reader,
        1.0,
        &output.writer,
        nvfp4.block_size,
    );

    const expected = [_]f16{
        0.0,  0.5,  1.0,  1.5,
        2.0,  3.0,  4.0,  6.0,
        0.0,  -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    };
    try testing.expectEqualSlices(u8, std.mem.asBytes(&expected), output.written());
}

test "stream kernel validates value count and short input" {
    var empty_weights = std.Io.Reader.fixed(&.{});
    var empty_scales = std.Io.Reader.fixed(&.{});
    var output: std.Io.Writer.Allocating = .init(testing.allocator);
    defer output.deinit();

    try testing.expectError(
        error.EmptyStream,
        nvfp4.dequantizeStreamF16(&empty_weights, &empty_scales, 1.0, &output.writer, 0),
    );
    try testing.expectError(
        error.InvalidBlockCount,
        nvfp4.dequantizeStreamF16(&empty_weights, &empty_scales, 1.0, &output.writer, 15),
    );

    const short_encoded = [_]u8{0} ** (nvfp4.packed_block_size - 1);
    const one_scale = [_]u8{0x38};
    var short_weights = std.Io.Reader.fixed(&short_encoded);
    var one_scale_reader = std.Io.Reader.fixed(&one_scale);
    try testing.expectError(
        error.EndOfStream,
        nvfp4.dequantizeStreamF16(
            &short_weights,
            &one_scale_reader,
            1.0,
            &output.writer,
            nvfp4.block_size,
        ),
    );
}

test "SafeTensors parser ignores metadata and supports BF16" {
    const json =
        "{\"__metadata__\":{\"format\":\"pt\"},\"tensor\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
    var bytes: [8 + json.len]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    std.mem.copyForwards(u8, bytes[8..], json);

    var reader = std.Io.Reader.fixed(&bytes);
    var header = try st.Header.parse(testing.allocator, &reader);
    defer header.deinit();

    try testing.expectEqual(@as(usize, 1), header.tensors.len);
    const tensor = header.find("tensor") orelse return error.TestExpectedTensor;
    try testing.expectEqual(st.DType.bf16, tensor.dtype);
    try testing.expectEqualSlices(usize, &[_]usize{2}, tensor.shape);
    try testing.expectEqual(@as(u64, 4), tensor.byteSize());
    try testing.expect(header.find("__metadata__") == null);
}

test "SafeTensors parser rejects an unsupported dtype" {
    const json =
        "{\"tensor\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var bytes: [8 + json.len]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    std.mem.copyForwards(u8, bytes[8..], json);

    var reader = std.Io.Reader.fixed(&bytes);
    try testing.expectError(error.UnsupportedDType, st.Header.parse(testing.allocator, &reader));
}

test "SafeTensors parser rejects a truncated header" {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, 1, .little);

    var reader = std.Io.Reader.fixed(&bytes);
    try testing.expectError(error.EndOfStream, st.Header.parse(testing.allocator, &reader));
}
