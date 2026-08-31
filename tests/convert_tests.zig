// AI generated tests for the project
const std = @import("std");
const testing = std.testing;

const convert = @import("convert");

const tiny_input = @embedFile("fixtures/tiny_nvfp4.safetensors");
const tiny_expected = @embedFile("fixtures/tiny_expected_f16.safetensors");

const GeneratedFixtureHeader = struct {
    blocks: usize,
    weight_end: u64,
    scale_end: u64,
    global_scale_end: u64,
    bias_end: u64,

    pub fn jsonStringify(self: GeneratedFixtureHeader, json: anytype) !void {
        try json.beginObject();

        try json.objectField("linear.weight");
        try json.beginObject();
        try json.objectField("dtype");
        try json.write("U8");
        try json.objectField("shape");
        try json.write([_]usize{ self.blocks, 8 });
        try json.objectField("data_offsets");
        try json.write([_]u64{ 0, self.weight_end });
        try json.endObject();

        try json.objectField("linear.weight_scale");
        try json.beginObject();
        try json.objectField("dtype");
        try json.write("F8_E4M3");
        try json.objectField("shape");
        try json.write([_]usize{self.blocks});
        try json.objectField("data_offsets");
        try json.write([_]u64{ self.weight_end, self.scale_end });
        try json.endObject();

        try json.objectField("linear.weight_scale_2");
        try json.beginObject();
        try json.objectField("dtype");
        try json.write("F32");
        try json.objectField("shape");
        try json.write([_]usize{});
        try json.objectField("data_offsets");
        try json.write([_]u64{ self.scale_end, self.global_scale_end });
        try json.endObject();

        try json.objectField("linear.bias");
        try json.beginObject();
        try json.objectField("dtype");
        try json.write("F16");
        try json.objectField("shape");
        try json.write([_]usize{3});
        try json.objectField("data_offsets");
        try json.write([_]u64{ self.global_scale_end, self.bias_end });
        try json.endObject();

        try json.endObject();
    }
};

const ParsedSafetensors = struct {
    header: std.json.Parsed(std.json.Value),
    data: []const u8,

    fn deinit(self: *ParsedSafetensors) void {
        self.header.deinit();
    }
};

fn parseSafetensors(bytes: []const u8) !ParsedSafetensors {
    if (bytes.len < 8) return error.TruncatedSafetensors;

    const header_length: usize = std.math.cast(
        usize,
        std.mem.readInt(u64, bytes[0..8], .little),
    ) orelse return error.HeaderTooLarge;
    const data_start = std.math.add(usize, 8, header_length) catch
        return error.HeaderTooLarge;
    if (data_start > bytes.len) return error.TruncatedSafetensors;

    return .{
        .header = try std.json.parseFromSlice(
            std.json.Value,
            testing.allocator,
            bytes[8..data_start],
            .{},
        ),
        .data = bytes[data_start..],
    };
}

fn expectJsonEqual(expected: std.json.Value, actual: std.json.Value) !void {
    try testing.expectEqual(std.meta.activeTag(expected), std.meta.activeTag(actual));

    switch (expected) {
        .null => {},
        .bool => |value| try testing.expectEqual(value, actual.bool),
        .integer => |value| try testing.expectEqual(value, actual.integer),
        .float => |value| try testing.expectEqual(value, actual.float),
        .number_string => |value| try testing.expectEqualStrings(value, actual.number_string),
        .string => |value| try testing.expectEqualStrings(value, actual.string),
        .array => |array| {
            try testing.expectEqual(array.items.len, actual.array.items.len);
            for (array.items, actual.array.items) |expected_item, actual_item| {
                try expectJsonEqual(expected_item, actual_item);
            }
        },
        .object => |object| {
            try testing.expectEqual(object.count(), actual.object.count());
            var iterator = object.iterator();
            while (iterator.next()) |entry| {
                const actual_value = actual.object.get(entry.key_ptr.*) orelse
                    return error.TestExpectedJsonField;
                try expectJsonEqual(entry.value_ptr.*, actual_value);
            }
        },
    }
}

fn expectSafetensorsEqual(actual_bytes: []const u8, expected_bytes: []const u8) !void {
    var actual = try parseSafetensors(actual_bytes);
    defer actual.deinit();
    var expected = try parseSafetensors(expected_bytes);
    defer expected.deinit();

    try expectJsonEqual(expected.header.value, actual.header.value);
    try testing.expectEqualSlices(u8, expected.data, actual.data);
}

fn fixturePath(allocator: std.mem.Allocator, tmp: *const testing.TmpDir, filename: []const u8) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/{s}",
        .{ tmp.sub_path, filename },
    );
}

fn makeGeneratedFixture(allocator: std.mem.Allocator, blocks: usize) ![]u8 {
    const weight_len = try std.math.mul(usize, blocks, 8);
    const scale_len = blocks;
    const global_scale_len = 4;
    const bias = [_]u8{ 0x00, 0x3c, 0x00, 0xc0, 0x00, 0x38 };

    const weight_end: u64 = @intCast(weight_len);
    const scale_end = weight_end + @as(u64, @intCast(scale_len));
    const global_scale_end = scale_end + global_scale_len;
    const bias_end = global_scale_end + bias.len;

    const header = try std.json.Stringify.valueAlloc(
        allocator,
        GeneratedFixtureHeader{
            .blocks = blocks,
            .weight_end = weight_end,
            .scale_end = scale_end,
            .global_scale_end = global_scale_end,
            .bias_end = bias_end,
        },
        .{},
    );
    defer allocator.free(header);

    const data_len: usize = @intCast(bias_end);
    const result = try allocator.alloc(u8, 8 + header.len + data_len);
    errdefer allocator.free(result);

    std.mem.writeInt(u64, result[0..8], @intCast(header.len), .little);
    std.mem.copyForwards(u8, result[8 .. 8 + header.len], header);

    const data = result[8 + header.len ..];
    const weight_bytes = data[0..weight_len];
    for (weight_bytes, 0..) |*byte, i| {
        byte.* = @truncate(i *% 37 +% i / 7 +% 11);
    }

    const scale_values = [_]u8{ 0x01, 0x20, 0x30, 0x38, 0x40, 0x48, 0xb0, 0xb8, 0xc0, 0x7e };
    const scale_bytes = data[weight_len .. weight_len + scale_len];
    for (scale_bytes, 0..) |*scale, i| {
        scale.* = scale_values[i % scale_values.len];
    }

    const global_scale_offset = weight_len + scale_len;
    std.mem.writeInt(
        u32,
        data[global_scale_offset..][0..4],
        @bitCast(@as(f32, 0.75)),
        .little,
    );
    std.mem.copyForwards(u8, data[global_scale_offset + 4 ..], &bias);

    return result;
}

fn convertFixture(
    tmp: *const testing.TmpDir,
    input_name: []const u8,
    output_name: []const u8,
    options: convert.Options,
) ![]u8 {
    const input_path = try fixturePath(testing.allocator, tmp, input_name);
    defer testing.allocator.free(input_path);
    const output_path = try fixturePath(testing.allocator, tmp, output_name);
    defer testing.allocator.free(output_path);

    try convert.dequantizeFile(
        testing.io,
        testing.allocator,
        input_path,
        output_path,
        options,
    );

    return tmp.dir.readFileAlloc(
        testing.io,
        output_name,
        testing.allocator,
        .unlimited,
    );
}

test "file converter matches the expected fixture with one-block chunks" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "input.safetensors",
        .data = tiny_input,
    });

    const input_path = try fixturePath(testing.allocator, &tmp, "input.safetensors");
    defer testing.allocator.free(input_path);
    const output_path = try fixturePath(testing.allocator, &tmp, "output.safetensors");
    defer testing.allocator.free(output_path);

    try convert.dequantizeFile(testing.io, testing.allocator, input_path, output_path, .{
        .chunk_blocks = 1,
    });

    const actual = try tmp.dir.readFileAlloc(
        testing.io,
        "output.safetensors",
        testing.allocator,
        .unlimited,
    );
    defer testing.allocator.free(actual);

    try expectSafetensorsEqual(actual, tiny_expected);
}

test "file converter rejects a zero chunk size" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "input.safetensors",
        .data = tiny_input,
    });

    const input_path = try fixturePath(testing.allocator, &tmp, "input.safetensors");
    defer testing.allocator.free(input_path);
    const output_path = try fixturePath(testing.allocator, &tmp, "output.safetensors");
    defer testing.allocator.free(output_path);

    try testing.expectError(
        error.InvalidChunkSize,
        convert.dequantizeFile(testing.io, testing.allocator, input_path, output_path, .{
            .chunk_blocks = 0,
        }),
    );
}

test "file converter output is identical for different thread counts" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const input = try makeGeneratedFixture(testing.allocator, 67);
    defer testing.allocator.free(input);
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "thread-input.safetensors",
        .data = input,
    });

    const baseline = try convertFixture(
        &tmp,
        "thread-input.safetensors",
        "threads-1.safetensors",
        .{ .threads = 1, .chunk_blocks = 19 },
    );
    defer testing.allocator.free(baseline);

    const thread_counts = [_]usize{ 2, 3, 4, 7, 19, 64 };
    for (thread_counts) |threads| {
        var output_name_buffer: [32]u8 = undefined;
        const output_name = try std.fmt.bufPrint(
            &output_name_buffer,
            "threads-{d}.safetensors",
            .{threads},
        );
        const actual = try convertFixture(
            &tmp,
            "thread-input.safetensors",
            output_name,
            .{ .threads = threads, .chunk_blocks = 19 },
        );
        defer testing.allocator.free(actual);

        try testing.expectEqualSlices(u8, baseline, actual);
    }
}

test "threaded output is identical across chunk sizes and repeated runs" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const input = try makeGeneratedFixture(testing.allocator, 257);
    defer testing.allocator.free(input);
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "chunk-input.safetensors",
        .data = input,
    });

    const baseline = try convertFixture(
        &tmp,
        "chunk-input.safetensors",
        "chunk-baseline.safetensors",
        .{ .threads = 1, .chunk_blocks = 257 },
    );
    defer testing.allocator.free(baseline);

    const cases = [_]struct {
        threads: usize,
        chunk_blocks: usize,
    }{
        .{ .threads = 2, .chunk_blocks = 1 },
        .{ .threads = 3, .chunk_blocks = 17 },
        .{ .threads = 4, .chunk_blocks = 64 },
        .{ .threads = 7, .chunk_blocks = 31 },
        .{ .threads = 16, .chunk_blocks = 128 },
        .{ .threads = 512, .chunk_blocks = 257 },
    };

    for (cases, 0..) |case, i| {
        var output_name_buffer: [32]u8 = undefined;
        const output_name = try std.fmt.bufPrint(
            &output_name_buffer,
            "chunk-case-{d}.safetensors",
            .{i},
        );
        const actual = try convertFixture(
            &tmp,
            "chunk-input.safetensors",
            output_name,
            .{
                .threads = case.threads,
                .chunk_blocks = case.chunk_blocks,
            },
        );
        defer testing.allocator.free(actual);

        try testing.expectEqualSlices(u8, baseline, actual);
    }

    for (0..4) |run_index| {
        var output_name_buffer: [32]u8 = undefined;
        const output_name = try std.fmt.bufPrint(
            &output_name_buffer,
            "repeat-{d}.safetensors",
            .{run_index},
        );
        const actual = try convertFixture(
            &tmp,
            "chunk-input.safetensors",
            output_name,
            .{ .threads = 8, .chunk_blocks = 64 },
        );
        defer testing.allocator.free(actual);

        try testing.expectEqualSlices(u8, baseline, actual);
    }
}

test "file converter rejects a zero thread count" {
    try testing.expectError(
        error.InvalidThreadCount,
        convert.dequantizeFile(
            testing.io,
            testing.allocator,
            "unused-input.safetensors",
            "unused-output.safetensors",
            .{ .threads = 0 },
        ),
    );
}
