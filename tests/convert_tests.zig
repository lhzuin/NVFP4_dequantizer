// AI generated tests for the project
const std = @import("std");
const testing = std.testing;

const convert = @import("convert");

const tiny_input = @embedFile("fixtures/tiny_nvfp4.safetensors");
const tiny_expected = @embedFile("fixtures/tiny_expected_f16.safetensors");

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
