// Implements .safetensors specific logic (agnostic to the exact data type), including:
// - parse header;
// - represent tensor metadata;
// - find tensors;
// - build/write output header.

const std = @import("std");

pub const DType = enum {
    u8,
    f8_e4m3,
    f16,
    f32,
    bf16,

    pub fn size(self: DType) usize {
        return switch (self) {
            .u8 => 1,
            .f8_e4m3 => 1,
            .f16 => 2,
            .f32 => 4,
            .bf16 => 2,
        };
    }

    pub fn toString(self: DType) []const u8 {
        return switch (self) {
            .u8 => "U8",
            .f8_e4m3 => "F8_E4M3",
            .f16 => "F16",
            .f32 => "F32",
            .bf16 => "BF16",
        };
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    begin: u64, // Relative to the start of the data section in the .safetensors file
    end: u64,

    pub fn byteSize(self: TensorInfo) u64 {
        return self.end - self.begin;
    }
};

const HeaderJson = struct {
    tensors: []const TensorInfo,

    pub fn jsonStringify(
        self: HeaderJson,
        json: anytype,
    ) !void {
        try json.beginObject();

        for (self.tensors) |tensor| {
            try json.objectField(
                tensor.name,
            );

            try json.beginObject();

            try json.objectField("dtype");
            try json.write(
                tensor.dtype.toString(),
            );

            try json.objectField("shape");
            try json.write(tensor.shape);

            try json.objectField(
                "data_offsets",
            );

            const offsets = [_]u64{
                tensor.begin,
                tensor.end,
            };

            try json.write(offsets);

            try json.endObject();
        }

        try json.endObject();
    }
};

pub const Header = struct {
    arena: std.heap.ArenaAllocator,
    tensors: []TensorInfo,
    data_start: u64,
    const header_begin = 8;

    pub fn parse(
        allocator: std.mem.Allocator,
        reader: *std.Io.Reader,
    ) !Header {
        // Reads header for the .safetensors file and creates the Header object

        // Create arena for allocating metadata
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const a = arena.allocator();

        // Read u64 header length
        var header_length_bytes: [8]u8 = undefined;
        try reader.readSliceAll(&header_length_bytes);
        const header_length = std.mem.readInt(u64, &header_length_bytes, .little);

        // Read JSON
        const header_end = Header.header_begin + header_length;
        const header_length_usize = std.math.cast(usize, header_length) orelse return error.HeaderTooLarge;

        const header_bytes = try a.alloc(u8, header_length_usize);

        try reader.readSliceAll(header_bytes);

        // Parse JSON
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
        defer parsed.deinit();

        // Count tensor entries
        var count: usize = 0;

        const object = parsed.value.object;

        for (object.keys()) |key| {
            // Ignore metadata entries
            if (std.mem.eql(u8, key, "__metadata__")) continue;

            count += 1;
        }

        // Construct TensorInfo array
        var tensors: []TensorInfo = try a.alloc(TensorInfo, count);

        var index: usize = 0;

        for (object.keys()) |key| {
            // Ignore metadata entries
            if (std.mem.eql(u8, key, "__metadata__")) continue;
            const name = try a.dupe(u8, key);
            const tensor_json = object.get(key).?;
            const dtype_str = tensor_json.object.get("dtype").?.string;
            const dtype = try parseDType(dtype_str);

            const shape_json = tensor_json.object.get("shape").?;
            var shape: []usize = try a.alloc(usize, shape_json.array.items.len);
            for (shape_json.array.items, 0..) |dim, j| {
                shape[j] = @intCast(dim.integer);
            }
            const data_offsets = tensor_json.object.get("data_offsets").?;
            const begin: u64 = @intCast(data_offsets.array.items[0].integer);
            const end: u64 = @intCast(data_offsets.array.items[1].integer);
            tensors[index] = TensorInfo{
                .name = name,
                .dtype = dtype,
                .shape = shape,
                .begin = begin,
                .end = end,
            };
            index += 1;
        }

        // Create and return Header object
        return Header{
            .arena = arena,
            .tensors = tensors,
            .data_start = header_end,
        };
    }

    pub fn find(
        self: *const Header,
        name: []const u8,
    ) ?*const TensorInfo {
        for (self.tensors) |*tensor| {
            if (std.mem.eql(u8, tensor.name, name)) {
                return tensor;
            }
        }

        return null;
    }

    pub fn fromInfos(
        allocator: std.mem.Allocator,
        infos: []const TensorInfo,
    ) !Header {
        // Creates a Header object from an array of TensorInfo objects
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const a = arena.allocator();

        var tensors: []TensorInfo = try a.alloc(TensorInfo, infos.len);

        for (infos, 0..) |info, i| {
            tensors[i] = info;
        }

        return Header{
            .arena = arena,
            .tensors = tensors,
            .data_start = 0, // Isn't needed for writing as it happens sequentially
        };
    }

    pub fn write(self: *const Header, allocator: std.mem.Allocator, writer: *std.Io.Writer) !void {
        // Writes the header to the given writer in .safetensors format

        const json_bytes =
            try std.json.Stringify.valueAlloc(
                allocator,
                HeaderJson{
                    .tensors = self.tensors,
                },
                .{},
            );
        defer allocator.free(json_bytes);

        var header_length_bytes: [Header.header_begin]u8 = undefined; // used for storing the length of the header in bytes

        std.mem.writeInt(u64, &header_length_bytes, @intCast(json_bytes.len), .little);

        try writer.writeAll(&header_length_bytes);

        try writer.writeAll(json_bytes);
    }

    pub fn deinit(self: *Header) void {
        self.arena.deinit();
    }
};

fn parseDType(name: []const u8) !DType {
    if (std.mem.eql(u8, name, "U8"))
        return .u8;

    if (std.mem.eql(u8, name, "F8_E4M3"))
        return .f8_e4m3;

    if (std.mem.eql(u8, name, "F16"))
        return .f16;

    if (std.mem.eql(u8, name, "F32"))
        return .f32;

    if (std.mem.eql(u8, name, "BF16"))
        return .bf16;

    return error.UnsupportedDType;
}
