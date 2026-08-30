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
    //TODO: add bf16

    pub fn size(self: DType) usize {
        return switch (self) {
            .u8 => 1,
            .f8_e4m3 => 1,
            .f16 => 2,
            .f32 => 4,
        };
    }

    pub fn toString(self: DType) []const u8 {
        return switch (self) {
            .u8 => "U8",
            .f8_e4m3 => "F8_E4M3",
            .f16 => "F16",
            .f32 => "F32",
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
        //TODO: fix this function to use the updated Zig JSON API

        // Create arena for allocating metadata
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const a = arena.allocator();

        // Read u64 header length
        var header_length_bytes: [8]u8 = undefined;
        try reader.readAll(&header_length_bytes);
        const header_length = std.mem.readInt(u64, &header_length_bytes, .little);

        // Read JSON
        const header_end = Header.header_begin + header_length;
        const header_bytes: []u8 = try a.alloc(u8, header_length);
        defer a.free(header_bytes);
        try reader.readAll(header_bytes);

        // Parse JSON
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
        defer parsed.deinit();

        // Count tensor entries
        var count: usize = 0;

        for (parsed.Object().entries()) |entry| {
            // Ignore metadata entries
            if (std.mem.eql(u8, entry.key, "__metadata__")) continue;

            count += 1;
        }

        // Construct TensorInfo array
        var tensors: []TensorInfo = try a.alloc(TensorInfo, count);

        var index: usize = 0;
        for (parsed.Object().entries()) |entry| {
            // Ignore metadata entries
            if (std.mem.eql(u8, entry.key, "__metadata__")) continue;
            const name = try a.dupe(u8, entry.key);
            const tensor_json = entry.value;
            const dtype_str = try tensor_json.Object().get("dtype").String();
            const dtype = try parseDType(dtype_str);
            const shape_json = try tensor_json.Object().get("shape");
            var shape: []usize = try a.alloc(usize, shape_json.Array().len());
            for (shape_json.Array(), 0..) |dim, j| {
                shape[j] = dim.Int();
            }
            const begin = try tensor_json.Object().get("data_offsets").Array()[0].Int();
            const end = try tensor_json.Object().get("data_offsets").Array()[1].Int();
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

    pub fn write(self: *const Header, writer: *std.Io.Writer) !void {
        // Writes the header to the given writer in .safetensors format
        var json_obj = std.json.Object.init(self.arena.allocator());
        defer json_obj.deinit();

        // We should probably leave a placeholder and at the end of the writing process, go back and write the actual header length (using Header.header_begin maybe).

        for (self.tensors) |tensor| {
            var tensor_obj = std.json.Object.init(self.arena.allocator());
            defer tensor_obj.deinit();

            try tensor_obj.put("dtype", std.json.Value.fromString(tensor.dtype.toString()));
            var shape_array = std.json.Array.init(self.arena.allocator());
            defer shape_array.deinit();
            for (tensor.shape) |dim| {
                try shape_array.append(std.json.Value.fromInt(dim));
            }
            try tensor_obj.put("shape", std.json.Value.fromArray(&shape_array));
            var offsets_array = std.json.Array.init(self.arena.allocator());
            defer offsets_array.deinit();
            try offsets_array.append(std.json.Value.fromInt(tensor.begin));
            try offsets_array.append(std.json.Value.fromInt(tensor.end));
            try tensor_obj.put("data_offsets", std.json.Value.fromArray(&offsets_array));

            try json_obj.put(tensor.name, std.json.Value.fromObject(&tensor_obj));
        }

        const json_str = try json_obj.toString();
        const json_bytes = json_str.?; // Check this syntax
        const header_length: u64 = @intCast(json_bytes.len);
        var header_length_bytes: [Header.header_begin]u8 = undefined;
        std.mem.writeInt(u64, &header_length_bytes, header_length, .little);

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

    return error.UnsupportedDType;
}
