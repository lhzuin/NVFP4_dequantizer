// Orchestrates the entire conversion pipeline.
// Will support parallel scheduling as a future improvement
// Assumes common Nvidia ModelOpt format with foo.weight, foo.weight_scale, foo.weight_scale_2

const std = @import("std");
const st = @import("safetensors.zig");
const nvfp4 = @import("nvfp4.zig");

pub const OutputType = enum {
    f16,
    pub fn toString(self: OutputType) []const u8 {
        return switch (self) {
            .f16 => "F16",
        };
    }

    pub fn toDType(self: OutputType) st.DType {
        return switch (self) {
            .f16 => .f16,
        };
    }
};

pub const Options = struct {
    output_type: OutputType = .f16,

    // Used later:
    threads: usize = 1,
    chunk_blocks: usize = 4096,
};

pub fn dequantizeFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
    options: Options,
) !void {
    // 1. Open input
    var file = try std.Io.Dir.cwd().openFile(io, input_path, .{ .mode = .read_only });
    defer file.close(io);
    var buffer: [1024]u8 = undefined;
    var file_r: std.Io.File.Reader = file.reader(&buffer);

    // 2. Parse input header
    var header = try st.Header.parse(allocator, &file_r.interface);
    defer header.deinit();

    // 3. Build ConversionPlan
    var plan = try ConversionPlan.buildPlan(allocator, &header, options.output_type);
    defer plan.deinit();

    // 4. Create output file

    var output_file = try std.Io.Dir.cwd().createFile(
        io,
        output_path,
        .{},
    );
    defer output_file.close(io);

    // 5. Create output Writer
    var output_buffer: [4096]u8 = undefined;
    var output_file_writer = output_file.writer(io, &output_buffer);

    const writer: *std.Io.Writer = &output_file_writer.interface;

    // 6. Create new header for output file
    var output_header = try st.Header.fromInfos(allocator, plan.tensors);
    defer output_header.deinit();

    // 7. Write output SafeTensors header
    try output_header.write(writer);

    // 8. Execute ConversionPlan
    plan.executePlan(&file_r, writer) catch |err| {
        std.debug.print("Error during conversion: {}\n", .{err});
        return err;
    };

    // 9. Flush output Writer
    try output_file.flush();
    try writer.flush();
}

const OutputTensor = struct {
    // Saves the intended structure of an output tensor so that we can easily process it later.

    info: st.TensorInfo, // Output metadata.
    source: Source, // How to generate the data using the input.
};

const Nvfp4Source = struct {
    weight: *const st.TensorInfo,
    block_scale: *const st.TensorInfo,
    global_scale: *const st.TensorInfo,
};

const Source = union(enum) {
    // Determines the source of the bytes for an output tensor. It can either be a direct copy of an input tensor, or a dequantization of a quantized weight tensor.
    copy: *const st.TensorInfo,
    nvfp4: Nvfp4Source,
};

const ConversionPlan = struct {
    // Saves the entire plan for converting an input safetensors file to an output safetensors file.
    tensors: []OutputTensor,
    allocator: std.mem.Allocator,
    input_header: *const st.Header,

    pub fn deinit(self: *ConversionPlan) void {
        self.allocator.free(self.tensors);
    }

    pub fn buildPlan(
        allocator: std.mem.Allocator,
        header: *const st.Header,
        output_type: OutputType,
    ) !ConversionPlan {
        // Builds a ConversionPlan by analyzing the input header and determining how to convert each tensor to the output format.

        const num_tensors = countOutputTensors(header);

        var plan = ConversionPlan{
            .tensors = try allocator.alloc(OutputTensor, num_tensors),
            .allocator = allocator,
            .input_header = header,
        };
        var count: usize = 0;
        var byte_cursor = 0;
        for (header.tensors) |*tensor| {
            // Calculate the output tensor's begin and end offsets based on the input tensor's shape and the output type's size

            if (isNvfp4Scale(tensor)) {
                // Skip scale tensors, they are not included in the output
                continue;
            }
            //const num_elements = std.math.prod(tensor.shape);
            // const output_element_size = tensor.byteSize(); //output_type.toDType().size(); // Check this!
            // const output_byte_size = num_elements * output_element_size;
            const logical_values = tensor.byteSize() * 2;

            const output_byte_size = logical_values * @sizeOf(f16); // Not sure if I should hardcode f16 here. Check
            const output_begin = byte_cursor;
            const output_end = output_begin + output_byte_size;
            byte_cursor = output_end;

            if (try matchNvfp4Weight(allocator, header, tensor)) |nvfp4_source| {
                // Create dequantized output tensor entry in the plan
                plan.tensors[count] = OutputTensor{ .info = .{ .name = tensor.name, .dtype = output_type.toDType(), .shape = tensor.shape, .begin = output_begin, .end = output_end }, .source = Source{ .nvfp4 = .{
                    .weight = tensor,
                    .block_scale = nvfp4_source.block_scale,
                    .global_scale = nvfp4_source.global_scale,
                } } };
                count += 1;
            } else {
                // For other tensors (like biases), we will just copy them to the output
                plan.tensors[count] = OutputTensor{
                    .info = .{ .name = tensor.name, .dtype = tensor.dtype, .shape = tensor.shape, .begin = output_begin, .end = output_end },
                    .source = Source{ .copy = tensor },
                };
                count += 1;
            }
        }
        return plan;
    }

    pub fn executePlan(
        self: *ConversionPlan,
        file_r: *std.Io.File.Reader,
        writer: *std.Io.Writer,
    ) !void {
        for (self.tensors) |tensor| {
            switch (tensor.source) {
                .copy => |source| {
                    try copyTensor(self.allocator, source, file_r, writer, self.input_header.data_start);
                },

                .nvfp4 => |source| {
                    // TODO: Here I should probably pass the source, and use it to find the scales (specially because it contains the offsets from the input file, which is important for the reader)
                    try convertQuantizedWeightF16(self.allocator, source, file_r, writer, self.input_header.data_start);
                },
            }
        }
    }
};

fn convertQuantizedWeightF16(allocator: std.mem.Allocator, source: *Nvfp4Source, reader: *std.Io.File.Reader, writer: *std.Io.Writer, input_data_start: u64) !void {
    // Dequantizes a quantized weight tensor from the input file and writes the dequantized values to the output file.
    const num_bytes = source.weight.info.byteSize();
    const buffer = try allocator.alloc(u8, @intCast(num_bytes));
    defer allocator.free(buffer);
    reader.seekTo(source.weight.info.begin + input_data_start);
    try reader.readAll(&buffer);
    // Find the scale and scale2 tensors
    const block_scale_tensor = source.block_scale;
    const global_scale_tensor = source.global_scale;
    const block_scale_bytes = try allocator.alloc(u8, block_scale_tensor.byteSize());
    defer allocator.free(block_scale_bytes);
    reader.seekTo(block_scale_tensor.begin + input_data_start);
    try reader.readAll(&block_scale_bytes);

    if (global_scale_tensor.byteSize() != 4)
        return error.InvalidGlobalScale;

    var global_scale_bytes: [4]u8 = undefined;
    reader.seekTo(global_scale_tensor.begin + input_data_start);
    try reader.readAll(&global_scale_bytes);

    // Dequantize the weight tensor using the global scale and block scale tensors
    const global_scale_bits = std.mem.readInt(u32, &global_scale_bytes, .little);
    const global_scale: f32 = @bitCast(global_scale_bits);
    const num_values = num_bytes * 2; // Each byte contains 2 values
    const weights_reader = std.Io.Reader.fromSlice(&buffer);
    const scales_reader = std.Io.Reader.fromSlice(&block_scale_bytes);
    try nvfp4.dequantizeStreamF16(&weights_reader, &scales_reader, global_scale, writer, num_values);
}

fn copyTensor(allocator: std.mem.Allocator, source: *const st.TensorInfo, reader: *std.Io.File.Reader, writer: *std.Io.Writer, input_data_start: u64) !void {
    // Copies the bytes of an input tensor to the output file. This is used for tensors that do not require dequantization, such as biases.

    const num_bytes = source.info.byteSize();
    const buffer = try allocator.alloc(u8, @intCast(num_bytes));
    defer allocator.free(buffer);
    reader.seekTo(source.info.begin + input_data_start);
    try reader.readAll(&buffer);
    try writer.writeAll(&buffer);
}

fn countOutputTensors(header: *const st.Header) usize {
    var count: usize = 0;
    for (header.tensors) |*tensor| {
        // Count the number of output tensors based on the input header.
        if (!isNvfp4Scale(tensor)) {
            // Scales are used for dequantization but are not included in the output.
            count += 1;
        }
    }
    return count;
}

fn isNvfp4Weight(
    allocator: std.mem.Allocator,
    header: *const st.Header,
    tensor: *const st.TensorInfo,
) bool { // TODO: delete this function

    // Checks that:
    // 1- tensor tensor.dtype == u8,
    // 2- tensor.name ends with ".weight"
    // 3- the corresponding scale and scale2 tensors exist in the header,
    // 4- block scale dtype == F8_E4M3
    // 5- global scale dtype == F32
    if (tensor.dtype != .u8) return false;
    if (!std.mem.endsWith(u8, tensor.name, ".weight")) return false;
    const block_scale_name =
        try std.fmt.allocPrint(
            allocator,
            "{s}_scale",
            .{tensor.name},
        );
    defer allocator.free(block_scale_name);

    const global_scale_name =
        try std.fmt.allocPrint(
            allocator,
            "{s}_scale_2",
            .{tensor.name},
        );
    defer allocator.free(global_scale_name);
    const block_scale_tensor = header.find(block_scale_name) orelse return false;
    const global_scale_tensor = header.find(global_scale_name) orelse return false;
    if (block_scale_tensor.dtype != .f8_e4m3) return false;
    if (global_scale_tensor.dtype != .f32) return false;
    return true;
}

fn isNvfp4Scale(
    tensor: *const st.TensorInfo,
) bool {
    // Detects if tensor.name ends with  weight_scale or weight_scale_2
    return std.mem.endsWith(u8, tensor.name, ".weight_scale") or std.mem.endsWith(u8, tensor.name, ".weight_scale_2"); // Maybe centralize this naming logic somewhere

}

fn matchNvfp4Weight(
    allocator: std.mem.Allocator,
    header: *const st.Header,
    weight: *const st.TensorInfo,
) !?Nvfp4Source {
    // Checks if the given weight tensor is a quantized weight tensor and returns the corresponding Nvfp4Source if it is.
    // In order to be considered a weight tensor, it must check:
    // 1- tensor tensor.dtype == u8,
    // 2- tensor.name ends with ".weight"
    // 3- the corresponding scale and scale2 tensors exist in the header,
    // 4- block scale dtype == F8_E4M3
    // 5- global scale dtype == F32

    if (weight.dtype != .u8) return null;
    if (!std.mem.endsWith(u8, weight.name, ".weight")) return null;
    const block_scale_name =
        try std.fmt.allocPrint(
            allocator,
            "{s}_scale",
            .{weight.name},
        );
    defer allocator.free(block_scale_name);

    const global_scale_name =
        try std.fmt.allocPrint(
            allocator,
            "{s}_scale_2",
            .{weight.name},
        );
    defer allocator.free(global_scale_name);
    const block_scale_tensor = header.find(block_scale_name) orelse return null;
    const global_scale_tensor = header.find(global_scale_name) orelse return null;
    if (block_scale_tensor.dtype != .f8_e4m3) return null;
    if (global_scale_tensor.dtype != .f32) return null;
    return Nvfp4Source{
        .weight = weight,
        .block_scale = block_scale_tensor,
        .global_scale = global_scale_tensor,
    };
}
