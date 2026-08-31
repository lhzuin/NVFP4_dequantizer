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
    chunk_blocks: usize = 4096, // Each chunk will contain this many blocks of 16 values each (except the last one that might be smaller). This is used for parallelization and memory management.
};

pub fn dequantizeFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
    options: Options,
) !void {
    if (options.threads == 0) return error.InvalidThreadCount;
    if (options.chunk_blocks == 0) return error.InvalidChunkSize;

    // 1. Open input
    var file = try std.Io.Dir.cwd().openFile(io, input_path, .{ .mode = .read_only });
    defer file.close(io);
    var buffer: [1024]u8 = undefined;
    var file_r: std.Io.File.Reader = file.reader(io, &buffer);

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
    const output_infos = try allocator.alloc(st.TensorInfo, plan.tensors.len);
    defer allocator.free(output_infos);

    for (plan.tensors, 0..) |tensor, i| {
        output_infos[i] = tensor.info;
    } // Needed because the plan.tensors array contains OutputTensor objects (not accessible by safetensors.zig's Header)
    var output_header = try st.Header.fromInfos(allocator, output_infos);
    defer output_header.deinit();

    // 7. Write output SafeTensors header
    try output_header.write(allocator, writer);

    // 8. Execute ConversionPlan
    plan.executePlan(io, &file_r, writer, options) catch |err| {
        std.debug.print("Error during conversion: {}\n", .{err});
        return err;
    };

    // 9. Flush output Writer
    try writer.flush();
}

const OutputTensor = struct {
    // Saves the intended structure of an output tensor so that we can easily process it later.

    info: st.TensorInfo, // Output metadata.
    source: Source, // How to generate the data using the input.

    // Only non-null when this plan had to create a new shape.
    owned_shape: ?[]usize = null,
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
        for (self.tensors) |tensor| {
            if (tensor.owned_shape) |shape| {
                self.allocator.free(shape);
            }
        }
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
        var byte_cursor: u64 = 0;
        for (header.tensors) |*tensor| {
            // Calculate the output tensor's begin and end offsets based on the input tensor's shape and the output type's size

            if (isNvfp4Scale(tensor)) {
                // Skip scale tensors, they are not included in the output
                continue;
            }

            if (try matchNvfp4Weight(allocator, header, tensor)) |nvfp4_source| {
                // Create dequantized output tensor entry in the plan

                if (tensor.shape.len == 0) return error.InvalidNvfp4Shape;

                const output_shape = try allocator.dupe(usize, tensor.shape);
                output_shape[output_shape.len - 1] *= 2; // Each byte contains 2 values, so we double the last dimension of the shape

                const logical_values = tensor.byteSize() * 2;
                const output_byte_size = logical_values * @as(u64, @intCast(output_type.toDType().size()));
                const output_begin = byte_cursor;
                const output_end = output_begin + output_byte_size;
                byte_cursor = output_end;
                plan.tensors[count] = OutputTensor{ .info = .{
                    .name = tensor.name,
                    .dtype = output_type.toDType(),
                    .shape = output_shape,
                    .begin = output_begin,
                    .end = output_end,
                }, .source = Source{ .nvfp4 = .{
                    .weight = tensor,
                    .block_scale = nvfp4_source.block_scale,
                    .global_scale = nvfp4_source.global_scale,
                } }, .owned_shape = output_shape };
                count += 1;
            } else {
                // For other tensors (like biases), we will just copy them to the output
                const output_byte_size = tensor.byteSize();
                const output_begin = byte_cursor;
                const output_end = output_begin + output_byte_size;
                byte_cursor = output_end;
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
        io: std.Io,
        file_r: *std.Io.File.Reader,
        writer: *std.Io.Writer,
        options: Options,
    ) !void {
        for (self.tensors) |tensor| {
            switch (tensor.source) {
                .copy => |source| {
                    //try copyTensor(self.allocator, source, file_r, writer, self.input_header.data_start);
                    try copyTensor(source, file_r, writer, self.input_header.data_start);
                },

                .nvfp4 => |source| {
                    try convertQuantizedWeightF16(self.allocator, io, source, &tensor.info, file_r, writer, self.input_header.data_start, options);
                },
            }
        }
    }
};

const ChunkScratch = struct {
    packed_bytes: []u8,
    scales: []u8,
    output: []f16,

    pub fn init(
        allocator: std.mem.Allocator,
        chunk_blocks: usize,
    ) !ChunkScratch {
        // Allocates scratch space for a chunk of blocks to be processed in parallel and returns a ChunkScratch object containing the allocated buffers.
        const packed_bytes = try allocator.alloc(u8, chunk_blocks * nvfp4.packed_block_size);
        errdefer allocator.free(packed_bytes);

        const scales = try allocator.alloc(u8, chunk_blocks);
        errdefer allocator.free(scales);

        const output = try allocator.alloc(f16, chunk_blocks * nvfp4.block_size);
        errdefer allocator.free(output);

        return .{
            .packed_bytes = packed_bytes,
            .scales = scales,
            .output = output,
        };
    }

    pub fn deinit(
        self: *ChunkScratch,
        allocator: std.mem.Allocator,
    ) void {
        allocator.free(self.output);
        allocator.free(self.scales);
        allocator.free(self.packed_bytes);
    }
};

const ChunkJob = struct {
    // Represents a job for processing a chunk of blocks. It contains the necessary information to read the packed bytes and scales from the input file, dequantize them, and write the dequantized values to the output writer.
    sequence: usize, // Index of the chunk in the sequence of chunks to be processed. Later used to restore output order when chunks are processed in parallel.
    // Obs: more fine-grained multi-thread approach was chosen instead to limit memory usage and avoid IO collisions.
    weight_offset: u64,
    scale_offset: u64,
    output_offset: u64,

    block_count: usize, // Actual number of blocks to process in this chunk (may be less than the chunk size for the last chunk).
};

fn convertQuantizedWeightF16(allocator: std.mem.Allocator, io: std.Io, source: Nvfp4Source, output_info: *const st.TensorInfo, reader: *std.Io.File.Reader, writer: *std.Io.Writer, input_data_start: u64, options: Options) !void {
    // Dequantizes a quantized weight tensor from the input file and writes the dequantized values to the output file.

    const chunk_blocks = options.chunk_blocks;
    const num_bytes = source.weight.byteSize();
    if (num_bytes % nvfp4.packed_block_size != 0) return error.InvalidNvfp4WeightSize;

    const num_blocks_u64 = num_bytes / nvfp4.packed_block_size;
    if (source.block_scale.byteSize() != num_blocks_u64) return error.InvalidBlockScaleCount;

    const num_blocks = std.math.cast(usize, num_blocks_u64) orelse return error.TensorTooLarge;

    // Find the scale and scale2 tensors
    const global_scale_tensor = source.global_scale; // As global scale is a single value, we can read it once and reuse it for all blocks.

    if (global_scale_tensor.byteSize() != 4)
        return error.InvalidGlobalScale;

    var global_scale_bytes: [4]u8 = undefined;
    try reader.seekTo(global_scale_tensor.begin + input_data_start);
    try reader.interface.readSliceAll(&global_scale_bytes);

    // Dequantize the weight tensor using the global scale and block scale tensors
    const global_scale_bits = std.mem.readInt(u32, &global_scale_bytes, .little);
    const global_scale: f32 = @bitCast(global_scale_bits);

    var scratch = try ChunkScratch.init(allocator, chunk_blocks);
    defer scratch.deinit(allocator);

    var first_block: usize = 0;
    var sequence: usize = 0;

    while (first_block < num_blocks) {
        const block_count = @min(chunk_blocks, num_blocks - first_block); // How many blocks to process in this chunk

        const first_block_u64: u64 = @intCast(first_block); // Position of first block in the chunk

        const job = ChunkJob{
            .sequence = sequence,
            .weight_offset = input_data_start + source.weight.begin + first_block_u64 * nvfp4.packed_block_size,
            .scale_offset = input_data_start + source.block_scale.begin + first_block_u64,
            .output_offset = output_info.begin + first_block_u64 * nvfp4.block_size * @sizeOf(f16), // Not sure if I should hardcode @sizeOf(f16) or centralize with the output type
            .block_count = block_count,
        };

        // Create slices based on the scratch buffers and the job's block count (à la volée)
        const weight_chunk_size = job.block_count * nvfp4.packed_block_size;
        const output_value_count = job.block_count * nvfp4.block_size;

        const weight_chunk = scratch.packed_bytes[0..weight_chunk_size];
        const scale_chunk = scratch.scales[0..job.block_count];
        const output_chunk = scratch.output[0..output_value_count];

        // Set the reader to the correct positions for this job and read the weights and scales
        try reader.seekTo(job.weight_offset);
        try reader.interface.readSliceAll(weight_chunk);

        try reader.seekTo(job.scale_offset);
        try reader.interface.readSliceAll(scale_chunk);

        // Dequantize the chunk of blocks (with multicore)
        try dequantizeChunkParallel(
            io,
            weight_chunk,
            scale_chunk,
            global_scale,
            output_chunk,
            options.threads,
        );

        //try nvfp4.dequantizeBlocksF16(weight_chunk, scale_chunk, global_scale, output_chunk);

        // Write the dequantized values to the output writer
        try writer.writeAll(std.mem.sliceAsBytes(output_chunk));

        // Increase counters for the next job/chunk
        first_block += block_count;
        sequence += 1;
    }
}

fn copyTensorOld(allocator: std.mem.Allocator, source: *const st.TensorInfo, reader: *std.Io.File.Reader, writer: *std.Io.Writer, input_data_start: u64) !void {
    // Copies the bytes of an input tensor to the output file. This is used for tensors that do not require dequantization, such as biases.

    const num_bytes = source.byteSize();
    const buffer = try allocator.alloc(u8, @intCast(num_bytes));
    defer allocator.free(buffer);
    try reader.seekTo(source.begin + input_data_start);
    try reader.interface.readSliceAll(buffer);
    try writer.writeAll(buffer);
}

fn copyTensor(source: *const st.TensorInfo, reader: *std.Io.File.Reader, writer: *std.Io.Writer, input_data_start: u64) !void {
    // Copies the bytes of an input tensor to the output file. This is used for tensors that do not require dequantization, such as biases.
    // This version uses streamExact64 to avoid allocating a buffer for the entire tensor, which is more efficient for large tensors.
    try reader.seekTo(source.begin + input_data_start);
    try reader.interface.streamExact64(writer, source.byteSize());
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

fn dequantizeBlocksWorker(
    packed_bytes: []const u8,
    scales: []const u8,
    global_scale: f32,
    output: []f16,
) void {
    // Wrapper to nvfp4's dequantization function. Needed because Io.Group.concurrent requires callbacks returning void or only error.Canceled
    nvfp4.dequantizeBlocksF16(packed_bytes, scales, global_scale, output) catch unreachable;
}

fn dequantizeChunkParallel(
    io: std.Io,
    packed_bytes: []const u8,
    scales: []const u8,
    global_scale: f32,
    output: []f16,
    threads: usize,
) !void {
    if (packed_bytes.len % nvfp4.packed_block_size != 0) {
        return error.InvalidPackedLength;
    }

    const num_blocks = packed_bytes.len / nvfp4.packed_block_size;

    if (scales.len != num_blocks) {
        return error.ScaleCountMismatch;
    }

    if (output.len != num_blocks * nvfp4.block_size) {
        return error.OutputLengthMismatch;
    }

    if (threads == 1 or num_blocks <= 1) {
        return nvfp4.dequantizeBlocksF16(
            packed_bytes,
            scales,
            global_scale,
            output,
        );
    }

    const worker_count = @min(threads, num_blocks);

    var group: std.Io.Group = .init;
    defer group.cancel(io);

    for (0..worker_count) |worker_index| {
        const first_block = worker_index * num_blocks / worker_count;
        const end_block = (worker_index + 1) * num_blocks / worker_count;

        const packed_part = packed_bytes[first_block * nvfp4.packed_block_size .. end_block * nvfp4.packed_block_size];

        const scale_part = scales[first_block..end_block];

        const output_part = output[first_block * nvfp4.block_size .. end_block * nvfp4.block_size];

        try group.concurrent(
            io,
            dequantizeBlocksWorker,
            .{
                packed_part,
                scale_part,
                global_scale,
                output_part,
            },
        );
    }

    try group.await(io);
}
