// Deals with CLI (path input, path output, output dtype) and calls convert.dequantizeFile(...)

const std = @import("std");
const convert = @import("convert.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    // parse CLI args:
    // input.safetensors
    // output.safetensors
    const minimal = init.minimal;
    const args = try minimal.args.toSlice(init.arena.allocator());

    if (args.len != 3) {
        std.debug.print(
            "Usage: {s} INPUT OUTPUT\n",
            .{args[0]},
        );
        return;
    }

    const input_path = args[1];
    const output_path = args[2];

    try convert.dequantizeFile(
        io,
        init.gpa,
        input_path,
        output_path,
        .{
            .output_type = .f16,
        },
    );
}
