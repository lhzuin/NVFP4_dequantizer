// Parses CLI options and invokes the SafeTensors conversion pipeline.

const std = @import("std");
const convert = @import("convert.zig");

fn printUsage(program: []const u8) void {
    std.debug.print(
        "Usage: {s} INPUT OUTPUT [--output-format FORMAT] [--threads N] [--chunk-blocks N]\n",
        .{program},
    );
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const minimal = init.minimal;
    const args = try minimal.args.toSlice(init.arena.allocator());

    if (args.len < 3) {
        printUsage(args[0]);
        return;
    }

    const input_path = args[1];
    const output_path = args[2];

    var options: convert.Options = .{
        .output_type = .f16,
    };

    var i: usize = 3;
    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "--output-format")) {
            if (i + 1 >= args.len) {
                std.debug.print("Missing value after --output-format\n", .{});
                printUsage(args[0]);
                return;
            }

            options.output_type = std.meta.stringToEnum(convert.OutputType, args[i + 1]) orelse {
                std.debug.print("Unsupported output format: {s} (supported: f16)\n", .{args[i + 1]});
                return;
            };

            i += 2;
        } else if (std.mem.eql(u8, args[i], "--threads")) {
            if (i + 1 >= args.len) {
                std.debug.print("Missing value after --threads\n", .{});
                printUsage(args[0]);
                return;
            }

            options.threads = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                std.debug.print("Invalid thread count: {s}\n", .{args[i + 1]});
                return;
            };

            if (options.threads == 0) {
                std.debug.print("Thread count must be greater than zero\n", .{});
                return;
            }

            i += 2;
        } else if (std.mem.eql(u8, args[i], "--chunk-blocks")) {
            if (i + 1 >= args.len) {
                std.debug.print("Missing value after --chunk-blocks\n", .{});
                printUsage(args[0]);
                return;
            }

            options.chunk_blocks = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                std.debug.print("Invalid chunk block count: {s}\n", .{args[i + 1]});
                return;
            };

            if (options.chunk_blocks == 0) {
                std.debug.print("Chunk block count must be greater than zero\n", .{});
                return;
            }

            i += 2;
        } else {
            std.debug.print("Unknown argument: {s}\n", .{args[i]});
            printUsage(args[0]);
            return;
        }
    }

    try convert.dequantizeFile(
        io,
        init.gpa,
        input_path,
        output_path,
        options,
    );
}
