const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable.
    const exe_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "nvfp4-dequant",
        .root_module = exe_module,
    });

    b.installArtifact(exe);

    // Convenient:
    //   zig build run -- input.safetensors output.safetensors
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the dequantizer");
    run_step.dependOn(&run_cmd.step);

    // ---------------------------------------------------------------------
    // Tests live under tests/, while production code lives under src/.
    //
    // We expose source files to tests as named modules instead of using
    // @import("../src/..."), which would cross the test module's package root.
    // ---------------------------------------------------------------------
    const tests_module = b.createModule(.{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    tests_module.addAnonymousImport("nvfp4", .{
        .root_source_file = b.path("src/nvfp4.zig"),
    });

    tests_module.addAnonymousImport("safetensors", .{
        .root_source_file = b.path("src/safetensors.zig"),
    });

    const unit_tests = b.addTest(.{
        .root_module = tests_module,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
