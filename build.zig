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

    const convert_tests_module = b.createModule(.{
        .root_source_file = b.path("tests/convert_tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    convert_tests_module.addAnonymousImport("convert", .{
        .root_source_file = b.path("src/convert.zig"),
    });

    const convert_tests = b.addTest(.{
        .root_module = convert_tests_module,
    });

    const run_convert_tests = b.addRunArtifact(convert_tests);

    const unit_test_step = b.step("test-unit", "Run Zig unit and converter tests");
    unit_test_step.dependOn(&run_unit_tests.step);
    unit_test_step.dependOn(&run_convert_tests.step);

    const integration_tests = b.addSystemCommand(&.{
        "python3",
        "tests/test_integration.py",
    });
    integration_tests.setCwd(b.path("."));
    integration_tests.setEnvironmentVariable("NVFP4_SKIP_BUILD", "1");
    integration_tests.step.dependOn(b.getInstallStep());

    const integration_test_step = b.step("test-integration", "Run CLI integration tests");
    integration_test_step.dependOn(&integration_tests.step);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(unit_test_step);
    test_step.dependOn(integration_test_step);
}
