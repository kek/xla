"""Build rules for XLA generated regression testing."""

load("//xla/tests:build_defs.bzl", "xla_test")

def hlo_test(name, hlo, **kwargs):
    xla_test(
        name = name,
        srcs = [],
        env = {"HLO_PATH": "$(location {})".format(hlo)},
        data = [hlo],
        real_hardware_only = True,
        deps = [
            "//xla/tests/fuzz:hlo_test_lib",
            "@tsl//tsl/platform:test_main",
        ],
        **kwargs
    )
