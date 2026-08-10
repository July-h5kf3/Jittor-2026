import argparse
import ast
import importlib.util
import io
import json
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np

import main
from main import parse_args


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_smoke_module():
    script = ROOT / "tests" / "smoke_jittor.py"
    specification = importlib.util.spec_from_file_location(
        "smoke_jittor_for_test", script
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_selective_scan_module(jittor_module):
    source = ROOT / "plr3d" / "ops" / "selective_scan.py"
    specification = importlib.util.spec_from_file_location(
        "selective_scan_for_test", source
    )
    module = importlib.util.module_from_spec(specification)
    with mock.patch.dict(sys.modules, {"jittor": jittor_module}):
        specification.loader.exec_module(module)
    return module


class Flags:
    def __init__(self):
        self.use_cuda = 0
        self.use_acl = 0


class FlagsWithoutAcl:
    __slots__ = ("use_cuda",)

    def __init__(self):
        self.use_cuda = 0


class EventFlags:
    def __init__(self, use_cuda=0, use_acl=0):
        self.events = []
        self._use_cuda = use_cuda
        self._use_acl = use_acl

    @property
    def use_cuda(self):
        return self._use_cuda

    @use_cuda.setter
    def use_cuda(self, value):
        self.events.append(("use_cuda", value))
        self._use_cuda = value

    @property
    def use_acl(self):
        return self._use_acl

    @use_acl.setter
    def use_acl(self, value):
        self.events.append(("use_acl", value))
        self._use_acl = value


class Compiler:
    def __init__(self, has_cuda=True, has_acl=True):
        self.has_cuda = has_cuda
        self.has_acl = has_acl
        self.nvcc_path = "/opt/cuda/bin/nvcc"
        self.tikcc_path = "/opt/ascend/bin/tikcc"


class FakeJittor:
    def __init__(self, has_cuda=True, has_acl=True):
        self.flags = Flags()
        self.compiler = Compiler(has_cuda=has_cuda, has_acl=has_acl)
        self.rank = 0
        self.world_size = 1
        self.in_mpi = False


class SparseJittor:
    def __init__(self):
        self.flags = type("Flags", (), {"use_cuda": None, "use_acl": None})()
        self.compiler = type(
            "Compiler",
            (),
            {"has_cuda": None, "has_acl": None, "nvcc_path": None, "tikcc_path": None},
        )()
        self.rank = None
        self.world_size = None
        self.in_mpi = None


class BackendTests(unittest.TestCase):
    def test_selective_scan_prefers_reference_when_acl_and_cuda_are_enabled(self):
        jittor = types.ModuleType("jittor")
        jittor.Var = object
        jittor.Function = type("Function", (), {"apply": classmethod(lambda cls: None)})
        jittor.flags = type("Flags", (), {"use_acl": 1, "use_cuda": 1})()
        selective_scan_module = load_selective_scan_module(jittor)

        class Value:
            def contiguous(self):
                return self

        values = tuple(Value() for _ in range(8))
        reference_result = object()

        with mock.patch.object(
            selective_scan_module, "_validate_inputs"
        ) as validate, mock.patch.object(
            selective_scan_module,
            "selective_scan_reference",
            return_value=reference_result,
        ) as reference, mock.patch.object(
            selective_scan_module._SelectiveScanCUDA, "apply"
        ) as cuda_apply:
            result = selective_scan_module.selective_scan(*values)

        self.assertIs(result, reference_result)
        validate.assert_called_once_with(*values)
        reference.assert_called_once_with(*values)
        cuda_apply.assert_not_called()

    def test_acl_smoke_never_selects_cuda_when_cuda_is_available(self):
        smoke = load_smoke_module()
        jt = FakeJittor(has_cuda=True, has_acl=True)

        class Parameter:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

        class Model:
            def state_dict(self):
                return {
                    str(index): Parameter(16278412 if index == 0 else 0)
                    for index in range(1280)
                }

        models = types.ModuleType("plr3d.models")
        models.DenoiseNet = Model
        ops = types.ModuleType("plr3d.ops")
        ops.__path__ = []
        selective_scan = types.ModuleType("plr3d.ops.selective_scan")
        selective_scan.selective_scan = lambda *values: values[0]
        selective_scan.selective_scan_reference = lambda *values: values[0]

        with mock.patch.dict(
            sys.modules,
            {
                "jittor": jt,
                "plr3d.models": models,
                "plr3d.ops": ops,
                "plr3d.ops.selective_scan": selective_scan,
            },
        ), mock.patch.object(smoke, "scan_inputs", return_value=[object()]), mock.patch.object(
            smoke,
            "run_scan",
            return_value=(np.array([1.0]), [np.array([1.0])] * 8),
        ) as run_scan, mock.patch("sys.stdout", io.StringIO()):
            smoke.main(["--device", "acl"])

        devices = [call.args[1] for call in run_scan.call_args_list]
        self.assertEqual(devices, ["cpu", "acl"])
        self.assertNotIn("cuda", devices)

    def test_acl_smoke_requires_all_eight_gradients(self):
        smoke = load_smoke_module()
        jt = FakeJittor(has_cuda=True, has_acl=True)

        class Parameter:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

        class Model:
            def state_dict(self):
                return {
                    str(index): Parameter(16278412 if index == 0 else 0)
                    for index in range(1280)
                }

        models = types.ModuleType("plr3d.models")
        models.DenoiseNet = Model
        ops = types.ModuleType("plr3d.ops")
        ops.__path__ = []
        selective_scan = types.ModuleType("plr3d.ops.selective_scan")
        selective_scan.selective_scan = lambda *values: values[0]
        selective_scan.selective_scan_reference = lambda *values: values[0]

        with mock.patch.dict(
            sys.modules,
            {
                "jittor": jt,
                "plr3d.models": models,
                "plr3d.ops": ops,
                "plr3d.ops.selective_scan": selective_scan,
            },
        ), mock.patch.object(
            smoke,
            "run_scan",
            return_value=(np.array([1.0]), [np.array([1.0])] * 7),
        ), mock.patch("sys.stdout", io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "expected 8 reference gradients"):
                smoke.main(["--device", "acl"])

    def test_acl_smoke_requires_all_eight_candidate_gradients(self):
        smoke = load_smoke_module()
        jt = FakeJittor(has_cuda=True, has_acl=True)

        class Parameter:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

        class Model:
            def state_dict(self):
                return {
                    str(index): Parameter(16278412 if index == 0 else 0)
                    for index in range(1280)
                }

        models = types.ModuleType("plr3d.models")
        models.DenoiseNet = Model
        ops = types.ModuleType("plr3d.ops")
        ops.__path__ = []
        selective_scan = types.ModuleType("plr3d.ops.selective_scan")
        selective_scan.selective_scan = lambda *values: values[0]
        selective_scan.selective_scan_reference = lambda *values: values[0]
        output = np.array([1.0])
        gradient = np.array([1.0])

        with mock.patch.dict(
            sys.modules,
            {
                "jittor": jt,
                "plr3d.models": models,
                "plr3d.ops": ops,
                "plr3d.ops.selective_scan": selective_scan,
            },
        ), mock.patch.object(
            smoke,
            "run_scan",
            side_effect=[
                (output, [gradient] * 8),
                (output, [gradient] * 7),
            ],
        ), mock.patch("sys.stdout", io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "expected 8 candidate gradients"):
                smoke.main(["--device", "acl"])

    def test_acl_smoke_asserts_acl_is_enabled_after_scanning(self):
        smoke = load_smoke_module()
        jt = FakeJittor(has_cuda=True, has_acl=True)

        class Parameter:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

        class Model:
            def state_dict(self):
                return {
                    str(index): Parameter(16278412 if index == 0 else 0)
                    for index in range(1280)
                }

        models = types.ModuleType("plr3d.models")
        models.DenoiseNet = Model
        ops = types.ModuleType("plr3d.ops")
        ops.__path__ = []
        selective_scan = types.ModuleType("plr3d.ops.selective_scan")
        selective_scan.selective_scan = lambda *values: values[0]
        selective_scan.selective_scan_reference = lambda *values: values[0]

        with mock.patch.dict(
            sys.modules,
            {
                "jittor": jt,
                "plr3d.models": models,
                "plr3d.ops": ops,
                "plr3d.ops.selective_scan": selective_scan,
            },
        ), mock.patch.object(smoke, "configure_device"), mock.patch.object(
            smoke,
            "run_scan",
            return_value=(np.array([1.0]), [np.array([1.0])] * 8),
        ), mock.patch("sys.stdout", io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "ACL smoke must leave ACL enabled"):
                smoke.main(["--device", "acl"])

    def test_acl_smoke_accepts_jittor_acl_cuda_alias_flags(self):
        smoke = load_smoke_module()
        jt = FakeJittor(has_cuda=True, has_acl=True)

        class Parameter:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

        class Model:
            def state_dict(self):
                return {
                    str(index): Parameter(16278412 if index == 0 else 0)
                    for index in range(1280)
                }

        models = types.ModuleType("plr3d.models")
        models.DenoiseNet = Model
        ops = types.ModuleType("plr3d.ops")
        ops.__path__ = []
        selective_scan = types.ModuleType("plr3d.ops.selective_scan")
        selective_scan.selective_scan = lambda *values: values[0]
        selective_scan.selective_scan_reference = lambda *values: values[0]

        def configure_acl_with_cuda_alias(device, jittor):
            self.assertEqual(device, "acl")
            jittor.flags.use_acl = 1
            jittor.flags.use_cuda = 1

        with mock.patch.dict(
            sys.modules,
            {
                "jittor": jt,
                "plr3d.models": models,
                "plr3d.ops": ops,
                "plr3d.ops.selective_scan": selective_scan,
            },
        ), mock.patch.object(
            smoke, "configure_device", side_effect=configure_acl_with_cuda_alias
        ), mock.patch.object(
            smoke,
            "run_scan",
            return_value=(np.array([1.0]), [np.array([1.0])] * 8),
        ), mock.patch("sys.stdout", io.StringIO()):
            smoke.main(["--device", "acl"])

    def test_scan_plan_switches_every_run_through_shared_backend(self):
        smoke = load_smoke_module()
        self.assertEqual(smoke.scan_validation_plan("cpu"), (("cpu", False),))
        self.assertEqual(
            smoke.scan_validation_plan("cuda"),
            (("cpu", False), ("cuda", True)),
        )
        plan = smoke.scan_validation_plan("acl")
        self.assertEqual(plan, (("cpu", False), ("acl", True)))

        class Value:
            def float32(self):
                return self

            def __pow__(self, exponent):
                return self

            def sum(self):
                return self

            def numpy(self):
                return np.array([1.0])

        class Jittor:
            @staticmethod
            def array(value):
                return Value()

            @staticmethod
            def grad(output, values):
                return [Value() for value in values]

        jt = Jittor()
        with mock.patch.object(smoke, "configure_device") as configure:
            for device, custom in plan:
                smoke.run_scan(
                    [np.array([1.0])],
                    device,
                    custom,
                    jt,
                    lambda *values: values[0],
                    lambda *values: values[0],
                )

        self.assertEqual(
            configure.call_args_list,
            [mock.call("cpu", jt), mock.call("acl", jt)],
        )

    def test_smoke_cli_wires_shared_device_configuration_before_model_creation(self):
        tree = ast.parse((ROOT / "tests" / "smoke_jittor.py").read_text(encoding="utf-8"))
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        parse = functions["parse_args"]
        self.assertEqual([argument.arg for argument in parse.args.args], ["argv"])
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "add_device_argument"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "parser"
                for node in ast.walk(parse)
            )
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "parse_args"
                and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id == "argv"
                for node in ast.walk(parse)
            )
        )

        smoke_main = functions["main"]
        self.assertEqual([argument.arg for argument in smoke_main.args.args], ["argv"])
        configure_index = next(
            index
            for index, statement in enumerate(smoke_main.body)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "configure_device"
            and len(statement.value.args) == 2
            and isinstance(statement.value.args[0], ast.Attribute)
            and isinstance(statement.value.args[0].value, ast.Name)
            and statement.value.args[0].value.id == "args"
            and statement.value.args[0].attr == "device"
            and isinstance(statement.value.args[1], ast.Name)
            and statement.value.args[1].id == "jt"
        )
        model_index = next(
            index
            for index, statement in enumerate(smoke_main.body)
            if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "DenoiseNet"
        )
        self.assertLess(configure_index, model_index)

    def test_doctor_parser_accepts_acl_device(self):
        self.assertEqual(parse_args(["doctor", "--device", "acl"]).device, "acl")

    def test_doctor_deep_help_describes_selected_device_smoke(self):
        output = io.StringIO()
        with mock.patch("sys.stdout", output), self.assertRaises(SystemExit):
            parse_args(["doctor", "--help"])

        self.assertIn("selected-device forward/backward smoke", output.getvalue())

    def test_main_only_prepares_cuda_layout_for_cuda_doctor(self):
        with mock.patch("main.prepare_conda_cuda_layout") as prepare, mock.patch(
            "main.doctor", return_value=0
        ) as run_doctor:
            self.assertEqual(main.main(["doctor", "--device", "acl"]), 0)

        prepare.assert_not_called()
        run_doctor.assert_called_once_with("acl", False)

        with mock.patch("main.prepare_conda_cuda_layout") as prepare, mock.patch(
            "main.doctor", return_value=0
        ) as run_doctor:
            self.assertEqual(main.main(["doctor", "--device", "cuda"]), 0)

        prepare.assert_called_once_with()
        run_doctor.assert_called_once_with("cuda", False)

    def test_deep_doctor_configures_selected_device_and_passes_it_to_smoke(self):
        jt = FakeJittor()
        jt.__version__ = "1.3.11"
        np = type("Numpy", (), {"__version__": "2.0.0"})()
        output = io.StringIO()

        with mock.patch.dict(sys.modules, {"jittor": jt, "numpy": np}), mock.patch(
            "main.run_command"
        ) as run_smoke, mock.patch("sys.stdout", output), mock.patch.object(
            main.sys, "version_info", (3, 10, 0)
        ):
            self.assertEqual(main.doctor("acl", deep=True), 0)

        report = json.loads(output.getvalue())
        self.assertEqual(report["device"], "acl")
        self.assertEqual(report["device_status"]["device"], "acl")
        self.assertEqual(jt.flags.use_acl, 1)
        self.assertEqual(jt.flags.use_cuda, 0)
        run_smoke.assert_called_once_with(
            [
                sys.executable,
                str(main.ROOT / "tests" / "smoke_jittor.py"),
                "--device",
                "acl",
            ],
            main.ROOT,
        )

    def test_add_device_argument_uses_cuda_default_and_supported_choices(self):
        from plr3d.backend import DEVICES, add_device_argument

        parser = argparse.ArgumentParser()
        self.assertIs(add_device_argument(parser), parser)

        action = next(action for action in parser._actions if action.dest == "device")
        self.assertEqual(parser.parse_args([]).device, "cuda")
        self.assertEqual(tuple(action.choices), DEVICES)

    def test_acl_enables_acl_and_disables_cuda(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        configure_device("acl", jt)

        self.assertEqual(jt.flags.use_acl, 1)
        self.assertEqual(jt.flags.use_cuda, 0)

    def test_cuda_enables_cuda_and_disables_acl(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        configure_device("cuda", jt)

        self.assertEqual(jt.flags.use_cuda, 1)
        self.assertEqual(jt.flags.use_acl, 0)

    def test_switching_from_cuda_to_acl_disables_cuda_first(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        jt.flags = EventFlags(use_cuda=1)

        configure_device("acl", jt)

        self.assertEqual(jt.flags.events, [("use_cuda", 0), ("use_acl", 1)])

    def test_switching_from_acl_to_cuda_disables_acl_first(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        jt.flags = EventFlags(use_acl=1)

        configure_device("cuda", jt)

        self.assertEqual(jt.flags.events, [("use_acl", 0), ("use_cuda", 1)])

    def test_cpu_disables_cuda_and_acl(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        jt.flags.use_cuda = 1
        jt.flags.use_acl = 1
        configure_device("cpu", jt)

        self.assertEqual(jt.flags.use_cuda, 0)
        self.assertEqual(jt.flags.use_acl, 0)

    def test_acl_requires_an_available_backend(self):
        from plr3d.backend import BackendError, configure_device

        with self.assertRaisesRegex(BackendError, "ACL backend is unavailable"):
            configure_device("acl", FakeJittor(has_acl=False))

    def test_cuda_requires_an_available_backend(self):
        from plr3d.backend import BackendError, configure_device

        with self.assertRaisesRegex(BackendError, "CUDA backend is unavailable"):
            configure_device("cuda", FakeJittor(has_cuda=False))

    def test_unknown_device_raises_backend_error(self):
        from plr3d.backend import BackendError, configure_device

        with self.assertRaises(BackendError):
            configure_device("metal", FakeJittor())

    def test_acl_requires_an_acl_flag(self):
        from plr3d.backend import BackendError, configure_device

        jt = FakeJittor()
        jt.flags = FlagsWithoutAcl()

        with self.assertRaisesRegex(
            BackendError, "Jittor does not expose the ACL device flag"
        ):
            configure_device("acl", jt)

    def test_cpu_allows_jittor_without_an_acl_flag(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        jt.flags = FlagsWithoutAcl()
        jt.flags.use_cuda = 1

        configure_device("cpu", jt)

        self.assertEqual(jt.flags.use_cuda, 0)

    def test_cuda_allows_jittor_without_an_acl_flag(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        jt.flags = FlagsWithoutAcl()

        configure_device("cuda", jt)

        self.assertEqual(jt.flags.use_cuda, 1)

    def test_status_reports_backend_environment_and_distributed_state(self):
        from plr3d.backend import device_status

        environment = {
            "NKAI_JITTOR_COMMIT": "fake-commit",
            "ASCEND_TOOLKIT_HOME": "/opt/ascend",
            "JITTOR_HOME": "/tmp/jittor-cache",
        }

        status = device_status("acl", FakeJittor(), environment)

        self.assertEqual(status["device"], "acl")
        self.assertEqual(status["jittor_commit"], "fake-commit")
        self.assertEqual(status["ascend_toolkit_home"], "/opt/ascend")
        self.assertEqual(status["jittor_home"], "/tmp/jittor-cache")
        self.assertEqual(status["rank"], 0)
        self.assertEqual(status["world_size"], 1)
        self.assertFalse(status["in_mpi"])
        self.assertTrue(status["has_acl"])
        self.assertEqual(status["tikcc_path"], "/opt/ascend/bin/tikcc")

    def test_status_normalizes_missing_values_and_has_no_environment_aliases(self):
        from plr3d.backend import device_status

        for environment in ({}, {"NKAI_JITTOR_COMMIT": None}):
            with self.subTest(environment=environment):
                status = device_status("cpu", SparseJittor(), environment)

                self.assertEqual(
                    set(status),
                    {
                        "device",
                        "jittor_commit",
                        "has_cuda",
                        "has_acl",
                        "use_cuda",
                        "use_acl",
                        "nvcc_path",
                        "tikcc_path",
                        "ascend_toolkit_home",
                        "jittor_home",
                        "in_mpi",
                        "rank",
                        "world_size",
                    },
                )
                self.assertEqual(status["jittor_commit"], "unknown")
                self.assertEqual(status["ascend_toolkit_home"], "")
                self.assertEqual(status["jittor_home"], "")
                self.assertEqual(status["nvcc_path"], "")
                self.assertEqual(status["tikcc_path"], "")
                self.assertEqual(status["rank"], 0)
                self.assertEqual(status["world_size"], 1)
                self.assertIs(status["use_cuda"], False)
                self.assertIs(status["use_acl"], False)
                self.assertIs(status["has_cuda"], False)
                self.assertIs(status["has_acl"], False)


if __name__ == "__main__":
    unittest.main()
