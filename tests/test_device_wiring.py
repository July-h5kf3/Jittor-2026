import ast
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "infer.py",
    "generate_rot.py",
    "train.py",
    "train_rot.py",
    "run_plr003.py",
)


class DeviceWiringTests(unittest.TestCase):
    def test_entrypoints_use_shared_device_helpers(self):
        for relative_path in ENTRYPOINTS:
            with self.subTest(entrypoint=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("add_device_argument", source)
                self.assertIn("configure_device(args.device", source)

    def test_entrypoints_do_not_configure_jittor_flags_directly(self):
        for relative_path in ENTRYPOINTS:
            with self.subTest(entrypoint=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertNotIn('choices=("cuda", "cpu")', source)
                self.assertNotIn("jt.flags.use_cuda =", source)

    def test_reproduce_config_accepts_all_supported_devices(self):
        source = (ROOT / "tools" / "build_reproduce_config.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        device_arguments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--device"
        ]
        self.assertEqual(len(device_arguments), 1)
        keywords = {keyword.arg: keyword.value for keyword in device_arguments[0].keywords}
        self.assertEqual(
            ast.literal_eval(keywords["choices"]), ("cpu", "cuda", "acl")
        )
        self.assertEqual(ast.literal_eval(keywords["default"]), "cuda")
        self.assertNotIn('"--device", "cuda"', source)

    def test_reproduce_config_wires_acl_to_every_device_dependent_stage(self):
        script = ROOT / "tools" / "build_reproduce_config.py"
        specification = importlib.util.spec_from_file_location(
            "build_reproduce_config_for_test", script
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = pathlib.Path(temporary_directory)
            (temporary_root / "configs").mkdir()
            generated_script = temporary_root / "tools" / script.name
            with mock.patch.object(module, "Path", return_value=generated_script):
                module.main(["--device", "acl"])

            payload = json.loads(
                (temporary_root / "configs" / "reproduce_full.json").read_text(
                    encoding="utf-8"
                )
            )

        device_dependent = {"train", "infer", "train-rot", "generate-rot"}
        stages = [
            stage
            for stage in payload["stages"]
            if stage["entrypoint"] in device_dependent
        ]
        self.assertTrue(stages)
        for stage in stages:
            with self.subTest(stage=stage["name"]):
                position = stage["args"].index("--device")
                self.assertEqual(stage["args"][position + 1], "acl")


if __name__ == "__main__":
    unittest.main()
