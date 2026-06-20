import unittest
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

if "jittor" not in sys.modules:
    fake_jittor = ModuleType("jittor")
    fake_dataset_module = ModuleType("jittor.dataset")
    fake_dataset_module.Dataset = object
    fake_jittor.optim = SimpleNamespace(SGD=object, Adam=object)
    fake_jittor.dataset = fake_dataset_module
    fake_jittor.Var = type("Var", (), {})
    fake_jittor.array = lambda value: value
    fake_jittor.in_mpi = False
    fake_jittor.rank = 0
    fake_jittor.world_size = 1
    fake_jittor.mpi = None
    sys.modules["jittor"] = fake_jittor
    sys.modules["jittor.dataset"] = fake_dataset_module

from src.system import spec as system_spec


class FakeMPI:
    def __init__(self, rank=0, world_size=1, local_rank=0):
        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank

    def world_rank(self):
        return self._rank

    def world_size(self):
        return self._world_size

    def local_rank(self):
        return self._local_rank


class DistributedContextTests(unittest.TestCase):
    def test_non_mpi_context_is_single_primary_process(self):
        fake_jt = SimpleNamespace(in_mpi=False, rank=0, world_size=1, mpi=None)

        with patch.object(system_spec, "jt", fake_jt):
            context = system_spec.get_distributed_context()

        self.assertFalse(context.enabled)
        self.assertEqual(context.rank, 0)
        self.assertEqual(context.local_rank, 0)
        self.assertEqual(context.world_size, 1)
        self.assertTrue(system_spec.is_primary_process(context))

    def test_mpi_context_uses_mpi_runtime_rank_information(self):
        fake_jt = SimpleNamespace(
            in_mpi=True,
            rank=99,
            world_size=99,
            mpi=FakeMPI(rank=1, world_size=2, local_rank=1),
        )

        with patch.object(system_spec, "jt", fake_jt):
            context = system_spec.get_distributed_context()

        self.assertTrue(context.enabled)
        self.assertEqual(context.rank, 1)
        self.assertEqual(context.local_rank, 1)
        self.assertEqual(context.world_size, 2)
        self.assertFalse(system_spec.is_primary_process(context))


class FakeModel:
    def __init__(self):
        self.saved_paths = []

    def parameters(self):
        return []

    def save(self, path):
        self.saved_paths.append(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(path)


class EarlyStoppingTests(unittest.TestCase):
    def make_system(self, save_dir, patience=1):
        fake_jt = SimpleNamespace(in_mpi=False, rank=0, world_size=1, mpi=None)
        with patch.object(system_spec, "jt", fake_jt):
            return system_spec.DummySystem(
                dataset_module=SimpleNamespace(),
                model=FakeModel(),
                trainer_config={
                    "epochs": 100,
                    "save_best_only": True,
                    "early_stopping": {
                        "enabled": True,
                        "monitor": "val/loss_sum",
                        "mode": "min",
                        "patience": patience,
                        "min_delta": 0.0,
                    },
                },
                ckpt_save_dir=save_dir,
                ckpt_save_name="checkpoint",
            )

    def test_initial_best_metric_preserves_existing_best_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            fake_jt = SimpleNamespace(in_mpi=False, rank=0, world_size=1, mpi=None)
            with patch.object(system_spec, "jt", fake_jt):
                system = system_spec.DummySystem(
                    dataset_module=SimpleNamespace(),
                    model=FakeModel(),
                    trainer_config={
                        "save_best_only": True,
                        "initial_best_metric": 0.5,
                        "initial_best_epoch": 3,
                        "early_stopping": {
                            "enabled": True,
                            "monitor": "val/loss_sum",
                            "mode": "min",
                            "patience": 2,
                        },
                    },
                    ckpt_save_dir=tmpdir,
                    ckpt_save_name="checkpoint",
                )

            self.assertEqual(system.best_metric, 0.5)
            self.assertEqual(system.best_epoch, 3)
            self.assertFalse(system._handle_epoch_checkpoint(epoch=4, metric=0.6))
            self.assertEqual(system.model.saved_paths, [])

            self.assertFalse(system._handle_epoch_checkpoint(epoch=5, metric=0.4))
            self.assertEqual(system.best_metric, 0.4)
            self.assertEqual(system.best_epoch, 5)
            self.assertEqual(len(system.model.saved_paths), 1)

    def test_validation_monitor_value_averages_all_class_loss_sums(self):
        with TemporaryDirectory() as tmpdir:
            system = self.make_system(tmpdir)
            system._validation_loss = {
                "val/chair_loss_sum": [1.0, 3.0],
                "val/table_loss_sum": [2.0],
                "val/chair_loss": [100.0],
            }

            self.assertEqual(system._get_validation_monitor_value(), 2.0)

    def test_best_only_checkpoint_overwrites_best_and_stops_after_patience(self):
        with TemporaryDirectory() as tmpdir:
            system = self.make_system(tmpdir, patience=1)

            self.assertFalse(system._handle_epoch_checkpoint(epoch=0, metric=2.0))
            self.assertEqual(system.best_metric, 2.0)
            self.assertEqual(system.best_epoch, 0)
            self.assertEqual(len(system.model.saved_paths), 1)
            self.assertTrue(system.model.saved_paths[0].endswith("checkpoint_best.pkl"))

            self.assertFalse(system._handle_epoch_checkpoint(epoch=1, metric=1.5))
            self.assertEqual(system.best_metric, 1.5)
            self.assertEqual(system.best_epoch, 1)
            self.assertEqual(system._epochs_without_improvement, 0)
            self.assertEqual(len(system.model.saved_paths), 2)

            self.assertTrue(system._handle_epoch_checkpoint(epoch=2, metric=1.6))
            self.assertEqual(system.best_metric, 1.5)
            self.assertEqual(system.best_epoch, 1)
            self.assertEqual(system._epochs_without_improvement, 1)
            self.assertEqual(len(system.model.saved_paths), 2)


if __name__ == "__main__":
    unittest.main()
