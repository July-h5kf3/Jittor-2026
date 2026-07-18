import tempfile
import unittest
from pathlib import Path

import jittor as jt

from src.system.spec import DummySystem


class FakeModel:
    def __init__(self):
        self.value = jt.array([1.0])

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value.assign(state["value"])

    def save(self, path):
        jt.save(self.state_dict(), path)


class MultiEMATests(unittest.TestCase):
    def test_saves_raw_and_all_decay_variants(self):
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            system = DummySystem(
                dataset_module=None,
                model=model,
                trainer_config={"ema_decays": [0.9, 0.99]},
                ckpt_save_dir=directory,
            )
            system._update_ema()
            model.value.assign(jt.array([3.0]))
            system._update_ema()
            target = Path(directory) / "checkpoint_best.pkl"
            system._save_model(str(target))

            self.assertTrue(target.is_file())
            self.assertTrue((Path(directory) / "checkpoint_best_ema09.pkl").is_file())
            self.assertTrue((Path(directory) / "checkpoint_best_ema099.pkl").is_file())
            self.assertAlmostEqual(float(model.value.item()), 3.0, places=6)
            self.assertAlmostEqual(
                float(jt.load(str(Path(directory) / "checkpoint_best_ema09.pkl"))["value"].item()),
                1.2,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()
