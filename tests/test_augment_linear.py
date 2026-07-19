from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from src.data.asset import Asset


ROOT = Path(__file__).resolve().parents[1]


class AugmentLinearTests(unittest.TestCase):
    def test_asset_transform_updates_mesh_and_sampled_clouds(self):
        clean = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        noisy = clean + np.asarray([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1]])
        asset = Asset(
            vertices=clean.copy(),
            sampled_vertices=clean.copy(),
            sampled_vertices_noisy=noisy.copy(),
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            [[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        )
        transform[:3, 3] = np.asarray([0.25, -0.5, 1.0])

        expected_clean = clean @ transform[:3, :3].T + transform[:3, 3]
        expected_noisy = noisy @ transform[:3, :3].T + transform[:3, 3]
        asset.transform(transform)

        np.testing.assert_allclose(asset.vertices, expected_clean)
        np.testing.assert_allclose(asset.sampled_vertices, expected_clean)
        np.testing.assert_allclose(asset.sampled_vertices_noisy, expected_noisy)
        np.testing.assert_allclose(
            asset.sampled_vertices_noisy - asset.sampled_vertices,
            (noisy - clean) @ transform[:3, :3].T,
        )

    def test_rot001_changes_rotation_only(self):
        transform = OmegaConf.load(
            ROOT / "configs" / "transform" / "spcf_n_rot001.yaml"
        )
        linear = next(
            item
            for item in transform.train_transform.augments
            if item["__target__"] == "linear"
        )
        self.assertEqual(linear.scale_p, 0.0)
        self.assertEqual(linear.rotate_p, 0.5)

        for name in (
            "train_spcfgfnrot001_cvm.yaml",
            "train_spcfgfnrot001.yaml",
            "predict_spcfgfnrot001a105_local2.yaml",
            "predict_spcfgfnrot001a105_local2_pass2.yaml",
            "predict_spcfgfnrot001a105_submit.yaml",
            "predict_spcfgfnrot001a105_submit_pass2.yaml",
        ):
            task = OmegaConf.load(ROOT / "configs" / "task" / name)
            for kind in ("data", "transform", "system", "model"):
                component = task.components[kind]
                self.assertTrue(
                    (ROOT / "configs" / kind / f"{component}.yaml").is_file()
                )


if __name__ == "__main__":
    unittest.main()
