import unittest

import numpy as np

from plr3d.routing import extrapolate, standard_route


class RoutingTests(unittest.TestCase):
    def test_standard_selected(self):
        noisy = np.zeros((8, 3), dtype=np.float32)
        rot = np.ones((8, 3), dtype=np.float32)
        member = np.full((8, 3), 2.0, dtype=np.float32)
        output, ratio, selected = standard_route(noisy, rot, member, 4.0)
        self.assertTrue(selected)
        self.assertAlmostEqual(ratio, 2.0)
        np.testing.assert_array_equal(output, np.full((8, 3), 1.75, dtype=np.float32))

    def test_standard_rejected(self):
        noisy = np.zeros((8, 3), dtype=np.float32)
        rot = np.full((8, 3), 0.1, dtype=np.float32)
        member = np.ones((8, 3), dtype=np.float32)
        output, _, selected = standard_route(noisy, rot, member, 4.0)
        self.assertFalse(selected)
        np.testing.assert_array_equal(output, rot)

    def test_extrapolation(self):
        full = np.ones((4, 3), dtype=np.float32)
        plr1 = np.full((4, 3), 2.0, dtype=np.float32)
        output = extrapolate(full, plr1, 1.25)
        np.testing.assert_array_equal(output, np.full((4, 3), 2.25, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
