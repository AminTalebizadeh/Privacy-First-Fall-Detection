"""Unit tests: finite-difference velocity/acceleration on synthetic motion."""

import unittest

import numpy as np

from src.features.frame_features import finite_diff, normalized_hip_dynamics
from src.features.sequence import FEATURE_DIM, sequence_features
from tests.test_normalize import synthetic_standing


class TestFiniteDiff(unittest.TestCase):
    def test_constant_velocity(self):
        fps = 30.0
        t = np.arange(60)
        y = 2.0 * t                                   # 2 px/frame
        vel, acc = finite_diff(y, fps)
        np.testing.assert_allclose(vel, 60.0, atol=1e-6)   # 2*30 px/s
        np.testing.assert_allclose(acc, 0.0, atol=1e-4)

    def test_constant_acceleration(self):
        fps = 30.0
        t = np.arange(90, dtype=np.float64)
        y = 0.5 * 3.0 * t ** 2                        # a = 3 px/frame^2
        vel, acc = finite_diff(y, fps)
        # interior points: central diffs are exact for quadratics
        np.testing.assert_allclose(vel[1:-1], 3.0 * t[1:-1] * fps, rtol=1e-6)
        np.testing.assert_allclose(acc[2:-2], 3.0 * fps * fps, rtol=1e-6)

    def test_nan_gap_interpolated_no_spike(self):
        fps = 30.0
        y = np.arange(30, dtype=np.float64)
        y[10:13] = np.nan                             # 3-frame detection gap
        vel, _ = finite_diff(y, fps)
        np.testing.assert_allclose(vel, fps, atol=1e-6)  # still 1 px/frame

    def test_all_nan_safe(self):
        vel, acc = finite_diff(np.full(10, np.nan), 30.0)
        self.assertFalse(np.isnan(vel).any())
        self.assertFalse(np.isnan(acc).any())

    def test_normalized_units(self):
        fps = 30.0
        hip_y = np.arange(30, dtype=np.float64) * 5.0  # 5 px/frame down
        scales = np.full(30, 100.0)                    # torso = 100 px
        vel, _ = normalized_hip_dynamics(hip_y, scales, fps)
        # 5 px/frame * 30 fps / 100 px-per-torso = 1.5 torso-lengths/s
        np.testing.assert_allclose(vel, 1.5, atol=1e-6)


class TestSequenceFeatures(unittest.TestCase):
    def test_shape_and_finiteness(self):
        seq = np.stack([synthetic_standing(hip_y=300 + 3 * t) for t in range(40)])
        feats = sequence_features(seq, fps=30.0)
        self.assertEqual(feats.shape, (40, FEATURE_DIM))
        self.assertTrue(np.isfinite(feats).all())

    def test_falling_sequence_has_downward_velocity(self):
        # hip drops 8 px/frame (image y grows downward) -> positive velocity
        seq = np.stack([synthetic_standing(hip_y=200 + 8 * t) for t in range(30)])
        feats = sequence_features(seq, fps=30.0)
        vel = feats[:, 53]
        self.assertTrue((vel[5:-5] > 0).all())
        # 8 px/frame * 30 fps / 100 px torso = 2.4 torso/s
        np.testing.assert_allclose(vel[5:-5], 2.4, atol=0.05)


if __name__ == "__main__":
    unittest.main()
