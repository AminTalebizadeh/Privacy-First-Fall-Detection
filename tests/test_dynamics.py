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

    def test_nan_gap_not_interpolated(self):
        fps = 30.0
        y = np.arange(30, dtype=np.float64)
        y[10:13] = np.nan  # 3-frame detection gap

        vel, acc = finite_diff(y, fps)

        # Valid regions retain the original constant velocity.
        np.testing.assert_allclose(vel[:10], fps, atol=1e-6)
        np.testing.assert_allclose(vel[13:], fps, atol=1e-6)

        # Long missing-data gap must not be interpolated.
        np.testing.assert_allclose(vel[10:13], 0.0, atol=1e-6)

        # No artificial acceleration spikes.
        np.testing.assert_allclose(acc, 0.0, atol=1e-6)

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

    def test_no_valid_torso_scale_returns_zero_dynamics(self):
        hip_y = np.arange(12, dtype=np.float64) * 5.0
        scales = np.full(12, np.nan)

        vel, acc = normalized_hip_dynamics(hip_y, scales, fps=30.0)

        np.testing.assert_array_equal(vel, np.zeros(12, dtype=np.float32))
        np.testing.assert_array_equal(acc, np.zeros(12, dtype=np.float32))

    def test_nonpositive_scales_are_not_valid(self):
        hip_y = np.arange(8, dtype=np.float64)
        scales = np.array([0.0, -1.0] * 4)

        vel, acc = normalized_hip_dynamics(hip_y, scales, fps=30.0)

        np.testing.assert_array_equal(vel, np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(acc, np.zeros(8, dtype=np.float32))

    def test_short_internal_gap_is_interpolated(self):
        y = np.arange(20, dtype=np.float64)
        y[8:10] = np.nan  # exactly max_gap=2

        vel, acc = finite_diff(y, fps=30.0, max_gap=2)

        np.testing.assert_allclose(vel, 30.0, atol=1e-6)
        np.testing.assert_allclose(acc, 0.0, atol=1e-6)

    def test_edge_gaps_are_not_interpolated(self):
        y = np.arange(20, dtype=np.float64)
        y[:3] = np.nan
        y[-2:] = np.nan

        vel, acc = finite_diff(y, fps=30.0)

        np.testing.assert_array_equal(vel[:3], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(vel[3:-2], 30.0, atol=1e-6)
        np.testing.assert_array_equal(vel[-2:], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(acc, 0.0, atol=1e-6)

    def test_mismatched_hip_and_scale_shapes_raise(self):
        with self.assertRaises(ValueError):
            normalized_hip_dynamics(
                np.arange(5, dtype=np.float64),
                np.ones(4, dtype=np.float64),
                fps=30.0,
            )



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
