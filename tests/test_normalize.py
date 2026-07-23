"""Unit tests: torso normalization & frame features on synthetic skeletons."""

import math
import unittest

import numpy as np

from src.features.frame_features import (L_HIP, L_SHOULDER, R_HIP, R_SHOULDER,
                                         bbox_aspect_ratio, torso_normalize,
                                         trunk_angle)


def synthetic_standing(cx=320.0, hip_y=300.0, torso=100.0, conf=0.9):
    """Upright stick figure: shoulders directly above hips."""
    k = np.zeros((17, 3), np.float32)
    # default every joint to the body midline so unspecified joints are
    # still positioned *relative to the person*, not at absolute (0,0)
    k[:, 0], k[:, 1], k[:, 2] = cx, hip_y - torso / 2, conf
    k[L_SHOULDER] = [cx - 30, hip_y - torso, conf]
    k[R_SHOULDER] = [cx + 30, hip_y - torso, conf]
    k[L_HIP] = [cx - 20, hip_y, conf]
    k[R_HIP] = [cx + 20, hip_y, conf]
    k[0] = [cx, hip_y - torso - 40, conf]            # nose
    k[15] = [cx - 20, hip_y + 180, conf]             # ankles
    k[16] = [cx + 20, hip_y + 180, conf]
    return k


class TestTorsoNormalize(unittest.TestCase):
    def test_translation_invariance(self):
        a = synthetic_standing(cx=100, hip_y=200)
        b = synthetic_standing(cx=500, hip_y=420)    # same pose, moved
        na, _, _ = torso_normalize(a)
        nb, _, _ = torso_normalize(b)
        np.testing.assert_allclose(na, nb, atol=1e-5)

    def test_scale_invariance(self):
        near = synthetic_standing(torso=200.0)       # person close to camera
        far = synthetic_standing(torso=50.0)         # same pose, far away
        # scale limb offsets consistently
        far[:, :2] = (far[:, :2] - far[[L_HIP, R_HIP], :2].mean(0)) * 1.0 \
            + far[[L_HIP, R_HIP], :2].mean(0)
        nn, _, sn = torso_normalize(near)
        nf, _, sf = torso_normalize(far)
        # torso length in normalized space must be exactly 1 for both
        for n in (nn, nf):
            hip_c = (n[L_HIP] + n[R_HIP]) / 2
            sho_c = (n[L_SHOULDER] + n[R_SHOULDER]) / 2
            self.assertAlmostEqual(float(np.linalg.norm(sho_c - hip_c)),
                                   1.0, places=5)
        self.assertAlmostEqual(sn, 200.0, places=3)
        self.assertAlmostEqual(sf, 50.0, places=3)

    def test_hip_center_at_origin(self):
        n, _, _ = torso_normalize(synthetic_standing())
        np.testing.assert_allclose((n[L_HIP] + n[R_HIP]) / 2, [0, 0], atol=1e-6)

    def test_low_conf_masked(self):
        k = synthetic_standing()
        k[9, 2] = 0.05                               # right wrist unreliable
        n, m, _ = torso_normalize(k, min_conf=0.30)
        self.assertEqual(m[9], 0.0)
        np.testing.assert_allclose(n[9], [0, 0], atol=1e-6)

    def test_occluded_torso_uses_fallback_scale(self):
        k = synthetic_standing()
        k[[L_HIP, R_HIP, L_SHOULDER, R_SHOULDER], 2] = 0.0
        _, _, s = torso_normalize(k, fallback_scale=123.0)
        self.assertAlmostEqual(s, 123.0, places=5)


class TestTrunkAngleAndAspect(unittest.TestCase):
    def test_standing_angle_zero(self):
        self.assertAlmostEqual(trunk_angle(synthetic_standing()), 0.0, places=5)

    def test_lying_angle_90deg(self):
        k = synthetic_standing()
        # rotate every point 90 deg about the hip center -> horizontal body
        hip_c = k[[L_HIP, R_HIP], :2].mean(0)
        rot = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)
        k[:, :2] = (k[:, :2] - hip_c) @ rot.T + hip_c
        self.assertAlmostEqual(trunk_angle(k), math.pi / 2, places=5)

    def test_angle_nan_when_torso_occluded(self):
        k = synthetic_standing()
        k[[L_HIP, R_HIP], 2] = 0.0
        self.assertTrue(math.isnan(trunk_angle(k)))

    def test_aspect_standing_lt_1_lying_gt_1(self):
        stand = synthetic_standing()
        self.assertLess(bbox_aspect_ratio(stand), 1.0)
        lie = stand.copy()
        lie[:, :2] = lie[:, [1, 0]][:, :2]           # swap x/y -> horizontal
        self.assertGreater(bbox_aspect_ratio(lie), 1.0)


if __name__ == "__main__":
    unittest.main()
