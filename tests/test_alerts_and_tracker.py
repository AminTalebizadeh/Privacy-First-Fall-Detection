"""Unit tests: tracker identity, alert debounce/cooldown, privacy schema."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pose.tracker import IoUCentroidTracker, iou_xyxy
from src.realtime.alerts import EVENT_SCHEMA, AlertManager


def box_kpts(x, y, w=60, h=160):
    box = np.array([x, y, x + w, y + h], np.float32)
    k = np.zeros((17, 3), np.float32)
    k[:, 0], k[:, 1], k[:, 2] = x + w / 2, y + h / 2, 0.9
    return box, k


class TestTracker(unittest.TestCase):
    def test_iou(self):
        a = np.array([0, 0, 10, 10], float)
        self.assertAlmostEqual(iou_xyxy(a, a), 1.0, places=6)
        self.assertAlmostEqual(iou_xyxy(a, np.array([20, 20, 30, 30], float)), 0.0)

    def test_stable_identity_while_moving(self):
        trk = IoUCentroidTracker(min_hits=1)
        ids = set()
        for t in range(10):
            b, k = box_kpts(100 + 5 * t, 100)
            out = trk.update(b[None], k[None])
            ids.update(o.track_id for o in out)
        self.assertEqual(len(ids), 1)

    def test_centroid_fallback_on_bbox_shape_shock(self):
        """Fall: tall box becomes wide box -> IoU may collapse, centroid saves id."""
        trk = IoUCentroidTracker(min_hits=1, iou_gate=0.6, centroid_gate_px=150)
        b1, k1 = box_kpts(100, 100, w=60, h=160)      # standing
        id1 = trk.update(b1[None], k1[None])[0].track_id
        b2, k2 = box_kpts(70, 200, w=160, h=60)       # lying, shifted
        id2 = trk.update(b2[None], k2[None])[0].track_id
        self.assertEqual(id1, id2)

    def test_track_dies_after_max_age(self):
        trk = IoUCentroidTracker(min_hits=1, max_age=2)
        b, k = box_kpts(100, 100)
        trk.update(b[None], k[None])
        for _ in range(3):
            trk.update(np.zeros((0, 4)), np.zeros((0, 17, 3)))
        self.assertEqual(len(trk.tracks), 0)


class TestAlertDebounce(unittest.TestCase):
    def test_requires_k_consecutive(self):
        am = AlertManager(threshold=0.5, k_consecutive=3, cooldown_s=10)
        self.assertIsNone(am.update(1, 0.9, now=0.0))
        self.assertIsNone(am.update(1, 0.9, now=0.2))
        self.assertIsNotNone(am.update(1, 0.9, now=0.4))   # 3rd -> fires

    def test_single_spike_suppressed(self):
        am = AlertManager(threshold=0.5, k_consecutive=3, cooldown_s=10)
        seq = [0.9, 0.2, 0.9, 0.2, 0.9, 0.2]              # never 3 in a row
        fired = [am.update(1, p, now=i * 0.2) for i, p in enumerate(seq)]
        self.assertTrue(all(f is None for f in fired))

    def test_cooldown_blocks_refire(self):
        am = AlertManager(threshold=0.5, k_consecutive=2, cooldown_s=10)
        am.update(1, 0.9, now=0.0)
        self.assertIsNotNone(am.update(1, 0.9, now=0.5))   # alert @ t=0.5
        # person still on floor: windows stay positive during cooldown
        for t in (1.0, 1.5, 2.0, 5.0, 9.0):
            self.assertIsNone(am.update(1, 0.9, now=t))
        self.assertIsNotNone(am.update(1, 0.9, now=12.0))  # after cooldown

    def test_tracks_independent(self):
        am = AlertManager(threshold=0.5, k_consecutive=2, cooldown_s=10)
        am.update(1, 0.9, now=0.0)
        self.assertIsNone(am.update(2, 0.9, now=0.1))      # track 2's first
        self.assertIsNotNone(am.update(1, 0.9, now=0.2))


class TestPrivacySchema(unittest.TestCase):
    """Acceptance criterion: event log schema contains no image payloads."""

    FORBIDDEN_KEYS = {"image", "frame", "jpeg", "png", "pixels", "data",
                      "b64", "base64", "thumbnail", "crop"}
    MAX_VALUE_LEN = 64          # no field long enough to smuggle an image

    def test_log_schema_is_exact_and_image_free(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "alerts.jsonl"
            am = AlertManager(threshold=0.5, k_consecutive=1, cooldown_s=0,
                              log_path=log)
            am.update(7, 0.93, now=1234.5)
            am.update(8, 0.88, now=1236.0)
            am.close()
            lines = log.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            ev = json.loads(line)
            self.assertEqual(tuple(ev.keys()), EVENT_SCHEMA)
            self.assertFalse(set(ev) & self.FORBIDDEN_KEYS)
            for v in ev.values():
                self.assertLess(len(str(v)), self.MAX_VALUE_LEN)
            self.assertIsInstance(ev["track_id"], int)
            self.assertIsInstance(ev["confidence"], float)
            self.assertIsInstance(ev["timestamp"], (int, float))


if __name__ == "__main__":
    unittest.main()
