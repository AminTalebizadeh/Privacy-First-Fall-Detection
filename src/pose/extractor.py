"""YOLOv8n-pose wrapper — inference only, never persists frames.

Privacy note: this module is the ONLY place raw pixels exist, and they
live exclusively in transient memory (the `frame` local). Nothing in
this module (or anywhere else in the repo) writes image data to disk;
`tests/test_privacy_schema.py` guards the event-log side of that
guarantee and `scripts/audit_no_frame_writes.py` greps the codebase for
image-writing calls as a CI check.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


class PoseExtractor:
    """Thin wrapper around ultralytics YOLOv8n-pose.

    Lazily imports ultralytics so that feature/model unit tests can run
    on machines without it installed.
    """

    def __init__(self, model: str = "yolov8n-pose.pt", imgsz: int = 640,
                 conf: float = 0.25, device: str = "cuda:0", half: bool = True):
        from ultralytics import YOLO  # lazy import
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            device, half = "cpu", False
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        # ultralytics>=8.4: `half` is deprecated; use quantize=16 for FP16
        self.quantize = 16 if half else None
        # warm-up so first real frame isn't slow
        self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                           imgsz=imgsz, device=device, quantize=self.quantize,
                           verbose=False)

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose on one BGR frame.

        Returns:
            boxes: (N,4) float32 xyxy
            kpts:  (N,17,3) float32 [x, y, conf] in pixel coords
        """
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, quantize=self.quantize,
                                 verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32))
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        xy = res.keypoints.xy.cpu().numpy().astype(np.float32)         # (N,17,2)
        kc = res.keypoints.conf
        kc = (kc.cpu().numpy().astype(np.float32) if kc is not None
              else np.ones(xy.shape[:2], np.float32))                  # (N,17)
        kpts = np.concatenate([xy, kc[..., None]], axis=-1)
        return boxes, kpts


def iter_video_frames(source, max_frames: Optional[int] = None
                      ) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR frame) from a video file path or webcam index.

    Frames are yielded and dropped — never buffered or written.
    """
    import cv2  # lazy import
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break
    finally:
        cap.release()
