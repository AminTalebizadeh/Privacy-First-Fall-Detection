"""Debounced fall alerting with cooldown and a privacy-safe JSON log.

Rules (all exposed via configs/default.yaml -> alerts):
  * A window is "positive" iff p(fall) >= threshold.
  * An alert fires only after K CONSECUTIVE positive windows for the
    same track (debouncing: single-window spikes from a fast sit-down
    or a tracking glitch don't page anyone).
  * After firing, the track enters a cooldown of `cooldown_s` seconds
    during which no re-fire occurs (one fall == one alert, not one
    alert per window while the person is on the floor).

Event log: JSON-lines, schema fixed to
    {"event", "timestamp", "track_id", "confidence"}
— by construction there is no field that could carry image data, and
`tests/test_privacy_schema.py` asserts exactly that.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, IO, List, Optional

EVENT_SCHEMA = ("event", "timestamp", "track_id", "confidence")


@dataclass
class _TrackAlertState:
    consecutive: int = 0
    last_alert_ts: float = -1e18
    prob_history: List[float] = field(default_factory=list)


@dataclass
class AlertEvent:
    event: str
    timestamp: float
    track_id: int
    confidence: float

    def to_json(self) -> str:
        d = {"event": self.event, "timestamp": self.timestamp,
             "track_id": self.track_id, "confidence": round(self.confidence, 4)}
        assert tuple(d.keys()) == EVENT_SCHEMA  # schema is frozen
        return json.dumps(d)


class AlertManager:
    def __init__(self, threshold: float = 0.6, k_consecutive: int = 3,
                 cooldown_s: float = 10.0,
                 log_path: Optional[str | Path] = None):
        self.threshold = threshold
        self.k = k_consecutive
        self.cooldown_s = cooldown_s
        self._states: Dict[int, _TrackAlertState] = {}
        self._log: Optional[IO] = (open(log_path, "a", buffering=1)
                                   if log_path else None)

    # ------------------------------------------------------------------
    def update(self, track_id: int, prob: float,
               now: Optional[float] = None) -> Optional[AlertEvent]:
        """Feed one classified window; returns an AlertEvent iff it fires."""
        now = time.time() if now is None else now
        st = self._states.setdefault(track_id, _TrackAlertState())
        st.prob_history.append(float(prob))

        if prob >= self.threshold:
            st.consecutive += 1
        else:
            st.consecutive = 0

        in_cooldown = (now - st.last_alert_ts) < self.cooldown_s
        if st.consecutive >= self.k and not in_cooldown:
            st.last_alert_ts = now
            st.consecutive = 0                      # re-arm from scratch
            ev = AlertEvent("fall_alert", now, int(track_id), float(prob))
            if self._log:
                self._log.write(ev.to_json() + "\n")
            return ev
        return None

    # ------------------------------------------------------------------
    def history(self, track_id: int, n: int = 90) -> List[float]:
        st = self._states.get(track_id)
        return st.prob_history[-n:] if st else []

    def close(self) -> None:
        if self._log:
            self._log.close()
            self._log = None
