"""Debounced fall alerting with bounded per-track history.

The on-disk event schema intentionally remains unchanged:
    {"event", "timestamp", "track_id", "confidence"}
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, IO, Iterable, List, Optional

EVENT_SCHEMA = ("event", "timestamp", "track_id", "confidence")
DEFAULT_HISTORY_SIZE = 512


@dataclass
class _TrackAlertState:
    consecutive: int = 0
    last_alert_ts: float = -1e18
    prob_history: Deque[float] = field(default_factory=deque)


@dataclass
class AlertEvent:
    event: str
    timestamp: float
    track_id: int
    confidence: float

    def to_json(self) -> str:
        payload = {
            "event": self.event,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 4),
        }
        # Keep exact field order and exact public schema for compatibility.
        assert tuple(payload.keys()) == EVENT_SCHEMA
        return json.dumps(payload)


class AlertManager:
    def __init__(self, threshold: float = 0.6,
                 k_consecutive: int = 3,
                 cooldown_s: float = 10.0,
                 log_path: Optional[str | Path] = None,
                 history_size: int = DEFAULT_HISTORY_SIZE):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if k_consecutive < 1:
            raise ValueError("k_consecutive must be >= 1")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")

        self.threshold = float(threshold)
        self.k = int(k_consecutive)
        self.cooldown_s = float(cooldown_s)
        self.history_size = int(history_size)
        self._states: Dict[int, _TrackAlertState] = {}
        self._log: Optional[IO[str]] = None

        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log = path.open("a", buffering=1, encoding="utf-8")

    def update(self, track_id: int, prob: float,
               now: Optional[float] = None) -> Optional[AlertEvent]:
        """Feed one classified window; return an event iff it fires."""
        prob = float(prob)
        if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be finite and in [0, 1]")

        now = time.time() if now is None else float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        track_id = int(track_id)
        state = self._states.get(track_id)
        if state is None:
            state = _TrackAlertState(
                prob_history=deque(maxlen=self.history_size)
            )
            self._states[track_id] = state

        state.prob_history.append(prob)
        if prob >= self.threshold:
            state.consecutive += 1
        else:
            state.consecutive = 0

        in_cooldown = (now - state.last_alert_ts) < self.cooldown_s
        if state.consecutive >= self.k and not in_cooldown:
            state.last_alert_ts = now
            state.consecutive = 0
            event = AlertEvent("fall_alert", now, track_id, prob)
            if self._log is not None:
                self._log.write(event.to_json() + "\n")
            return event
        return None

    def history(self, track_id: int, n: int = 90) -> List[float]:
        state = self._states.get(int(track_id))
        if state is None or n <= 0:
            return []
        return list(state.prob_history)[-n:]

    def retain_tracks(self, active_track_ids: Iterable[int]) -> None:
        """Drop state for tracks the tracker has permanently removed.

        Call once per frame (or periodically) with ``tracker.tracks.keys()``.
        This prevents the state dictionary from growing for the lifetime of
        a long-running camera process.
        """
        active = {int(track_id) for track_id in active_track_ids}
        for track_id in list(self._states):
            if track_id not in active:
                del self._states[track_id]

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
