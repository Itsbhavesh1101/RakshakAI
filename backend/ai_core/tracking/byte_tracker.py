from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import system_config
from ai_core.detectors import utils
from ai_core.model_service import YoloModelService


@dataclass
class TrackObservation:
    track_id: Optional[int]
    box: List[int]
    confidence: float
    label: str = "person"
    center: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed_px_s: float = 0.0
    direction: str = "stationary"
    age_frames: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    dwell_seconds: float = 0.0
    path: List[Tuple[float, float]] = field(default_factory=list)
    suspicious_behaviors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        person_label = f"Person {self.track_id}" if self.track_id is not None else "Person"
        return {
            "track_id": self.track_id,
            "label": person_label,
            "class_label": self.label,
            "box": [int(value) for value in self.box],
            "confidence": round(float(self.confidence), 4),
            "center": [round(float(self.center[0]), 2), round(float(self.center[1]), 2)],
            "velocity": [round(float(self.velocity[0]), 2), round(float(self.velocity[1]), 2)],
            "speed_px_s": round(float(self.speed_px_s), 2),
            "direction": self.direction,
            "age_frames": int(self.age_frames),
            "dwell_seconds": round(float(self.dwell_seconds), 2),
            "path": [
                [round(float(point[0]), 2), round(float(point[1]), 2)]
                for point in self.path[-system_config.tracking_history_size:]
            ],
            "suspicious_behaviors": list(self.suspicious_behaviors),
        }


@dataclass
class _TrackState:
    first_seen: float
    last_seen: float
    center: Tuple[float, float]
    path: Deque[Tuple[float, float]]
    age_frames: int = 1


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class ByteTrackPersonTracker:
    """ByteTrack person tracker backed by Ultralytics' built-in tracker."""

    model_name = "yolo11n"

    def __init__(self, model_service: YoloModelService):
        self.model_service = model_service
        self.states: Dict[int, _TrackState] = {}

    def reset(self) -> None:
        self.states.clear()
        model = self.model_service.models.get(self.model_name)
        predictor = getattr(model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            delattr(predictor, "trackers")

    def update(self, frame: np.ndarray, conf_thresh: float) -> List[TrackObservation]:
        if not system_config.tracking_enabled:
            return []

        track_conf = min(conf_thresh, system_config.tracking_confidence_threshold)
        model, results = self.model_service.track(
            self.model_name,
            frame,
            track_conf,
            classes=[0],
            tracker=system_config.tracker_config,
        )
        raw_tracks = self._extract_person_tracks(model, results, frame)
        return self._update_motion_state(raw_tracks)

    def _extract_person_tracks(self, model: Any, results: Any, frame: np.ndarray) -> List[TrackObservation]:
        frame_height, frame_width = frame.shape[:2]
        min_person_area = max(250, int(frame_width * frame_height * 0.0012))
        tracks: List[TrackObservation] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_values = _to_numpy(getattr(boxes, "xyxy", None))
            conf_values = _to_numpy(getattr(boxes, "conf", None))
            cls_values = _to_numpy(getattr(boxes, "cls", None))
            id_values = _to_numpy(getattr(boxes, "id", None))
            if xyxy_values is None or conf_values is None or cls_values is None:
                continue

            names = getattr(result, "names", None) or getattr(model, "names", {})
            for index, xyxy in enumerate(xyxy_values):
                cls = int(cls_values[index])
                label = utils.normalize_label(str(names.get(cls, cls)))
                if label != "person":
                    continue

                x1, y1, x2, y2 = [int(value) for value in xyxy]
                box = utils.clamp_box(x1, y1, x2, y2, frame_width, frame_height)
                if box is None or utils.box_area(box) < min_person_area:
                    continue

                track_id = None
                if id_values is not None and index < len(id_values):
                    track_id = int(id_values[index])
                center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                tracks.append(
                    TrackObservation(
                        track_id=track_id,
                        box=box,
                        confidence=float(conf_values[index]),
                        label=label,
                        center=center,
                        first_seen=time.time(),
                        last_seen=time.time(),
                    )
                )

        return tracks

    def _update_motion_state(self, tracks: Sequence[TrackObservation]) -> List[TrackObservation]:
        now = time.time()
        observed_ids = set()
        enriched: List[TrackObservation] = []

        for track in tracks:
            if track.track_id is None:
                track.first_seen = now
                track.last_seen = now
                track.path = [track.center]
                enriched.append(track)
                continue

            observed_ids.add(track.track_id)
            previous = self.states.get(track.track_id)
            if previous is None:
                path: Deque[Tuple[float, float]] = deque(maxlen=system_config.tracking_history_size)
                path.append(track.center)
                state = _TrackState(
                    first_seen=now,
                    last_seen=now,
                    center=track.center,
                    path=path,
                    age_frames=1,
                )
                velocity = (0.0, 0.0)
                speed = 0.0
            else:
                dt = max(now - previous.last_seen, 1e-3)
                dx = track.center[0] - previous.center[0]
                dy = track.center[1] - previous.center[1]
                velocity = (dx / dt, dy / dt)
                speed = math.hypot(velocity[0], velocity[1])
                previous.path.append(track.center)
                previous.center = track.center
                previous.last_seen = now
                previous.age_frames += 1
                state = previous

            self.states[track.track_id] = state
            track.first_seen = state.first_seen
            track.last_seen = now
            track.age_frames = state.age_frames
            track.velocity = velocity
            track.speed_px_s = speed
            track.direction = self._direction_from_velocity(velocity)
            track.dwell_seconds = now - state.first_seen
            track.path = list(state.path)
            if speed >= system_config.running_speed_px_s:
                track.suspicious_behaviors.append("running_behavior")
            enriched.append(track)

        self._cleanup_missing(now, observed_ids)
        return enriched

    def _cleanup_missing(self, now: float, observed_ids: set[int]) -> None:
        expired = [
            track_id for track_id, state in self.states.items()
            if track_id not in observed_ids and now - state.last_seen > system_config.tracking_lost_seconds
        ]
        for track_id in expired:
            self.states.pop(track_id, None)

    @staticmethod
    def _direction_from_velocity(velocity: Tuple[float, float]) -> str:
        vx, vy = velocity
        if math.hypot(vx, vy) < 35.0:
            return "stationary"

        horizontal = ""
        vertical = ""
        if abs(vx) >= 25.0:
            horizontal = "right" if vx > 0 else "left"
        if abs(vy) >= 25.0:
            vertical = "down" if vy > 0 else "up"
        if horizontal and vertical:
            return f"{vertical}-{horizontal}"
        return vertical or horizontal or "moving"
