from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from config import system_config
from ai_core.detectors import utils
from ai_core.model_service import YoloModelService
from ai_core.tracking import ByteTrackPersonTracker


class IntrusionPersonDetector:
    """Person localization layer shared by intrusion, crowd, and weapon context."""

    model_name = "yolo11n"

    def __init__(self, model_service: YoloModelService):
        self.model_service = model_service
        self.tracker = ByteTrackPersonTracker(model_service)

    def preload(self) -> None:
        self.model_service.load_model(self.model_name)

    def detect_people(self, frame: np.ndarray, conf_thresh: float) -> List[List[int]]:
        return [person["box"] for person in self.detect_tracked_people(frame, conf_thresh)]

    def detect_tracked_people(self, frame: np.ndarray, conf_thresh: float) -> List[Dict[str, Any]]:
        if system_config.tracking_enabled:
            try:
                return [track.as_dict() for track in self.tracker.update(frame, conf_thresh)]
            except Exception as exc:
                print(f"ByteTrack person tracking failed; falling back to detection-only people: {exc}")

        frame_height, frame_width = frame.shape[:2]
        model, results = self.model_service.predict(self.model_name, frame, conf_thresh)
        min_person_area = max(250, int(frame_width * frame_height * 0.0012))
        people: List[Dict[str, Any]] = []

        for item in self.model_service.extract_detections(model, results, frame_width, frame_height, conf_thresh):
            if item["label"] != "person":
                continue
            if utils.box_area(item["box"]) < min_person_area:
                continue
            x, y, width, height = item["box"]
            people.append({
                "track_id": None,
                "label": "Person",
                "class_label": "person",
                "box": item["box"],
                "confidence": round(float(item["confidence"]), 4),
                "center": [round(x + width / 2, 2), round(y + height / 2, 2)],
                "velocity": [0.0, 0.0],
                "speed_px_s": 0.0,
                "direction": "unknown",
                "age_frames": 1,
                "dwell_seconds": 0.0,
                "path": [[round(x + width / 2, 2), round(y + height / 2, 2)]],
                "suspicious_behaviors": [],
            })

        return people
