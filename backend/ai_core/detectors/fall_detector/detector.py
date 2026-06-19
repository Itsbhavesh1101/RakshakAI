from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from config import system_config
from ai_core.detectors import utils
from ai_core.model_service import YoloModelService


FALL_ALIASES = {
    "fall",
    "fallen",
    "person_fall",
    "person falling",
    "lying",
    "down",
}


class FallDetector:
    """Specialized fall detector with posture fallback validation."""

    module_name = "fall_detection"
    model_name = "fall"

    def __init__(self, model_service: YoloModelService):
        self.model_service = model_service

    def preload(self) -> None:
        self.model_service.load_model(self.model_name)

    def label_matches(self, label: str) -> bool:
        return utils.label_matches(label, FALL_ALIASES)

    def detect(
        self,
        frame: np.ndarray,
        person_boxes: List[List[int]],
        conf_thresh: float,
    ) -> List[Dict[str, Any]]:
        frame_height, frame_width = frame.shape[:2]
        model, results = self.model_service.predict(self.model_name, frame, conf_thresh)
        hits: List[Dict[str, Any]] = []

        for item in self.model_service.extract_detections(model, results, frame_width, frame_height, conf_thresh):
            if not self.label_matches(item["label"]):
                continue

            _, _, box_width, box_height = item["box"]
            aspect_ratio = box_width / max(1, box_height)
            area_ratio = (box_width * box_height) / max(1, frame_width * frame_height)
            if aspect_ratio >= 0.70 and area_ratio >= 0.002:
                hits.append(item)

        if system_config.fall_posture_fallback_enabled:
            hits.extend(self._posture_candidates(person_boxes, frame_width, frame_height))

        return utils.dedupe_detections(hits, iou_thresh=0.50)

    @staticmethod
    def _posture_candidates(
        person_boxes: List[List[int]],
        frame_width: int,
        frame_height: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for pbox in person_boxes:
            px, py, person_width, person_height = pbox
            aspect_ratio = person_width / max(1, person_height)
            area_ratio = (person_width * person_height) / max(1, frame_width * frame_height)
            lower_frame_bias = (py + person_height) / max(1, frame_height)
            if aspect_ratio >= 1.20 and area_ratio >= 0.006 and lower_frame_bias >= 0.42:
                candidates.append({
                    "box": pbox,
                    "confidence": min(0.74, 0.48 + (aspect_ratio - 1.20) * 0.18),
                    "label": "posture_fall_candidate",
                    "source": "person_posture",
                })
        return candidates
