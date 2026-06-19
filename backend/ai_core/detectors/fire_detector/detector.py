from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from config import system_config
from ai_core.detectors import utils
from ai_core.model_service import YoloModelService


FIRE_ALIASES = {
    "fire",
    "flame",
    "flames",
    "smoke",
    "default",
}


class FireSmokeDetector:
    """Specialized fire/smoke detector with texture-color validation hooks."""

    module_name = "fire_detection"
    model_name = "fire"

    def __init__(self, model_service: YoloModelService):
        self.model_service = model_service

    def preload(self) -> None:
        self.model_service.load_model(self.model_name)

    def label_matches(self, label: str) -> bool:
        return utils.label_matches(label, FIRE_ALIASES)

    @staticmethod
    def fire_color_density(frame: np.ndarray, box: List[int]) -> float:
        x, y, width, height = box
        roi = frame[y:y + height, x:x + width]
        if roi.size == 0:
            return 0.0

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([0, 70, 90])
        upper_orange = np.array([35, 255, 255])
        lower_red = np.array([165, 70, 90])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_roi, lower_orange, upper_orange)
        mask2 = cv2.inRange(hsv_roi, lower_red, upper_red)
        fire_mask = cv2.bitwise_or(mask1, mask2)
        return cv2.countNonZero(fire_mask) / max(1, roi.shape[0] * roi.shape[1])

    def detect(self, frame: np.ndarray, conf_thresh: float) -> List[Dict[str, Any]]:
        frame_height, frame_width = frame.shape[:2]
        model, results = self.model_service.predict(self.model_name, frame, conf_thresh)
        hits: List[Dict[str, Any]] = []

        for item in self.model_service.extract_detections(model, results, frame_width, frame_height, conf_thresh):
            label = item["label"]
            if not self.label_matches(label):
                continue

            is_smoke = "smoke" in label
            color_density = self.fire_color_density(frame, item["box"])
            _, _, box_width, box_height = item["box"]
            area_ratio = (box_width * box_height) / max(1, frame_width * frame_height)

            if not is_smoke and color_density < system_config.fire_min_color_density:
                continue
            if is_smoke and area_ratio < 0.002 and item["confidence"] < 0.62:
                continue

            item["color_density"] = color_density
            hits.append(item)

        return utils.dedupe_detections(hits, iou_thresh=0.55)
