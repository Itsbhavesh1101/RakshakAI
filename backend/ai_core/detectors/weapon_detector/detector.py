from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from config import system_config
from ai_core.detectors import utils
from ai_core.model_service import YoloModelService


WEAPON_ALIASES = {
    "weapon",
    "gun",
    "guns",
    "pistol",
    "rifle",
    "knife",
    "knives",
    "firearm",
    "handgun",
    "revolver",
    "shotgun",
    "blade",
}


class WeaponDetector:
    """Specialized detector for small weapon candidates near people."""

    module_name = "weapon_detection"
    model_name = "weapon"

    def __init__(self, model_service: YoloModelService):
        self.model_service = model_service

    def preload(self) -> None:
        self.model_service.load_model(self.model_name)

    def label_matches(self, label: str) -> bool:
        return utils.label_matches(label, WEAPON_ALIASES)

    @staticmethod
    def is_near_person(candidate_box: List[int], person_boxes: List[List[int]]) -> bool:
        if not person_boxes:
            return True

        x, y, width, height = candidate_box
        cx = x + width / 2
        cy = y + height / 2
        for pbox in person_boxes:
            px, py, pw, ph = pbox
            pcx = px + pw / 2
            pcy = py + ph / 2
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            max_dist = max(180.0, ph * 1.5, pw * 2.0)
            if dist <= max_dist:
                return True
        return False

    def detect(
        self,
        frame: np.ndarray,
        person_boxes: List[List[int]],
        conf_thresh: float,
    ) -> List[Dict[str, Any]]:
        frame_height, frame_width = frame.shape[:2]
        model, results = self.model_service.predict(self.model_name, frame, conf_thresh)
        hits = [
            item for item in self.model_service.extract_detections(
                model,
                results,
                frame_width,
                frame_height,
                conf_thresh,
            )
            if self.label_matches(item["label"]) and self.is_near_person(item["box"], person_boxes)
        ]
        hits.extend(self.detect_in_person_rois(frame, person_boxes, conf_thresh))
        return utils.dedupe_detections(hits, iou_thresh=0.50)

    def detect_in_person_rois(
        self,
        frame: np.ndarray,
        person_boxes: List[List[int]],
        conf_thresh: float,
    ) -> List[Dict[str, Any]]:
        if not system_config.weapon_roi_pass_enabled or not person_boxes:
            return []

        frame_height, frame_width = frame.shape[:2]
        min_area = max(200, int(frame_width * frame_height * 0.00035))
        candidates: List[Dict[str, Any]] = []
        largest_people = sorted(person_boxes, key=utils.box_area, reverse=True)[:system_config.weapon_roi_max_persons]

        for person_box in largest_people:
            roi_box = utils.expand_box(person_box, system_config.weapon_roi_scale, frame_width, frame_height)
            if roi_box is None:
                continue

            rx, ry, rw, rh = roi_box
            roi = frame[ry:ry + rh, rx:rx + rw]
            if roi.size == 0:
                continue

            roi_conf = max(0.05, conf_thresh * 0.75)
            try:
                model, results = self.model_service.predict(
                    self.model_name,
                    roi,
                    roi_conf,
                    imgsz=system_config.weapon_roi_imgsz,
                )
            except Exception as exc:
                print(f"Error in weapon ROI pass: {exc}")
                continue

            for item in self.model_service.extract_detections(model, results, rw, rh, roi_conf):
                if not self.label_matches(item["label"]):
                    continue

                x, y, width, height = item["box"]
                full_box = utils.clamp_box(rx + x, ry + y, rx + x + width, ry + y + height, frame_width, frame_height)
                if full_box is None or utils.box_area(full_box) < min_area:
                    continue

                candidates.append({
                    "box": full_box,
                    "confidence": min(0.99, float(item["confidence"]) + 0.08),
                    "label": item["label"],
                    "source": "person_roi",
                })

        return candidates
