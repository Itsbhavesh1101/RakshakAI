from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Optional[List[int]]:
    if width <= 0 or height <= 0:
        return None

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def normalize_label(label: str) -> str:
    return str(label).lower().strip().replace("-", "_")


def label_matches(label: str, aliases: Sequence[str]) -> bool:
    normalized = normalize_label(label)
    return any(alias in normalized for alias in aliases)


def detection_confidence(detections: Sequence[Dict[str, Any]]) -> float:
    if not detections:
        return 0.0
    return max(float(item["confidence"]) for item in detections)


def box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2])) * max(0, int(box[3]))


def xywh_to_xyxy(box: Sequence[int]) -> Tuple[int, int, int, int]:
    x, y, width, height = [int(value) for value in box]
    return x, y, x + width, y + height


def box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    left_x1, left_y1, left_x2, left_y2 = xywh_to_xyxy(left)
    right_x1, right_y1, right_x2, right_y2 = xywh_to_xyxy(right)

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union else 0.0


def box_center(box: Sequence[int]) -> Tuple[float, float]:
    x, y, width, height = [int(value) for value in box]
    return x + width / 2, y + height / 2


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Sequence[int]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def box_in_ignore_zone(box: Sequence[int], ignore_zones: Sequence[Sequence[Sequence[int]]]) -> bool:
    if not ignore_zones:
        return False
    center = box_center(box)
    return any(point_in_polygon(center, zone) for zone in ignore_zones)


def filter_ignore_zone_detections(
    detections: Sequence[Dict[str, Any]],
    ignore_zones: Sequence[Sequence[Sequence[int]]],
) -> List[Dict[str, Any]]:
    if not ignore_zones:
        return [dict(item) for item in detections]
    return [
        dict(item)
        for item in detections
        if not box_in_ignore_zone(item.get("box", []), ignore_zones)
    ]


def dedupe_detections(detections: Sequence[Dict[str, Any]], iou_thresh: float = 0.55) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for item in sorted(detections, key=lambda hit: float(hit["confidence"]), reverse=True):
        if any(box_iou(item["box"], kept["box"]) >= iou_thresh for kept in deduped):
            continue
        deduped.append(dict(item))
    return deduped


def expand_box(box: Sequence[int], scale: float, width: int, height: int) -> Optional[List[int]]:
    x, y, box_width, box_height = [int(value) for value in box]
    cx = x + box_width / 2
    cy = y + box_height / 2
    new_width = box_width * scale
    new_height = box_height * scale
    x1 = int(round(cx - new_width / 2))
    y1 = int(round(cy - new_height / 2))
    x2 = int(round(cx + new_width / 2))
    y2 = int(round(cy + new_height / 2))
    return clamp_box(x1, y1, x2, y2, width, height)


def draw_label(frame: np.ndarray, box: Sequence[int], text: str, color: Tuple[int, int, int]) -> None:
    x, y, width, height = [int(value) for value in box]
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
    cv2.putText(
        frame,
        text,
        (x, max(16, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
