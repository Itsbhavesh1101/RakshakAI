import time
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np


def _point_inside_or_near(poly_pts: np.ndarray, point: Tuple[float, float], margin_px: float) -> bool:
    return cv2.pointPolygonTest(poly_pts, point, True) >= -margin_px


def _box_zone_overlap_ratio(box: List[int], poly_pts: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    h, w = frame_shape
    x, y, bw, bh = box
    x1 = max(0, min(w - 1, x))
    y1 = max(0, min(h - 1, y))
    x2 = max(0, min(w, x + bw))
    y2 = max(0, min(h, y + bh))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    zone_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [poly_pts], 255)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
    overlap = cv2.countNonZero(cv2.bitwise_and(zone_mask, box_mask))
    return overlap / max(1, (x2 - x1) * (y2 - y1))


def _box_in_restricted_zone(
    box: List[int],
    poly_pts: np.ndarray,
    frame_shape: Tuple[int, int],
    margin_px: float,
) -> bool:
    bx, by, bw, bh = box
    cx = bx + bw / 2
    left_x = bx + bw * 0.28
    right_x = bx + bw * 0.72
    center_y = by + bh * 0.50
    lower_y = by + bh * 0.78
    foot_y = by + bh

    anchors = [
        (cx, foot_y),
        (left_x, foot_y),
        (right_x, foot_y),
        (cx, lower_y),
        (cx, center_y),
    ]
    if any(_point_inside_or_near(poly_pts, point, margin_px) for point in anchors):
        return True

    return _box_zone_overlap_ratio(box, poly_pts, frame_shape) >= 0.18

def check_restricted_zones(
    frame: np.ndarray, 
    coords: List[Tuple[int, int]], 
    person_boxes: List[List[int]],
    authorized_people: List[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[List[int]]]:
    """
    Renders a premium semi-transparent polygon zone overlay.
    Validates if person foot-coordinates are inside the polygon using cv2.pointPolygonTest.
    
    Args:
        frame: The raw image frame.
        coords: List of polygon coordinate nodes. If empty, default values are assigned.
        person_boxes: BBoxes of people detected: [[x, y, w, h], ...]
        
    Returns:
        - annotated_frame: Frame with visual overlays drawn.
        - alerts: List of unified alert dicts triggered.
        - violators: List of unauthorized bounding boxes violating the boundary.
    """
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    _ = authorized_people
    
    # 1. Fallback default restricted polygon
    if not coords or len(coords) < 3:
        # Default polygon spanning the lower-mid track region
        coords = [
            (int(w * 0.15), int(h * 0.65)),
            (int(w * 0.85), int(h * 0.65)),
            (int(w * 0.95), int(h * 0.95)),
            (int(w * 0.05), int(h * 0.95))
        ]
        
    poly_pts = np.array(coords, dtype=np.int32)
    
    # 2. Check each person's feet coordinate
    zone_entries = []
    margin_px = max(6.0, min(w, h) * 0.012)
    for box in person_boxes:
        if _box_in_restricted_zone(box, poly_pts, (h, w), margin_px):
            zone_entries.append(box)

    violators = zone_entries
            
    # 3. Draw premium semi-transparent overlay
    # Pulse color on alert status (using time for a soft flashing neon border effect)
    violation_active = len(violators) > 0
    t = time.time()
    
    if violation_active:
        # Flashing crimson red
        alpha = 0.35 + 0.1 * np.sin(t * 8)
        color_fill = (0, 0, 180)
        color_line = (0, 0, 255)
        text_color = (0, 0, 255)
        text_str = "WARNING: RESTRICTED ZONE INTRUSION!"
    else:
        # Harmony safe green
        alpha = 0.20
        color_fill = (0, 180, 0)
        color_line = (0, 255, 0)
        text_color = (0, 255, 0)
        text_str = "PERIMETER WATCH ACTIVE"
        
    # Draw transparent filled polygon
    overlay = annotated_frame.copy()
    cv2.fillPoly(overlay, [poly_pts], color_fill)
    annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1.0 - alpha, 0)
    
    # Draw thick neon contour lines
    cv2.polylines(annotated_frame, [poly_pts], True, color_line, 2, cv2.LINE_AA)
    
    # 4. Highlight violators. Authorized roster entries are not allowed to suppress
    # intrusion until a verified identity matcher can map a person to a box.
    for box in violators:
        bx, by, bw, bh = box
        cx = bx + bw // 2
        foot_y = by + bh
        
        # Red bounding box + neon alert HUD marker
        cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
        cv2.circle(annotated_frame, (cx, foot_y), 6, (0, 0, 255), -1)
        cv2.putText(
            annotated_frame, 
            "INTRUDER", 
            (bx, by - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.55, 
            (0, 0, 255), 
            2,
            cv2.LINE_AA
        )
        
    # Draw status overlay header card
    cv2.rectangle(annotated_frame, (8, h - 35), (380, h - 8), (15, 15, 22), -1)
    cv2.putText(
        annotated_frame, 
        text_str, 
        (18, h - 16), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        text_color, 
        1, 
        cv2.LINE_AA
    )
    
    alerts = []
    if violation_active:
        alerts.append({
            "camera_id": "CAM_MAIN_ENTRANCE_01",
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "anomaly_detected": True,
            "type": "trespassing_detection",
            "severity": "CRITICAL",
            "confidence": 0.95,
            "bounding_boxes": violators,
            "live_count": len(violators),
            "authorized_count": 0,
        })
        
    return annotated_frame, alerts, violators
