from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import MODELS_DIR, system_config
from ai_core.advanced_ai import AdvancedBehaviorAnalyzer
from ai_core.agents import MultiAgentOrchestrator
from ai_core.detectors import utils
from ai_core.detectors.fall_detector import FALL_ALIASES, FallDetector
from ai_core.detectors.fire_detector import FIRE_ALIASES, FireSmokeDetector
from ai_core.detectors.intrusion_detector import IntrusionPersonDetector
from ai_core.detectors.weapon_detector import WEAPON_ALIASES, WeaponDetector
from ai_core.model_service import YoloModelService
from ai_core.scene_memory import SceneMemoryStore
from ai_core.temporal_validation.scorer import TemporalThreatScorer
from ai_core.threat_engine import ThreatScoringEngine
from ai_core.tracking import RestrictedZoneMotionTracker


class SurveillanceDetectionPipeline:
    """Stable Phase 1 detection core with specialized detector ownership."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.model_service = YoloModelService(models_dir)
        self.device = self.model_service.device
        self.use_half_precision = self.model_service.use_half_precision
        self.models = self.model_service.models
        self.model_files = self.model_service.model_files

        self.temporal_scorer = TemporalThreatScorer()
        self.persistence_threshold = self.temporal_scorer.persistence_threshold
        self.crowd_count_threshold = 12
        self.class_aliases = {
            "weapon_detection": WEAPON_ALIASES,
            "fire_detection": FIRE_ALIASES,
            "fall_detection": FALL_ALIASES,
        }

        self.person_detector = IntrusionPersonDetector(self.model_service)
        self.weapon_detector = WeaponDetector(self.model_service)
        self.fire_detector = FireSmokeDetector(self.model_service)
        self.fall_detector = FallDetector(self.model_service)
        self.zone_motion_tracker = RestrictedZoneMotionTracker()
        self.threat_scoring_engine = ThreatScoringEngine()
        self.scene_memory = SceneMemoryStore()
        self.multi_agent_orchestrator = MultiAgentOrchestrator()
        self.advanced_behavior_analyzer = AdvancedBehaviorAnalyzer()
        self.detectors = {
            "weapon_detection": self.weapon_detector,
            "fire_detection": self.fire_detector,
            "fall_detection": self.fall_detector,
            "crowd_detection": self.person_detector,
            "trespassing_detection": self.person_detector,
        }

        print(f"DetectionEngine initialized. Using active device for YOLO: '{self.device}'")

    @property
    def consecutive_detections(self) -> Dict[str, int]:
        return self.temporal_scorer.consecutive_detections

    @consecutive_detections.setter
    def consecutive_detections(self, values: Dict[str, int]) -> None:
        self.temporal_scorer.consecutive_detections = values

    @property
    def temporal_scores(self) -> Dict[str, float]:
        return self.temporal_scorer.temporal_scores

    @temporal_scores.setter
    def temporal_scores(self, values: Dict[str, float]) -> None:
        self.temporal_scorer.temporal_scores = values

    def preload_module(self, module: str) -> None:
        normalized_module = system_config.normalize_module(module)
        detector = self.detectors.get(normalized_module)
        if detector is not None:
            detector.preload()

    def reset_tracking(self) -> None:
        self.person_detector.tracker.reset()
        self.zone_motion_tracker.reset()
        self.temporal_scorer.reset()
        self.scene_memory.reset()

    def score_alerts(
        self,
        alerts: List[Dict[str, Any]],
        tracked_people: List[Dict[str, Any]],
        zone_updates: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return self.threat_scoring_engine.score_alerts(alerts, tracked_people, zone_updates)

    def remember_scene(
        self,
        alerts: List[Dict[str, Any]],
        tracked_people: List[Dict[str, Any]],
        scene_threat: Dict[str, Any],
        zone_updates: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not system_config.scene_memory_enabled:
            return {
                "created_events": [],
                "timeline": [],
                "object_history": {},
                "memory_size": 0,
                "backend": self.scene_memory.active_backend,
            }
        return self.scene_memory.record_frame(alerts, tracked_people, scene_threat, zone_updates)

    def analyze_advanced_behavior(
        self,
        tracked_people: List[Dict[str, Any]],
        alerts: List[Dict[str, Any]],
        zone_updates: Optional[Dict[int, Dict[str, Any]]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.advanced_behavior_analyzer.analyze(
            tracked_people,
            alerts,
            zone_updates or {},
            stats or {},
        )

    def run_multi_agent_analysis(
        self,
        alerts: List[Dict[str, Any]],
        tracked_people: List[Dict[str, Any]],
        stats: Dict[str, Any],
        zone_updates: Optional[Dict[int, Dict[str, Any]]] = None,
        scene_memory: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        report = self.multi_agent_orchestrator.analyze(
            alerts,
            tracked_people,
            stats,
            zone_updates or {},
            scene_memory or {},
        )
        enriched_alerts = self.multi_agent_orchestrator.enrich_alerts(alerts, report)
        report["input_alert_count"] = len(enriched_alerts)

        if system_config.multi_agent_suppress_unverified:
            filtered_alerts = [
                alert for alert in enriched_alerts
                if (alert.get("agent_verification") or {}).get("decision") == "verified"
            ]
        else:
            filtered_alerts = enriched_alerts

        report["output_alert_count"] = len(filtered_alerts)
        report["suppressed_alert_count"] = len(enriched_alerts) - len(filtered_alerts)
        return filtered_alerts, report

    def _load_model(self, name: str):
        return self.model_service.load_model(name)

    def _predict(self, name: str, frame: np.ndarray, conf_thresh: float, imgsz: Optional[int] = None):
        return self.model_service.predict(name, frame, conf_thresh, imgsz)

    def _extract_detections(
        self,
        model,
        results: Any,
        frame_width: int,
        frame_height: int,
        conf_thresh: float,
    ) -> List[Dict[str, Any]]:
        return self.model_service.extract_detections(model, results, frame_width, frame_height, conf_thresh)

    @staticmethod
    def _clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Optional[List[int]]:
        return utils.clamp_box(x1, y1, x2, y2, width, height)

    @staticmethod
    def _normalize_label(label: str) -> str:
        return utils.normalize_label(label)

    def _label_matches(self, module: str, label: str) -> bool:
        return utils.label_matches(label, self.class_aliases.get(module, set()))

    @staticmethod
    def _confidence(alert_boxes: List[Dict[str, Any]]) -> float:
        return utils.detection_confidence(alert_boxes)

    @staticmethod
    def _box_area(box: List[int]) -> int:
        return utils.box_area(box)

    @staticmethod
    def _xywh_to_xyxy(box: List[int]) -> Tuple[int, int, int, int]:
        return utils.xywh_to_xyxy(box)

    @classmethod
    def _box_iou(cls, left: List[int], right: List[int]) -> float:
        return utils.box_iou(left, right)

    @classmethod
    def _dedupe_detections(cls, detections: List[Dict[str, Any]], iou_thresh: float = 0.55) -> List[Dict[str, Any]]:
        return utils.dedupe_detections(detections, iou_thresh)

    @staticmethod
    def _is_near_person(candidate_box: List[int], person_boxes: List[List[int]]) -> bool:
        return WeaponDetector.is_near_person(candidate_box, person_boxes)

    @staticmethod
    def _fire_color_density(frame: np.ndarray, box: List[int]) -> float:
        return FireSmokeDetector.fire_color_density(frame, box)

    @staticmethod
    def _expand_box(box: List[int], scale: float, width: int, height: int) -> Optional[List[int]]:
        return utils.expand_box(box, scale, width, height)

    def _detect_weapons_in_person_rois(
        self,
        frame: np.ndarray,
        person_boxes: List[List[int]],
        conf_thresh: float,
    ) -> List[Dict[str, Any]]:
        return self.weapon_detector.detect_in_person_rois(frame, person_boxes, conf_thresh)

    def _update_temporal_score(self, module: str, hits: List[Dict[str, Any]], weight: float = 1.0) -> float:
        return self.temporal_scorer.update_score(module, hits, weight)

    def _should_alert(self, module: str, hits: List[Dict[str, Any]], weight: float = 1.0) -> Tuple[bool, float, float]:
        return self.temporal_scorer.should_alert(module, hits, weight)

    @staticmethod
    def _draw_label(frame: np.ndarray, box: List[int], text: str, color: Tuple[int, int, int]) -> None:
        utils.draw_label(frame, box, text, color)

    def _person_threshold(self, enabled: set[str], confidence_thresholds: Dict[str, float]) -> float:
        modules = ("crowd_detection", "trespassing_detection", "weapon_detection", "fall_detection")
        values = [
            confidence_thresholds.get(module, 0.40)
            for module in modules
            if module in enabled
        ]
        if not values:
            values = [system_config.person_confidence_threshold]
        return min(min(values), system_config.person_confidence_threshold)

    def _detect_tracked_people(
        self,
        frame: np.ndarray,
        enabled: set[str],
        confidence_thresholds: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        if not {
            "crowd_detection",
            "trespassing_detection",
            "weapon_detection",
            "fall_detection",
        }.intersection(enabled):
            return []

        try:
            return self.person_detector.detect_tracked_people(
                frame,
                self._person_threshold(enabled, confidence_thresholds),
            )
        except Exception as exc:
            print(f"Error in pre-detecting people: {exc}")
            return []

    def _detect_people(
        self,
        frame: np.ndarray,
        enabled: set[str],
        confidence_thresholds: Dict[str, float],
    ) -> List[List[int]]:
        return [
            person["box"] for person in self._detect_tracked_people(frame, enabled, confidence_thresholds)
        ]

    @staticmethod
    def _person_boxes(tracked_people: List[Dict[str, Any]]) -> List[List[int]]:
        return [[int(value) for value in person["box"]] for person in tracked_people]

    @staticmethod
    def _track_ids_for_boxes(
        boxes: List[List[int]],
        tracked_people: List[Dict[str, Any]],
        iou_thresh: float = 0.15,
    ) -> List[int]:
        track_ids: List[int] = []
        for box in boxes:
            best_track_id = None
            best_iou = 0.0
            for person in tracked_people:
                track_id = person.get("track_id")
                if track_id is None:
                    continue
                iou = utils.box_iou(box, person["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = int(track_id)
            if best_track_id is not None and best_iou >= iou_thresh and best_track_id not in track_ids:
                track_ids.append(best_track_id)
                continue

            nearest_track_id = SurveillanceDetectionPipeline._nearest_track_id(box, tracked_people)
            if nearest_track_id is not None and nearest_track_id not in track_ids:
                track_ids.append(nearest_track_id)
        return track_ids

    @staticmethod
    def _nearest_track_id(
        box: List[int],
        tracked_people: List[Dict[str, Any]],
    ) -> Optional[int]:
        x, y, width, height = box
        cx = x + width / 2
        cy = y + height / 2
        best_track_id = None
        best_distance = float("inf")

        for person in tracked_people:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            px, py, pw, ph = [int(value) for value in person["box"]]
            pcx = px + pw / 2
            pcy = py + ph / 2
            distance = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            max_distance = max(180.0, ph * 1.5, pw * 2.0)
            if distance <= max_distance and distance < best_distance:
                best_distance = distance
                best_track_id = int(track_id)

        return best_track_id

    @staticmethod
    def _subjects_for_track_ids(
        track_ids: List[int],
        tracked_people: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        wanted = set(track_ids)
        return [
            person for person in tracked_people
            if person.get("track_id") is not None and int(person["track_id"]) in wanted
        ]

    @staticmethod
    def _append_tracking_context(
        alert: Dict[str, Any],
        track_ids: List[int],
        tracked_people: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        alert["track_ids"] = track_ids
        alert["subjects"] = SurveillanceDetectionPipeline._subjects_for_track_ids(track_ids, tracked_people)
        if track_ids:
            alert["primary_subject"] = f"Person {track_ids[0]}"
        return alert

    @staticmethod
    def _base_alert(
        alert_type: str,
        severity: str,
        confidence: float,
        temporal_score: float,
        bounding_boxes: List[List[int]],
        timestamp: str,
        live_count: Optional[int] = None,
        validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        alert = {
            "camera_id": "CAM_MAIN_ENTRANCE_01",
            "timestamp": timestamp,
            "anomaly_detected": True,
            "type": alert_type,
            "severity": severity,
            "confidence": confidence,
            "temporal_score": round(temporal_score, 3),
            "bounding_boxes": bounding_boxes,
            "live_count": live_count,
        }
        if validation is not None:
            alert["validation"] = validation
        return alert

    def _run_weapon(
        self,
        frame: np.ndarray,
        display_frame: np.ndarray,
        person_boxes: List[List[int]],
        tracked_people: List[Dict[str, Any]],
        conf_thresh: float,
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        hits = self.weapon_detector.detect(frame, person_boxes, conf_thresh)
        boxes = [item["box"] for item in hits]
        validation = self.temporal_scorer.validate_weapon(hits)
        if not validation["validated"]:
            return []
        temporal_score = validation["score"]
        max_conf = validation["confidence"]

        for box in boxes:
            self._draw_label(display_frame, box, f"WEAPON {max_conf:.2f}", (255, 0, 255))
        alert = self._base_alert(
            "weapon_detection",
            "CRITICAL",
            max_conf,
            temporal_score,
            boxes,
            timestamp,
            validation=validation["evidence"],
        )
        track_ids = self._track_ids_for_boxes(boxes, tracked_people, iou_thresh=0.02)
        alert = self._append_tracking_context(alert, track_ids, tracked_people)
        return self.score_alerts([alert], tracked_people)

    def _run_fire(
        self,
        frame: np.ndarray,
        display_frame: np.ndarray,
        conf_thresh: float,
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        hits = self.fire_detector.detect(frame, conf_thresh)
        boxes = [item["box"] for item in hits]
        flame_weight = 1.14 if any("smoke" not in item["label"] for item in hits) else 0.92
        validation = self.temporal_scorer.validate_fire(hits, weight=flame_weight)
        if not validation["validated"]:
            return []
        temporal_score = validation["score"]
        max_conf = validation["confidence"]

        for item in hits:
            label = "SMOKE" if "smoke" in item["label"] else "FIRE"
            color = (0, 69, 255) if label == "FIRE" else (192, 192, 192)
            self._draw_label(display_frame, item["box"], f"{label} {max_conf:.2f}", color)
        alert = self._base_alert(
            "fire_detection",
            "CRITICAL",
            max_conf,
            temporal_score,
            boxes,
            timestamp,
            validation=validation["evidence"],
        )
        return self.score_alerts([alert], [])

    def _run_fall(
        self,
        frame: np.ndarray,
        display_frame: np.ndarray,
        person_boxes: List[List[int]],
        tracked_people: List[Dict[str, Any]],
        conf_thresh: float,
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        hits = self.fall_detector.detect(frame, person_boxes, conf_thresh)
        boxes = [item["box"] for item in hits]
        track_ids = self._track_ids_for_boxes(boxes, tracked_people, iou_thresh=0.20)
        validation = self.temporal_scorer.validate_fall(hits, track_ids, tracked_people)
        if not validation["validated"]:
            return []
        temporal_score = validation["score"]
        max_conf = validation["confidence"]
        validated_ids = validation["evidence"].get("validated_track_ids") or track_ids

        for box in boxes:
            self._draw_label(display_frame, box, f"FALL DETECTED {max_conf:.2f}", (0, 0, 255))
        alert = self._base_alert(
            "fall_detection",
            "MEDIUM",
            max_conf,
            temporal_score,
            boxes,
            timestamp,
            validation=validation["evidence"],
        )
        if validated_ids:
            track_ids = [int(track_id) for track_id in validated_ids]
        alert = self._append_tracking_context(alert, track_ids, tracked_people)
        return self.score_alerts([alert], tracked_people)

    def _run_crowd(
        self,
        display_frame: np.ndarray,
        tracked_people: List[Dict[str, Any]],
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        person_boxes = self._person_boxes(tracked_people)
        for pbox in person_boxes:
            px, py, pw, ph = pbox
            cv2.rectangle(display_frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

        crowd_active = len(person_boxes) >= self.crowd_count_threshold
        self.temporal_scorer.update_consecutive("crowd_detection", crowd_active)
        hits = [
            {
                "box": pbox,
                "confidence": min(0.98, len(person_boxes) / max(1, self.crowd_count_threshold)),
                "label": "person",
            }
            for pbox in person_boxes
        ] if crowd_active else []
        should_alert, temporal_score, max_conf = self._should_alert("crowd_detection", hits)
        if not should_alert:
            return []

        alert = self._base_alert(
            "crowd_detection",
            "WARNING",
            max(0.90, max_conf),
            temporal_score,
            person_boxes,
            timestamp,
            live_count=len(person_boxes),
        )
        track_ids = [
            int(person["track_id"]) for person in tracked_people
            if person.get("track_id") is not None
        ]
        alert = self._append_tracking_context(alert, track_ids, tracked_people)
        return self.score_alerts([alert], tracked_people)

    @staticmethod
    def _draw_track_labels(display_frame: np.ndarray, tracked_people: List[Dict[str, Any]]) -> None:
        for person in tracked_people:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            x, y, width, _ = [int(value) for value in person["box"]]
            label = f"Person {track_id} {person.get('direction', 'moving')}"
            cv2.putText(
                display_frame,
                label,
                (x, max(16, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _merge_zone_updates(
        tracked_people: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for person in tracked_people:
            copy = dict(person)
            track_id = copy.get("track_id")
            if track_id is not None and int(track_id) in zone_updates:
                update = zone_updates[int(track_id)]
                merged_behaviors = set(copy.get("suspicious_behaviors") or [])
                merged_behaviors.update(update.get("suspicious_behaviors") or [])
                copy.update(update)
                copy["suspicious_behaviors"] = sorted(merged_behaviors)
            enriched.append(copy)
        return enriched

    def enrich_zone_alerts(
        self,
        alerts: List[Dict[str, Any]],
        violator_boxes: List[List[int]],
        tracked_people: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        track_ids = self._track_ids_for_boxes(violator_boxes, tracked_people, iou_thresh=0.15)
        zone_updates = self.zone_motion_tracker.update(tracked_people, track_ids)
        enriched_people = self._merge_zone_updates(tracked_people, zone_updates)

        enriched_alerts: List[Dict[str, Any]] = []
        for alert in alerts:
            if alert.get("type") != "trespassing_detection":
                enriched_alerts.append(alert)
                continue

            enriched_alert = self._append_tracking_context(alert, track_ids, enriched_people)
            related_updates = [zone_updates[track_id] for track_id in track_ids if track_id in zone_updates]
            if related_updates:
                enriched_alert["zone_events"] = sorted({item["zone_event"] for item in related_updates})
                behaviors = set()
                for item in related_updates:
                    behaviors.update(item.get("suspicious_behaviors") or [])
                enriched_alert["suspicious_behaviors"] = sorted(behaviors)
                enriched_alert["boundary_crossing_count"] = max(
                    item.get("boundary_crossing_count", 0) for item in related_updates
                )
                enriched_alert["restricted_zone_dwell_seconds"] = max(
                    item.get("restricted_zone_dwell_seconds", 0.0) for item in related_updates
                )
            enriched_alerts.append(enriched_alert)

        return enriched_alerts, enriched_people, zone_updates

    def run_yolo_checks(
        self,
        frame: np.ndarray,
        enabled_modules: List[str],
        confidence_thresholds: Dict[str, float],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run specialized detector engines and draw visual overlays.

        The public method name is retained for backward compatibility with the
        live worker and offline evaluation tooling.
        """
        enabled = set(system_config.normalize_modules(enabled_modules))
        display_frame = frame.copy()
        alerts: List[Dict[str, Any]] = []
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        tracked_people = self._detect_tracked_people(frame, enabled, confidence_thresholds)
        person_boxes = self._person_boxes(tracked_people)
        active_track_ids = [
            int(person["track_id"]) for person in tracked_people
            if person.get("track_id") is not None
        ]
        stats = {
            "crowd_count": len(person_boxes),
            "person_boxes": person_boxes,
            "tracked_people": tracked_people,
            "tracked_person_count": len(active_track_ids),
            "active_track_ids": active_track_ids,
            "tracking_enabled": system_config.tracking_enabled,
            "tracker": system_config.tracker_config if system_config.tracking_enabled else None,
            "temporal_validation": {
                "consecutive_detections": dict(self.temporal_scorer.consecutive_detections),
                "temporal_scores": {
                    module: round(float(score), 3)
                    for module, score in self.temporal_scorer.temporal_scores.items()
                },
                "rules": {
                    "weapon_required_frames": system_config.weapon_required_frames,
                    "fire_required_seconds": system_config.fire_required_seconds,
                    "fall_no_movement_seconds": system_config.fall_no_movement_seconds,
                    "fall_no_movement_speed_px_s": system_config.fall_no_movement_speed_px_s,
                },
            },
            "scene_threat": self.threat_scoring_engine.score_scene(tracked_people),
            "detector_architecture": "ai_core.specialized_detectors.v1",
        }

        if "weapon_detection" in enabled:
            try:
                alerts.extend(
                    self._run_weapon(
                        frame,
                        display_frame,
                        person_boxes,
                        tracked_people,
                        confidence_thresholds.get("weapon_detection", 0.45),
                        timestamp,
                    )
                )
            except Exception as exc:
                print(f"Error in weapon detection: {exc}")

        if "fire_detection" in enabled:
            try:
                alerts.extend(
                    self._run_fire(
                        frame,
                        display_frame,
                        confidence_thresholds.get("fire_detection", 0.45),
                        timestamp,
                    )
                )
            except Exception as exc:
                print(f"Error in fire detection: {exc}")

        if "fall_detection" in enabled:
            try:
                alerts.extend(
                    self._run_fall(
                        frame,
                        display_frame,
                        person_boxes,
                        tracked_people,
                        confidence_thresholds.get("fall_detection", 0.45),
                        timestamp,
                    )
                )
            except Exception as exc:
                print(f"Error in fall detection: {exc}")

        if "crowd_detection" in enabled:
            try:
                alerts.extend(self._run_crowd(display_frame, tracked_people, timestamp))
            except Exception as exc:
                print(f"Error in crowd drawing/trigger: {exc}")

        self._draw_track_labels(display_frame, tracked_people)

        return display_frame, alerts, stats
