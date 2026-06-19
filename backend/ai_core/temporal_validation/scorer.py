from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from config import system_config
from ai_core.detectors.utils import detection_confidence


DEFAULT_TEMPORAL_MODULES = (
    "weapon_detection",
    "fire_detection",
    "fall_detection",
    "crowd_detection",
)


@dataclass
class FallCandidateState:
    first_seen: float
    last_seen: float
    no_movement_since: Optional[float] = None
    last_speed_px_s: float = 0.0
    alert_ready: bool = False


class TemporalThreatScorer:
    """Rules-based temporal validation for alert reliability."""

    def __init__(self, modules=DEFAULT_TEMPORAL_MODULES):
        self.consecutive_detections: Dict[str, int] = {module: 0 for module in modules}
        self.temporal_scores: Dict[str, float] = {module: 0.0 for module in modules}
        self.persistence_threshold = system_config.weapon_required_hits
        self.first_seen_at: Dict[str, Optional[float]] = {module: None for module in modules}
        self.last_seen_at: Dict[str, Optional[float]] = {module: None for module in modules}
        self.weapon_window_size = system_config.weapon_temporal_window
        self.weapon_frame_window: Deque[bool] = deque(maxlen=self.weapon_window_size)
        self.weapon_confidence_window: Deque[float] = deque(maxlen=self.weapon_window_size)
        self.fall_candidates: Dict[str, FallCandidateState] = {}

    def reset(self) -> None:
        self._sync_weapon_window_config(clear=True)
        for module in set(self.consecutive_detections) | set(DEFAULT_TEMPORAL_MODULES):
            self.consecutive_detections[module] = 0
            self.temporal_scores[module] = 0.0
            self.first_seen_at[module] = None
            self.last_seen_at[module] = None
        self.fall_candidates.clear()

    def update_score(self, module: str, hits: List[Dict[str, Any]], weight: float = 1.0) -> float:
        score = self._legacy_score(module, hits, weight)
        self.temporal_scores[module] = score
        return score

    def should_alert(
        self,
        module: str,
        hits: List[Dict[str, Any]],
        weight: float = 1.0,
    ) -> Tuple[bool, float, float]:
        if module == "weapon_detection":
            validation = self.validate_weapon(hits)
        elif module == "fire_detection":
            validation = self.validate_fire(hits, weight=weight)
        else:
            score = self.update_score(module, hits, weight)
            validation = {
                "validated": bool(hits) and score >= system_config.alert_trigger_scores.get(module, 1.0),
                "score": score,
                "confidence": detection_confidence(hits),
            }
        return validation["validated"], validation["score"], validation["confidence"]

    def update_consecutive(self, module: str, has_hits: bool) -> None:
        now = time.time()
        if has_hits:
            self.consecutive_detections[module] = self.consecutive_detections.get(module, 0) + 1
            if self.first_seen_at.get(module) is None:
                self.first_seen_at[module] = now
            self.last_seen_at[module] = now
        else:
            self.consecutive_detections[module] = 0
            self.first_seen_at[module] = None
            self.last_seen_at[module] = None
            self.temporal_scores[module] = 0.0

    def validate_weapon(self, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._sync_weapon_window_config()
        self.update_consecutive("weapon_detection", bool(hits))
        frames = self.consecutive_detections.get("weapon_detection", 0)
        confidence = detection_confidence(hits)
        min_confidence = system_config.weapon_min_alert_confidence
        current_positive = bool(hits) and confidence >= min_confidence
        self.weapon_frame_window.append(current_positive)
        self.weapon_confidence_window.append(confidence if current_positive else 0.0)

        window_size = self.weapon_frame_window.maxlen or system_config.weapon_temporal_window
        required_hits = min(max(int(system_config.weapon_required_hits), 1), window_size)
        observed_hits = sum(1 for has_hit in self.weapon_frame_window if has_hit)
        max_window_confidence = max(self.weapon_confidence_window, default=0.0)
        score = observed_hits / max(1, required_hits)
        self.temporal_scores["weapon_detection"] = score
        return {
            "validated": current_positive and observed_hits >= required_hits,
            "score": score,
            "confidence": confidence,
            "evidence": {
                "required_hits": required_hits,
                "window_size": window_size,
                "observed_hits": observed_hits,
                "min_alert_confidence": min_confidence,
                "current_frame_confidence": round(confidence, 3),
                "max_window_confidence": round(max_window_confidence, 3),
                "required_consecutive_frames": system_config.weapon_required_frames,
                "observed_consecutive_frames": frames,
                "validation_rule": "weapon_sliding_window_hits",
            },
        }

    def validate_fire(self, hits: List[Dict[str, Any]], weight: float = 1.0) -> Dict[str, Any]:
        self.update_consecutive("fire_detection", bool(hits))
        now = time.time()
        first_seen = self.first_seen_at.get("fire_detection")
        duration = (now - first_seen) if hits and first_seen is not None else 0.0
        required = system_config.fire_required_seconds
        confidence = detection_confidence(hits)
        score = min(3.0, (duration / max(required, 1e-3)) * weight)
        self.temporal_scores["fire_detection"] = score
        return {
            "validated": bool(hits) and duration >= required,
            "score": score,
            "confidence": confidence,
            "evidence": {
                "required_seconds": required,
                "observed_seconds": round(duration, 3),
                "observed_consecutive_frames": self.consecutive_detections.get("fire_detection", 0),
                "validation_rule": "fire_duration_seconds",
            },
        }

    def validate_fall(
        self,
        hits: List[Dict[str, Any]],
        track_ids: Iterable[int],
        tracked_people: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self.update_consecutive("fall_detection", bool(hits))
        now = time.time()
        track_id_list = [int(track_id) for track_id in track_ids if track_id is not None]
        if not hits:
            self._cleanup_fall_candidates(now, active_keys=set())
            self.temporal_scores["fall_detection"] = 0.0
            return self._fall_result(False, 0.0, 0.0, [], track_id_list)

        if not track_id_list:
            track_id_list = [None]

        active_keys = {self._fall_key(track_id) for track_id in track_id_list}
        subject_lookup = {
            int(person["track_id"]): person
            for person in tracked_people
            if person.get("track_id") is not None
        }
        validated_ids: List[int] = []
        evidence_items: List[Dict[str, Any]] = []
        best_score = 0.0

        for track_id in track_id_list:
            key = self._fall_key(track_id)
            person = subject_lookup.get(track_id) if track_id is not None else None
            speed = float(person.get("speed_px_s") or 0.0) if person else 0.0
            state = self.fall_candidates.get(key)
            if state is None:
                state = FallCandidateState(first_seen=now, last_seen=now)
                self.fall_candidates[key] = state

            state.last_seen = now
            state.last_speed_px_s = speed
            if speed <= system_config.fall_no_movement_speed_px_s:
                if state.no_movement_since is None:
                    state.no_movement_since = now
            else:
                state.no_movement_since = None
                state.alert_ready = False

            no_movement_seconds = (
                now - state.no_movement_since
                if state.no_movement_since is not None
                else 0.0
            )
            score = min(3.0, no_movement_seconds / max(system_config.fall_no_movement_seconds, 1e-3))
            best_score = max(best_score, score)
            evidence = {
                "track_id": track_id,
                "required_no_movement_seconds": system_config.fall_no_movement_seconds,
                "observed_no_movement_seconds": round(no_movement_seconds, 3),
                "movement_speed_px_s": round(speed, 3),
                "movement_threshold_px_s": system_config.fall_no_movement_speed_px_s,
                "validation_rule": "fall_no_movement_after_detection",
            }
            evidence_items.append(evidence)

            if no_movement_seconds >= system_config.fall_no_movement_seconds:
                state.alert_ready = True
                if track_id is not None:
                    validated_ids.append(track_id)

        self._cleanup_fall_candidates(now, active_keys)
        confidence = detection_confidence(hits)
        validated = any(self.fall_candidates[key].alert_ready for key in active_keys if key in self.fall_candidates)
        self.temporal_scores["fall_detection"] = best_score
        return self._fall_result(validated, best_score, confidence, evidence_items, validated_ids)

    def _legacy_score(self, module: str, hits: List[Dict[str, Any]], weight: float) -> float:
        hit_score = min(1.0, detection_confidence(hits)) * weight if hits else 0.0
        previous_score = self.temporal_scores.get(module, 0.0)
        decayed_score = previous_score * system_config.temporal_decay
        score = min(3.0, decayed_score + hit_score)
        if not hits:
            score = decayed_score
        return score

    def _sync_weapon_window_config(self, clear: bool = False) -> None:
        window_size = min(max(int(system_config.weapon_temporal_window), 3), 60)
        required_hits = min(max(int(system_config.weapon_required_hits), 1), window_size)
        self.persistence_threshold = required_hits

        if clear or self.weapon_frame_window.maxlen != window_size:
            existing_frames = [] if clear else list(self.weapon_frame_window)[-window_size:]
            existing_confidences = [] if clear else list(self.weapon_confidence_window)[-window_size:]
            self.weapon_frame_window = deque(existing_frames, maxlen=window_size)
            self.weapon_confidence_window = deque(existing_confidences, maxlen=window_size)
            self.weapon_window_size = window_size

    @staticmethod
    def _fall_key(track_id: Optional[int]) -> str:
        return f"track:{track_id}" if track_id is not None else "untracked"

    @staticmethod
    def _fall_result(
        validated: bool,
        score: float,
        confidence: float,
        evidence_items: List[Dict[str, Any]],
        validated_ids: List[int],
    ) -> Dict[str, Any]:
        return {
            "validated": validated,
            "score": score,
            "confidence": confidence,
            "evidence": {
                "validation_rule": "fall_no_movement_after_detection",
                "subjects": evidence_items,
                "validated_track_ids": validated_ids,
            },
        }

    def _cleanup_fall_candidates(self, now: float, active_keys: set[str]) -> None:
        expired = [
            key for key, state in self.fall_candidates.items()
            if key not in active_keys and now - state.last_seen > system_config.fall_candidate_lost_seconds
        ]
        for key in expired:
            self.fall_candidates.pop(key, None)
