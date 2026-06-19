from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

from config import system_config


class AdvancedBehaviorAnalyzer:
    """Heuristic Phase 4 behavior analysis over tracker and zone evidence."""

    def analyze(
        self,
        tracked_people: List[Dict[str, Any]],
        alerts: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]],
        stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not system_config.advanced_behavior_enabled:
            return {
                "enabled": False,
                "events": [],
                "summary": "Advanced behavior analysis disabled.",
            }

        events: List[Dict[str, Any]] = []
        events.extend(self._fight_candidates(tracked_people))
        events.extend(self._panic_candidates(tracked_people, alerts, stats))
        events.extend(self._stampede_candidates(tracked_people, stats))
        events.extend(self._abandoned_bag_candidates(stats))
        events.extend(self._suspicious_movement(tracked_people, zone_updates))

        events = sorted(events, key=lambda event: event["risk_score"], reverse=True)
        return {
            "enabled": True,
            "events": events,
            "max_risk_score": max([event["risk_score"] for event in events], default=0),
            "summary": self._summary(events),
            "signals": {
                "tracked_people": len(tracked_people),
                "crowd_count": int(stats.get("crowd_count") or len(tracked_people)),
                "running_subjects": len(self._running_subjects(tracked_people)),
            },
        }

    def _fight_candidates(self, tracked_people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for left, right in combinations(tracked_people, 2):
            left_id = left.get("track_id")
            right_id = right.get("track_id")
            if left_id is None or right_id is None:
                continue
            distance = self._center_distance(left.get("box"), right.get("box"))
            if distance is None or distance > system_config.fight_proximity_px:
                continue
            left_speed = float(left.get("speed_px_s") or 0.0)
            right_speed = float(right.get("speed_px_s") or 0.0)
            if max(left_speed, right_speed) < system_config.suspicious_movement_speed_px_s:
                continue
            risk = min(100, int(45 + (system_config.fight_proximity_px - distance) * 0.2 + max(left_speed, right_speed) * 0.05))
            events.append({
                "type": "fight_candidate",
                "risk_score": risk,
                "track_ids": [int(left_id), int(right_id)],
                "evidence": {
                    "distance_px": round(distance, 2),
                    "max_speed_px_s": round(max(left_speed, right_speed), 2),
                    "rule": "close_fast_subjects",
                },
            })
        return events

    def _panic_candidates(
        self,
        tracked_people: List[Dict[str, Any]],
        alerts: List[Dict[str, Any]],
        stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        running = self._running_subjects(tracked_people)
        crowd_count = int(stats.get("crowd_count") or len(tracked_people))
        hazard_alert = any(alert.get("type") in {"weapon_detection", "fire_detection"} for alert in alerts)
        if len(running) < system_config.panic_running_count and not (hazard_alert and running):
            return []

        risk = min(100, 35 + len(running) * 12 + (15 if hazard_alert else 0) + max(0, crowd_count - 5) * 2)
        return [{
            "type": "panic_candidate",
            "risk_score": risk,
            "track_ids": [int(person["track_id"]) for person in running if person.get("track_id") is not None],
            "evidence": {
                "running_count": len(running),
                "crowd_count": crowd_count,
                "hazard_alert_present": hazard_alert,
                "rule": "running_subjects_with_hazard_or_cluster",
            },
        }]

    def _stampede_candidates(self, tracked_people: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        crowd_count = int(stats.get("crowd_count") or len(tracked_people))
        running = self._running_subjects(tracked_people)
        if crowd_count < system_config.stampede_crowd_threshold or len(running) < system_config.panic_running_count:
            return []

        directions = [str(person.get("direction") or "") for person in running if person.get("direction")]
        dominant_direction_count = max([directions.count(direction) for direction in set(directions)], default=0)
        if dominant_direction_count < max(2, len(running) // 2):
            return []

        risk = min(100, 50 + crowd_count * 2 + len(running) * 5)
        return [{
            "type": "crowd_stampede_candidate",
            "risk_score": risk,
            "track_ids": [int(person["track_id"]) for person in running if person.get("track_id") is not None],
            "evidence": {
                "crowd_count": crowd_count,
                "running_count": len(running),
                "dominant_direction_count": dominant_direction_count,
                "rule": "dense_crowd_running_same_direction",
            },
        }]

    @staticmethod
    def _abandoned_bag_candidates(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidate_objects = stats.get("object_candidates") or []
        events = []
        for item in candidate_objects:
            if item.get("label") not in {"bag", "backpack", "suitcase", "handbag"}:
                continue
            dwell_seconds = float(item.get("stationary_seconds") or 0.0)
            if dwell_seconds < system_config.abandoned_object_seconds:
                continue
            events.append({
                "type": "abandoned_bag_candidate",
                "risk_score": min(100, int(35 + dwell_seconds)),
                "track_ids": [],
                "evidence": {
                    "label": item.get("label"),
                    "stationary_seconds": round(dwell_seconds, 2),
                    "box": item.get("box"),
                    "rule": "stationary_unattended_object",
                },
            })
        return events

    def _suspicious_movement(
        self,
        tracked_people: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        events = []
        for person in tracked_people:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            update = zone_updates.get(int(track_id), {})
            behaviors = set(person.get("suspicious_behaviors") or [])
            behaviors.update(update.get("suspicious_behaviors") or [])
            speed = float(person.get("speed_px_s") or 0.0)
            if not behaviors and speed < system_config.suspicious_movement_speed_px_s:
                continue

            risk = min(100, int(20 + speed * 0.08 + len(behaviors) * 12))
            events.append({
                "type": "suspicious_movement",
                "risk_score": risk,
                "track_ids": [int(track_id)],
                "evidence": {
                    "speed_px_s": round(speed, 2),
                    "behaviors": sorted(behaviors),
                    "zone_event": update.get("zone_event"),
                    "rule": "speed_or_zone_behavior",
                },
            })
        return events

    @staticmethod
    def _running_subjects(tracked_people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        running = []
        for person in tracked_people:
            behaviors = set(person.get("suspicious_behaviors") or [])
            speed = float(person.get("speed_px_s") or 0.0)
            if speed >= system_config.running_speed_px_s or "running_behavior" in behaviors or "running" in behaviors:
                running.append(person)
        return running

    @staticmethod
    def _center_distance(left_box: Any, right_box: Any) -> float | None:
        if not left_box or not right_box or len(left_box) != 4 or len(right_box) != 4:
            return None
        lx, ly, lw, lh = [float(value) for value in left_box]
        rx, ry, rw, rh = [float(value) for value in right_box]
        left_center = (lx + lw / 2.0, ly + lh / 2.0)
        right_center = (rx + rw / 2.0, ry + rh / 2.0)
        return ((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2) ** 0.5

    @staticmethod
    def _summary(events: List[Dict[str, Any]]) -> str:
        if not events:
            return "No advanced behavior anomalies detected."
        top = events[0]
        return f"{top['type'].replace('_', ' ')} detected with risk {top['risk_score']}/100."
