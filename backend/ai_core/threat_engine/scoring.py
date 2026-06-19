from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from config import system_config


@dataclass
class ThreatScore:
    score: int
    level: str
    factors: Dict[str, float]
    summary: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ThreatScoringEngine:
    """Converts validated detections and scene context into normalized risk."""

    def score_alert(
        self,
        alert: Dict[str, Any],
        tracked_people: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if "threat_score" in alert and "alert_level" in alert:
            return dict(alert)

        threat_score = self.calculate(alert, tracked_people, zone_updates or {})
        enriched = dict(alert)
        enriched["threat_score"] = threat_score.score
        enriched["alert_level"] = threat_score.level
        enriched["risk_factors"] = threat_score.factors
        enriched["risk_summary"] = threat_score.summary
        enriched["severity"] = threat_score.level
        return enriched

    def score_alerts(
        self,
        alerts: Iterable[Dict[str, Any]],
        tracked_people: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        return [
            self.score_alert(alert, tracked_people, zone_updates)
            for alert in alerts
        ]

    def score_scene(
        self,
        tracked_people: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        zone_updates = zone_updates or {}
        if not tracked_people:
            score = ThreatScore(
                score=0,
                level=self.level_for_score(0),
                factors={
                    "base_scene": 0.0,
                    "movement_aggression": 0.0,
                    "zone_risk": 0.0,
                },
                summary="No active subject risk in the current scene.",
            )
            return score.as_dict()

        movement = self._movement_aggression(tracked_people, {"type": "scene"})
        zone_risk = 0.0
        if any(update.get("in_restricted_zone") for update in zone_updates.values()):
            zone_risk = min(system_config.threat_score_weights.get("zone_risk", 15.0), system_config.zone_base_risk)
        total = min(100, max(0, int(round(10.0 + movement + zone_risk))))
        score = ThreatScore(
            score=total,
            level=self.level_for_score(total),
            factors={
                "base_scene": 10.0,
                "movement_aggression": round(float(movement), 2),
                "zone_risk": round(float(zone_risk), 2),
            },
            summary=f"Scene scored {total}/100 ({self.level_for_score(total)}) from tracked subject motion and zone state.",
        )
        return score.as_dict()

    def calculate(
        self,
        alert: Dict[str, Any],
        tracked_people: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]],
    ) -> ThreatScore:
        alert_type = str(alert.get("type") or "")
        track_ids = [
            int(track_id) for track_id in alert.get("track_ids", [])
            if track_id is not None
        ]
        subjects = self._subjects(track_ids, tracked_people)
        factors = {
            "base_event": self._base_event_score(alert_type),
            "weapon_confidence": self._weapon_confidence(alert),
            "intrusion_severity": self._intrusion_severity(alert, zone_updates),
            "movement_aggression": self._movement_aggression(subjects, alert),
            "time_sensitivity": self._time_sensitivity(alert),
            "zone_risk": self._zone_risk(alert, zone_updates),
        }
        score = min(100, max(0, int(round(sum(factors.values())))))
        level = self.level_for_score(score)
        return ThreatScore(
            score=score,
            level=level,
            factors={key: round(float(value), 2) for key, value in factors.items()},
            summary=self._summary(alert_type, score, level, factors),
        )

    @staticmethod
    def level_for_score(score: int) -> str:
        if score <= 30:
            return "SAFE"
        if score <= 60:
            return "WARNING"
        if score <= 85:
            return "HIGH RISK"
        return "CRITICAL"

    @staticmethod
    def _subjects(track_ids: List[int], tracked_people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        wanted = set(track_ids)
        if not wanted:
            return list(tracked_people)
        return [
            person for person in tracked_people
            if person.get("track_id") is not None and int(person["track_id"]) in wanted
        ]

    @staticmethod
    def _base_event_score(alert_type: str) -> float:
        return {
            "weapon_detection": 45.0,
            "fire_detection": 55.0,
            "fall_detection": 45.0,
            "trespassing_detection": 5.0,
            "crowd_detection": 20.0,
        }.get(alert_type, 10.0)

    @staticmethod
    def _weapon_confidence(alert: Dict[str, Any]) -> float:
        if alert.get("type") != "weapon_detection":
            return 0.0
        confidence = min(1.0, max(0.0, float(alert.get("confidence") or 0.0)))
        return confidence * system_config.threat_score_weights.get("weapon_confidence", 45.0)

    @staticmethod
    def _intrusion_severity(alert: Dict[str, Any], zone_updates: Dict[int, Dict[str, Any]]) -> float:
        if alert.get("type") != "trespassing_detection":
            return 0.0
        live_count = max(1, int(alert.get("live_count") or 1))
        max_score = system_config.threat_score_weights.get("intrusion_severity", 35.0)
        score = min(max_score, 15.0 + (live_count - 1) * 5.0)
        behaviors = set(alert.get("suspicious_behaviors") or [])
        for update in zone_updates.values():
            behaviors.update(update.get("suspicious_behaviors") or [])
        if "repeated_boundary_crossing" in behaviors:
            score += 8.0
        if "loitering" in behaviors:
            score += 6.0
        return min(max_score, score)

    @staticmethod
    def _movement_aggression(subjects: List[Dict[str, Any]], alert: Dict[str, Any]) -> float:
        speeds = [float(person.get("speed_px_s") or 0.0) for person in subjects]
        max_speed = max(speeds, default=0.0)
        max_score = system_config.threat_score_weights.get("movement_aggression", 20.0)
        score = min(max_score, (max_speed / max(system_config.running_speed_px_s, 1.0)) * max_score)
        behaviors = set(alert.get("suspicious_behaviors") or [])
        for person in subjects:
            behaviors.update(person.get("suspicious_behaviors") or [])
        if "running_behavior" in behaviors:
            score = max(score, max_score * 0.90)
        return score

    @staticmethod
    def _time_sensitivity(alert: Dict[str, Any]) -> float:
        alert_type = str(alert.get("type") or "")
        validation = alert.get("validation") or {}
        max_score = system_config.threat_score_weights.get("time_sensitivity", 12.0)
        if alert_type == "fire_detection":
            observed = float(validation.get("observed_seconds") or 0.0)
            return min(max_score, observed * 2.0)
        if alert_type == "fall_detection":
            subjects = validation.get("subjects") or []
            observed = max(
                [float(item.get("observed_no_movement_seconds") or 0.0) for item in subjects],
                default=0.0,
            )
            return min(max_score, observed * 2.0)
        if alert_type == "weapon_detection":
            frames = float(validation.get("observed_consecutive_frames") or 0.0)
            return min(max_score, frames)
        return 0.0

    @staticmethod
    def _zone_risk(alert: Dict[str, Any], zone_updates: Dict[int, Dict[str, Any]]) -> float:
        if alert.get("type") != "trespassing_detection":
            return 0.0
        max_score = system_config.threat_score_weights.get("zone_risk", 25.0)
        dwell_seconds = float(alert.get("restricted_zone_dwell_seconds") or 0.0)
        crossing_count = int(alert.get("boundary_crossing_count") or 0)
        score = system_config.zone_base_risk
        score += min(10.0, dwell_seconds / max(system_config.loitering_seconds, 1.0) * 10.0)
        score += min(10.0, max(0, crossing_count - 1) * 2.5)
        if any(update.get("in_restricted_zone") for update in zone_updates.values()):
            score += 5.0
        return min(max_score, score)

    @staticmethod
    def _summary(alert_type: str, score: int, level: str, factors: Dict[str, float]) -> str:
        top_factors = sorted(factors.items(), key=lambda item: item[1], reverse=True)[:3]
        factor_text = ", ".join(f"{name}={value:.1f}" for name, value in top_factors if value > 0)
        if not factor_text:
            factor_text = "baseline scene risk"
        return f"{alert_type.replace('_', ' ')} scored {score}/100 ({level}) from {factor_text}."
