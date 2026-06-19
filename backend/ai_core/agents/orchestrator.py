from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from config import system_config

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # LangGraph is the preferred runtime; this keeps local demos running.
    END = START = StateGraph = None


class AgentState(TypedDict, total=False):
    alerts: List[Dict[str, Any]]
    tracked_people: List[Dict[str, Any]]
    stats: Dict[str, Any]
    zone_updates: Dict[int, Dict[str, Any]]
    scene_memory: Dict[str, Any]
    vision: Dict[str, Any]
    context: Dict[str, Any]
    behavior: Dict[str, Any]
    verification: Dict[str, Any]
    threat_reasoning: Dict[str, Any]
    narration: Dict[str, Any]


class MultiAgentOrchestrator:
    """Controlled six-agent workflow for deterministic threat verification."""

    node_order = (
        "vision_agent",
        "context_agent",
        "behavior_agent",
        "verification_agent",
        "threat_reasoning_agent",
        "narrator_agent",
    )

    def __init__(self):
        self.runtime = "sequential_fallback"
        self.graph = self._build_graph()

    def analyze(
        self,
        alerts: List[Dict[str, Any]],
        tracked_people: List[Dict[str, Any]],
        stats: Dict[str, Any],
        zone_updates: Dict[int, Dict[str, Any]],
        scene_memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not system_config.multi_agent_enabled:
            return {
                "enabled": False,
                "runtime": self.runtime,
                "summary": "Multi-agent analysis disabled.",
                "agents": {},
                "alert_narratives": {},
            }

        state: AgentState = {
            "alerts": [dict(alert) for alert in alerts],
            "tracked_people": [dict(person) for person in tracked_people],
            "stats": dict(stats),
            "zone_updates": {int(track_id): dict(update) for track_id, update in zone_updates.items()},
            "scene_memory": dict(scene_memory or {}),
        }

        try:
            final_state = self.graph.invoke(state) if self.graph is not None else self._invoke_sequential(state)
        except Exception as exc:
            print(f"Multi-agent graph failed; using sequential fallback: {exc}")
            final_state = self._invoke_sequential(state)

        return {
            "enabled": True,
            "runtime": self.runtime,
            "framework": system_config.multi_agent_framework,
            "workflow": list(self.node_order),
            "agents": {
                "vision": final_state.get("vision", {}),
                "context": final_state.get("context", {}),
                "behavior": final_state.get("behavior", {}),
                "verification": final_state.get("verification", {}),
                "threat_reasoning": final_state.get("threat_reasoning", {}),
                "narrator": final_state.get("narration", {}),
            },
            "summary": final_state.get("threat_reasoning", {}).get("assessment", "No active threat reasoning."),
            "recommendation": final_state.get("threat_reasoning", {}).get("recommended_response", "Continue monitoring."),
            "alert_narratives": final_state.get("narration", {}).get("alert_narratives", {}),
            "generated_at": self._iso_timestamp(time.time()),
        }

    def enrich_alerts(
        self,
        alerts: List[Dict[str, Any]],
        report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not report.get("enabled"):
            return alerts

        narratives = report.get("alert_narratives") or {}
        verification = report.get("agents", {}).get("verification", {})
        decisions = verification.get("alert_decisions") or {}
        threat_reasoning = report.get("agents", {}).get("threat_reasoning", {})

        enriched: List[Dict[str, Any]] = []
        for index, alert in enumerate(alerts):
            copy = dict(alert)
            alert_key = self._alert_key(alert, index)
            decision = decisions.get(alert_key, {})
            copy["agent_alert_key"] = alert_key
            copy["agent_verification"] = decision
            copy["agent_narrative"] = narratives.get(alert_key)
            copy["agent_threat_reasoning"] = {
                "assessment": threat_reasoning.get("assessment"),
                "probability": threat_reasoning.get("probability"),
                "recommended_response": threat_reasoning.get("recommended_response"),
                "final_score": threat_reasoning.get("final_score"),
                "level": threat_reasoning.get("level"),
            }
            enriched.append(copy)
        return enriched

    def _build_graph(self):
        if system_config.multi_agent_framework != "langgraph" or StateGraph is None:
            self.runtime = "sequential_fallback"
            return None

        graph = StateGraph(AgentState)
        graph.add_node("vision_agent", self._vision_agent)
        graph.add_node("context_agent", self._context_agent)
        graph.add_node("behavior_agent", self._behavior_agent)
        graph.add_node("verification_agent", self._verification_agent)
        graph.add_node("threat_reasoning_agent", self._threat_reasoning_agent)
        graph.add_node("narrator_agent", self._narrator_agent)
        graph.add_edge(START, "vision_agent")
        graph.add_edge("vision_agent", "context_agent")
        graph.add_edge("context_agent", "behavior_agent")
        graph.add_edge("behavior_agent", "verification_agent")
        graph.add_edge("verification_agent", "threat_reasoning_agent")
        graph.add_edge("threat_reasoning_agent", "narrator_agent")
        graph.add_edge("narrator_agent", END)
        self.runtime = "langgraph"
        return graph.compile()

    def _invoke_sequential(self, state: AgentState) -> AgentState:
        self.runtime = "sequential_fallback"
        for node in (
            self._vision_agent,
            self._context_agent,
            self._behavior_agent,
            self._verification_agent,
            self._threat_reasoning_agent,
            self._narrator_agent,
        ):
            state.update(node(state))
        return state

    def _vision_agent(self, state: AgentState) -> Dict[str, Any]:
        alerts = state.get("alerts", [])
        tracked_people = state.get("tracked_people", [])
        detections = []
        flags = {
            "weapon": False,
            "fire_or_smoke": False,
            "fall": False,
            "crowd": False,
            "restricted_entry": False,
        }

        for index, alert in enumerate(alerts):
            alert_type = str(alert.get("type") or "unknown")
            confidence = float(alert.get("confidence") or 0.0)
            detections.append({
                "alert_key": self._alert_key(alert, index),
                "type": alert_type,
                "confidence": round(confidence, 3),
                "track_ids": alert.get("track_ids") or [],
                "bounding_boxes": alert.get("bounding_boxes") or [],
                "validation_rule": (alert.get("validation") or {}).get("validation_rule"),
            })
            if alert_type == "weapon_detection":
                flags["weapon"] = True
            elif alert_type == "fire_detection":
                flags["fire_or_smoke"] = True
            elif alert_type == "fall_detection":
                flags["fall"] = True
            elif alert_type == "crowd_detection":
                flags["crowd"] = True
            elif alert_type == "trespassing_detection":
                flags["restricted_entry"] = True

        return {
            "vision": {
                "agent": "vision_agent",
                "purpose": ["object_detection", "region_localization", "pose_or_fall_evidence"],
                **flags,
                "max_confidence": round(max([item["confidence"] for item in detections], default=0.0), 3),
                "detections": detections,
                "tracked_subjects": [
                    {
                        "track_id": person.get("track_id"),
                        "box": person.get("box"),
                        "direction": person.get("direction"),
                        "speed_px_s": person.get("speed_px_s"),
                    }
                    for person in tracked_people
                    if person.get("track_id") is not None
                ],
                "segmentation": {
                    "mode": "bounding_box_regions",
                    "available": any(item.get("bounding_boxes") for item in detections),
                },
                "pose_estimation": {
                    "fall_evidence_available": flags["fall"],
                    "source": "fall_detector_temporal_validation" if flags["fall"] else "not_triggered",
                },
            }
        }

    def _context_agent(self, state: AgentState) -> Dict[str, Any]:
        alerts = state.get("alerts", [])
        stats = state.get("stats", {})
        zone_updates = state.get("zone_updates", {})
        camera_id = self._camera_id(alerts)
        timestamp = self._alert_timestamp(alerts)
        hour = self._hour_from_timestamp(timestamp)
        time_band = self._time_band(hour)
        crowd_count = int(stats.get("crowd_count") or stats.get("tracked_person_count") or len(state.get("tracked_people", [])))
        zone_active = any(update.get("in_restricted_zone") for update in zone_updates.values())
        zone_type = "restricted_zone" if zone_active or any(alert.get("type") == "trespassing_detection" for alert in alerts) else self._camera_zone_type(camera_id)

        modifiers = []
        if time_band == "night":
            modifiers.append({"factor": "night_time", "risk_delta": 5})
        if zone_type == "restricted_zone":
            modifiers.append({"factor": "restricted_zone", "risk_delta": 10})
        if crowd_count >= 12:
            modifiers.append({"factor": "high_crowd_density", "risk_delta": 6})

        return {
            "context": {
                "agent": "context_agent",
                "camera_id": camera_id,
                "zone_type": zone_type,
                "time_band": time_band,
                "environment": {
                    "source_type": system_config.source_type,
                    "camera_role": self._camera_zone_type(camera_id),
                },
                "crowd_density": self._crowd_density(crowd_count),
                "crowd_count": crowd_count,
                "risk_modifiers": modifiers,
                "summary": f"{zone_type.replace('_', ' ')} scene during {time_band} with {crowd_count} tracked subject(s).",
            }
        }

    def _behavior_agent(self, state: AgentState) -> Dict[str, Any]:
        tracked_people = state.get("tracked_people", [])
        zone_updates = state.get("zone_updates", {})
        alerts = state.get("alerts", [])
        running_subjects = []
        loitering_subjects = []

        for person in tracked_people:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            speed = float(person.get("speed_px_s") or 0.0)
            behaviors = set(person.get("suspicious_behaviors") or [])
            update = zone_updates.get(int(track_id), {})
            behaviors.update(update.get("suspicious_behaviors") or [])

            if speed >= system_config.running_speed_px_s or "running_behavior" in behaviors or "running" in behaviors:
                running_subjects.append({
                    "track_id": int(track_id),
                    "speed_px_s": round(speed, 2),
                    "direction": person.get("direction"),
                })
            if "loitering" in behaviors or update.get("zone_event") == "loitering":
                loitering_subjects.append({
                    "track_id": int(track_id),
                    "dwell_seconds": update.get("restricted_zone_dwell_seconds", person.get("dwell_seconds")),
                })

        crowd_alert = any(alert.get("type") == "crowd_detection" for alert in alerts)
        panic_indicators = bool(crowd_alert and running_subjects)
        directions = [item.get("direction") for item in running_subjects if item.get("direction")]
        possible_chasing = len(running_subjects) >= 2 and len(set(directions)) <= 1

        return {
            "behavior": {
                "agent": "behavior_agent",
                "running": bool(running_subjects),
                "panic": panic_indicators,
                "loitering": bool(loitering_subjects),
                "chasing": possible_chasing,
                "running_subjects": running_subjects,
                "loitering_subjects": loitering_subjects,
                "summary": self._behavior_summary(running_subjects, loitering_subjects, panic_indicators, possible_chasing),
            }
        }

    def _verification_agent(self, state: AgentState) -> Dict[str, Any]:
        alerts = state.get("alerts", [])
        tracked_people = state.get("tracked_people", [])
        scene_memory = state.get("scene_memory", {})
        tracked_ids = {
            int(person["track_id"])
            for person in tracked_people
            if person.get("track_id") is not None
        }
        decisions = {}

        for index, alert in enumerate(alerts):
            alert_key = self._alert_key(alert, index)
            confidence = float(alert.get("confidence") or 0.0)
            alert_type = str(alert.get("type") or "")
            track_ids = [int(track_id) for track_id in alert.get("track_ids", []) if track_id is not None]
            validation = alert.get("validation") or {}
            temporal_score = float(alert.get("temporal_score") or 0.0)

            checks = {
                "confidence_present": confidence > 0.0,
                "temporal_consistency": bool(validation) or temporal_score >= 1.0,
                "multi_frame_validation": bool(validation.get("validation_rule")) or alert_type in {"crowd_detection", "trespassing_detection"},
                "tracking_reliability": self._tracking_reliable(alert_type, track_ids, tracked_ids),
                "scene_memory_recorded": bool(scene_memory.get("created_events") or scene_memory.get("timeline")),
            }
            score = self._verification_score(confidence, temporal_score, checks)
            decision = "verified" if score >= system_config.multi_agent_min_verification_score else "review"
            if score < 0.25:
                decision = "reject"

            decisions[alert_key] = {
                "decision": decision,
                "score": round(score, 3),
                "checks": checks,
                "false_alarm_risk": "low" if decision == "verified" else "elevated",
            }

        scores = [item["score"] for item in decisions.values()]
        return {
            "verification": {
                "agent": "verification_agent",
                "purpose": ["temporal_consistency", "multi_frame_validation", "tracking_reliability"],
                "overall_score": round(sum(scores) / len(scores), 3) if scores else 1.0,
                "alert_decisions": decisions,
                "suppression_enabled": system_config.multi_agent_suppress_unverified,
                "summary": self._verification_summary(decisions),
            }
        }

    def _threat_reasoning_agent(self, state: AgentState) -> Dict[str, Any]:
        alerts = state.get("alerts", [])
        vision = state.get("vision", {})
        context = state.get("context", {})
        behavior = state.get("behavior", {})
        verification = state.get("verification", {})
        base_score = max([int(alert.get("threat_score") or 0) for alert in alerts], default=0)

        modifier = 0
        modifier += sum(int(item.get("risk_delta") or 0) for item in context.get("risk_modifiers", []))
        if behavior.get("running"):
            modifier += 8
        if behavior.get("loitering"):
            modifier += 5
        if behavior.get("panic"):
            modifier += 7
        if behavior.get("chasing"):
            modifier += 6

        verification_score = float(verification.get("overall_score", 1.0))
        if verification_score < system_config.multi_agent_min_verification_score:
            modifier -= 20

        final_score = min(100, max(0, int(round(base_score + modifier))))
        level = self._level_for_score(final_score)
        probability = self._probability(final_score, verification_score)
        scenario = self._scenario(vision, context, behavior)

        return {
            "threat_reasoning": {
                "agent": "threat_reasoning_agent",
                "base_score": base_score,
                "context_behavior_modifier": modifier,
                "verification_score": round(verification_score, 3),
                "final_score": final_score,
                "level": level,
                "probability": probability,
                "scenario": scenario,
                "assessment": f"{probability.title()} probability {scenario}.",
                "recommended_response": self._recommended_response(level, scenario, verification_score),
            }
        }

    def _narrator_agent(self, state: AgentState) -> Dict[str, Any]:
        alerts = state.get("alerts", [])
        context = state.get("context", {})
        behavior = state.get("behavior", {})
        verification = state.get("verification", {})
        reasoning = state.get("threat_reasoning", {})
        decisions = verification.get("alert_decisions", {})
        narratives = {}

        for index, alert in enumerate(alerts):
            alert_key = self._alert_key(alert, index)
            narratives[alert_key] = self._alert_narrative(
                alert,
                context,
                behavior,
                decisions.get(alert_key, {}),
                reasoning,
            )

        return {
            "narration": {
                "agent": "narrator_agent",
                "alert_narratives": narratives,
                "incident_summary": self._incident_summary(narratives, reasoning),
            }
        }

    @staticmethod
    def _alert_key(alert: Dict[str, Any], index: int) -> str:
        track_ids = ",".join(str(track_id) for track_id in alert.get("track_ids", []))
        return f"{index}:{alert.get('type', 'alert')}:{track_ids}"

    @staticmethod
    def _camera_id(alerts: List[Dict[str, Any]]) -> str:
        for alert in alerts:
            if alert.get("camera_id"):
                return str(alert["camera_id"])
        return "CAM_MAIN_ENTRANCE_01"

    @staticmethod
    def _alert_timestamp(alerts: List[Dict[str, Any]]) -> Optional[str]:
        for alert in alerts:
            if alert.get("timestamp"):
                return str(alert["timestamp"])
        return None

    @staticmethod
    def _hour_from_timestamp(timestamp: Optional[str]) -> int:
        if not timestamp:
            return time.gmtime().tm_hour
        normalized = timestamp.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1]
        normalized = normalized[:19]
        for pattern in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return time.strptime(normalized, pattern).tm_hour
            except ValueError:
                continue
        return time.gmtime().tm_hour

    @staticmethod
    def _time_band(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 20:
            return "evening"
        return "night"

    @staticmethod
    def _camera_zone_type(camera_id: str) -> str:
        if "ENTRANCE" in camera_id or "LOBBY" in camera_id:
            return "public_lobby"
        if "GATE" in camera_id:
            return "gate_area"
        return "monitored_area"

    @staticmethod
    def _crowd_density(count: int) -> str:
        if count >= 12:
            return "high"
        if count >= 6:
            return "medium"
        return "low"

    @staticmethod
    def _behavior_summary(
        running_subjects: List[Dict[str, Any]],
        loitering_subjects: List[Dict[str, Any]],
        panic: bool,
        chasing: bool,
    ) -> str:
        parts = []
        if running_subjects:
            parts.append(f"{len(running_subjects)} running subject(s)")
        if loitering_subjects:
            parts.append(f"{len(loitering_subjects)} loitering subject(s)")
        if panic:
            parts.append("panic indicators")
        if chasing:
            parts.append("possible chasing pattern")
        return ", ".join(parts) if parts else "No elevated behavior pattern detected."

    @staticmethod
    def _tracking_reliable(alert_type: str, track_ids: List[int], tracked_ids: set[int]) -> bool:
        if alert_type in {"fire_detection", "crowd_detection"}:
            return True
        return bool(track_ids) and all(track_id in tracked_ids for track_id in track_ids)

    @staticmethod
    def _verification_score(confidence: float, temporal_score: float, checks: Dict[str, bool]) -> float:
        score = 0.0
        score += min(0.30, max(0.0, confidence) * 0.30)
        score += 0.25 if checks.get("temporal_consistency") else 0.0
        score += 0.20 if checks.get("multi_frame_validation") else 0.0
        score += 0.15 if checks.get("tracking_reliability") else 0.0
        score += 0.05 if checks.get("scene_memory_recorded") else 0.0
        score += min(0.05, max(0.0, temporal_score) * 0.03)
        return min(1.0, score)

    @staticmethod
    def _verification_summary(decisions: Dict[str, Dict[str, Any]]) -> str:
        if not decisions:
            return "No active alert required verification."
        counts: Dict[str, int] = {}
        for item in decisions.values():
            decision = str(item.get("decision") or "unknown")
            counts[decision] = counts.get(decision, 0) + 1
        return ", ".join(f"{count} {decision}" for decision, count in sorted(counts.items()))

    @staticmethod
    def _level_for_score(score: int) -> str:
        if score <= 30:
            return "SAFE"
        if score <= 60:
            return "WARNING"
        if score <= 85:
            return "HIGH RISK"
        return "CRITICAL"

    @staticmethod
    def _probability(score: int, verification_score: float) -> str:
        if verification_score < 0.35:
            return "low"
        if score >= 76 and verification_score >= 0.55:
            return "high"
        if score >= 45:
            return "medium"
        return "low"

    @staticmethod
    def _scenario(vision: Dict[str, Any], context: Dict[str, Any], behavior: Dict[str, Any]) -> str:
        restricted_context = context.get("zone_type") == "restricted_zone"
        if vision.get("weapon") and (vision.get("restricted_entry") or restricted_context):
            return "armed restricted-zone intrusion"
        if vision.get("weapon"):
            return "armed threat"
        if vision.get("fire_or_smoke"):
            return "fire or smoke hazard"
        if vision.get("fall"):
            return "medical safety incident"
        if vision.get("crowd") or behavior.get("panic"):
            return "crowd instability event"
        if vision.get("restricted_entry"):
            return "restricted-zone intrusion"
        return "anomalous surveillance event"

    @staticmethod
    def _recommended_response(level: str, scenario: str, verification_score: float) -> str:
        if verification_score < 0.35:
            return "Route to operator review before dispatch."
        if level == "CRITICAL":
            return f"Immediate response recommended for {scenario}."
        if level == "HIGH RISK":
            return f"Dispatch security to assess {scenario}."
        if level == "WARNING":
            return f"Monitor and prepare response for {scenario}."
        return "Continue monitoring."

    @staticmethod
    def _alert_narrative(
        alert: Dict[str, Any],
        context: Dict[str, Any],
        behavior: Dict[str, Any],
        decision: Dict[str, Any],
        reasoning: Dict[str, Any],
    ) -> str:
        subject = alert.get("primary_subject") or "A subject"
        alert_type = str(alert.get("type") or "alert").replace("_", " ")
        score = int(alert.get("threat_score") or reasoning.get("final_score") or 0)
        level = str(alert.get("alert_level") or alert.get("severity") or reasoning.get("level") or "UNKNOWN")
        confidence = float(alert.get("confidence") or 0.0)
        behaviors = behavior.get("summary") or "No elevated behavior pattern detected."
        zone = str(context.get("zone_type") or "monitored_area").replace("_", " ")
        verification = decision.get("decision", "review")
        recommendation = reasoning.get("recommended_response") or "Continue monitoring."
        return (
            f"{subject} triggered {alert_type} in the {zone}. "
            f"Model confidence is {confidence:.2f}; threat score is {score}/100 ({level}). "
            f"Behavior analysis: {behaviors}. Verification decision: {verification}. "
            f"{recommendation}"
        )

    @staticmethod
    def _incident_summary(narratives: Dict[str, str], reasoning: Dict[str, Any]) -> str:
        if not narratives:
            return "No active incident narrative generated."
        return f"{reasoning.get('assessment', 'Threat evidence reviewed.')} {reasoning.get('recommended_response', '')}".strip()

    @staticmethod
    def _iso_timestamp(timestamp: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
