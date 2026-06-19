from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config import system_config


@dataclass
class _ZoneTrackState:
    in_zone: bool = False
    entered_at: Optional[float] = None
    last_seen: float = 0.0
    crossing_count: int = 0


class RestrictedZoneMotionTracker:
    """Tracks entering/leaving and suspicious restricted-zone behavior by ID."""

    def __init__(self):
        self.states: Dict[int, _ZoneTrackState] = {}

    def reset(self) -> None:
        self.states.clear()

    def update(
        self,
        tracked_people: Sequence[Dict[str, Any]],
        in_zone_track_ids: Iterable[int],
    ) -> Dict[int, Dict[str, Any]]:
        now = time.time()
        in_zone_ids = {int(track_id) for track_id in in_zone_track_ids if track_id is not None}
        active_people = {
            int(person["track_id"]): person
            for person in tracked_people
            if person.get("track_id") is not None
        }
        updates: Dict[int, Dict[str, Any]] = {}

        for track_id, person in active_people.items():
            current_in_zone = track_id in in_zone_ids
            state = self.states.get(track_id)
            if state is None:
                state = _ZoneTrackState(last_seen=now)
                self.states[track_id] = state

            if current_in_zone and not state.in_zone:
                state.crossing_count += 1
                state.entered_at = now
                zone_event = "entering_restricted_zone"
            elif not current_in_zone and state.in_zone:
                state.entered_at = None
                zone_event = "leaving_restricted_zone"
            else:
                zone_event = "inside_restricted_zone" if current_in_zone else "outside_restricted_zone"

            state.in_zone = current_in_zone
            state.last_seen = now
            dwell_seconds = now - state.entered_at if current_in_zone and state.entered_at else 0.0
            suspicious_behaviors = list(person.get("suspicious_behaviors") or [])
            if current_in_zone and dwell_seconds >= system_config.loitering_seconds:
                suspicious_behaviors.append("loitering")
            if state.crossing_count >= system_config.boundary_crossing_threshold:
                suspicious_behaviors.append("repeated_boundary_crossing")
            if float(person.get("speed_px_s") or 0.0) >= system_config.running_speed_px_s:
                suspicious_behaviors.append("running_behavior")

            updates[track_id] = {
                "track_id": track_id,
                "in_restricted_zone": current_in_zone,
                "zone_event": zone_event,
                "restricted_zone_dwell_seconds": round(dwell_seconds, 2),
                "boundary_crossing_count": state.crossing_count,
                "suspicious_behaviors": sorted(set(suspicious_behaviors)),
            }

        self._cleanup(now, set(active_people))
        return updates

    def _cleanup(self, now: float, active_track_ids: set[int]) -> None:
        expired = [
            track_id for track_id, state in self.states.items()
            if track_id not in active_track_ids and now - state.last_seen > system_config.tracking_lost_seconds
        ]
        for track_id in expired:
            self.states.pop(track_id, None)
