from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Optional

from config import system_config

try:
    import redis
except ImportError:  # Redis is optional for local/demo runs.
    redis = None


SUPPORTED_BACKENDS = {"memory", "redis"}


@dataclass
class MemoryEvent:
    event_id: str
    timestamp: str
    monotonic_time: float
    camera_id: str
    event_type: str
    summary: str
    threat_score: int
    alert_level: str
    track_ids: List[int] = field(default_factory=list)
    risk_factors: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload = self.storage_dict()
        payload["age_seconds"] = round(max(0.0, time.time() - self.monotonic_time), 2)
        return payload

    def storage_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryEvent":
        return cls(
            event_id=str(payload.get("event_id") or ""),
            timestamp=str(payload.get("timestamp") or ""),
            monotonic_time=float(payload.get("monotonic_time") or time.time()),
            camera_id=str(payload.get("camera_id") or "CAM_MAIN_ENTRANCE_01"),
            event_type=str(payload.get("event_type") or "event"),
            summary=str(payload.get("summary") or ""),
            threat_score=int(payload.get("threat_score") or 0),
            alert_level=str(payload.get("alert_level") or "SAFE"),
            track_ids=[int(track_id) for track_id in payload.get("track_ids", []) if track_id is not None],
            risk_factors=dict(payload.get("risk_factors") or {}),
            metadata=dict(payload.get("metadata") or {}),
            embedding=[float(value) for value in payload.get("embedding", [])],
        )


class SceneMemoryStore:
    """Event timeline memory with embeddings, object history, and optional Redis persistence."""

    def __init__(self):
        self._lock = RLock()
        self.events: Deque[MemoryEvent] = deque(maxlen=system_config.scene_memory_max_events)
        self.object_history: Dict[int, Deque[Dict[str, Any]]] = defaultdict(self._new_history_deque)
        self.last_event_keys: Dict[str, float] = {}
        self.requested_backend = "memory"
        self.active_backend = "memory"
        self.redis_client = None
        self.redis_error: Optional[str] = None
        self.reconfigure(clear=False)

    def reconfigure(self, clear: bool = False) -> None:
        """Apply current scene-memory config and reconnect/resize storage if needed."""
        with self._lock:
            self.requested_backend = self._configured_backend()
            self.redis_error = None

            if self.requested_backend == "redis":
                client = self._connect_redis()
                if client is not None:
                    self.redis_client = client
                    self.active_backend = "redis"
                    if clear:
                        self._reset_redis()
                    self._trim_redis_limits()
                    return

            self.redis_client = None
            self.active_backend = "memory"
            self._resize_memory_deques()
            if clear:
                self._reset_memory()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requested_backend": self.requested_backend,
                "active_backend": self.active_backend,
                "redis_connected": self.active_backend == "redis" and self.redis_client is not None,
                "redis_url": self._safe_redis_url(system_config.scene_memory_redis_url),
                "key_prefix": system_config.scene_memory_key_prefix,
                "redis_error": self.redis_error,
                "memory_size": self._memory_size_unlocked(),
            }

    def reset(self) -> None:
        with self._lock:
            self._reset_memory()
            if self.active_backend == "redis" and self.redis_client is not None:
                self._reset_redis()

    def record_frame(
        self,
        alerts: List[Dict[str, Any]],
        tracked_people: List[Dict[str, Any]],
        scene_threat: Dict[str, Any],
        zone_updates: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            zone_updates = zone_updates or {}
            self._append_object_history(tracked_people, zone_updates, now)

            created: List[MemoryEvent] = []
            for alert in alerts:
                event = self._event_from_alert(alert, now)
                if self._should_store(event):
                    self._store_event(event)
                    created.append(event)

            scene_event = self._scene_event(scene_threat, tracked_people, zone_updates, now)
            if scene_event and self._should_store(scene_event):
                self._store_event(scene_event)
                created.append(scene_event)

            self._expire_old(now)
            return {
                "created_events": [event.as_dict() for event in created],
                "timeline": self.timeline(limit=system_config.scene_memory_timeline_limit),
                "object_history": self.object_history_snapshot(limit=system_config.scene_memory_object_history),
                "memory_size": self._memory_size_unlocked(),
                "backend": self.active_backend,
            }

    def timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            if self.active_backend == "redis" and self.redis_client is not None:
                return [event.as_dict() for event in self._redis_events(limit=max(1, limit))]
            return [event.as_dict() for event in list(self.events)[-max(1, limit):]]

    def object_history_snapshot(self, limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            if self.active_backend == "redis" and self.redis_client is not None:
                return self._redis_object_history(limit=max(1, limit))

            snapshot: Dict[str, List[Dict[str, Any]]] = {}
            for track_id, history in self.object_history.items():
                snapshot[str(track_id)] = list(history)[-max(1, limit):]
            return snapshot

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            query_embedding = self._embed(query)
            ranked = sorted(
                self._all_events_unlocked(),
                key=lambda event: self._cosine_similarity(query_embedding, event.embedding),
                reverse=True,
            )
            return [event.as_dict() for event in ranked[:max(1, limit)]]

    def _append_object_history(
        self,
        tracked_people: Iterable[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]],
        now: float,
    ) -> None:
        for person in tracked_people:
            entry = self._object_history_entry(person, zone_updates, now)
            if entry is None:
                continue

            track_id = int(entry.pop("track_id"))
            if self.active_backend == "redis" and self.redis_client is not None:
                self._append_redis_object_history(track_id, entry)
            else:
                self.object_history[track_id].append(entry)

    def _event_from_alert(self, alert: Dict[str, Any], now: float) -> MemoryEvent:
        track_ids = [
            int(track_id) for track_id in alert.get("track_ids", [])
            if track_id is not None
        ]
        event_type = str(alert.get("type") or "alert")
        score = int(alert.get("threat_score") or 0)
        level = str(alert.get("alert_level") or alert.get("severity") or "SAFE")
        summary = str(alert.get("risk_summary") or self._alert_summary(alert, track_ids))
        metadata = {
            "validation": alert.get("validation"),
            "zone_events": alert.get("zone_events"),
            "suspicious_behaviors": alert.get("suspicious_behaviors"),
            "primary_subject": alert.get("primary_subject"),
            "bounding_boxes": alert.get("bounding_boxes"),
        }
        return MemoryEvent(
            event_id=self._event_id(event_type, track_ids, now),
            timestamp=str(alert.get("timestamp") or self._iso_timestamp(now)),
            monotonic_time=now,
            camera_id=str(alert.get("camera_id") or "CAM_MAIN_ENTRANCE_01"),
            event_type=event_type,
            summary=summary,
            threat_score=score,
            alert_level=level,
            track_ids=track_ids,
            risk_factors=dict(alert.get("risk_factors") or {}),
            metadata={key: value for key, value in metadata.items() if value not in (None, [], {})},
            embedding=self._embed(summary),
        )

    def _scene_event(
        self,
        scene_threat: Dict[str, Any],
        tracked_people: List[Dict[str, Any]],
        zone_updates: Dict[int, Dict[str, Any]],
        now: float,
    ) -> Optional[MemoryEvent]:
        score = int(scene_threat.get("score") or 0)
        if score < system_config.scene_memory_min_scene_score:
            return None

        track_ids = [
            int(person["track_id"]) for person in tracked_people
            if person.get("track_id") is not None
        ]
        summary = str(scene_threat.get("summary") or f"Scene risk score {score}.")
        if zone_updates:
            active_events = sorted({
                str(update.get("zone_event"))
                for update in zone_updates.values()
                if update.get("zone_event")
            })
            if active_events:
                summary = f"{summary} Zone activity: {', '.join(active_events)}."

        return MemoryEvent(
            event_id=self._event_id("scene_threat", track_ids, now),
            timestamp=self._iso_timestamp(now),
            monotonic_time=now,
            camera_id="CAM_MAIN_ENTRANCE_01",
            event_type="scene_threat",
            summary=summary,
            threat_score=score,
            alert_level=str(scene_threat.get("level") or "SAFE"),
            track_ids=track_ids,
            risk_factors=dict(scene_threat.get("factors") or {}),
            metadata={"zone_updates": zone_updates},
            embedding=self._embed(summary),
        )

    def _should_store(self, event: MemoryEvent) -> bool:
        key = f"{event.event_type}:{','.join(map(str, event.track_ids))}:{event.alert_level}"
        now = event.monotonic_time

        if self.active_backend == "redis" and self.redis_client is not None:
            dedupe_seconds = system_config.scene_memory_dedupe_seconds
            if dedupe_seconds <= 0:
                return True
            redis_key = f"{self._dedupe_key_prefix()}:{hashlib.sha1(key.encode('utf-8')).hexdigest()}"
            return bool(self.redis_client.set(redis_key, "1", ex=max(1, math.ceil(dedupe_seconds)), nx=True))

        last_seen = self.last_event_keys.get(key, 0.0)
        if now - last_seen < system_config.scene_memory_dedupe_seconds:
            return False
        self.last_event_keys[key] = now
        return True

    def _store_event(self, event: MemoryEvent) -> None:
        if self.active_backend == "redis" and self.redis_client is not None:
            self.redis_client.rpush(self._events_key(), self._json_dumps(event.storage_dict()))
            self.redis_client.ltrim(self._events_key(), -system_config.scene_memory_max_events, -1)
            self.redis_client.expire(self._events_key(), max(1, math.ceil(system_config.scene_memory_ttl_seconds)))
            return

        self.events.append(event)

    def _expire_old(self, now: float) -> None:
        ttl = system_config.scene_memory_ttl_seconds

        if self.active_backend == "redis" and self.redis_client is not None:
            fresh_events = [
                event for event in self._redis_events(limit=system_config.scene_memory_max_events)
                if now - event.monotonic_time <= ttl
            ]
            if len(fresh_events) != self.redis_client.llen(self._events_key()):
                pipe = self.redis_client.pipeline()
                pipe.delete(self._events_key())
                if fresh_events:
                    pipe.rpush(self._events_key(), *[self._json_dumps(event.storage_dict()) for event in fresh_events])
                    pipe.expire(self._events_key(), max(1, math.ceil(ttl)))
                pipe.execute()
            return

        while self.events and now - self.events[0].monotonic_time > ttl:
            self.events.popleft()

        stale_keys = [
            key for key, timestamp in self.last_event_keys.items()
            if now - timestamp > ttl
        ]
        for key in stale_keys:
            self.last_event_keys.pop(key, None)

    def _all_events_unlocked(self) -> List[MemoryEvent]:
        if self.active_backend == "redis" and self.redis_client is not None:
            return self._redis_events(limit=system_config.scene_memory_max_events)
        return list(self.events)

    def _memory_size_unlocked(self) -> int:
        if self.active_backend == "redis" and self.redis_client is not None:
            return int(self.redis_client.llen(self._events_key()))
        return len(self.events)

    def _reset_memory(self) -> None:
        self.events.clear()
        self.object_history.clear()
        self.last_event_keys.clear()

    def _reset_redis(self) -> None:
        if self.redis_client is None:
            return
        keys = [self._events_key(), self._objects_key()]
        keys.extend(list(self.redis_client.scan_iter(f"{self._dedupe_key_prefix()}:*")))
        if keys:
            self.redis_client.delete(*keys)

    def _resize_memory_deques(self) -> None:
        if self.events.maxlen != system_config.scene_memory_max_events:
            self.events = deque(list(self.events)[-system_config.scene_memory_max_events:], maxlen=system_config.scene_memory_max_events)

        if any(history.maxlen != system_config.scene_memory_object_history for history in self.object_history.values()):
            resized: Dict[int, Deque[Dict[str, Any]]] = defaultdict(self._new_history_deque)
            for track_id, history in self.object_history.items():
                resized[track_id] = deque(
                    list(history)[-system_config.scene_memory_object_history:],
                    maxlen=system_config.scene_memory_object_history,
                )
            self.object_history = resized

    def _new_history_deque(self) -> Deque[Dict[str, Any]]:
        return deque(maxlen=system_config.scene_memory_object_history)

    def _connect_redis(self):
        if redis is None:
            self.redis_error = "redis package is not installed"
            return None

        try:
            client = redis.Redis.from_url(
                system_config.scene_memory_redis_url,
                decode_responses=True,
                socket_connect_timeout=0.35,
                socket_timeout=0.35,
            )
            client.ping()
            return client
        except Exception as exc:
            self.redis_error = str(exc)
            print(f"Scene memory Redis unavailable; falling back to in-process memory: {exc}")
            return None

    def _trim_redis_limits(self) -> None:
        if self.redis_client is None:
            return

        self.redis_client.ltrim(self._events_key(), -system_config.scene_memory_max_events, -1)
        histories = self.redis_client.hgetall(self._objects_key())
        for track_id, raw_history in histories.items():
            history = self._json_loads(raw_history, [])
            if isinstance(history, list):
                self.redis_client.hset(
                    self._objects_key(),
                    track_id,
                    self._json_dumps(history[-system_config.scene_memory_object_history:]),
                )

    def _redis_events(self, limit: int) -> List[MemoryEvent]:
        if self.redis_client is None:
            return []

        raw_events = self.redis_client.lrange(self._events_key(), -max(1, limit), -1)
        events: List[MemoryEvent] = []
        for raw_event in raw_events:
            payload = self._json_loads(raw_event, {})
            if isinstance(payload, dict):
                try:
                    events.append(MemoryEvent.from_dict(payload))
                except (TypeError, ValueError):
                    continue
        return events

    def _append_redis_object_history(self, track_id: int, entry: Dict[str, Any]) -> None:
        if self.redis_client is None:
            return

        track_key = str(track_id)
        raw_history = self.redis_client.hget(self._objects_key(), track_key)
        history = self._json_loads(raw_history, [])
        if not isinstance(history, list):
            history = []
        history.append(entry)
        history = history[-system_config.scene_memory_object_history:]

        self.redis_client.hset(self._objects_key(), track_key, self._json_dumps(history))
        self.redis_client.expire(self._objects_key(), max(1, math.ceil(system_config.scene_memory_ttl_seconds)))

    def _redis_object_history(self, limit: int) -> Dict[str, List[Dict[str, Any]]]:
        if self.redis_client is None:
            return {}

        snapshot: Dict[str, List[Dict[str, Any]]] = {}
        for track_id, raw_history in self.redis_client.hgetall(self._objects_key()).items():
            history = self._json_loads(raw_history, [])
            if isinstance(history, list):
                snapshot[str(track_id)] = history[-max(1, limit):]
        return snapshot

    def _object_history_entry(
        self,
        person: Dict[str, Any],
        zone_updates: Dict[int, Dict[str, Any]],
        now: float,
    ) -> Optional[Dict[str, Any]]:
        track_id = person.get("track_id")
        if track_id is None:
            return None

        track_id = int(track_id)
        zone_state = zone_updates.get(track_id, {})
        return {
            "track_id": track_id,
            "timestamp": self._iso_timestamp(now),
            "box": person.get("box"),
            "direction": person.get("direction"),
            "speed_px_s": person.get("speed_px_s"),
            "in_restricted_zone": zone_state.get("in_restricted_zone", person.get("in_restricted_zone", False)),
            "zone_event": zone_state.get("zone_event", person.get("zone_event")),
            "threat_tags": person.get("suspicious_behaviors") or [],
        }

    def _events_key(self) -> str:
        return f"{system_config.scene_memory_key_prefix}:events"

    def _objects_key(self) -> str:
        return f"{system_config.scene_memory_key_prefix}:objects"

    def _dedupe_key_prefix(self) -> str:
        return f"{system_config.scene_memory_key_prefix}:dedupe"

    @staticmethod
    def _configured_backend() -> str:
        backend = str(system_config.scene_memory_backend or "memory").strip().lower()
        return backend if backend in SUPPORTED_BACKENDS else "memory"

    @staticmethod
    def _safe_redis_url(url: str) -> str:
        if "@" not in url:
            return url
        scheme, rest = url.split("://", 1) if "://" in url else ("redis", url)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"

    @staticmethod
    def _json_dumps(payload: Any) -> str:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _json_loads(raw_value: Optional[str], fallback: Any) -> Any:
        if not raw_value:
            return fallback
        try:
            return json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _alert_summary(alert: Dict[str, Any], track_ids: List[int]) -> str:
        subject = f"Person {track_ids[0]} " if track_ids else ""
        event = str(alert.get("type") or "event").replace("_", " ")
        score = int(alert.get("threat_score") or 0)
        level = str(alert.get("alert_level") or alert.get("severity") or "SAFE")
        return f"{subject}{event} scored {score}/100 ({level})."

    @staticmethod
    def _event_id(event_type: str, track_ids: List[int], now: float) -> str:
        raw = f"{event_type}:{','.join(map(str, track_ids))}:{now:.3f}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _iso_timestamp(timestamp: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))

    @staticmethod
    def _embed(text: str, dimensions: int = 32) -> List[float]:
        vector = [0.0] * dimensions
        for token in text.lower().replace("/", " ").replace("_", " ").split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        return sum(left[index] * right[index] for index in range(size))
