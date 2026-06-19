import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = Path(os.getenv("MODELS_DIR", BACKEND_DIR / "models")).resolve()
SNAPSHOTS_DIR = Path(os.getenv("SNAPSHOTS_DIR", PROJECT_ROOT / "snapshots")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", PROJECT_ROOT / "uploads")).resolve()
FRONTEND_DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", PROJECT_ROOT / "frontend" / "dist")).resolve()
DB_PATH = Path(os.getenv("ALERTS_DB_PATH", PROJECT_ROOT / "alerts.db")).resolve()
ULTRALYTICS_CONFIG_DIR = Path(
    os.getenv("YOLO_CONFIG_DIR", PROJECT_ROOT / ".cache" / "ultralytics")
).resolve()
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")


def _env_list(name: str, default: List[str]) -> List[str]:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _env_float_map(prefix: str, defaults: Dict[str, float]) -> Dict[str, float]:
    values = defaults.copy()
    for key in defaults:
        env_key = f"{prefix}_{key.upper()}"
        if env_key in os.environ:
            try:
                values[key] = float(os.environ[env_key])
            except ValueError:
                    print(f"Ignoring invalid float value for {env_key}: {os.environ[env_key]}")
    return values


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _clamp_int(value: Any, lower: int, upper: int) -> int:
    return min(max(int(value), lower), upper)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_point(raw_point: Any, width: int, height: int) -> Optional[Tuple[int, int]]:
    if isinstance(raw_point, dict):
        raw_x = raw_point.get("x")
        raw_y = raw_point.get("y")
    elif isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
        raw_x = raw_point[0]
        raw_y = raw_point[1]
    else:
        return None

    try:
        return _clamp_int(raw_x, 0, width), _clamp_int(raw_y, 0, height)
    except (TypeError, ValueError):
        return None


def _looks_like_point(raw_value: Any) -> bool:
    if isinstance(raw_value, dict):
        return "x" in raw_value and "y" in raw_value
    return (
        isinstance(raw_value, (list, tuple))
        and len(raw_value) >= 2
        and _is_number(raw_value[0])
        and _is_number(raw_value[1])
    )


def _normalize_box_zone(raw_zone: Any, width: int, height: int) -> List[Tuple[int, int]]:
    try:
        if isinstance(raw_zone, dict):
            if {"x", "y", "width", "height"}.issubset(raw_zone):
                x1 = int(raw_zone["x"])
                y1 = int(raw_zone["y"])
                x2 = x1 + int(raw_zone["width"])
                y2 = y1 + int(raw_zone["height"])
            elif {"x1", "y1", "x2", "y2"}.issubset(raw_zone):
                x1 = int(raw_zone["x1"])
                y1 = int(raw_zone["y1"])
                x2 = int(raw_zone["x2"])
                y2 = int(raw_zone["y2"])
            else:
                return []
        elif (
            isinstance(raw_zone, (list, tuple))
            and len(raw_zone) == 4
            and all(_is_number(value) for value in raw_zone)
        ):
            x1 = int(raw_zone[0])
            y1 = int(raw_zone[1])
            x2 = x1 + int(raw_zone[2])
            y2 = y1 + int(raw_zone[3])
        else:
            return []
    except (TypeError, ValueError):
        return []

    left = _clamp_int(min(x1, x2), 0, width)
    right = _clamp_int(max(x1, x2), 0, width)
    top = _clamp_int(min(y1, y2), 0, height)
    bottom = _clamp_int(max(y1, y2), 0, height)
    if right <= left or bottom <= top:
        return []
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _normalize_zone(raw_zone: Any, width: int, height: int) -> List[Tuple[int, int]]:
    if isinstance(raw_zone, dict) and "points" in raw_zone:
        raw_zone = raw_zone["points"]

    box_zone = _normalize_box_zone(raw_zone, width, height)
    if box_zone:
        return box_zone

    if not isinstance(raw_zone, (list, tuple)):
        return []

    points = [
        point
        for point in (_normalize_point(raw_point, width, height) for raw_point in raw_zone)
        if point is not None
    ]
    return points if len(points) >= 3 else []


def normalize_ignore_zones(raw_zones: Any, width: int, height: int) -> List[List[Tuple[int, int]]]:
    if isinstance(raw_zones, dict) and "zones" in raw_zones:
        raw_zones = raw_zones["zones"]
    elif isinstance(raw_zones, dict):
        raw_zones = [raw_zones]

    if not isinstance(raw_zones, (list, tuple)):
        return []

    if raw_zones and all(_looks_like_point(item) for item in raw_zones):
        zone_items = [raw_zones]
    else:
        zone_items = raw_zones

    zones: List[List[Tuple[int, int]]] = []
    for raw_zone in zone_items:
        zone = _normalize_zone(raw_zone, width, height)
        if zone:
            zones.append(zone)
    return zones[:50]


def _env_ignore_zones(width: int, height: int) -> List[List[Tuple[int, int]]]:
    raw_value = os.getenv("IGNORE_ZONE_COORDS", "")
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        print(f"Ignoring invalid JSON value for IGNORE_ZONE_COORDS: {raw_value}")
        return []
    return normalize_ignore_zones(parsed, width, height)


def _normalize_authorized_person(raw_person: Dict[str, Any], index: int) -> Dict[str, Any]:
    name = str(raw_person.get("name") or "").strip()
    person_id = str(raw_person.get("id") or "").strip()
    if not name:
        name = f"Authorized Person {index + 1}"
    if not person_id:
        person_id = name.lower().replace(" ", "-")

    access_zones = raw_person.get("access_zones", ["CAM_MAIN_ENTRANCE_01"])
    if not isinstance(access_zones, list):
        access_zones = ["CAM_MAIN_ENTRANCE_01"]

    return {
        "id": person_id,
        "name": name,
        "role": str(raw_person.get("role") or "Authorized").strip() or "Authorized",
        "access_zones": [str(zone).strip() for zone in access_zones if str(zone).strip()],
        "enabled": bool(raw_person.get("enabled", True)),
        "present": bool(raw_person.get("present", False)),
        "intrusion_bypass": bool(raw_person.get("intrusion_bypass", False)),
    }


def _env_authorized_people() -> List[Dict[str, Any]]:
    raw_value = os.getenv("AUTHORIZED_PEOPLE", "")
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = [{"name": item.strip()} for item in raw_value.split(",") if item.strip()]

    if not isinstance(parsed, list):
        return []

    people = []
    for index, item in enumerate(parsed):
        if isinstance(item, str):
            item = {"name": item}
        if isinstance(item, dict):
            people.append(_normalize_authorized_person(item, index))
    return people

class SystemConfig:
    allowed_modules = {
        "weapon_detection",
        "fire_detection",
        "fall_detection",
        "crowd_detection",
        "trespassing_detection",
    }
    module_aliases = {
        "intrusion_detection": "trespassing_detection",
        "smoke_detection": "fire_detection",
        "crowd_anomaly": "crowd_detection",
    }

    def __init__(self):
        # Frame and camera configuration
        self.camera_device_id = int(os.getenv("CAMERA_DEVICE_ID", "0"))
        self.frame_width = int(os.getenv("FRAME_WIDTH", "640"))
        self.frame_height = int(os.getenv("FRAME_HEIGHT", "480"))
        self.target_fps = max(1, int(os.getenv("TARGET_FPS", "20")))
        self.inference_fps = max(0.2, float(os.getenv("INFERENCE_FPS", "6.0")))
        self.jpeg_quality = min(max(int(os.getenv("JPEG_QUALITY", "65")), 35), 95)
        self.model_imgsz = min(max(int(os.getenv("MODEL_IMGSZ", "640")), 256), 1280)
        self.detection_iou = _clamp(float(os.getenv("DETECTION_IOU", "0.50")), 0.10, 0.90)
        self.augment_inference = _env_bool("AUGMENT_INFERENCE", False)
        self.model_files = {
            "weapon": os.getenv("WEAPON_MODEL_FILE", "weapon.pt"),
            "weapon_secondary": os.getenv("WEAPON_SECONDARY_MODEL_FILE", "rtdetr_weapon.pt"),
            "fire": os.getenv("FIRE_MODEL_FILE", "fire.pt"),
            "fall": os.getenv("FALL_MODEL_FILE", "fall_model.pt"),
            "yolo11n": os.getenv("PERSON_MODEL_FILE", "yolo11n.pt"),
        }
        self.person_confidence_threshold = _clamp(float(os.getenv("PERSON_CONFIDENCE", "0.35")), 0.05, 0.99)
        self.tracking_enabled = _env_bool("TRACKING_ENABLED", True)
        self.tracker_config = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")
        self.tracking_confidence_threshold = _clamp(float(os.getenv("TRACKING_CONFIDENCE", "0.10")), 0.01, 0.99)
        self.tracking_history_size = min(max(int(os.getenv("TRACKING_HISTORY_SIZE", "12")), 2), 60)
        self.tracking_lost_seconds = _clamp(float(os.getenv("TRACKING_LOST_SECONDS", "4.0")), 0.5, 30.0)
        self.loitering_seconds = _clamp(float(os.getenv("LOITERING_SECONDS", "20.0")), 2.0, 600.0)
        self.boundary_crossing_threshold = min(max(int(os.getenv("BOUNDARY_CROSSING_THRESHOLD", "3")), 2), 20)
        self.running_speed_px_s = _clamp(float(os.getenv("RUNNING_SPEED_PX_S", "420.0")), 50.0, 5000.0)
        self.weapon_roi_pass_enabled = _env_bool("WEAPON_ROI_PASS", True)
        self.weapon_roi_max_persons = min(max(int(os.getenv("WEAPON_ROI_MAX_PERSONS", "4")), 0), 12)
        self.weapon_roi_scale = _clamp(float(os.getenv("WEAPON_ROI_SCALE", "1.35")), 1.0, 2.5)
        self.weapon_roi_imgsz = min(max(int(os.getenv("WEAPON_ROI_IMGSZ", "640")), 256), 1280)
        self.weapon_secondary_verifier_enabled = _env_bool("WEAPON_SECONDARY_VERIFIER_ENABLED", False)
        self.weapon_secondary_min_confidence = _clamp(float(os.getenv("WEAPON_SECONDARY_MIN_CONFIDENCE", "0.65")), 0.05, 0.99)
        self.weapon_secondary_iou = _clamp(float(os.getenv("WEAPON_SECONDARY_IOU", "0.25")), 0.01, 0.90)
        self.temporal_decay = _clamp(float(os.getenv("TEMPORAL_DECAY", "0.55")), 0.0, 0.95)
        self.weapon_temporal_window = min(max(int(os.getenv("WEAPON_TEMPORAL_WINDOW", "10")), 3), 60)
        self.weapon_required_hits = min(
            max(int(os.getenv("WEAPON_REQUIRED_HITS", "8")), 1),
            self.weapon_temporal_window,
        )
        self.weapon_min_alert_confidence = _clamp(float(os.getenv("WEAPON_MIN_ALERT_CONFIDENCE", "0.75")), 0.05, 0.99)
        self.weapon_required_frames = min(
            max(int(os.getenv("WEAPON_REQUIRED_FRAMES", str(self.weapon_required_hits))), 1),
            self.weapon_temporal_window,
        )
        self.fire_required_seconds = _clamp(float(os.getenv("FIRE_REQUIRED_SECONDS", "3.0")), 0.1, 60.0)
        self.fall_no_movement_seconds = _clamp(float(os.getenv("FALL_NO_MOVEMENT_SECONDS", "5.0")), 0.5, 60.0)
        self.fall_no_movement_speed_px_s = _clamp(float(os.getenv("FALL_NO_MOVEMENT_SPEED_PX_S", "30.0")), 0.0, 500.0)
        self.fall_candidate_lost_seconds = _clamp(float(os.getenv("FALL_CANDIDATE_LOST_SECONDS", "8.0")), 1.0, 120.0)
        self.zone_base_risk = _clamp(float(os.getenv("ZONE_BASE_RISK", "10.0")), 0.0, 40.0)
        self.threat_score_weights = _env_float_map("THREAT_WEIGHT", {
            "weapon_confidence": 50.0,
            "intrusion_severity": 25.0,
            "movement_aggression": 25.0,
            "time_sensitivity": 12.0,
            "zone_risk": 15.0,
        })
        self.scene_memory_enabled = _env_bool("SCENE_MEMORY_ENABLED", True)
        self.scene_memory_backend = os.getenv("SCENE_MEMORY_BACKEND", "memory").strip().lower()
        self.scene_memory_redis_url = os.getenv("SCENE_MEMORY_REDIS_URL", "redis://localhost:6379/0")
        self.scene_memory_key_prefix = os.getenv("SCENE_MEMORY_KEY_PREFIX", "rakshak:scene_memory").strip() or "rakshak:scene_memory"
        self.scene_memory_max_events = min(max(int(os.getenv("SCENE_MEMORY_MAX_EVENTS", "500")), 20), 5000)
        self.scene_memory_ttl_seconds = _clamp(float(os.getenv("SCENE_MEMORY_TTL_SECONDS", "900.0")), 30.0, 86400.0)
        self.scene_memory_dedupe_seconds = _clamp(float(os.getenv("SCENE_MEMORY_DEDUPE_SECONDS", "2.0")), 0.0, 60.0)
        self.scene_memory_timeline_limit = min(max(int(os.getenv("SCENE_MEMORY_TIMELINE_LIMIT", "25")), 5), 200)
        self.scene_memory_object_history = min(max(int(os.getenv("SCENE_MEMORY_OBJECT_HISTORY", "20")), 3), 200)
        self.scene_memory_min_scene_score = min(max(int(os.getenv("SCENE_MEMORY_MIN_SCENE_SCORE", "10")), 0), 100)
        self.multi_agent_enabled = _env_bool("MULTI_AGENT_ENABLED", True)
        self.multi_agent_framework = os.getenv("MULTI_AGENT_FRAMEWORK", "langgraph").strip().lower()
        self.multi_agent_min_verification_score = _clamp(float(os.getenv("MULTI_AGENT_MIN_VERIFICATION_SCORE", "0.55")), 0.0, 1.0)
        self.multi_agent_suppress_unverified = _env_bool("MULTI_AGENT_SUPPRESS_UNVERIFIED", False)
        self.advanced_behavior_enabled = _env_bool("ADVANCED_BEHAVIOR_ENABLED", True)
        self.fight_proximity_px = _clamp(float(os.getenv("FIGHT_PROXIMITY_PX", "85.0")), 20.0, 300.0)
        self.panic_running_count = min(max(int(os.getenv("PANIC_RUNNING_COUNT", "3")), 1), 20)
        self.stampede_crowd_threshold = min(max(int(os.getenv("STAMPEDE_CROWD_THRESHOLD", "10")), 2), 100)
        self.abandoned_object_seconds = _clamp(float(os.getenv("ABANDONED_OBJECT_SECONDS", "30.0")), 5.0, 600.0)
        self.suspicious_movement_speed_px_s = _clamp(float(os.getenv("SUSPICIOUS_MOVEMENT_SPEED_PX_S", "300.0")), 30.0, 5000.0)
        self.face_recognition_enabled = _env_bool("FACE_RECOGNITION_ENABLED", False)
        self.face_gallery_path = os.getenv("FACE_GALLERY_PATH", str(PROJECT_ROOT / "face_gallery.json"))
        self.face_blacklist_path = os.getenv("FACE_BLACKLIST_PATH", str(PROJECT_ROOT / "face_blacklist.json"))
        self.face_match_threshold = _clamp(float(os.getenv("FACE_MATCH_THRESHOLD", "0.45")), 0.1, 0.95)
        self.face_model_name = os.getenv("FACE_MODEL_NAME", "buffalo_l")
        self.face_ctx_id = int(os.getenv("FACE_CTX_ID", "-1"))
        self.audio_intelligence_enabled = _env_bool("AUDIO_INTELLIGENCE_ENABLED", True)
        self.audio_gunshot_peak_threshold = _clamp(float(os.getenv("AUDIO_GUNSHOT_PEAK_THRESHOLD", "0.80")), 0.05, 1.0)
        self.audio_scream_centroid_threshold = _clamp(float(os.getenv("AUDIO_SCREAM_CENTROID_HZ", "2800.0")), 200.0, 12000.0)
        self.audio_glass_high_freq_ratio = _clamp(float(os.getenv("AUDIO_GLASS_HIGH_FREQ_RATIO", "0.28")), 0.01, 1.0)
        self.edge_ai_enabled = _env_bool("EDGE_AI_ENABLED", False)
        self.edge_profile = os.getenv("EDGE_PROFILE", "cpu").strip().lower()
        self.edge_device_hint = os.getenv("EDGE_DEVICE_HINT", "").strip()
        self.fire_min_color_density = _clamp(float(os.getenv("FIRE_MIN_COLOR_DENSITY", "0.010")), 0.0, 1.0)
        self.fall_posture_fallback_enabled = _env_bool("FALL_POSTURE_FALLBACK", True)
        
        # Default Active Modules
        self.active_modules = self.normalize_modules(_env_list("ACTIVE_MODULES", [
            "weapon_detection",
            "fire_detection",
            "fall_detection",
            "crowd_detection",
            "trespassing_detection"
        ]))
        
        # Candidate thresholds are class-specific so small-object recall does not
        # force critical alerts to accept weak detections.
        self.confidence_thresholds = {
            key: _clamp(value, 0.05, 0.99)
            for key, value in _env_float_map("CONFIDENCE", {
            "weapon_detection": 0.60,
            "fire_detection": 0.50,
            "fall_detection": 0.45,
            "crowd_detection": 0.35,
            "trespassing_detection": 0.35
            }).items()
        }

        self.alert_trigger_scores = {
            key: _clamp(value, 0.05, 3.0)
            for key, value in _env_float_map("ALERT_SCORE", {
            "weapon_detection": 0.72,
            "fire_detection": 0.78,
            "fall_detection": 0.82,
            "crowd_detection": 1.20,
            "trespassing_detection": 1.00,
            }).items()
        }
        self.instant_confidence = {
            key: _clamp(value, 0.05, 0.99)
            for key, value in _env_float_map("INSTANT_CONFIDENCE", {
            "weapon_detection": 0.85,
            "fire_detection": 0.80,
            "fall_detection": 0.80,
            "crowd_detection": 0.99,
            "trespassing_detection": 0.95,
            }).items()
        }
        
        # Alert Cooldown periods per module (in seconds)
        self.cooldown_periods = _env_float_map("COOLDOWN", {
            "weapon_detection": 15.0,
            "fire_detection": 15.0,
            "fall_detection": 15.0,
            "crowd_detection": 15.0,
            "trespassing_detection": 15.0
        })
        
        # SMTP email alerts toggles
        self.smtp_enabled = False
        
        # SMS Mobile Alerts toggles
        self.sms_enabled = False
        self.to_phone = os.getenv("TO_PHONE", "")
        
        # Default restricted zone vertices (scaled for 640x480 resolution)
        self.restricted_zone_coords = [
            (96, 312),
            (544, 312),
            (608, 456),
            (32, 456)
        ]
        self.ignore_zone_coords = _env_ignore_zones(self.frame_width, self.frame_height)
        self.authorized_people = _env_authorized_people()
        
        # Video stream source: "webcam" or "file"
        self.source_type = os.getenv("SOURCE_TYPE", "webcam")
        if self.source_type not in {"webcam", "file"}:
            self.source_type = "webcam"
        self.video_filepath = os.getenv("VIDEO_FILEPATH", "")

        self.cors_origins = _env_list("CORS_ORIGINS", [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ])

    @classmethod
    def normalize_module(cls, module: str) -> str:
        normalized = str(module).strip()
        return cls.module_aliases.get(normalized, normalized)

    @classmethod
    def normalize_modules(cls, modules: List[str]) -> List[str]:
        deduped = []
        for module in modules:
            normalized = cls.normalize_module(module)
            if normalized in cls.allowed_modules and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    @staticmethod
    def normalize_authorized_people(people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_people = []
        seen_ids = set()
        for index, item in enumerate(people):
            if not isinstance(item, dict):
                continue
            person = _normalize_authorized_person(item, index)
            if person["id"] in seen_ids:
                person["id"] = f"{person['id']}-{index + 1}"
            seen_ids.add(person["id"])
            normalized_people.append(person)
        return normalized_people[:50]

    def authorized_people_for_camera(self, camera_id: str) -> List[Dict[str, Any]]:
        authorized = []
        for person in self.authorized_people:
            zones = person.get("access_zones") or []
            if (
                person.get("enabled")
                and person.get("present")
                and person.get("intrusion_bypass")
                and (camera_id in zones or "*" in zones)
            ):
                authorized.append(person)
        return authorized

system_config = SystemConfig()
