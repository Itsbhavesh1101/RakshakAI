import asyncio
import base64
import re
import threading
import time
import shutil
from typing import Annotated, List, Dict, Tuple, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_modules.detection_engine import detection_engine
from ai_modules.perimeter_engine import check_restricted_zones
from services.notifier import notifier_service
import database as db
from config import FRONTEND_DIST_DIR, UPLOADS_DIR, system_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load active modules in a background thread to prevent first-hit detection latency
    def warm_up():
        print("Warm-up: Pre-loading active YOLO models in the background...")
        for module in system_config.active_modules:
            try:
                detection_engine.preload_module(module)
            except Exception as e:
                print(f"Warm-up error pre-loading model for {module}: {e}")
        print("Warm-up complete. All active YOLO models are pre-loaded and ready!")

    threading.Thread(target=warm_up, daemon=True).start()
    video_capture_worker.start()
    yield
    video_capture_worker.stop()

app = FastAPI(title="Rakshak AI Surveillance Platform API", version="1.0.0", lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=system_config.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db.init_db()

# --- Frame Ingestion Layer (Singleton Worker Thread) ---
class VideoCaptureWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {
            "anomaly_detected": False,
            "alerts": [],
            "stats": {
                "crowd_count": 0,
                "inference_ms": 0,
                "inference_fps": system_config.inference_fps,
            }
        }
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_frame_id = 0
        self.last_processed_frame_id = 0
        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.last_alert_times: Dict[str, float] = {}
        self.cached_alerts: List[Dict[str, Any]] = []
        self.cached_person_boxes: List[List[int]] = []
        self.cached_tracked_people: List[Dict[str, Any]] = []
        self.cached_trespassing_boxes: List[List[int]] = []
        self.last_inference_ms = 0.0
        self.trespassing_score = 0

    def _set_latest(
        self,
        frame_jpeg: Optional[bytes] = None,
        telemetry: Optional[Dict[str, Any]] = None
    ):
        with self.state_lock:
            if frame_jpeg is not None:
                self.latest_frame_jpeg = frame_jpeg
            if telemetry is not None:
                self.latest_telemetry = telemetry

    def _set_raw_frame(self, frame: np.ndarray):
        with self.frame_lock:
            self.latest_raw_frame = frame
            self.latest_frame_id += 1

    def _get_raw_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self.frame_lock:
            if self.latest_raw_frame is None:
                return None, self.latest_frame_id
            return self.latest_raw_frame.copy(), self.latest_frame_id

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(system_config.jpeg_quality)]
        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
        return jpeg.tobytes() if ok else b""

    @staticmethod
    def _dispatch_alert_notification(
        alert_type: str,
        severity: str,
        frame_jpeg: bytes,
        alert_id: int
    ):
        def update_description(description: str):
            db.update_alert_description(alert_id, description)

        try:
            asyncio.run(
                notifier_service.schedule_notification_dispatch(
                    alert_type=alert_type,
                    severity=severity,
                    frame_jpeg=frame_jpeg,
                    callback_on_enrichment=update_description,
                )
            )
        except Exception as exc:
            print(f"Alert notification worker failed for alert #{alert_id}: {exc}")

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        if frame.size == 0:
            return frame
        return cv2.resize(frame, (system_config.frame_width, system_config.frame_height))

    def _draw_hud_header(self, frame: np.ndarray):
        cv2.rectangle(frame, (8, 8), (385, 35), (15, 15, 22), -1)
        cv2.putText(
            frame,
            f"RAKSHAK SURVEILLANCE SUITE | AI {self.last_inference_ms:.0f}ms",
            (18, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_alert_boxes(frame: np.ndarray, alerts: List[Dict[str, Any]]):
        styles = {
            "weapon_detection": ((255, 0, 255), "WEAPON"),
            "fire_detection": ((0, 69, 255), "FIRE/SMOKE"),
            "fall_detection": ((0, 0, 255), "FALL"),
            "crowd_detection": ((0, 255, 255), "CROWD"),
        }
        for alert in alerts:
            alert_type = alert.get("type")
            if alert_type == "trespassing_detection":
                continue
            color, label = styles.get(alert_type, ((0, 220, 255), "ALERT"))
            confidence = float(alert.get("confidence") or 0.0)
            for box in alert.get("bounding_boxes", []):
                if len(box) != 4:
                    continue
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x, max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    @staticmethod
    def _draw_track_labels(frame: np.ndarray, tracked_people: List[Dict[str, Any]]):
        for person in tracked_people:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            box = person.get("box") or []
            if len(box) != 4:
                continue
            x, y, _, _ = [int(value) for value in box]
            label = f"Person {track_id} {person.get('direction', 'moving')}"
            cv2.putText(
                frame,
                label,
                (x, max(16, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

    def _compose_display_frame(self, frame: np.ndarray) -> np.ndarray:
        with self.state_lock:
            alerts = [alert.copy() for alert in self.cached_alerts]
            person_boxes = [box[:] for box in self.cached_person_boxes]
            tracked_people = [person.copy() for person in self.cached_tracked_people]

        display_frame = frame.copy()
        if "trespassing_detection" in system_config.active_modules:
            display_frame, _, _ = check_restricted_zones(
                display_frame,
                list(system_config.restricted_zone_coords),
                person_boxes,
                system_config.authorized_people_for_camera("CAM_MAIN_ENTRANCE_01"),
            )
        self._draw_alert_boxes(display_frame, alerts)
        self._draw_track_labels(display_frame, tracked_people)
        self._draw_hud_header(display_frame)
        return display_frame

    def _handle_alerts(self, alerts: List[Dict[str, Any]], raw_frame: np.ndarray, frame_bytes: bytes):
        current_time = time.time()
        for alert in alerts:
            alert_type = alert["type"]
            severity = alert["severity"]
            last_alert_time = self.last_alert_times.get(alert_type, 0.0)
            cooldown = system_config.cooldown_periods.get(alert_type, 15.0)

            if current_time - last_alert_time < cooldown:
                continue

            self.last_alert_times[alert_type] = current_time
            snapshot_filename = f"snap_{alert_type}_{int(current_time)}.jpg"
            snapshot_path = db.SNAPSHOTS_DIR / snapshot_filename
            cv2.imwrite(str(snapshot_path), raw_frame)

            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            # Create descriptive, context-aware dynamic alert messages
            primary_subject = alert.get("primary_subject")
            subject_prefix = f"{primary_subject}: " if primary_subject else ""
            message_templates = {
                "weapon_detection": f"[CRITICAL ALARM] {subject_prefix}Active threat: Individual armed with a weapon/firearm detected at CAM_MAIN_ENTRANCE_01. Stand by for guard dispatch.",
                "fire_detection": "[CRITICAL ALARM] Thermal hazard: Active flame/smoke coordinates detected at CAM_MAIN_ENTRANCE_01. Dispatching local extinguishers.",
                "fall_detection": f"[WARNING] {subject_prefix}Safety breach: Individual down (possible fall or injury) at CAM_MAIN_ENTRANCE_01. Check-in initiated.",
                "crowd_detection": f"[WARNING] Assembly alert: Ingress flow congestion with {alert.get('live_count', 0)} individuals gathered at CAM_MAIN_ENTRANCE_01.",
                "trespassing_detection": f"[CRITICAL ALARM] {subject_prefix}Restricted breach: Unauthorised entry of {alert.get('live_count', 1)} intruder(s) inside danger polygon zone at CAM_MAIN_ENTRANCE_01."
            }
            message = message_templates.get(
                alert_type, 
                f"{alert_type.replace('_', ' ').title()} threat alert detected at ingress nodes."
            )
            if "threat_score" in alert:
                message = f"{message} Threat score: {alert['threat_score']}/100 ({alert.get('alert_level', severity)})."
            if alert.get("agent_narrative"):
                message = f"{message} Agent assessment: {alert['agent_narrative']}"

            alert_id = db.log_alert_to_db(
                timestamp=timestamp_str,
                module_name=alert_type.replace("_", " ").title(),
                severity=severity,
                message=message,
                snapshot_filename=f"snapshots/{snapshot_filename}",
            )

            if frame_bytes:
                notification_thread = threading.Thread(
                    target=self._dispatch_alert_notification,
                    args=(alert_type, severity, frame_bytes, alert_id),
                    daemon=True,
                )
                notification_thread.start()

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
                self.capture_thread.start()
                self.inference_thread.start()
                print("VideoCaptureWorker capture and inference threads started.")

    def stop(self):
        with self.lock:
            self.running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        for thread in (self.capture_thread, self.inference_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        print("VideoCaptureWorker threads stopped.")

    def _get_capture(self) -> cv2.VideoCapture:
        """Configures and opens the capture device or file."""
        if system_config.source_type == "webcam":
            print(f"Opening webcam camera device index {system_config.camera_device_id}...")
            cap = cv2.VideoCapture(system_config.camera_device_id)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, system_config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, system_config.frame_height)
        else:
            print(f"Opening video file stream '{system_config.video_filepath}'...")
            cap = cv2.VideoCapture(system_config.video_filepath)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        while self.running:
            loop_started_at = time.perf_counter()
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = self._get_capture()
                    if not self.cap.isOpened():
                        print("Failed to open capture source. Retrying in 3 seconds...")
                        self.cap = None
                        # Generate temporary warning card
                        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            dummy, 
                            "CAMERA CONNECTION LOST", 
                            (100, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (0, 0, 255), 
                            2
                        )
                        frame_bytes = self._encode_jpeg(dummy)
                        if frame_bytes:
                            self._set_latest(frame_jpeg=frame_bytes)
                        time.sleep(3.0)
                        continue

                ret, frame = self.cap.read()
                if not ret:
                    if system_config.source_type == "file":
                        # Loop video file seamlessly
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Camera feed frames dropped. Releasing and reconnecting...")
                        self.cap.release()
                        self.cap = None
                        time.sleep(1.0)
                        continue

                # Process the frame
                frame = self._normalize_frame(frame)
                self._set_raw_frame(frame)

                display_frame = self._compose_display_frame(frame)
                frame_bytes = self._encode_jpeg(display_frame)
                if frame_bytes:
                    self._set_latest(frame_jpeg=frame_bytes)
                
                elapsed = time.perf_counter() - loop_started_at
                frame_interval = 1.0 / max(1, system_config.target_fps)
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
            except Exception as e:
                print(f"Error in video capture thread: {e}")
                time.sleep(1.0)

    def _inference_loop(self):
        while self.running:
            loop_started_at = time.perf_counter()
            try:
                frame, frame_id = self._get_raw_frame()
                if frame is None or frame_id == self.last_processed_frame_id:
                    time.sleep(0.02)
                    continue

                self.last_processed_frame_id = frame_id
                enabled_modules = list(system_config.active_modules)
                thresholds = dict(system_config.confidence_thresholds)

                display_frame, alerts, stats = detection_engine.run_yolo_checks(
                    frame,
                    enabled_modules,
                    thresholds,
                )

                person_boxes = stats.get("person_boxes", [])
                tracked_people = stats.get("tracked_people", [])
                violators: List[List[int]] = []
                zone_updates: Dict[int, Dict[str, Any]] = {}
                if "trespassing_detection" in enabled_modules:
                    display_frame, trespassing_alerts, violators = check_restricted_zones(
                        display_frame,
                        list(system_config.restricted_zone_coords),
                        person_boxes,
                        system_config.authorized_people_for_camera("CAM_MAIN_ENTRANCE_01"),
                    )
                    trespassing_alerts, tracked_people, zone_updates = detection_engine.enrich_zone_alerts(
                        trespassing_alerts,
                        violators,
                        tracked_people,
                    )
                    stats["scene_threat"] = detection_engine.threat_scoring_engine.score_scene(
                        tracked_people,
                        zone_updates,
                    )
                    if violators:
                        self.trespassing_score = min(self.trespassing_score + 2, 6)
                    else:
                        self.trespassing_score = max(self.trespassing_score - 1, 0)
                        
                    if self.trespassing_score >= 3 and trespassing_alerts:
                        alerts.extend(trespassing_alerts)

                alerts = detection_engine.score_alerts(alerts, tracked_people, zone_updates)
                scene_memory = detection_engine.remember_scene(
                    alerts,
                    tracked_people,
                    stats.get("scene_threat", {}),
                    zone_updates,
                )
                alerts, multi_agent_report = detection_engine.run_multi_agent_analysis(
                    alerts,
                    tracked_people,
                    stats,
                    zone_updates,
                    scene_memory,
                )

                inference_ms = (time.perf_counter() - loop_started_at) * 1000.0
                self.last_inference_ms = inference_ms
                frame_bytes = self._encode_jpeg(display_frame)
                self._handle_alerts(alerts, frame, frame_bytes)

                telemetry = {
                    "anomaly_detected": len(alerts) > 0,
                    "alerts": alerts,
                    "stats": {
                        "crowd_count": stats.get("crowd_count", 0),
                        "person_count": len(person_boxes),
                        "tracked_person_count": stats.get("tracked_person_count", 0),
                        "active_track_ids": stats.get("active_track_ids", []),
                        "tracked_people": tracked_people,
                        "tracking_enabled": stats.get("tracking_enabled", system_config.tracking_enabled),
                        "tracker": stats.get("tracker"),
                        "trespassing_count": len(violators) if "trespassing_detection" in enabled_modules else 0,
                        "trespassing_score": self.trespassing_score,
                        "zone_motion": zone_updates,
                        "scene_threat": stats.get("scene_threat", {}),
                        "scene_memory": scene_memory,
                        "multi_agent": multi_agent_report,
                        "max_threat_score": max(
                            [int(alert.get("threat_score") or 0) for alert in alerts],
                            default=0,
                        ),
                        "inference_ms": round(inference_ms, 1),
                        "inference_fps": system_config.inference_fps,
                    }
                }

                with self.state_lock:
                    self.cached_alerts = alerts
                    self.cached_person_boxes = [box[:] for box in person_boxes]
                    self.cached_tracked_people = [person.copy() for person in tracked_people]
                    self.cached_trespassing_boxes = [box[:] for box in violators] if "trespassing_detection" in enabled_modules else []
                    self.latest_telemetry = telemetry

                if frame_bytes:
                    self._set_latest(frame_jpeg=frame_bytes)

                elapsed = time.perf_counter() - loop_started_at
                inference_interval = 1.0 / max(0.2, system_config.inference_fps)
                if elapsed < inference_interval:
                    time.sleep(inference_interval - elapsed)
            except Exception as e:
                print(f"Error in inference thread: {e}")
                time.sleep(1.0)

    def get_latest_data(self) -> Tuple[Optional[bytes], Dict[str, Any]]:
        with self.state_lock:
            return self.latest_frame_jpeg, self.latest_telemetry

video_capture_worker = VideoCaptureWorker()



# --- REST API Routing layer ---

@app.get("/api/config")
def get_config():
    """Returns current active model and source parameters."""
    return {
        "active_modules": system_config.active_modules,
        "confidence_thresholds": system_config.confidence_thresholds,
        "cooldown_periods": system_config.cooldown_periods,
        "restricted_zone_coords": system_config.restricted_zone_coords,
        "authorized_people": system_config.authorized_people,
        "source_type": system_config.source_type,
        "video_filepath": system_config.video_filepath,
        "smtp_enabled": system_config.smtp_enabled,
        "smtp_server": notifier_service.smtp_server,
        "to_address": notifier_service.to_address,
        "sms_enabled": system_config.sms_enabled,
        "to_phone": notifier_service.to_phone,
        "camera_device_id": system_config.camera_device_id,
        "target_fps": system_config.target_fps,
        "inference_fps": system_config.inference_fps,
        "jpeg_quality": system_config.jpeg_quality,
        "model_imgsz": system_config.model_imgsz,
        "model_files": system_config.model_files,
        "detection_iou": system_config.detection_iou,
        "augment_inference": system_config.augment_inference,
        "person_confidence_threshold": system_config.person_confidence_threshold,
        "tracking_enabled": system_config.tracking_enabled,
        "tracker_config": system_config.tracker_config,
        "tracking_confidence_threshold": system_config.tracking_confidence_threshold,
        "tracking_history_size": system_config.tracking_history_size,
        "tracking_lost_seconds": system_config.tracking_lost_seconds,
        "loitering_seconds": system_config.loitering_seconds,
        "boundary_crossing_threshold": system_config.boundary_crossing_threshold,
        "running_speed_px_s": system_config.running_speed_px_s,
        "weapon_roi_pass_enabled": system_config.weapon_roi_pass_enabled,
        "weapon_roi_max_persons": system_config.weapon_roi_max_persons,
        "weapon_roi_scale": system_config.weapon_roi_scale,
        "weapon_roi_imgsz": system_config.weapon_roi_imgsz,
        "temporal_decay": system_config.temporal_decay,
        "weapon_required_frames": system_config.weapon_required_frames,
        "fire_required_seconds": system_config.fire_required_seconds,
        "fall_no_movement_seconds": system_config.fall_no_movement_seconds,
        "fall_no_movement_speed_px_s": system_config.fall_no_movement_speed_px_s,
        "fall_candidate_lost_seconds": system_config.fall_candidate_lost_seconds,
        "zone_base_risk": system_config.zone_base_risk,
        "threat_score_weights": system_config.threat_score_weights,
        "scene_memory_enabled": system_config.scene_memory_enabled,
        "scene_memory_backend": system_config.scene_memory_backend,
        "scene_memory_redis_url": system_config.scene_memory_redis_url,
        "scene_memory_key_prefix": system_config.scene_memory_key_prefix,
        "scene_memory_max_events": system_config.scene_memory_max_events,
        "scene_memory_ttl_seconds": system_config.scene_memory_ttl_seconds,
        "scene_memory_dedupe_seconds": system_config.scene_memory_dedupe_seconds,
        "scene_memory_timeline_limit": system_config.scene_memory_timeline_limit,
        "scene_memory_object_history": system_config.scene_memory_object_history,
        "scene_memory_min_scene_score": system_config.scene_memory_min_scene_score,
        "scene_memory_status": detection_engine.scene_memory.status(),
        "multi_agent_enabled": system_config.multi_agent_enabled,
        "multi_agent_framework": system_config.multi_agent_framework,
        "multi_agent_min_verification_score": system_config.multi_agent_min_verification_score,
        "multi_agent_suppress_unverified": system_config.multi_agent_suppress_unverified,
        "alert_level_ranges": {
            "SAFE": [0, 30],
            "WARNING": [31, 60],
            "HIGH RISK": [61, 85],
            "CRITICAL": [86, 100],
        },
        "alert_trigger_scores": system_config.alert_trigger_scores,
        "instant_confidence": system_config.instant_confidence,
        "fire_min_color_density": system_config.fire_min_color_density,
        "fall_posture_fallback_enabled": system_config.fall_posture_fallback_enabled,
    }

@app.post("/api/config")
def update_config(data: dict = Body(...)):
    """Dynamically updates backend settings in real-time."""
    if "active_modules" in data:
        system_config.active_modules = system_config.normalize_modules(data["active_modules"])
    if "confidence_thresholds" in data:
        for module, threshold in data["confidence_thresholds"].items():
            normalized_module = system_config.normalize_module(module)
            if normalized_module in system_config.confidence_thresholds:
                system_config.confidence_thresholds[normalized_module] = min(max(float(threshold), 0.05), 0.99)
    if "alert_trigger_scores" in data:
        for module, score in data["alert_trigger_scores"].items():
            normalized_module = system_config.normalize_module(module)
            if normalized_module in system_config.alert_trigger_scores:
                system_config.alert_trigger_scores[normalized_module] = min(max(float(score), 0.05), 3.0)
    if "instant_confidence" in data:
        for module, threshold in data["instant_confidence"].items():
            normalized_module = system_config.normalize_module(module)
            if normalized_module in system_config.instant_confidence:
                system_config.instant_confidence[normalized_module] = min(max(float(threshold), 0.05), 0.99)
    if "restricted_zone_coords" in data:
        # Convert JSON arrays back to tuple coords
        raw_coords = data["restricted_zone_coords"]
        parsed_coords = []
        for point in raw_coords:
            if not isinstance(point, list) or len(point) != 2:
                continue
            try:
                x = min(max(int(point[0]), 0), system_config.frame_width)
                y = min(max(int(point[1]), 0), system_config.frame_height)
            except (TypeError, ValueError):
                continue
            parsed_coords.append((x, y))
        system_config.restricted_zone_coords = parsed_coords[:8]
    if "authorized_people" in data:
        if not isinstance(data["authorized_people"], list):
            raise HTTPException(status_code=400, detail="Authorized people must be a list.")
        system_config.authorized_people = system_config.normalize_authorized_people(data["authorized_people"])
    if "smtp_enabled" in data:
        system_config.smtp_enabled = bool(data["smtp_enabled"])
        notifier_service.smtp_enabled = bool(data["smtp_enabled"])
    if "to_address" in data:
        notifier_service.to_address = data["to_address"]
    if "sms_enabled" in data:
        system_config.sms_enabled = bool(data["sms_enabled"])
        notifier_service.sms_enabled = bool(data["sms_enabled"])
    if "to_phone" in data:
        notifier_service.to_phone = data["to_phone"]
    
    # Check if video source has changed
    source_changed = False
    if (
        "source_type" in data
        and data["source_type"] in {"webcam", "file"}
        and data["source_type"] != system_config.source_type
    ):
        system_config.source_type = data["source_type"]
        source_changed = True
    if "video_filepath" in data and data["video_filepath"] != system_config.video_filepath:
        requested_path = str(data["video_filepath"]).strip()
        if requested_path:
            video_path = Path(requested_path).expanduser().resolve()
            if not video_path.exists() or not video_path.is_file():
                raise HTTPException(status_code=400, detail="Configured video file does not exist.")
            system_config.video_filepath = str(video_path)
        else:
            system_config.video_filepath = ""
        source_changed = True
    if "camera_device_id" in data and data["camera_device_id"] != system_config.camera_device_id:
        system_config.camera_device_id = max(0, int(data["camera_device_id"]))
        source_changed = True
    if "target_fps" in data:
        system_config.target_fps = min(max(int(data["target_fps"]), 1), 30)
    if "inference_fps" in data:
        system_config.inference_fps = min(max(float(data["inference_fps"]), 0.2), 10.0)
    if "jpeg_quality" in data:
        system_config.jpeg_quality = min(max(int(data["jpeg_quality"]), 35), 95)
    if "model_imgsz" in data:
        system_config.model_imgsz = min(max(int(data["model_imgsz"]), 256), 1280)
    if "model_files" in data:
        if not isinstance(data["model_files"], dict):
            raise HTTPException(status_code=400, detail="model_files must be an object.")
        model_file_changed = False
        for key, filename in data["model_files"].items():
            if key not in system_config.model_files:
                continue
            safe_filename = Path(str(filename)).name
            if safe_filename:
                system_config.model_files[key] = safe_filename
                detection_engine.model_files[key] = safe_filename
                detection_engine.model_service.model_files[key] = safe_filename
                detection_engine.models[key] = None
                detection_engine.model_service.models[key] = None
                model_file_changed = True
        if model_file_changed:
            detection_engine.reset_tracking()
    if "detection_iou" in data:
        system_config.detection_iou = min(max(float(data["detection_iou"]), 0.10), 0.90)
    if "augment_inference" in data:
        system_config.augment_inference = bool(data["augment_inference"])
    if "person_confidence_threshold" in data:
        system_config.person_confidence_threshold = min(max(float(data["person_confidence_threshold"]), 0.05), 0.99)
    if "tracking_enabled" in data:
        system_config.tracking_enabled = bool(data["tracking_enabled"])
        detection_engine.reset_tracking()
    if "tracker_config" in data:
        tracker_config = Path(str(data["tracker_config"])).name
        if tracker_config not in {"bytetrack.yaml", "botsort.yaml"}:
            raise HTTPException(status_code=400, detail="tracker_config must be bytetrack.yaml or botsort.yaml.")
        system_config.tracker_config = tracker_config
        detection_engine.reset_tracking()
    if "tracking_confidence_threshold" in data:
        system_config.tracking_confidence_threshold = min(max(float(data["tracking_confidence_threshold"]), 0.01), 0.99)
    if "tracking_history_size" in data:
        system_config.tracking_history_size = min(max(int(data["tracking_history_size"]), 2), 60)
        detection_engine.reset_tracking()
    if "tracking_lost_seconds" in data:
        system_config.tracking_lost_seconds = min(max(float(data["tracking_lost_seconds"]), 0.5), 30.0)
    if "loitering_seconds" in data:
        system_config.loitering_seconds = min(max(float(data["loitering_seconds"]), 2.0), 600.0)
    if "boundary_crossing_threshold" in data:
        system_config.boundary_crossing_threshold = min(max(int(data["boundary_crossing_threshold"]), 2), 20)
    if "running_speed_px_s" in data:
        system_config.running_speed_px_s = min(max(float(data["running_speed_px_s"]), 50.0), 5000.0)
    if "weapon_roi_pass_enabled" in data:
        system_config.weapon_roi_pass_enabled = bool(data["weapon_roi_pass_enabled"])
    if "weapon_roi_max_persons" in data:
        system_config.weapon_roi_max_persons = min(max(int(data["weapon_roi_max_persons"]), 0), 12)
    if "weapon_roi_scale" in data:
        system_config.weapon_roi_scale = min(max(float(data["weapon_roi_scale"]), 1.0), 2.5)
    if "weapon_roi_imgsz" in data:
        system_config.weapon_roi_imgsz = min(max(int(data["weapon_roi_imgsz"]), 256), 1280)
    if "temporal_decay" in data:
        system_config.temporal_decay = min(max(float(data["temporal_decay"]), 0.0), 0.95)
    if "weapon_required_frames" in data:
        system_config.weapon_required_frames = min(max(int(data["weapon_required_frames"]), 1), 60)
        detection_engine.reset_tracking()
    if "fire_required_seconds" in data:
        system_config.fire_required_seconds = min(max(float(data["fire_required_seconds"]), 0.1), 60.0)
        detection_engine.reset_tracking()
    if "fall_no_movement_seconds" in data:
        system_config.fall_no_movement_seconds = min(max(float(data["fall_no_movement_seconds"]), 0.5), 60.0)
        detection_engine.reset_tracking()
    if "fall_no_movement_speed_px_s" in data:
        system_config.fall_no_movement_speed_px_s = min(max(float(data["fall_no_movement_speed_px_s"]), 0.0), 500.0)
        detection_engine.reset_tracking()
    if "fall_candidate_lost_seconds" in data:
        system_config.fall_candidate_lost_seconds = min(max(float(data["fall_candidate_lost_seconds"]), 1.0), 120.0)
        detection_engine.reset_tracking()
    if "zone_base_risk" in data:
        system_config.zone_base_risk = min(max(float(data["zone_base_risk"]), 0.0), 40.0)
    if "threat_score_weights" in data:
        if not isinstance(data["threat_score_weights"], dict):
            raise HTTPException(status_code=400, detail="threat_score_weights must be an object.")
        for key, value in data["threat_score_weights"].items():
            if key in system_config.threat_score_weights:
                system_config.threat_score_weights[key] = min(max(float(value), 0.0), 100.0)
    scene_memory_reconfigure = False
    if "scene_memory_enabled" in data:
        system_config.scene_memory_enabled = bool(data["scene_memory_enabled"])
    if "scene_memory_backend" in data:
        backend = str(data["scene_memory_backend"]).strip().lower()
        if backend not in {"memory", "redis"}:
            raise HTTPException(status_code=400, detail="scene_memory_backend must be memory or redis.")
        system_config.scene_memory_backend = backend
        scene_memory_reconfigure = True
    if "scene_memory_redis_url" in data:
        redis_url = str(data["scene_memory_redis_url"]).strip()
        if not redis_url:
            raise HTTPException(status_code=400, detail="scene_memory_redis_url cannot be empty.")
        system_config.scene_memory_redis_url = redis_url
        scene_memory_reconfigure = True
    if "scene_memory_key_prefix" in data:
        key_prefix = str(data["scene_memory_key_prefix"]).strip()
        if not key_prefix:
            raise HTTPException(status_code=400, detail="scene_memory_key_prefix cannot be empty.")
        system_config.scene_memory_key_prefix = key_prefix
        scene_memory_reconfigure = True
    if "scene_memory_max_events" in data:
        system_config.scene_memory_max_events = min(max(int(data["scene_memory_max_events"]), 20), 5000)
        scene_memory_reconfigure = True
    if "scene_memory_ttl_seconds" in data:
        system_config.scene_memory_ttl_seconds = min(max(float(data["scene_memory_ttl_seconds"]), 30.0), 86400.0)
        scene_memory_reconfigure = True
    if "scene_memory_dedupe_seconds" in data:
        system_config.scene_memory_dedupe_seconds = min(max(float(data["scene_memory_dedupe_seconds"]), 0.0), 60.0)
    if "scene_memory_timeline_limit" in data:
        system_config.scene_memory_timeline_limit = min(max(int(data["scene_memory_timeline_limit"]), 5), 200)
    if "scene_memory_object_history" in data:
        system_config.scene_memory_object_history = min(max(int(data["scene_memory_object_history"]), 3), 200)
        scene_memory_reconfigure = True
    if "scene_memory_min_scene_score" in data:
        system_config.scene_memory_min_scene_score = min(max(int(data["scene_memory_min_scene_score"]), 0), 100)
    if scene_memory_reconfigure:
        detection_engine.scene_memory.reconfigure()
    if "multi_agent_enabled" in data:
        system_config.multi_agent_enabled = bool(data["multi_agent_enabled"])
    if "multi_agent_framework" in data:
        framework = str(data["multi_agent_framework"]).strip().lower()
        if framework not in {"langgraph", "sequential"}:
            raise HTTPException(status_code=400, detail="multi_agent_framework must be langgraph or sequential.")
        system_config.multi_agent_framework = framework
        detection_engine.multi_agent_orchestrator.graph = detection_engine.multi_agent_orchestrator._build_graph()
    if "multi_agent_min_verification_score" in data:
        system_config.multi_agent_min_verification_score = min(
            max(float(data["multi_agent_min_verification_score"]), 0.0),
            1.0,
        )
    if "multi_agent_suppress_unverified" in data:
        system_config.multi_agent_suppress_unverified = bool(data["multi_agent_suppress_unverified"])
    if "fire_min_color_density" in data:
        system_config.fire_min_color_density = min(max(float(data["fire_min_color_density"]), 0.0), 1.0)
    if "fall_posture_fallback_enabled" in data:
        system_config.fall_posture_fallback_enabled = bool(data["fall_posture_fallback_enabled"])

    if source_changed:
        print("Surveillance source changed. Re-initializing capture worker...")
        detection_engine.reset_tracking()
        video_capture_worker.stop()
        video_capture_worker.start()

    return {"status": "success", "message": "Configurations updated successfully."}

@app.get("/api/health")
def health_check():
    """Lightweight readiness probe for the API and capture loop."""
    frame_bytes, telemetry = video_capture_worker.get_latest_data()
    return {
        "status": "ok",
        "capture_running": video_capture_worker.running,
        "frame_ready": frame_bytes is not None,
        "target_fps": system_config.target_fps,
        "inference_fps": system_config.inference_fps,
        "jpeg_quality": system_config.jpeg_quality,
        "model_imgsz": system_config.model_imgsz,
        "model_files": system_config.model_files,
        "detection_iou": system_config.detection_iou,
        "tracking_enabled": system_config.tracking_enabled,
        "tracker": system_config.tracker_config,
        "stats": telemetry.get("stats", {}),
    }

@app.get("/api/alerts")
def get_alerts(limit: int = Query(default=100, ge=1, le=500)):
    """Serves the persistent logged history of threats."""
    return db.fetch_alert_history(limit=limit)

@app.post("/api/alerts/clear")
def clear_alerts():
    """Wipes alert database and cleans snapshots folders."""
    db.clear_alerts_table()
    return {"status": "success", "message": "Alert history and snapshot buffers cleared."}

@app.get("/api/stats")
def get_stats():
    """Fetches real-time status stats."""
    _, telemetry = video_capture_worker.get_latest_data()
    return telemetry["stats"]

@app.get("/api/scene_memory")
def get_scene_memory(limit: Annotated[int, Query(ge=1, le=200)] = 25):
    """Returns the recent scene memory timeline and object history."""
    return {
        "enabled": system_config.scene_memory_enabled,
        "status": detection_engine.scene_memory.status(),
        "timeline": detection_engine.scene_memory.timeline(limit=limit),
        "object_history": detection_engine.scene_memory.object_history_snapshot(
            limit=system_config.scene_memory_object_history
        ),
    }

@app.get("/api/scene_memory/search")
def search_scene_memory(
    q: Annotated[str, Query(min_length=1, max_length=300)],
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
):
    """Semantic search over scene memory summaries."""
    return {
        "query": q,
        "results": detection_engine.scene_memory.semantic_search(q, limit=limit),
    }

@app.post("/api/scene_memory/clear")
def clear_scene_memory():
    """Clears in-memory scene timeline and object history."""
    detection_engine.scene_memory.reset()
    return {"status": "success", "message": "Scene memory cleared."}

@app.post("/api/upload_video")
def upload_video(file: UploadFile = File(...)):
    """Accepts video file uploads to /uploads directory for testing."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename or "upload.mp4")
    file_path = UPLOADS_DIR / safe_filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "filepath": str(file_path)}

@app.get("/snapshots/{filename}")
def serve_snapshot(filename: str):
    """Serves visual alert snapshots directly."""
    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    fpath = db.SNAPSHOTS_DIR / safe_filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Snapshot evidence file not found.")
    return FileResponse(fpath)


# --- Global WebSocket Connection Layer ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast_json(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
                
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/api/ws/telemetry")
async def ws_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Periodically fetch latest JPEG frames and telemetry data, broadcasting it
            frame_bytes, telemetry = video_capture_worker.get_latest_data()
            if frame_bytes is not None:
                # Convert binary frame to base64 to bundle in text payload
                frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                payload = {
                    "frame": f"data:image/jpeg;base64,{frame_b64}",
                    "anomaly_detected": telemetry["anomaly_detected"],
                    "alerts": telemetry["alerts"],
                    "stats": telemetry["stats"]
                }
                await websocket.send_json(payload)
            await asyncio.sleep(1.0 / max(1, system_config.target_fps))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket exception: {e}")
        manager.disconnect(websocket)


# Serve compiled React frontend
if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
