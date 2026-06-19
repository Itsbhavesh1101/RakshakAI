from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import system_config

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


class FaceRecognitionService:
    """Optional InsightFace-backed employee verification and blacklist matching."""

    def __init__(self):
        self.face_app = None
        self.gallery = self._load_gallery()
        self.blacklist = self._load_blacklist()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": system_config.face_recognition_enabled,
            "provider": "insightface",
            "available": FaceAnalysis is not None,
            "gallery_size": len(self.gallery),
            "blacklist_size": len(self.blacklist),
            "threshold": system_config.face_match_threshold,
        }

    def analyze_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        if not system_config.face_recognition_enabled:
            return {"enabled": False, "matches": [], "faces": [], "status": self.status()}
        if FaceAnalysis is None:
            return {
                "enabled": True,
                "matches": [],
                "faces": [],
                "status": self.status(),
                "message": "InsightFace is not installed. Install insightface and onnxruntime to enable recognition.",
            }

        frame = self._decode_image(image_bytes)
        if frame is None:
            return {"enabled": True, "matches": [], "faces": [], "status": self.status(), "message": "Invalid image."}

        app = self._app()
        faces = app.get(frame)
        results = []
        for index, face in enumerate(faces):
            embedding = np.asarray(face.embedding, dtype=np.float32)
            match = self._best_match(embedding, self.gallery)
            blacklist_match = self._best_match(embedding, self.blacklist)
            result = {
                "face_index": index,
                "box": [int(value) for value in face.bbox.tolist()],
                "employee_match": match,
                "blacklist_match": blacklist_match,
                "verified_employee": bool(match and match["score"] >= system_config.face_match_threshold),
                "blacklist_alert": bool(blacklist_match and blacklist_match["score"] >= system_config.face_match_threshold),
            }
            results.append(result)

        return {
            "enabled": True,
            "status": self.status(),
            "faces": results,
            "matches": [item for item in results if item["verified_employee"] or item["blacklist_alert"]],
        }

    def _app(self):
        if self.face_app is None:
            self.face_app = FaceAnalysis(name=system_config.face_model_name)
            self.face_app.prepare(ctx_id=system_config.face_ctx_id, det_size=(640, 640))
        return self.face_app

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def _load_gallery(self) -> List[Dict[str, Any]]:
        return self._load_embeddings(system_config.face_gallery_path)

    def _load_blacklist(self) -> List[Dict[str, Any]]:
        return self._load_embeddings(system_config.face_blacklist_path)

    @staticmethod
    def _load_embeddings(path: str) -> List[Dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            return []
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        entries = payload if isinstance(payload, list) else payload.get("entries", [])
        normalized = []
        for item in entries:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                continue
            normalized.append({
                "id": str(item.get("id") or item.get("name") or "unknown"),
                "name": str(item.get("name") or item.get("id") or "unknown"),
                "role": str(item.get("role") or ""),
                "embedding": [float(value) for value in embedding],
            })
        return normalized

    @staticmethod
    def _best_match(embedding: np.ndarray, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        norm = np.linalg.norm(embedding) or 1.0
        for candidate in candidates:
            candidate_embedding = np.asarray(candidate["embedding"], dtype=np.float32)
            candidate_norm = np.linalg.norm(candidate_embedding) or 1.0
            score = float(np.dot(embedding, candidate_embedding) / (norm * candidate_norm))
            if best is None or score > best["score"]:
                best = {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "role": candidate.get("role", ""),
                    "score": round(score, 4),
                }
        return best

    @staticmethod
    def embedding_to_base64(embedding: List[float]) -> str:
        return base64.b64encode(np.asarray(embedding, dtype=np.float32).tobytes()).decode("ascii")
