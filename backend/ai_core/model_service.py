from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from config import MODELS_DIR, system_config
from ai_core.detectors.utils import clamp_box, normalize_label


DEFAULT_MODEL_FILES = {
    "weapon": "weapon.pt",
    "fire": "fire.pt",
    "fall": "fall_model.pt",
    "yolo11n": "yolo11n.pt",
}


class YoloModelService:
    """Shared model loader and inference adapter for specialized detectors."""

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        model_files: Optional[Dict[str, str]] = None,
    ):
        self.models_dir = Path(models_dir)
        self.model_files = (model_files or getattr(system_config, "model_files", None) or DEFAULT_MODEL_FILES).copy()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half_precision = self.device == "cuda"
        self.models: Dict[str, Optional[YOLO]] = {
            name: None for name in self.model_files
        }

    def load_model(self, name: str) -> YOLO:
        if name not in self.model_files:
            raise KeyError(f"Unknown YOLO model key: {name}")

        if self.models.get(name) is None:
            model_path = self.models_dir / self.model_files[name]
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model weights not found: {model_path}")

            print(f"Loading YOLO weights for '{name}' model from '{model_path}'...")
            model = YOLO(str(model_path))
            try:
                model.fuse()
            except Exception as exc:
                print(f"Failed to fuse YOLO conv+bn layers for {name}: {exc}")
            try:
                model.to(self.device)
            except Exception as exc:
                print(f"Could not pre-load '{name}' on {self.device}; YOLO will choose fallback: {exc}")
            self.models[name] = model

        loaded_model = self.models[name]
        if loaded_model is None:
            raise RuntimeError(f"YOLO model '{name}' could not be loaded.")
        return loaded_model

    def predict(
        self,
        name: str,
        frame: np.ndarray,
        conf_thresh: float,
        imgsz: Optional[int] = None,
    ) -> Tuple[YOLO, Any]:
        model = self.load_model(name)
        with torch.inference_mode():
            results = model(
                frame,
                conf=conf_thresh,
                device=self.device,
                iou=system_config.detection_iou,
                imgsz=imgsz or system_config.model_imgsz,
                verbose=False,
                half=self.use_half_precision,
                augment=system_config.augment_inference,
            )
        return model, results

    def track(
        self,
        name: str,
        frame: np.ndarray,
        conf_thresh: float,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        tracker: Optional[str] = None,
    ) -> Tuple[YOLO, Any]:
        model = self.load_model(name)
        with torch.inference_mode():
            results = model.track(
                source=frame,
                persist=True,
                conf=conf_thresh,
                device=self.device,
                iou=system_config.detection_iou,
                imgsz=imgsz or system_config.model_imgsz,
                verbose=False,
                half=self.use_half_precision,
                augment=system_config.augment_inference,
                classes=classes,
                tracker=tracker or system_config.tracker_config,
            )
        return model, results

    def extract_detections(
        self,
        model: YOLO,
        results: Any,
        frame_width: int,
        frame_height: int,
        conf_thresh: float,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        for result in results:
            for raw_box in result.boxes:
                cls = int(raw_box.cls[0])
                conf = float(raw_box.conf[0].item())
                if conf < conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, raw_box.xyxy[0])
                box = clamp_box(x1, y1, x2, y2, frame_width, frame_height)
                if box is None:
                    continue

                detections.append({
                    "box": box,
                    "confidence": conf,
                    "label": normalize_label(str(model.names.get(cls, cls))),
                })
        return detections
