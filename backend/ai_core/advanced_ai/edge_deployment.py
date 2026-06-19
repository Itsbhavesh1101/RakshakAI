from __future__ import annotations

from typing import Any, Dict

from config import system_config


class EdgeDeploymentAdvisor:
    """Reports edge AI deployment profile and recommended runtime settings."""

    def status(self) -> Dict[str, Any]:
        profile = system_config.edge_profile
        recommendations = {
            "nvidia_jetson": {
                "runtime": "TensorRT / ONNX Runtime GPU",
                "model_imgsz": min(system_config.model_imgsz, 640),
                "inference_fps": min(system_config.inference_fps, 8.0),
                "notes": [
                    "Export YOLO weights to TensorRT engine for best latency.",
                    "Use half precision where supported.",
                    "Pin camera capture and inference to separate threads.",
                ],
            },
            "raspberry_pi_coral": {
                "runtime": "TensorFlow Lite Edge TPU",
                "model_imgsz": min(system_config.model_imgsz, 416),
                "inference_fps": min(system_config.inference_fps, 4.0),
                "notes": [
                    "Use Edge TPU compatible TFLite models.",
                    "Keep person detection lightweight and push heavy reasoning to server.",
                    "Prefer lower JPEG quality for network-constrained links.",
                ],
            },
            "cpu": {
                "runtime": "OpenCV / PyTorch CPU",
                "model_imgsz": min(system_config.model_imgsz, 416),
                "inference_fps": min(system_config.inference_fps, 3.0),
                "notes": [
                    "Use small detector weights.",
                    "Keep advanced AI features optional.",
                    "Lower inference FPS when stream latency rises.",
                ],
            },
        }
        selected = recommendations.get(profile, recommendations["cpu"])
        return {
            "enabled": system_config.edge_ai_enabled,
            "profile": profile,
            "device_hint": system_config.edge_device_hint,
            "recommended_runtime": selected["runtime"],
            "recommended_model_imgsz": selected["model_imgsz"],
            "recommended_inference_fps": selected["inference_fps"],
            "notes": selected["notes"],
        }
