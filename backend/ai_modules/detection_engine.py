from ai_core.pipeline import SurveillanceDetectionPipeline


class DetectionEngine(SurveillanceDetectionPipeline):
    """Backward-compatible facade for the Phase 1 AI core pipeline."""


detection_engine = DetectionEngine()
