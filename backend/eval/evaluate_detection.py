"""
Evaluate Rakshak detection modules against YOLO-format labels.

Expected dataset layout:
  dataset/
    images/
      frame_001.jpg
    labels/
      frame_001.txt

Each label row should be: <class_id> <x_center> <y_center> <width> <height>
with normalized YOLO coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai_modules.detection_engine import DetectionEngine  # noqa: E402
from config import MODELS_DIR, system_config  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_MAP = {
    "0": "weapon_detection",
    "1": "fire_detection",
    "2": "fall_detection",
    "3": "person",
}
EVAL_MODULES = ("weapon_detection", "fire_detection", "fall_detection")


@dataclass
class BoxRecord:
    image: str
    label: str
    box: List[int]
    confidence: float = 1.0
    matched: bool = False


@dataclass
class ClassMetrics:
    label: str
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    ap50: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate filtered Rakshak detector outputs against YOLO labels."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset root containing images/ and labels/ folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/eval/runs/latest"),
        help="Directory where reports and crops will be written.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=list(EVAL_MODULES),
        choices=list(EVAL_MODULES),
        help="Detection modules to evaluate.",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        help=(
            "Optional JSON mapping YOLO class IDs or names to labels, e.g. "
            '{"0":"weapon_detection","1":"fire_detection"}.'
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to labels.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=100,
        help="Maximum false positive and false negative crops to export.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing weapon.pt, fire.pt, fall_model.pt, and yolo11n.pt.",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Skip exporting false-positive and false-negative crops.",
    )
    return parser.parse_args()


def load_class_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return DEFAULT_CLASS_MAP.copy()
    with path.open("r", encoding="utf-8") as handle:
        raw_map = json.load(handle)
    return {str(key): str(value) for key, value in raw_map.items()}


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def yolo_to_xywh(parts: Sequence[str], width: int, height: int) -> Optional[List[int]]:
    try:
        x_center, y_center, box_width, box_height = map(float, parts)
    except ValueError:
        return None

    x1 = int(round((x_center - box_width / 2) * width))
    y1 = int(round((y_center - box_height / 2) * height))
    x2 = int(round((x_center + box_width / 2) * width))
    y2 = int(round((y_center + box_height / 2) * height))
    return DetectionEngine._clamp_box(x1, y1, x2, y2, width, height)


def label_path_for(image_path: Path, dataset_dir: Path) -> Path:
    relative = image_path.relative_to(dataset_dir / "images")
    return dataset_dir / "labels" / relative.with_suffix(".txt")


def read_ground_truth(
    image_path: Path,
    dataset_dir: Path,
    image_shape: Tuple[int, int, int],
    class_map: Dict[str, str],
) -> List[BoxRecord]:
    height, width = image_shape[:2]
    label_path = label_path_for(image_path, dataset_dir)
    if not label_path.exists():
        return []

    records: List[BoxRecord] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            print(f"Skipping malformed label {label_path}:{line_no}: {raw_line}")
            continue

        label = class_map.get(parts[0], parts[0])
        box = yolo_to_xywh(parts[1:], width, height)
        if box is None:
            print(f"Skipping invalid box {label_path}:{line_no}: {raw_line}")
            continue

        records.append(BoxRecord(str(image_path), label, box))
    return records


def xywh_to_xyxy(box: Sequence[int]) -> Tuple[int, int, int, int]:
    x, y, width, height = box
    return x, y, x + width, y + height


def box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    left_x1, left_y1, left_x2, left_y2 = xywh_to_xyxy(left)
    right_x1, right_y1, right_x2, right_y2 = xywh_to_xyxy(right)

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    left_area = max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1)
    right_area = max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def run_filtered_predictions(
    engine: DetectionEngine,
    image: np.ndarray,
    image_path: Path,
    modules: Iterable[str],
) -> List[BoxRecord]:
    engine.consecutive_detections = {
        key: engine.persistence_threshold for key in engine.consecutive_detections
    }
    engine.temporal_scores = {
        key: system_config.alert_trigger_scores.get(key, 1.0)
        for key in engine.temporal_scores
    }
    _, alerts, _ = engine.run_yolo_checks(
        image,
        enabled_modules=list(modules),
        confidence_thresholds=system_config.confidence_thresholds,
    )

    predictions: List[BoxRecord] = []
    for alert in alerts:
        label = str(alert.get("type", "unknown"))
        confidence = float(alert.get("confidence") or 0.0)
        for box in alert.get("bounding_boxes") or []:
            predictions.append(
                BoxRecord(str(image_path), label, [int(value) for value in box], confidence)
            )
    return predictions


def match_predictions(
    ground_truth: List[BoxRecord],
    predictions: List[BoxRecord],
    label: str,
    iou_threshold: float,
) -> Tuple[int, List[BoxRecord], List[BoxRecord]]:
    gt_for_label = [item for item in ground_truth if item.label == label]
    pred_for_label = sorted(
        [item for item in predictions if item.label == label],
        key=lambda item: item.confidence,
        reverse=True,
    )

    true_positives = 0
    false_positives: List[BoxRecord] = []
    for prediction in pred_for_label:
        best_gt: Optional[BoxRecord] = None
        best_iou = 0.0
        for gt_item in gt_for_label:
            if gt_item.matched:
                continue
            iou = box_iou(prediction.box, gt_item.box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_item

        if best_gt is not None and best_iou >= iou_threshold:
            prediction.matched = True
            best_gt.matched = True
            true_positives += 1
        else:
            false_positives.append(prediction)

    false_negatives = [item for item in gt_for_label if not item.matched]
    return true_positives, false_positives, false_negatives


def average_precision(
    ground_truth_count: int,
    predictions: List[BoxRecord],
    ground_truth: List[BoxRecord],
    label: str,
    iou_threshold: float,
) -> float:
    if ground_truth_count == 0:
        return 0.0

    gt_for_label = [
        BoxRecord(item.image, item.label, item.box, item.confidence)
        for item in ground_truth
        if item.label == label
    ]
    pred_for_label = sorted(
        [item for item in predictions if item.label == label],
        key=lambda item: item.confidence,
        reverse=True,
    )

    tp_values: List[float] = []
    fp_values: List[float] = []
    for prediction in pred_for_label:
        best_gt: Optional[BoxRecord] = None
        best_iou = 0.0
        for gt_item in gt_for_label:
            if gt_item.matched:
                continue
            iou = box_iou(prediction.box, gt_item.box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_item

        if best_gt is not None and best_iou >= iou_threshold:
            best_gt.matched = True
            tp_values.append(1.0)
            fp_values.append(0.0)
        else:
            tp_values.append(0.0)
            fp_values.append(1.0)

    if not tp_values:
        return 0.0

    tp_cumsum = np.cumsum(tp_values)
    fp_cumsum = np.cumsum(fp_values)
    recalls = tp_cumsum / ground_truth_count
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)

    precision_envelope = np.concatenate(([0.0], precisions, [0.0]))
    recall_envelope = np.concatenate(([0.0], recalls, [1.0]))
    for index in range(len(precision_envelope) - 2, -1, -1):
        precision_envelope[index] = max(precision_envelope[index], precision_envelope[index + 1])

    changing_points = np.where(recall_envelope[1:] != recall_envelope[:-1])[0]
    return float(
        np.sum(
            (recall_envelope[changing_points + 1] - recall_envelope[changing_points])
            * precision_envelope[changing_points + 1]
        )
    )


def crop_box(image: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x, y, width, height = box
    return image[y : y + height, x : x + width]


def export_crops(records: List[BoxRecord], output_dir: Path, prefix: str, max_crops: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records[:max_crops], 1):
        image = cv2.imread(record.image)
        if image is None:
            continue
        crop = crop_box(image, record.box)
        if crop.size == 0:
            continue
        source_name = Path(record.image).stem
        filename = f"{prefix}_{index:04d}_{record.label}_{source_name}.jpg"
        cv2.imwrite(str(output_dir / filename), crop)


def write_metrics(output_dir: Path, metrics: List[ClassMetrics], summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "summary": summary,
        "classes": [asdict(item) for item in metrics],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8"
    )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Dataset must contain images/ and labels/: {dataset_dir}", file=sys.stderr)
        return 2

    image_paths = iter_images(images_dir)
    if not image_paths:
        print(f"No images found in {images_dir}", file=sys.stderr)
        return 2

    class_map = load_class_map(args.class_map)
    modules = list(args.modules)
    engine = DetectionEngine(args.models_dir)

    all_ground_truth: List[BoxRecord] = []
    all_predictions: List[BoxRecord] = []
    latencies_ms: List[float] = []

    print(f"Evaluating {len(image_paths)} images with modules: {', '.join(modules)}")
    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        ground_truth = read_ground_truth(image_path, dataset_dir, image.shape, class_map)
        started = time.perf_counter()
        predictions = run_filtered_predictions(engine, image, image_path, modules)
        latencies_ms.append((time.perf_counter() - started) * 1000)

        all_ground_truth.extend(ground_truth)
        all_predictions.extend(predictions)

        if index % 25 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} images")

    metrics: List[ClassMetrics] = []
    all_false_positives: List[BoxRecord] = []
    all_false_negatives: List[BoxRecord] = []
    for label in modules:
        gt_count = sum(1 for item in all_ground_truth if item.label == label)
        pred_count = sum(1 for item in all_predictions if item.label == label)
        tp_count, false_positives, false_negatives = match_predictions(
            all_ground_truth, all_predictions, label, args.iou
        )
        all_false_positives.extend(false_positives)
        all_false_negatives.extend(false_negatives)

        precision = tp_count / pred_count if pred_count else 0.0
        recall = tp_count / gt_count if gt_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        ap50 = average_precision(gt_count, all_predictions, all_ground_truth, label, args.iou)
        metrics.append(
            ClassMetrics(
                label=label,
                ground_truth=gt_count,
                predictions=pred_count,
                true_positives=tp_count,
                false_positives=len(false_positives),
                false_negatives=len(false_negatives),
                precision=round(precision, 6),
                recall=round(recall, 6),
                f1=round(f1, 6),
                ap50=round(ap50, 6),
            )
        )

    mean_ap50 = float(np.mean([item.ap50 for item in metrics])) if metrics else 0.0
    summary = {
        "dataset": str(dataset_dir),
        "images": len(image_paths),
        "modules": modules,
        "iou_threshold": args.iou,
        "model_imgsz": system_config.model_imgsz,
        "detection_iou": system_config.detection_iou,
        "augment_inference": system_config.augment_inference,
        "weapon_roi_pass_enabled": system_config.weapon_roi_pass_enabled,
        "temporal_decay": system_config.temporal_decay,
        "alert_trigger_scores": {
            module: system_config.alert_trigger_scores.get(module) for module in modules
        },
        "instant_confidence": {
            module: system_config.instant_confidence.get(module) for module in modules
        },
        "confidence_thresholds": {
            module: system_config.confidence_thresholds.get(module) for module in modules
        },
        "mean_ap50": round(mean_ap50, 6),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3) if latencies_ms else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3)
        if latencies_ms
        else 0.0,
    }

    output_dir = args.output.resolve()
    write_metrics(output_dir, metrics, summary)
    if not args.no_crops:
        export_crops(all_false_positives, output_dir / "false_positives", "fp", args.max_crops)
        export_crops(all_false_negatives, output_dir / "false_negatives", "fn", args.max_crops)

    print("\nEvaluation complete")
    print(f"mAP@{args.iou:.2f}: {summary['mean_ap50']:.4f}")
    print(f"Mean latency: {summary['mean_latency_ms']:.1f} ms")
    print(f"Wrote reports to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
