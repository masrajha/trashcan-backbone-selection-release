"""COCO bounding-box evaluation with per-class research artifacts."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


SUMMARY_KEYS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _mean_valid(values):
    values = np.asarray(values)
    values = values[values > -1]
    return float(np.mean(values)) if values.size else None


def _bbox_iou_xywh(left, right):
    """Return IoU for two COCO-format [x, y, width, height] boxes."""
    left_x1, left_y1, left_w, left_h = (float(value) for value in left)
    right_x1, right_y1, right_w, right_h = (float(value) for value in right)
    left_x2, left_y2 = left_x1 + max(0.0, left_w), left_y1 + max(0.0, left_h)
    right_x2, right_y2 = right_x1 + max(0.0, right_w), right_y1 + max(0.0, right_h)
    intersection_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    union = (
        max(0.0, left_w) * max(0.0, left_h)
        + max(0.0, right_w) * max(0.0, right_h)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def critical_false_negative_metrics(
    coco_gt,
    predictions,
    image_ids,
    *,
    category_id=14,
    score_threshold=0.05,
    iou_threshold=0.50,
):
    """Compute one-to-one medium/large target-class false negatives."""
    image_ids = {int(image_id) for image_id in image_ids}
    gt_annotations = [
        annotation
        for annotation in coco_gt.loadAnns(
            coco_gt.getAnnIds(imgIds=sorted(image_ids), catIds=[int(category_id)])
        )
        if not int(annotation.get("iscrowd", 0))
        and not int(annotation.get("ignore", 0))
        and float(annotation.get("area", 0.0)) >= 32.0**2
    ]

    predictions_by_image = {}
    for prediction in predictions:
        if (
            int(prediction["image_id"]) in image_ids
            and int(prediction["category_id"]) == int(category_id)
            and float(prediction["score"]) >= float(score_threshold)
        ):
            predictions_by_image.setdefault(int(prediction["image_id"]), []).append(
                prediction
            )
    for rows in predictions_by_image.values():
        rows.sort(key=lambda row: float(row["score"]), reverse=True)

    ground_truths_by_image = {}
    for annotation in gt_annotations:
        ground_truths_by_image.setdefault(int(annotation["image_id"]), []).append(
            annotation
        )

    matched_annotation_ids = set()
    for image_id, image_predictions in predictions_by_image.items():
        image_ground_truths = ground_truths_by_image.get(image_id, [])
        for prediction in image_predictions:
            candidates = [
                (annotation, _bbox_iou_xywh(prediction["bbox"], annotation["bbox"]))
                for annotation in image_ground_truths
                if int(annotation["id"]) not in matched_annotation_ids
            ]
            if not candidates:
                continue
            best_annotation, best_iou = max(candidates, key=lambda row: row[1])
            if best_iou >= float(iou_threshold):
                matched_annotation_ids.add(int(best_annotation["id"]))

    false_negatives = []
    area_counts = {
        "medium": {"num_gt": 0, "num_matched": 0},
        "large": {"num_gt": 0, "num_matched": 0},
    }
    for annotation in gt_annotations:
        area_name = "large" if float(annotation["area"]) >= 96.0**2 else "medium"
        matched = int(annotation["id"]) in matched_annotation_ids
        area_counts[area_name]["num_gt"] += 1
        area_counts[area_name]["num_matched"] += int(matched)
        if not matched:
            false_negatives.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "image_id": int(annotation["image_id"]),
                    "category_id": int(annotation["category_id"]),
                    "bbox": [float(value) for value in annotation["bbox"]],
                    "area": float(annotation["area"]),
                    "area_range": area_name,
                }
            )

    for values in area_counts.values():
        values["false_negative_count"] = values["num_gt"] - values["num_matched"]
        values["false_negative_rate"] = (
            values["false_negative_count"] / values["num_gt"]
            if values["num_gt"]
            else None
        )
    num_gt = len(gt_annotations)
    false_negative_count = len(false_negatives)
    return {
        "definition": {
            "category_id": int(category_id),
            "score_threshold": float(score_threshold),
            "iou_threshold": float(iou_threshold),
            "area_scope": ["medium", "large"],
            "area_definition": "COCO_area_ranges_at_evaluation_scale",
            "matching": "greedy_descending_score_one_to_one",
            "crowd_and_ignored_gt_excluded": True,
        },
        "num_gt": num_gt,
        "num_matched": num_gt - false_negative_count,
        "false_negative_count": false_negative_count,
        "critical_false_negative_rate": (
            false_negative_count / num_gt if num_gt else None
        ),
        "by_area": area_counts,
        "false_negatives": false_negatives,
    }


def target_class_ap(metrics):
    """Return the plastic-centric checkpoint metric with strict validation."""
    value = metrics.get("target_class", {}).get("AP")
    if value is None or not np.isfinite(value):
        raise ValueError("AP kelas target tidak tersedia atau tidak finite")
    return float(value)


class DetectionEvaluator:
    def __init__(
        self,
        model,
        dataloader,
        device,
        annotation_file,
        category_map,
        output_dir,
        target_class="trash_plastic",
        score_threshold=0.0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.annotation_file = Path(annotation_file)
        self.category_map = {int(key): value for key, value in category_map.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_class = target_class
        self.score_threshold = float(score_threshold)

    def _predict(self, images):
        if hasattr(self.model, "predict"):
            return self.model.predict(images)
        return self.model(images)

    def collect_predictions(self, max_batches=None):
        self.model.eval()
        predictions = []
        image_ids = []
        timings = []

        with torch.no_grad():
            for batch_index, (images, targets) in enumerate(self.dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = [image.to(self.device, non_blocking=True) for image in images]
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                outputs = self._predict(images)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                timings.append(time.perf_counter() - started)

                for target, output in zip(targets, outputs):
                    image_id = int(target["image_id"].item())
                    image_ids.append(image_id)
                    boxes = output["boxes"].detach().cpu().tolist()
                    labels = output["labels"].detach().cpu().tolist()
                    scores = output["scores"].detach().cpu().tolist()
                    for box, label, score in zip(boxes, labels, scores):
                        if float(score) < self.score_threshold:
                            continue
                        predictions.append(
                            {
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": xyxy_to_xywh(box),
                                "score": float(score),
                            }
                        )
        return predictions, sorted(set(image_ids)), timings

    def _per_class_metrics(self, coco_eval, coco_gt, image_ids):
        precision = coco_eval.eval["precision"]
        recall = coco_eval.eval["recall"]
        iou_thresholds = coco_eval.params.iouThrs
        area_index = 0
        max_dets_index = len(coco_eval.params.maxDets) - 1
        rows = []

        for category_index, category_id in enumerate(coco_eval.params.catIds):
            ap = _mean_valid(precision[:, :, category_index, area_index, max_dets_index])
            ap50 = _mean_valid(
                precision[np.isclose(iou_thresholds, 0.50), :, category_index, area_index, max_dets_index]
            )
            ap75 = _mean_valid(
                precision[np.isclose(iou_thresholds, 0.75), :, category_index, area_index, max_dets_index]
            )
            ar100 = _mean_valid(recall[:, category_index, area_index, max_dets_index])
            num_gt = len(coco_gt.getAnnIds(imgIds=image_ids, catIds=[category_id], iscrowd=None))
            rows.append(
                {
                    "category_id": int(category_id),
                    "class_name": self.category_map.get(int(category_id), f"category_{category_id}"),
                    "AP": ap,
                    "AP50": ap50,
                    "AP75": ap75,
                    "AR100": ar100,
                    "num_gt": num_gt,
                }
            )
        return rows

    def _write_artifacts(self, predictions, metrics, per_class):
        prediction_json = self.output_dir / "predictions_bbox.json"
        prediction_json.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

        with (self.output_dir / "predictions_bbox.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["image_id", "category_id", "x", "y", "width", "height", "score"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in predictions:
                x, y, width, height = row["bbox"]
                writer.writerow(
                    {
                        "image_id": row["image_id"],
                        "category_id": row["category_id"],
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "score": row["score"],
                    }
                )

        (self.output_dir / "detection_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (self.output_dir / "detection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
            writer.writeheader()
            for key, value in metrics["overall"].items():
                writer.writerow({"metric": key, "value": value})

        with (self.output_dir / "per_class_ap.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["category_id", "class_name", "AP", "AP50", "AP75", "AR100", "num_gt"]
            )
            writer.writeheader()
            writer.writerows(per_class)

        critical = metrics["critical_false_negatives"]
        (self.output_dir / "critical_false_negatives.json").write_text(
            json.dumps(critical, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (self.output_dir / "critical_false_negatives.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = [
                "annotation_id",
                "image_id",
                "category_id",
                "x",
                "y",
                "width",
                "height",
                "area",
                "area_range",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in critical["false_negatives"]:
                x, y, width, height = row["bbox"]
                writer.writerow(
                    {
                        "annotation_id": row["annotation_id"],
                        "image_id": row["image_id"],
                        "category_id": row["category_id"],
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "area": row["area"],
                        "area_range": row["area_range"],
                    }
                )

    def evaluate(self, max_batches=None):
        predictions, image_ids, timings = self.collect_predictions(max_batches=max_batches)
        if not image_ids:
            raise ValueError("Dataloader evaluasi tidak menghasilkan citra")

        coco_gt = COCO(str(self.annotation_file))
        if predictions:
            coco_dt = coco_gt.loadRes(predictions)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": [coco_gt.imgs[image_id] for image_id in image_ids],
                "categories": list(coco_gt.cats.values()),
                "annotations": [],
            }
            coco_dt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.params.catIds = sorted(self.category_map)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        overall = {key: float(value) for key, value in zip(SUMMARY_KEYS, coco_eval.stats)}
        per_class = self._per_class_metrics(coco_eval, coco_gt, image_ids)
        target_rows = [row for row in per_class if row["class_name"] == self.target_class]
        if len(target_rows) != 1:
            raise ValueError(f"Metrik kelas target {self.target_class!r} tidak ditemukan secara unik")
        if self.score_threshold > 0.05:
            raise ValueError(
                "Critical FN membutuhkan prediction score threshold <= 0.05"
            )
        critical_false_negatives = critical_false_negative_metrics(
            coco_gt,
            predictions,
            image_ids,
            category_id=int(target_rows[0]["category_id"]),
            score_threshold=0.05,
            iou_threshold=0.50,
        )

        metrics = {
            "iou_type": "bbox",
            "annotation_file": str(self.annotation_file),
            "score_threshold": self.score_threshold,
            "num_images": len(image_ids),
            "num_predictions": len(predictions),
            "overall": overall,
            "target_class": target_rows[0],
            "critical_false_negatives": critical_false_negatives,
            "per_class": per_class,
            "timing": {
                "mean_seconds_per_batch": float(np.mean(timings)) if timings else None,
                "p50_seconds_per_batch": float(np.percentile(timings, 50)) if timings else None,
                "p95_seconds_per_batch": float(np.percentile(timings, 95)) if timings else None,
            },
        }
        self._write_artifacts(predictions, metrics, per_class)
        return metrics
