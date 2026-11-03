#!/usr/bin/env python3
"""Formal, protocol-driven inference benchmark for Phase 3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import get_config
from data.dataset import get_dataloaders
from evaluation.detection_evaluator import SUMMARY_KEYS, xyxy_to_xywh
from inference.inference import TrashCANInference
from models.backbone_registry import backbone_provenance
from utils.experiment import sha256_file
from utils.phase3_protocol import apply_phase3_protocol, load_phase3_protocol
from utils.reproducibility import seed_everything


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def percentile_summary(milliseconds):
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Timing benchmark kosong atau mengandung nilai non-finite")
    mean = float(np.mean(values))
    return {
        "count": int(values.size),
        "mean_ms": mean,
        "median_ms": float(np.median(values)),
        "sample_standard_deviation_ms": (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        ),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput_fps": float(1000.0 / mean),
    }


def counterbalanced_order(rows, repetition):
    """Rotate the model order between repetitions to reduce order bias."""
    ordered = sorted(rows, key=lambda row: (row["job"], row["run_id"]))
    shift = (repetition - 1) % len(ordered)
    return ordered[shift:] + ordered[:shift]


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata():
    commit = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--porcelain"])
    return {"source_git_commit": commit, "git_worktree_clean": status == ""}


def gpu_metadata(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    driver = command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(index)]
    )
    return {
        "gpu_name": props.name,
        "gpu_index": index,
        "gpu_total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_driver_version": driver,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def load_registry(path: Path, jobs=None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["runs"]
    if jobs:
        requested = set(jobs)
        rows = [row for row in rows if int(row["job"]) in requested]
        missing = requested - {int(row["job"]) for row in rows}
        if missing:
            raise ValueError(f"Job tidak terdaftar: {sorted(missing)}")
    return rows


def parse_job_spec(value):
    if not value:
        return None
    jobs = set()
    for part in value.split(","):
        bounds = part.strip().split("-", 1)
        if len(bounds) == 1:
            jobs.add(int(bounds[0]))
        else:
            jobs.update(range(int(bounds[0]), int(bounds[1]) + 1))
    return sorted(jobs)


def prepare_config(row, protocol, protocol_path, protocol_hash, manifest_path):
    config = get_config("rtx3060")
    config.BACKBONE = row["backbone"]
    config.BACKBONE_PROVENANCE = backbone_provenance(row["backbone"])
    config.PRETRAINED = False
    config.DATASET_VERSION = "material"
    config = apply_phase3_protocol(config, protocol, stage="formal", seed=row["seed"])
    config.PHASE3_TRAINING_PROTOCOL = protocol
    config.PHASE3_TRAINING_PROTOCOL_PATH = protocol_path
    config.PHASE3_TRAINING_PROTOCOL_SHA256 = protocol_hash

    annotation = Path(config.TRAIN_JSON)
    if not annotation.is_absolute():
        annotation = PROJECT_ROOT / annotation
    categories = json.loads(annotation.read_text(encoding="utf-8"))["categories"]
    config.NUM_CLASSES = len(categories)
    config.CATEGORY_MAP = {int(item["id"]): item["name"] for item in categories}
    matches = [key for key, name in config.CATEGORY_MAP.items() if name == "trash_plastic"]
    if len(matches) != 1:
        raise ValueError("Kelas target trash_plastic tidak unik")
    config.TARGET_CLASS = "trash_plastic"
    config.TARGET_CATEGORY_ID = matches[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config.SPLIT_MANIFEST_PATH = manifest_path
    config.SPLIT_MANIFEST = manifest
    config.SPLIT_MANIFEST_SHA256 = manifest.get("manifest_sha256")
    if not config.SPLIT_MANIFEST_SHA256:
        config.SPLIT_MANIFEST_SHA256 = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    config.EVAL_ANNOTATION_JSON = annotation
    config.EVAL_SCORE_THRESHOLD = 0.0
    return config


def prediction_rows(outputs, targets, score_threshold=0.0):
    rows = []
    image_ids = []
    for output, target in zip(outputs, targets):
        image_id = int(target["image_id"].item())
        image_ids.append(image_id)
        boxes = output["boxes"].detach().cpu().tolist()
        labels = output["labels"].detach().cpu().tolist()
        scores = output["scores"].detach().cpu().tolist()
        for box, label, score in zip(boxes, labels, scores):
            if float(score) >= score_threshold:
                rows.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                })
    return rows, image_ids


def evaluate_quality(predictions, image_ids, annotation_file, category_map, target_class):
    coco_gt = COCO(str(annotation_file))
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
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(set(image_ids))
    evaluator.params.catIds = sorted(category_map)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    overall = {key: float(value) for key, value in zip(SUMMARY_KEYS, evaluator.stats)}
    precision = evaluator.eval["precision"]
    category_ids = evaluator.params.catIds
    target_id = next(key for key, value in category_map.items() if value == target_class)
    category_index = category_ids.index(target_id)
    target_values = precision[:, :, category_index, 0, -1]
    valid = target_values[target_values > -1]
    target_ap = float(np.mean(valid)) if valid.size else None
    return {"overall": overall, "target_class": {"class_name": target_class, "AP": target_ap}}


def warm_up(model, val_loader, device, iterations, amp_enabled):
    iterator = iter(val_loader)
    with torch.inference_mode():
        for _ in range(iterations):
            try:
                images, _ = next(iterator)
            except StopIteration:
                iterator = iter(val_loader)
                images, _ = next(iterator)
            images = [image.to(device, non_blocking=True) for image in images]
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                model.predict(images)
    torch.cuda.synchronize(device)


def measure_complete_pass(model, val_loader, device, amp_enabled, collect_predictions):
    model_only_ms = []
    end_to_end_ms = []
    predictions = []
    image_ids = []
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for cpu_images, targets in val_loader:
            gpu_images = [image.to(device, non_blocking=True) for image in cpu_images]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                model.predict(gpu_images)
            end_event.record()
            torch.cuda.synchronize(device)
            model_only_ms.append(float(start_event.elapsed_time(end_event)))

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            gpu_images = [image.to(device, non_blocking=True) for image in cpu_images]
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                outputs = model.predict(gpu_images)
            cpu_outputs = [
                {key: value.detach().cpu() for key, value in output.items()}
                for output in outputs
            ]
            torch.cuda.synchronize(device)
            end_to_end_ms.append((time.perf_counter() - started) * 1000.0)
            if collect_predictions:
                rows, ids = prediction_rows(cpu_outputs, targets, score_threshold=0.0)
                predictions.extend(rows)
                image_ids.extend(ids)

    return {
        "model_only": percentile_summary(model_only_ms),
        "end_to_end": percentile_summary(end_to_end_ms),
        "memory": {
            "peak_allocated_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
            "peak_reserved_vram_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        },
        "predictions": predictions,
        "image_ids": image_ids,
    }


def validate_checkpoint(row, checkpoint, resolved_config, protocol_hash, manifest_hash):
    if not checkpoint.is_file() or not resolved_config.is_file():
        raise FileNotFoundError(f"Artefak run tidak lengkap: {row['run_id']}")
    snapshot = json.loads(resolved_config.read_text(encoding="utf-8"))
    preflight = snapshot.get("phase3_formal_preflight", {})
    checks = {
        "backbone": snapshot.get("backbone") == row["backbone"],
        "seed": int(row["seed"]) in [int(value) for value in preflight.get("requested_seeds", [])],
        "protocol_sha256": snapshot.get("phase3_training_protocol_sha256") == protocol_hash,
        "manifest_sha256": snapshot.get("split_manifest_sha256") == manifest_hash,
        "formal_stage": snapshot.get("phase3_stage") == "formal",
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Checkpoint provenance gagal {row['run_id']}: {failed}")
    return snapshot, checks


def benchmark_one(row, repetition, order_index, args, protocol_info, environment):
    protocol, protocol_path, protocol_hash, inference_protocol = protocol_info
    run_dir = PROJECT_ROOT / "Riset" / "output" / "runs" / row["run_id"]
    checkpoint = run_dir / "checkpoints" / "best_model.pth"
    resolved_config = run_dir / "resolved_config.json"
    manifest_hash = args.manifest.stem.rsplit("_", 1)[-1]
    snapshot, provenance_checks = validate_checkpoint(
        row, checkpoint, resolved_config, protocol_hash, manifest_hash
    )
    config = prepare_config(row, protocol, protocol_path, protocol_hash, args.manifest)
    seed_everything(row["seed"], deterministic=True)
    _, val_loader, _, val_dataset = get_dataloaders(config)
    if len(val_dataset) != args.expected_images:
        raise ValueError(f"Jumlah val_internal {len(val_dataset)} != {args.expected_images}")

    inference = TrashCANInference(checkpoint, args.device, config)
    amp_enabled = protocol["benchmark_protocol"]["precision"] == "amp_fp16"
    warm_up(inference.model, val_loader, args.device, inference_protocol["warmup_iterations"], amp_enabled)
    started = utc_now()
    measured = measure_complete_pass(
        inference.model, val_loader, args.device, amp_enabled, collect_predictions=repetition == 1
    )
    finished = utc_now()

    quality = None
    quality_gate = {"required": True, "passed": None}
    if repetition == 1:
        quality = evaluate_quality(
            measured.pop("predictions"), measured.pop("image_ids"),
            config.EVAL_ANNOTATION_JSON, config.CATEGORY_MAP, config.TARGET_CLASS,
        )
        target_ap = quality["target_class"]["AP"]
        delta = abs(target_ap - float(row["expected_target_ap"]))
        quality_gate.update({
            "observed_target_ap": target_ap,
            "registered_target_ap": row["expected_target_ap"],
            "absolute_difference": delta,
            "tolerance": args.quality_tolerance,
            "passed": bool(math.isfinite(target_ap) and delta <= args.quality_tolerance),
        })
        if not quality_gate["passed"]:
            raise RuntimeError(f"Quality gate gagal untuk {row['run_id']}: {quality_gate}")
    else:
        measured.pop("predictions")
        measured.pop("image_ids")
        quality_gate["passed"] = True
        quality_gate["basis"] = "passed_on_repetition_1"

    return {
        "status": "completed",
        "job": row["job"],
        "run_id": row["run_id"],
        "backbone": row["backbone"],
        "seed": row["seed"],
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": config.SPLIT_MANIFEST_SHA256,
        "protocol_sha256": protocol_hash,
        "source_git_commit": snapshot.get("phase3_formal_preflight", {}).get("head_commit"),
        "git_worktree_clean_at_training": bool(
            snapshot.get("phase3_formal_preflight", {}).get("git_worktree_clean")
        ),
        "benchmark_git": git_metadata(),
        "environment": environment,
        "power_condition": protocol["benchmark_protocol"]["power_condition"],
        "input_shape": [1, 3, protocol["benchmark_protocol"]["input_height"], protocol["benchmark_protocol"]["input_width"]],
        "precision": protocol["benchmark_protocol"]["precision"],
        "batch_size": 1,
        "warmup_iterations": inference_protocol["warmup_iterations"],
        "measured_image_count": len(val_dataset),
        "repetition": repetition,
        "model_order": order_index,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "provenance_checks": provenance_checks,
        "quality_gate": quality_gate,
        "quality_metrics": quality,
        **measured,
    }


def write_csv(path, records):
    fields = [
        "job", "run_id", "backbone", "seed", "repetition", "model_order",
        "measured_image_count", "precision", "model_only_median_ms",
        "end_to_end_median_ms", "end_to_end_p95_ms", "throughput_fps",
        "peak_allocated_vram_mb", "peak_reserved_vram_mb", "checkpoint_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "job": row["job"], "run_id": row["run_id"], "backbone": row["backbone"],
                "seed": row["seed"], "repetition": row["repetition"], "model_order": row["model_order"],
                "measured_image_count": row["measured_image_count"], "precision": row["precision"],
                "model_only_median_ms": row["model_only"]["median_ms"],
                "end_to_end_median_ms": row["end_to_end"]["median_ms"],
                "end_to_end_p95_ms": row["end_to_end"]["p95_ms"],
                "throughput_fps": row["end_to_end"]["throughput_fps"],
                "peak_allocated_vram_mb": row["memory"]["peak_allocated_vram_mb"],
                "peak_reserved_vram_mb": row["memory"]["peak_reserved_vram_mb"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "Riset/configs/phase3_formal_inference_runs.json")
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "Riset/configs/phase3_training_protocol.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "Riset/output/research_phase1/visual_review/web_app/rebuilds/candidate_10ab2a514764/visual_review/web_app/rebuilds/candidate_eb4dae55b2c4/visual_review/web_app/frozen/internal_split_manifest_eb4dae55b2c4afb4a2167448ad81b5cd8fed7bc7399ae57a36b27851300c2914.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Riset/output/research_phase3/formal_inference_benchmark")
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--jobs", help="Contoh: 1-5,7,9")
    parser.add_argument("--expected-images", type=int, default=1211)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument("--allow-device-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    protocol, protocol_path, protocol_hash = load_phase3_protocol(args.protocol, project_root=PROJECT_ROOT)
    inference_protocol = protocol["benchmark_protocol"]["inference"]
    rows = load_registry(args.registry, parse_job_spec(args.jobs))
    excluded = [row for row in rows if not row.get("eligible", True)]
    rows = [row for row in rows if row.get("eligible", True)]
    if not rows:
        raise ValueError("Tidak ada run eligible")

    plan = []
    for repetition in range(1, inference_protocol["repetitions"] + 1):
        for order_index, row in enumerate(counterbalanced_order(rows, repetition), start=1):
            plan.append({"repetition": repetition, "model_order": order_index, **row})
    if args.dry_run:
        print(json.dumps({"protocol_sha256": protocol_hash, "excluded": excluded, "plan": plan}, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark formal mensyaratkan CUDA")
    args.device = torch.device("cuda:0")
    environment = gpu_metadata(args.device)
    expected_gpu = protocol["benchmark_protocol"]["device"]
    if environment["gpu_name"] != expected_gpu and not args.allow_device_mismatch:
        raise RuntimeError(f"GPU {environment['gpu_name']!r} != perangkat beku {expected_gpu!r}")
    torch.backends.cuda.matmul.allow_tf32 = bool(protocol["benchmark_protocol"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(protocol["benchmark_protocol"]["allow_tf32"])

    output = args.output_dir / args.tag
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "benchmark_state.json"
    state = {
        "status": "running", "tag": args.tag, "started_at_utc": utc_now(),
        "protocol_sha256": protocol_hash, "registry": str(args.registry),
        "excluded_runs": excluded, "plan": plan, "records": [], "environment": environment,
    }
    atomic_json(state_path, state)
    try:
        protocol_info = (protocol, protocol_path, protocol_hash, inference_protocol)
        for item in plan:
            print(f"=== repetition {item['repetition']}/{inference_protocol['repetitions']} | order {item['model_order']}/{len(rows)} | job {item['job']} {item['backbone']} seed {item['seed']} ===", flush=True)
            record = benchmark_one(
                item, item["repetition"], item["model_order"], args, protocol_info, environment
            )
            state["records"].append(record)
            atomic_json(output / f"job{item['job']:02d}_rep{item['repetition']}.json", record)
            atomic_json(state_path, state)
            del record
            torch.cuda.empty_cache()
        state["status"] = "completed"
        state["finished_at_utc"] = utc_now()
        atomic_json(state_path, state)
        atomic_json(output / "formal_inference_benchmark.json", state)
        write_csv(output / "formal_inference_benchmark.csv", state["records"])
    except Exception as exc:
        state["status"] = "failed"
        state["finished_at_utc"] = utc_now()
        state["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(state_path, state)
        raise
    print(f"Benchmark formal selesai: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
