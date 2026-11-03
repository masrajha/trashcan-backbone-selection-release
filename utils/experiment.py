"""Resolve one reproducible experiment context for the research pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import subprocess

import numpy as np
import torch
import torchvision

from utils.split_manifest import canonical_sha256, infer_group_id, sha256_file


DEFAULT_MANIFEST = Path("Riset/output/research_phase1/internal_split_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("Riset/output/runs")


def _git_commit(project_root: Path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _snapshot_path(path: Path, project_root: Path) -> str:
    """Return a relative project path, or an absolute external-data path."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _manifest_hash(manifest: dict):
    stored_hash = manifest.get("manifest_sha256")
    if stored_hash:
        core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        calculated = canonical_sha256(core)
        if calculated != stored_hash:
            raise ValueError(
                f"Hash manifest tidak cocok: tersimpan={stored_hash}, dihitung={calculated}"
            )
        return stored_hash

    return canonical_sha256(manifest)


def _source_tree_hash(project_root: Path):
    digest = hashlib.sha256()
    paths = [project_root / "requirements.txt", project_root / "RESEARCH_WORKFLOW.md"]
    for folder in ("configs", "data", "evaluation", "inference", "models", "training", "utils", "scripts"):
        paths.extend(sorted((project_root / folder).glob("*.py")))
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_state(project_root: Path):
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        return {
            "commit": _git_commit(project_root),
            "dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {"commit": _git_commit(project_root), "dirty": None}


def _validate_manifest(manifest, coco_data, train_json: Path):
    if manifest.get("official_test_role") != "locked_final_test":
        raise ValueError("Manifest tidak menyatakan official test sebagai locked_final_test")
    source_hash = manifest.get("source_annotation_sha256")
    if not source_hash:
        raise ValueError("Manifest tidak memuat source_annotation_sha256")
    actual_source_hash = sha256_file(train_json)
    if source_hash != actual_source_hash:
        raise ValueError("Hash anotasi sumber manifest berbeda dari COCO train aktif")

    expected_ids = {int(row["id"]) for row in coco_data.get("images", [])}
    train_ids = {int(value) for value in manifest["train_internal"]["image_ids"]}
    val_ids = {int(value) for value in manifest["val_internal"]["image_ids"]}
    if train_ids & val_ids:
        raise ValueError("Manifest memiliki overlap image ID")
    if train_ids | val_ids != expected_ids:
        raise ValueError("Manifest tidak mencakup train resmi tepat satu kali")

    train_groups = set(manifest["train_internal"].get("group_ids", []))
    val_groups = set(manifest["val_internal"].get("group_ids", []))
    if train_groups & val_groups:
        raise ValueError("Manifest memiliki overlap group ID")
    aliases = manifest.get("group_rule", {}).get("merged_group_aliases", {})
    derived = {
        aliases.get(
            infer_group_id(row["file_name"], row["id"]),
            infer_group_id(row["file_name"], row["id"]),
        ): int(row["id"])
        for row in coco_data.get("images", [])
    }
    if not train_groups | val_groups <= set(derived):
        raise ValueError("Manifest memuat group ID yang tidak sesuai aturan aktif")

    counts_by_partition = {}
    annotations_by_image = {}
    for annotation in coco_data.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    for partition, ids in (("train_internal", train_ids), ("val_internal", val_ids)):
        counts = {}
        for image_id in ids:
            for annotation in annotations_by_image.get(image_id, []):
                key = str(int(annotation["category_id"]))
                counts[key] = counts.get(key, 0) + 1
        expected_counts = {str(k): int(v) for k, v in manifest[partition]["category_counts"].items()}
        counts = {key: counts.get(key, 0) for key in expected_counts}
        if counts != expected_counts:
            raise ValueError(f"Statistik kategori manifest tidak cocok untuk {partition}")
        counts_by_partition[partition] = counts
    zero_val = [key for key, value in counts_by_partition["val_internal"].items() if value == 0]
    if zero_val:
        raise ValueError(f"Validasi internal tidak mewakili kategori: {zero_val}")


def resolve_experiment_config(
    config,
    *,
    manifest_path=None,
    output_root=None,
    run_id=None,
    target_class="trash_plastic",
    score_threshold=0.0,
    mode="train",
    require_manifest=True,
    model_path=None,
    resume_from=None,
):
    """Attach a validated research context to an existing hardware config."""
    project_root = Path(__file__).resolve().parent.parent
    train_json = Path(config.TRAIN_JSON).resolve()
    if not train_json.exists():
        raise FileNotFoundError(f"Anotasi train tidak ditemukan: {train_json}")

    coco_data = _read_json(train_json)
    categories = sorted(coco_data.get("categories", []), key=lambda row: int(row["id"]))
    if not categories:
        raise ValueError(f"Daftar kategori kosong: {train_json}")

    category_map = {int(row["id"]): row["name"] for row in categories}
    target_matches = [category_id for category_id, name in category_map.items() if name == target_class]
    if len(target_matches) != 1:
        raise ValueError(
            f"Kelas target {target_class!r} harus ditemukan tepat satu kali; ditemukan {target_matches}"
        )

    resolved_manifest_path = Path(manifest_path or DEFAULT_MANIFEST)
    if not resolved_manifest_path.is_absolute():
        resolved_manifest_path = project_root / resolved_manifest_path

    manifest = None
    manifest_sha256 = None
    if resolved_manifest_path.exists():
        manifest = _read_json(resolved_manifest_path)
        manifest_sha256 = _manifest_hash(manifest)
        _validate_manifest(manifest, coco_data, train_json)
        manifest_category_map = {
            int(key): value for key, value in manifest.get("category_mapping", {}).items()
        }
        if manifest_category_map and manifest_category_map != category_map:
            raise ValueError("Pemetaan kategori manifest berbeda dari anotasi COCO aktif")
    elif require_manifest:
        raise FileNotFoundError(
            f"Manifest split internal tidak ditemukan: {resolved_manifest_path}. "
            "Jalankan notebook Fase 1 terlebih dahulu."
        )

    resolved_output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    if not resolved_output_root.is_absolute():
        resolved_output_root = project_root / resolved_output_root
    resolved_run_id = run_id or f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = resolved_output_root / resolved_run_id
    resolved_resume_path = None
    if resume_from:
        resolved_resume_path = Path(resume_from)
        if not resolved_resume_path.is_absolute():
            resolved_resume_path = (project_root / resolved_resume_path).resolve()
        else:
            resolved_resume_path = resolved_resume_path.resolve()
        if not resolved_resume_path.is_file():
            raise FileNotFoundError(
                f"Training-state checkpoint tidak ditemukan: {resolved_resume_path}"
            )
    if run_dir.exists() and any(run_dir.iterdir()):
        if resolved_resume_path is None:
            raise FileExistsError(
                f"Run directory sudah berisi artefak: {run_dir}. Gunakan --run-id baru."
            )
        checkpoint_root = (run_dir / "checkpoints").resolve()
        if checkpoint_root not in resolved_resume_path.parents:
            raise ValueError(
                "Checkpoint resume harus berada di subfolder checkpoints dari run yang sama"
            )
        if not (run_dir / "resolved_config.json").is_file():
            raise FileNotFoundError(
                f"Snapshot konfigurasi run asal tidak ditemukan: {run_dir / 'resolved_config.json'}"
            )
    for child in ("checkpoints", "metrics", "predictions", "benchmarks"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    config.NUM_CLASSES = len(categories)
    config.CATEGORY_MAP = category_map
    config.TARGET_CLASS = target_class
    config.TARGET_CATEGORY_ID = target_matches[0]
    config.SPLIT_MANIFEST_PATH = resolved_manifest_path if manifest is not None else None
    config.SPLIT_MANIFEST = manifest
    config.SPLIT_MANIFEST_SHA256 = manifest_sha256
    config.EVAL_ANNOTATION_JSON = train_json
    config.EVAL_IOU_TYPE = "bbox"
    config.EVAL_SCORE_THRESHOLD = float(score_threshold)
    config.OUTPUT_ROOT = resolved_output_root
    config.RUN_ID = resolved_run_id
    config.RUN_DIR = run_dir

    git_state = _git_state(project_root)
    environment_lock = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        # Keep checkpoint metadata primitive-only so PyTorch's restricted
        # weights-only loader can read it without broad pickle allow-lists.
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(config.DEVICE),
        "gpu_name": config.GPU_NAME,
        "requirements_sha256": sha256_file(project_root / "requirements.txt"),
    }
    resolved_model_path = None
    model_sha256 = None
    if model_path:
        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = (project_root / resolved_model_path).resolve()
        if resolved_model_path.is_file():
            model_sha256 = sha256_file(resolved_model_path)
    snapshot = {
        "mode": mode,
        "run_id": resolved_run_id,
        "dataset_version": config.DATASET_VERSION,
        "train_annotation": _snapshot_path(train_json, project_root),
        "split_manifest": (
            _snapshot_path(resolved_manifest_path, project_root)
            if manifest is not None
            else None
        ),
        "split_manifest_sha256": manifest_sha256,
        "official_test_role": manifest.get("official_test_role") if manifest else None,
        "category_map": {str(key): value for key, value in category_map.items()},
        "num_object_classes": config.NUM_CLASSES,
        "target_class": target_class,
        "target_category_id": config.TARGET_CATEGORY_ID,
        "model": getattr(config, "MODEL_NAME", "maskrcnn_resnet50_fpn"),
        "backbone": getattr(config, "BACKBONE", "resnet50"),
        "backbone_pretraining_scope": getattr(
            config, "BACKBONE_PRETRAINING_SCOPE", "backbone_only"
        ),
        "backbone_provenance": getattr(config, "BACKBONE_PROVENANCE", None),
        "phase3_stage": getattr(config, "PHASE3_STAGE", None),
        "phase3_training_protocol": (
            str(getattr(config, "PHASE3_TRAINING_PROTOCOL_PATH", "")) or None
        ),
        "phase3_training_protocol_sha256": getattr(
            config, "PHASE3_TRAINING_PROTOCOL_SHA256", None
        ),
        "phase3_formal_preflight": getattr(
            config, "PHASE3_FORMAL_PREFLIGHT", None
        ),
        "phase4_stage": getattr(config, "PHASE4_STAGE", None),
        "phase4_fixed_fusion": bool(getattr(config, "PHASE4_FIXED_FUSION", False)),
        "phase4_protocol_sha256": getattr(config, "PHASE4_PROTOCOL_SHA256", None),
        "phase4_branches": (
            getattr(config, "PHASE4_FIXED_FUSION_PROTOCOL", {}) or {}
        ).get("branches"),
        "checkpoint_path": str(resolved_model_path) if resolved_model_path else None,
        "checkpoint_sha256": model_sha256,
        "input_height": config.INPUT_HEIGHT,
        "input_width": config.INPUT_WIDTH,
        "min_size": config.MIN_SIZE,
        "max_size": config.MAX_SIZE,
        "batch_size": config.BATCH_SIZE,
        "data_fraction": config.DATA_FRACTION,
        "learning_rate": config.LEARNING_RATE,
        "num_epochs": getattr(config, "NUM_EPOCHS", None),
        "momentum": getattr(config, "MOMENTUM", None),
        "weight_decay": getattr(config, "WEIGHT_DECAY", None),
        "scheduler": {
            "name": "StepLR",
            "step_size": getattr(config, "STEP_SIZE", None),
            "gamma": getattr(config, "GAMMA", None),
        },
        "early_stopping_patience": getattr(config, "PATIENCE", None),
        "validate_every": getattr(config, "VALIDATE_EVERY", 1),
        "pretrained": bool(getattr(config, "PRETRAINED", False)),
        "num_workers": getattr(config, "NUM_WORKERS", None),
        "pin_memory": bool(getattr(config, "PIN_MEMORY", False)),
        "persistent_workers": bool(getattr(config, "PERSISTENT_WORKERS", False)),
        "precision": (
            "amp_fp16"
            if bool(getattr(config, "USE_AMP", False))
            and getattr(config, "DEVICE", torch.device("cpu")).type == "cuda"
            else "fp32"
        ),
        "deterministic": bool(getattr(config, "DETERMINISTIC", True)),
        "allow_tf32": bool(getattr(config, "ALLOW_TF32", False)),
        "reproducibility": getattr(config, "REPRODUCIBILITY", None),
        "evaluation": {
            "iou_type": "bbox",
            "score_threshold": float(score_threshold),
            "primary_class": target_class,
        },
        "seed": getattr(config, "RANDOM_SEED", 42),
        "device": str(config.DEVICE),
        "gpu_name": config.GPU_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state,
        "source_tree_sha256": _source_tree_hash(project_root),
        "environment": environment_lock,
    }
    if resolved_resume_path is not None:
        original_snapshot = _read_json(run_dir / "resolved_config.json")
        compatibility_fields = (
            "run_id", "dataset_version", "split_manifest_sha256", "target_class",
            "target_category_id", "backbone", "phase3_training_protocol_sha256",
            "phase4_protocol_sha256", "phase4_fixed_fusion",
            "input_height", "input_width", "min_size", "max_size", "batch_size",
            "data_fraction", "learning_rate", "num_epochs", "momentum",
            "weight_decay", "scheduler", "validate_every", "precision",
            "deterministic", "seed",
        )
        mismatches = {
            field: {"original": original_snapshot.get(field), "runtime": snapshot.get(field)}
            for field in compatibility_fields
            if original_snapshot.get(field) != snapshot.get(field)
        }
        if mismatches:
            raise ValueError(f"Konfigurasi resume berbeda dari run asal: {mismatches}")
        config.RESOLVED_CONFIG = original_snapshot
        config.RESUME_FROM = resolved_resume_path
        (run_dir / "resume_environment_lock.json").write_text(
            json.dumps(environment_lock, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        config.RESOLVED_CONFIG = snapshot
        (run_dir / "resolved_config.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "environment_lock.json").write_text(
            json.dumps(environment_lock, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return config
