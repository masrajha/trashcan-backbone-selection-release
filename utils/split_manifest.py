"""Deterministic group-aware internal split construction and validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import random
import re
import shutil
import statistics


MANIFEST_SCHEMA_VERSION = 2
GROUP_RULE_VERSION = "trashcan_video_prefix_v2"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_group_id(file_name, image_id) -> str:
    stem = Path(file_name).stem
    patterns = (
        r"(?i)^(vid[_-]?\d+)",
        r"(?i)^(video[_-]?\d+)",
        r"(?i)^(.+?)[_-](?:frame|img|image)[_-]?\d+$",
    )
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match:
            return match.group(1).lower().replace("-", "_")
    return f"image_{int(image_id)}"


def _partition_counts(image_ids, annotations_by_image, category_ids):
    counts = Counter()
    for image_id in image_ids:
        counts.update(int(row["category_id"]) for row in annotations_by_image[image_id])
    return {str(category_id): int(counts[category_id]) for category_id in category_ids}


def _candidate_score(
    val_groups,
    *,
    group_to_images,
    group_category_counts,
    total_images,
    total_category_counts,
    category_ids,
    val_fraction,
):
    val_images = sum(len(group_to_images[group_id]) for group_id in val_groups)
    image_error = abs(val_images / total_images - val_fraction)
    category_errors = []
    for category_id in category_ids:
        total = total_category_counts[category_id]
        selected = sum(group_category_counts[group_id][category_id] for group_id in val_groups)
        category_errors.append(abs(selected / total - val_fraction))
    return 2.0 * image_error + statistics.fmean(category_errors), image_error, category_errors


def build_internal_split(
    annotation_path,
    *,
    seed=42,
    val_fraction=0.20,
    search_candidates=10000,
    min_val_annotations_per_class=5,
    group_aliases=None,
):
    """Build a full, disjoint split with mandatory class representation."""
    annotation_path = Path(annotation_path).resolve()
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = sorted(data.get("categories", []), key=lambda row: int(row["id"]))
    if not images or not categories:
        raise ValueError("COCO train harus memuat images dan categories")
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction harus berada pada rentang (0, 1)")

    image_by_id = {int(row["id"]): row for row in images}
    annotations_by_image = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    group_aliases = {str(key): str(value) for key, value in (group_aliases or {}).items()}
    group_to_images = defaultdict(list)
    group_category_counts = defaultdict(Counter)
    for image_id, image in image_by_id.items():
        raw_group_id = infer_group_id(image["file_name"], image_id)
        group_id = group_aliases.get(raw_group_id, raw_group_id)
        group_to_images[group_id].append(image_id)
        group_category_counts[group_id].update(
            int(row["category_id"]) for row in annotations_by_image[image_id]
        )

    group_ids = sorted(group_to_images)
    category_ids = [int(row["id"]) for row in categories]
    total_category_counts = Counter(int(row["category_id"]) for row in annotations)
    target_images = len(images) * float(val_fraction)
    min_counts = {
        category_id: min(
            int(min_val_annotations_per_class),
            max(1, total_category_counts[category_id] - 1),
        )
        for category_id in category_ids
    }

    best = None
    rng = random.Random(int(seed))
    feasible_candidates = 0
    for _ in range(int(search_candidates)):
        shuffled = group_ids.copy()
        rng.shuffle(shuffled)
        selected = []
        selected_images = 0
        for group_id in shuffled:
            if selected_images >= target_images and selected:
                break
            selected.append(group_id)
            selected_images += len(group_to_images[group_id])

        val_counts = Counter()
        for group_id in selected:
            val_counts.update(group_category_counts[group_id])
        if any(val_counts[category_id] < min_counts[category_id] for category_id in category_ids):
            continue
        if any(
            total_category_counts[category_id] - val_counts[category_id] < 1
            for category_id in category_ids
        ):
            continue

        feasible_candidates += 1
        score, image_error, category_errors = _candidate_score(
            selected,
            group_to_images=group_to_images,
            group_category_counts=group_category_counts,
            total_images=len(images),
            total_category_counts=total_category_counts,
            category_ids=category_ids,
            val_fraction=float(val_fraction),
        )
        ranking = (score, max(category_errors), image_error, tuple(sorted(selected)))
        if best is None or ranking < best[0]:
            best = (ranking, set(selected), val_counts)

    if best is None:
        raise RuntimeError(
            "Tidak ditemukan split feasible. Tambah search_candidates atau turunkan "
            "min_val_annotations_per_class."
        )

    _, val_groups, _ = best
    train_groups = set(group_ids) - val_groups
    val_ids = sorted(image_id for group_id in val_groups for image_id in group_to_images[group_id])
    train_ids = sorted(image_id for group_id in train_groups for image_id in group_to_images[group_id])
    if set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(image_by_id):
        raise AssertionError("Split tidak eksklusif atau tidak mencakup seluruh train resmi")

    train_counts = _partition_counts(train_ids, annotations_by_image, category_ids)
    val_counts = _partition_counts(val_ids, annotations_by_image, category_ids)
    source_sha256 = sha256_file(annotation_path)
    group_rule = {
        "version": GROUP_RULE_VERSION,
        "description": "Prefix video/frame dari nama file; fallback singleton berdasarkan image_id",
    }
    if group_aliases:
        group_rule["merged_group_aliases"] = group_aliases
    manifest_payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": int(seed),
        "internal_val_fraction_target": float(val_fraction),
        "search_candidates": int(search_candidates),
        "feasible_candidates": feasible_candidates,
        "min_val_annotations_per_class": int(min_val_annotations_per_class),
        "group_rule": group_rule,
        "source_annotation": annotation_path.name,
        "source_annotation_sha256": source_sha256,
        "train_internal": {
            "image_ids": train_ids,
            "group_ids": sorted(train_groups),
            "category_counts": train_counts,
        },
        "val_internal": {
            "image_ids": val_ids,
            "group_ids": sorted(val_groups),
            "category_counts": val_counts,
        },
        "selection_score": float(best[0][0]),
        "category_mapping": {str(row["id"]): row["name"] for row in categories},
        "official_test_role": "locked_final_test",
    }
    manifest = {**manifest_payload, "manifest_sha256": canonical_sha256(manifest_payload)}

    fractions = {}
    for category_id in category_ids:
        total = total_category_counts[category_id]
        fractions[str(category_id)] = float(val_counts[str(category_id)] / total)
    group_sizes = [len(value) for value in group_to_images.values()]
    quality_report = {
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_annotation_sha256": source_sha256,
        "image_coverage": len(set(train_ids) | set(val_ids)),
        "expected_images": len(images),
        "image_overlap": len(set(train_ids) & set(val_ids)),
        "group_overlap": len(train_groups & val_groups),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "zero_count_val_categories": [key for key, value in val_counts.items() if value == 0],
        "minimum_val_count": min(val_counts.values()),
        "val_category_fractions": fractions,
        "group_size": {
            "minimum": min(group_sizes),
            "median": float(statistics.median(group_sizes)),
            "maximum": max(group_sizes),
        },
        "review_note": (
            "Pemisahan internal lolos pemeriksaan otomatis. Ancaman kemiripan frame yang tidak "
            "terwakili oleh prefix video tetap harus dilaporkan sebagai keterbatasan."
        ),
    }
    return manifest, quality_report, image_by_id


def write_split_artifacts(manifest, quality_report, image_by_id, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "internal_split_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_hash = previous.get("manifest_sha256", "unknown")
        archive_dir = output_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"internal_split_manifest_{previous_hash}.json"
        if not archive_path.exists():
            shutil.copy2(manifest_path, archive_path)

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "internal_split_quality_report.json").write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "internal_split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["partition", "image_id", "file_name", "group_id"]
        )
        writer.writeheader()
        aliases = manifest.get("group_rule", {}).get("merged_group_aliases", {})
        for partition in ("train_internal", "val_internal"):
            for image_id in manifest[partition]["image_ids"]:
                image = image_by_id[int(image_id)]
                writer.writerow(
                    {
                        "partition": partition,
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        "group_id": aliases.get(
                            infer_group_id(image["file_name"], image_id),
                            infer_group_id(image["file_name"], image_id),
                        ),
                    }
                )
    return manifest_path
