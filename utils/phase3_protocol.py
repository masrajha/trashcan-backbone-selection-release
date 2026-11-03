"""Load and enforce the controlled Phase 3 detector/training protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEFAULT_PHASE3_PROTOCOL = Path("Riset/configs/phase3_training_protocol.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phase3_protocol(path=None, *, project_root=None):
    root = Path(project_root or Path(__file__).resolve().parent.parent)
    resolved = Path(path or DEFAULT_PHASE3_PROTOCOL)
    if not resolved.is_absolute():
        resolved = root / resolved
    with resolved.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_phase3_protocol(protocol)
    return protocol, resolved, _sha256(resolved)


def validate_phase3_protocol(protocol):
    required_top = {
        "model",
        "training",
        "screening",
        "evaluation",
        "formal_decision",
        "benchmark_protocol",
        "run_validity",
        "resource_limits",
        "freeze",
    }
    missing = required_top - set(protocol)
    if missing:
        raise ValueError(f"Phase 3 protocol missing sections: {sorted(missing)}")

    if protocol.get("status") not in {"ready_for_freeze", "frozen"}:
        raise ValueError("Protocol status must be ready_for_freeze or frozen")

    model = protocol["model"]
    training = protocol["training"]
    evaluation = protocol["evaluation"]
    decision = protocol["formal_decision"]
    benchmark = protocol["benchmark_protocol"]
    validity = protocol["run_validity"]

    if model["architecture"] != "MaskRCNN" or model["num_object_classes"] != 16:
        raise ValueError("Phase 3 requires MaskRCNN with 16 object classes")
    if model["feature_strides"] != [4, 8, 16, 32] or model["fpn_out_channels"] != 256:
        raise ValueError("Invalid frozen FPN contract")
    if model["pretraining_scope"] != "backbone_only" or model["detector_pretrained"]:
        raise ValueError("Only ImageNet backbone pretraining is allowed")
    if training["optimizer"] != "SGD" or training["scheduler"] != "StepLR":
        raise ValueError("Frozen optimizer/scheduler contract changed")
    if training["early_stopping"]:
        raise ValueError("Early stopping is disabled for equal training budgets")
    if evaluation["split"] != "val_internal" or evaluation["official_test_access"]:
        raise ValueError("Evaluation must use val_internal while official test remains locked")

    seeds = training["formal_seeds"]
    if len(seeds) != len(set(seeds)) or len(seeds) < 3:
        raise ValueError("At least three distinct formal seeds are required")

    expected_families = {
        "cnn": ["resnet50", "efficientnet_b0", "convnext_tiny"],
        "transformer": ["swin_t", "pvtv2_b1"],
    }
    if decision["selection_scope"] != "one_winner_per_family":
        raise ValueError("Formal selection must choose one winner per architecture family")
    if decision["eligible_set"] != "pareto_frontier_per_family":
        raise ValueError("Formal winners must be selected from each family Pareto frontier")
    if decision["families"] != expected_families:
        raise ValueError("Formal candidate families differ from the registered Phase 3 set")
    if decision["reserve_candidates"] != ["swin_v2_t"]:
        raise ValueError("Swin-V2-T must remain the sole reserve candidate")
    if decision["primary_metric"] != evaluation["primary_metric"]:
        raise ValueError("Formal primary metric must match the evaluation contract")
    aggregation = decision["aggregation"]
    if aggregation["formal_seed_count"] != len(seeds) or not aggregation["paired_by_seed"]:
        raise ValueError("Formal aggregation must use every registered seed and pair by seed")
    if aggregation["inferential_claims_permitted"]:
        raise ValueError("Three-seed Phase 3 results cannot be used for inferential claims")
    tolerance = decision["ap_equivalence_tolerance"]
    if not isinstance(tolerance, (int, float)) or not 0 < tolerance <= 0.05:
        raise ValueError("AP equivalence tolerance must be in (0, 0.05]")
    expected_tie_break = [
        "mean_AP_trash_plastic",
        "lower_AP_sample_standard_deviation",
        "lower_critical_false_negative_rate",
        "higher_AP75_trash_plastic",
        "lower_peak_reserved_vram_mb",
        "lower_median_end_to_end_latency_ms",
        "lower_parameter_count",
        "simpler_implementation",
    ]
    if decision["tie_break_order"] != expected_tie_break:
        raise ValueError("Formal tie-break order changed")
    if decision["pareto"] != {
        "separate_by_family": True,
        "maximize": ["mean_AP_trash_plastic"],
        "minimize": [
            "peak_reserved_vram_mb",
            "median_end_to_end_latency_ms",
        ],
    }:
        raise ValueError("Formal Pareto objectives changed")
    if decision["composite_score_allowed"] or decision["pilot_accuracy_ranking_allowed"]:
        raise ValueError("Composite scoring and pilot accuracy ranking are prohibited")

    critical_fn = decision["critical_false_negative"]
    if (
        critical_fn["category"] != "trash_plastic"
        or critical_fn["category_id"] != 14
        or critical_fn["score_threshold"] != 0.05
        or critical_fn["iou_threshold"] != 0.5
        or critical_fn["area_scope"] != ["medium", "large"]
        or critical_fn["area_definition"] != "COCO_area_ranges_at_evaluation_scale"
    ):
        raise ValueError("Critical false-negative definition changed")

    if benchmark["device"] != protocol["resource_limits"]["device"]:
        raise ValueError("Benchmark device must match the frozen resource device")
    if (
        benchmark["input_height"] != training["input_height"]
        or benchmark["input_width"] != training["input_width"]
        or benchmark["precision"] != training["precision"]
        or benchmark["allow_tf32"] != training["allow_tf32"]
    ):
        raise ValueError("Benchmark input/precision must match the training protocol")
    inference = benchmark["inference"]
    if inference["split"] != evaluation["split"] or inference["batch_size"] != 1:
        raise ValueError("Inference benchmark must use val_internal at batch size 1")
    if inference["sample_scope"] != "one_complete_split_pass_per_repetition":
        raise ValueError("Inference benchmark must measure the complete validation split")
    if inference["warmup_iterations"] < 10 or inference["repetitions"] < 3:
        raise ValueError("Benchmark requires >=10 warmups and >=3 repetitions")
    if not inference["cuda_synchronize"] or not inference["quality_gate_required"]:
        raise ValueError("Synchronized timing and a quality gate are mandatory")
    required_statistics = {"mean", "median", "p90", "p95", "p99", "throughput_fps"}
    if not required_statistics.issubset(inference["statistics"]):
        raise ValueError("Benchmark statistics are incomplete")
    if (
        inference["ap_score_threshold"] != evaluation["score_threshold"]
        or inference["max_detections_per_image"] != model["box_detections_per_img"]
    ):
        raise ValueError("Benchmark detection settings differ from evaluation/model settings")
    if (
        benchmark["training"]["batch_size"] != training["batch_size"]
        or benchmark["training"]["gradient_accumulation_steps"]
        != training["gradient_accumulation_steps"]
    ):
        raise ValueError("Training benchmark batch contract changed")
    required_metadata = {
        "run_id",
        "backbone",
        "checkpoint_sha256",
        "manifest_sha256",
        "protocol_sha256",
        "source_git_commit",
        "git_worktree_clean",
        "python_version",
        "pytorch_version",
        "torchvision_version",
        "cuda_runtime_version",
        "cudnn_version",
        "gpu_name",
        "gpu_driver_version",
        "power_condition",
        "input_shape",
        "precision",
        "batch_size",
        "warmup_iterations",
        "measured_image_count",
        "repetition",
        "model_order",
        "started_at_utc",
        "finished_at_utc",
    }
    if not required_metadata.issubset(benchmark["required_metadata"]):
        raise ValueError("Benchmark metadata contract is incomplete")

    required_validity = {
        "complete_epoch_budget",
        "finite_loss_and_metrics",
        "matching_manifest_and_protocol_hashes",
        "clean_source_commit",
        "official_test_untouched",
    }
    if not required_validity.issubset(validity["required"]):
        raise ValueError("Run validity contract is incomplete")
    if not validity["report_all_failures"] or not validity["require_clean_git_worktree"]:
        raise ValueError("All failures and source cleanliness must be auditable")

    if protocol["status"] == "frozen":
        freeze = protocol["freeze"]
        missing_freeze = [
            key
            for key in ("frozen_at_utc", "git_commit", "protocol_sha256", "reviewer")
            if not freeze.get(key)
        ]
        if missing_freeze:
            raise ValueError(f"Frozen protocol missing metadata: {missing_freeze}")
    return True


def apply_phase3_protocol(config, protocol, *, stage="formal", seed=None):
    training = protocol["training"]
    stages = {
        "formal": {
            "data_fraction": training["formal_data_fraction"],
            "epochs": training["formal_epochs"],
            "seed": training["formal_seeds"][0],
        },
        **protocol["screening"],
    }
    if stage not in stages:
        raise ValueError(f"Unknown Phase 3 stage {stage!r}; choose one of {sorted(stages)}")
    selected = stages[stage]
    selected_seed = selected["seed"]
    if seed is not None:
        seed = int(seed)
        if stage == "formal" and seed not in training["formal_seeds"]:
            raise ValueError(
                f"Formal seed {seed} is not registered; choose {training['formal_seeds']}"
            )
        if stage != "formal" and seed != selected_seed:
            raise ValueError(f"Stage {stage} is frozen to seed {selected_seed}")
        selected_seed = seed
    values = {
        "DATA_FRACTION": selected["data_fraction"],
        "NUM_EPOCHS": selected["epochs"],
        "RANDOM_SEED": selected_seed,
        "INPUT_HEIGHT": training["input_height"],
        "INPUT_WIDTH": training["input_width"],
        "MIN_SIZE": training["min_size"],
        "MAX_SIZE": training["max_size"],
        "LEARNING_RATE": training["learning_rate"],
        "MOMENTUM": training["momentum"],
        "WEIGHT_DECAY": training["weight_decay"],
        "STEP_SIZE": training["step_size"],
        "GAMMA": training["gamma"],
        "VALIDATE_EVERY": training["validate_every"],
        "USE_AMP": training["precision"] == "amp_fp16",
        "ALLOW_TF32": training["allow_tf32"],
        "DETERMINISTIC": training["deterministic"],
        "NUM_WORKERS": training["num_workers"],
        "PIN_MEMORY": training["pin_memory"],
        "PERSISTENT_WORKERS": training["persistent_workers"],
        "GRADIENT_ACCUMULATION_STEPS": training["gradient_accumulation_steps"],
        "BACKBONE_PRETRAINING_SCOPE": protocol["model"]["pretraining_scope"],
        "SKIP_VALIDATION": False,
    }
    model_fields = {
        "FPN_OUT_CHANNELS": "fpn_out_channels",
        "ANCHOR_SIZES": "anchor_sizes",
        "ASPECT_RATIOS": "aspect_ratios",
        "BOX_ROI_OUTPUT_SIZE": "box_roi_output_size",
        "BOX_ROI_SAMPLING_RATIO": "box_roi_sampling_ratio",
        "MASK_ROI_OUTPUT_SIZE": "mask_roi_output_size",
        "MASK_ROI_SAMPLING_RATIO": "mask_roi_sampling_ratio",
        "RPN_FG_IOU_THRESH": "rpn_fg_iou_thresh",
        "RPN_BG_IOU_THRESH": "rpn_bg_iou_thresh",
        "RPN_NMS_THRESH": "rpn_nms_thresh",
        "RPN_BATCH_SIZE_PER_IMAGE": "rpn_batch_size_per_image",
        "RPN_POSITIVE_FRACTION": "rpn_positive_fraction",
        "RPN_PRE_NMS_TOP_N_TRAIN": "rpn_pre_nms_top_n_train",
        "RPN_PRE_NMS_TOP_N_TEST": "rpn_pre_nms_top_n_test",
        "RPN_POST_NMS_TOP_N_TRAIN": "rpn_post_nms_top_n_train",
        "RPN_POST_NMS_TOP_N_TEST": "rpn_post_nms_top_n_test",
        "BOX_SCORE_THRESH": "box_score_thresh",
        "BOX_NMS_THRESH": "box_nms_thresh",
        "BOX_DETECTIONS_PER_IMG": "box_detections_per_img",
        "BOX_FG_IOU_THRESH": "box_fg_iou_thresh",
        "BOX_BG_IOU_THRESH": "box_bg_iou_thresh",
        "BOX_BATCH_SIZE_PER_IMAGE": "box_batch_size_per_image",
        "BOX_POSITIVE_FRACTION": "box_positive_fraction",
    }
    values.update(
        {config_name: protocol["model"][field] for config_name, field in model_fields.items()}
    )
    for name, value in values.items():
        if name == "BATCH_SIZE":
            continue
        setattr(config, name, value)
    config._batch_size_override = int(training["batch_size"])
    config.PHASE3_STAGE = stage
    return config
