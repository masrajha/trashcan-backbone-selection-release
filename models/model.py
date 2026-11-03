"""
Model definitions for TrashCAN segmentation.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from config import config as default_config
from utils.checkpoint import load_checkpoint
from .backbone_registry import backbone_provenance, build_backbone


def create_model(num_classes=None, pretrained=True, cfg=None):
    """Create a Mask R-CNN model.

    Args:
        num_classes: Number of object classes, excluding background.
        pretrained: Use the registered ImageNet-1K backbone weights only.
        cfg: Runtime config object.
    """
    cfg = cfg or default_config
    if num_classes is None:
        num_classes = cfg.NUM_CLASSES

    num_classes = int(num_classes)
    num_classes_for_model = num_classes + 1
    print(f"Creating model with {num_classes_for_model} classes ({num_classes} objects + 1 background)")

    backbone_name = getattr(cfg, "BACKBONE", "resnet50")
    fpn_out_channels = int(getattr(cfg, "FPN_OUT_CHANNELS", 256))
    if getattr(cfg, "PHASE4_FIXED_FUSION", False):
        from pathlib import Path
        from .fixed_fusion import build_fixed_fusion_backbone

        backbone = build_fixed_fusion_backbone(
            cfg.PHASE4_FIXED_FUSION_PROTOCOL,
            project_root=Path(__file__).resolve().parent.parent,
            initialize_branches=getattr(cfg, "PHASE4_INITIALIZE_BRANCHES", True),
        )
    else:
        backbone = build_backbone(
            backbone_name, pretrained=pretrained, out_channels=fpn_out_channels
        )
    anchor_sizes = tuple(
        (int(value),)
        for value in getattr(cfg, "ANCHOR_SIZES", (32, 64, 128, 256, 512))
    )
    ratios = tuple(
        float(value) for value in getattr(cfg, "ASPECT_RATIOS", (0.5, 1.0, 2.0))
    )
    aspect_ratios = (ratios,) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=int(getattr(cfg, "BOX_ROI_OUTPUT_SIZE", 7)),
        sampling_ratio=int(getattr(cfg, "BOX_ROI_SAMPLING_RATIO", 2)),
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=int(getattr(cfg, "MASK_ROI_OUTPUT_SIZE", 14)),
        sampling_ratio=int(getattr(cfg, "MASK_ROI_SAMPLING_RATIO", 2)),
    )
    model = MaskRCNN(
        backbone,
        num_classes=num_classes_for_model,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=cfg.MIN_SIZE,
        max_size=cfg.MAX_SIZE,
        rpn_fg_iou_thresh=float(getattr(cfg, "RPN_FG_IOU_THRESH", 0.7)),
        rpn_bg_iou_thresh=float(getattr(cfg, "RPN_BG_IOU_THRESH", 0.3)),
        rpn_nms_thresh=float(getattr(cfg, "RPN_NMS_THRESH", 0.7)),
        rpn_batch_size_per_image=int(getattr(cfg, "RPN_BATCH_SIZE_PER_IMAGE", 256)),
        rpn_positive_fraction=float(getattr(cfg, "RPN_POSITIVE_FRACTION", 0.5)),
        rpn_pre_nms_top_n_train=int(getattr(cfg, "RPN_PRE_NMS_TOP_N_TRAIN", 2000)),
        rpn_pre_nms_top_n_test=int(getattr(cfg, "RPN_PRE_NMS_TOP_N_TEST", 1000)),
        rpn_post_nms_top_n_train=int(getattr(cfg, "RPN_POST_NMS_TOP_N_TRAIN", 2000)),
        rpn_post_nms_top_n_test=int(getattr(cfg, "RPN_POST_NMS_TOP_N_TEST", 1000)),
        box_score_thresh=float(getattr(cfg, "BOX_SCORE_THRESH", 0.05)),
        box_nms_thresh=float(getattr(cfg, "BOX_NMS_THRESH", 0.5)),
        box_detections_per_img=int(getattr(cfg, "BOX_DETECTIONS_PER_IMG", 100)),
        box_fg_iou_thresh=float(getattr(cfg, "BOX_FG_IOU_THRESH", 0.5)),
        box_bg_iou_thresh=float(getattr(cfg, "BOX_BG_IOU_THRESH", 0.5)),
        box_batch_size_per_image=int(getattr(cfg, "BOX_BATCH_SIZE_PER_IMAGE", 512)),
        box_positive_fraction=float(getattr(cfg, "BOX_POSITIVE_FRACTION", 0.25)),
    )
    if getattr(cfg, "PHASE4_FIXED_FUSION", False):
        model.phase4_baseline = "efficientnet_b0+pvtv2_b1_fixed_0.5"
    else:
        model.phase3_backbone = backbone_provenance(backbone_name)

    return model


class TrashCANModel(nn.Module):
    """Wrapper class for TrashCAN Mask R-CNN with training utilities."""

    def __init__(self, num_classes=None, cfg=None):
        super().__init__()
        self.cfg = cfg or default_config
        self.num_classes = int(num_classes if num_classes is not None else self.cfg.NUM_CLASSES)
        self.model = create_model(self.num_classes, self.cfg.PRETRAINED, self.cfg)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Targets must be provided during training")
        return self.model(images, targets)

    def train_step(self, images, targets):
        self.train()
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses, loss_dict

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        return self.model(images)

    def save(self, path):
        num_output_features = self.model.roi_heads.box_predictor.cls_score.out_features
        num_classes = num_output_features - 1
        print(f"Saving model with {num_classes} object classes (+1 background)")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "num_classes": num_classes,
            "config": {
                "num_classes": num_classes,
                "total_output_features": num_output_features,
                "dataset_version": getattr(self.cfg, "DATASET_VERSION", None),
                "backbone": getattr(self.cfg, "BACKBONE", "resnet50"),
                "phase4_fixed_fusion": bool(getattr(self.cfg, "PHASE4_FIXED_FUSION", False)),
                "phase4_protocol_sha256": getattr(self.cfg, "PHASE4_PROTOCOL_SHA256", None),
                "backbone_provenance": getattr(self.cfg, "BACKBONE_PROVENANCE", None),
                "phase3_stage": getattr(self.cfg, "PHASE3_STAGE", None),
                "phase3_training_protocol_sha256": getattr(
                    self.cfg, "PHASE3_TRAINING_PROTOCOL_SHA256", None
                ),
                "data_fraction": getattr(self.cfg, "DATA_FRACTION", None),
                "split_manifest_sha256": getattr(self.cfg, "SPLIT_MANIFEST_SHA256", None),
                "category_map": getattr(self.cfg, "CATEGORY_MAP", None),
                "target_class": getattr(self.cfg, "TARGET_CLASS", None),
                "target_category_id": getattr(self.cfg, "TARGET_CATEGORY_ID", None),
                "primary_metric": f"AP_{getattr(self.cfg, 'TARGET_CLASS', 'trash_plastic')}",
                "source_annotation_sha256": (
                    getattr(self.cfg, "SPLIT_MANIFEST", {}) or {}
                ).get("source_annotation_sha256"),
                "resolved_config": getattr(self.cfg, "RESOLVED_CONFIG", None),
            }
        }, path)

    def load(self, path):
        checkpoint = load_checkpoint(path, map_location="cpu")

        if isinstance(checkpoint, dict):
            checkpoint_config = checkpoint.get("config", {})
            stored_backbone = checkpoint_config.get("backbone") if isinstance(checkpoint_config, dict) else None
            current_backbone = getattr(self.cfg, "BACKBONE", "resnet50")
            resolved = checkpoint_config.get("resolved_config", {}) if isinstance(checkpoint_config, dict) else {}
            stored_fusion = bool(checkpoint_config.get(
                "phase4_fixed_fusion",
                resolved.get("phase4_fixed_fusion", False) if isinstance(resolved, dict) else False,
            )) if isinstance(checkpoint_config, dict) else False
            if stored_fusion != bool(getattr(self.cfg, "PHASE4_FIXED_FUSION", False)):
                raise ValueError("Checkpoint model variant differs from current fusion setting")
            if stored_backbone and stored_backbone != current_backbone:
                raise ValueError(
                    "Checkpoint backbone mismatch: "
                    f"checkpoint={stored_backbone!r}, runtime={current_backbone!r}. "
                    "Select the matching --backbone before loading this checkpoint."
                )

            all_keys_have_model_prefix = all(k.startswith("model.") for k in checkpoint.keys())

            if all_keys_have_model_prefix:
                print("Removing 'model.' prefix from state dict keys")
                new_state_dict = {}
                for key, value in checkpoint.items():
                    new_key = key[6:] if key.startswith("model.") else key
                    new_state_dict[new_key] = value
                self.model.load_state_dict(new_state_dict)
                print("Loaded model with removed prefix")

            elif "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded model from checkpoint")

            else:
                try:
                    self.model.load_state_dict(checkpoint)
                    print("Loaded model directly")
                except Exception as exc:
                    print(f"Direct load failed: {exc}")
                    self.model.load_state_dict(checkpoint, strict=False)
                    print("Loaded model with strict=False")
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded model")

        return self

    def train(self, mode=True):
        self.training = mode
        self.model.train(mode)
        return self

    def eval(self):
        self.training = False
        self.model.eval()
        return self
