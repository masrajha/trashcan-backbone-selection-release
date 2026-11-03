"""Controlled backbone registry for Phase 3 single-branch experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    role: str
    provider: str
    implementation_id: str
    weights_id: str
    pretraining_dataset: str = "ImageNet-1K"
    status: str = "primary"


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(
        name="resnet50",
        family="cnn",
        role="baseline_control",
        provider="torchvision",
        implementation_id="torchvision.models.resnet50",
        weights_id="ResNet50_Weights.IMAGENET1K_V2",
    ),
    "efficientnet_b0": BackboneSpec(
        name="efficientnet_b0",
        family="cnn",
        role="efficient_candidate",
        provider="timm",
        implementation_id="efficientnet_b0.ra_in1k",
        weights_id="timm/efficientnet_b0.ra_in1k",
    ),
    "convnext_tiny": BackboneSpec(
        name="convnext_tiny",
        family="cnn",
        role="modern_cnn_candidate",
        provider="timm",
        implementation_id="convnext_tiny.fb_in1k",
        weights_id="timm/convnext_tiny.fb_in1k",
        status="screening_required",
    ),
    "swin_t": BackboneSpec(
        name="swin_t",
        family="transformer",
        role="window_transformer_candidate",
        provider="timm",
        implementation_id="swin_tiny_patch4_window7_224.ms_in1k",
        weights_id="timm/swin_tiny_patch4_window7_224.ms_in1k",
    ),
    "pvtv2_b1": BackboneSpec(
        name="pvtv2_b1",
        family="transformer",
        role="dense_pyramid_transformer_candidate",
        provider="timm",
        implementation_id="pvt_v2_b1.in1k",
        weights_id="timm/pvt_v2_b1.in1k",
    ),
    "swin_v2_t": BackboneSpec(
        name="swin_v2_t",
        family="transformer",
        role="reserve_window_transformer",
        provider="timm",
        implementation_id="swinv2_tiny_window8_256.ms_in1k",
        weights_id="timm/swinv2_tiny_window8_256.ms_in1k",
        status="reserve",
    ),
}


def available_backbones():
    return tuple(BACKBONE_SPECS)


def get_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(available_backbones())
        raise ValueError(f"Unknown backbone {name!r}; choose one of: {choices}") from exc


def backbone_provenance(name: str) -> dict:
    return asdict(get_backbone_spec(name))


class TimmPyramidWithFPN(nn.Module):
    """Adapt timm hierarchical feature models to Mask R-CNN's FPN contract."""

    required_reductions = (4, 8, 16, 32)

    def __init__(self, model_name: str, *, pretrained: bool, out_channels: int = 256):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "The Phase 3 backbone registry requires timm==1.0.29. "
                "Install the pinned research dependency before screening."
            ) from exc

        create_kwargs = {
            "pretrained": pretrained,
            "features_only": True,
        }
        if model_name.startswith("swin"):
            create_kwargs["strict_img_size"] = False
        self.body = timm.create_model(model_name, **create_kwargs)
        reductions = tuple(int(value) for value in self.body.feature_info.reduction())
        channels = tuple(int(value) for value in self.body.feature_info.channels())
        try:
            self.selected_indices = tuple(reductions.index(value) for value in self.required_reductions)
        except ValueError as exc:
            raise ValueError(
                f"{model_name} does not expose required strides {self.required_reductions}; "
                f"reported reductions={reductions}"
            ) from exc

        selected_channels = [channels[index] for index in self.selected_indices]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=selected_channels,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.feature_reductions = self.required_reductions
        self.feature_channels = tuple(selected_channels)
        self._selected_channels = tuple(selected_channels)

    def forward(self, images: torch.Tensor):
        raw_features = self.body(images)
        selected = OrderedDict()
        for level, (index, expected_channels) in enumerate(
            zip(self.selected_indices, self._selected_channels)
        ):
            feature = raw_features[index]
            if feature.ndim == 4 and feature.shape[1] != expected_channels:
                if feature.shape[-1] != expected_channels:
                    raise ValueError(
                        f"Unexpected channel layout for feature {level}: {tuple(feature.shape)}"
                    )
                feature = feature.permute(0, 3, 1, 2).contiguous()
            selected[str(level)] = feature
        return self.fpn(selected)


def build_backbone(name: str, *, pretrained: bool, out_channels: int = 256):
    """Build a backbone with a common five-level Mask R-CNN FPN output."""
    spec = get_backbone_spec(name)
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights,
            trainable_layers=5,
            returned_layers=[1, 2, 3, 4],
            extra_blocks=LastLevelMaxPool(),
        )
        backbone.feature_reductions = (4, 8, 16, 32)
        backbone.feature_channels = (256, 512, 1024, 2048)
    else:
        backbone = TimmPyramidWithFPN(
            spec.implementation_id,
            pretrained=pretrained,
            out_channels=out_channels,
        )
    backbone.phase3_spec = backbone_provenance(name)
    return backbone
