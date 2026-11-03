"""Phase 4 fixed-weight CNN--Transformer feature-fusion control."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .backbone_registry import get_backbone_spec


class PyramidFeatures(nn.Module):
    """Return four timm feature levels without a branch-specific FPN."""

    reductions = (4, 8, 16, 32)

    def __init__(self, name: str):
        super().__init__()
        import timm

        spec = get_backbone_spec(name)
        self.body = timm.create_model(spec.implementation_id, pretrained=False, features_only=True)
        available = tuple(int(x) for x in self.body.feature_info.reduction())
        channels = tuple(int(x) for x in self.body.feature_info.channels())
        if any(level not in available for level in self.reductions):
            raise ValueError(f"{name}: expected strides {self.reductions}, got {available}")
        self.indices = tuple(available.index(level) for level in self.reductions)
        self.channels = tuple(channels[index] for index in self.indices)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        raw = self.body(x)
        result = []
        for index, channels in zip(self.indices, self.channels):
            feature = raw[index]
            if feature.ndim != 4:
                raise ValueError(f"Expected 4D feature tensor, got {tuple(feature.shape)}")
            if feature.shape[1] != channels:
                if feature.shape[-1] != channels:
                    raise ValueError(f"Unexpected channel layout: {tuple(feature.shape)}")
                feature = feature.permute(0, 3, 1, 2).contiguous()
            result.append(feature)
        return result


def load_branch_weights(branch: PyramidFeatures, path: str, expected_sha256: str) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = stored.get("model_state_dict", stored)
    prefix = "backbone.body."
    features = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not features:
        raise ValueError(f"No branch feature weights in {checkpoint}")
    branch.body.load_state_dict(features, strict=True)


class FixedFusionBackbone(nn.Module):
    """Align P2--P5, average both branches, then use one shared FPN."""

    def __init__(self, cnn: nn.Module, transformer: nn.Module, *,
                 cnn_channels: tuple[int, ...], transformer_channels: tuple[int, ...],
                 out_channels: int = 256):
        super().__init__()
        if len(cnn_channels) != 4 or len(transformer_channels) != 4:
            raise ValueError("Both branches must expose four feature levels")
        self.cnn = cnn
        self.transformer = transformer
        self.cnn_alignment = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in cnn_channels)
        self.transformer_alignment = nn.ModuleList(
            nn.Conv2d(c, out_channels, 1) for c in transformer_channels
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels] * 4,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.feature_reductions = (4, 8, 16, 32)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        cnn = self.cnn(x)
        transformer = self.transformer(x)
        if len(cnn) != 4 or len(transformer) != 4:
            raise ValueError("Expected four features from each branch")
        combined = OrderedDict()
        for level, (fc, ft, ac, at) in enumerate(zip(
            cnn, transformer, self.cnn_alignment, self.transformer_alignment
        )):
            if fc.shape[0] != ft.shape[0]:
                raise ValueError("Branch batch sizes differ")
            fc = ac(fc)
            ft = at(ft)
            if ft.shape[-2:] != fc.shape[-2:]:
                ft = F.interpolate(ft, size=fc.shape[-2:], mode="bilinear", align_corners=False)
            combined[str(level)] = 0.5 * fc + 0.5 * ft
        return self.fpn(combined)


def build_fixed_fusion_backbone(protocol: dict, *, project_root: Path,
                                initialize_branches: bool = True) -> FixedFusionBackbone:
    branches = protocol["branches"]
    if (branches["cnn"]["name"], branches["transformer"]["name"]) != (
        "efficientnet_b0", "pvtv2_b1"
    ):
        raise ValueError("Phase 4 baseline is fixed to EfficientNet-B0 and PVTv2-B1")
    cnn = PyramidFeatures("efficientnet_b0")
    transformer = PyramidFeatures("pvtv2_b1")
    if initialize_branches:
        for model, role in ((cnn, "cnn"), (transformer, "transformer")):
            record = branches[role]
            load_branch_weights(
                model, str(project_root / record["checkpoint"]), record["sha256"]
            )
    return FixedFusionBackbone(
        cnn, transformer,
        cnn_channels=cnn.channels,
        transformer_channels=transformer.channels,
        out_channels=int(protocol["feature_contract"]["channels"]),
    )
