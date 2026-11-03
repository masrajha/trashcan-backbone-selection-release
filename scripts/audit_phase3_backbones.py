#!/usr/bin/env python3
"""Audit Phase 3 backbone contracts and optionally load pretrained weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision

from models.backbone_registry import available_backbones, build_backbone, get_backbone_spec


def state_dict_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audit_one(name: str, *, pretrained: bool, height: int, width: int) -> dict:
    spec = get_backbone_spec(name)
    result = {"name": name, "spec": spec.__dict__, "pretrained_loaded": pretrained}
    try:
        backbone = build_backbone(name, pretrained=pretrained).eval()
        with torch.no_grad():
            outputs = backbone(torch.zeros(1, 3, height, width))
        shapes = {key: list(value.shape) for key, value in outputs.items()}
        expected = {
            "0": [height // 4, width // 4],
            "1": [height // 8, width // 8],
            "2": [height // 16, width // 16],
            "3": [height // 32, width // 32],
        }
        for key, spatial in expected.items():
            if key not in shapes or shapes[key][1] != 256 or shapes[key][-2:] != spatial:
                raise ValueError(
                    f"Invalid {name} feature {key}: got {shapes.get(key)}, "
                    f"expected [1, 256, {spatial[0]}, {spatial[1]}]"
                )
        result.update(
            status="passed",
            feature_shapes=shapes,
            parameter_count=sum(value.numel() for value in backbone.parameters()),
            pretrained_body_sha256=state_dict_sha256(backbone.body) if pretrained else None,
        )
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", action="append", choices=available_backbones())
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument(
        "--output",
        default="Riset/output/research_phase3/pretrained_weight_audit.json",
    )
    args = parser.parse_args()
    names = args.backbone or list(available_backbones())
    results = [
        audit_one(name, pretrained=args.pretrained, height=args.height, width=args.width)
        for name in names
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pretrained": args.pretrained,
        "input_size": [args.height, args.width],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
        "results": results,
        "passed": sum(row["status"] == "passed" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(1 if payload["failed"] else 0)


if __name__ == "__main__":
    main()
