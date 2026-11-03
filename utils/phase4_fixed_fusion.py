"""Load the recorded Phase 4 fixed-fusion transition and runtime contract."""

import hashlib
import json
from pathlib import Path

from utils.phase3_protocol import apply_phase3_protocol, load_phase3_protocol


DEFAULT_PROTOCOL = Path("Riset/configs/phase4_fixed_fusion_protocol.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_phase4_fixed_fusion(config, *, stage: str, seed=None, path=None):
    root = Path(__file__).resolve().parent.parent
    source = Path(path or DEFAULT_PROTOCOL)
    if not source.is_absolute():
        source = root / source
    protocol = json.loads(source.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_phase4_transition_and_fixed_fusion_baseline":
        raise ValueError("Phase 4 fixed-fusion protocol has an unexpected status")
    if protocol["feature_contract"]["fusion"] != "0.5 * cnn + 0.5 * transformer":
        raise ValueError("Fixed-fusion weights changed")
    for key in ("phase3_training_protocol", "split_manifest"):
        expected = protocol[key + "_sha256"]
        if _sha256(root / protocol[key]) != expected:
            raise ValueError(f"Frozen input hash changed: {key}")
    formal_freeze = root / "Riset/output/research_phase3/phase3_formal_protocol_freeze.json"
    if _sha256(formal_freeze) != protocol["phase3_formal_freeze_sha256"]:
        raise ValueError("Phase 3 formal freeze hash changed")
    phase3, _, phase3_hash = load_phase3_protocol(root / protocol["phase3_training_protocol"])
    if phase3_hash != protocol["phase3_training_protocol_sha256"]:
        raise ValueError("Phase 3 training protocol hash changed")
    if stage not in ("smoke", "full"):
        raise ValueError("Phase 4 fixed-fusion stage must be smoke or full")
    config = apply_phase3_protocol(
        config, phase3, stage="formal" if stage == "full" else "smoke", seed=seed
    )
    config.PHASE3_STAGE = None
    config.PHASE4_STAGE = stage
    config.PHASE4_FIXED_FUSION = True
    config.PHASE4_FIXED_FUSION_PROTOCOL = protocol
    config.PHASE4_PROTOCOL_SHA256 = _sha256(source)
    config.PHASE3_TRAINING_PROTOCOL_PATH = root / protocol["phase3_training_protocol"]
    config.PHASE3_TRAINING_PROTOCOL_SHA256 = phase3_hash
    config.MODEL_NAME = "maskrcnn_fixed_fusion_efficientnet_b0_pvtv2_b1"
    config.BACKBONE = "efficientnet_b0+pvtv2_b1"
    config.BACKBONE_PROVENANCE = {role: record["name"] for role, record in protocol["branches"].items()}
    return config
