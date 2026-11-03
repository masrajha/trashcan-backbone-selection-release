"""Validate and freeze the formal Phase 3 decision/benchmark protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.phase3_protocol import load_phase3_protocol  # noqa: E402


DEFAULT_INPUTS = {
    "protocol": Path("Riset/configs/phase3_training_protocol.json"),
    "backbone_registry": Path("models/backbone_registry.py"),
    "backbone_contract_audit": Path(
        "Riset/output/research_phase3/backbone_contract_audit.json"
    ),
    "pretrained_weight_audit": Path(
        "Riset/output/research_phase3/pretrained_weight_audit.json"
    ),
    "activation_freeze": Path(
        "Riset/output/research_phase3/backbone_activation_freeze.json"
    ),
    "selection_protocol_document": Path("PHASE3_BACKBONE_SELECTION_PROTOCOL.md"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Riset/output/research_phase3/phase3_formal_protocol_freeze.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_INPUTS["protocol"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol, protocol_path, protocol_digest = load_phase3_protocol(
        args.protocol, project_root=PROJECT_ROOT
    )
    input_paths = dict(DEFAULT_INPUTS)
    input_paths["protocol"] = args.protocol

    missing = [str(path) for path in map(resolve, input_paths.values()) if not path.is_file()]
    if missing:
        raise SystemExit(f"Required freeze inputs are missing: {missing}")

    activation_path = resolve(input_paths["activation_freeze"])
    activation = json.loads(activation_path.read_text(encoding="utf-8-sig"))
    if activation.get("status") != "frozen":
        raise SystemExit("Backbone activation decision is not frozen")

    source_commit = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain")
    clean = not bool(porcelain)
    input_hashes = {
        name: {
            "path": path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(path),
        }
        for name, raw_path in input_paths.items()
        for path in [resolve(raw_path)]
    }

    validation = {
        "protocol_valid": True,
        "protocol_status": protocol["status"],
        "protocol_sha256": protocol_digest,
        "source_git_commit": source_commit,
        "git_worktree_clean": clean,
        "activation_freeze_status": activation["status"],
        "official_test_access": protocol["evaluation"]["official_test_access"],
        "input_hashes": input_hashes,
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return 0

    if not args.reviewer or not args.reviewer.strip():
        raise SystemExit("--reviewer is required for final freeze")
    if not clean:
        raise SystemExit(
            "Refusing to freeze a dirty worktree. Commit the protocol implementation, "
            "review the diff, then rerun."
        )
    if protocol["status"] != "ready_for_freeze":
        raise SystemExit("Protocol must have status=ready_for_freeze")
    if protocol["evaluation"]["official_test_access"]:
        raise SystemExit("Official test must remain locked")

    output = resolve(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"Freeze record already exists: {output}; use --force to replace it")

    record = {
        "schema_version": 1,
        "status": "frozen",
        "scope": "phase3_formal_backbone_selection_protocol",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": args.reviewer.strip(),
        "source_git_commit": source_commit,
        "git_worktree_clean": True,
        "rules": {
            "formal_decision": protocol["formal_decision"],
            "benchmark_protocol": protocol["benchmark_protocol"],
            "run_validity": protocol["run_validity"],
        },
        "inputs": input_hashes,
        "activation_freeze_sha256": sha256(activation_path),
        "activation_freeze_record_sha256": activation.get("freeze_record_sha256"),
        "official_test_access": False,
        "amendment_policy": (
            "Any post-freeze change requires a new versioned protocol, explicit rationale, "
            "reviewer, source commit, and SHA-256 record before affected formal runs."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")

    print(json.dumps({
        "freeze_record": output.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": digest,
        "sha256_sidecar": sidecar.relative_to(PROJECT_ROOT).as_posix(),
        "source_git_commit": source_commit,
        "reviewer": record["reviewer"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
