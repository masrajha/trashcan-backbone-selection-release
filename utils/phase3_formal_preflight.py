"""Cryptographic preflight gate for Phase 3 formal experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from utils.phase3_protocol import load_phase3_protocol


DEFAULT_FREEZE_RECORD = Path(
    "Riset/output/research_phase3/phase3_formal_protocol_freeze.json"
)
_ALLOWED_POST_FREEZE_PATHS = {
    "Riset/output/research_phase3/phase3_formal_protocol_freeze.json",
    "Riset/output/research_phase3/phase3_formal_protocol_freeze.json.sha256",
}
_ALLOWED_POST_FREEZE_PREFIXES = (
    "Riset/output/research_phase3/formal_",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _git(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_inside(project_root: Path, raw_path) -> Path:
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"Freeze input berada di luar project root: {raw_path}") from error
    return resolved


def validate_phase3_formal_preflight(
    *,
    project_root=None,
    freeze_record=None,
    backbones=None,
    seeds=None,
    require_clean=True,
    source_attestation=None,
):
    """Verify frozen inputs, source provenance, and requested formal matrix."""
    root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    freeze_path = _resolve_inside(root, freeze_record or DEFAULT_FREEZE_RECORD)
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Formal freeze record tidak ditemukan: {freeze_path}")

    sidecar_path = freeze_path.with_suffix(freeze_path.suffix + ".sha256")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"SHA-256 sidecar tidak ditemukan: {sidecar_path}")
    sidecar_parts = sidecar_path.read_text(encoding="utf-8-sig").strip().split()
    if not sidecar_parts:
        raise ValueError("SHA-256 sidecar kosong")
    actual_freeze_hash = _sha256(freeze_path)
    if sidecar_parts[0].lower() != actual_freeze_hash:
        raise ValueError("Hash formal freeze record tidak cocok dengan sidecar")

    freeze = _read_json(freeze_path)
    if freeze.get("status") != "frozen":
        raise ValueError("Formal protocol belum berstatus frozen")
    if freeze.get("official_test_access") is not False:
        raise ValueError("Formal freeze tidak menjamin official test tetap terkunci")
    if freeze.get("git_worktree_clean") is not True:
        raise ValueError("Formal freeze tidak berasal dari worktree bersih")

    input_results = {}
    for name, metadata in freeze.get("inputs", {}).items():
        path = _resolve_inside(root, metadata["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Freeze input hilang: {path}")
        actual = _sha256(path)
        expected = str(metadata["sha256"]).lower()
        if actual != expected:
            raise ValueError(
                f"Hash freeze input berubah untuk {name}: expected={expected}, actual={actual}"
            )
        input_results[name] = {"path": metadata["path"], "sha256": actual}
    if "protocol" not in input_results or "activation_freeze" not in input_results:
        raise ValueError("Formal freeze wajib memuat protocol dan activation_freeze")

    activation_path = _resolve_inside(
        root, freeze["inputs"]["activation_freeze"]["path"]
    )
    activation = _read_json(activation_path)
    if activation.get("status") != "frozen":
        raise ValueError("Backbone activation freeze tidak berstatus frozen")
    if activation.get("official_test_accessed") is not False:
        raise ValueError("Activation freeze tidak menjamin official test terkunci")

    protocol_path = _resolve_inside(root, freeze["inputs"]["protocol"]["path"])
    protocol, _, protocol_hash = load_phase3_protocol(protocol_path, project_root=root)
    if protocol_hash != freeze["inputs"]["protocol"]["sha256"]:
        raise ValueError("Hash protocol aktif berbeda dari formal freeze")
    if protocol["evaluation"]["official_test_access"]:
        raise ValueError("Protocol aktif membuka official test")

    source_commit = freeze.get("source_git_commit")
    if not source_commit:
        raise ValueError("Formal freeze tidak mencatat source_git_commit")
    attestation_result = None
    if source_attestation:
        attestation_path = Path(source_attestation).resolve()
        if not attestation_path.is_file():
            raise FileNotFoundError(
                f"Kaggle source attestation tidak ditemukan: {attestation_path}"
            )
        attestation_sidecar = attestation_path.with_suffix(
            attestation_path.suffix + ".sha256"
        )
        if not attestation_sidecar.is_file():
            raise FileNotFoundError(
                f"SHA-256 source attestation tidak ditemukan: {attestation_sidecar}"
            )
        sidecar_parts = attestation_sidecar.read_text(
            encoding="utf-8-sig"
        ).strip().split()
        if not sidecar_parts:
            raise ValueError("SHA-256 source attestation kosong")
        actual_attestation_hash = _sha256(attestation_path)
        if sidecar_parts[0].lower() != actual_attestation_hash:
            raise ValueError("Hash Kaggle source attestation tidak cocok")
        attestation = _read_json(attestation_path)
        if attestation.get("status") != "frozen":
            raise ValueError("Kaggle source bundle belum berstatus frozen")
        if attestation.get("formal_freeze_sha256") != actual_freeze_hash:
            raise ValueError("Kaggle source attestation merujuk formal freeze berbeda")
        if attestation.get("formal_freeze_source_git_commit") != source_commit:
            raise ValueError("Commit sumber pada Kaggle attestation tidak cocok")
        if attestation.get("git_worktree_clean_at_bundle_creation") is not True:
            raise ValueError("Kaggle source bundle tidak dibuat dari worktree bersih")
        head_commit = attestation.get("execution_git_commit")
        if not head_commit:
            raise ValueError("Kaggle source attestation tidak mencatat execution commit")
        changed_since_source = {
            str(row).replace("\\", "/")
            for row in attestation.get("changed_since_source", [])
            if str(row).strip()
        }
        source_manifest_path = attestation_path.parent / attestation["source_manifest"]
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"Source file manifest tidak ditemukan: {source_manifest_path}"
            )
        if _sha256(source_manifest_path) != attestation["source_manifest_sha256"]:
            raise ValueError("Hash source file manifest tidak cocok")
        source_manifest = _read_json(source_manifest_path)
        for item in source_manifest.get("files", []):
            source_path = _resolve_inside(root, item["path"])
            if not source_path.is_file() or _sha256(source_path) != item["sha256"]:
                raise ValueError(f"Source bundle file berubah: {item['path']}")
        worktree_clean = True
        attestation_result = {
            "path": str(attestation_path),
            "sha256": actual_attestation_hash,
            "source_manifest_sha256": attestation["source_manifest_sha256"],
        }
    else:
        head_commit = _git(root, "rev-parse", "HEAD")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, head_commit],
            cwd=root,
            capture_output=True,
        )
        if ancestor.returncode != 0:
            raise ValueError("Commit sumber freeze bukan ancestor HEAD")
        changed_since_source = {
            row.replace("\\", "/")
            for row in _git(
                root, "diff", "--name-only", source_commit, head_commit
            ).splitlines()
            if row.strip()
        }
        worktree_status = _git(root, "status", "--porcelain")
        worktree_clean = not bool(worktree_status)

    if require_clean and not worktree_clean:
        raise ValueError("Formal run mensyaratkan git worktree bersih")

    decision = freeze["rules"]["formal_decision"]
    registered_backbones = [
        name for family in ("cnn", "transformer") for name in decision["families"][family]
    ]
    registered_seeds = [int(seed) for seed in protocol["training"]["formal_seeds"]]
    if list(activation.get("primary_candidates", [])) != registered_backbones:
        raise ValueError("Kandidat formal berbeda dari activation freeze")
    if [int(seed) for seed in activation.get("formal_seeds", [])] != registered_seeds:
        raise ValueError("Seed formal berbeda dari activation freeze")
    expected_run_count = len(registered_backbones) * len(registered_seeds)
    if int(activation.get("formal_run_count", -1)) != expected_run_count:
        raise ValueError("Jumlah run formal berbeda dari activation freeze")

    requested_backbones = list(backbones) if backbones is not None else registered_backbones
    requested_seeds = [int(seed) for seed in seeds] if seeds is not None else registered_seeds
    if len(requested_backbones) != len(set(requested_backbones)):
        raise ValueError("Daftar backbone formal memiliki duplikat")
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError("Daftar seed formal memiliki duplikat")
    invalid_backbones = sorted(set(requested_backbones) - set(registered_backbones))
    invalid_seeds = sorted(set(requested_seeds) - set(registered_seeds))
    if invalid_backbones:
        raise ValueError(f"Backbone tidak aktif untuk formal run: {invalid_backbones}")
    if invalid_seeds:
        raise ValueError(f"Seed tidak terdaftar untuk formal run: {invalid_seeds}")

    return {
        "status": "passed",
        "freeze_record": freeze_path.relative_to(root).as_posix(),
        "freeze_record_sha256": actual_freeze_hash,
        "source_git_commit": source_commit,
        "head_commit": head_commit,
        "source_is_ancestor": True,
        "changed_since_source": sorted(changed_since_source),
        "git_worktree_clean": worktree_clean,
        "official_test_access": False,
        "protocol_sha256": protocol_hash,
        "manifest_sha256": activation.get("manifest_sha256"),
        "registered_backbones": registered_backbones,
        "registered_seeds": registered_seeds,
        "requested_backbones": requested_backbones,
        "requested_seeds": requested_seeds,
        "planned_run_count": len(requested_backbones) * len(requested_seeds),
        "input_hashes": input_results,
        "source_attestation": attestation_result,
    }
