# TrashCan Phase 3 backbone selection release

This package accompanies the paper on leakage-aware, multi-seed, resource-constrained selection of CNN and Transformer backbones for underwater trash detection. It contains the training/evaluation source, frozen Phase 3 protocol and split, per-job metric records, and formal inference benchmark outputs. It excludes raw images, annotations, pretrained weight files, and trained checkpoints.

## Contents

- `main.py`, `models/`, `data/`, `configs/`, `utils/`: experiment implementation.
- `Riset/configs/`: frozen training protocol, pretrained weights registry, and formal job registry.
- `splits/` and `Riset/output/research_phase1/internal_split_manifest.json`: identical final internal split.
- `Riset/output/research_phase3/`: freeze and audit records, evaluation report, and formal inference results.
- `Riset/output/runs/`: lightweight metadata and metrics for the registered jobs; checkpoints are excluded.
- `DATA.md`: original data source and required local layout.
- `SHA256SUMS.txt`: checksums of this release.

## Reproduce

1. Obtain the original TrashCan material dataset and place it as described in `DATA.md`. Confirm that the annotation SHA-256 matches the frozen split manifest.
2. Create a Python environment and install `requirements.txt`. Use the GPU/PyTorch combination recorded in `environment_lock.json` for each run when matching the published resource numbers. The version ranges in `requirements.txt` alone do not lock the full environment.
3. Supply the pretrained backbone weights described in `Riset/configs/phase3_pretrained_weights_lock.json` and verify their hashes. No pretrained binaries are included.
4. Review `PHASE3_BACKBONE_SELECTION_PROTOCOL.md` and `Riset/output/research_phase3/phase3_formal_protocol_freeze.json`. Use the formal job registry and the scripts under `scripts/` for preflight and execution. The PowerShell runner assumes a local `.trashcan-env` and RTX 3060; adapt that path for another platform and document deviations.
5. For inference benchmarking, trained best checkpoints are required; they are not in this package. Published benchmark records are included for inspection.

The release is a **research artifact**, not a one-command rerun of all 15 training jobs. The paper's original measurements require the original data, exact weights, checkpoints, and hardware conditions. The frozen protocol records the historical source commit; this package also contains lightweight outputs from those runs.

## Citation and licensing

The Python and PowerShell source code in this release is licensed under the MIT License. Copyright (c) 2026 Didik Kurniawan. See `LICENSE` for the full English license text. Please cite the associated article when using this artifact. The TrashCan dataset, pretrained weights, and any third-party materials retain their own terms and are not relicensed by this notice.

Code development was assisted by OpenAI Codex; the author reviewed and validated the code.

The source files are kept byte-for-byte intact so the recorded source-tree and frozen-input hashes remain verifiable; the repository-level `LICENSE` applies to the code files without changing their contents.
