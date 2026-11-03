"""CLI for the Phase 3 formal cryptographic preflight gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.phase3_formal_preflight import validate_phase3_formal_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-record")
    parser.add_argument("--backbone", action="append", dest="backbones")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    try:
        report = validate_phase3_formal_preflight(
            project_root=PROJECT_ROOT,
            freeze_record=args.freeze_record,
            backbones=args.backbones,
            seeds=args.seeds,
            require_clean=not args.allow_dirty,
        )
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 1
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
