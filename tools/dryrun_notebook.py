"""Dry-run train.ipynb locally: executes every cell in order on this machine.

Skips the Colab-only bits (pip magic, Drive upload, file download) and shrinks the
config to a couple of games, so it verifies the pipeline path end to end in a minute.

    python tools/dryrun_notebook.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "train.ipynb"

# tiny config so the dry run is quick
OVERRIDES = {"GAMES": 2, "DEPTH": 1, "CELL_SAMPLES": 2, "BATCH": 2, "EVAL_GAMES": 1}


def cells() -> list[dict]:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return nb["cells"]


def strip_magics(src: str) -> str:
    """Drop notebook shell magics (!pip etc.) - they are Colab-only."""
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!"))


def main() -> int:
    ns: dict = {}
    failures = 0
    for i, cell in enumerate(cells()):
        src = "".join(cell.get("source", []))
        head = src.strip().splitlines()[0][:70] if src.strip() else ""
        if cell["cell_type"] != "code":
            print(f"cell {i} [markdown] skip")
            continue
        if "google.colab" in src:
            print(f"cell {i} [code] skipped (Colab-only: Drive + download)")
            continue
        code = strip_magics(src)
        print(f"cell {i} [code] {head}")
        t0 = time.time()
        try:
            exec(compile(code, f"notebook_cell_{i}", "exec"), ns)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAILED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            break
        if "GAMES" in ns and i == 2:  # config cell: shrink it for the dry run
            ns.update(OVERRIDES)
            print(f"  overrides: {OVERRIDES}")
        print(f"  ok ({time.time() - t0:.1f}s)")
    print("\ndry run:", "FAILED" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
