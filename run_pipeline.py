"""Rebuild every data table from the raw Spotify export, in order.

    python run_pipeline.py            # run all steps
    python run_pipeline.py 2          # start from step 02 (earlier outputs already exist)

Each step is a numbered script in pipeline/. They run one after another; if one fails, the run stops.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / "pipeline"


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = sorted(p for p in PIPELINE.glob("[0-9][0-9]_*.py") if int(p.name[:2]) >= start)
    for step in steps:
        print(f"\n=== {step.name} ===", flush=True)
        # UTF-8 output, so song titles in any alphabet can be printed on Windows too
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        subprocess.run([sys.executable, step.name], cwd=PIPELINE, check=True, env=env)


if __name__ == "__main__":
    main()
