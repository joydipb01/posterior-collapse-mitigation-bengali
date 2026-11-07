"""Cross-product experiment runner.

Each cell is launched in its own process so a crashed run does not take the
grid down and so memory is released between training jobs.
"""

from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def _slug(value) -> str:
    text = str(value).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "x"


def expand_grid(spec: dict) -> list[dict]:
    axes: dict[str, list] = spec.get("axes") or {}
    if not axes:
        raise ValueError("grid spec needs a non-empty 'axes' mapping")
    seeds = spec.get("seeds") or [spec.get("seed", 13)]
    keys = list(axes)
    runs = []
    for combo in itertools.product(*(axes[key] for key in keys)):
        for seed in seeds:
            runs.append(
                {
                    "overrides": dict(zip(keys, combo, strict=True)),
                    "seed": seed,
                    "name": "-".join(_slug(v) for v in combo) + f"-s{seed}",
                }
            )
    return runs


def run_grid(grid_path: str | Path, *, dry_run: bool = False, max_runs: int | None = None) -> None:
    grid_path = Path(grid_path)
    with grid_path.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}

    base = (grid_path.parent / spec.get("base", "../base.yaml")).resolve()
    out_dir = spec.get("out_dir", "runs/grid")
    runs = expand_grid(spec)
    if max_runs is None:
        max_runs = spec.get("max_runs")
    if max_runs is not None:
        runs = runs[:max_runs]

    for run in runs:
        command = [
            sys.executable,
            "-m",
            "bcvae",
            "train",
            "--config",
            str(base),
            "--run-dir",
            out_dir,
            "--run-name",
            run["name"],
        ]
        for key, value in run["overrides"].items():
            command += ["--set", f"{key}={json.dumps(value)}"]
        command += ["--set", f"seed={run['seed']}"]

        if dry_run:
            print(" ".join(command))
            continue

        print(f"=== {run['name']}", flush=True)
        subprocess.run(command, check=True)
