"""Aggregate per-run metrics into a paper-ready CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import load_config

PROBE_FEATURES = ("tense", "person", "honorific")


def summarize_runs(runs_dir: str | Path, out_path: str | Path | None = None) -> Path:
    runs_dir = Path(runs_dir)
    rows: list[dict] = []

    for metrics_path in sorted(runs_dir.glob("**/metrics.jsonl")):
        run_dir = metrics_path.parent
        records = []
        with metrics_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        val_records = [row for row in records if row.get("split") == "val"]
        if not val_records:
            continue

        last = val_records[-1]
        cfg = load_config(run_dir / "config.yaml")
        row = {
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "tokenizer": cfg.get_path("tokenizer.type"),
            "schedule": cfg.get_path("train.kl.schedule"),
            "decoder": cfg.get_path("model.decoder"),
            "skip_connection": cfg.get_path("model.skip_connection"),
            "seed": cfg.get_path("seed"),
            "epoch": last.get("epoch"),
            "step": last.get("step"),
            "val_nll": last.get("nll_per_token"),
            "val_kl": last.get("kl"),
            "active_units": last.get("active_units"),
            "effective_dim": last.get("effective_dim"),
        }
        probe = last.get("probe") or {}
        for feature in PROBE_FEATURES:
            entry = probe.get(feature) or {}
            prefix = f"probe_{feature}"
            if entry.get("skipped", True):
                row[prefix] = None
                row[f"{prefix}_balanced"] = None
                row[f"{prefix}_macro_f1"] = None
            else:
                row[prefix] = entry.get("accuracy")
                row[f"{prefix}_balanced"] = entry.get("balanced_accuracy")
                row[f"{prefix}_macro_f1"] = entry.get("macro_f1")
        rows.append(row)

    out_path = Path(out_path) if out_path else runs_dir / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "run_dir",
        "tokenizer",
        "schedule",
        "decoder",
        "skip_connection",
        "seed",
        "epoch",
        "step",
        "val_nll",
        "val_kl",
        "active_units",
        "effective_dim",
        *[
            column
            for feature in PROBE_FEATURES
            for column in (
                f"probe_{feature}",
                f"probe_{feature}_balanced",
                f"probe_{feature}_macro_f1",
            )
        ],
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return out_path
