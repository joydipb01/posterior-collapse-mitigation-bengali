import json
from pathlib import Path

from bcvae.config import load_config
from bcvae.training.trainer import train

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_training(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg.set_path("train.epochs", 1)

    run_dir = train(cfg, run_dir=tmp_path / "run", quiet=True)

    assert (run_dir / "model.pt").exists()
    assert (run_dir / "tokenizer" / "tokenizer.json").exists()
    assert (run_dir / "config.yaml").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] > 0
    assert summary["best_val_nll"] < 1e6

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["split"] == "val" for row in rows)


def test_grid_expansion():
    from bcvae.grid import expand_grid

    runs = expand_grid(
        {"axes": {"tokenizer.type": ["bpe", "char"], "train.kl.schedule": ["linear"]}, "seeds": [1, 2]}
    )
    assert len(runs) == 4
    assert len({run["name"] for run in runs}) == 4
