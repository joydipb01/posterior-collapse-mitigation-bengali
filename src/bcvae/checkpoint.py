"""Checkpoint and run-directory handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config, load_config
from .models import SentenceVAE
from .tokenization import Tokenizer, load_tokenizer, save_tokenizer


def model_from_config(cfg: Config, tokenizer: Tokenizer) -> SentenceVAE:
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    return SentenceVAE(
        vocab_size=tokenizer.vocab_size,
        emb_dim=model_cfg["emb_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        enc_layers=model_cfg["enc_layers"],
        dec_layers=model_cfg["dec_layers"],
        dropout=model_cfg["dropout"],
        decoder=model_cfg["decoder"],
        bidirectional=model_cfg["bidirectional"],
        skip_connection=model_cfg["skip_connection"],
        tie_embeddings=model_cfg["tie_embeddings"],
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        word_dropout=train_cfg["word_dropout"],
        free_bits=train_cfg["free_bits"],
    )


def save_run(
    run_dir: str | Path,
    model: SentenceVAE,
    cfg: Config,
    tokenizer: Tokenizer,
    meta: dict[str, Any],
    *,
    name: str = "model.pt",
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, run_dir / "tokenizer")
    path = run_dir / name
    torch.save(
        {"model": model.state_dict(), "meta": meta},
        path,
    )
    return path


def load_run(run_dir: str | Path, device: str | torch.device = "cpu"):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    tokenizer = load_tokenizer(run_dir / "tokenizer")
    model = model_from_config(cfg, tokenizer)
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, cfg, ckpt.get("meta", {})
