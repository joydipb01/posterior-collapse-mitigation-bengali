"""Training loop with the collapse-mitigation levers wired in."""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..checkpoint import model_from_config, save_run
from ..config import Config
from ..corpus import read_sentences
from ..dataset import SentenceDataset, collate_batch
from ..eval.reconstruct import evaluate
from ..probing import collect_latents, probe_report
from ..seeding import resolve_device, set_seed
from ..tokenization import build_tokenizer, load_tokenizer
from .schedules import beta_at


def _unique_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.name}-{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free run directory near {path}")


def _run_dir(cfg: Config, tokenizer_type: str) -> Path:
    name = cfg.get_path("logging.run_name") or f"{tokenizer_type}-{cfg.get_path('train.kl.schedule')}"
    return _unique_dir(Path(cfg.get_path("logging.run_dir", "runs")) / str(name))


def _load_texts(path: str, cfg: Config, fallback: list[str] | None = None) -> list[str]:
    path_obj = Path(path)
    if not path_obj.exists() or path_obj.stat().st_size == 0:
        if fallback is None:
            raise FileNotFoundError(f"corpus not found: {path}")
        return fallback
    return read_sentences(
        path_obj,
        max_sentences=cfg.get_path("data.max_sentences"),
        min_chars=cfg.get_path("data.min_chars", 2),
    )


def train(cfg: Config, run_dir: str | Path | None = None, *, quiet: bool = False) -> Path:
    seed = int(cfg.get_path("seed", 13))
    set_seed(seed)
    device = resolve_device(cfg.get_path("device", "auto"))

    train_texts = _load_texts(cfg["data"]["train"], cfg)
    val_texts = _load_texts(cfg["data"]["val"], cfg, fallback=train_texts)
    if not train_texts:
        raise ValueError("training corpus is empty")

    tok_cfg = cfg["tokenizer"]
    model_path = tok_cfg.get("model_path")
    if model_path:
        tokenizer = load_tokenizer(model_path)
    else:
        tokenizer = build_tokenizer(
            tok_cfg["type"],
            train_texts,
            vocab_size=int(tok_cfg.get("vocab_size", 8000)),
            min_frequency=int(tok_cfg.get("min_frequency", 2)),
            morph_suffixes=cfg.get_path("tokenizer.morph.suffixes"),
        )

    run_dir = Path(run_dir) if run_dir else _run_dir(cfg, tokenizer.type_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)

    max_tokens = int(cfg.get_path("data.max_tokens", 40))
    batch_size = int(cfg.get_path("train.batch_size", 64))
    collate = partial(collate_batch, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        SentenceDataset(train_texts, tokenizer, max_tokens),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=len(train_texts) > batch_size,
        num_workers=int(cfg.get_path("train.num_workers", 0)),
    )
    val_loader = DataLoader(
        SentenceDataset(val_texts, tokenizer, max_tokens),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise ValueError("no training batches; check max_tokens and batch_size")

    model = model_from_config(cfg, tokenizer).to(device)
    lr = float(cfg.get_path("train.lr", 1e-3))
    weight_decay = float(cfg.get_path("train.weight_decay", 0.0))
    lag = max(1, int(cfg.get_path("train.encoder_lag", 1)))

    if lag > 1:
        # Disjoint parameter sets: the shared embedding follows the decoder
        # optimizer, so it still updates every step.  This is an approximation
        # of He et al.'s lagging inference network, not a bit-exact replica.
        enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        opt_enc = torch.optim.Adam(enc_params, lr=lr, weight_decay=weight_decay)
        opt_dec = torch.optim.Adam(dec_params, lr=lr, weight_decay=weight_decay)
        opt = None
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        opt_enc = opt_dec = None

    epochs = int(cfg.get_path("train.epochs", 20))
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(float(cfg.get_path("train.kl.warmup_epochs", 10)) * steps_per_epoch)
    schedule = cfg.get_path("train.kl.schedule", "linear")
    cycles = int(cfg.get_path("train.kl.cycles", 4))
    ratio = float(cfg.get_path("train.kl.ratio", 0.5))
    clip = float(cfg.get_path("train.clip", 5.0))
    max_batches = cfg.get_path("train.max_batches")
    log_every = int(cfg.get_path("train.log_every", 50))
    eval_every = max(1, int(cfg.get_path("train.eval_every", 1)))
    probe_every = max(1, int(cfg.get_path("train.probe_every", 2)))
    gold = None
    if cfg.get_path("data.labels"):
        from ..labels import load_gold

        gold = load_gold(cfg.get_path("data.labels"))

    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8")

    def record(row: dict) -> None:
        metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_file.flush()

    def log(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr, flush=True)

    log(f"run: {run_dir}  device={device}  vocab={tokenizer.vocab_size}")

    best_nll = float("inf")
    global_step = 0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            progress = tqdm(
                train_loader,
                desc=f"epoch {epoch}/{epochs}",
                disable=quiet or not sys.stderr.isatty(),
                leave=False,
            )
            for batch_idx, batch in enumerate(progress):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                input_ids = batch["input_ids"].to(device)
                lengths = batch["lengths"].to(device)
                beta = beta_at(
                    global_step,
                    schedule=schedule,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    cycles=cycles,
                    ratio=ratio,
                )

                if opt is not None:
                    opt.zero_grad(set_to_none=True)
                else:
                    opt_enc.zero_grad(set_to_none=True)
                    opt_dec.zero_grad(set_to_none=True)

                out = model(input_ids, lengths, beta=beta)
                out["loss"].backward()
                clip_grad_norm_(model.parameters(), clip)

                if opt is not None:
                    opt.step()
                else:
                    opt_dec.step()
                    if global_step % lag == 0:
                        opt_enc.step()

                global_step += 1
                progress.set_postfix(
                    nll=f"{float(out['nll_per_token']):.3f}",
                    kl=f"{float(out['kl']):.2f}",
                    beta=f"{beta:.2f}",
                )

                if global_step % log_every == 0:
                    row = {
                        "split": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "nll_per_token": float(out["nll_per_token"]),
                        "kl": float(out["kl"]),
                        "beta": beta,
                        "active_units": int((out["kl_per_dim"] > 1e-2).sum()),
                    }
                    record(row)
                    log(
                        f"epoch {epoch} step {global_step} "
                        f"nll={row['nll_per_token']:.3f} kl={row['kl']:.2f} beta={beta:.2f}"
                    )

            if epoch % eval_every == 0:
                val = evaluate(
                    model,
                    val_loader,
                    tokenizer,
                    device,
                    beta=1.0,
                    max_batches=max_batches,
                )
                probe = None
                if epoch % probe_every == 0:
                    mu, texts = collect_latents(model, val_loader, device, max_batches)
                    probe = probe_report(mu, texts, gold, seed=seed)

                row = {
                    "split": "val",
                    "epoch": epoch,
                    "step": global_step,
                    **{k: v for k, v in val.items() if k != "samples"},
                }
                if probe is not None:
                    row["probe"] = probe
                record(row)

                meta = {
                    "epoch": epoch,
                    "step": global_step,
                    "val_nll": val["nll_per_token"],
                    "probe": probe,
                }
                save_run(run_dir, model, cfg, tokenizer, meta, name="last.pt")
                if val["nll_per_token"] < best_nll:
                    best_nll = val["nll_per_token"]
                    save_run(run_dir, model, cfg, tokenizer, meta, name="model.pt")

                log(
                    f"[val] epoch {epoch} nll={val['nll_per_token']:.3f} "
                    f"kl={val['kl']:.2f} active={val['active_units']} "
                    f"eff_dim={val['effective_dim']:.2f}"
                )
                if not quiet:
                    for sample in val["samples"][:2]:
                        log(f"  ref: {sample['reference']}")
                        log(f"  hyp: {sample['reconstruction']}")
    finally:
        metrics_file.close()

    summary = {
        "run_dir": str(run_dir),
        "best_val_nll": best_nll,
        "steps": global_step,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer.type_name,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"done: best val nll/token = {best_nll:.3f}")
    return run_dir
