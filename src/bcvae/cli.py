"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import summarize_runs
from .checkpoint import load_run
from .config import load_config
from .corpus import read_sentences, split_corpus, write_lines
from .corpus_builder import build_corpus
from .dataset import SentenceDataset, collate_batch
from .eval.interpolate import interpolate_pairs, read_pairs, write_results
from .eval.reconstruct import evaluate
from .grid import run_grid
from .probing import collect_latents, probe_report
from .seeding import resolve_device
from .training.trainer import train


def _add_set(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config leaf, e.g. --set max_sentences=100000 (repeatable)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bcvae", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="normalise and split a corpus")
    p_prepare.add_argument("--input", nargs="+", required=True)
    p_prepare.add_argument("--out", default="data/processed")
    p_prepare.add_argument("--val-frac", type=float, default=0.05)
    p_prepare.add_argument("--test-frac", type=float, default=0.05)
    p_prepare.add_argument("--seed", type=int, default=13)
    p_prepare.add_argument("--max-sentences", type=int, default=None)

    p_corpus = sub.add_parser(
        "build-corpus",
        aliases=["fetch"],
        help="stream a Hugging Face corpus, filter it, and write document-level splits",
    )
    p_corpus.add_argument(
        "--config",
        default="configs/corpus/fineweb2_ben.yaml",
        help="corpus recipe YAML (default: FineWeb-2 Bengali)",
    )
    p_corpus.add_argument("--out", default=None)
    p_corpus.add_argument("--max-sentences", type=int, default=None)
    p_corpus.add_argument("--max-documents", type=int, default=None)
    p_corpus.add_argument("--dry-run", action="store_true")
    p_corpus.add_argument("--quiet", action="store_true")
    _add_set(p_corpus)

    p_train = sub.add_parser("train", help="train a sentence VAE")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--run-dir", default=None, help="parent directory for runs")
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--quiet", action="store_true")
    _add_set(p_train)

    p_eval = sub.add_parser("evaluate", help="reconstruction + collapse metrics")
    p_eval.add_argument("--run", required=True)
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--max-batches", type=int, default=None)
    p_eval.add_argument("--out", default=None)
    p_eval.add_argument("--device", default="auto")

    p_interp = sub.add_parser("interpolate", help="decode interpolation chains")
    p_interp.add_argument("--run", required=True)
    p_interp.add_argument("--pairs", required=True)
    p_interp.add_argument("--steps", type=int, default=7)
    p_interp.add_argument("--mode", default="linear", choices=["linear", "slerp"])
    p_interp.add_argument("--out", default=None)
    p_interp.add_argument("--device", default="auto")

    p_probe = sub.add_parser("probe", help="linear probes on the posterior mean")
    p_probe.add_argument("--run", required=True)
    p_probe.add_argument("--split", default="val", choices=["train", "val", "test"])
    p_probe.add_argument("--max-batches", type=int, default=None)
    p_probe.add_argument("--device", default="auto")

    p_grid = sub.add_parser("grid", help="run a cross-product experiment grid")
    p_grid.add_argument("--grid", required=True)
    p_grid.add_argument("--dry-run", action="store_true")
    p_grid.add_argument("--max-runs", type=int, default=None)

    p_sum = sub.add_parser("summarize", help="collect run metrics into a CSV")
    p_sum.add_argument("--runs", default="runs")
    p_sum.add_argument("--out", default=None)

    return parser


def _cmd_prepare(args: argparse.Namespace) -> int:
    sentences: list[str] = []
    for path in args.input:
        sentences.extend(
            read_sentences(path, max_sentences=args.max_sentences, min_chars=2)
        )
    unique = list(dict.fromkeys(sentences))
    train, val, test = split_corpus(
        unique, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    out = Path(args.out)
    write_lines(out / "train.txt", train)
    write_lines(out / "val.txt", val)
    write_lines(out / "test.txt", test)
    print(
        json.dumps(
            {"train": len(train), "val": len(val), "test": len(test), "out": str(out)},
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.set)
    if args.run_dir:
        cfg.set_path("logging.run_dir", args.run_dir)
    if args.run_name:
        cfg.set_path("logging.run_name", args.run_name)
    train(cfg, quiet=args.quiet)
    return 0


def _cmd_build_corpus(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.set)
    if args.out:
        cfg.set_path("out", args.out)
    if args.max_sentences is not None:
        cfg.set_path("max_sentences", args.max_sentences)
        # Keep quick trials proportional unless the user set these explicitly.
        provided = {item.split("=", 1)[0].strip() for item in args.set if "=" in item}
        if "val_sentences" not in provided:
            cfg.set_path("val_sentences", max(1, args.max_sentences // 100))
        if "test_sentences" not in provided:
            cfg.set_path("test_sentences", max(1, args.max_sentences // 100))
    if args.max_documents is not None:
        cfg.set_path("max_documents", args.max_documents)
    build_corpus(cfg, dry_run=args.dry_run, quiet=args.quiet)
    return 0


def _loader_for_run(cfg, tokenizer, split: str, max_tokens: int):
    from functools import partial

    from torch.utils.data import DataLoader

    path = cfg["data"].get(split, cfg["data"]["val"])
    texts = read_sentences(path, max_sentences=cfg.get_path("data.max_sentences"))
    if not texts:
        raise ValueError(f"no sentences in {path}")
    dataset = SentenceDataset(texts, tokenizer, max_tokens)
    collate = partial(collate_batch, pad_id=tokenizer.pad_id)
    return DataLoader(dataset, batch_size=cfg.get_path("train.batch_size", 64), collate_fn=collate)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, tokenizer, cfg, _ = load_run(args.run, device)
    loader = _loader_for_run(cfg, tokenizer, args.split, cfg.get_path("data.max_tokens", 40))
    metrics = evaluate(
        model,
        loader,
        tokenizer,
        device,
        beta=1.0,
        max_batches=args.max_batches,
        sample_n=5,
        max_len=cfg.get_path("data.max_tokens", 40),
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def _cmd_interpolate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, tokenizer, cfg, _ = load_run(args.run, device)
    pairs = read_pairs(args.pairs)
    results = interpolate_pairs(
        model,
        tokenizer,
        pairs,
        steps=args.steps,
        mode=args.mode,
        device=device,
        max_tokens=cfg.get_path("data.max_tokens", 40),
    )
    out = args.out or str(Path(args.run) / "interpolations.jsonl")
    write_results(results, out)

    summary = {
        "pairs": len(results),
        "mean_smoothness": sum(r["metrics"]["smoothness"] for r in results) / len(results),
        "mean_max_jump": sum(r["metrics"]["max_jump"] for r in results) / len(results),
        "mean_monotonicity": sum(r["metrics"]["monotonicity"] for r in results)
        / len(results),
        "out": out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, tokenizer, cfg, _ = load_run(args.run, device)
    loader = _loader_for_run(cfg, tokenizer, args.split, cfg.get_path("data.max_tokens", 40))
    mu, texts = collect_latents(model, loader, device, args.max_batches)
    if len(texts) == 0:
        raise ValueError("no sentences collected for probing")
    gold = None
    if cfg.get_path("data.labels"):
        from .labels import load_gold

        gold = load_gold(cfg.get_path("data.labels"))
    print(json.dumps(probe_report(mu, texts, gold), ensure_ascii=False, indent=2))
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    run_grid(args.grid, dry_run=args.dry_run, max_runs=args.max_runs)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    out = summarize_runs(args.runs, args.out)
    print(str(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "prepare": _cmd_prepare,
        "build-corpus": _cmd_build_corpus,
        "fetch": _cmd_build_corpus,
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
        "interpolate": _cmd_interpolate,
        "probe": _cmd_probe,
        "grid": _cmd_grid,
        "summarize": _cmd_summarize,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
