import json
import re
from pathlib import Path

from bcvae.cli import main
from bcvae.config import Config
from bcvae.corpus_builder import (
    assign_split,
    bengali_ratio,
    build_corpus,
    passes_filters,
    resolve_source,
    script_ratio,
    split_sentences,
)

ROOT = Path(__file__).resolve().parents[1]


def _filter_config(**overrides):
    base = {
        "min_chars": 10,
        "max_chars": 200,
        "min_words": 3,
        "max_words": 40,
        "script": "bengali",
        "min_script_ratio": 0.8,
        "max_digit_ratio": 0.2,
    }
    base.update(overrides)
    return Config(base)


def test_split_sentences_handles_bengali_punctuation():
    text = "আজ আকাশ পরিষ্কার। আমি ভাত খাই? হ্যাঁ! নতুন লাইন\nআরেক বাক্য।"
    assert split_sentences(text) == [
        "আজ আকাশ পরিষ্কার।",
        "আমি ভাত খাই?",
        "হ্যাঁ!",
        "নতুন লাইন",
        "আরেক বাক্য।",
    ]


def test_split_sentences_keeps_decimals():
    assert split_sentences("দাম ৩.৫ টাকা। পরের বাক্য।") == [
        "দাম ৩.৫ টাকা।",
        "পরের বাক্য।",
    ]


def test_bengali_ratio():
    assert bengali_ratio("আমি ভাত খাই।") == 1.0
    assert bengali_ratio("I eat rice.") == 0.0
    assert 0.0 < bengali_ratio("আমি eat rice।") < 1.0


def test_script_ratio_supports_latin_control():
    assert script_ratio("I eat rice.", "latin") == 1.0
    assert script_ratio("আমি ভাত খাই।", "latin") == 0.0
    cfg = _filter_config(script="latin", min_script_ratio=0.8)
    assert passes_filters("I eat rice every single day.", cfg)[0]
    assert not passes_filters("আমি প্রতিদিন সকালে ভাত খাই।", cfg)[0]


def test_filters_reject_noise_and_short_lines():
    cfg = _filter_config()
    assert passes_filters("আমি প্রতিদিন সকালে ভাত খাই।", cfg)[0]
    assert not passes_filters("ছোট।", cfg)[0]
    assert not passes_filters("I eat rice every single day.", cfg)[0]
    assert not passes_filters(
        "আমি প্রতিদিন সকালে ভাত খাই এবং https://example.com দেখি।", cfg
    )[0]
    assert not passes_filters("আমি ১২৩ ৪৫৬ ৭৮৯ ০১২ ৩৪৫ ৬৭৮", cfg)[0]


def test_terminal_punctuation_filter_rejects_headings():
    cfg = _filter_config(require_terminal_punctuation=True)
    assert passes_filters("আমি প্রতিদিন সকালে ভাত খাই।", cfg)[0]
    assert not passes_filters("বাংলা ভাষার আন্তর্জাতিক নীতি", cfg)[0]


def test_assign_split_is_stable_and_roughly_balanced():
    assert assign_split("doc-1", 0.1, 0.1, seed=13) == assign_split(
        "doc-1", 0.1, 0.1, seed=13
    )
    splits = [assign_split(f"doc-{i}", 0.1, 0.1, seed=13) for i in range(10_000)]
    assert 0.07 < splits.count("val") / len(splits) < 0.13
    assert 0.07 < splits.count("test") / len(splits) < 0.13
    assert 0.74 < splits.count("train") / len(splits) < 0.86


def test_resolve_sangraha_defaults():
    spec = resolve_source(Config({"source": "sangraha"}))
    assert spec.dataset == "ai4bharat/sangraha"
    assert spec.config == "verified"
    assert spec.split == "ben"
    assert spec.language is None


def test_resolve_english_control_recipe(monkeypatch):
    monkeypatch.chdir(ROOT)
    from bcvae.config import load_config

    cfg = load_config(ROOT / "configs" / "corpus" / "fineweb_eng.yaml")
    spec = resolve_source(cfg)
    assert spec.dataset == "HuggingFaceFW/fineweb"
    assert spec.config == "sample-10BT"
    assert cfg.get_path("script") == "latin"


def test_dry_run_does_not_write_or_import_datasets(tmp_path):
    cfg = Config(
        {
            "source": "fineweb2",
            "out": str(tmp_path / "out"),
            "max_sentences": 1234,
            "val_sentences": 12,
            "test_sentences": 12,
        }
    )
    plan = build_corpus(cfg, dry_run=True, quiet=True)
    assert plan["source"]["dataset"] == "HuggingFaceFW/fineweb-2"
    assert plan["source"]["config"] == "ben_Beng"
    assert plan["targets"]["train"] == 1234
    assert not (tmp_path / "out").exists()


def test_build_corpus_from_jsonl(tmp_path):
    docs = []
    for i in range(100):
        sentences = [f"এটি ডকুমেন্ট {i} এর বাক্য {j}।" for j in range(10)]
        docs.append({"id": f"doc-{i}", "text": "\n".join(sentences)})
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "out"
    cfg = Config(
        {
            "source": "jsonl",
            "path": str(path),
            "text_field": "text",
            "id_field": "id",
            "out": str(out),
            "max_sentences": 200,
            "val_sentences": 20,
            "test_sentences": 20,
            "max_documents": None,
            "min_chars": 5,
            "max_chars": 200,
            "min_words": 3,
            "max_words": 40,
            "script": "bengali",
            "min_script_ratio": 0.5,
            "max_digit_ratio": 0.5,
            "dedupe": "normalized",
            "val_frac": 0.1,
            "test_frac": 0.1,
            "seed": 13,
            "show_progress": False,
        }
    )
    manifest = build_corpus(cfg, quiet=True)

    assert manifest["source"]["kind"] == "jsonl"
    assert manifest["files"]["train"]["sentences"] > 0
    assert manifest["files"]["val"]["sentences"] > 0
    assert manifest["files"]["test"]["sentences"] > 0
    assert manifest["counts"]["sentences_kept"] == sum(
        manifest["files"][split]["sentences"] for split in ("train", "val", "test")
    )
    assert manifest["files"]["train"]["sentences"] <= 200
    assert (out / "manifest.json").exists()

    # A document must not leak across splits.
    doc_re = re.compile(r"ডকুমেন্ট (\d+)")
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        lines = (out / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        assert lines
        for line in lines:
            match = doc_re.search(line)
            assert match is not None
            doc_id = match.group(1)
            assert doc_id not in seen or seen[doc_id] == split
            seen[doc_id] = split


def test_cli_build_corpus_dry_run(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    assert (
        main(
            [
                "build-corpus",
                "--config",
                "configs/corpus/fineweb2_ben.yaml",
                "--dry-run",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "HuggingFaceFW/fineweb-2" in printed
    assert '"train": 2000000' in printed
