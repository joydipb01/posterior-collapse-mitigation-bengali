from bcvae.eval.interpolate import chain_metrics
from bcvae.eval.text_metrics import (
    char_ngram_similarity,
    distinct_n,
    spearman,
    token_f1,
)


def test_token_f1():
    assert token_f1("আমি ভাত খাই", "আমি ভাত খাই") == 1.0
    assert token_f1("আমি ভাত খাই", "সে ভাত খায়") < 1.0
    assert token_f1("", "কিছু") == 0.0


def test_char_similarity_is_bounded():
    a = "আজ আকাশ পরিষ্কার।"
    assert char_ngram_similarity(a, a) == 1.0
    assert 0.0 <= char_ngram_similarity(a, "তিনি অফিসে যান।") < 1.0


def test_distinct_and_spearman():
    assert distinct_n(["এক দুই", "এক দুই"], 1) == 0.5
    assert round(spearman([0, 1, 2, 3], [0, 1, 2, 3]), 6) == 1.0


def test_chain_metrics():
    chain = ["আজ আকাশ পরিষ্কার।", "আজ আকাশ মেঘলা।", "আজ আকাশ খুব মেঘলা।"]
    metrics = chain_metrics(chain)
    assert metrics["n_steps"] == 3
    assert metrics["f1_to_start"][0] == 1.0
    assert metrics["f1_to_end"][-1] == 1.0
    assert 0.0 <= metrics["smoothness"] <= 1.0
