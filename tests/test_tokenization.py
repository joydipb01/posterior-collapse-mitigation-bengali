import pytest

from bcvae.tokenization import (
    BPETokenizer,
    CharTokenizer,
    MorphAnalyzer,
    MorphTokenizer,
    WordTokenizer,
    load_tokenizer,
    save_tokenizer,
)

SENTENCES = [
    "আমি প্রতিদিন সকালে ভাত খাই।",
    "সে গতকাল স্কুলে গিয়েছিল।",
    "তিনি অফিসে দেরি করে পৌঁছেছিলেন।",
    "শিশুরা বাগানে খেলছে।",
]


@pytest.mark.parametrize("cls", [CharTokenizer, WordTokenizer, MorphTokenizer])
def test_roundtrip_and_save(cls, tmp_path):
    tokenizer = cls.train(SENTENCES, vocab_size=500, min_frequency=1)
    text = SENTENCES[0]
    ids = tokenizer.encode(text)
    assert ids
    assert tokenizer.decode(ids) == text

    save_tokenizer(tokenizer, tmp_path / "tok")
    loaded = load_tokenizer(tmp_path / "tok")
    assert loaded.encode(text) == ids
    assert loaded.vocab_size == tokenizer.vocab_size


def test_bpe_roundtrip(tmp_path):
    tokenizer = BPETokenizer.train(SENTENCES, vocab_size=300, min_frequency=1)
    text = SENTENCES[1]
    assert tokenizer.decode(tokenizer.encode(text)) == text

    save_tokenizer(tokenizer, tmp_path / "tok")
    loaded = load_tokenizer(tmp_path / "tok")
    assert loaded.encode(text) == tokenizer.encode(text)


def test_morph_segments_concatenate():
    analyzer = MorphAnalyzer()
    for word in ["বইগুলো", "ছেলেটি", "খাচ্ছিলাম", "স্কুলে"]:
        pieces = analyzer.segment(word)
        assert "".join(pieces) == word
        assert len(pieces) >= 1


def test_specials_are_stable():
    tokenizer = WordTokenizer.train(SENTENCES, vocab_size=500, min_frequency=1)
    assert tokenizer.itos[:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]
    assert tokenizer.encode("অজানা-শব্দ")[0] == tokenizer.unk_id
