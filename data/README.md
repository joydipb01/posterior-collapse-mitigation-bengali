# Data layout

```
data/
  raw/                 # optional hand-managed corpus, not versioned
    train.txt          # one UTF-8 sentence per line
    val.txt
    test.txt
  processed/           # output of `bcvae build-corpus` or `bcvae prepare`
    train.txt
    val.txt
    test.txt
    manifest.json
  sample/              # 40-sentence smoke corpus + 4-document JSONL
  pairs/               # interpolation pairs (versioned)
  labels/              # optional gold labels
```

## Building the training corpus

The study uses a **streaming** builder so the 87 GB FineWeb-2 Bengali split is
never downloaded in full.  Install the optional dependency first:

```bash
uv pip install -e ".[corpus]"
```

Preview the plan without touching the network:

```bash
python -m bcvae build-corpus --config configs/corpus/fineweb2_ben.yaml --dry-run
```

Build it:

```bash
python -m bcvae build-corpus --config configs/corpus/fineweb2_ben.yaml
```

That writes `data/processed/{train,val,test}.txt` and `manifest.json`.  The
manifest records the resolved source, filters, counts, and target shortfalls.

Available recipes:

| Recipe | Role | License |
|---|---|---|
| `configs/corpus/fineweb2_ben.yaml` | Primary training corpus | ODC-By 1.0 |
| `configs/corpus/sangraha_ben.yaml` | Clean replication corpus | CC-BY-4.0 |
| `configs/corpus/fineweb_eng.yaml` | English control for H1 | ODC-By 1.0 |
| `configs/corpus/smoke_jsonl.yaml` | Offline smoke test | bundled sample |

Sangraha exposes one split per language, so its recipe uses `split: ben`
rather than a `language` filter.  FineWeb-2's `ben_Beng` config is already
language-specific.  FineWeb-2 has no English config, so the control uses the
original FineWeb (`sample-10BT`); swap in a matching `CC-MAIN-*` config if you
need the same crawl window.  The control sets `script: latin`; the default is
`script: bengali`, and `script: any` disables the script filter.

`--max-sentences` scales the validation and test targets to 1% automatically
unless you pass `--set val_sentences=...` or `--set test_sentences=...`.

`source: jsonl` lets you build from a local JSONL file without `datasets`:

```json
{"id": "doc-1", "text": "প্রথম বাক্য।\nদ্বিতীয় বাক্য।"}
```

### What the builder guarantees

- **Document-level splits.**  Every sentence from a document goes to exactly
  one of train/val/test.  Sentence-level random splits leak context and
  inflate probe accuracy.
- **Stable hashing.**  Split assignment is `blake2b(seed:document_id)`, so the
  same corpus and seed produce the same split.
- **Cheap filters.**  Length, word count, script ratio, digit ratio, URLs,
  e-mail addresses, and HTML tags.
- **De-duplication.**  `exact`, `normalized` (default; removes punctuation and
  spacing variants), or `none`.
- **No silent shortfall.**  If a split does not reach its target, the manifest
  lists a warning.

The sentence splitter is a deterministic baseline (danda, `?`, `!`, newlines,
and full stops followed by a letter).  Abbreviations are the known failure
mode.  For a camera-ready corpus, replace it with a Bengali-aware splitter and
report the difference.

### Sizing

For a single 12 GB GPU:

- **Grid / Stage A–B:** 300K train sentences.
- **Finalists:** 2M train sentences.
- **Val/test:** 20K each, or 1% of train.

Set `max_documents` for a quick trial:

```bash
python -m bcvae build-corpus --config configs/corpus/fineweb2_ben.yaml \
    --max-documents 2000 --max-sentences 50000 --out data/processed-trial
```

### Tokenizer-fair comparison

Char tokenizers produce roughly 5x more tokens than word tokenizers, so a
single `data.max_tokens` truncates char sentences far more often.  Set the cap
per tokenizer and report the truncation rate:

| Tokenizer | Suggested `max_tokens` |
|---|---|
| char | 256 |
| morph | 96 |
| BPE | 64 |
| word | 48 |

## Raw corpus (without the builder)

Plain text, one sentence per line, UTF-8.  Blank lines are skipped and
duplicates are removed.  Text is NFC-normalised and control characters are
stripped; **ZWNJ/ZWJ are preserved** because they change Bengali conjuncts.

```bash
python -m bcvae prepare --input data/raw/all.txt --out data/processed
```

## Gold labels (optional)

Silver labels (regex heuristics over pronouns and verb endings) are generated
automatically.  If you hand-label a subset, use a tab-separated file with a
header:

```
text	tense	person	honorific
আমি ভাত খাই।	present	1	none
তিনি আসবেন।	future	3	formal
```

Allowed values: `tense ∈ {past, present, future, none}`, `person ∈ {1, 2, 3, none}`,
`honorific ∈ {formal, informal, intimate, none}`.  Point `data.labels` at the
file; matching is by exact sentence text.

Bengali verb suffixes fuse tense, person, and honorific.  Treat the three
labels as correlated, report balanced accuracy or macro-F1, and validate a
sample by hand before interpreting probe results.

## Interpolation pairs

```
sentence A ||| sentence B
```

Both sides are required.  Prefer hand-built minimal pairs that differ in one
controllable dimension (tense, honorific, polarity, person).  BanglaParaphrase
is a useful secondary naturalness set, but most of its pairs change several
things at once.
