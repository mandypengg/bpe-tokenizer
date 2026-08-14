# bpe-tokenizer

A byte-pair-encoding tokenizer written from scratch, whose encoder reproduces
GPT-2's tokenization exactly (verified against `tiktoken`).

## Layout

```
bpe/
  base.py     get_stats / merge primitives, Tokenizer base class, save+load
  basic.py    BasicTokenizer  - byte-level BPE, no splitting
  regex.py    RegexTokenizer  - GPT-2/GPT-4 split patterns + special tokens
  gpt2.py     GPT2Tokenizer   - loads OpenAI's encoder.json / vocab.bpe
tests/        pytest suite, including exact-match tests against tiktoken
benchmarks/   compression ratio vs vocab size, with a matplotlib plot
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

## Usage

```python
from bpe import RegexTokenizer, GPT2Tokenizer

tok = RegexTokenizer()
tok.train(open("corpus.txt").read(), vocab_size=1024)
tok.register_special_tokens({"<|endoftext|>": 1024})
tok.encode("hello world<|endoftext|>", allowed_special="all")
tok.save("models/mine")          # -> mine.model, mine.vocab

gpt2 = GPT2Tokenizer.from_pretrained()   # downloads to data/gpt2/ once
gpt2.encode_ordinary("hello world")      # [31373, 995]
```

## Benchmarks

```bash
.venv/bin/python benchmarks/compression_ratio.py --input corpus.txt
```

Trains at a range of vocab sizes and plots bytes-per-token against vocab size,
with GPT-2's ratio on the same corpus as a reference line.

## Notes

- Merges are ordered; the position in `Tokenizer.merges` is the priority.
  Encoding always applies the lowest-ranked eligible pair, which is what makes
  encoding agree with training.
- GPT-2's single-byte tokens are not ids 0..255 — `encoder.json` assigns them
  in `bytes_to_unicode` order. `GPT2Tokenizer` overrides `_byte_ids` to seed
  encoding with that permutation; every tokenizer trained here uses the
  identity mapping instead.
- Requires the `regex` package rather than stdlib `re`, for `\p{L}` and
  possessive quantifiers in the split patterns.
