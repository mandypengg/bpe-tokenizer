"""
Roundtrip property over real English: decode(encode(t)) == t.

Trains on a few hundred KB of Project Gutenberg prose (see the `english_corpus`
fixture in conftest.py) and then checks the property over a few thousand
strings drawn from that same corpus. The samples deliberately include slices
cut at arbitrary character offsets, not just clean paragraph or sentence
boundaries — a tokenizer that only roundtrips well-formed input is not
roundtripping, it is getting lucky.

Training happens once per session per tokenizer (session-scoped fixtures);
these are the slowest tests in the suite by a wide margin (~35s, against
well under a second for everything else).

`test_first_50_merges_are_sane` prints the merge table on every run, no -s
needed — it writes through `capsys.disabled()`.
"""

from __future__ import annotations

import random

import pytest

from bpe import BasicTokenizer, RegexTokenizer

# how much of the corpus to train on, and how big a vocabulary to learn.
# 300 KB / 512 is a few seconds of training; the property under test does not
# get more true with a bigger vocab, only slower to check.
TRAIN_BYTES = 300_000
VOCAB_SIZE = 512

NUM_SAMPLES = 3000
SEED = 0


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="session")
def training_text(english_corpus: str) -> str:
    return english_corpus[:TRAIN_BYTES]


@pytest.fixture(scope="session")
def basic_tokenizer(training_text: str) -> BasicTokenizer:
    tok = BasicTokenizer()
    tok.train(training_text, VOCAB_SIZE)
    return tok


@pytest.fixture(scope="session")
def regex_tokenizer(training_text: str) -> RegexTokenizer:
    tok = RegexTokenizer()
    tok.train(training_text, VOCAB_SIZE)
    return tok


@pytest.fixture(scope="session")
def samples(english_corpus: str) -> list[str]:
    """
    A few thousand strings from the corpus, in four flavours.

    Drawn from the whole corpus, not just the slice used for training, so a
    good share of these contain text the tokenizer never saw.
    """
    rng = random.Random(SEED)
    text = english_corpus
    out: list[str] = []

    # 1. paragraphs
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    out.extend(rng.sample(paragraphs, min(750, len(paragraphs))))

    # 2. single lines
    lines = [ln for ln in text.split("\n") if ln.strip()]
    out.extend(rng.sample(lines, min(750, len(lines))))

    # 3. sentence-ish fragments
    sentences = [s for s in text.replace("\n", " ").split(". ") if s.strip()]
    out.extend(rng.sample(sentences, min(750, len(sentences))))

    # 4. arbitrary slices: start and end fall wherever they fall, so these cut
    #    through the middle of words, quotes, and merge-able runs
    for _ in range(NUM_SAMPLES - len(out)):
        start = rng.randrange(len(text))
        length = rng.randint(1, 300)
        out.append(text[start : start + length])

    return out


# -- the property -------------------------------------------------------------


@pytest.mark.parametrize("name", ["basic", "regex"])
def test_roundtrip_over_corpus_samples(name, samples, basic_tokenizer, regex_tokenizer):
    tok = {"basic": basic_tokenizer, "regex": regex_tokenizer}[name]
    for text in samples:
        assert tok.decode(tok.encode(text)) == text, f"roundtrip failed on {text!r}"


@pytest.mark.parametrize("name", ["basic", "regex"])
def test_roundtrip_over_whole_corpus(name, english_corpus, basic_tokenizer, regex_tokenizer):
    """The corpus as one string, including the ~275 KB never trained on."""
    tok = {"basic": basic_tokenizer, "regex": regex_tokenizer}[name]
    assert tok.decode(tok.encode(english_corpus)) == english_corpus


@pytest.mark.parametrize("name", ["basic", "regex"])
def test_every_emitted_id_is_in_vocab(name, samples, basic_tokenizer, regex_tokenizer):
    tok = {"basic": basic_tokenizer, "regex": regex_tokenizer}[name]
    for text in samples[:500]:
        assert all(idx in tok.vocab for idx in tok.encode(text))


def test_training_actually_compressed_the_corpus(basic_tokenizer, training_text):
    nbytes = len(training_text.encode("utf-8"))
    ntokens = len(basic_tokenizer.encode(training_text))
    # 512 tokens over English should comfortably beat 1.5 bytes/token
    assert nbytes / ntokens > 1.5


# -- eyeball the merges -------------------------------------------------------


def test_first_50_merges_are_sane(basic_tokenizer, capsys):
    """
    Print the first 50 merges learned, and assert the structural invariants
    that make them meaningful: learned in order, ids assigned 256 upward, and
    every merged token equal to the concatenation of its two children.
    """
    tok = basic_tokenizer
    first_50 = list(tok.merges.items())[:50]

    with capsys.disabled():
        print(f"\nfirst {len(first_50)} merges (BasicTokenizer, "
              f"{TRAIN_BYTES // 1000} KB English, vocab {VOCAB_SIZE}):\n")
        for rank, ((p0, p1), idx) in enumerate(first_50):
            left = tok.vocab[p0].decode("utf-8", errors="replace")
            right = tok.vocab[p1].decode("utf-8", errors="replace")
            merged = tok.vocab[idx].decode("utf-8", errors="replace")
            print(f"  {rank:>3}  ({p0:>4},{p1:>4}) -> {idx:<5} "
                  f"[{left}] + [{right}] = [{merged}]")
        print()

    for rank, ((p0, p1), idx) in enumerate(first_50):
        assert idx == 256 + rank, "ids must be assigned in merge order"
        assert tok.vocab[idx] == tok.vocab[p0] + tok.vocab[p1]
        # a merge can only be built from tokens that already existed
        assert p0 < idx and p1 < idx
