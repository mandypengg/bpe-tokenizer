"""
Full-corpus parity against tiktoken: our GPT-2 encoder must agree token for
token with `tiktoken.get_encoding("gpt2")` on all 5,000 strings in
tests/corpus.json, plus a few hundred KB of real English prose.

tests/test_gpt2.py already pins a dozen hand-picked strings. This file is the
broad sweep: every category the corpus generator produces (emoji, CJK, Arabic,
Cyrillic, source code, pathological whitespace, lone combining marks, the
empty string) run through both encoders and compared.
"""

from __future__ import annotations

import collections

import pytest

from bpe.gpt2 import GPT2Tokenizer
from tests.build_corpus import load_corpus
from tests.parity_report import MAX_REPORTS, format_mismatch, report, sweep


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def ours() -> GPT2Tokenizer:
    try:
        return GPT2Tokenizer.from_pretrained()
    except Exception as e:  # offline, or the blob store is unreachable
        pytest.skip(f"GPT-2 vocab files unavailable: {e}")


@pytest.fixture(scope="module")
def reference():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as e:
        pytest.skip(f"tiktoken gpt2 encoding unavailable: {e}")


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return load_corpus()


# -- the sweep ----------------------------------------------------------------


def test_corpus_is_fully_loaded(corpus):
    """
    Guard the sweep's denominator. Every parity test below is a loop over this
    list, so a corpus that silently shrank to 3 cases would still pass green.
    """
    assert len(corpus) == 5000
    assert len({c["text"] for c in corpus}) == 5000
    assert len({c["category"] for c in corpus}) == 13


def test_encode_parity_over_full_corpus(ours, reference, corpus):
    """encode_ordinary over all 5,000 corpus strings, specials included as text."""
    reports, by_category = sweep(
        corpus, ours.encode_ordinary, reference.encode_ordinary, ours, reference
    )
    assert not by_category, report(reports, len(corpus), by_category)


def test_decode_parity_over_full_corpus(ours, reference, corpus):
    """Decoding tiktoken's own ids must give back what tiktoken gives back."""
    reports: list[str] = []
    by_category: collections.Counter = collections.Counter()

    for i, case in enumerate(corpus):
        ids = reference.encode_ordinary(case["text"])
        got, want = ours.decode(ids), reference.decode(ids)
        if got != want:
            by_category[case["category"]] += 1
            if len(reports) < MAX_REPORTS:
                reports.append(
                    f"case #{i} ({case['category']}): ours {got!r} != theirs {want!r}"
                )
    assert not by_category, report(reports, len(corpus), by_category)


def test_roundtrip_over_full_corpus(ours, corpus):
    """decode(encode(t)) == t for every corpus string."""
    reports: list[str] = []
    by_category: collections.Counter = collections.Counter()

    for i, case in enumerate(corpus):
        text = case["text"]
        got = ours.decode(ours.encode_ordinary(text))
        if got != text:
            by_category[case["category"]] += 1
            if len(reports) < MAX_REPORTS:
                reports.append(
                    f"case #{i} ({case['category']}): {text!r} -> {got!r}"
                )
    assert not by_category, report(reports, len(corpus), by_category)


def test_special_token_parity(ours, reference, corpus):
    """
    The special_tokens category, encoded with <|endoftext|> actually recognized.

    encode_ordinary BPEs the literal text; this is the other path, where the
    token is split out before BPE runs and emitted as id 50256.
    """
    cases = [c for c in corpus if c["category"] == "special_tokens"]
    assert cases, "corpus has no special_tokens category"

    reports, by_category = sweep(
        cases,
        lambda t: ours.encode(t, allowed_special="all"),
        lambda t: reference.encode(t, allowed_special="all"),
        ours,
        reference,
    )
    assert not by_category, report(reports, len(cases), by_category)


@pytest.mark.parametrize("category", [
    "degenerate", "whitespace", "special_tokens", "emoji", "cjk", "arabic",
    "cyrillic", "code_python", "code_javascript", "code_c", "markdown",
    "english_prose", "mixed_script",
])
def test_encode_parity_by_category(ours, reference, corpus, category):
    """
    The same comparison split per category, so a regression names its pattern
    in the test id rather than in the failure text.
    """
    cases = [c for c in corpus if c["category"] == category]
    assert cases, f"corpus has no {category} category"

    reports, by_category = sweep(
        cases, ours.encode_ordinary, reference.encode_ordinary, ours, reference
    )
    assert not by_category, report(reports, len(cases), by_category)


# -- real prose ---------------------------------------------------------------


def test_encode_parity_over_english_corpus(ours, reference, english_corpus):
    """
    A few hundred KB of Sherlock Holmes, whole and in paragraphs.

    The corpus.json cases are short by design; this is the check that parity
    holds over long continuous text, where a single wrong merge early on
    would knock every later token out of alignment.
    """
    text = english_corpus[:200_000]
    got, want = ours.encode_ordinary(text), reference.encode_ordinary(text)
    if got != want:
        pytest.fail(format_mismatch("english corpus", text[:200] + "...",
                                    got, want, ours, reference))

    paragraphs = [p for p in text.split("\n\n") if p.strip()][:200]
    mismatches = []
    for i, para in enumerate(paragraphs):
        p_got = ours.encode_ordinary(para)
        p_want = reference.encode_ordinary(para)
        if p_got != p_want and len(mismatches) < MAX_REPORTS:
            mismatches.append(
                format_mismatch(f"paragraph #{i}", para, p_got, p_want, ours, reference)
            )
    assert not mismatches, "\n" + "\n".join(mismatches)
