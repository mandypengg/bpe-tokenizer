"""
Full-corpus parity against tiktoken: our GPT-2 encoder must agree token for
token with `tiktoken.get_encoding("gpt2")` on all 5,000 strings in
tests/corpus.json, plus a few hundred KB of real English prose.

tests/test_gpt2.py already pins a dozen hand-picked strings. This file is the
broad sweep: every category the corpus generator produces (emoji, CJK, Arabic,
Cyrillic, source code, pathological whitespace, lone combining marks, the
empty string) run through both encoders and compared.

Failures report the input, both token sequences windowed around the problem,
and the first index where they diverge, because "3,412 mismatches" is not a
bug report and "index 4 of ' café': ours 269 (b' ca') vs theirs 40304
(b' caf')" is.
"""

from __future__ import annotations

import collections

import pytest

from bpe.gpt2 import GPT2Tokenizer
from tests.build_corpus import load_corpus

# how many detailed mismatch reports to print before summarizing the rest
MAX_REPORTS = 20
# tokens of context to show either side of the first divergence
WINDOW = 6


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


# -- failure reporting --------------------------------------------------------


def first_divergence(ours_ids: list[int], theirs: list[int]) -> int:
    """Index of the first differing token, or the length of the shorter list."""
    for i, (a, b) in enumerate(zip(ours_ids, theirs)):
        if a != b:
            return i
    return min(len(ours_ids), len(theirs))


def _token_repr(tokenizer, ids: list[int], i: int) -> str:
    if i >= len(ids):
        return "<past end>"
    idx = ids[i]
    try:
        raw = tokenizer.decode_single_token_bytes(idx)  # tiktoken
    except AttributeError:
        raw = tokenizer.vocab[idx]
    return f"{idx} ({raw!r})"


def _seq(ids: list[int], limit: int = 40) -> str:
    body = ", ".join(str(i) for i in ids[:limit])
    if len(ids) > limit:
        body += f", ... +{len(ids) - limit} more"
    return f"[{body}]  ({len(ids)} tokens)"


def format_mismatch(label, text, ours_ids, theirs, ours_tok, ref_tok) -> str:
    """Full report for one disagreement: input, both sequences, divergence point."""
    i = first_divergence(ours_ids, theirs)
    lo, hi = max(0, i - WINDOW), i + WINDOW + 1

    lines = [
        f"{label}",
        f"  text      : {text!r}",
        f"  utf-8     : {text.encode('utf-8')!r}",
        f"  ours      : {_seq(ours_ids)}",
        f"  theirs    : {_seq(theirs)}",
        f"  diverge at index {i}:",
        f"    ours  [{i}] = {_token_repr(ours_tok, ours_ids, i)}",
        f"    theirs[{i}] = {_token_repr(ref_tok, theirs, i)}",
    ]

    if i > 0:
        agreed = ours_tok.decode(ours_ids[:i])
        lines.append(f"  agreed prefix decodes to {agreed!r}")

    lines.append(f"  window [{lo}:{hi}]")
    for name, ids, tok in (("ours", ours_ids, ours_tok), ("theirs", theirs, ref_tok)):
        window = [
            ("->" if j == i else "  ") + _token_repr(tok, ids, j)
            for j in range(lo, min(hi, len(ids)))
        ]
        lines.append(f"    {name:<7}: " + "  ".join(window))
    return "\n".join(lines)


def report(reports: list[str], total: int, by_category: dict) -> str:
    """
    Assembled failure message: per-case detail, then a category breakdown.

    `reports` is only the capped sample that got formatted; the real mismatch
    count comes from `by_category`, which counts every case.
    """
    failed = sum(by_category.values())
    parts = [f"\n{failed} of {total} cases disagree with tiktoken.\n"]
    parts.extend(reports)
    if failed > len(reports):
        parts.append(f"\n... and {failed - len(reports)} more not shown.")
    if by_category:
        parts.append("\nmismatches by category (failed / total in category):")
        for name, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
            parts.append(f"  {name:<18} {count}")
    return "\n".join(parts)


# -- the sweep ----------------------------------------------------------------


def test_corpus_is_fully_loaded(corpus):
    """
    Guard the sweep's denominator. Every parity test below is a loop over this
    list, so a corpus that silently shrank to 3 cases would still pass green.
    """
    assert len(corpus) == 5000
    assert len({c["text"] for c in corpus}) == 5000
    assert len({c["category"] for c in corpus}) == 13


def sweep(cases, encode_ours, encode_theirs, ours_tok, ref_tok):
    """
    Run both encoders over `cases`, counting every disagreement but formatting
    only the first MAX_REPORTS of them.

    Returns (reports, by_category). Counting and formatting are deliberately
    separate: the headline number has to reflect all 5,000 cases, not just the
    handful small enough to print.
    """
    reports: list[str] = []
    by_category: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()

    for i, case in enumerate(cases):
        text, category = case["text"], case["category"]
        seen[category] += 1
        got, want = encode_ours(text), encode_theirs(text)
        if got != want:
            by_category[category] += 1
            if len(reports) < MAX_REPORTS:
                label = f"case #{i} ({category} #{seen[category]})"
                reports.append(
                    format_mismatch(label, text, got, want, ours_tok, ref_tok)
                )
    return reports, by_category


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
