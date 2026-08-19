"""
The caches encoding hangs off the merge list: the pair-priority table and the
per-chunk encodings.

Both are pure speed, so the thing worth testing is that they are invisible.
Every test here asserts the same answer comes back cold, warm, and after the
vocabulary underneath has been replaced. A stale chunk cache is the dangerous
failure: it would keep returning a correct-looking token sequence from a
vocabulary that no longer exists.
"""

from __future__ import annotations

from bpe import BasicTokenizer, RegexTokenizer

TEXT = "the cat sat on the mat, the cat sat again " * 20


def trained(vocab_size: int = 320, text: str = TEXT) -> RegexTokenizer:
    tok = RegexTokenizer()
    tok.train(text, vocab_size)
    return tok


# -- the caches are invisible -------------------------------------------------


def test_warm_cache_encodes_the_same_as_cold():
    tok = trained()
    cold = tok.encode_ordinary(TEXT)
    assert tok.encode_ordinary(TEXT) == cold


def test_reset_cache_changes_nothing():
    tok = trained()
    before = tok.encode_ordinary(TEXT)
    tok.reset_cache()
    assert tok.encode_ordinary(TEXT) == before


def test_cache_is_actually_used():
    tok = trained()
    tok.encode_ordinary(TEXT)
    assert tok._chunk_cache, "encoding populated no chunks"
    assert b" cat" in tok._chunk_cache


def test_encoding_matches_the_uncached_path():
    """`_encode_chunk` bypasses the cache, so it is an independent answer."""
    tok = trained()
    ids = tok.encode_ordinary("the cat sat")
    direct = []
    import regex as re

    for chunk in re.findall(tok.compiled_pattern, "the cat sat"):
        direct.extend(tok._encode_chunk(chunk.encode("utf-8")))
    assert ids == direct


# -- invalidation -------------------------------------------------------------


def test_retraining_invalidates_the_chunk_cache():
    tok = trained(vocab_size=320)
    first = tok.encode_ordinary(TEXT)
    tok.train(TEXT, 260)  # a much smaller vocabulary: fewer merges available
    second = tok.encode_ordinary(TEXT)
    assert second != first, "retraining should change the encoding"
    assert len(second) > len(first)


def test_retraining_invalidates_the_rank_table():
    tok = trained(vocab_size=320)
    tok.encode_ordinary(TEXT)
    tok.train(TEXT, 300)
    assert tok.merge_ranks() == {pair: i for i, pair in enumerate(tok.merges)}


def test_loading_a_model_invalidates_the_caches(tmp_path):
    tok = trained(vocab_size=320)
    tok.encode_ordinary(TEXT)
    other = trained(vocab_size=280)
    other.save(str(tmp_path / "small"))

    tok.load(str(tmp_path / "small.model"))
    assert tok.encode_ordinary(TEXT) == other.encode_ordinary(TEXT)


def test_assigning_merges_directly_invalidates():
    tok = trained()
    tok.encode_ordinary(TEXT)
    tok.merges = {}
    assert tok.merge_ranks() == {}
    assert tok._chunk_cache == {}
    # with no merges left, encoding is one token per byte
    assert tok.encode_ordinary("cat") == list(b"cat")


# -- the size cap -------------------------------------------------------------


def test_cache_is_capped(monkeypatch):
    tok = trained()
    monkeypatch.setattr(type(tok), "CHUNK_CACHE_MAX", 8)
    text = " ".join(f"word{i}" for i in range(200))
    ids = tok.encode_ordinary(text)
    assert len(tok._chunk_cache) <= 8
    # and the answer is unaffected by all that churn
    tok.reset_cache()
    assert tok.encode_ordinary(text) == ids


# -- the same holds without a split pattern -----------------------------------


def test_basic_tokenizer_shares_the_rank_cache():
    tok = BasicTokenizer()
    tok.train(TEXT, 300)
    first = tok.encode(TEXT)
    assert tok.encode(TEXT) == first
    tok.train(TEXT, 260)
    assert tok.encode(TEXT) != first


# -- long chunks --------------------------------------------------------------


def test_long_runs_do_not_go_quadratic():
    """
    A guard on the shape of the merge loop, not on absolute speed.

    Each round merges every occurrence of the winning pair, so a long run of
    one character costs a round per distinct pair. Merging a single
    occurrence per round instead is equally correct and around a thousand
    times slower here, which is exactly the kind of regression that hides
    behind a prose benchmark. The budget is deliberately loose: real runs
    take a few milliseconds, the broken shape takes minutes.
    """
    import time

    from bpe.gpt2 import GPT2Tokenizer

    try:
        tok = GPT2Tokenizer.from_pretrained()
    except Exception as e:
        import pytest

        pytest.skip(f"GPT-2 vocab files unavailable: {e}")

    start = time.perf_counter()
    for text in (" " * 10_000, "a" * 10_000, "\n" * 5_000, "!" * 5_000):
        tok.reset_cache()
        assert tok.decode(tok.encode_ordinary(text)) == text
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"long runs took {elapsed:.1f}s; the merge loop regressed"
