"""
The `.tiktoken` encodings: r50k, p50k, cl100k (GPT-3.5/GPT-4) and o200k
(GPT-4o).

Two things are under test. First, merge recovery: these files publish token
bytes and ranks but not the merges, so `recover_merges` reconstructs the
merge list, and if it reconstructs the wrong one the tokenizer is silently
wrong rather than broken. Second, the usual parity sweep against tiktoken.

The sharpest check is `test_recovered_r50k_merges_match_gpt2`: r50k_base is
GPT-2's vocabulary in the new file format, so the merges we recover from it
must equal the ones vocab.bpe states outright. Recovery is validated against
ground truth, not just against its own output.
"""

from __future__ import annotations

import pytest

from bpe.gpt2 import GPT2Tokenizer
from bpe.ranks import (
    ENCODINGS,
    RanksTokenizer,
    load_mergeable_ranks,
    recover_merges,
)
from tests.build_corpus import load_corpus
from tests.parity_report import format_mismatch, report, sweep

# every published encoding this package knows how to load
ALL = sorted(ENCODINGS)
# the two the package is really here for; swept over the whole corpus
HEADLINE = ["cl100k_base", "o200k_base"]


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="session")
def tokenizers() -> dict[str, RanksTokenizer]:
    """Every encoding, loaded once for the whole session (o200k takes ~2s)."""
    loaded = {}
    for name in ALL:
        try:
            loaded[name] = RanksTokenizer.from_pretrained(name)
        except Exception as e:  # offline, or the blob store is unreachable
            pytest.skip(f"{name} unavailable: {e}")
    return loaded


@pytest.fixture(scope="session")
def references() -> dict:
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return {name: tiktoken.get_encoding(name) for name in ALL}
    except Exception as e:
        pytest.skip(f"tiktoken encodings unavailable: {e}")


@pytest.fixture(scope="session")
def corpus() -> list[dict]:
    return load_corpus()


# -- merge recovery on a vocabulary small enough to check by hand --------------


def toy_ranks() -> dict[bytes, int]:
    """
    A hand-built rank table over the alphabet {a, b}, ranks 0 and 1 for the
    base tokens and then three merges: ab, then abab, then ababb.
    """
    return {b"a": 0, b"b": 1, b"ab": 2, b"abab": 3, b"ababb": 4}


def test_recover_merges_on_a_toy_vocabulary():
    assert recover_merges(toy_ranks()) == {
        (0, 1): 2,  # a + b -> ab
        (2, 2): 3,  # ab + ab -> abab
        (3, 1): 4,  # abab + b -> ababb
    }


def test_recovered_merges_are_in_rank_order():
    merges = recover_merges(toy_ranks())
    assert list(merges.values()) == sorted(merges.values())


def test_base_tokens_produce_no_merges():
    assert recover_merges({b"a": 0, b"b": 1}) == {}


def test_unreachable_token_is_rejected():
    """A token whose halves aren't both in the vocabulary can't have been BPE'd."""
    with pytest.raises(AssertionError, match="not reachable by BPE"):
        recover_merges({b"a": 0, b"b": 1, b"c": 2, b"abc": 3})


def test_recovery_respects_rank_order_not_just_membership():
    """
    `ab` exists but ranks *after* `ba`, so "bab" must recover as (ba, b) and
    not (b, ab). Ignoring the rank cutoff would pick the wrong pair here.
    """
    ranks = {b"a": 0, b"b": 1, b"ba": 2, b"bab": 3, b"ab": 4}
    assert recover_merges(ranks)[(2, 1)] == 3


# -- recovery against ground truth --------------------------------------------


def test_recovered_r50k_merges_match_gpt2(tokenizers):
    """
    r50k_base is GPT-2's vocabulary in the `.tiktoken` format. The merges we
    recover from it must be exactly the ones vocab.bpe lists, in the same
    order: same pairs, same resulting ids, same priority.
    """
    try:
        gpt2 = GPT2Tokenizer.from_pretrained()
    except Exception as e:
        pytest.skip(f"GPT-2 vocab files unavailable: {e}")

    recovered = tokenizers["r50k_base"].merges
    assert recovered == gpt2.merges
    assert list(recovered) == list(gpt2.merges), "merge order differs"
    assert tokenizers["r50k_base"].byte_to_id == gpt2.byte_to_id


@pytest.mark.parametrize("name", ALL)
def test_merges_rebuild_the_published_vocabulary(name, tokenizers, references):
    """Every token's bytes must come back out of the recovered merge tree."""
    ours = tokenizers[name]
    theirs = references[name]._mergeable_ranks
    special_ids = set(ours.special_tokens.values())
    assert {token: idx for idx, token in ours.vocab.items()
            if idx not in special_ids} == theirs


@pytest.mark.parametrize("name", ALL)
def test_vocabulary_size_matches_tiktoken(name, tokenizers, references):
    assert max(tokenizers[name].vocab) == references[name].max_token_value


# -- parity -------------------------------------------------------------------


@pytest.mark.parametrize("name", HEADLINE)
def test_encode_parity_over_full_corpus(name, tokenizers, references, corpus):
    reports, by_category = sweep(
        corpus,
        tokenizers[name].encode_ordinary,
        references[name].encode_ordinary,
        tokenizers[name],
        references[name],
    )
    assert not by_category, report(reports, len(corpus), by_category)


@pytest.mark.parametrize("name", ALL)
def test_encode_parity_over_a_corpus_sample(name, tokenizers, references, corpus):
    """Every encoding, over one case from each category plus the awkward ones."""
    seen, cases = set(), []
    for case in corpus:
        if case["category"] not in seen:
            seen.add(case["category"])
            cases.append(case)
    cases += [c for c in corpus if c["category"] in ("degenerate", "whitespace")][:200]

    reports, by_category = sweep(
        cases,
        tokenizers[name].encode_ordinary,
        references[name].encode_ordinary,
        tokenizers[name],
        references[name],
    )
    assert not by_category, report(reports, len(cases), by_category)


@pytest.mark.parametrize("name", ALL)
def test_encode_parity_over_english_prose(name, tokenizers, references, english_corpus):
    """
    Long continuous text, where one wrong merge early knocks every later
    token out of alignment.
    """
    text = english_corpus[:100_000]
    got = tokenizers[name].encode_ordinary(text)
    want = references[name].encode_ordinary(text)
    if got != want:
        pytest.fail(format_mismatch(f"{name}: english corpus", text[:200] + "...",
                                    got, want, tokenizers[name], references[name]))


@pytest.mark.parametrize("name", ALL)
def test_roundtrip_over_corpus(name, tokenizers, corpus):
    tok = tokenizers[name]
    bad = [c["text"] for c in corpus
           if tok.decode(tok.encode_ordinary(c["text"])) != c["text"]]
    assert not bad, f"{len(bad)} strings failed to roundtrip, e.g. {bad[:3]!r}"


# -- special tokens -----------------------------------------------------------


def test_cl100k_special_tokens(tokenizers):
    assert tokenizers["cl100k_base"].special_tokens == {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    }


@pytest.mark.parametrize("name", ALL)
def test_special_tokens_match_tiktoken(name, tokenizers, references):
    assert tokenizers[name].special_tokens == references[name]._special_tokens


@pytest.mark.parametrize("name", HEADLINE)
def test_special_token_parity(name, tokenizers, references, corpus):
    """<|endoftext|> and friends split out before BPE, as tiktoken does."""
    cases = [c for c in corpus if c["category"] == "special_tokens"]
    reports, by_category = sweep(
        cases,
        lambda t: tokenizers[name].encode(t, allowed_special="all"),
        lambda t: references[name].encode(t, allowed_special="all"),
        tokenizers[name],
        references[name],
    )
    assert not by_category, report(reports, len(cases), by_category)


def test_fim_tokens_are_single_ids(tokenizers):
    tok = tokenizers["cl100k_base"]
    text = "<|fim_prefix|>def f(<|fim_suffix|>):<|fim_middle|>"
    assert tok.encode(text, allowed_special="all")[0] == 100258
    assert tok.decode(tok.encode(text, allowed_special="all")) == text


# -- the loader's edges -------------------------------------------------------


def test_unknown_encoding_name_is_rejected():
    with pytest.raises(ValueError, match="unknown encoding"):
        RanksTokenizer.from_pretrained("gpt5_base")


def test_no_download_and_no_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RanksTokenizer.from_pretrained("cl100k_base", cache_dir=tmp_path, download=False)


def test_load_from_cache_without_downloading(tokenizers):
    """The cache the session fixture warmed is enough on its own."""
    tok = RanksTokenizer.from_pretrained("cl100k_base", download=False)
    assert tok.encode_ordinary("hello world") == \
        tokenizers["cl100k_base"].encode_ordinary("hello world")


def test_training_is_refused(tokenizers):
    with pytest.raises(NotImplementedError, match="pretrained"):
        tokenizers["cl100k_base"].train("text", 300)


def test_load_is_refused(tokenizers):
    with pytest.raises(NotImplementedError, match=".model"):
        tokenizers["cl100k_base"].load("x.model")


def test_load_mergeable_ranks_tolerates_a_trailing_newline(tmp_path):
    import base64

    path = tmp_path / "toy.tiktoken"
    path.write_bytes(b"\n".join(
        base64.b64encode(t) + b" " + str(r).encode() for t, r in toy_ranks().items()
    ) + b"\n\n")
    assert load_mergeable_ranks(path) == toy_ranks()


# -- refusing special tokens the way tiktoken does ----------------------------


DISALLOWED_CASES = [
    # (text, allowed_special) -> both must raise, or neither, and agree on ids
    ("plain text with no specials at all", "none_raise"),
    ("a<|endoftext|>b", "none_raise"),
    ("a<|endoftext|>b", "all"),
    ("a<|endoftext|>b", frozenset({"<|endoftext|>"})),
    ("a<|endoftext|>b<|fim_prefix|>c", frozenset({"<|endoftext|>"})),
    ("a<|fim_prefix|>b", frozenset({"<|endoftext|>"})),
    ("<|endofprompt|>", frozenset({"<|endoftext|>"})),
    ("<|endoftext|><|endoftext|>", "all"),
]


@pytest.mark.parametrize("text,allowed", DISALLOWED_CASES)
def test_disallowed_specials_agree_with_tiktoken(text, allowed, tokenizers, references):
    """
    Allowing one special token is not consent to silently BPE another. Both
    implementations must refuse the same inputs, and agree on the rest.
    """
    ours, theirs = tokenizers["cl100k_base"], references["cl100k_base"]
    allowed_them = allowed if isinstance(allowed, frozenset) else (
        "all" if allowed == "all" else set()
    )

    def run(fn, arg):
        try:
            return fn(text, allowed_special=arg)
        except ValueError:
            return "raised"

    assert run(ours.encode, allowed) == run(theirs.encode, allowed_them)


def test_disallowed_check_scales_to_a_large_special_vocabulary(tokenizers):
    """
    The check used to be a substring scan per special token. o200k_harmony
    registers over a thousand of them, so that shape costs more than the
    encoding it guards; this runs one alternation instead.
    """
    import time

    tok = tokenizers["o200k_base"]
    original = dict(tok.special_tokens)
    try:
        tok.register_special_tokens(
            {**original, **{f"<|reserved_{i}|>": 200013 + i for i in range(1075)}}
        )
        text = "the quick brown fox jumps over the lazy dog " * 500

        start = time.perf_counter()
        for _ in range(5):
            tok.encode(text)
        assert time.perf_counter() - start < 5.0
    finally:
        # the fixture is session-scoped; leaving 1,075 extra specials
        # registered would break every test that runs after this one
        tok.register_special_tokens(original)
