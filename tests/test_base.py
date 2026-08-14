"""Tests for the primitives and the base class's bookkeeping."""

import pytest

from bpe import BasicTokenizer, get_stats, merge, render_token
from bpe.base import replace_control_characters


def test_get_stats_counts_adjacent_pairs():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_get_stats_accumulates_into_existing_dict():
    counts = {(1, 2): 5}
    get_stats([1, 2], counts)
    assert counts == {(1, 2): 6}


def test_get_stats_on_short_input():
    assert get_stats([]) == {}
    assert get_stats([1]) == {}


def test_merge_on_empty_list():
    assert merge([], (1, 2), 4) == []


def test_merge_on_single_element():
    # nothing to pair with, so the element survives untouched
    assert merge([1], (1, 2), 4) == [1]
    assert merge([1], (1, 1), 4) == [1]


def test_merge_at_sequence_start():
    assert merge([1, 2, 3], (1, 2), 4) == [4, 3]


def test_merge_at_sequence_end():
    # the i+1 lookahead must not run off the end
    assert merge([3, 1, 2], (1, 2), 4) == [3, 4]


def test_merge_replaces_every_occurrence():
    assert merge([1, 2, 3, 1, 2], (1, 2), 4) == [4, 3, 4]


def test_merge_does_not_overlap_matches():
    # [1,1,1] must consume the first two, leaving a trailing 1
    assert merge([1, 1, 1], (1, 1), 4) == [4, 1]
    assert merge([1, 1, 1, 1], (1, 1), 4) == [4, 4]


def test_merge_leaves_input_untouched_when_pair_absent():
    ids = [1, 2, 3]
    assert merge(ids, (9, 9), 4) == [1, 2, 3]
    assert ids == [1, 2, 3]  # merge must not mutate its argument


def test_render_token_escapes_control_characters():
    assert render_token(b"hi\nthere") == "hi\\u000athere"
    assert replace_control_characters("a\x00b") == "a\\u0000b"


def test_merge_ranks_follow_learned_order():
    tok = BasicTokenizer()
    tok.train("abababab", 258)
    ranks = tok.merge_ranks()
    # rank must be position in the learned sequence, not the token id
    assert sorted(ranks.values()) == [0, 1]
    assert ranks[next(iter(tok.merges))] == 0


def test_save_and_load_roundtrip(tmp_path):
    text = "the quick brown fox jumps over the lazy dog " * 5
    tok = BasicTokenizer()
    tok.train(text, 300)
    tok.register_special_tokens({"<|endoftext|>": 300})

    prefix = str(tmp_path / "model")
    tok.save(prefix)

    reloaded = BasicTokenizer()
    reloaded.load(prefix + ".model")

    assert reloaded.merges == tok.merges
    assert list(reloaded.merges) == list(tok.merges)  # order preserved
    assert reloaded.special_tokens == tok.special_tokens
    assert reloaded.vocab == tok.vocab
    assert reloaded.encode(text) == tok.encode(text)


def test_save_writes_readable_vocab_file(tmp_path):
    tok = BasicTokenizer()
    tok.train("abababab", 258)
    prefix = str(tmp_path / "model")
    tok.save(prefix)
    contents = (tmp_path / "model.vocab").read_text(encoding="utf-8")
    assert "[ab]" in contents


def test_base_class_methods_are_abstract():
    from bpe.base import Tokenizer

    tok = Tokenizer()
    with pytest.raises(NotImplementedError):
        tok.train("hi", 300)
    with pytest.raises(NotImplementedError):
        tok.encode("hi")
    with pytest.raises(NotImplementedError):
        tok.decode([1])
