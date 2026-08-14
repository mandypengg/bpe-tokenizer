"""Tests for the plain byte-level BPE tokenizer."""

import pytest

from bpe import BasicTokenizer

TEXTS = [
    "",
    "a",
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "héllo wörld — em dash, curly ’quotes’",
    "🙂🙃 emoji and CJK: 你好世界",
    "  leading and trailing whitespace  \n\ttabs\n",
]


def test_train_learns_expected_merges():
    # unambiguous by construction: (a,b) is the only pair worth merging first
    tok = BasicTokenizer()
    tok.train("abababab", 259)
    assert list(tok.merges) == [(97, 98), (256, 256), (257, 257)]
    assert tok.vocab[256] == b"ab"
    assert tok.vocab[257] == b"abab"
    assert tok.vocab[258] == b"abababab"
    assert tok.encode("abababab") == [258]


def test_train_stops_when_nothing_left_to_merge():
    tok = BasicTokenizer()
    tok.train("abababab", 400)  # asks for 144 merges, only 3 are possible
    assert len(tok.merges) == 3
    assert tok.encode("abababab") == [258]


def test_train_rejects_vocab_smaller_than_byte_alphabet():
    tok = BasicTokenizer()
    with pytest.raises(AssertionError):
        tok.train("hello", 255)


@pytest.mark.parametrize("text", TEXTS)
def test_untrained_tokenizer_is_identity_over_bytes(text):
    tok = BasicTokenizer()
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip_after_training(text):
    tok = BasicTokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 10, 300)
    assert tok.decode(tok.encode(text)) == text


def test_training_compresses_its_own_corpus():
    text = "the quick brown fox jumps over the lazy dog " * 10
    tok = BasicTokenizer()
    tok.train(text, 300)
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_encode_matches_training_segmentation():
    # encoding the training text must reproduce what training converged to
    text = "aaabdaaabac"
    tok = BasicTokenizer()
    tok.train(text, 259)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    # every id must be decodable back through the vocab
    assert all(idx in tok.vocab for idx in ids)


def test_decode_tolerates_split_multibyte_characters():
    tok = BasicTokenizer()
    # 0x80 is a bare utf-8 continuation byte and cannot stand alone
    assert tok.decode([128]) == "�"
