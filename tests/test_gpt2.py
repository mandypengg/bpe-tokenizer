"""
The correctness target: our GPT-2 encoder must agree with tiktoken exactly.

Both the vocab files and tiktoken's own data are fetched from the network on
first use, so every test here skips cleanly when offline.
"""

import pytest

from bpe import bytes_to_unicode, unicode_to_bytes
from bpe.gpt2 import GPT2Tokenizer

TEXTS = [
    "",
    "hello world",
    "Hello, world! I've got 1234 reasons — don't you think?",
    "   leading spaces and\ttabs\n\nand newlines\n",
    "héllo wörld, naïve café, ÄÖÜ",
    "🙂🙃 emoji, ZWJ family 👨‍👩‍👧‍👦, and CJK 你好世界",
    "def f(x):\n    return x ** 2  # code\n",
    "a" * 200,
    "the" * 50,
    "MiXeD CaSe WoRdS and UPPERCASE and lowercase",
    "1234567890 3.14159 1e-9 0x1F",
    "<|endoftext|>",
]


# -- byte <-> unicode mapping (no network needed) -----------------------------


def test_bytes_to_unicode_covers_all_256_bytes():
    mapping = bytes_to_unicode()
    assert len(mapping) == 256
    assert sorted(mapping) == list(range(256))


def test_bytes_to_unicode_is_reversible():
    mapping = bytes_to_unicode()
    inverse = unicode_to_bytes()
    assert len(set(mapping.values())) == 256
    assert all(inverse[ch] == b for b, ch in mapping.items())


def test_bytes_to_unicode_avoids_whitespace_and_control_characters():
    # the whole point: a rendered token never contains whitespace or controls
    for ch in bytes_to_unicode().values():
        assert not ch.isspace()
        assert ch.isprintable()


def test_printable_ascii_maps_to_itself():
    mapping = bytes_to_unicode()
    for b in range(ord("!"), ord("~") + 1):
        assert mapping[b] == chr(b)
    # space is one of the shifted bytes
    assert mapping[ord(" ")] != " "


# -- the real thing -----------------------------------------------------------


@pytest.fixture(scope="module")
def ours():
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


def test_vocab_size(ours):
    assert len(ours.vocab) == 50257
    assert ours.special_tokens == {"<|endoftext|>": 50256}


def test_merge_count(ours):
    assert len(ours.merges) == 50000


@pytest.mark.parametrize("text", TEXTS)
def test_encode_matches_tiktoken(ours, reference, text):
    assert ours.encode_ordinary(text) == reference.encode_ordinary(text)


@pytest.mark.parametrize("text", TEXTS)
def test_decode_matches_tiktoken(ours, reference, text):
    ids = reference.encode_ordinary(text)
    assert ours.decode(ids) == reference.decode(ids)


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip(ours, text):
    assert ours.decode(ours.encode_ordinary(text)) == text


def test_endoftext_is_handled_as_a_special_token(ours, reference):
    text = "hello<|endoftext|>world"
    expected = reference.encode(text, allowed_special="all")
    assert ours.encode(text, allowed_special="all") == expected
    assert 50256 in expected


def test_known_token_ids(ours):
    # spot checks that pin the byte-to-id permutation and the merge order
    assert ours.encode_ordinary("hello world") == [31373, 995]
    assert ours.encode_ordinary(" the") == [262]
    assert ours.byte_to_id[ord("!")] == 0


def test_training_is_refused(ours):
    with pytest.raises(NotImplementedError):
        ours.train("hello", 300)


def test_load_is_refused(ours):
    with pytest.raises(NotImplementedError):
        ours.load("model.model")
