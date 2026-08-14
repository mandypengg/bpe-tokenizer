"""Tests for the regex-split tokenizer and special-token handling."""

import pytest
import regex as re

from bpe import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, RegexTokenizer

TEXTS = [
    "",
    "hello world",
    "I've got 1234 reasons, don't you think?",
    "héllo wörld — em dash",
    "🙂🙃 emoji and CJK: 你好世界",
    "  runs   of  spaces\n\n\nand newlines\t\ttabs",
]


@pytest.mark.parametrize("pattern", [GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN])
@pytest.mark.parametrize("text", TEXTS)
def test_split_pattern_is_lossless(pattern, text):
    # the chunks must reassemble into the original text exactly
    assert "".join(re.findall(re.compile(pattern), text)) == text


def test_gpt2_pattern_splits_words_from_punctuation():
    chunks = re.findall(re.compile(GPT2_SPLIT_PATTERN), "Hello world's dog.")
    assert chunks == ["Hello", " world", "'s", " dog", "."]


def test_gpt4_pattern_caps_digit_runs_at_three():
    chunks = re.findall(re.compile(GPT4_SPLIT_PATTERN), "1234567")
    assert chunks == ["123", "456", "7"]


def test_default_pattern_is_gpt2():
    assert RegexTokenizer().pattern == GPT2_SPLIT_PATTERN


# -- whitespace ---------------------------------------------------------------
# The ` ?` in ` ?\p{L}+` binds a single space forward onto the following word;
# see the comment on GPT2_SPLIT_PATTERN. These pin that behaviour down.

WHITESPACE_TEXTS = [
    " ",
    "   ",
    "\n",
    "\t",
    " leading",
    "trailing ",
    "  leading and trailing  ",
    "repeated    interior    spaces",
    "\n\nleading newlines",
    "trailing newlines\n\n",
    "mixed \t \n whitespace",
    "\r\n\r\n",
    " word",  # NBSP: \s to the regex module, but not the ASCII space in ` ?`
    "word  word",
]


@pytest.mark.parametrize("text", WHITESPACE_TEXTS)
def test_whitespace_split_is_lossless(text):
    assert "".join(re.findall(re.compile(GPT2_SPLIT_PATTERN), text)) == text


def test_single_space_binds_to_the_following_word():
    assert re.findall(re.compile(GPT2_SPLIT_PATTERN), "a dog") == ["a", " dog"]


def test_only_one_space_is_absorbed_by_a_word():
    # three spaces before "dog" -> two spaces stand alone, one joins the word
    assert re.findall(re.compile(GPT2_SPLIT_PATTERN), "a   dog") == ["a", "  ", " dog"]


def test_trailing_space_is_its_own_chunk():
    assert re.findall(re.compile(GPT2_SPLIT_PATTERN), "dog ") == ["dog", " "]


def test_leading_space_makes_a_different_chunk_than_the_bare_word():
    tok = RegexTokenizer()
    tok.train("the dog sat. the dog ran. the dog slept. " * 20, 300)
    # " dog" and "dog" are different chunks, so they need not encode alike
    assert tok.encode(" dog") != tok.encode("dog")
    assert tok.decode(tok.encode(" dog")) == " dog"


@pytest.mark.parametrize("text", WHITESPACE_TEXTS)
def test_whitespace_roundtrips(text):
    tok = RegexTokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 10, 300)
    assert tok.decode(tok.encode(text)) == text


def test_whitespace_only_text_never_merges_into_a_word():
    tok = RegexTokenizer()
    tok.train("hello world " * 50, 300)
    # encoding "   " must produce only whitespace bytes back
    assert tok.decode(tok.encode("   ")) == "   "
    assert all(t.strip() == b"" for t in (tok.vocab[i] for i in tok.encode("   ")))


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip(text):
    tok = RegexTokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 10, 300)
    assert tok.decode(tok.encode(text)) == text


def test_merges_never_cross_a_chunk_boundary():
    tok = RegexTokenizer(GPT2_SPLIT_PATTERN)
    tok.train("dog. dog. dog. dog. dog. dog. dog. dog.", 280)
    tokens = set(tok.vocab.values())
    assert b"dog" in tokens or b" dog" in tokens
    # "dog" and "." are separate chunks, so no token may span them
    assert not any(b"g." in t for t in tokens)


def test_untrained_tokenizer_is_identity_over_bytes():
    tok = RegexTokenizer()
    text = "hello world"
    assert tok.encode(text) == list(text.encode("utf-8"))


def test_training_compresses_its_own_corpus():
    text = "the quick brown fox jumps over the lazy dog " * 10
    tok = RegexTokenizer()
    tok.train(text, 300)
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


# -- special tokens -----------------------------------------------------------


@pytest.fixture
def tok_with_specials():
    tok = RegexTokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 10, 300)
    tok.register_special_tokens({"<|endoftext|>": 300})
    return tok


def test_special_token_is_split_out_before_bpe(tok_with_specials):
    tok = tok_with_specials
    ids = tok.encode("hello<|endoftext|>world", allowed_special="all")
    assert 300 in ids
    # the surrounding text must encode exactly as it would on its own,
    # i.e. no merge reached across the special token
    split = ids.index(300)
    assert ids[:split] == tok.encode_ordinary("hello")
    assert ids[split + 1:] == tok.encode_ordinary("world")


def test_special_token_roundtrips(tok_with_specials):
    text = "hello<|endoftext|>world"
    tok = tok_with_specials
    assert tok.decode(tok.encode(text, allowed_special="all")) == text


def test_none_raise_is_the_default(tok_with_specials):
    with pytest.raises(AssertionError):
        tok_with_specials.encode("hello<|endoftext|>world")


def test_allowed_special_none_treats_it_as_ordinary_text(tok_with_specials):
    tok = tok_with_specials
    ids = tok.encode("hello<|endoftext|>world", allowed_special="none")
    assert 300 not in ids
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_allowed_special_accepts_an_explicit_set(tok_with_specials):
    tok = tok_with_specials
    tok.register_special_tokens({"<|endoftext|>": 300, "<|pad|>": 301})
    ids = tok.encode("a<|endoftext|>b<|pad|>c", allowed_special={"<|endoftext|>"})
    assert 300 in ids and 301 not in ids


def test_unknown_allowed_special_is_rejected(tok_with_specials):
    with pytest.raises(ValueError):
        tok_with_specials.encode("hi", allowed_special="sometimes")


def test_decode_rejects_unknown_ids(tok_with_specials):
    with pytest.raises(ValueError):
        tok_with_specials.decode([99999])


@pytest.mark.parametrize("text", [
    "<|endoftext|>",                       # nothing but the special token
    "<|endoftext|>trailing text",          # at the very start
    "leading text<|endoftext|>",           # at the very end
    "<|endoftext|><|endoftext|>",          # adjacent, nothing in between
    "a<|endoftext|> <|endoftext|>b",       # separated only by a space
    " <|endoftext|> ",                     # surrounded by whitespace
    "the<|endoftext|>the<|endoftext|>the", # around text that WOULD merge
])
def test_special_tokens_roundtrip_in_any_position(tok_with_specials, text):
    tok = tok_with_specials
    ids = tok.encode(text, allowed_special="all")
    assert tok.decode(ids) == text


def test_special_token_is_one_id_and_never_merged_through(tok_with_specials):
    tok = tok_with_specials
    # "the " is frequent enough in the training text to have merged into one
    # or more multi-byte tokens; the special token must still cut cleanly
    text = "the<|endoftext|>the"
    ids = tok.encode(text, allowed_special="all")
    assert ids.count(300) == 1
    split = ids.index(300)
    assert ids[:split] == tok.encode_ordinary("the")
    assert ids[split + 1:] == tok.encode_ordinary("the")


def test_adjacent_special_tokens_stay_separate(tok_with_specials):
    tok = tok_with_specials
    ids = tok.encode("<|endoftext|><|endoftext|>", allowed_special="all")
    assert ids == [300, 300]


def test_longest_special_token_wins_when_one_is_a_prefix_of_another():
    tok = RegexTokenizer()
    tok.train("hello world " * 20, 300)
    tok.register_special_tokens({"<|end|>": 300, "<|end|>of<|end|>": 301})
    ids = tok.encode("a<|end|>of<|end|>b", allowed_special="all")
    # the longer token must match, not "<|end|>" followed by literal "of..."
    assert 301 in ids and 300 not in ids
    assert tok.decode(ids) == "a<|end|>of<|end|>b"


def test_unregistered_special_looking_text_is_just_text(tok_with_specials):
    tok = tok_with_specials
    text = "not a real <|token|> here"
    assert tok.decode(tok.encode(text)) == text
