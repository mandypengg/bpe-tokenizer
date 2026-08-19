"""
BPE with a regex split pattern in front of it, the way GPT-2 and GPT-4 do it.

Text is first chopped into chunks (words, runs of digits, runs of punctuation,
whitespace) and BPE runs *within* each chunk independently. Merges can never
cross a chunk boundary, which is what stops the tokenizer from learning
tokens like " dog." or " the cat".

Uses the third-party `regex` module, not stdlib `re`: the split patterns need
Unicode property escapes (\\p{L}, \\p{N}) and possessive quantifiers, which
`re` does not support.
"""

from __future__ import annotations

import regex as re

from .base import Tokenizer, get_stats, merge

# the two patterns in the wild. GPT-4's differs in that it case-folds the
# contraction list, caps number runs at 3 digits, and treats newlines specially.
#
# The ` ?` in ` ?\p{L}+` is the whole reason GPT-2 tokens look the way they do.
# A word chunk may swallow ONE leading space, so " dog" is a single chunk rather
# than " " + "dog". Three consequences for token boundaries:
#
#   1. Space binds forward, to the word that follows it, never backward. No
#      token ever ends in a space (in a trained vocab), and mid-sentence tokens
#      carry their space as a prefix.
#   2. A word has two distinct token identities depending on what precedes it.
#      "dog" at the start of a line and " dog" after a space are different
#      chunks, so BPE learns them as different tokens with different ids.
#   3. Only one space is absorbed. Given "  dog", ` ?\p{L}+` cannot match at the
#      first space (the optional space is followed by another space, not a
#      letter), so `\s+(?!\S)` takes the extra space as its own chunk and the
#      word chunk is " dog". A run of n spaces before a word splits as n-1
#      spaces + " word".
GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# o200k_base (GPT-4o and later). Two letter alternatives instead of one, so
# that a capitalized word splits as one chunk rather than shedding its first
# letter, and the contraction suffix is matched as part of the word chunk
# rather than as its own alternative.
O200K_SPLIT_PATTERN = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str | None = None):
        super().__init__()
        # GPT-2's pattern is the default: this package's correctness target is
        # GPT-2, and GPT2Tokenizer inherits from here.
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256, "vocab_size must leave room for the 256 bytes"
        num_merges = vocab_size - 256

        # split first, then work per chunk for the rest of training
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for chunk_ids in ids:
                # accumulate across chunks into one tally
                get_stats(chunk_ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]!r}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    # -- decoding -------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        parts = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                parts.append(self.vocab[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- which special tokens are in play -------------------------------------

    def _allowed_specials(self, allowed_special) -> dict[str, int]:
        """The special tokens to split out of the text, per `allowed_special`."""
        if allowed_special == "all":
            return self.special_tokens
        if allowed_special in ("none", "none_raise"):
            return {}
        if isinstance(allowed_special, (set, frozenset, list, tuple)):
            unknown = set(allowed_special) - set(self.special_tokens)
            if unknown:
                raise ValueError(f"not registered special tokens: {sorted(unknown)}")
            return {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        raise ValueError(f"allowed_special={allowed_special!r} not understood")

    def _disallowed_specials(self, allowed_special, disallowed_special,
                             special: dict[str, int]) -> set[str]:
        """
        The special tokens whose mere presence in the text is an error.

        Defaults follow `allowed_special`: naming some tokens means the rest
        are refused, while "none" asks for them to be treated as plain text
        and so refuses nothing.
        """
        if disallowed_special is None:
            disallowed_special = "none" if allowed_special == "none" else "all"
        if disallowed_special == "all":
            return set(self.special_tokens) - set(special)
        if disallowed_special == "none":
            return set()
        if isinstance(disallowed_special, (set, frozenset, list, tuple)):
            return set(disallowed_special)
        raise ValueError(f"disallowed_special={disallowed_special!r} not understood")

    def _reject_disallowed(self, text: str, forbidden: set[str]):
        """
        Raise if any forbidden special token appears in `text`.

        One alternation rather than a substring scan per token: with 1,000+
        special tokens (o200k_harmony has that many) the naive loop costs more
        than the encoding it guards. This was an `assert`, which `python -O`
        strips, taking the check with it.
        """
        pattern = "|".join(re.escape(token) for token in sorted(forbidden))
        match = re.search(pattern, text)
        if match is None:
            return
        raise ValueError(
            f"text contains the special token {match.group()!r}, which is "
            f"disallowed. Pass allowed_special={{{match.group()!r}, ...}} to "
            f"encode it as one token, allowed_special=\"none\" to encode it as "
            f"ordinary text, or disallowed_special=\"none\" to ignore this."
        )

    # -- encoding -------------------------------------------------------------

    def _encode_chunk(self, text_bytes: bytes,
                      ranks: dict[tuple[int, int], int] | None = None) -> list[int]:
        # callers looping over chunks pass `ranks` in; rebuilding it per chunk
        # is O(chunks x merges), which dominates encoding on a 50k-merge vocab
        ranks = self.merge_ranks() if ranks is None else ranks
        return self._apply_merges(self._byte_ids(text_bytes), ranks)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, ignoring special tokens entirely (they get BPE'd)."""
        ranks = self.merge_ranks()
        cache = self._chunk_cache
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            # each chunk is BPE'd on its own and the results concatenated, so
            # no merge can ever reach across a chunk boundary
            piece = chunk.encode("utf-8")
            encoded = cache.get(piece)
            if encoded is None:
                encoded = tuple(self._encode_chunk(piece, ranks))
                # the split pattern makes chunks words, so the same handful
                # recur constantly and this hits on most of a real document.
                # Full: drop everything rather than let one adversarial
                # document's chunks lock the common words out for good.
                if len(cache) >= self.CHUNK_CACHE_MAX:
                    cache.clear()
                cache[piece] = encoded
            ids.extend(encoded)
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none_raise",
               disallowed_special: str | set[str] | None = None):
        """
        Encode text, handling special tokens per `allowed_special`:

          "all"         - every registered special token is recognized
          "none"        - none are; they are encoded as ordinary text
          "none_raise"  - none are, and their presence in `text` raises (default)
          set of str    - only these are recognized, and any *other* special
                          token appearing in `text` raises

        `disallowed_special` overrides which tokens raise: "all" for every
        special token not allowed, or an explicit set. This mirrors tiktoken,
        where encoding a control token that arrived in untrusted text is a
        mistake worth an exception rather than a silent fallback to BPE.

        Special tokens are split out of the text BEFORE any BPE runs, so their
        bytes are never eligible for merging.
        """
        special = self._allowed_specials(allowed_special)
        forbidden = self._disallowed_specials(allowed_special, disallowed_special, special)
        if forbidden:
            self._reject_disallowed(text, forbidden)

        if not special:
            return self.encode_ordinary(text)

        # capture-group split keeps the special tokens themselves in the output.
        # longest first, so that when one special token is a prefix of another
        # ("<|end|>" vs "<|endoftext|>") alternation picks the longer one.
        ordered = sorted(special, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(k) for k in ordered) + ")"
        chunks = re.split(special_pattern, text)

        ids = []
        for part in chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
