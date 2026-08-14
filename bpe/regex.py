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
GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str | None = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
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

    # -- encoding -------------------------------------------------------------

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        ids = self._byte_ids(text_bytes)
        ranks = self.merge_ranks()
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: ranks.get(p, float("inf")))
            if pair not in ranks:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, ignoring special tokens entirely (they get BPE'd)."""
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none_raise"):
        """
        Encode text, handling special tokens per `allowed_special`:

          "all"         - every registered special token is recognized
          "none"        - none are; they are encoded as ordinary text
          "none_raise"  - none are, and their presence in `text` raises (default)
          set of str    - only these are recognized

        Special tokens are split out of the text BEFORE any BPE runs, so their
        bytes are never eligible for merging.
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens), (
                "text contains a special token; pass allowed_special to say "
                "how it should be handled"
            )
        elif isinstance(allowed_special, (set, frozenset)):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special!r} not understood")

        if not special:
            return self.encode_ordinary(text)

        # capture-group split keeps the special tokens themselves in the output
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        chunks = re.split(special_pattern, text)

        ids = []
        for part in chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
