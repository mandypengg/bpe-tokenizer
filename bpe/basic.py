"""
The simplest possible BPE: byte-level, no regex splitting, no special tokens.

Merges can span any byte boundary, including spaces and punctuation, so this
will happily learn a token like " the cat". That is exactly why real
tokenizers add a split pattern (see regex.py) — but it makes the algorithm
easy to read and easy to test.
"""

from __future__ import annotations

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256, "vocab_size must leave room for the 256 bytes"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                # text collapsed to a single token; nothing left to merge
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]!r}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # a token boundary can split a multi-byte character, so decoding an
        # arbitrary id sequence must tolerate invalid utf-8
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        # no split pattern, so the whole text is one chunk: the eligible pair
        # learned earliest wins, repeatedly, until none are left
        ids = self._byte_ids(text.encode("utf-8"))
        return self._apply_merges(ids, self.merge_ranks())
