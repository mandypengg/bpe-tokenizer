"""
Shared machinery for every tokenizer in this package.

Contains the two primitives the BPE algorithm is built out of (`get_stats`
and `merge`), plus the `Tokenizer` base class that owns the vocabulary,
decoding, and (de)serialization.

The single most important invariant here: **merges are ordered**. The order
in which a pair was learned is its priority, and encoding must replay merges
in exactly that order. We store them in `self.merges`, a dict keyed by pair
and valued by the resulting token id. Dicts preserve insertion order, so the
learned order survives; `merge_ranks()` exposes it explicitly for encoding.
Never rebuild `merges` from an unordered source.
"""

from __future__ import annotations

import unicodedata

# -----------------------------------------------------------------------------
# the two primitives


def get_stats(ids: list[int], counts: dict[tuple[int, int], int] | None = None):
    """
    Count how often each adjacent pair occurs in `ids`.

    Passing an existing `counts` dict accumulates into it, which lets the
    regex tokenizer tally statistics across many chunks without allocating
    a dict per chunk.

        get_stats([1, 2, 3, 1, 2]) -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    Replace every non-overlapping occurrence of `pair` in `ids` with `idx`.

        merge([1, 2, 3, 1, 2], (1, 2), 4) -> [4, 3, 4]

    Occurrences are consumed left to right, so a run like [1, 1, 1] with
    pair (1, 1) yields [idx, 1] rather than overlapping matches.
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# -----------------------------------------------------------------------------
# helpers for printing tokens without wrecking the terminal


def replace_control_characters(s: str) -> str:
    """Escape control characters (category C*) so printing a token is safe."""
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


def render_token(t: bytes) -> str:
    """Human-readable form of a token's bytes, for the .vocab debug file."""
    s = t.decode("utf-8", errors="replace")
    return replace_control_characters(s)


# -----------------------------------------------------------------------------
# base class


class Tokenizer:
    """
    Base class: owns `merges`, `vocab`, the split `pattern`, and special tokens.

    Subclasses implement `train`, `encode`, and `decode`.
    """

    def __init__(self):
        # (int, int) -> int. Insertion order IS merge order / priority.
        self.merges: dict[tuple[int, int], int] = {}
        # str, the regex used to split text before BPE ("" means no splitting)
        self.pattern: str = ""
        # str -> int, e.g. {"<|endoftext|>": 50256}
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        # int -> bytes
        self.vocab: dict[int, bytes] = self._build_vocab()

    # -- interface implemented by subclasses ---------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    # -- shared implementation -----------------------------------------------

    def merge_ranks(self) -> dict[tuple[int, int], int]:
        """
        Map each pair to its priority, lowest = merged first.

        Encoding always applies the *lowest-ranked* mergeable pair present,
        which is what makes encoding agree with training.
        """
        return {pair: rank for rank, pair in enumerate(self.merges)}

    def _byte_ids(self, text_bytes: bytes) -> list[int]:
        """
        Seed ids for a piece of text: one token per raw byte.

        For tokenizers we train ourselves, token id == byte value for the
        first 256 ids, so this is the identity. GPT-2 permutes that mapping
        and overrides this hook.
        """
        return list(text_bytes)

    def _build_vocab(self) -> dict[int, bytes]:
        """Derive int -> bytes from the 256 byte tokens plus the merges."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # replayed in insertion order, so children always precede parents
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def register_special_tokens(self, special_tokens: dict[str, int]):
        """e.g. tokenizer.register_special_tokens({"<|endoftext|>": 50256})"""
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        for special, idx in special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")

    # -- serialization --------------------------------------------------------

    def save(self, file_prefix: str):
        """
        Write `<prefix>.model` (loadable) and `<prefix>.vocab` (human-readable).

        Each merge is stored as `child0 child1 idx`. Storing the resulting id
        explicitly, rather than assuming `256 + rank`, keeps the format honest
        for vocabularies whose ids are not assigned in merge order.
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for (idx1, idx2), idx in self.merges.items():
                f.write(f"{idx1} {idx2} {idx}\n")

        vocab_file = file_prefix + ".vocab"
        inverted = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted:
                    idx0, idx1 = inverted[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file: str):
        """Inverse of `save`; reads a `.model` file only."""
        assert model_file.endswith(".model")
        merges: dict[tuple[int, int], int] = {}
        special_tokens: dict[str, int] = {}
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "bpe v1", f"unknown model version: {version!r}"
            self.pattern = f.readline().rstrip("\n")
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().rstrip("\n").rsplit(" ", 1)
                special_tokens[special] = int(special_idx)
            for line in f:
                if not line.strip():
                    continue
                idx1, idx2, idx = map(int, line.split())
                merges[(idx1, idx2)] = idx
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()
