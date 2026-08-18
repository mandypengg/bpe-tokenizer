"""
Tokenizers loaded from OpenAI's `.tiktoken` rank files: r50k, p50k, cl100k
(GPT-3.5/GPT-4) and o200k (GPT-4o).

These files have a different shape from GPT-2's vocab.bpe, and the difference
is the whole problem this module solves.

`vocab.bpe` lists the merges: one line per merge, in learned order, naming the
two halves. `RegexTokenizer` encodes by replaying exactly that list, so GPT-2
loads in a dozen lines.

A `.tiktoken` file lists only the *result*: base64 token bytes and a rank, with
no record of which two tokens were joined to make it. tiktoken doesn't need
that record, because it merges by looking up the concatenation of an adjacent
pair directly in the rank table. Our encoder is built on `merges`, keyed by a
pair of ids. So we reconstruct the merge list at load time.

## Recovering the merges

The rank IS the merge order: rank 256 was the first pair learned, 257 the
second, and so on. So a token of rank r was assembled out of tokens that all
have rank < r, and at the moment its merge fired, its bytes stood in exactly
the state that replaying every merge below rank r produces. Replay that and
you are left with two pieces: the pair that made it.

    "attention" (say rank 9000) -> replay every merge below 9000 over
    a-t-t-e-n-t-i-o-n, which leaves ["atten", "tion"], so the merge is
    (id["atten"], id["tion"]) -> 9000.

Every composite token in every encoding here decomposes this way, which is not
luck: it is what it means for the vocabulary to have been produced by BPE. We
assert it rather than trusting it, since a vocabulary that failed would encode
subtly wrong rather than loudly wrong.

## The tradeoff

Recovery costs a pass over the vocabulary at load, about a second for
cl100k. Encoding straight off the rank table the way tiktoken does would skip
it. We pay the second to keep one encoder for every tokenizer in the package,
and to keep the "merges are ordered, the order is the priority" invariant true
everywhere rather than in most places.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path

from .download import DATA_DIR, download_file
from .regex import (
    GPT2_SPLIT_PATTERN,
    GPT4_SPLIT_PATTERN,
    O200K_SPLIT_PATTERN,
    RegexTokenizer,
)

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

DEFAULT_CACHE_DIR = DATA_DIR / "openai"
BASE_URL = "https://openaipublic.blob.core.windows.net/encodings"


@dataclass(frozen=True)
class Encoding:
    """Everything needed to build one published encoding."""

    name: str
    sha256: str
    pattern: str
    special_tokens: dict[str, int] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.name}.tiktoken"

    @property
    def filename(self) -> str:
        return f"{self.name}.tiktoken"


# r50k_base is GPT-2's vocabulary in the newer file format, which makes it a
# free cross-check: merges recovered from it must equal the ones vocab.bpe
# states outright (tests/test_ranks.py asserts exactly that).
ENCODINGS: dict[str, Encoding] = {
    "r50k_base": Encoding(
        name="r50k_base",
        sha256="306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        pattern=GPT2_SPLIT_PATTERN,
        special_tokens={ENDOFTEXT: 50256},
    ),
    "p50k_base": Encoding(
        name="p50k_base",
        sha256="94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        pattern=GPT2_SPLIT_PATTERN,
        special_tokens={ENDOFTEXT: 50256},
    ),
    "cl100k_base": Encoding(
        name="cl100k_base",
        sha256="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        pattern=GPT4_SPLIT_PATTERN,
        special_tokens={
            ENDOFTEXT: 100257,
            FIM_PREFIX: 100258,
            FIM_MIDDLE: 100259,
            FIM_SUFFIX: 100260,
            ENDOFPROMPT: 100276,
        },
    ),
    "o200k_base": Encoding(
        name="o200k_base",
        sha256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        pattern=O200K_SPLIT_PATTERN,
        special_tokens={ENDOFTEXT: 199999, ENDOFPROMPT: 200018},
    ),
}


# -- loading ------------------------------------------------------------------


def download_encoding(name: str, cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR) -> Path:
    """Fetch one `.tiktoken` file into `cache_dir` if it isn't already there."""
    encoding = ENCODINGS[name]
    return download_file(
        encoding.url,
        Path(cache_dir) / encoding.filename,
        expected_sha256=encoding.sha256,
    )


def load_mergeable_ranks(path: str | os.PathLike) -> dict[bytes, int]:
    """
    Parse a `.tiktoken` file into token bytes -> rank.

    One token per line, `<base64 of the token bytes> <rank>`. Blank lines are
    tolerated because the files end with a newline.
    """
    ranks: dict[bytes, int] = {}
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


# -- merge recovery -----------------------------------------------------------


def _split_token(ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    """
    Replay BPE over `token`'s bytes using only merges below `max_rank`.

    The same greedy loop the encoder runs, but over byte strings rather than
    ids, since ids are what we are in the middle of working out.
    """
    parts = [bytes([b]) for b in token]
    while len(parts) > 1:
        best_at, best_rank = None, max_rank
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < best_rank:
                best_at, best_rank = i, rank
        if best_at is None:
            break
        parts[best_at:best_at + 2] = [parts[best_at] + parts[best_at + 1]]
    return parts


def recover_merges(ranks: dict[bytes, int]) -> dict[tuple[int, int], int]:
    """
    Reconstruct the ordered merge list a `.tiktoken` rank table implies.

    Returned in increasing rank order, so the dict's insertion order is the
    merge priority, exactly as everywhere else in this package.
    """
    merges: dict[tuple[int, int], int] = {}
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) == 1:
            continue  # a base token; nothing was merged to make it
        parts = _split_token(ranks, token, rank)
        assert len(parts) == 2, (
            f"token {token!r} (rank {rank}) replays to {len(parts)} pieces, "
            f"not 2: {parts!r}. The vocabulary is not reachable by BPE."
        )
        merges[(ranks[parts[0]], ranks[parts[1]])] = rank
    return merges


class RanksTokenizer(RegexTokenizer):
    """
    A tokenizer built from a `.tiktoken` rank file.

        tok = RanksTokenizer.from_pretrained("cl100k_base")
        tok.encode("hello world")

    Training is not supported; these vocabularies are OpenAI's, and the point
    is to reproduce them.
    """

    def __init__(self, pattern: str | None = None):
        super().__init__(pattern=pattern)
        self.name: str | None = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_ranks(cls, mergeable_ranks: dict[bytes, int], pattern: str,
                   special_tokens: dict[str, int] | None = None):
        """Build from an already-parsed rank table."""
        self = cls(pattern=pattern)

        base = {token: rank for token, rank in mergeable_ranks.items() if len(token) == 1}
        assert len(base) == 256, f"expected 256 single-byte tokens, found {len(base)}"
        self.byte_to_id = {token[0]: rank for token, rank in base.items()}

        self.merges = recover_merges(mergeable_ranks)
        self.vocab = self._build_vocab()

        rebuilt = {token: idx for idx, token in self.vocab.items()}
        assert rebuilt == mergeable_ranks, (
            "recovered merges do not rebuild the published vocabulary"
        )

        if special_tokens:
            self.register_special_tokens(dict(special_tokens))
        return self

    @classmethod
    def from_file(cls, path: str | os.PathLike, pattern: str,
                  special_tokens: dict[str, int] | None = None):
        """Build from a `.tiktoken` file on disk."""
        return cls.from_ranks(load_mergeable_ranks(path), pattern, special_tokens)

    @classmethod
    def from_pretrained(cls, name: str = "cl100k_base",
                        cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR,
                        download: bool = True):
        """
        Load a published encoding by name, e.g. "cl100k_base".

        Pass `download=False` to fail instead of hitting the network.
        """
        if name not in ENCODINGS:
            raise ValueError(
                f"unknown encoding {name!r}; known: {sorted(ENCODINGS)}"
            )
        encoding = ENCODINGS[name]
        path = Path(cache_dir) / encoding.filename
        if download:
            path = download_encoding(name, cache_dir)
        elif not path.exists():
            raise FileNotFoundError(
                f"{path} not found; call download_encoding({name!r}) or pass "
                f"download=True"
            )
        self = cls.from_file(path, encoding.pattern, encoding.special_tokens)
        self.name = encoding.name
        return self

    # -- not applicable to a published vocabulary -----------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        raise NotImplementedError(
            "RanksTokenizer is pretrained; train a RegexTokenizer instead"
        )

    def load(self, model_file: str):
        raise NotImplementedError(
            "the single-byte token ids live in the .tiktoken file, which the "
            ".model format does not carry; use RanksTokenizer.from_file()"
        )
