"""
GPT-2's tokenizer, loaded from the original `encoder.json` / `vocab.bpe`.

This is the correctness target for the whole package: encoding with this class
must reproduce tiktoken's "gpt2" encoding token for token.

Two GPT-2 specifics matter here:

1. The byte <-> unicode mapping. GPT-2 does BPE over a reversible rendering of
   bytes as printable unicode characters (`bytes_to_unicode`), so that its
   vocab files stay plain text. We invert that mapping to recover the raw
   bytes behind every token.
2. The 256 single-byte tokens are NOT ids 0..255. `encoder.json` assigns them
   in `bytes_to_unicode` order, so byte 0x21 ('!') is id 0. We hand that
   permutation to the base class as `byte_to_id`, which seeds encoding and
   the vocabulary from it.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .download import DATA_DIR, download_file
from .regex import GPT2_SPLIT_PATTERN, RegexTokenizer

ENDOFTEXT = "<|endoftext|>"

# url and expected sha256 per file. The hashes are the ones tiktoken pins for
# the same files served from its own url; both paths serve identical bytes.
GPT2_FILES = {
    "encoder.json": (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    ),
    "vocab.bpe": (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
}

DEFAULT_CACHE_DIR = DATA_DIR / "gpt2"


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """
    Reversible map from the 256 byte values to printable unicode characters.

    The printable ASCII/Latin-1 ranges map to themselves; the remaining 68
    bytes (control characters, space, and a few others) are shifted up into
    the 256+ range so that no token ever contains whitespace or a control
    character. Reversible by construction — see `unicode_to_bytes`.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache()
def unicode_to_bytes() -> dict[str, int]:
    """Inverse of `bytes_to_unicode`."""
    return {ch: b for b, ch in bytes_to_unicode().items()}


def download_gpt2_files(cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR) -> Path:
    """Fetch encoder.json and vocab.bpe into `cache_dir` if not already there."""
    cache_dir = Path(cache_dir)
    for name, (url, digest) in GPT2_FILES.items():
        download_file(url, cache_dir / name, expected_sha256=digest)
    return cache_dir


class GPT2Tokenizer(RegexTokenizer):
    """
    Pretrained GPT-2 tokenizer. Construct it with `from_pretrained()`.

    Training is not supported: the merges come from OpenAI's files, and the
    point of this class is to reproduce them exactly.
    """

    def __init__(self):
        super().__init__(pattern=GPT2_SPLIT_PATTERN)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_files(cls, encoder_json: str | os.PathLike, vocab_bpe: str | os.PathLike):
        """Build from the two original GPT-2 files."""
        with open(encoder_json, "r", encoding="utf-8") as f:
            encoder: dict[str, int] = json.load(f)
        with open(vocab_bpe, "r", encoding="utf-8") as f:
            bpe_data = f.read()

        self = cls()

        # ids of the 256 single-byte tokens, in byte order
        byte_encoder = bytes_to_unicode()
        self.byte_to_id = {b: encoder[ch] for b, ch in byte_encoder.items()}

        # skip the "#version: 0.2" header; each remaining line is one merge,
        # and the LINE ORDER is the merge priority
        lines = bpe_data.split("\n")[1:]
        merges: dict[tuple[int, int], int] = {}
        for line in lines:
            if not line:
                continue
            first, second = line.split()
            merges[(encoder[first], encoder[second])] = encoder[first + second]
        self.merges = merges

        self.vocab = self._build_vocab()
        # any token in encoder.json that isn't reachable through the merges
        # (i.e. <|endoftext|>) is a special token
        specials = {
            tok: idx for tok, idx in encoder.items() if idx not in self.vocab
        }
        if specials:
            self.register_special_tokens(specials)

        assert len(self.vocab) == len(encoder), (
            f"rebuilt {len(self.vocab)} tokens but encoder.json has {len(encoder)}"
        )
        return self

    @classmethod
    def from_pretrained(cls, cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR,
                        download: bool = True):
        """
        Load from `cache_dir`, downloading OpenAI's files there if missing.

        Pass `download=False` to fail instead of hitting the network.
        """
        cache_dir = Path(cache_dir)
        if download:
            download_gpt2_files(cache_dir)
        encoder_json = cache_dir / "encoder.json"
        vocab_bpe = cache_dir / "vocab.bpe"
        for path in (encoder_json, vocab_bpe):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found; call download_gpt2_files() or pass "
                    f"download=True"
                )
        return cls.from_files(encoder_json, vocab_bpe)

    # -- GPT-2 specific overrides --------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        raise NotImplementedError(
            "GPT2Tokenizer is pretrained; train a RegexTokenizer instead"
        )

    def load(self, model_file: str):
        raise NotImplementedError(
            "GPT-2's single-byte token ids live in encoder.json, which the "
            ".model format does not carry; use GPT2Tokenizer.from_files()"
        )
