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
   in `bytes_to_unicode` order, so byte 0x21 ('!') is id 0. Everything else in
   this package assumes id == byte value for the base tokens, so we override
   `_byte_ids` to seed encoding with GPT-2's permuted ids instead.
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import urllib.request
from functools import lru_cache
from pathlib import Path

from .regex import GPT2_SPLIT_PATTERN, RegexTokenizer

ENDOFTEXT = "<|endoftext|>"

GPT2_FILE_URLS = {
    "encoder.json": "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
    "vocab.bpe": "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "gpt2"


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


def _ssl_context() -> ssl.SSLContext:
    """
    Default SSL context, backed by certifi's roots when they're available.

    A stock macOS python often has no usable CA store, so a plain urlopen
    fails with CERTIFICATE_VERIFY_FAILED. certifi ships as a transitive
    dependency of tiktoken; fall back to the system store without it.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download_gpt2_files(cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR) -> Path:
    """Fetch encoder.json and vocab.bpe into `cache_dir` if not already there."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    context = _ssl_context()
    for name, url in GPT2_FILE_URLS.items():
        target = cache_dir / name
        if target.exists():
            continue
        # download to a temp name first so an interrupted fetch can't leave a
        # truncated file that later runs treat as cached
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, context=context, timeout=60) as response:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(response, f)
        tmp.replace(target)
    return cache_dir


class GPT2Tokenizer(RegexTokenizer):
    """
    Pretrained GPT-2 tokenizer. Construct it with `from_pretrained()`.

    Training is not supported: the merges come from OpenAI's files, and the
    point of this class is to reproduce them exactly.
    """

    def __init__(self):
        # must exist before Tokenizer.__init__ calls _build_vocab()
        self.byte_to_id: dict[int, int] = {}
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

    def _byte_ids(self, text_bytes: bytes) -> list[int]:
        return [self.byte_to_id[b] for b in text_bytes]

    def _build_vocab(self) -> dict[int, bytes]:
        if not self.byte_to_id:
            return {}  # not loaded yet
        vocab = {idx: bytes([b]) for b, idx in self.byte_to_id.items()}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        raise NotImplementedError(
            "GPT2Tokenizer is pretrained; train a RegexTokenizer instead"
        )

    def load(self, model_file: str):
        raise NotImplementedError(
            "GPT-2's single-byte token ids live in encoder.json, which the "
            ".model format does not carry; use GPT2Tokenizer.from_files()"
        )
