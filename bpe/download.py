"""
Fetching and caching the vocabulary files the pretrained tokenizers need.

Every pretrained tokenizer here loads from a file OpenAI publishes, so they all
want the same three things: a certifi-backed SSL context, a download that can't
leave a truncated file behind, and a cache directory that is checked first.

Downloads are verified against a known SHA-256 where we have one. A truncated
or corrupted cache file is worse than a missing one: it fails much later, as a
wrong token id rather than an I/O error.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def ssl_context() -> ssl.SSLContext:
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


def sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, target: str | os.PathLike,
                  expected_sha256: str | None = None,
                  user_agent: str | None = None) -> Path:
    """
    Fetch `url` to `target` unless it's already there, and return the path.

    A cached file whose hash doesn't match is re-downloaded once, on the
    assumption that it was truncated or is from an older release.
    """
    target = Path(target)
    if target.exists():
        if expected_sha256 is None or sha256(target) == expected_sha256:
            return target
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent} if user_agent else {}
    request = urllib.request.Request(url, headers=headers)

    # download to a temp name in the same directory, then rename. An
    # interrupted fetch can't leave a partial file that later runs cache.
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with urllib.request.urlopen(request, context=ssl_context(), timeout=60) as response:
            with os.fdopen(fd, "wb") as f:
                shutil.copyfileobj(response, f)
        if expected_sha256 is not None:
            got = sha256(tmp)
            if got != expected_sha256:
                raise ValueError(
                    f"{url} hashed to {got}, expected {expected_sha256}"
                )
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target
