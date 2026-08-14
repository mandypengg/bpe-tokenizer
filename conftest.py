# Present so pytest puts the repo root on sys.path and `import bpe` resolves.
"""
Shared fixtures. The English corpus used by the roundtrip tests is downloaded
once and cached under `data/`, the same arrangement `bpe.gpt2` uses for
encoder.json / vocab.bpe: `data/` is gitignored, so nothing large lands in the
repo, and tests that need the network skip cleanly when it is unavailable.
"""

from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# "The Adventures of Sherlock Holmes", public domain, ~575 KB of English prose
# once the Project Gutenberg boilerplate is stripped.
CORPUS_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
CORPUS_FILE = "pg1661.txt"

_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _ssl_context() -> ssl.SSLContext:
    """Same certifi-then-system fallback as bpe/gpt2.py; see the note there."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download_corpus(cache_dir: Path = CORPUS_DIR) -> Path:
    """Fetch the corpus into `cache_dir` if not already there."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / CORPUS_FILE
    if target.exists():
        return target
    # download to a temp name first so an interrupted fetch can't leave a
    # truncated file that later runs treat as cached
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        CORPUS_URL,
        # gutenberg.org refuses urllib's default User-Agent
        headers={"User-Agent": "Mozilla/5.0 (compatible; bpe-tokenizer tests)"},
    )
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=60) as response:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(response, f)
    tmp.replace(target)
    return target


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Drop the license header/footer, keeping only the book itself."""
    start = raw.index(_START_MARKER)
    start = raw.index("\n", start) + 1
    end = raw.index(_END_MARKER)
    return raw[start:end].strip()


@pytest.fixture(scope="session")
def english_corpus() -> str:
    """A few hundred KB of real English prose, normalized to \\n line endings."""
    try:
        path = download_corpus()
    except Exception as e:  # offline, DNS failure, gutenberg down, ...
        pytest.skip(f"corpus unavailable: {e}")
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return strip_gutenberg_boilerplate(raw)
