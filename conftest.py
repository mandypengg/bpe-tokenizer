# Present so pytest puts the repo root on sys.path and `import bpe` resolves.
"""
Shared fixtures. The English corpus used by the roundtrip tests is downloaded
once and cached under `data/` through the same helper the pretrained
tokenizers use: `data/` is gitignored, so nothing large lands in the repo, and
tests that need the network skip cleanly when it is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bpe.download import DATA_DIR, download_file

CORPUS_DIR = DATA_DIR / "corpus"

# "The Adventures of Sherlock Holmes", public domain, ~575 KB of English prose
# once the Project Gutenberg boilerplate is stripped.
CORPUS_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
CORPUS_FILE = "pg1661.txt"

_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def download_corpus(cache_dir: Path = CORPUS_DIR) -> Path:
    """Fetch the corpus into `cache_dir` if not already there."""
    return download_file(
        CORPUS_URL,
        cache_dir / CORPUS_FILE,
        # gutenberg.org refuses urllib's default User-Agent
        user_agent="Mozilla/5.0 (compatible; bpe-tokenizer tests)",
    )


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
