"""
Tests for benchmarks/compression_ratio.py.

Two things here are worth pinning. The first is the prefix property the sweep
depends on: `truncated(tok, V)` is claimed to be indistinguishable from having
trained to V in the first place, and if that is ever false the whole curve is
quietly wrong. The second is that the file runs as a script at all, which only
a subprocess can check — a script's own directory goes first on sys.path, so a
benchmark named after a stdlib module breaks on import rather than at the line
that uses it.

The benchmark lives outside any package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bpe import BasicTokenizer, RegexTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "benchmarks" / "compression_ratio.py"

# enough text to support a few hundred merges without being slow
SAMPLE = (REPO_ROOT / "bpe" / "regex.py").read_text(encoding="utf-8") * 3


@pytest.fixture(scope="module")
def bench():
    """The benchmark is not importable by name; load it from its path."""
    spec = importlib.util.spec_from_file_location("benchmark_compression", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the prefix property ------------------------------------------------------


@pytest.mark.parametrize("cls", [BasicTokenizer, RegexTokenizer])
@pytest.mark.parametrize("vocab_size", [256, 300, 400])
def test_truncated_matches_training_at_that_size(bench, cls, vocab_size):
    """
    Training to 512 and truncating to V == training to V.

    This is the load-bearing assumption of the sweep: it trains once at the
    largest size and reads every smaller vocabulary off the same merge list.
    Merges are ordered and training is greedy, so the merge list for a small
    vocab must be an exact prefix of the list for a larger one, ids included.
    """
    big = cls()
    big.train(SAMPLE, 512)

    direct = cls()
    direct.train(SAMPLE, vocab_size)

    got = bench.truncated(big, vocab_size)
    assert got.merges == direct.merges
    assert list(got.merges) == list(direct.merges), "merge ORDER must match too"
    assert got.vocab == direct.vocab


@pytest.mark.parametrize("cls", [BasicTokenizer, RegexTokenizer])
def test_truncated_encodes_identically(bench, cls):
    """The stronger claim: the truncated tokenizer emits the same token ids."""
    big = cls()
    big.train(SAMPLE, 512)
    direct = cls()
    direct.train(SAMPLE, 400)

    text = SAMPLE[:4000]
    assert bench.truncated(big, 400).encode(text) == direct.encode(text)


def test_truncated_leaves_the_original_alone(bench):
    """Truncating must not mutate the tokenizer it copies from."""
    tok = RegexTokenizer()
    tok.train(SAMPLE, 400)
    before = dict(tok.merges)

    small = bench.truncated(tok, 300)
    assert tok.merges == before
    assert len(small.merges) == 44
    assert small.pattern == tok.pattern


def test_vocab_256_is_the_no_merge_baseline(bench):
    """A 256-token vocab has no merges, so it is one token per byte, exactly."""
    tok = RegexTokenizer()
    tok.train(SAMPLE, 512)
    baseline = bench.truncated(tok, 256)

    assert baseline.merges == {}
    text = SAMPLE[:2000]
    ratio, ntokens, _ = bench.ratio(baseline, text)
    assert ratio == 1.0
    assert ntokens == len(text.encode("utf-8"))


def test_ratio_checks_the_roundtrip(bench):
    """`ratio` is also the benchmark's correctness check; it must actually fail."""
    tok = RegexTokenizer()
    tok.train(SAMPLE, 300)
    tok.vocab[257] = b"corrupted"
    with pytest.raises(AssertionError, match="roundtrip"):
        bench.ratio(tok, SAMPLE[:2000])


# -- sweep bookkeeping --------------------------------------------------------


def test_log_spaced_sizes(bench):
    sizes = bench.log_spaced()
    assert sizes[0] == 256 and sizes[-1] == 8192
    assert sizes == sorted(set(sizes))
    # even spacing on a log axis: every consecutive ratio is the same half-octave
    ratios = [b / a for a, b in zip(sizes, sizes[1:])]
    assert all(abs(r - 2 ** 0.5) < 0.01 for r in ratios), ratios


def test_split_corpus_is_contiguous_and_complete(bench):
    train, holdout = bench.split_corpus("abcdefghij", 0.2)
    assert train == "abcdefgh"
    assert holdout == "ij"
    assert train + holdout == "abcdefghij"


# -- the filename hazard ------------------------------------------------------


def test_no_module_shadows_a_stdlib_module():
    """
    No .py file in the repo may be named after a top-level stdlib module.

    Running any script puts its own directory first on sys.path, so such a file
    shadows the real module for everything imported afterwards. This bit once:
    the benchmark was briefly named compression.py, and `compression` became a
    stdlib package in Python 3.14, which broke bz2 -> shutil -> matplotlib.
    Renaming was the fix; this keeps it renamed.
    """
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in REPO_ROOT.rglob("*.py")
        if ".venv" not in path.parts and path.stem in sys.stdlib_module_names
    ]
    assert not offenders, f"these shadow stdlib modules: {offenders}"


def test_runs_as_a_script(tmp_path):
    """
    End-to-end run in a subprocess.

    Only a real `python benchmarks/compression_ratio.py` reproduces the script
    sys.path layout, so an import-time collision would show up here and nowhere
    else in the suite.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE[:20_000], encoding="utf-8")
    out_png, out_json = tmp_path / "c.png", tmp_path / "c.json"

    result = subprocess.run(
        [sys.executable, str(BENCHMARK), "--input", str(corpus),
         "--sizes", "256", "320", "--out", str(out_png), "--json", str(out_json)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out_png.exists() and out_png.stat().st_size > 0

    data = json.loads(out_json.read_text())
    assert {r["name"] for r in data["results"]} == {"BasicTokenizer", "RegexTokenizer"}
    for row in data["results"]:
        if row["vocab_size"] == 256:
            assert row["holdout_ratio"] == 1.0
        else:
            assert row["holdout_ratio"] > 1.0
