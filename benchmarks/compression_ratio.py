"""
How much does a bigger vocabulary actually buy you?

Trains BasicTokenizer and RegexTokenizer at a range of vocab sizes and plots
compression (bytes per token) against vocab size, with GPT-2's own ratio on
the same corpus drawn as a reference line.

    python benchmarks/compression_ratio.py                     # bundled corpus
    python benchmarks/compression_ratio.py --input book.txt
    python benchmarks/compression_ratio.py --sizes 300 500 1000 --no-plot

(Not named compression.py: as a script its own directory leads sys.path, and
`compression` is a stdlib package as of Python 3.14 — shadowing it breaks
gzip, and therefore matplotlib.)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bpe import BasicTokenizer, RegexTokenizer  # noqa: E402
from bpe.gpt2 import GPT2Tokenizer  # noqa: E402

DEFAULT_SIZES = [300, 512, 1024, 2048, 4096]


def load_corpus(path: Path | None) -> str:
    """Read the corpus, defaulting to this repo's own Python source."""
    if path is not None:
        return path.read_text(encoding="utf-8")
    sources = sorted(REPO_ROOT.glob("bpe/*.py")) + sorted(REPO_ROOT.glob("tests/*.py"))
    return "\n".join(p.read_text(encoding="utf-8") for p in sources)


def measure(tokenizer_cls, text: str, vocab_size: int) -> dict:
    tok = tokenizer_cls()
    t0 = time.perf_counter()
    tok.train(text, vocab_size)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ids = tok.encode(text)
    encode_s = time.perf_counter() - t0

    assert tok.decode(ids) == text, f"{tokenizer_cls.__name__} failed to roundtrip"
    nbytes = len(text.encode("utf-8"))
    return {
        "name": tokenizer_cls.__name__,
        "vocab_size": vocab_size,
        "tokens": len(ids),
        "ratio": nbytes / len(ids),
        "train_s": train_s,
        "encode_s": encode_s,
    }


def gpt2_ratio(text: str) -> float | None:
    """GPT-2's bytes per token on this corpus, or None if it can't be loaded."""
    try:
        tok = GPT2Tokenizer.from_pretrained()
    except Exception as e:
        print(f"(skipping GPT-2 reference line: {e})")
        return None
    return len(text.encode("utf-8")) / len(tok.encode_ordinary(text))


def plot(results: list[dict], reference: float | None, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in dict.fromkeys(r["name"] for r in results):
        rows = [r for r in results if r["name"] == name]
        ax.plot(
            [r["vocab_size"] for r in rows],
            [r["ratio"] for r in rows],
            marker="o",
            label=name,
        )
    if reference is not None:
        ax.axhline(
            reference,
            linestyle="--",
            color="gray",
            label=f"GPT-2 (50257) = {reference:.2f}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("vocab size")
    ax.set_ylabel("bytes per token")
    ax.set_title("BPE compression vs vocab size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="corpus file")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "compression.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    text = load_corpus(args.input)
    nbytes = len(text.encode("utf-8"))
    print(f"corpus: {nbytes} bytes, {len(text)} characters\n")

    header = f"{'tokenizer':<16}{'vocab':>7}{'tokens':>9}{'bytes/tok':>11}{'train s':>9}{'enc s':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for vocab_size in args.sizes:
        for cls in (BasicTokenizer, RegexTokenizer):
            r = measure(cls, text, vocab_size)
            results.append(r)
            print(
                f"{r['name']:<16}{r['vocab_size']:>7}{r['tokens']:>9}"
                f"{r['ratio']:>11.3f}{r['train_s']:>9.2f}{r['encode_s']:>8.2f}"
            )

    reference = gpt2_ratio(text)
    if reference is not None:
        print(f"\nGPT-2 (vocab 50257): {reference:.3f} bytes/token")

    if not args.no_plot:
        plot(results, reference, args.out)


if __name__ == "__main__":
    main()
