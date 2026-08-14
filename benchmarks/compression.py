"""
Bytes per token as a function of vocabulary size, measured on held-out text.

Trains BasicTokenizer and RegexTokenizer on one slice of a corpus and measures
compression on a slice they never saw, across log-spaced vocab sizes from 256
(no merges at all, 1.00 bytes/token by definition) up to 8192. GPT-2's ratio on
the same held-out text is drawn as a reference.

    python benchmarks/compression.py                  # bundled Sherlock Holmes
    python benchmarks/compression.py --quick          # ~1 min, coarser sweep
    python benchmarks/compression.py --input book.txt --holdout-frac 0.2
    python benchmarks/compression.py --train a.txt --holdout b.txt

Writes benchmarks/compression.png and benchmarks/compression.json; --replot
redraws the figure from that json without retraining.
"""

from __future__ import annotations

# This module is named `compression`, which since Python 3.14 is also a stdlib
# package (bz2 does `from compression._common import _streams`). Running this
# file as a script puts benchmarks/ at the front of sys.path, where it would
# shadow that package and break bz2 -> shutil -> matplotlib. So drop our own
# directory from sys.path and add the repo root, before importing anything
# else. `os` and `sys` are already loaded at interpreter startup.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
sys.path.insert(0, _REPO_ROOT)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from bpe import BasicTokenizer, RegexTokenizer  # noqa: E402
from bpe.base import Tokenizer  # noqa: E402
from bpe.gpt2 import GPT2Tokenizer  # noqa: E402

REPO_ROOT = Path(_REPO_ROOT)
OUT_PNG = REPO_ROOT / "benchmarks" / "compression.png"
OUT_JSON = REPO_ROOT / "benchmarks" / "compression.json"

TOKENIZERS = {"BasicTokenizer": BasicTokenizer, "RegexTokenizer": RegexTokenizer}

# design tokens, light surface
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"
SERIES = {"BasicTokenizer": "#2a78d6", "RegexTokenizer": "#eb6834"}


def log_spaced(lo: int = 256, hi: int = 8192, per_octave: int = 2) -> list[int]:
    """Vocab sizes spaced evenly on a log2 axis, endpoints included."""
    import math

    steps = round(math.log2(hi / lo) * per_octave)
    sizes = [round(lo * 2 ** (i / per_octave)) for i in range(steps + 1)]
    return sorted(dict.fromkeys(sizes))


# -----------------------------------------------------------------------------
# corpus


def load_default_corpus() -> str:
    """
    The repo's standard corpus: Project Gutenberg's Sherlock Holmes.

    Reuses conftest's downloader rather than duplicating it, so the benchmark
    and the tests measure the same bytes from the same cache.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from conftest import download_corpus, strip_gutenberg_boilerplate

    raw = download_corpus().read_text(encoding="utf-8").replace("\r\n", "\n")
    return strip_gutenberg_boilerplate(raw)


def split_corpus(text: str, holdout_frac: float) -> tuple[str, str]:
    """
    Contiguous split: the first (1-frac) trains, the last frac is held out.

    Contiguous rather than interleaved, so the held-out slice is genuinely
    unseen prose and not sentences surrounded by their own neighbours. It is
    still the same author and register as the training half, so these numbers
    describe in-domain generalization; use --train/--holdout with two unrelated
    files for the out-of-domain question.
    """
    cut = int(len(text) * (1.0 - holdout_frac))
    return text[:cut], text[cut:]


# -----------------------------------------------------------------------------
# measurement


def truncated(tok: Tokenizer, vocab_size: int) -> Tokenizer:
    """
    A copy of `tok` keeping only its first `vocab_size - 256` merges.

    This is the trick that makes the sweep affordable. Training is greedy and
    its state at step i does not depend on the target vocab size, so the merge
    list learned for 8192 has the list for any smaller vocab as an exact
    prefix. One training run therefore yields every point on the curve;
    training each size separately would repeat the same work eleven times.
    tests/test_compression_benchmark.py pins that equivalence.
    """
    clone = copy.copy(tok)  # shares the compiled split pattern; merges below are fresh
    clone.merges = dict(list(tok.merges.items())[: vocab_size - 256])
    clone.vocab = clone._build_vocab()
    return clone


def ratio(tok: Tokenizer, text: str) -> tuple[float, int, float]:
    """(bytes per token, token count, seconds) for `text` under `tok`."""
    t0 = time.perf_counter()
    ids = tok.encode(text)
    elapsed = time.perf_counter() - t0
    assert tok.decode(ids) == text, f"{type(tok).__name__} failed to roundtrip"
    return len(text.encode("utf-8")) / len(ids), len(ids), elapsed


def gpt2_reference(text: str) -> float | None:
    """GPT-2's bytes per token on `text`, or None if its files can't be loaded."""
    try:
        tok = GPT2Tokenizer.from_pretrained()
    except Exception as e:
        print(f"(skipping GPT-2 reference: {e})")
        return None
    return len(text.encode("utf-8")) / len(tok.encode_ordinary(text))


def run(train_text: str, holdout_text: str, sizes: list[int]) -> dict:
    """Train each tokenizer once at max(sizes), then measure every prefix."""
    results = []
    header = (
        f"{'tokenizer':<16}{'vocab':>7}{'held-out':>10}{'train':>8}"
        f"{'tokens':>9}{'enc s':>8}"
    )

    for name, cls in TOKENIZERS.items():
        print(f"\ntraining {name} to vocab {max(sizes)} "
              f"on {len(train_text.encode('utf-8')):,} bytes ...", flush=True)
        tok = cls()
        t0 = time.perf_counter()
        tok.train(train_text, max(sizes))
        train_s = time.perf_counter() - t0
        print(f"  {train_s:.1f}s\n")
        print(header)
        print("-" * len(header))

        for vocab_size in sizes:
            small = truncated(tok, vocab_size)
            held_ratio, ntokens, enc_s = ratio(small, holdout_text)
            train_ratio, _, _ = ratio(small, train_text)
            results.append({
                "name": name,
                "vocab_size": vocab_size,
                "holdout_ratio": held_ratio,
                "train_ratio": train_ratio,
                "holdout_tokens": ntokens,
                "encode_s": enc_s,
            })
            print(f"{name:<16}{vocab_size:>7}{held_ratio:>10.3f}{train_ratio:>8.3f}"
                  f"{ntokens:>9}{enc_s:>8.2f}", flush=True)

    reference = gpt2_reference(holdout_text)
    if reference is not None:
        print(f"\nGPT-2 (vocab 50257) on held-out text: {reference:.3f} bytes/token")

    return {
        "train_bytes": len(train_text.encode("utf-8")),
        "holdout_bytes": len(holdout_text.encode("utf-8")),
        "sizes": sizes,
        "gpt2_ratio": reference,
        "results": results,
    }


# -----------------------------------------------------------------------------
# plot


def plot(data: dict, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    results, reference = data["results"], data["gpt2_ratio"]
    sizes = data["sizes"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # recessive solid hairline grid, y only: the reader compares heights
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)

    if reference is not None:
        ax.axhline(reference, color=INK_MUTED, linewidth=1.1, linestyle=(0, (5, 4)),
                   zorder=1)
        # anchored left, where the curves are still far below it
        ax.text(sizes[0] * 1.05, reference + 0.07,
                f"GPT-2, vocab 50,257  ·  {reference:.2f}",
                color=INK_MUTED, fontsize=8.5, va="bottom", ha="left")

    # nudge colliding endpoint labels apart; RegexTokenizer lands on the GPT-2 rule
    label_offset = {"BasicTokenizer": 0, "RegexTokenizer": -13}
    for name in TOKENIZERS:
        rows = [r for r in results if r["name"] == name]
        if not rows:
            continue
        xs = [r["vocab_size"] for r in rows]
        ys = [r["holdout_ratio"] for r in rows]
        ax.plot(xs, ys, color=SERIES[name], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        # direct label at the right endpoint, so identity never rests on color alone
        ax.annotate(f"{name}  {ys[-1]:.2f}", (xs[-1], ys[-1]),
                    textcoords="offset points",
                    xytext=(11, label_offset.get(name, 0)),
                    color=SERIES[name], fontsize=8.5, va="center", ha="left",
                    annotation_clip=False)

    ax.set_xscale("log", base=2)
    octaves = [s for s in sizes if s & (s - 1) == 0]
    ax.xaxis.set_major_locator(FixedLocator(octaves))
    ax.xaxis.set_minor_locator(FixedLocator([s for s in sizes if s not in octaves]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    # the half-octave points get a tick but no label; matplotlib's default log
    # minor formatter renders them as "1.41406 x 2^8"
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(sizes[0] * 0.94, sizes[-1] * 1.06)

    ax.set_xlabel("vocabulary size", color=INK_MUTED, fontsize=9.5, labelpad=9)
    ax.set_ylabel("bytes per token, held out", color=INK_MUTED, fontsize=9.5,
                  labelpad=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)

    # title block in axes coords, so it can't land on top of itself
    ax.text(0, 1.13, "Bigger vocabularies compress better, with diminishing returns",
            transform=ax.transAxes, color=INK, fontsize=12, va="bottom", ha="left")
    ax.text(0, 1.045,
            f"trained on {data['train_bytes']:,} bytes of English prose, "
            f"measured on a held-out {data['holdout_bytes']:,}",
            transform=ax.transAxes, color=INK_MUTED, fontsize=8.5,
            va="bottom", ha="left")

    # lower right is the one empty corner
    legend = ax.legend(frameon=False, loc="lower right", fontsize=8.5,
                       handlelength=2.4, borderaxespad=1.2)
    for text in legend.get_texts():
        text.set_color(INK_MUTED)

    # right margin holds the endpoint labels, top margin holds the title block
    fig.subplots_adjust(left=0.095, right=0.80, top=0.845, bottom=0.115)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"wrote {out_path}")


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="one corpus, split by --holdout-frac")
    parser.add_argument("--train", type=Path, help="training corpus (with --holdout)")
    parser.add_argument("--holdout", type=Path, help="held-out corpus (with --train)")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="powers of two up to 2048; roughly a minute")
    parser.add_argument("--out", type=Path, default=OUT_PNG)
    parser.add_argument("--json", type=Path, default=OUT_JSON)
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved json, no training")
    args = parser.parse_args()

    if args.replot:
        plot(json.loads(args.json.read_text()), args.out)
        return

    if bool(args.train) != bool(args.holdout):
        parser.error("--train and --holdout must be given together")

    if args.sizes:
        sizes = sorted(args.sizes)
    elif args.quick:
        sizes = [256, 512, 1024, 2048]
    else:
        sizes = log_spaced()
    if sizes[0] < 256:
        parser.error("vocab sizes must be at least 256")

    if args.train:
        train_text = args.train.read_text(encoding="utf-8")
        holdout_text = args.holdout.read_text(encoding="utf-8")
    else:
        text = (args.input.read_text(encoding="utf-8") if args.input
                else load_default_corpus())
        train_text, holdout_text = split_corpus(text, args.holdout_frac)

    print(f"train {len(train_text.encode('utf-8')):,} bytes | "
          f"held out {len(holdout_text.encode('utf-8')):,} bytes | "
          f"vocab sizes {sizes}")

    t0 = time.perf_counter()
    data = run(train_text, holdout_text, sizes)
    print(f"\ntotal {time.perf_counter() - t0:.0f}s")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.json}")
    plot(data, args.out)


if __name__ == "__main__":
    main()
