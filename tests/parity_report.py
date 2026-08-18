"""
Shared failure reporting for the parity sweeps.

Failures report the input, both token sequences windowed around the problem,
and the first index where they diverge, because "3,412 mismatches" is not a
bug report and "index 4 of ' café': ours 269 (b' ca') vs theirs 40304
(b' caf')" is.
"""

from __future__ import annotations

import collections

# how many detailed mismatch reports to print before summarizing the rest
MAX_REPORTS = 20
# tokens of context to show either side of the first divergence
WINDOW = 6


def first_divergence(ours_ids: list[int], theirs: list[int]) -> int:
    """Index of the first differing token, or the length of the shorter list."""
    for i, (a, b) in enumerate(zip(ours_ids, theirs)):
        if a != b:
            return i
    return min(len(ours_ids), len(theirs))


def _token_repr(tokenizer, ids: list[int], i: int) -> str:
    if i >= len(ids):
        return "<past end>"
    idx = ids[i]
    try:
        raw = tokenizer.decode_single_token_bytes(idx)  # tiktoken
    except AttributeError:
        raw = tokenizer.vocab[idx]
    return f"{idx} ({raw!r})"


def _seq(ids: list[int], limit: int = 40) -> str:
    body = ", ".join(str(i) for i in ids[:limit])
    if len(ids) > limit:
        body += f", ... +{len(ids) - limit} more"
    return f"[{body}]  ({len(ids)} tokens)"


def format_mismatch(label, text, ours_ids, theirs, ours_tok, ref_tok) -> str:
    """Full report for one disagreement: input, both sequences, divergence point."""
    i = first_divergence(ours_ids, theirs)
    lo, hi = max(0, i - WINDOW), i + WINDOW + 1

    lines = [
        f"{label}",
        f"  text      : {text!r}",
        f"  utf-8     : {text.encode('utf-8')!r}",
        f"  ours      : {_seq(ours_ids)}",
        f"  theirs    : {_seq(theirs)}",
        f"  diverge at index {i}:",
        f"    ours  [{i}] = {_token_repr(ours_tok, ours_ids, i)}",
        f"    theirs[{i}] = {_token_repr(ref_tok, theirs, i)}",
    ]

    if i > 0:
        agreed = ours_tok.decode(ours_ids[:i])
        lines.append(f"  agreed prefix decodes to {agreed!r}")

    lines.append(f"  window [{lo}:{hi}]")
    for name, ids, tok in (("ours", ours_ids, ours_tok), ("theirs", theirs, ref_tok)):
        window = [
            ("->" if j == i else "  ") + _token_repr(tok, ids, j)
            for j in range(lo, min(hi, len(ids)))
        ]
        lines.append(f"    {name:<7}: " + "  ".join(window))
    return "\n".join(lines)


def report(reports: list[str], total: int, by_category: dict) -> str:
    """
    Assembled failure message: per-case detail, then a category breakdown.

    `reports` is only the capped sample that got formatted; the real mismatch
    count comes from `by_category`, which counts every case.
    """
    failed = sum(by_category.values())
    parts = [f"\n{failed} of {total} cases disagree with tiktoken.\n"]
    parts.extend(reports)
    if failed > len(reports):
        parts.append(f"\n... and {failed - len(reports)} more not shown.")
    if by_category:
        parts.append("\nmismatches by category (failed / total in category):")
        for name, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
            parts.append(f"  {name:<18} {count}")
    return "\n".join(parts)


def sweep(cases, encode_ours, encode_theirs, ours_tok, ref_tok):
    """
    Run both encoders over `cases`, counting every disagreement but formatting
    only the first MAX_REPORTS of them.

    Returns (reports, by_category). Counting and formatting are deliberately
    separate: the headline number has to reflect all 5,000 cases, not just the
    handful small enough to print.
    """
    reports: list[str] = []
    by_category: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()

    for i, case in enumerate(cases):
        text, category = case["text"], case["category"]
        seen[category] += 1
        got, want = encode_ours(text), encode_theirs(text)
        if got != want:
            by_category[category] += 1
            if len(reports) < MAX_REPORTS:
                label = f"case #{i} ({category} #{seen[category]})"
                reports.append(
                    format_mismatch(label, text, got, want, ours_tok, ref_tok)
                )
    return reports, by_category
