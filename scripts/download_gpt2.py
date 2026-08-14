#!/usr/bin/env python
"""
Fetch OpenAI's encoder.json / vocab.bpe if they aren't already cached, then
load them and print a sanity check.

    python scripts/download_gpt2.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe.gpt2 import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    ENDOFTEXT,
    GPT2Tokenizer,
    bytes_to_unicode,
    download_gpt2_files,
)

PROBE_IDS = (0, 100, 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)

    cache_dir = download_gpt2_files(args.cache_dir)
    print(f"vocab files in {cache_dir}")

    tokenizer = GPT2Tokenizer.from_pretrained(cache_dir, download=False)
    print(f"vocab entries: {len(tokenizer.vocab)}")
    print(f"merges:        {len(tokenizer.merges)}")
    print(f"specials:      {tokenizer.special_tokens}")

    byte_encoder = bytes_to_unicode()
    eot_id = tokenizer.special_tokens[ENDOFTEXT]
    for idx in (*PROBE_IDS, eot_id):
        raw = tokenizer.vocab[idx]
        # the rendered form is the string that actually appears in encoder.json
        rendered = "".join(byte_encoder[b] for b in raw)
        text = raw.decode("utf-8", errors="replace")
        print(f"  id {idx:<6} bytes={raw!r:<18} text={text!r:<17} rendered={rendered!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
