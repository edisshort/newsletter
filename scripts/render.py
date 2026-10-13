#!/usr/bin/env python3
"""Render top10.json (a Digest) into a single self-contained HTML page."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digest.config import load_settings  # noqa: E402
from digest.models import Digest  # noqa: E402
from digest.render.render import render, source_counts  # noqa: E402

log = logging.getLogger("render")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=Path, default=None, help="Digest JSON (default: settings.paths.top10)")
    p.add_argument("--out", type=Path, default=None, help="HTML file (default: settings.paths.html)")
    p.add_argument("--raw", type=Path, default=None,
                   help="RawItem JSONL for the footer's per-source counts (default: settings.paths.raw; skipped if missing)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_settings()
    inp = args.inp or cfg.paths.top10
    out = args.out or cfg.paths.html
    raw = args.raw or cfg.paths.raw

    digest = Digest.model_validate_json(inp.read_text(encoding="utf-8"))
    counts = source_counts(raw) if raw.is_file() else None
    if counts is None:
        log.warning("raw snapshot %s not found; footer will omit source counts", raw)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(digest, cfg, counts), encoding="utf-8")
    print(f"{len(digest.top10)} items rendered -> {out}")
    return 0 if len(digest.top10) == cfg.generate.top_n else 1


if __name__ == "__main__":
    sys.exit(main())
