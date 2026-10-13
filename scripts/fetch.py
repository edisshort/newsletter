#!/usr/bin/env python3
"""Fetch all enabled sources into a RawItem JSONL snapshot (internet-connected machine only)."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digest.config import load_settings, load_sources  # noqa: E402
from digest.fetch import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=None, help="look-back window (default: settings.fetch.window_days)")
    p.add_argument("--out", type=Path, default=None, help="output JSONL (default: settings.paths.raw)")
    p.add_argument("--sources-file", type=Path, default=None, help="alternative sources.yaml")
    p.add_argument("--only", nargs="+", default=None, help="fetch only these source names")
    p.add_argument("--no-extract", action="store_true", help="skip full-page extraction")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_settings()
    sources = load_sources(args.sources_file, enabled_only=True)
    if args.only:
        sources = [s for s in sources if s.name in args.only]
    days = args.days or cfg.fetch.window_days
    out = args.out or cfg.paths.raw
    since = datetime.now(timezone.utc) - timedelta(days=days)

    items, reports = run(cfg, sources, since, do_extract=not args.no_extract)
    items.sort(key=lambda it: (it.published_at, it.id), reverse=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")

    kept: dict[str, int] = {}
    for item in items:
        kept[item.source] = kept.get(item.source, 0) + 1
    print(f"\n{'source':<20} {'fetched':>8} {'kept':>6} {'secs':>6}  status")
    for r in reports:
        status = f"FAILED  {r.error}" if r.error else "ok"
        print(f"{r.name:<20} {r.count:>8} {kept.get(r.name, 0):>6} {r.seconds:>6.1f}  {status}")
    non_empty = sum(1 for v in kept.values() if v)
    print(f"\n{len(items)} items from {non_empty}/{len(reports)} sources (window {days}d) -> {out}")
    return 0 if items else 1


if __name__ == "__main__":
    sys.exit(main())
