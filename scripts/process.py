#!/usr/bin/env python3
"""Embed, cluster and rank a RawItem JSONL snapshot into ranked_clusters.json (offline)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digest.config import load_settings, load_sources  # noqa: E402
from digest.models import RawItem  # noqa: E402
from digest.process import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=Path, default=None, help="RawItem JSONL (default: settings.paths.raw)")
    p.add_argument("--out", type=Path, default=None, help="ranked clusters JSON (default: settings.paths.processed)")
    p.add_argument("--threshold", type=float, default=None, help="override cluster.cluster_threshold")
    p.add_argument("--relevance-min", type=float, default=None, help="override relevance.relevance_min")
    p.add_argument("--rows", type=int, default=None, help="table rows to print (default: settings.score.table_rows)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_settings()
    if args.threshold is not None:
        cfg.cluster.cluster_threshold = args.threshold
    if args.relevance_min is not None:
        cfg.relevance.relevance_min = args.relevance_min
    inp = args.inp or cfg.paths.raw
    out = args.out or cfg.paths.processed

    items = [RawItem.model_validate_json(line) for line in inp.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = run(cfg, load_sources(), items)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.model_dump(mode="json") for c in result.selected]
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    selected_ids = {c.cluster_id for c in result.selected}
    rows = args.rows or cfg.score.table_rows
    print(f"\n{'rank':>4} {'sel':>3} {'score':>6} {'cov':>5} {'eng':>5} {'rel':>5} {'rec':>5} {'raw_rel':>7} {'n':>3}  "
          f"{'anchor':<24} headline  [sources]")
    for c in result.ranked[:rows]:
        mark = "*" if c.cluster_id in selected_ids else ""
        print(f"{c.cluster_id:>4} {mark:>3} {c.score:>6.3f} {c.coverage:>5.2f} {c.engagement:>5.2f} {c.relevance:>5.2f} "
              f"{c.recency:>5.2f} {result.raw_relevance[c.cluster_id]:>7.3f} {len(c.items):>3}  "
              f"{c.best_anchor[:24]:<24} {c.headline[:70]}  [{', '.join(c.sources)}]")
    print(f"\n{len(items)} items -> {len(result.ranked)} relevant clusters "
          f"({result.dropped_irrelevant} dropped as off-topic); {len(result.selected)} selected (*) -> {out}")
    return 0 if result.selected else 1


if __name__ == "__main__":
    sys.exit(main())
