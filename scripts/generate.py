#!/usr/bin/env python3
"""Summarise ranked clusters into top10.json with the internal LLM (or extractively with --dry-run)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digest.config import load_settings  # noqa: E402
from digest.generate import report_dict, run  # noqa: E402
from digest.generate.llm import LLM  # noqa: E402
from digest.models import Cluster  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=Path, default=None, help="ranked clusters JSON (default: settings.paths.processed)")
    p.add_argument("--out", type=Path, default=None, help="Digest JSON (default: settings.paths.top10)")
    p.add_argument("--report", type=Path, default=None, help="grounding report JSON (default: settings.generate.report)")
    p.add_argument("--dry-run", action="store_true", help="skip the LLM; extractive summaries only")
    p.add_argument("--days", type=int, default=None, help="window shown in the digest (default: settings.fetch.window_days)")
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
    inp = args.inp or cfg.paths.processed
    out = args.out or cfg.paths.top10
    report_path = args.report or cfg.generate.report

    clusters = [Cluster.model_validate(c) for c in json.loads(inp.read_text(encoding="utf-8"))]
    llm = None if args.dry_run else LLM(cfg)
    digest, reports = run(cfg, clusters, llm, args.days or cfg.fetch.window_days)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(digest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report = report_dict(reports)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'#':>2} {'mode':<10} {'tries':>5} {'score':>6}  {'category':<13} title")
    for item, r in zip(digest.top10, [r for r in reports if r.cluster_id and r.summary_mode in ("llm", "extractive")]):
        print(f"{item.rank:>2} {item.summary_mode:<10} {r.attempts:>5} {item.trend_score:>6.3f}  {item.category:<13} {item.title[:70]}")
    print(f"\n{len(digest.top10)} initiatives ({report['llm']} llm, {report['extractive']} extractive); "
          f"unresolved grounding misses in llm items: {report['unresolved_llm_grounding_misses']}")
    print(f"overview ({digest.overview_mode}): {digest.headline}")
    print(f"-> {out}\n-> {report_path}")
    return 0 if len(digest.top10) == cfg.generate.top_n else 1


if __name__ == "__main__":
    sys.exit(main())
