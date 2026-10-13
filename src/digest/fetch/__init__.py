"""Fetchers. Each module exposes fetch(cfg, source, since) -> list[RawItem]. Network-only stage.

Must not import anything from process/generate/render: it runs on a separate machine
with only requirements-fetch.txt installed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from digest.config import Settings, Source
from digest.fetch import arxiv, extract, github, hf, hn, rss
from digest.models import RawItem

log = logging.getLogger(__name__)

Fetcher = Callable[[Settings, Source, datetime], list[RawItem]]

FETCHERS: dict[str, Fetcher] = {
    "hf_models": hf.fetch,
    "hf_papers": hf.fetch,
    "arxiv": arxiv.fetch,
    "github": github.fetch,
    "hn": hn.fetch,
    "rss": rss.fetch,
}


@dataclass
class SourceReport:
    name: str
    count: int = 0
    seconds: float = 0.0
    error: str | None = None


def run(
    cfg: Settings, sources: list[Source], since: datetime, do_extract: bool = True
) -> tuple[list[RawItem], list[SourceReport]]:
    """Fetch every source; a failing source is logged and reported, never fatal. Dedup by id."""
    items: list[RawItem] = []
    reports: list[SourceReport] = []
    for source in sources:
        report = SourceReport(source.name)
        start = time.monotonic()
        try:
            fetcher = FETCHERS[source.type]
            got = fetcher(cfg, source, since)
            report.count = len(got)
            items.extend(got)
        except Exception as exc:  # noqa: BLE001 - one source must never stop the run
            report.error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:200]
            log.warning("source %s failed: %s", source.name, report.error)
        report.seconds = time.monotonic() - start
        reports.append(report)

    items = dedup(items)
    if do_extract:
        items = extract.enrich(cfg, items)
    return items, reports


def dedup(items: list[RawItem]) -> list[RawItem]:
    """Keep one item per id, preferring the one with the longest text (first seen on ties)."""
    best: dict[str, RawItem] = {}
    for item in items:
        prev = best.get(item.id)
        if prev is None or len(item.text) > len(prev.text):
            best[item.id] = item
    dropped = len(items) - len(best)
    if dropped:
        log.info("dedup dropped %d duplicate items", dropped)
    return list(best.values())
