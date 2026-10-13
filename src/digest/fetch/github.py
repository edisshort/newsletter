"""GitHub release Atom feeds for a list of repos (no API token needed)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from digest.config import Settings, Source
from digest.fetch.common import entry_datetime, entry_text, fetch_feed, make_client, make_item
from digest.models import RawItem

log = logging.getLogger(__name__)


def fetch(cfg: Settings, source: Source, since: datetime) -> list[RawItem]:
    """One failing repo is logged and skipped; the source fails only if every repo fails."""
    repos: list[str] = source.extra.get("repos", [])
    now = datetime.now(timezone.utc)
    items: list[RawItem] = []
    failures = 0
    with make_client(cfg) as client:
        for repo in repos:
            try:
                entries = fetch_feed(client, cfg, source.url.format(repo=repo))
            except Exception as exc:  # noqa: BLE001 - isolate per-repo failures
                failures += 1
                log.warning("%s: repo %s failed: %s", source.name, repo, exc)
                continue
            for entry in entries:
                published = entry_datetime(entry)
                if not published or published < since or not entry.get("link"):
                    continue
                items.append(
                    make_item(
                        cfg, source,
                        title=f"{repo} {entry.get('title', '')}",
                        url=entry["link"],
                        published_at=published,
                        text=entry_text(entry),
                        fetched_at=now,
                    )
                )
    if repos and failures == len(repos):
        raise RuntimeError(f"all {failures} repos failed")
    return items
