"""Generic RSS/Atom fetcher for lab and org blogs."""

from __future__ import annotations

from datetime import datetime, timezone

from digest.config import Settings, Source
from digest.fetch.common import entry_datetime, entry_text, fetch_feed, make_client, make_item
from digest.models import RawItem


def fetch(cfg: Settings, source: Source, since: datetime) -> list[RawItem]:
    now = datetime.now(timezone.utc)
    items: list[RawItem] = []
    with make_client(cfg) as client:
        for entry in fetch_feed(client, cfg, source.url):
            published = entry_datetime(entry)
            if not published or published < since or not entry.get("link"):
                continue
            items.append(
                make_item(
                    cfg, source,
                    title=entry.get("title", ""),
                    url=entry["link"],
                    published_at=published,
                    text=entry_text(entry),
                    fetched_at=now,
                )
            )
    return items
