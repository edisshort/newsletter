"""arXiv category RSS (rss.arxiv.org). Only new submissions and cross-lists; revisions are skipped."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from digest.config import Settings, Source
from digest.fetch.common import clean_text, entry_datetime, fetch_feed, make_client, make_item
from digest.models import RawItem

# Summary looks like "arXiv:2610.10550v1 Announce Type: new  Abstract: ..."
_PREFIX_RE = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*", re.IGNORECASE)


def fetch(cfg: Settings, source: Source, since: datetime) -> list[RawItem]:
    keep_types = set(source.extra.get("announce_types", ["new", "cross"]))
    now = datetime.now(timezone.utc)
    items: list[RawItem] = []
    with make_client(cfg) as client:
        for entry in fetch_feed(client, cfg, source.url):
            if entry.get("arxiv_announce_type", "new") not in keep_types:
                continue
            published = entry_datetime(entry)
            if not published or published < since:
                continue
            items.append(
                make_item(
                    cfg, source,
                    title=entry.get("title", ""),
                    url=entry["link"],
                    published_at=published,
                    text=_PREFIX_RE.sub("", clean_text(entry.get("summary", ""))),
                    fetched_at=now,
                )
            )
    return items
