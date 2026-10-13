"""Hacker News stories via the Algolia search_by_date API."""

from __future__ import annotations

from datetime import datetime, timezone

from digest.config import Settings, Source
from digest.fetch.common import get, make_client, make_item, parse_iso
from digest.models import RawItem

HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"


def fetch(cfg: Settings, source: Source, since: datetime) -> list[RawItem]:
    min_points = source.extra.get("min_points", 30)
    hits_per_page = source.extra.get("hits_per_page", 100)
    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    items: list[RawItem] = []
    with make_client(cfg) as client:
        for query in source.extra.get("queries", []):
            params = {
                "query": query,
                "tags": "story",
                "numericFilters": f"created_at_i>{int(since.timestamp())},points>{min_points}",
                "hitsPerPage": hits_per_page,
            }
            for hit in get(client, cfg, source.url, params=params).json().get("hits", []):
                if hit["objectID"] in seen or not hit.get("title"):
                    continue
                seen.add(hit["objectID"])
                items.append(
                    make_item(
                        cfg, source,
                        title=hit["title"],
                        url=hit.get("url") or HN_ITEM_URL.format(id=hit["objectID"]),
                        published_at=parse_iso(hit["created_at"]),
                        text=hit.get("story_text") or "",
                        metrics={"points": hit.get("points"), "comments": hit.get("num_comments")},
                        fetched_at=now,
                    )
                )
    return items
