"""Shared HTTP, URL-canonicalisation and parsing helpers for fetchers."""

from __future__ import annotations

import calendar
import hashlib
import html
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import feedparser
import httpx

from digest.config import Settings, Source
from digest.models import RawItem

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def make_client(cfg: Settings) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": cfg.fetch.user_agent},
        timeout=cfg.fetch.timeout_s,
        follow_redirects=True,
        proxy=cfg.fetch.https_proxy or None,
    )


def get(client: httpx.Client, cfg: Settings, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
    """GET with `cfg.fetch.retries` retries and exponential backoff. Raises on final failure.

    4xx responses (other than 429) are not retried: they will not get better.
    """
    attempts = cfg.fetch.retries + 1
    for attempt in range(attempts):
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
            if not retryable or attempt == attempts - 1:
                raise
            log.debug("retrying %s after HTTP %s", url, exc.response.status_code)
        except httpx.TransportError as exc:
            if attempt == attempts - 1:
                raise
            log.debug("retrying %s after %s", url, type(exc).__name__)
        time.sleep(cfg.fetch.backoff_s * 2**attempt)
    raise AssertionError("unreachable")


def canonical_url(url: str, keep_query_hosts: list[str]) -> str:
    """Lower-case scheme/host, drop www., fragment, trailing slash and (usually) the query string."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    query = parts.query if host in keep_query_hosts else ""
    path = parts.path.rstrip("/")
    return urlunsplit(("https", host, path, query, ""))


def item_id(url: str, keep_query_hosts: list[str]) -> str:
    return hashlib.sha1(canonical_url(url, keep_query_hosts).encode("utf-8")).hexdigest()


def clean_text(raw: str | None) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not raw:
        return ""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", raw))).strip()


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def struct_to_dt(st: time.struct_time) -> datetime:
    """feedparser *_parsed values are UTC struct_time."""
    return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)


def make_item(
    cfg: Settings,
    source: Source,
    *,
    title: str,
    url: str,
    published_at: datetime,
    text: str = "",
    metrics: dict[str, float] | None = None,
    fetched_at: datetime,
) -> RawItem:
    return RawItem(
        id=item_id(url, cfg.fetch.keep_query_hosts),
        title=clean_text(title),
        url=url,
        source=source.name,
        type=source.item_type,
        published_at=published_at,
        text=clean_text(text)[: cfg.fetch.max_text_chars],
        metrics={k: float(v) for k, v in (metrics or {}).items() if v is not None},
        fetched_at=fetched_at,
    )


def fetch_feed(client: httpx.Client, cfg: Settings, url: str) -> list[feedparser.FeedParserDict]:
    """Download a feed with our client (UA, proxy, retries), then parse it."""
    parsed = feedparser.parse(get(client, cfg, url).content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"unparseable feed {url}: {parsed.bozo_exception}")
    return parsed.entries


def entry_datetime(entry: feedparser.FeedParserDict) -> datetime | None:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    return struct_to_dt(st) if st else None


def entry_text(entry: feedparser.FeedParserDict) -> str:
    """Prefer full content over summary when the feed provides it."""
    content = entry.get("content") or []
    if content and content[0].get("value"):
        return content[0]["value"]
    return entry.get("summary", "")
