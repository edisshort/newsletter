"""Hugging Face trending models and daily papers."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from digest.config import Settings, Source
from digest.fetch.common import get, make_client, make_item, parse_iso
from digest.models import RawItem

log = logging.getLogger(__name__)

HF_MODEL_URL = "https://huggingface.co/{id}"
HF_PAPER_URL = "https://huggingface.co/papers/{id}"
# Tag prefixes that are noise for embedding/clustering.
_SKIP_TAG_PREFIXES = ("region:", "endpoints_compatible", "autotrain_compatible", "deploy:", "text-generation-inference")


def fetch(cfg: Settings, source: Source, since: datetime) -> list[RawItem]:
    if source.type == "hf_models":
        return _fetch_models(cfg, source, since)
    if source.type == "hf_papers":
        return _fetch_papers(cfg, source, since)
    raise ValueError(f"unknown HF source type {source.type!r}")


def _model_text(m: dict[str, Any]) -> str:
    """Model listings carry no prose, so build a short description from metadata."""
    card = m.get("cardData") or {}
    base = card.get("base_model")
    if isinstance(base, list):
        base = ", ".join(base)
    tags = [t for t in m.get("tags", []) if not t.startswith(_SKIP_TAG_PREFIXES) and ":" not in t]
    parts = [
        f"{m['id']} is a Hugging Face model",
        f"for {m['pipeline_tag']}" if m.get("pipeline_tag") else "",
        f"by {m['author']}" if m.get("author") else "",
    ]
    lines = [" ".join(p for p in parts if p) + "."]
    if card.get("license"):
        lines.append(f"License: {card['license']}.")
    if base:
        lines.append(f"Base model: {base}.")
    if m.get("library_name"):
        lines.append(f"Library: {m['library_name']}.")
    if tags:
        lines.append(f"Tags: {', '.join(tags[:15])}.")
    return " ".join(lines)


def _fetch_models(cfg: Settings, source: Source, since: datetime) -> list[RawItem]:
    # Trending is itself a "this week" signal, so accept models up to max_age_days old.
    max_age = timedelta(days=source.extra.get("max_age_days", 30))
    oldest = min(since, datetime.now(timezone.utc) - max_age)
    now = datetime.now(timezone.utc)
    with make_client(cfg) as client:
        try:
            models = get(client, cfg, source.url).json()
        except httpx.HTTPStatusError as exc:
            if not source.fallback_url or exc.response.status_code != 400:
                raise
            log.warning("%s: %s rejected (HTTP 400), falling back to %s", source.name, source.url, source.fallback_url)
            models = get(client, cfg, source.fallback_url).json()
    items = []
    for m in models:
        created = parse_iso(m["createdAt"]) if m.get("createdAt") else None
        if not created or created < oldest:
            continue
        items.append(
            make_item(
                cfg, source,
                title=m["id"],
                url=HF_MODEL_URL.format(id=m["id"]),
                published_at=created,
                text=_model_text(m),
                metrics={"likes": m.get("likes"), "downloads": m.get("downloads"), "trending": m.get("trendingScore")},
                fetched_at=now,
            )
        )
    return items


def _fetch_papers(cfg: Settings, source: Source, since: datetime) -> list[RawItem]:
    """The endpoint returns one day per call; walk every day in the window."""
    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    items: list[RawItem] = []
    day = since.date()
    with make_client(cfg) as client:
        while day <= now.date():
            for entry in get(client, cfg, source.url, params={"date": day.isoformat()}).json():
                paper = entry["paper"]
                if paper["id"] in seen:
                    continue
                seen.add(paper["id"])
                published = paper.get("submittedOnDailyAt") or entry.get("publishedAt") or paper["publishedAt"]
                items.append(
                    make_item(
                        cfg, source,
                        title=entry.get("title") or paper["title"],
                        url=HF_PAPER_URL.format(id=paper["id"]),
                        published_at=parse_iso(published),
                        text=entry.get("summary") or paper.get("summary", ""),
                        metrics={"upvotes": paper.get("upvotes"), "comments": entry.get("numComments")},
                        fetched_at=now,
                    )
                )
            day += timedelta(days=1)
    return items
