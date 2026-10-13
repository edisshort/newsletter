"""Full-page text extraction (trafilatura) for items whose feed text is too short."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import trafilatura

from digest.config import Settings
from digest.fetch.common import clean_text, get, make_client
from digest.models import RawItem

log = logging.getLogger(__name__)


def enrich(cfg: Settings, items: list[RawItem]) -> list[RawItem]:
    """Return items with `text` replaced by extracted page text where that is longer.

    Types in `extract_append_types` (HF models) are always extracted and the page text is appended
    to the existing metadata line instead of replacing it.
    Failures are logged at DEBUG and leave the item unchanged (blogs often block bots).
    """
    append = set(cfg.fetch.extract_append_types)
    targets = [
        i for i, it in enumerate(items)
        if it.type in append
        or (it.type in cfg.fetch.extract_types and len(it.text) < cfg.fetch.extract_min_chars)
    ]
    if not targets:
        return items
    log.info("extracting full text for %d short items", len(targets))
    out = list(items)
    with make_client(cfg) as client, ThreadPoolExecutor(cfg.fetch.extract_workers) as pool:

        def work(idx: int) -> tuple[int, str | None]:
            item = items[idx]
            try:
                page = get(client, cfg, str(item.url)).text
                return idx, trafilatura.extract(page, include_comments=False, include_tables=False)
            except Exception as exc:  # noqa: BLE001 - extraction is best-effort
                log.debug("extract failed for %s: %s", item.url, exc)
                return idx, None

        ok = 0
        for idx, text in pool.map(work, targets):
            text = clean_text(text)
            if text and items[idx].type in append:
                text = f"{items[idx].text}\n\n{text}"
            text = text[: cfg.fetch.max_text_chars]
            if len(text) > len(items[idx].text):
                out[idx] = items[idx].model_copy(update={"text": text})
                ok += 1
    log.info("extraction improved %d/%d items", ok, len(targets))
    return out
