"""Render a Digest into a single self-contained, print-ready A4 HTML report: NSE masthead + numbered points."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from collections import Counter
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from digest.config import Settings
from digest.models import Digest, Initiative

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_NAME = "digest.html.j2"


def source_counts(raw_path: Path) -> Counter[str]:
    """Items per source in a RawItem JSONL snapshot (for the footer). Only the `source` key is read."""
    counts: Counter[str] = Counter()
    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                counts[json.loads(line).get("source", "unknown")] += 1
    return counts


def link_label(url: str) -> str:
    """Short, readable link text: host without 'www.'."""
    host = urlsplit(url).netloc
    return host.removeprefix("www.") or url


def oss_label(item: Initiative) -> str | None:
    """'Open source · apache-2.0', 'Closed', 'License: x', or None when nothing is known."""
    if item.open_source is True:
        return f"Open source · {item.license}" if item.license else "Open source"
    if item.open_source is False:
        return f"Closed · {item.license}" if item.license else "Closed"
    return f"License: {item.license}" if item.license else None


def logo_data_uri(path: Path) -> str | None:
    """Embed the logo so the HTML stays a single file; None (text fallback) if it is missing."""
    if not path.is_file():
        log.warning("logo not found at %s; using text fallback", path)
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _entry(it: Initiative, cfg: Settings) -> dict:
    rc = cfg.render
    return {
        "rank": it.rank,
        "title": it.title,
        "category": it.category,
        "color": rc.category_colors.get(it.category, rc.category_colors.get("Other", "#6b7280")),
        "what": it.what_it_is,
        "why": it.why_it_matters,
        "oss": oss_label(it),
        "links": [{"url": str(u), "label": link_label(str(u))} for u in it.sources],
        "score": f"{it.trend_score:.2f}",
        "published": it.published.strftime(rc.short_date_format),
        "extractive": it.summary_mode == "extractive",
    }


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(digest: Digest, cfg: Settings, counts: Counter[str] | None = None) -> str:
    """Return the digest as one HTML string with inline CSS and an embedded logo."""
    rc = cfg.render
    end = digest.generated_at
    start = end - timedelta(days=digest.window_days)
    srcs = sorted((counts or Counter()).items(), key=lambda kv: (-kv[1], kv[0]))
    if len(digest.top10) != cfg.generate.top_n:
        log.warning("rendering %d items (expected %d)", len(digest.top10), cfg.generate.top_n)
    return _environment().get_template(TEMPLATE_NAME).render(
        title=rc.title,
        org=rc.org,
        logo=logo_data_uri(rc.logo_path),
        c=rc.colors,
        date=end.strftime(rc.date_format),
        window_start=start.strftime(rc.short_date_format),
        window_end=end.strftime(rc.short_date_format),
        window_days=digest.window_days,
        disclaimer=rc.disclaimer,
        items=[_entry(it, cfg) for it in sorted(digest.top10, key=lambda i: i.rank)],
        quick_hits=[{"title": h.title, "url": str(h.url), "label": link_label(str(h.url))} for h in digest.quick_hits],
        sources=srcs,
        total_items=sum(n for _, n in srcs),
        generated_at=end.strftime(f"{rc.short_date_format} %H:%M UTC"),
    )
