"""Relevance, trend score components and diversity-aware selection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

import numpy as np

from digest.config import Settings
from digest.models import Cluster, RawItem

_BASE_MODEL = re.compile(r"Base model: ([\w.-]+)/")


@dataclass
class RawScores:
    """Un-normalised components for one cluster (kept for logging/tuning)."""

    relevance: float
    best_anchor: str
    coverage: float
    engagement: float | None  # None: no item reports an engagement metric
    recency: float
    authority: float


def centroid(vectors: np.ndarray) -> np.ndarray:
    c = vectors.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm else c


def relevance(centroid_vec: np.ndarray, anchor_vecs: np.ndarray, anchors: list[str]) -> tuple[float, str]:
    sims = anchor_vecs @ centroid_vec
    best = int(np.argmax(sims))
    return float(sims[best]), anchors[best]


def engagement(items: list[RawItem], metrics: list[str]) -> float | None:
    """log1p of summed metrics, or None when no item reports any of them (e.g. blog posts)."""
    if not any(m in it.metrics for it in items for m in metrics):
        return None
    return math.log1p(sum(it.metrics.get(m, 0.0) for it in items for m in metrics))


def impute_missing(values: list[float | None]) -> list[float]:
    """Replace None with the median of the known values (0 if none are known)."""
    known = sorted(v for v in values if v is not None)
    median = float(np.median(known)) if known else 0.0
    return [median if v is None else v for v in values]


def recency(items: list[RawItem], reference: datetime, half_life_h: float) -> float:
    newest = max(it.published_at for it in items)
    age_h = max((reference - newest).total_seconds() / 3600.0, 0.0)
    return 0.5 ** (age_h / half_life_h)


def authority(items: list[RawItem], trust: dict[str, float], default: float) -> float:
    return sum(trust.get(it.source, default) for it in items) / len(items)


def is_derivative(item: RawItem) -> bool:
    """HF model whose card names a base model from a different owner (fine-tune, quant, merge)."""
    if item.type != "model" or "/" not in item.title:
        return False
    m = _BASE_MODEL.search(item.text)
    return bool(m) and m.group(1).lower() != item.title.split("/", 1)[0].lower()


def has_text(cluster: Cluster, min_chars: int) -> bool:
    return any(len(it.text) >= min_chars for it in cluster.items)


def looks_like_launch(cluster: Cluster, title_markers: list[str], hosts: list[str]) -> bool:
    """Clusters made only of discussions (HN) must point at a launch, not commentary.
    Any non-discussion item (model, release, blog, paper) counts as a launch on its own."""
    if not title_markers and not hosts:
        return True
    if any(it.type != "discussion" for it in cluster.items):
        return True
    for it in cluster.items:
        title = f" {it.title.lower()}"
        host = urlsplit(str(it.url)).netloc.lower().removeprefix("www.")
        if any(m in title for m in title_markers) or host in hosts:
            return True
    return False


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values, default=0.0), max(values, default=0.0)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def order_items(items: list[RawItem], trust: dict[str, float], default: float, metrics: list[str]) -> list[RawItem]:
    """Most authoritative first: trust weight, then engagement, then newest, then id."""
    return sorted(
        items,
        key=lambda it: (
            -trust.get(it.source, default),
            -sum(it.metrics.get(m, 0.0) for m in metrics),
            -it.published_at.timestamp(),
            it.id,
        ),
    )


def build_clusters(
    groups: list[list[RawItem]], raw: list[RawScores], cfg: Settings, trust: dict[str, float]
) -> list[Cluster]:
    """Normalise each component across clusters, combine with configured weights, rank."""
    sc = cfg.score
    w = sc.weights
    eng = [r.engagement for r in raw]
    eng_filled = impute_missing(eng) if sc.impute_missing_engagement else [v or 0.0 for v in eng]
    norm = {
        name: minmax([getattr(r, name) for r in raw])
        for name in ("relevance", "coverage", "recency", "authority")
    }
    norm["engagement"] = minmax(eng_filled)
    clusters = []
    for i, (items, r) in enumerate(zip(groups, raw)):
        comp = {name: round(norm[name][i], 6) for name in norm}
        score = (
            w.coverage * comp["coverage"]
            + w.engagement * comp["engagement"]
            + w.relevance * comp["relevance"]
            + w.recency * comp["recency"]
            + w.authority * comp["authority"]
        )
        ordered = order_items(items, trust, sc.default_trust_weight, sc.engagement_metrics)
        if is_derivative(ordered[0]):
            score *= sc.derivative_penalty
        clusters.append(
            Cluster(
                cluster_id=0,  # assigned after ranking
                items=ordered,
                headline=ordered[0].title,
                sources=sorted({it.source for it in items}),
                score=round(score, 6),
                best_anchor=r.best_anchor,
                **comp,
            )
        )
    clusters.sort(key=lambda c: (-c.score, -c.coverage, c.headline, c.items[0].id))
    for rank, c in enumerate(clusters, start=1):
        c.cluster_id = rank
    return clusters


def select_diverse(
    ranked: list[Cluster],
    max_per_anchor: int,
    top_n: int,
    max_paper_clusters: int | None = None,
    min_text_chars: int = 0,
    launch_title_markers: list[str] | None = None,
    launch_hosts: list[str] | None = None,
) -> list[Cluster]:
    """Walk the ranking, allowing at most `max_per_anchor` clusters per best_anchor.

    If `max_paper_clusters` is set, clusters made only of papers are capped by that instead of
    the anchor limit (papers vastly outnumber launches and would otherwise use up anchor slots).
    Clusters with no item of at least `min_text_chars` text are skipped: there is nothing to summarise.
    Discussion-only clusters that do not look like a launch (opinion, commentary) are skipped.
    """
    per_anchor: dict[str, int] = {}
    papers = 0
    chosen = []
    for c in ranked:
        if len(chosen) >= top_n:
            break
        if not has_text(c, min_text_chars):
            continue
        if not looks_like_launch(c, launch_title_markers or [], launch_hosts or []):
            continue
        if max_paper_clusters is not None and all(it.type == "paper" for it in c.items):
            if papers >= max_paper_clusters:
                continue
            papers += 1
        else:
            if per_anchor.get(c.best_anchor, 0) >= max_per_anchor:
                continue
            per_anchor[c.best_anchor] = per_anchor.get(c.best_anchor, 0) + 1
        chosen.append(c)
    return chosen
