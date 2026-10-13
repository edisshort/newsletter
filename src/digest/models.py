"""Pydantic data contracts shared by every pipeline stage.

Frozen after Phase 1: ask before changing any field.
2026-10-12 (Phase 6, approved): Digest gained headline/overview/overview_mode/quick_hits, all with
defaults so older top10.json files still load.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

ItemType = Literal["model", "paper", "repo", "release", "news", "discussion"]
Category = Literal[
    "LLM", "Embedding", "Agent/Tooling", "Research", "Indic AI", "Regulation", "Other"
]


class RawItem(BaseModel):
    id: str  # sha1 of canonical url
    title: str
    url: HttpUrl
    source: str  # e.g. "hf_models", "arxiv_cs.CL", "hn"
    type: ItemType
    published_at: datetime
    text: str = ""  # extracted body / abstract / model card summary
    metrics: dict[str, float] = Field(default_factory=dict)  # likes, downloads, stars, points, comments
    fetched_at: datetime


class Cluster(BaseModel):
    cluster_id: int
    items: list[RawItem]
    headline: str  # title of the most authoritative item
    sources: list[str]  # distinct source names
    relevance: float
    coverage: float
    engagement: float
    recency: float
    authority: float
    score: float
    best_anchor: str  # which topic anchor it matched


class Initiative(BaseModel):
    rank: int
    title: str
    category: Category
    what_it_is: str  # <= 40 words
    why_it_matters: str  # <= 30 words
    open_source: bool | None
    license: str | None
    sources: list[HttpUrl]  # must come from the cluster's items
    trend_score: float
    published: date
    summary_mode: Literal["llm", "extractive"]


class QuickHit(BaseModel):
    """One-line mention of a candidate that did not make the main list (title copied from source)."""

    title: str
    url: HttpUrl
    source: str


class Digest(BaseModel):
    generated_at: datetime
    window_days: int
    top10: list[Initiative]
    headline: str = ""  # page headline for the week
    overview: str = ""  # lead paragraph summarising the week, grounded in the top10 summaries
    overview_mode: Literal["llm", "extractive", ""] = ""
    quick_hits: list[QuickHit] = Field(default_factory=list)
