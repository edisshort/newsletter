"""Load config/settings.yaml and config/sources.yaml (plus env vars) into typed settings.

Only depends on pydantic + pyyaml so the fetch machine can import it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


class PathsCfg(BaseModel):
    raw: Path
    processed: Path
    top10: Path
    html: Path
    fixture: Path


class FetchCfg(BaseModel):
    window_days: int
    timeout_s: float
    retries: int
    backoff_s: float
    user_agent: str
    extract_min_chars: int
    extract_types: list[str]
    extract_workers: int
    max_text_chars: int
    keep_query_hosts: list[str]
    extract_append_types: list[str] = []
    https_proxy: str | None = None  # from HTTPS_PROXY


class EmbedCfg(BaseModel):
    model_path: str
    max_text_chars: int
    batch_size: int
    device: str
    anchor_prefix: str = ""


class ClusterCfg(BaseModel):
    cluster_threshold: float
    entity_merge: bool
    entity_item_types: list[str]
    entity_max_items: int
    entity_stopwords: list[str]


class RelevanceCfg(BaseModel):
    relevance_min: float
    anchors: list[str]


class ScoreWeights(BaseModel):
    coverage: float
    engagement: float
    relevance: float
    recency: float
    authority: float


class ScoreCfg(BaseModel):
    weights: ScoreWeights
    recency_half_life_h: float
    engagement_metrics: list[str]
    default_trust_weight: float
    max_per_anchor: int
    max_paper_clusters: int | None = None
    derivative_penalty: float = 1.0
    min_text_chars: int = 0
    impute_missing_engagement: bool = False
    launch_title_markers: list[str] = []
    launch_hosts: list[str] = []
    top_n_candidates: int
    table_rows: int
    seed: int


class GenerateCfg(BaseModel):
    top_n: int
    temperature: float
    response_format: str
    max_items_per_cluster: int
    item_text_chars: int
    max_retries: int
    what_it_is_max_words: int
    why_it_matters_max_words: int
    max_sources: int
    max_what_why_overlap: float
    number_qualifiers: dict[str, list[str]] = {}
    headline_max_words: int = 14
    overview_max_words: int = 90
    overview_fallback_items: int = 3
    quick_hits: int = 0
    write_overview: bool = True
    qualifier_window_chars: int = 30
    open_source_markers: list[str]
    open_licenses: list[str]
    report: Path
    anchor_categories: dict[str, str]
    category_keywords: dict[str, list[str]]


class LLMCfg(BaseModel):
    base_url: str
    model: str
    api_key: str
    timeout_s: float
    max_tokens: int
    reasoning_effort: str | None = None


class RenderCfg(BaseModel):
    title: str
    org: str
    logo_path: Path
    disclaimer: str
    date_format: str
    short_date_format: str
    colors: dict[str, str]
    category_colors: dict[str, str]


class Settings(BaseModel):
    paths: PathsCfg
    fetch: FetchCfg
    embed: EmbedCfg
    cluster: ClusterCfg
    relevance: RelevanceCfg
    score: ScoreCfg
    generate: GenerateCfg
    llm: LLMCfg
    render: RenderCfg


class Source(BaseModel):
    """One entry in sources.yaml. Fetcher-specific keys (repos, queries, ...) land in `extra`."""

    name: str
    type: str
    url: str
    item_type: str
    enabled: bool = True
    trust_weight: float = 0.5
    coverage_group: str | None = None  # sources sharing a group count once for coverage
    fallback_url: str | None = None
    note: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


_ENV_OVERRIDES: dict[tuple[str, str], str] = {
    ("embed", "model_path"): "EMBED_MODEL_PATH",
    ("llm", "base_url"): "LLM_BASE_URL",
    ("llm", "model"): "LLM_MODEL",
    ("llm", "api_key"): "LLM_API_KEY",
    ("llm", "reasoning_effort"): "LLM_REASONING_EFFORT",
    ("fetch", "https_proxy"): "HTTPS_PROXY",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(path: Path | None = None) -> Settings:
    raw = _read_yaml(path or CONFIG_DIR / "settings.yaml")
    for (section, key), env in _ENV_OVERRIDES.items():
        value = os.environ.get(env) or (os.environ.get(env.lower()) if env == "HTTPS_PROXY" else None)
        if value:
            raw.setdefault(section, {})[key] = value
    settings = Settings.model_validate(raw)
    missing = set(settings.relevance.anchors) - set(settings.generate.anchor_categories)
    if missing:
        raise ValueError(f"generate.anchor_categories is missing anchors: {sorted(missing)}")
    if not settings.generate.report.is_absolute():
        settings.generate.report = REPO_ROOT / settings.generate.report
    if not settings.render.logo_path.is_absolute():
        settings.render.logo_path = REPO_ROOT / settings.render.logo_path
    # Resolve relative data paths against the repo root so scripts work from any cwd.
    for field in PathsCfg.model_fields:
        p = getattr(settings.paths, field)
        if not p.is_absolute():
            setattr(settings.paths, field, REPO_ROOT / p)
    return settings


def load_sources(path: Path | None = None, enabled_only: bool = False) -> list[Source]:
    raw = _read_yaml(path or CONFIG_DIR / "sources.yaml")
    known = set(Source.model_fields) - {"extra"}
    sources = []
    for entry in raw.get("sources", []):
        base = {k: v for k, v in entry.items() if k in known}
        extra = {k: v for k, v in entry.items() if k not in known}
        sources.append(Source(**base, extra=extra))
    names = [s.name for s in sources]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate source names in sources.yaml: {sorted(dupes)}")
    return [s for s in sources if s.enabled] if enabled_only else sources
