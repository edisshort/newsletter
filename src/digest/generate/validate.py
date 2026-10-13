"""Schema, word-limit, source and grounding checks for generated initiatives, plus the extractive fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import HttpUrl, TypeAdapter, ValidationError

from digest.config import Settings
from digest.generate.prompts import (
    LLMInitiative,
    LLMOverview,
    entries_text,
    is_paper_cluster,
    item_body,
    prompt_items,
)
from digest.models import Cluster, Initiative, RawItem

_URL = TypeAdapter(HttpUrl)
# Capitalised multi-char tokens that contain a digit or hyphen: Qwen3-14B, GLM-5.3, EmbeddingGemma-2, V4.
_NAME_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*\b")
_NUMBER = re.compile(r"(?<![\w.-])\d+(?:[.,]\d+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LICENSE_IN_TEXT = re.compile(r"License: ([\w.+-]+)\.")
_NULLISH = {"", "null", "none", "n/a", "unknown", "not stated"}
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"that", "this", "with", "from", "into", "than", "their", "which", "while", "using", "over", "under"}


@dataclass
class Check:
    """Outcome of validating one LLM answer."""

    initiative: LLMInitiative | None = None
    errors: list[str] = field(default_factory=list)  # fatal: trigger a retry
    grounding_misses: list[str] = field(default_factory=list)
    dropped_sources: list[str] = field(default_factory=list)  # fixed in place, not fatal
    corrections: list[str] = field(default_factory=list)  # unsupported claims removed in place


def _canon(url: str) -> str:
    return str(_URL.validate_python(url))


def word_count(text: str) -> int:
    return len(text.split())


def cluster_text(cluster: Cluster, cfg: Settings) -> str:
    """The text the LLM was shown. Engagement numbers are deliberately excluded, so a summary
    that quotes likes/points fails grounding."""
    return "\n".join(f"{it.title}\n{it.url}\n{item_body(it)}" for it in prompt_items(cluster, cfg))


def _squash(text: str) -> str:
    return "".join(_WORD.findall(text.lower()))


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOPWORDS}


def overlap(a: str, b: str) -> float:
    """Share of the smaller text's content words that also appear in the other."""
    wa, wb = _content_words(a), _content_words(b)
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0


def open_source_supported(text: str, license_: str | None, cfg: Settings) -> bool:
    low = text.lower()
    if any(m in low for m in cfg.generate.open_source_markers):
        return True
    return license_ is not None and _squash(license_) in {_squash(x) for x in cfg.generate.open_licenses}


def grounding_misses(summary: str, source_text: str) -> list[str]:
    """Model-name tokens and numbers in `summary` that do not occur in `source_text`."""
    haystack = source_text.lower()
    numbers_in_source = {n.replace(",", "") for n in _NUMBER.findall(source_text)}
    misses = []
    for tok in _NAME_TOKEN.findall(summary):
        if len(tok) > 1 and (any(ch.isdigit() for ch in tok) or "-" in tok) and tok.lower() not in haystack:
            misses.append(tok)
    for num in _NUMBER.findall(summary):
        if num.replace(",", "") not in numbers_in_source:
            misses.append(num)
    return sorted(set(misses))


def qualifier_misses(summary: str, source_text: str, cfg: Settings) -> list[str]:
    """Numbers whose qualifier in the summary ("under 0.37") has no same-meaning qualifier
    within the window before that number anywhere in the source ("an average of $0.37")."""
    g = cfg.generate
    low_src = source_text.lower()
    misses = []
    for group, words in g.number_qualifiers.items():
        alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        for m in re.finditer(rf"\b(?:{alt})\s+[$~≈]?(\d[\d.,]*)", summary.lower()):
            num = m.group(1).rstrip(".,")
            found = False
            for n in re.finditer(rf"(?<![\d.]){re.escape(num)}(?![\d])", low_src):
                window = low_src[max(0, n.start() - g.qualifier_window_chars): n.start()]
                if any(re.search(rf"\b{re.escape(w)}\b", window) for w in words):
                    found = True
                    break
            if not found:
                misses.append(f"{m.group(0).strip()} (qualifier '{group}' not in source)")
    return sorted(set(misses))


def check(raw_json: str, cluster: Cluster, cfg: Settings) -> Check:
    g = cfg.generate
    result = Check()
    try:
        data = json.loads(raw_json)
        draft = LLMInitiative.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        result.errors.append(f"invalid JSON for the schema: {str(exc).splitlines()[0][:300]}")
        return result

    if word_count(draft.what_it_is) > g.what_it_is_max_words:
        result.errors.append(f"what_it_is has {word_count(draft.what_it_is)} words; max {g.what_it_is_max_words}")
    if word_count(draft.why_it_matters) > g.why_it_matters_max_words:
        result.errors.append(f"why_it_matters has {word_count(draft.why_it_matters)} words; max {g.why_it_matters_max_words}")
    if not draft.what_it_is.strip() or not draft.title.strip():
        result.errors.append("title and what_it_is must not be empty")

    if overlap(draft.what_it_is, draft.why_it_matters) > g.max_what_why_overlap:
        result.errors.append("why_it_matters repeats what_it_is; state the result or use it enables instead")

    if draft.license is not None and draft.license.strip().lower() in _NULLISH:
        draft.license = None

    # Claims about openness must be backed by the text the LLM saw; drop them otherwise.
    text = cluster_text(cluster, cfg)
    if draft.license is not None and _squash(draft.license) not in _squash(text):
        result.corrections.append(f"license {draft.license!r} not in source text")
        draft.license = None
    if draft.open_source and not open_source_supported(text, draft.license, cfg):
        result.corrections.append("open_source=true not supported by source text")
        draft.open_source = None

    if is_paper_cluster(cluster):
        if draft.category != "Research":
            result.corrections.append(f"category {draft.category} -> Research (paper)")
            draft.category = "Research"

    # Sources must come from the cluster; drop anything else, fall back to the top item.
    allowed = {_canon(str(it.url)): str(it.url) for it in prompt_items(cluster, cfg)}
    kept: list[str] = []
    for url in draft.sources:
        try:
            canon = _canon(url)
        except ValidationError:
            canon = None
        if canon in allowed and allowed[canon] not in kept:
            kept.append(allowed[canon])
        else:
            result.dropped_sources.append(url)
    draft.sources = kept[: g.max_sources] or [str(cluster.items[0].url)]

    summary = f"{draft.title}\n{draft.what_it_is}\n{draft.why_it_matters}\n{draft.license or ''}"
    result.grounding_misses = grounding_misses(summary, text) + qualifier_misses(summary, text, cfg)
    if result.grounding_misses:
        result.errors.append(
            "these names/numbers do not appear in the provided text, remove or correct them: "
            + ", ".join(result.grounding_misses)
        )
    result.initiative = draft
    return result


# --- extractive fallback --------------------------------------------------------

def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "…"


def _first_sentences(text: str, max_words: int) -> str:
    out: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        if out and word_count(" ".join(out + [sentence])) > max_words:
            break
        out.append(sentence)
    return _truncate_words(" ".join(out), max_words)


def _category(cluster: Cluster, cfg: Settings) -> str:
    for category, keywords in cfg.generate.category_keywords.items():
        if any(re.search(rf"\b{re.escape(k)}", cluster.headline, re.IGNORECASE) for k in keywords):
            return category
    if is_paper_cluster(cluster):
        return "Research"
    return cfg.generate.anchor_categories.get(cluster.best_anchor, "Other")


def _license(items: list[RawItem], cfg: Settings) -> tuple[bool | None, str | None]:
    for it in items:
        m = _LICENSE_IN_TEXT.search(it.text)
        if m:
            lic = m.group(1)
            return (True if lic.lower() in cfg.generate.open_licenses else None), lic
    return None, None


def _engagement_phrase(item: RawItem, metrics: list[str]) -> str:
    """Engagement of one item only: clusters can hold several distinct items, so sums would mislead."""
    present = [(k, item.metrics[k]) for k in metrics if item.metrics.get(k)]
    if not present:
        return ""
    k, v = max(present, key=lambda kv: kv[1])
    return f"; the top item has {int(v):,} {k}"


def _sources(cluster: Cluster, max_sources: int) -> list[str]:
    """Top item, then the first item of each other source. Paper clusters often group distinct
    papers on one theme, so they link only the top item."""
    if is_paper_cluster(cluster):
        return [str(cluster.items[0].url)]
    seen: set[str] = set()
    urls: list[str] = []
    for it in cluster.items:
        if it.source not in seen:
            seen.add(it.source)
            urls.append(str(it.url))
    return urls[:max_sources]


def extractive(cluster: Cluster, cfg: Settings) -> LLMInitiative:
    """Deterministic summary built only from the cluster's own text and metadata."""
    g = cfg.generate
    top = cluster.items[0]
    open_source, license_ = _license(cluster.items, cfg)
    n_sources = len(cluster.sources)
    why = (
        f"Seen in {n_sources} source{'s' if n_sources != 1 else ''} "
        f"({', '.join(cluster.sources)}){_engagement_phrase(top, cfg.score.engagement_metrics)}."
    )
    return LLMInitiative(
        title=_truncate_words(cluster.headline, 16),
        category=_category(cluster, cfg),
        what_it_is=_first_sentences(item_body(top) or top.title, g.what_it_is_max_words),
        why_it_matters=_truncate_words(why, g.why_it_matters_max_words),
        open_source=open_source,
        license=license_,
        sources=_sources(cluster, g.max_sources),
    )


# --- weekly overview ------------------------------------------------------------

def check_overview(raw_json: str, initiatives: list[Initiative], cfg: Settings) -> tuple[LLMOverview | None, list[str]]:
    g = cfg.generate
    try:
        draft = LLMOverview.model_validate(json.loads(raw_json))
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, [f"invalid JSON for the schema: {str(exc).splitlines()[0][:300]}"]
    errors = []
    if not draft.headline.strip() or not draft.overview.strip():
        errors.append("headline and overview must not be empty")
    if word_count(draft.headline) > g.headline_max_words:
        errors.append(f"headline has {word_count(draft.headline)} words; max {g.headline_max_words}")
    if word_count(draft.overview) > g.overview_max_words:
        errors.append(f"overview has {word_count(draft.overview)} words; max {g.overview_max_words}")
    source = entries_text(initiatives)
    text = f"{draft.headline}\n{draft.overview}"
    misses = grounding_misses(text, source) + qualifier_misses(text, source, cfg)
    if misses:
        errors.append("these names/numbers do not appear in the entries, remove or correct them: " + ", ".join(misses))
    return draft, errors


def extractive_overview(initiatives: list[Initiative], cfg: Settings) -> LLMOverview:
    """Lead item title as headline; first sentence of the top items' what_it_is as the paragraph."""
    g = cfg.generate
    if not initiatives:
        return LLMOverview(headline="", overview="")
    firsts = [_SENTENCE_END.split(it.what_it_is.strip())[0] for it in initiatives[: g.overview_fallback_items]]
    return LLMOverview(
        headline=_truncate_words(initiatives[0].title, g.headline_max_words),
        overview=_truncate_words(" ".join(firsts), g.overview_max_words),
    )
