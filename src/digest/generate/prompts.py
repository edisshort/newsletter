"""Prompt templates and the LLM output schema (one call per cluster)."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from digest.config import Settings
from digest.models import Category, Cluster, RawItem

_HF_TAGS = re.compile(r"\s*Tags: .*$")


class LLMInitiative(BaseModel):
    """What the LLM writes. rank, trend_score, published and summary_mode are filled in code."""

    title: str = Field(description="Short name of the initiative, e.g. the model or product name")
    category: Category
    what_it_is: str = Field(description="What was released or announced. Max 40 words.")
    why_it_matters: str = Field(description="Why a practitioner should care. Max 30 words.")
    open_source: bool | None = Field(description="true/false only if the text says so, else null")
    license: str | None = Field(description="License name only if stated in the text, else null")
    sources: list[str] = Field(description="URLs copied exactly from the provided items")


SYSTEM_PROMPT = """You write entries for an internal daily digest of important AI initiatives.
Rules:
- Use ONLY facts stated in the provided items. Do not add outside knowledge, background
  explanations or likely benefits that the items do not state. Keep names and attributions
  exactly as written (e.g. "the creator of X" is a person, not "the X team").
- If the items say little, keep both fields short and restate only what the text says.
- Do not mention likes, downloads, points, upvotes, stars, or the site an item was posted on.
  Do not copy tag lists or model-card metadata; describe what the thing is and does.
- why_it_matters must add something what_it_is does not say (the capability, result or use it
  enables, as stated in the text). Do not repeat what_it_is.
- Be concise, factual and neutral. No hype words (revolutionary, groundbreaking, game-changing).
- Every model name, product name and number you write must appear in the provided text, with the
  same qualifier ("average", "up to", "under", "over") the text puts next to it.
- Keep the strength of every claim exactly as stated: "gold-medal level" is not "won gold",
  "validated up to" is not "supports", and keep caveats such as "unofficial" or "preliminary".
- why_it_matters must be supported by the text: a stated result, number or capability. Do not
  write generic benefits ("for creative workflows", "for developers") the text does not state.
- what_it_is: at most {what_words} words. why_it_matters: at most {why_words} words.
- category must be one of: {categories}.
- open_source and license: set only if the text states them; otherwise null.
- sources: copy 1-3 URLs exactly as given in the items; never invent URLs.
- Reply with a single JSON object and nothing else."""

USER_TEMPLATE = """Cluster headline: {headline}

Items:
{items}

Write one digest entry for this cluster."""

SCHEMA_HINT = "\n\nThe JSON object must match this JSON schema:\n{schema}"

CATEGORIES: tuple[str, ...] = Category.__args__  # type: ignore[attr-defined]


def system_prompt(cfg: Settings, include_schema: bool) -> str:
    g = cfg.generate
    text = SYSTEM_PROMPT.format(
        what_words=g.what_it_is_max_words, why_words=g.why_it_matters_max_words, categories=", ".join(CATEGORIES)
    )
    if include_schema:
        text += SCHEMA_HINT.format(schema=json.dumps(LLMInitiative.model_json_schema()))
    return text


def is_paper_cluster(cluster: Cluster) -> bool:
    return all(it.type == "paper" for it in cluster.items)


def item_body(item: RawItem) -> str:
    """Item text without the HF tag list or a repeated title prefix."""
    text = _HF_TAGS.sub("", item.text) if item.type == "model" else item.text
    if item.type in ("news", "discussion") and text.lower().startswith(item.title.lower()):
        text = text[len(item.title):].lstrip(" .:-")
    return text


def prompt_items(cluster: Cluster, cfg: Settings) -> list[RawItem]:
    """Items the LLM sees. Paper clusters group distinct papers on one theme, so only the top one."""
    return cluster.items[:1] if is_paper_cluster(cluster) else cluster.items[: cfg.generate.max_items_per_cluster]


def user_prompt(cluster: Cluster, cfg: Settings) -> str:
    g = cfg.generate
    blocks = []
    for n, item in enumerate(prompt_items(cluster, cfg), start=1):
        blocks.append(
            f"[{n}] title: {item.title}\n    url: {item.url}\n"
            f"    text: {item_body(item)[: g.item_text_chars] or '(no text)'}"
        )
    return USER_TEMPLATE.format(headline=cluster.headline, items="\n\n".join(blocks))


class LLMOverview(BaseModel):
    """Page headline and lead paragraph for the week, written from the validated entries only."""

    headline: str = Field(description="Headline for the week's issue. Max {headline_words} words.")
    overview: str = Field(description="Lead paragraph covering the main developments. Max {overview_words} words.")


OVERVIEW_SYSTEM_PROMPT = """You write the lead paragraph of an internal weekly AI digest.
You are given the week's entries, already written and checked. Rules:
- Use ONLY facts in the entries. No outside knowledge, no predictions, no hype words.
- Every name and number must appear in the entries, with the same qualifier.
- headline: at most {headline_words} words, naming the most important development (entry 1 ranks highest).
- overview: one paragraph of at most {overview_words} words that connects the main developments,
  leading with entry 1. Plain, factual, like a briefing note.
- Reply with a single JSON object with keys "headline" and "overview" and nothing else."""


def overview_system_prompt(cfg: Settings) -> str:
    g = cfg.generate
    return OVERVIEW_SYSTEM_PROMPT.format(headline_words=g.headline_max_words, overview_words=g.overview_max_words)


def entries_text(initiatives: list) -> str:
    """The validated entries as the overview prompt (and its grounding text) shows them."""
    return "\n\n".join(
        f"[{it.rank}] {it.title} ({it.category})\nWhat: {it.what_it_is}\nWhy: {it.why_it_matters}"
        for it in initiatives
    )


def retry_prompt(errors: list[str]) -> str:
    return "Your previous answer had these problems:\n- " + "\n- ".join(errors) + "\nReturn a corrected JSON object."


ResponseMode = Literal["json_schema", "json_object"]
