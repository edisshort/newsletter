"""LLM summarisation of selected clusters with grounding validation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from digest.config import Settings
from digest.generate import prompts, validate
from digest.generate.llm import LLM
from digest.models import Cluster, Digest, Initiative, QuickHit

log = logging.getLogger(__name__)


@dataclass
class ClusterReport:
    """Per-cluster audit trail written next to top10.json."""

    cluster_id: int
    headline: str
    summary_mode: str = "extractive"
    attempts: int = 0
    errors: list[list[str]] = field(default_factory=list)  # errors per failed attempt
    final_grounding_misses: list[str] = field(default_factory=list)
    dropped_sources: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    llm_error: str | None = None


def summarise(cluster: Cluster, cfg: Settings, llm: LLM | None) -> tuple[prompts.LLMInitiative, ClusterReport]:
    report = ClusterReport(cluster.cluster_id, cluster.headline)
    if llm is not None:
        messages = [
            {"role": "system", "content": prompts.system_prompt(cfg, include_schema=True)},
            {"role": "user", "content": prompts.user_prompt(cluster, cfg)},
        ]
        for _ in range(cfg.generate.max_retries + 1):
            report.attempts += 1
            try:
                answer = llm.complete(messages)
            except Exception as exc:  # noqa: BLE001 - endpoint trouble -> extractive fallback
                report.llm_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
                log.warning("cluster %d: LLM call failed: %s", cluster.cluster_id, report.llm_error)
                break
            result = validate.check(answer, cluster, cfg)
            report.dropped_sources += result.dropped_sources
            report.corrections += result.corrections
            if not result.errors and result.initiative is not None:
                report.summary_mode = "llm"
                report.final_grounding_misses = []
                return result.initiative, report
            report.errors.append(result.errors)
            report.final_grounding_misses = result.grounding_misses
            log.info("cluster %d attempt %d rejected: %s", cluster.cluster_id, report.attempts, "; ".join(result.errors))
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": prompts.retry_prompt(result.errors)},
            ]
        log.warning("cluster %d: falling back to extractive summary", cluster.cluster_id)
    report.final_grounding_misses = []  # extractive text is copied from the source
    return validate.extractive(cluster, cfg), report


def run(
    cfg: Settings, clusters: list[Cluster], llm: LLM | None, window_days: int
) -> tuple[Digest, list[ClusterReport]]:
    """Summarise clusters in score order until top_n initiatives are produced."""
    ordered = sorted(clusters, key=lambda c: (-c.score, c.cluster_id))
    initiatives: list[Initiative] = []
    reports: list[ClusterReport] = []
    for cluster in ordered:
        if len(initiatives) >= cfg.generate.top_n:
            break
        draft, report = summarise(cluster, cfg, llm)
        reports.append(report)
        try:
            initiatives.append(
                Initiative(
                    rank=len(initiatives) + 1,
                    trend_score=cluster.score,
                    published=max(it.published_at for it in cluster.items).date(),
                    summary_mode=report.summary_mode,  # type: ignore[arg-type]
                    **draft.model_dump(),
                )
            )
        except Exception as exc:  # noqa: BLE001 - skip a cluster that cannot form a valid item
            report.summary_mode = "skipped"
            log.error("cluster %d produced an invalid Initiative, skipping: %s", cluster.cluster_id, exc)
    if len(initiatives) < cfg.generate.top_n:
        log.warning("only %d initiatives from %d clusters", len(initiatives), len(clusters))
    overview, overview_report = summarise_overview(initiatives, cfg, llm)
    reports.append(overview_report)
    used = {r.cluster_id for r in reports}
    quick_hits = [
        QuickHit(title=c.headline, url=c.items[0].url, source=c.items[0].source)
        for c in ordered if c.cluster_id not in used
    ][: cfg.generate.quick_hits]
    digest = Digest(
        generated_at=datetime.now(timezone.utc),
        window_days=window_days,
        top10=initiatives,
        headline=overview.headline,
        overview=overview.overview,
        overview_mode=overview_report.summary_mode,  # type: ignore[arg-type]
        quick_hits=quick_hits,
    )
    return digest, reports


OVERVIEW_REPORT_ID = 0  # cluster ids start at 1; 0 marks the weekly overview in the report


def summarise_overview(
    initiatives: list[Initiative], cfg: Settings, llm: LLM | None
) -> tuple[prompts.LLMOverview, ClusterReport]:
    """Headline + lead paragraph from the validated entries, with an extractive fallback."""
    report = ClusterReport(OVERVIEW_REPORT_ID, "weekly overview")
    if llm is not None and initiatives and cfg.generate.write_overview:
        messages = [
            {"role": "system", "content": prompts.overview_system_prompt(cfg)},
            {"role": "user", "content": prompts.entries_text(initiatives)},
        ]
        for _ in range(cfg.generate.max_retries + 1):
            report.attempts += 1
            try:
                answer = llm.complete(messages, prompts.LLMOverview)
            except Exception as exc:  # noqa: BLE001 - endpoint trouble -> extractive fallback
                report.llm_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
                log.warning("overview: LLM call failed: %s", report.llm_error)
                break
            draft, errors = validate.check_overview(answer, initiatives, cfg)
            if draft is not None and not errors:
                report.summary_mode = "llm"
                return draft, report
            report.errors.append(errors)
            log.info("overview attempt %d rejected: %s", report.attempts, "; ".join(errors))
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": prompts.retry_prompt(errors)},
            ]
        log.warning("overview: falling back to extractive summary")
    return validate.extractive_overview(initiatives, cfg), report


def report_dict(reports: list[ClusterReport]) -> dict:
    overview = [r for r in reports if r.cluster_id == OVERVIEW_REPORT_ID]
    reports = [r for r in reports if r.cluster_id != OVERVIEW_REPORT_ID]
    llm_items = [r for r in reports if r.summary_mode == "llm"]
    return {
        "clusters": len(reports),
        "llm": len(llm_items),
        "extractive": len(reports) - len(llm_items),
        "overview": asdict(overview[0]) if overview else None,
        "unresolved_llm_grounding_misses": sum(len(r.final_grounding_misses) for r in llm_items),
        "details": [asdict(r) for r in reports],
    }
