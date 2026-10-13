"""Embedding, clustering and scoring. Runs offline on the airgapped server."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

from digest.config import Settings, Source
from digest.models import Cluster, RawItem
from digest.process import cluster, embed, score

log = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    ranked: list[Cluster]  # every relevant cluster, best first
    selected: list[Cluster]  # diversity-filtered candidates for generation
    raw_relevance: dict[int, float]  # cluster_id -> un-normalised cosine (for tuning)
    dropped_irrelevant: int


def run(cfg: Settings, sources: list[Source], items: list[RawItem], model=None) -> ProcessResult:
    if not items:
        raise ValueError("no input items")
    random.seed(cfg.score.seed)
    np.random.seed(cfg.score.seed)

    items = sorted(items, key=lambda it: it.id)  # input order must not affect output
    model = model or embed.load_model(cfg)
    vectors = embed.embed_items(model, items, cfg)
    anchors = cfg.relevance.anchors
    anchor_vecs = embed.embed_texts(model, [cfg.embed.anchor_prefix + a for a in anchors], cfg)

    labels = cluster.cluster_embeddings(vectors, cfg.cluster.cluster_threshold)
    log.info("agglomerative clustering: %d items -> %d clusters", len(items), len(set(labels.tolist())))
    if cfg.cluster.entity_merge:
        labels = cluster.merge_by_entity(items, labels, cfg)
    index_groups = cluster.group(labels)
    log.info("after entity merge: %d clusters", len(index_groups))

    trust = {s.name: s.trust_weight for s in sources}
    coverage_group = {s.name: s.coverage_group or s.name for s in sources}
    reference = max(it.fetched_at for it in items)  # snapshot time, so reruns are identical
    sc = cfg.score
    groups: list[list[RawItem]] = []
    raws: list[score.RawScores] = []
    dropped = 0
    for idxs in index_groups:
        rel, anchor = score.relevance(score.centroid(vectors[idxs]), anchor_vecs, anchors)
        if rel < cfg.relevance.relevance_min:
            dropped += 1
            continue
        members = [items[i] for i in idxs]
        groups.append(members)
        raws.append(
            score.RawScores(
                relevance=rel,
                best_anchor=anchor,
                coverage=float(len({coverage_group.get(it.source, it.source) for it in members})),
                engagement=score.engagement(members, sc.engagement_metrics),
                recency=score.recency(members, reference, sc.recency_half_life_h),
                authority=score.authority(members, trust, sc.default_trust_weight),
            )
        )
    log.info("relevance filter dropped %d of %d clusters (min %.2f)", dropped, len(index_groups), cfg.relevance.relevance_min)

    ranked = score.build_clusters(groups, raws, cfg, trust)
    raw_rel = {}
    for c in ranked:
        # map back to raw relevance via the (unique) item ids of the cluster
        ids = {it.id for it in c.items}
        raw_rel[c.cluster_id] = next(r.relevance for g, r in zip(groups, raws) if {it.id for it in g} == ids)
    selected = score.select_diverse(
        ranked, sc.max_per_anchor, sc.top_n_candidates, sc.max_paper_clusters, sc.min_text_chars,
        sc.launch_title_markers, sc.launch_hosts,
    )
    return ProcessResult(ranked, selected, raw_rel, dropped)
