"""Agglomerative clustering of item embeddings, then merging clusters that name the same model/repo."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from digest.config import Settings
from digest.models import RawItem

log = logging.getLogger(__name__)

# owner/repo (HF model ids, GitHub repos). Both parts need a letter; avoids "24/7", "and/or".
_OWNER_REPO = re.compile(r"\b([A-Za-z][\w.-]{1,38})/([A-Za-z0-9][\w.-]+[A-Za-z0-9])\b")
# Model-like tokens: letters and digits joined by - or . (Qwen3-14B, GLM-5.3-Flash, DeepSeek-V4.1).
_MODEL_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)+\b")
# "Name 2" / "Name 5.3" written with a space (EmbeddingGemma 2, GLM 5.3 Flash).
_NAME_SPACE_VERSION = re.compile(r"\b([A-Z][A-Za-z]{2,})\s+v?(\d+(?:\.\d+)?)\b")
# Leading family + version of a token: "embeddinggemma-2-gguf" -> ("embeddinggemma", "2").
_FAMILY_VERSION = re.compile(r"^([a-z]{3,})[-_ ]?v?(\d+(?:\.\d+)?)")


def _family_key(token: str, stopwords: set[str]) -> str | None:
    m = _FAMILY_VERSION.match(token.lower())
    if not m or m.group(1) in stopwords:
        return None
    return f"{m.group(1)}{m.group(2)}"


def entity_keys(title: str, stopwords: set[str]) -> set[str]:
    """Names in a title that identify a specific model or repo, normalised for matching."""
    keys: set[str] = set()
    candidates: list[str] = []
    for owner, repo in _OWNER_REPO.findall(title):
        if owner.lower() in stopwords or repo.lower() in stopwords:
            continue
        keys.add(f"repo:{owner.lower()}/{repo.lower()}")
        candidates.append(repo)
    candidates += _MODEL_TOKEN.findall(title)
    candidates += [f"{name}-{ver}" for name, ver in _NAME_SPACE_VERSION.findall(title)]
    for token in candidates:
        key = _family_key(token, stopwords)
        if key:
            keys.add(key)
    return keys


def cluster_embeddings(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster labels for each row. Deterministic for a given input order."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
    )
    return model.fit_predict(embeddings)


def merge_by_entity(items: list[RawItem], labels: np.ndarray, cfg: Settings) -> np.ndarray:
    """Union clusters whose items share an entity key. Over-common keys are ignored as too generic."""
    c = cfg.cluster
    stopwords = set(c.entity_stopwords)
    key_to_items: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        if item.type not in c.entity_item_types:
            continue
        for key in entity_keys(item.title, stopwords):
            key_to_items[key].append(idx)

    parent = {int(lab): int(lab) for lab in set(labels.tolist())}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = 0
    for key in sorted(key_to_items):
        idxs = key_to_items[key]
        if len(idxs) > c.entity_max_items:
            log.debug("entity %r on %d items: too generic, not merging", key, len(idxs))
            continue
        roots = sorted({find(int(labels[i])) for i in idxs})
        for other in roots[1:]:
            parent[other] = roots[0]
            merges += 1
            log.debug("merged clusters on entity %r", key)
    if merges:
        log.info("entity merge joined %d cluster pairs", merges)
    return np.array([find(int(lab)) for lab in labels])


def group(labels: np.ndarray) -> list[list[int]]:
    """Item indices per cluster, clusters ordered by their smallest item index (stable)."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels.tolist()):
        buckets[lab].append(idx)
    return sorted(buckets.values(), key=lambda idxs: idxs[0])
