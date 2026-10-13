"""Sentence embeddings from a local bge-small-en-v1.5 (CPU, normalised)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Never let sentence-transformers / transformers reach the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from digest.config import Settings  # noqa: E402
from digest.models import RawItem  # noqa: E402

log = logging.getLogger(__name__)


def load_model(cfg: Settings) -> SentenceTransformer:
    path = cfg.embed.model_path
    if not path or not Path(path).is_dir():
        raise FileNotFoundError(
            f"embedding model not found at {path!r}; set EMBED_MODEL_PATH to a local bge-small-en-v1.5 folder"
        )
    log.info("loading embedding model from %s", path)
    return SentenceTransformer(path, device=cfg.embed.device, local_files_only=True)


def item_text(item: RawItem, max_chars: int) -> str:
    body = item.text[:max_chars]
    return f"{item.title}. {body}" if body else item.title


def embed_texts(model: SentenceTransformer, texts: list[str], cfg: Settings) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=cfg.embed.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def embed_items(model: SentenceTransformer, items: list[RawItem], cfg: Settings) -> np.ndarray:
    return embed_texts(model, [item_text(it, cfg.embed.max_text_chars) for it in items], cfg)
