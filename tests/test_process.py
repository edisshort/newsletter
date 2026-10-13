import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from digest.config import REPO_ROOT, load_settings, load_sources
from digest.models import Cluster, RawItem
from digest.process import cluster, score

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.jsonl"
NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)
STOP = set(load_settings().cluster.entity_stopwords)


def _item(title: str, type_: str = "model", source: str = "hf_models", hours_ago: float = 0, **metrics) -> RawItem:
    return RawItem(
        id=title, title=title, url=f"https://x.test/{abs(hash(title))}", source=source, type=type_,
        published_at=NOW - timedelta(hours=hours_ago), metrics=metrics, fetched_at=NOW,
    )


def _cluster(anchor: str, type_: str = "model") -> Cluster:
    return Cluster(cluster_id=0, items=[_item(f"{anchor}-{random.random()}", type_)], headline="h", sources=["s"],
                   relevance=0, coverage=0, engagement=0, recency=0, authority=0, score=0, best_anchor=anchor)


# --- entity extraction --------------------------------------------------------

@pytest.mark.parametrize("title,key", [
    ("EmbeddingGemma 2: an open, lightweight multimodal embedding model", "embeddinggemma2"),
    ("unsloth/embeddinggemma-2-GGUF", "embeddinggemma2"),
    ("One month coding with GLM 5.3 Flash", "glm5.3"),
    ("Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw", "glm5.3"),
    ("autotrust/GLM5.3-Flash-E224-DGX-Spark", "glm5.3"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek4.1"),
    ("vllm-project/vllm v0.31.0", "repo:vllm-project/vllm"),
])
def test_entity_keys(title, key):
    assert key in cluster.entity_keys(title, STOP)


@pytest.mark.parametrize("title", ["24/7 support and/or more", "5 Steps to Create Assets", "Release 5 notes"])
def test_entity_keys_ignores_noise(title):
    assert cluster.entity_keys(title, STOP) == set()


def test_merge_by_entity_unions_clusters():
    cfg = load_settings()
    items = [_item("google/embeddinggemma-2"), _item("EmbeddingGemma 2 launch", "news"), _item("Unrelated", "news")]
    merged = cluster.merge_by_entity(items, np.array([0, 1, 2]), cfg)
    assert merged[0] == merged[1] != merged[2]


def test_merge_by_entity_skips_papers():
    cfg = load_settings()
    items = [_item("Qwen3-14B release"), _item("We fine-tune Qwen3-14B", "paper", "arxiv_cs.CL")]
    merged = cluster.merge_by_entity(items, np.array([0, 1]), cfg)
    assert merged[0] != merged[1]


# --- scoring ------------------------------------------------------------------

def test_minmax_and_constant():
    assert score.minmax([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert score.minmax([2.0, 2.0]) == [0.0, 0.0]


def test_recency_half_life():
    items = [_item("a", hours_ago=48), _item("b", hours_ago=96)]
    assert score.recency(items, NOW, 48) == pytest.approx(0.5)


def test_engagement_sums_configured_metrics_only():
    items = [_item("a", likes=9, downloads=10_000), _item("b", points=10)]
    assert score.engagement(items, ["likes", "points"]) == pytest.approx(np.log1p(19))


def test_select_diverse_caps_anchor_and_papers():
    ranked = [_cluster("A"), _cluster("A"), _cluster("A"), _cluster("B"),
              _cluster("P", "paper"), _cluster("P", "paper"), _cluster("C")]
    chosen = score.select_diverse(ranked, max_per_anchor=2, top_n=10, max_paper_clusters=1)
    assert [c.best_anchor for c in chosen] == ["A", "A", "B", "P", "C"]
    assert len(score.select_diverse(ranked, 2, top_n=2)) == 2


# --- end to end on the fixture (needs the local embedding model) --------------

MODEL_PATH = os.environ.get("EMBED_MODEL_PATH") or str(REPO_ROOT / "models" / "bge-small-en-v1.5")
needs_model = pytest.mark.skipif(not Path(MODEL_PATH).is_dir(), reason="embedding model not available")


@pytest.fixture(scope="module")
def processed():
    from digest import process

    cfg = load_settings()
    cfg.embed.model_path = MODEL_PATH
    items = [RawItem.model_validate_json(line) for line in FIXTURE.read_text().splitlines()]
    model = process.embed.load_model(cfg)
    first = process.run(cfg, load_sources(), items, model=model)
    shuffled = items[:]
    random.Random(0).shuffle(shuffled)
    second = process.run(cfg, load_sources(), shuffled, model=model)
    return cfg, first, second


def _cluster_of(clusters, title_prefix):
    return next(c for c in clusters if any(it.title.startswith(title_prefix) for it in c.items))


@needs_model
def test_near_duplicates_share_a_cluster(processed):
    _, result, _ = processed
    gemma = _cluster_of(result.ranked, "EmbeddingGemma 2")
    titles = {it.title for it in gemma.items}
    assert {"google/embeddinggemma-2", "unsloth/embeddinggemma-2-GGUF"} <= titles
    assert {it.source for it in gemma.items} >= {"blog_deepmind", "hn", "hf_models"}
    glm = _cluster_of(result.ranked, "One month coding with GLM 5.3")
    assert sum("GLM" in it.title for it in glm.items) >= 3


@needs_model
def test_off_topic_items_dropped(processed):
    _, result, _ = processed
    titles = {it.title for c in result.ranked for it in c.items}
    assert "The lamps in my house" not in titles
    assert not any("Bahrain F1" in t for t in titles)


@needs_model
def test_selection_shape(processed):
    cfg, result, _ = processed
    # fixture HN items carry no text, so fewer than top_n_candidates survive the min_text filter
    assert cfg.generate.top_n <= len(result.selected) <= cfg.score.top_n_candidates
    assert all(score.has_text(c, cfg.score.min_text_chars) for c in result.selected)
    assert sum(all(it.type == "paper" for it in c.items) for c in result.selected) <= cfg.score.max_paper_clusters
    scores = [c.score for c in result.ranked]
    assert scores == sorted(scores, reverse=True)
    assert result.selected[0].headline.startswith("EmbeddingGemma 2")
    for c in result.ranked:
        for comp in (c.coverage, c.engagement, c.relevance, c.recency, c.authority):
            assert 0.0 <= comp <= 1.0


@needs_model
def test_rerun_is_identical_even_with_shuffled_input(processed):
    _, first, second = processed
    dump = lambda r: [c.model_dump_json() for c in r.ranked]  # noqa: E731
    assert dump(first) == dump(second)


def test_is_derivative():
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)

    def model(title: str, text: str) -> RawItem:
        return RawItem(id=title, title=title, url=f"https://hf.test/{title}", source="hf_models", type="model",
                       published_at=now, text=text, fetched_at=now)

    assert score.is_derivative(model("autotrust/JEV-27B-VL", "x. Base model: Qwen/Qwen3.8-27B. Tags: a."))
    assert not score.is_derivative(model("google/embeddinggemma-2", "x. License: apache-2.0. Tags: a."))
    assert not score.is_derivative(model("Qwen/Qwen3.8-27B-FP8", "x. Base model: Qwen/Qwen3.8-27B."))


def _hn_cluster(title: str, url: str, type_: str = "discussion") -> Cluster:
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)
    it = RawItem(id=title, title=title, url=url, source="hn", type=type_, published_at=now, text="x" * 300,
                 fetched_at=now)
    return Cluster(cluster_id=1, items=[it], headline=title, sources=["hn"], relevance=1, coverage=0, engagement=1,
                   recency=1, authority=0.5, score=1.0, best_anchor="a")


def test_looks_like_launch():
    cfg = load_settings()
    ok = lambda c: score.looks_like_launch(c, cfg.score.launch_title_markers, cfg.score.launch_hosts)  # noqa: E731
    assert not ok(_hn_cluster("The gap in shared understanding of LLM capability is widening", "https://twitter.com/x/1"))
    assert not ok(_hn_cluster("Why aren't AI companies being charged for hacking?", "https://mortoray.com/why"))
    assert ok(_hn_cluster("Janas: Run LLMs larger than RAM", "https://github.com/prabanta-dev/janas"))
    assert ok(_hn_cluster("Launch HN: Agent37 (YC F26)", "https://news.ycombinator.com/item?id=1"))
    assert ok(_hn_cluster("Strands Decider 2B: a small, open-source, decision model", "https://strandsagents.com/blog"))
    assert ok(_hn_cluster("Some opinion", "https://openai.com/x", type_="news"))  # non-discussion always passes


def test_missing_engagement_is_imputed_with_median():
    assert score.impute_missing([None, 1.0, 3.0, 5.0]) == [3.0, 1.0, 3.0, 5.0]
    assert score.impute_missing([None, None]) == [0.0, 0.0]
    assert score.engagement([_hn_cluster("t", "https://a.test").items[0]], ["points"]) is None
