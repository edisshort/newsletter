from digest.config import REPO_ROOT, load_settings, load_sources


def test_settings_load_and_weights_sum_to_one():
    s = load_settings()
    w = s.score.weights
    assert abs(w.coverage + w.engagement + w.relevance + w.recency + w.authority - 1.0) < 1e-9
    assert s.paths.raw.is_relative_to(REPO_ROOT)
    assert len(s.relevance.anchors) >= 8


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.internal:8000/v1")
    monkeypatch.setenv("EMBED_MODEL_PATH", "/models/bge-small-en-v1.5")
    s = load_settings()
    assert s.llm.base_url == "http://llm.internal:8000/v1"
    assert s.embed.model_path == "/models/bge-small-en-v1.5"


def test_sources_load():
    sources = load_sources()
    names = {s.name for s in sources}
    assert {"hf_models", "hf_papers", "hn", "github_releases"} <= names
    gh = next(s for s in sources if s.name == "github_releases")
    assert "vllm-project/vllm" in gh.extra["repos"]
