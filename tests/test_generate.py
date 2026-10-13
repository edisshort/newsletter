import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import openai
import pytest

from digest.config import load_settings
from digest.generate import prompts, report_dict, run, validate
from digest.generate.llm import LLM
from digest.models import Cluster, Digest, RawItem

NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)


def _item(n: int, title: str, text: str, source: str = "blog_x", type_: str = "news", **metrics) -> RawItem:
    return RawItem(id=f"id{n}", title=title, url=f"https://src.test/{n}", source=source, type=type_,
                   published_at=NOW, text=text, metrics=metrics, fetched_at=NOW)


def _cluster(cid: int, score: float = 0.5, items=None, anchor="release of a new open-weight large language model"):
    items = items or [
        _item(cid * 10, f"Acme-7B {cid} released", "Acme released Acme-7B, a 7B model under apache-2.0. It scores 81.5 on MMLU."),
        _item(cid * 10 + 1, f"Acme-7B {cid} on HN", "", source="hn", type_="discussion", points=120),
    ]
    return Cluster(cluster_id=cid, items=items, headline=items[0].title, sources=sorted({i.source for i in items}),
                   relevance=0.5, coverage=0.5, engagement=0.5, recency=0.5, authority=0.5, score=score,
                   best_anchor=anchor)


def _answer(**overrides) -> str:
    base = {
        "title": "Acme-7B", "category": "LLM",
        "what_it_is": "Acme released Acme-7B, a 7B open model.",
        "why_it_matters": "It scores 81.5 on MMLU.",
        "open_source": True, "license": "apache-2.0",
        "sources": ["https://src.test/10", "https://evil.test/made-up"],
    }
    base.update(overrides)
    return json.dumps(base)


class FakeCompletions:
    """Returns queued answers; optionally rejects json_schema like an older server."""

    def __init__(self, answers, reject_schema=False):
        self.answers = list(answers)
        self.reject_schema = reject_schema
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_schema and kwargs["response_format"]["type"] == "json_schema":
            req = httpx.Request("POST", "http://llm.test")
            raise openai.BadRequestError("json_schema unsupported", response=httpx.Response(400, request=req), body=None)
        content = self.answers.pop(0) if self.answers else _answer()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _llm(cfg, answers, reject_schema=False):
    fake = FakeCompletions(answers, reject_schema)
    return LLM(cfg, client=SimpleNamespace(chat=SimpleNamespace(completions=fake))), fake


@pytest.fixture
def cfg():
    return load_settings()


# --- validate -------------------------------------------------------------------

def test_grounding_flags_unknown_names_and_numbers():
    src = "Acme released Acme-7B, a 7B model. It scores 81.5 on MMLU with 1,200 likes."
    assert validate.grounding_misses("Acme-7B scores 81.5 with 1200 likes", src) == []
    assert validate.grounding_misses("Acme-8B beats GPT-5 by 12%", src) == ["12", "Acme-8B", "GPT-5"]


def test_check_drops_foreign_sources_and_keeps_cluster_urls(cfg):
    res = validate.check(_answer(), _cluster(1), cfg)
    assert res.errors == []
    assert res.initiative.sources == ["https://src.test/10"]
    assert res.dropped_sources == ["https://evil.test/made-up"]


def test_check_falls_back_to_top_item_url_when_no_source_survives(cfg):
    res = validate.check(_answer(sources=["https://evil.test/x"]), _cluster(1), cfg)
    assert res.initiative.sources == ["https://src.test/10"]


def test_check_reports_word_limit_and_bad_json(cfg):
    long = " ".join(["word"] * 41)
    assert any("what_it_is has 41 words" in e for e in validate.check(_answer(what_it_is=long), _cluster(1), cfg).errors)
    assert validate.check("not json", _cluster(1), cfg).errors
    assert validate.check(_answer(category="Crypto"), _cluster(1), cfg).errors


def test_extractive_is_grounded_and_within_limits(cfg):
    c = _cluster(1)
    draft = validate.extractive(c, cfg)
    assert draft.what_it_is.startswith("Acme released Acme-7B")
    assert validate.word_count(draft.what_it_is) <= cfg.generate.what_it_is_max_words
    assert validate.word_count(draft.why_it_matters) <= cfg.generate.why_it_matters_max_words
    assert set(draft.sources) <= {str(i.url) for i in c.items}
    assert draft.category == "LLM"


def test_extractive_paper_cluster_links_only_top_item(cfg):
    papers = [_item(1, "Paper A", "We study A.", "hf_papers", "paper", upvotes=5),
              _item(2, "Paper B", "We study B.", "arxiv_cs.CL", "paper")]
    draft = validate.extractive(_cluster(1, items=papers), cfg)
    assert draft.sources == ["https://src.test/1"] and draft.category == "Research"


# --- orchestration ----------------------------------------------------------------

def test_dry_run_produces_top_n_extractive(cfg):
    clusters = [_cluster(i, score=1 - i / 100) for i in range(1, 13)]
    digest, reports = run(cfg, clusters, llm=None, window_days=7)
    assert len(digest.top10) == cfg.generate.top_n
    assert [i.rank for i in digest.top10] == list(range(1, 11))
    assert all(i.summary_mode == "extractive" for i in digest.top10)
    assert digest.top10[0].trend_score == clusters[0].score
    Digest.model_validate_json(digest.model_dump_json())


def test_llm_success_retry_and_fallback(cfg):
    clusters = [_cluster(1, 0.9), _cluster(2, 0.8), _cluster(3, 0.7)]
    bad = _answer(what_it_is="Acme-9B launched.")  # ungrounded name
    llm, fake = _llm(cfg, [_answer(), bad, _answer(), bad, bad, bad])
    digest, reports = run(cfg, clusters, llm, window_days=7)
    modes = [(r.summary_mode, r.attempts) for r in reports[:3]]
    assert modes == [("llm", 1), ("llm", 2), ("extractive", 3)]
    # retry message carried the grounding error back to the model
    assert "Acme-9B" in fake.calls[2]["messages"][-1]["content"]
    rep = report_dict(reports)
    assert rep["unresolved_llm_grounding_misses"] == 0 and rep["extractive"] == 1


def test_json_schema_rejected_falls_back_to_json_object(cfg):
    llm, fake = _llm(cfg, [_answer()], reject_schema=True)
    digest, reports = run(cfg, [_cluster(1)], llm, window_days=7)
    assert reports[0].summary_mode == "llm"
    assert [c["response_format"]["type"] for c in fake.calls[:2]] == ["json_schema", "json_object"]
    assert "JSON schema" in fake.calls[1]["messages"][0]["content"]


def test_endpoint_down_falls_back_without_retries(cfg):
    class Down:
        def create(self, **kwargs):
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://llm.test"))

    llm = LLM(cfg, client=SimpleNamespace(chat=SimpleNamespace(completions=Down())))
    digest, reports = run(cfg, [_cluster(1)], llm, window_days=7)
    assert reports[0].summary_mode == "extractive" and reports[0].attempts == 1 and reports[0].llm_error


def test_llm_requires_base_url(cfg):
    cfg.llm.base_url = ""
    with pytest.raises(ValueError):
        LLM(cfg)


def test_engagement_numbers_are_not_grounding(cfg):
    # metrics are not shown to the LLM, so quoting them (e.g. "63 points on GitHub") must fail
    res = validate.check(_answer(why_it_matters="Discussed on HN with 120 points."), _cluster(1), cfg)
    assert res.grounding_misses == ["120"]
    assert "engagement" not in prompts.user_prompt(_cluster(1), cfg)


def test_what_why_repetition_rejected(cfg):
    res = validate.check(_answer(why_it_matters="Acme released Acme-7B, an open 7B model."), _cluster(1), cfg)
    assert any("repeats" in e for e in res.errors)


def test_unsupported_open_source_and_license_removed(cfg):
    items = [_item(1, "Acme-7B released", "Acme released Acme-7B, a 7B model. It scores 81.5 on MMLU.")]
    res = validate.check(_answer(license="mit", sources=["https://src.test/1"]), _cluster(1, items=items), cfg)
    assert res.initiative.license is None and res.initiative.open_source is None
    assert len(res.corrections) == 2 and res.errors == []
    # supported by the default cluster text ("under apache-2.0")
    res = validate.check(_answer(), _cluster(1), cfg)
    assert res.initiative.open_source is True and res.initiative.license == "apache-2.0"


def test_paper_cluster_uses_top_paper_only(cfg):
    papers = [_item(i, f"Paper {i}", f"We present Paper {i}, reaching 81.5 on MMLU.", source="hf_papers", type_="paper")
              for i in range(1, 4)]
    answer = _answer(title="Paper 1", category="LLM", what_it_is="We present Paper 1.", why_it_matters="It reaches 81.5 on MMLU.",
                     open_source=None, license=None, sources=["https://src.test/1", "https://src.test/2"])
    res = validate.check(answer, _cluster(1, items=papers), cfg)
    assert res.initiative.category == "Research"
    assert [str(u) for u in res.initiative.sources] == ["https://src.test/1"]
    assert "Paper 2" not in prompts.user_prompt(_cluster(1, items=papers), cfg)


def test_hf_tags_not_shown_to_llm(cfg):
    model = _item(1, "acme/Acme-7B", "acme/Acme-7B is a Hugging Face model. License: mit. Tags: foo, bar, baz.",
                  source="hf_models", type_="model")
    assert "Tags:" not in prompts.user_prompt(_cluster(1, items=[model]), cfg)


def test_nullish_license_and_source_cap(cfg):
    items = [_item(i, f"Acme-7B note {i}", "Acme-7B is out.") for i in range(5)]
    urls = [f"https://src.test/{i}" for i in range(5)]
    res = validate.check(_answer(license="null", sources=urls, why_it_matters="Acme-7B is out."), _cluster(1, items=items), cfg)
    assert res.initiative.license is None
    assert res.initiative.sources == urls[: cfg.generate.max_sources]


def test_qualifier_mismatch_is_a_grounding_miss(cfg):
    src = "Our customers spend an average of $0.37/month per sandbox. Up to 90% faster."
    assert validate.qualifier_misses("Sandboxes cost under $0.37/month.", src, cfg)
    assert validate.qualifier_misses("Sandboxes average $0.37/month, up to 90% faster.", src, cfg) == []


# --- weekly overview ------------------------------------------------------------

def _initiatives(cfg, n=3):
    digest, _ = run(cfg, [_cluster(i, score=1 - i / 100) for i in range(1, n + 1)], llm=None, window_days=7)
    return digest.top10


def test_overview_grounding_rejects_new_numbers(cfg):
    items = _initiatives(cfg)
    ok = json.dumps({"headline": "Acme-7B released", "overview": "Acme released Acme-7B, a 7B model."})
    bad = json.dumps({"headline": "Acme-7B released", "overview": "Acme-7B scores 95 on MMLU."})
    assert validate.check_overview(ok, items, cfg)[1] == []
    assert any("95" in e for e in validate.check_overview(bad, items, cfg)[1])


def test_overview_falls_back_and_quick_hits_list_unused_candidates(cfg):
    cfg.generate.top_n = 2
    cfg.generate.quick_hits = 6  # off in settings since 2026-10-12; the feature stays tested
    clusters = [_cluster(i, score=1 - i / 100) for i in range(1, 6)]
    digest, reports = run(cfg, clusters, llm=None, window_days=7)
    assert digest.overview_mode == "extractive" and digest.headline == digest.top10[0].title
    assert digest.overview.startswith("Acme released Acme-7B")
    assert [h.title for h in digest.quick_hits] == [c.headline for c in clusters[2:]]
    assert report_dict(reports)["clusters"] == 2


def test_overview_llm_path(cfg):
    cfg.generate.write_overview = True
    overview = json.dumps({"headline": "Acme-7B released", "overview": "Acme released Acme-7B, a 7B model."})
    llm, fake = _llm(cfg, [_answer(), overview])
    digest, reports = run(cfg, [_cluster(1)], llm, window_days=7)
    assert digest.overview_mode == "llm" and digest.headline == "Acme-7B released"
    assert fake.calls[-1]["response_format"]["json_schema"]["name"] == "llmoverview"


def test_overview_disabled_makes_no_llm_call(cfg):
    cfg.generate.write_overview = False
    llm, fake = _llm(cfg, [_answer()])
    digest, _ = run(cfg, [_cluster(1)], llm, window_days=7)
    assert len(fake.calls) == 1 and digest.overview_mode == "extractive"
