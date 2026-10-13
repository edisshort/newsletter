import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from digest import fetch as fetch_pkg
from digest.config import Source, load_settings, load_sources
from digest.fetch import arxiv, common, dedup, hf, rss
from digest.models import RawItem

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.jsonl"
NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)
KEEP = ["news.ycombinator.com"]


@pytest.fixture
def cfg():
    s = load_settings()
    return s.model_copy(update={"fetch": s.fetch.model_copy(update={"backoff_s": 0.0})})


def _item(url: str, text: str = "", source: str = "hn") -> RawItem:
    return RawItem(
        id=common.item_id(url, KEEP), title="t", url=url, source=source,
        type="news", published_at=NOW, text=text, fetched_at=NOW,
    )


def _mock_client(handler):
    return lambda cfg: httpx.Client(transport=httpx.MockTransport(handler))


# --- URL canonicalisation / ids ---------------------------------------------

@pytest.mark.parametrize("variant", [
    "https://example.com/post",
    "https://example.com/post/",
    "http://www.example.com/post?utm_source=x",
    "https://EXAMPLE.com/post#section",
])
def test_canonical_url_variants_share_id(variant):
    assert common.item_id(variant, KEEP) == common.item_id("https://example.com/post", KEEP)


def test_hn_item_query_is_kept():
    a = common.item_id("https://news.ycombinator.com/item?id=1", KEEP)
    b = common.item_id("https://news.ycombinator.com/item?id=2", KEEP)
    assert a != b


def test_dedup_prefers_longer_text():
    short, long_ = _item("https://x.com/a"), _item("https://x.com/a/", text="body text")
    out = dedup([short, long_])
    assert len(out) == 1 and out[0].text == "body text"


def test_clean_text_strips_html():
    assert common.clean_text("<p>Hello&nbsp;<b>world</b></p>\n\n x") == "Hello world x"


# --- retries ------------------------------------------------------------------

def test_get_retries_5xx_then_succeeds(cfg):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] < 3 else 200, text="ok")

    with _mock_client(handler)(cfg) as client:
        assert common.get(client, cfg, "https://x.test/").text == "ok"
    assert calls["n"] == cfg.fetch.retries + 1


def test_get_does_not_retry_404(cfg):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    with _mock_client(handler)(cfg) as client, pytest.raises(httpx.HTTPStatusError):
        common.get(client, cfg, "https://x.test/")
    assert calls["n"] == 1


# --- parsers ------------------------------------------------------------------

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>New &amp; shiny model</title><link>https://blog.test/new</link>
<pubDate>Wed, 08 Oct 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;We release X-7B.&lt;/p&gt;</description></item>
<item><title>Old post</title><link>https://blog.test/old</link>
<pubDate>Mon, 01 Sep 2026 10:00:00 GMT</pubDate><description>old</description></item>
</channel></rss>"""


def test_rss_parses_and_filters_by_since(cfg, monkeypatch):
    monkeypatch.setattr(rss, "make_client", _mock_client(lambda r: httpx.Response(200, text=RSS)))
    src = Source(name="blog_test", type="rss", url="https://blog.test/feed", item_type="news")
    items = rss.fetch(cfg, src, NOW - timedelta(days=7))
    assert [i.title for i in items] == ["New & shiny model"]
    assert items[0].text == "We release X-7B." and items[0].source == "blog_test"


ARXIV = """<?xml version="1.0"?><rss version="2.0" xmlns:arxiv="http://arxiv.org/schemas/atom"><channel>
<item><title>Paper A</title><link>https://arxiv.org/abs/2610.00001</link>
<description>arXiv:2610.00001v1 Announce Type: new
Abstract: We study things.</description><pubDate>Thu, 09 Oct 2026 00:00:00 -0400</pubDate>
<arxiv:announce_type>new</arxiv:announce_type></item>
<item><title>Paper B</title><link>https://arxiv.org/abs/2501.00002</link>
<description>arXiv:2501.00002v3 Announce Type: replace
Abstract: Old.</description><pubDate>Thu, 09 Oct 2026 00:00:00 -0400</pubDate>
<arxiv:announce_type>replace</arxiv:announce_type></item>
</channel></rss>"""


def test_arxiv_strips_prefix_and_skips_revisions(cfg, monkeypatch):
    monkeypatch.setattr(arxiv, "make_client", _mock_client(lambda r: httpx.Response(200, text=ARXIV)))
    src = Source(name="arxiv_cs.CL", type="arxiv", url="https://rss.test", item_type="paper",
                 extra={"announce_types": ["new", "cross"]})
    items = arxiv.fetch(cfg, src, NOW - timedelta(days=7))
    assert [i.title for i in items] == ["Paper A"]
    assert items[0].text == "We study things."


def test_hf_models_falls_back_on_400(cfg, monkeypatch):
    model = {"id": "org/Model-7B", "likes": 5, "downloads": 10, "trendingScore": 3,
             "createdAt": "2026-10-08T00:00:00.000Z", "pipeline_tag": "text-generation",
             "cardData": {"license": "apache-2.0"}, "tags": ["transformers", "region:us"]}

    def handler(request):
        if "trendingScore" in str(request.url):
            return httpx.Response(400)
        return httpx.Response(200, json=[model])

    monkeypatch.setattr(hf, "make_client", _mock_client(handler))
    src = Source(name="hf_models", type="hf_models", item_type="model",
                 url="https://hf.test/api/models?sort=trendingScore",
                 fallback_url="https://hf.test/api/models?sort=likes")
    [item] = hf.fetch(cfg, src, NOW - timedelta(days=7))
    assert str(item.url) == "https://huggingface.co/org/Model-7B"
    assert "License: apache-2.0" in item.text and "region:us" not in item.text
    assert item.metrics == {"likes": 5.0, "downloads": 10.0, "trending": 3.0}


# --- orchestration ------------------------------------------------------------

def test_one_failing_source_does_not_stop_run(cfg, monkeypatch):
    good = _item("https://ok.test/a")

    def boom(cfg, source, since):
        raise httpx.ConnectError("dns failure")

    monkeypatch.setitem(fetch_pkg.FETCHERS, "bad", boom)
    monkeypatch.setitem(fetch_pkg.FETCHERS, "good", lambda cfg, source, since: [good])
    sources = [
        Source(name="s_bad", type="bad", url="https://bad.invalid", item_type="news"),
        Source(name="s_good", type="good", url="https://ok.test", item_type="news"),
    ]
    items, reports = fetch_pkg.run(cfg, sources, NOW, do_extract=False)
    assert items == [good]
    assert reports[0].error and "ConnectError" in reports[0].error
    assert reports[1].error is None and reports[1].count == 1


def test_every_configured_source_type_has_a_fetcher():
    assert {s.type for s in load_sources()} <= set(fetch_pkg.FETCHERS)


def test_fetch_package_does_not_import_processing_stack():
    code = (
        "import sys, digest.fetch;"
        "bad = [m for m in ('sklearn', 'sentence_transformers', 'torch', 'openai', 'jinja2',"
        " 'digest.process', 'digest.generate', 'digest.render') if m in sys.modules];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1] / "src")


# --- fixture ------------------------------------------------------------------

def test_fixture_is_valid_and_diverse():
    items = [RawItem.model_validate_json(line) for line in FIXTURE.read_text().splitlines()]
    assert 50 <= len(items) <= 80
    assert len({i.id for i in items}) == len(items)
    assert len({i.source for i in items}) >= 8
    assert {i.type for i in items} >= {"model", "paper", "release", "news", "discussion"}
    # Deliberate near-duplicate group: same launch from several sources.
    gemma = {i.source for i in items if "embeddinggemma" in i.title.lower().replace(" ", "")}
    assert len(gemma) >= 3


# --- live (skipped by default; run with `pytest -m network`) -------------------

@pytest.mark.network
def test_live_hn_fetch():
    cfg = load_settings()
    src = next(s for s in load_sources() if s.name == "hn")
    items = fetch_pkg.FETCHERS[src.type](cfg, src, datetime.now(timezone.utc) - timedelta(days=7))
    assert items and all(i.source == "hn" for i in items)
