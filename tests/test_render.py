import json
from collections import Counter
from datetime import date, datetime, timezone

from digest.config import load_settings
from digest.models import Digest, Initiative
from digest.render.render import link_label, oss_label, render, source_counts

NOW = datetime(2026, 10, 9, 6, 30, tzinfo=timezone.utc)


def _initiative(rank: int, **overrides) -> Initiative:
    base = dict(
        rank=rank, title=f"Acme-{rank}B", category="LLM",
        what_it_is="Acme released a 7B open model.", why_it_matters="It scores 81.5 on MMLU.",
        open_source=True, license="apache-2.0", sources=[f"https://www.src.test/{rank}", "https://hf.test/x"],
        trend_score=1 - rank / 20, published=date(2026, 10, 8), summary_mode="llm",
    )
    base.update(overrides)
    return Initiative(**base)


def _digest(n: int = 10, **overrides) -> Digest:
    return Digest(generated_at=NOW, window_days=7, top10=[_initiative(i, **overrides) for i in range(1, n + 1)])


def test_render_has_all_items_header_and_footer():
    cfg = load_settings()
    html = render(_digest(), cfg, Counter({"hn": 3, "arxiv_cs.CL": 5}))
    for i in range(1, 11):
        assert f"Acme-{i}B" in html
        assert f'href="https://www.src.test/{i}"' in html
    assert "verify before forwarding" not in html  # disclaimer removed at review (2026-10-12)
    assert "02 Oct 2026" in html and "09 Oct 2026" in html  # window
    assert "8 items from 2 sources" in html
    assert html.index("arxiv_cs.CL (5)") < html.index("hn (3)")  # sorted by count
    assert "Open source · apache-2.0" in html
    assert cfg.render.title in html and "Vol." not in html and "Issue" not in html
    assert cfg.render.category_colors["LLM"] in html
    assert html.count("<b>What:</b>") == 10 and html.count("<b>Why it matters:</b>") == 10
    assert html.index("Acme-1B") < html.index("Acme-2B") < html.index("Acme-10B")  # rank order
    assert "trend score <span class=\"score\">0.95</span>" in html


def test_render_is_self_contained(tmp_path):
    cfg = load_settings()
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"\x89PNG fake")
    cfg.render.logo_path = logo
    html = render(_digest(), cfg)
    assert "<script" not in html and "<link" not in html and 'src="http' not in html
    assert 'src="data:image/png;base64,' in html
    assert "Sources scanned" not in html  # no counts given


def test_missing_logo_falls_back_to_text():
    cfg = load_settings()
    cfg.render.logo_path = cfg.render.logo_path.with_name("does-not-exist.png")
    assert 'class="logo-text">NSE<' in render(_digest(), cfg)


def test_render_escapes_text_and_flags_extractive():
    html = render(_digest(1, title="<b>x</b> & y", summary_mode="extractive"), load_settings())
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in html
    assert ">extract<" in html


def test_oss_label_and_link_label():
    assert oss_label(_initiative(1, open_source=None, license=None)) is None
    assert oss_label(_initiative(1, open_source=False, license=None)) == "Closed"
    assert oss_label(_initiative(1, open_source=None, license="mit")) == "License: mit"
    assert link_label("https://www.example.com/a?b=1") == "example.com"


def test_source_counts(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps({"source": s}) for s in ["hn", "hn", "hf_models"]) + "\n\n", encoding="utf-8")
    assert source_counts(p) == Counter({"hn": 2, "hf_models": 1})


def test_disclaimer_shown_only_when_configured():
    cfg = load_settings()
    cfg.render.disclaimer = "Check before sending."
    assert render(_digest(1), cfg).count("Check before sending.") == 2
