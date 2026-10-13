from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from digest.models import Digest, Initiative, RawItem

NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)


def test_raw_item_defaults():
    item = RawItem(
        id="abc", title="t", url="https://example.com/x", source="hn",
        type="discussion", published_at=NOW, fetched_at=NOW,
    )
    assert item.text == "" and item.metrics == {}


def test_raw_item_rejects_bad_type():
    with pytest.raises(ValidationError):
        RawItem(
            id="abc", title="t", url="https://example.com/x", source="hn",
            type="tweet", published_at=NOW, fetched_at=NOW,
        )


def test_digest_roundtrip():
    init = Initiative(
        rank=1, title="X", category="LLM", what_it_is="a", why_it_matters="b",
        open_source=True, license="apache-2.0", sources=["https://example.com/x"],
        trend_score=0.9, published=date(2026, 10, 8), summary_mode="extractive",
    )
    d = Digest(generated_at=NOW, window_days=7, top10=[init])
    assert Digest.model_validate_json(d.model_dump_json()) == d
