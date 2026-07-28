import json

import pandas as pd
import pytest

from sentiment_analyzer.dashboard import (
    build_dashboard_payload,
    render_dashboard_html,
    write_dashboard,
)


def _write_temporal_fixture(path, *, malicious_emotion=False):
    path.mkdir()
    emotion = (
        "</script><script>alert(1)</script>"
        if malicious_emotion
        else "joy"
    )
    coverage = pd.DataFrame(
        {
            "country_code": ["BR", "BR", "US", "US"],
            "analysis_year": [2023, 2024, 2023, 2024],
            "total_reviews": [11, 10, 20, 20],
            "analyzed_reviews": [10, 10, 20, 19],
            "empty_text_reviews": [1, 0, 0, 1],
            "other_status_reviews": [0, 0, 0, 0],
            "analysis_coverage_percent": [90.91, 100, 100, 95],
            "is_partial_year": [False, True, False, True],
        }
    )
    rows = []
    for country, year, total in (
        ("BR", 2023, 10),
        ("BR", 2024, 10),
        ("US", 2023, 20),
        ("US", 2024, 19),
    ):
        for rank, label, count, score in (
            (1, emotion, total - 2, 0.7),
            (2, "neutral", 2, 0.3),
        ):
            rows.append(
                {
                    "country_code": country,
                    "analysis_year": year,
                    "emotion": label,
                    "analyzed_reviews": total,
                    "mean_score": score,
                    "emotion_count": count,
                    "emotion_percent": count / total * 100,
                    "threshold": None,
                    "rank": rank,
                    "is_partial_year": year == 2024,
                }
            )
    primary = pd.DataFrame(rows)
    multilabel = primary.copy()
    multilabel["threshold"] = 0.5
    coverage.to_csv(path / "country_year_coverage.csv", index=False)
    primary.to_csv(
        path / "country_year_emotion_primary.csv",
        index=False,
    )
    multilabel.to_csv(
        path / "country_year_emotion_multilabel.csv",
        index=False,
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-28T00:00:00",
                "partial_year": 2024,
                "emotion_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )


def test_dashboard_payload_aggregates_years_without_raw_reviews(tmp_path):
    temporal = tmp_path / "temporal"
    _write_temporal_fixture(temporal)

    payload = build_dashboard_payload(temporal)

    assert payload["metadata"]["input_reviews"] == 61
    assert payload["metadata"]["analyzed_reviews"] == 59
    assert payload["metadata"]["empty_text_reviews"] == 2
    assert payload["metadata"]["partial_year"] == 2024
    br_primary = payload["emotions"]["primary"]["all_years"]
    br_joy = next(
        row
        for row in br_primary
        if row["country_code"] == "BR" and row["emotion"] == "joy"
    )
    assert br_joy["analyzed_reviews"] == 20
    assert br_joy["emotion_count"] == 16
    assert br_joy["emotion_percent"] == 80.0
    assert br_joy["rank"] == 1
    assert "review_text" not in json.dumps(payload)


def test_dashboard_html_is_self_contained_and_escapes_embedded_json(
    tmp_path,
):
    temporal = tmp_path / "temporal"
    _write_temporal_fixture(temporal, malicious_emotion=True)

    html = render_dashboard_html(build_dashboard_payload(temporal))

    assert "__DASHBOARD_DATA__" not in html
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html
    assert "<script src=" not in html
    assert "<link " not in html
    assert "fetch(" not in html
    assert "Dashboard de Emoções" in html


def test_dashboard_writer_creates_portable_html(tmp_path):
    temporal = tmp_path / "temporal"
    output = tmp_path / "dashboard.html"
    _write_temporal_fixture(temporal)

    result = write_dashboard(temporal, output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 10_000


def test_dashboard_rejects_missing_required_columns(tmp_path):
    temporal = tmp_path / "temporal"
    _write_temporal_fixture(temporal)
    coverage = pd.read_csv(temporal / "country_year_coverage.csv")
    coverage.drop(columns=["empty_text_reviews"]).to_csv(
        temporal / "country_year_coverage.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="empty_text_reviews"):
        build_dashboard_payload(temporal)
