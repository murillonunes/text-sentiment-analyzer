import pandas as pd
import pytest

from sentiment_analyzer.temporal import build_temporal_tables


def _frame():
    return pd.DataFrame(
        {
            "recommendation_id": [1, 2, 3, 4, 5],
            "country_code": ["BR", "BR", "BR", "BR", "US"],
            "date_created": [
                "2023-01-01",
                "2023-06-01",
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
            ],
            "analysis_status": [
                "analyzed",
                "analyzed",
                "analyzed",
                "empty_text",
                "analyzed",
            ],
            "emotion": ["joy", "neutral", "joy", None, "neutral"],
            "emotion_joy_score": [0.8, 0.6, 0.7, None, 0.2],
            "emotion_neutral_score": [0.4, 0.7, 0.6, None, 0.8],
        }
    )


def test_temporal_tables_use_analyzed_reviews_as_denominator():
    result = build_temporal_tables(_frame(), threshold=0.5)
    primary = result["primary"]
    multilabel = result["multilabel"]

    br_2023_joy = primary[
        (primary.country_code == "BR")
        & (primary.analysis_year == 2023)
        & (primary.emotion == "joy")
    ].iloc[0]
    br_2023_neutral = multilabel[
        (multilabel.country_code == "BR")
        & (multilabel.analysis_year == 2023)
        & (multilabel.emotion == "neutral")
    ].iloc[0]

    assert br_2023_joy.analyzed_reviews == 2
    assert br_2023_joy.emotion_count == 1
    assert br_2023_joy.emotion_percent == 50.0
    assert br_2023_neutral.emotion_count == 1
    assert br_2023_neutral.emotion_percent == 50.0
    assert br_2023_neutral.mean_score == pytest.approx(0.55)


def test_temporal_coverage_keeps_empty_text_separate():
    result = build_temporal_tables(_frame(), partial_year=2024)
    coverage = result["coverage"]
    br_2024 = coverage[
        (coverage.country_code == "BR")
        & (coverage.analysis_year == 2024)
    ].iloc[0]

    assert br_2024.total_reviews == 2
    assert br_2024.analyzed_reviews == 1
    assert br_2024.empty_text_reviews == 1
    assert br_2024.analysis_coverage_percent == 50.0
    assert bool(br_2024.is_partial_year)
    assert result["summary"]["partial_year"] == 2024
    assert result["summary"]["latest_year_is_partial"] is True


def test_temporal_top3_and_year_over_year_changes_are_reproducible():
    result = build_temporal_tables(_frame())
    top3 = result["top3"]
    yoy = result["yoy"]

    assert set(top3["mode"]) == {"primary", "multilabel"}
    br_2023_primary = top3[
        (top3["mode"] == "primary")
        & (top3.country_code == "BR")
        & (top3.analysis_year == 2023)
    ]
    assert br_2023_primary.sort_values("rank").emotion.tolist() == [
        "joy",
        "neutral",
    ]

    br_joy_2024 = yoy[
        (yoy["mode"] == "primary")
        & (yoy.country_code == "BR")
        & (yoy.emotion == "joy")
        & (yoy.analysis_year == 2024)
    ].iloc[0]
    assert br_joy_2024.previous_year == 2023
    assert br_joy_2024.previous_percent == 50.0
    assert br_joy_2024.percent_change_pp == 50.0
