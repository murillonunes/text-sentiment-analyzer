import numpy as np
import pandas as pd

from sentiment_analyzer.statistical import (
    benjamini_hochberg,
    build_statistical_tables,
    wilson_interval,
)


def test_benjamini_hochberg_preserves_order_and_monotonicity():
    adjusted = benjamini_hochberg([0.04, 0.001, 0.03])
    assert np.allclose(adjusted, [0.04, 0.003, 0.04])


def test_wilson_interval_contains_observed_proportion():
    lower, upper = wilson_interval(50, 100)
    assert lower < 0.5 < upper


def test_statistical_tables_separate_complete_and_partial_periods():
    rows = []
    recommendation_id = 0
    for country, language in (("BR", "brazilian"), ("US", "english")):
        for year in (2020, 2021, 2022):
            for index in range(20):
                recommendation_id += 1
                joy = index < (5 + (year - 2020) * 3)
                rows.append(
                    {
                        "recommendation_id": recommendation_id,
                        "country_code": country,
                        "language": language,
                        "date_created": f"{year}-01-01",
                        "analysis_status": "analyzed",
                        "emotion": "joy" if joy else "neutral",
                        "emotion_joy_score": 0.8 if joy else 0.2,
                        "emotion_neutral_score": 0.2 if joy else 0.8,
                    }
                )
    result = build_statistical_tables(
        pd.DataFrame(rows),
        threshold=0.5,
        complete_through=2021,
        partial_year=2022,
        dominant_languages={"BR": "brazilian", "US": "english"},
    )

    assert result["summary"]["population_counts"] == {
        "all_languages": 120,
        "dominant_languages": 120,
    }
    assert set(result["trends"]["period"]) == {
        "complete_years",
        "including_partial_year",
    }
    assert len(result["annual"]) == 48
    assert result["annual"]["is_partial_year"].sum() == 16
    assert result["distribution_tests"]["p_value"].between(0, 1).all()
    assert result["trends"]["model_converged"].all()
    assert result["interactions"]["reduced_converged"].all()
    assert result["interactions"]["full_converged"].all()
