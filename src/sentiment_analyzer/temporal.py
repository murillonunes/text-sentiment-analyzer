from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from sentiment_analyzer.analyzer import SentimentAnalyzer


def _score_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        column[len("emotion_") : -len("_score")]: column
        for column in df.columns
        if column.startswith("emotion_")
        and column.endswith("_score")
        and column != "emotion_score"
    }


def _rank_rows(rows: list[Dict[str, Any]]) -> None:
    rows.sort(
        key=lambda row: (
            row["country_code"],
            row["analysis_year"],
            -row["emotion_count"],
            -row["mean_score"],
            row["emotion"],
        )
    )
    previous_group = None
    rank = 0
    for row in rows:
        group = (row["country_code"], row["analysis_year"])
        if group != previous_group:
            previous_group = group
            rank = 1
        else:
            rank += 1
        row["rank"] = rank


def build_temporal_tables(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    partial_year: int | None = None,
) -> Dict[str, Any]:
    """Builds country/year tables without changing the analyzed input."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    required = {"country_code", "analysis_status", "emotion"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    working = SentimentAnalyzer.add_analysis_year(df)
    if "analysis_year" not in working.columns:
        raise ValueError("input must contain a supported review date column")

    score_columns = _score_columns(working)
    if not score_columns:
        raise ValueError("input does not contain per-emotion score columns")

    dated = working[
        working["country_code"].notna() & working["analysis_year"].notna()
    ].copy()
    dated["analysis_year"] = dated["analysis_year"].astype(int)
    valid = dated[dated["analysis_status"] == "analyzed"].copy()

    coverage_rows = []
    for (country, year), group in dated.groupby(
        ["country_code", "analysis_year"],
        sort=True,
    ):
        statuses = group["analysis_status"].value_counts()
        analyzed = int(statuses.get("analyzed", 0))
        empty = int(statuses.get("empty_text", 0))
        coverage_rows.append(
            {
                "country_code": country,
                "analysis_year": int(year),
                "total_reviews": len(group),
                "analyzed_reviews": analyzed,
                "empty_text_reviews": empty,
                "other_status_reviews": int(len(group) - analyzed - empty),
                "analysis_coverage_percent": round(
                    analyzed / len(group) * 100,
                    2,
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage["is_partial_year"] = coverage["analysis_year"].eq(partial_year)

    primary_rows: list[Dict[str, Any]] = []
    multilabel_rows: list[Dict[str, Any]] = []
    for (country, year), group in valid.groupby(
        ["country_code", "analysis_year"],
        sort=True,
    ):
        denominator = len(group)
        for emotion, column in sorted(score_columns.items()):
            scores = pd.to_numeric(group[column], errors="coerce")
            mean_score = (
                round(float(scores.mean()), 6)
                if scores.notna().any()
                else 0.0
            )
            common = {
                "country_code": country,
                "analysis_year": int(year),
                "emotion": emotion,
                "analyzed_reviews": denominator,
                "mean_score": mean_score,
            }
            primary_count = int(group["emotion"].eq(emotion).sum())
            primary_rows.append(
                {
                    **common,
                    "emotion_count": primary_count,
                    "emotion_percent": round(
                        primary_count / denominator * 100,
                        2,
                    ),
                    "threshold": None,
                }
            )
            multilabel_count = int(scores.ge(threshold).sum())
            multilabel_rows.append(
                {
                    **common,
                    "emotion_count": multilabel_count,
                    "emotion_percent": round(
                        multilabel_count / denominator * 100,
                        2,
                    ),
                    "threshold": threshold,
                }
            )

    _rank_rows(primary_rows)
    _rank_rows(multilabel_rows)
    primary = pd.DataFrame(primary_rows)
    multilabel = pd.DataFrame(multilabel_rows)
    primary["is_partial_year"] = primary["analysis_year"].eq(partial_year)
    multilabel["is_partial_year"] = multilabel["analysis_year"].eq(
        partial_year
    )

    top3 = pd.concat(
        [
            primary[primary["rank"] <= 3].assign(mode="primary"),
            multilabel[multilabel["rank"] <= 3].assign(mode="multilabel"),
        ],
        ignore_index=True,
    )
    top3 = top3[
        [
            "mode",
            "country_code",
            "analysis_year",
            "rank",
            "emotion",
            "analyzed_reviews",
            "emotion_count",
            "emotion_percent",
            "mean_score",
            "threshold",
            "is_partial_year",
        ]
    ].sort_values(
        ["mode", "country_code", "analysis_year", "rank"],
        ignore_index=True,
    )

    yoy_frames = []
    for mode, table in (("primary", primary), ("multilabel", multilabel)):
        yoy = table.sort_values(
            ["country_code", "emotion", "analysis_year"]
        ).copy()
        groups = yoy.groupby(["country_code", "emotion"], sort=False)
        yoy["previous_year"] = groups["analysis_year"].shift(1).astype(
            "Int64"
        )
        yoy["previous_count"] = groups["emotion_count"].shift(1).astype(
            "Int64"
        )
        yoy["previous_percent"] = groups["emotion_percent"].shift(1)
        yoy["count_change"] = (
            yoy["emotion_count"] - yoy["previous_count"]
        ).astype("Int64")
        yoy["percent_change_pp"] = (
            yoy["emotion_percent"] - yoy["previous_percent"]
        ).round(2)
        yoy["mean_score_change"] = (
            yoy["mean_score"] - groups["mean_score"].shift(1)
        ).round(6)
        yoy.insert(0, "mode", mode)
        yoy_frames.append(yoy)
    yoy = pd.concat(yoy_frames, ignore_index=True)

    years = sorted(valid["analysis_year"].unique().tolist())
    countries = sorted(valid["country_code"].astype(str).unique().tolist())
    summary = {
        "input_reviews": int(len(working)),
        "dated_reviews": int(len(dated)),
        "analyzed_reviews": int(len(valid)),
        "empty_text_reviews": int(
            (working["analysis_status"] == "empty_text").sum()
        ),
        "other_status_reviews": int(
            (~working["analysis_status"].isin(["analyzed", "empty_text"])).sum()
        ),
        "countries": countries,
        "years": years,
        "partial_year": partial_year,
        "latest_year_is_partial": (
            partial_year == max(years) if years else False
        ),
        "latest_year": max(years) if years else None,
        "emotion_threshold": threshold,
        "emotions": sorted(score_columns),
        "primary_percentages_are_exclusive": True,
        "multilabel_percentages_may_sum_above_100": True,
    }
    return {
        "coverage": coverage,
        "primary": primary,
        "multilabel": multilabel,
        "top3": top3,
        "yoy": yoy,
        "summary": summary,
    }
