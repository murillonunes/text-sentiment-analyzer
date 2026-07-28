from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


REQUIRED_COVERAGE_COLUMNS = {
    "country_code",
    "analysis_year",
    "total_reviews",
    "analyzed_reviews",
    "empty_text_reviews",
    "other_status_reviews",
    "analysis_coverage_percent",
    "is_partial_year",
}
REQUIRED_EMOTION_COLUMNS = {
    "country_code",
    "analysis_year",
    "emotion",
    "analyzed_reviews",
    "mean_score",
    "emotion_count",
    "emotion_percent",
    "rank",
    "is_partial_year",
}


def _native_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    return [
        {key: _native_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def _aggregate_emotions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, emotion), group in frame.groupby(
        ["country_code", "emotion"],
        sort=True,
    ):
        total = int(group["analyzed_reviews"].sum())
        count = int(group["emotion_count"].sum())
        weighted_score = (
            group["mean_score"] * group["analyzed_reviews"]
        ).sum()
        rows.append(
            {
                "country_code": country,
                "analysis_year": None,
                "emotion": emotion,
                "analyzed_reviews": total,
                "mean_score": (
                    round(float(weighted_score / total), 6)
                    if total
                    else None
                ),
                "emotion_count": count,
                "emotion_percent": (
                    round(count / total * 100, 4) if total else 0.0
                ),
                "threshold": _native_value(group["threshold"].iloc[0])
                if "threshold" in group
                else None,
                "is_partial_year": False,
            }
        )
    aggregate = pd.DataFrame(rows).sort_values(
        ["country_code", "emotion_count", "mean_score", "emotion"],
        ascending=[True, False, False, True],
        ignore_index=True,
    )
    aggregate["rank"] = aggregate.groupby("country_code").cumcount() + 1
    return aggregate


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dashboard_payload(
    temporal_dir: str | Path,
    *,
    statistical_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Loads and validates aggregated research artifacts for the dashboard."""
    temporal_path = Path(temporal_dir)
    coverage = pd.read_csv(temporal_path / "country_year_coverage.csv")
    primary = pd.read_csv(
        temporal_path / "country_year_emotion_primary.csv"
    )
    multilabel = pd.read_csv(
        temporal_path / "country_year_emotion_multilabel.csv"
    )
    temporal_summary = _read_json(temporal_path / "summary.json")

    _validate_columns(
        coverage,
        REQUIRED_COVERAGE_COLUMNS,
        source="country_year_coverage.csv",
    )
    _validate_columns(
        primary,
        REQUIRED_EMOTION_COLUMNS,
        source="country_year_emotion_primary.csv",
    )
    _validate_columns(
        multilabel,
        REQUIRED_EMOTION_COLUMNS,
        source="country_year_emotion_multilabel.csv",
    )

    countries = sorted(coverage["country_code"].dropna().unique().tolist())
    years = sorted(
        int(year) for year in coverage["analysis_year"].dropna().unique()
    )
    emotions = sorted(primary["emotion"].dropna().unique().tolist())
    if set(emotions) != set(multilabel["emotion"].dropna().unique()):
        raise ValueError("primary and multilabel emotion sets do not match")

    confidence = pd.DataFrame()
    statistical_summary: Dict[str, Any] = {}
    if statistical_dir is not None:
        statistical_path = Path(statistical_dir)
        confidence = pd.read_csv(
            statistical_path / "annual_emotion_estimates.csv"
        )
        required_confidence = {
            "population",
            "mode",
            "country_code",
            "analysis_year",
            "emotion",
            "ci95_lower_percent",
            "ci95_upper_percent",
        }
        _validate_columns(
            confidence,
            required_confidence,
            source="annual_emotion_estimates.csv",
        )
        confidence = confidence[
            confidence["population"] == "all_languages"
        ].copy()
        statistical_summary = _read_json(
            statistical_path / "summary.json"
        )

    coverage_aggregate = (
        coverage.groupby("country_code", as_index=False)
        .agg(
            total_reviews=("total_reviews", "sum"),
            analyzed_reviews=("analyzed_reviews", "sum"),
            empty_text_reviews=("empty_text_reviews", "sum"),
            other_status_reviews=("other_status_reviews", "sum"),
        )
    )
    coverage_aggregate["analysis_coverage_percent"] = (
        coverage_aggregate["analyzed_reviews"]
        / coverage_aggregate["total_reviews"]
        * 100
    ).round(4)
    coverage_aggregate["analysis_year"] = None
    coverage_aggregate["is_partial_year"] = False

    threshold_values = pd.to_numeric(
        multilabel.get("threshold"),
        errors="coerce",
    ).dropna()
    threshold = (
        float(threshold_values.iloc[0])
        if len(threshold_values)
        else temporal_summary.get("emotion_threshold", 0.5)
    )
    partial_year = temporal_summary.get("partial_year")
    if partial_year is None:
        partial_rows = coverage[coverage["is_partial_year"].astype(bool)]
        partial_year = (
            int(partial_rows["analysis_year"].iloc[0])
            if len(partial_rows)
            else None
        )

    return {
        "metadata": {
            "title": "Dashboard de Emoções — Cyberpunk 2077",
            "countries": countries,
            "years": years,
            "emotions": emotions,
            "partial_year": partial_year,
            "emotion_threshold": threshold,
            "input_reviews": int(coverage["total_reviews"].sum()),
            "analyzed_reviews": int(coverage["analyzed_reviews"].sum()),
            "empty_text_reviews": int(
                coverage["empty_text_reviews"].sum()
            ),
            "generated_at": temporal_summary.get("generated_at"),
            "statistical_analysis_available": not confidence.empty,
            "complete_through": statistical_summary.get(
                "complete_through"
            ),
            "score_note": (
                "Probabilidade atribuída pelo modelo a cada emoção; "
                "não representa diretamente a intensidade emocional interna."
            ),
        },
        "coverage": {
            "annual": _records(coverage),
            "all_years": _records(coverage_aggregate),
        },
        "emotions": {
            "primary": {
                "annual": _records(primary),
                "all_years": _records(_aggregate_emotions(primary)),
            },
            "multilabel": {
                "annual": _records(multilabel),
                "all_years": _records(_aggregate_emotions(multilabel)),
            },
        },
        "confidence_intervals": _records(confidence),
    }


def render_dashboard_html(payload: Dict[str, Any]) -> str:
    """Renders a self-contained HTML document with safely embedded JSON."""
    template = (
        resources.files("sentiment_analyzer")
        .joinpath("templates/dashboard.html")
        .read_text(encoding="utf-8")
    )
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    serialized = (
        serialized.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return template.replace("__DASHBOARD_DATA__", serialized)


def write_dashboard(
    temporal_dir: str | Path,
    output_path: str | Path,
    *,
    statistical_dir: str | Path | None = None,
) -> Path:
    payload = build_dashboard_payload(
        temporal_dir,
        statistical_dir=statistical_dir,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_dashboard_html(payload),
        encoding="utf-8",
    )
    return output
