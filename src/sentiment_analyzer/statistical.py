from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, chi2_contingency, norm

from sentiment_analyzer.analyzer import SentimentAnalyzer


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    """Returns Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[valid] = restored
    return adjusted


def wilson_interval(
    count: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - (1 - confidence) / 2)
    proportion = count / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def _score_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        column[len("emotion_") : -len("_score")]: column
        for column in df.columns
        if column.startswith("emotion_")
        and column.endswith("_score")
        and column != "emotion_score"
    }


def _fit_binomial(
    counts: np.ndarray,
    totals: np.ndarray,
    design: np.ndarray,
) -> Dict[str, Any]:
    counts = np.asarray(counts, dtype=float)
    totals = np.asarray(totals, dtype=float)
    design = np.asarray(design, dtype=float)

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(
            -np.sum(counts * eta - totals * np.logaddexp(0.0, eta))
        )

    def gradient(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        return design.T @ (totals * probabilities - counts)

    overall = np.clip(counts.sum() / totals.sum(), 1e-6, 1 - 1e-6)
    start = np.zeros(design.shape[1])
    start[0] = np.log(overall / (1 - overall))
    fitted = minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=gradient,
        options={
            "maxiter": 10_000,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )
    beta = fitted.x
    probabilities = expit(design @ beta)
    weights = totals * probabilities * (1 - probabilities)
    information = design.T @ (design * weights[:, None])
    covariance = np.linalg.pinv(information)
    return {
        "beta": beta,
        "covariance": covariance,
        "log_likelihood": -objective(beta),
        "converged": bool(fitted.success),
    }


def _rank_bh(
    frame: pd.DataFrame,
    *,
    p_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["p_value_adjusted"] = np.nan
    for _, indices in result.groupby(group_columns).groups.items():
        result.loc[indices, "p_value_adjusted"] = benjamini_hochberg(
            result.loc[indices, p_column]
        )
    result["significant_fdr_0_05"] = (
        result["p_value_adjusted"] < 0.05
    )
    return result


def _annual_counts(
    frame: pd.DataFrame,
    *,
    threshold: float,
    population: str,
) -> pd.DataFrame:
    score_columns = _score_columns(frame)
    rows = []
    for (country, year), group in frame.groupby(
        ["country_code", "analysis_year"],
        sort=True,
    ):
        total = len(group)
        for emotion, column in sorted(score_columns.items()):
            for mode, count in (
                ("primary", int(group["emotion"].eq(emotion).sum())),
                (
                    "multilabel",
                    int(
                        pd.to_numeric(group[column], errors="coerce")
                        .ge(threshold)
                        .sum()
                    ),
                ),
            ):
                lower, upper = wilson_interval(count, total)
                rows.append(
                    {
                        "population": population,
                        "mode": mode,
                        "country_code": country,
                        "analysis_year": int(year),
                        "emotion": emotion,
                        "review_count": count,
                        "analyzed_reviews": total,
                        "emotion_percent": round(count / total * 100, 4),
                        "ci95_lower_percent": round(lower * 100, 4),
                        "ci95_upper_percent": round(upper * 100, 4),
                        "threshold": (
                            threshold if mode == "multilabel" else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _binary_temporal_tests(
    annual: pd.DataFrame,
    *,
    years: list[int],
) -> pd.DataFrame:
    rows = []
    selected = annual[annual["analysis_year"].isin(years)]
    for keys, group in selected.groupby(
        ["population", "mode", "country_code", "emotion"],
        sort=True,
    ):
        group = group.sort_values("analysis_year")
        table = np.column_stack(
            [
                group["review_count"].to_numpy(),
                (
                    group["analyzed_reviews"] - group["review_count"]
                ).to_numpy(),
            ]
        )
        statistic, p_value, degrees, _ = chi2_contingency(table)
        total = table.sum()
        effect = np.sqrt(statistic / total) if total else np.nan
        rows.append(
            {
                "population": keys[0],
                "mode": keys[1],
                "country_code": keys[2],
                "emotion": keys[3],
                "start_year": min(years),
                "end_year": max(years),
                "chi_square": statistic,
                "degrees_of_freedom": degrees,
                "p_value": p_value,
                "cramers_v": effect,
            }
        )
    result = pd.DataFrame(rows)
    return _rank_bh(
        result,
        p_column="p_value",
        group_columns=["population", "mode"],
    )


def _primary_distribution_tests(
    frame: pd.DataFrame,
    *,
    population: str,
    period: str,
) -> Dict[str, Any]:
    rows = []
    for country, group in frame.groupby("country_code", sort=True):
        table = pd.crosstab(group["analysis_year"], group["emotion"])
        statistic, p_value, degrees, _ = chi2_contingency(table)
        denominator = len(group) * min(
            table.shape[0] - 1,
            table.shape[1] - 1,
        )
        effect = (
            np.sqrt(statistic / denominator)
            if denominator > 0
            else np.nan
        )
        rows.append(
            {
                "population": population,
                "period": period,
                "country_code": country,
                "start_year": int(group["analysis_year"].min()),
                "end_year": int(group["analysis_year"].max()),
                "analyzed_reviews": len(group),
                "chi_square": statistic,
                "degrees_of_freedom": degrees,
                "p_value": p_value,
                "cramers_v": effect,
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_adjusted"] = benjamini_hochberg(result["p_value"])
    result["significant_fdr_0_05"] = result["p_value_adjusted"] < 0.05
    return result


def _logistic_results(
    annual: pd.DataFrame,
    *,
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trend_rows = []
    interaction_rows = []
    for (population, mode, emotion), group in annual.groupby(
        ["population", "mode", "emotion"],
        sort=True,
    ):
        group = group.sort_values(["country_code", "analysis_year"])
        countries = sorted(group["country_code"].unique())
        reference = countries[0]
        centered_year = (
            group["analysis_year"] - group["analysis_year"].min()
        ).to_numpy(dtype=float)
        dummies = [
            group["country_code"].eq(country).to_numpy(dtype=float)
            for country in countries[1:]
        ]
        reduced = np.column_stack(
            [np.ones(len(group)), centered_year, *dummies]
        )
        interactions = [
            centered_year * dummy for dummy in dummies
        ]
        full = np.column_stack([reduced, *interactions])
        counts = group["review_count"].to_numpy(dtype=float)
        totals = group["analyzed_reviews"].to_numpy(dtype=float)
        reduced_fit = _fit_binomial(counts, totals, reduced)
        full_fit = _fit_binomial(counts, totals, full)

        likelihood_ratio = max(
            0.0,
            2
            * (
                full_fit["log_likelihood"]
                - reduced_fit["log_likelihood"]
            ),
        )
        interaction_df = len(interactions)
        interaction_p = chi2.sf(likelihood_ratio, interaction_df)
        interaction_rows.append(
            {
                "population": population,
                "period": period,
                "mode": mode,
                "emotion": emotion,
                "reference_country": reference,
                "likelihood_ratio": likelihood_ratio,
                "degrees_of_freedom": interaction_df,
                "p_value": interaction_p,
                "reduced_converged": reduced_fit["converged"],
                "full_converged": full_fit["converged"],
            }
        )

        beta = full_fit["beta"]
        covariance = full_fit["covariance"]
        for country_index, country in enumerate(countries):
            contrast = np.zeros(len(beta))
            contrast[1] = 1.0
            if country_index > 0:
                interaction_index = len(reduced[0]) + country_index - 1
                contrast[interaction_index] = 1.0
            slope = float(contrast @ beta)
            variance = float(contrast @ covariance @ contrast)
            standard_error = np.sqrt(max(variance, 0.0))
            z_value = slope / standard_error if standard_error else np.nan
            p_value = (
                2 * norm.sf(abs(z_value))
                if np.isfinite(z_value)
                else np.nan
            )
            trend_rows.append(
                {
                    "population": population,
                    "period": period,
                    "mode": mode,
                    "country_code": country,
                    "emotion": emotion,
                    "start_year": int(group["analysis_year"].min()),
                    "end_year": int(group["analysis_year"].max()),
                    "log_odds_change_per_year": slope,
                    "odds_ratio_per_year": np.exp(slope),
                    "ci95_lower": np.exp(slope - 1.96 * standard_error),
                    "ci95_upper": np.exp(slope + 1.96 * standard_error),
                    "z_value": z_value,
                    "p_value": p_value,
                    "model_converged": full_fit["converged"],
                }
            )

    trends = _rank_bh(
        pd.DataFrame(trend_rows),
        p_column="p_value",
        group_columns=["population", "period", "mode"],
    )
    interactions = _rank_bh(
        pd.DataFrame(interaction_rows),
        p_column="p_value",
        group_columns=["population", "period", "mode"],
    )
    return trends, interactions


def build_statistical_tables(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    complete_through: int,
    partial_year: int | None,
    dominant_languages: Dict[str, str],
) -> Dict[str, Any]:
    """Builds inferential temporal results and language sensitivity tables."""
    working = SentimentAnalyzer.add_analysis_year(df)
    required = {
        "country_code",
        "language",
        "analysis_status",
        "analysis_year",
        "emotion",
    }
    missing = sorted(required - set(working.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if not _score_columns(working):
        raise ValueError("input does not contain per-emotion score columns")

    valid = working[
        (working["analysis_status"] == "analyzed")
        & working["country_code"].notna()
        & working["analysis_year"].notna()
    ].copy()
    valid["analysis_year"] = valid["analysis_year"].astype(int)

    populations = {
        "all_languages": valid,
        "dominant_languages": valid[
            valid.apply(
                lambda row: dominant_languages.get(row["country_code"])
                == row["language"],
                axis=1,
            )
        ],
    }
    annual_frames = [
        _annual_counts(frame, threshold=threshold, population=name)
        for name, frame in populations.items()
    ]
    annual = pd.concat(annual_frames, ignore_index=True)
    annual["is_partial_year"] = annual["analysis_year"].eq(partial_year)

    periods = {
        "complete_years": list(
            range(int(valid["analysis_year"].min()), complete_through + 1)
        )
    }
    if partial_year is not None:
        periods["including_partial_year"] = list(
            range(int(valid["analysis_year"].min()), partial_year + 1)
        )

    binary_frames = []
    distribution_frames = []
    trend_frames = []
    interaction_frames = []
    for period, years in periods.items():
        annual_period = annual[annual["analysis_year"].isin(years)]
        binary = _binary_temporal_tests(annual_period, years=years)
        binary.insert(1, "period", period)
        binary_frames.append(binary)
        for population, frame in populations.items():
            period_frame = frame[frame["analysis_year"].isin(years)]
            distribution_frames.append(
                _primary_distribution_tests(
                    period_frame,
                    population=population,
                    period=period,
                )
            )
        trends, interactions = _logistic_results(
            annual_period,
            period=period,
        )
        trend_frames.append(trends)
        interaction_frames.append(interactions)

    summary = {
        "input_reviews": int(len(working)),
        "analyzed_reviews": int(len(valid)),
        "complete_through": complete_through,
        "partial_year": partial_year,
        "emotion_threshold": threshold,
        "dominant_languages": dominant_languages,
        "population_counts": {
            name: int(len(frame)) for name, frame in populations.items()
        },
        "multiple_testing_correction": "Benjamini-Hochberg FDR",
        "significance_level": 0.05,
    }
    return {
        "annual": annual,
        "binary_tests": pd.concat(binary_frames, ignore_index=True),
        "distribution_tests": pd.concat(
            distribution_frames,
            ignore_index=True,
        ),
        "trends": pd.concat(trend_frames, ignore_index=True),
        "interactions": pd.concat(interaction_frames, ignore_index=True),
        "summary": summary,
    }
