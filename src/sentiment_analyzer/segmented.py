from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def build_token_windows(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """Splits text into overlapping windows that fit the model input."""
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_capacity = max_length - special_tokens
    if content_capacity <= 0:
        raise ValueError("max_length does not leave room for content tokens")
    if not 0 <= stride < content_capacity:
        raise ValueError("stride must be between 0 and content capacity - 1")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    windows: List[Dict[str, Any]] = []
    step = content_capacity - stride
    for index, start in enumerate(range(0, len(token_ids), step)):
        window_ids = token_ids[start : start + content_capacity]
        if not window_ids:
            break
        overlap = 0 if index == 0 else min(stride, len(window_ids))
        windows.append(
            {
                "segment_index": index,
                "start_token": start,
                "content_token_count": len(window_ids),
                "aggregation_weight": len(window_ids) - overlap,
                "text": tokenizer.decode(
                    window_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }
        )
        if start + content_capacity >= len(token_ids):
            break
    return windows


def aggregate_segment_scores(
    predictions: Iterable[Dict[str, Any]],
    weights: Iterable[int],
    *,
    threshold: float,
) -> Dict[str, Any]:
    """Aggregates independent emotion probabilities across text segments."""
    predictions = list(predictions)
    weights = list(weights)
    if len(predictions) != len(weights) or not predictions:
        raise ValueError("predictions and weights must have the same non-zero length")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if any(weight <= 0 for weight in weights):
        raise ValueError("all aggregation weights must be positive")

    labels = sorted(
        {
            label
            for prediction in predictions
            for label in prediction.get("scores", {})
        }
    )
    total_weight = sum(weights)
    result: Dict[str, Any] = {}
    for label in labels:
        scores = [
            float(prediction.get("scores", {}).get(label, 0.0))
            for prediction in predictions
        ]
        result[f"segmented_mean_{label}_score"] = sum(
            score * weight for score, weight in zip(scores, weights)
        ) / total_weight
        result[f"segmented_max_{label}_score"] = max(scores)
        result[f"segments_above_threshold_{label}"] = sum(
            score >= threshold for score in scores
        )

    if labels:
        result["segmented_mean_emotion"] = max(
            labels,
            key=lambda label: result[f"segmented_mean_{label}_score"],
        )
        result["segmented_mean_emotion_score"] = result[
            f"segmented_mean_{result['segmented_mean_emotion']}_score"
        ]
        result["segmented_max_emotion"] = max(
            labels,
            key=lambda label: result[f"segmented_max_{label}_score"],
        )
        result["segmented_max_emotion_score"] = result[
            f"segmented_max_{result['segmented_max_emotion']}_score"
        ]
    return result
