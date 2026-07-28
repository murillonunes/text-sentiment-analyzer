import pytest

from sentiment_analyzer.segmented import (
    aggregate_segment_scores,
    build_token_windows,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [int(value) for value in text.split()]

    def decode(self, token_ids, **_kwargs):
        return " ".join(str(value) for value in token_ids)

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 2


def test_token_windows_respect_capacity_and_overlap():
    windows = build_token_windows(
        FakeTokenizer(),
        " ".join(str(value) for value in range(12)),
        max_length=8,
        stride=2,
    )

    assert [window["content_token_count"] for window in windows] == [6, 6, 4]
    assert [window["start_token"] for window in windows] == [0, 4, 8]
    assert [window["aggregation_weight"] for window in windows] == [6, 4, 2]
    assert windows[1]["text"] == "4 5 6 7 8 9"


def test_token_windows_reject_invalid_stride():
    with pytest.raises(ValueError):
        build_token_windows(
            FakeTokenizer(),
            "1 2 3",
            max_length=8,
            stride=6,
        )


def test_segment_scores_include_weighted_mean_max_and_threshold_count():
    result = aggregate_segment_scores(
        [
            {"scores": {"neutral": 0.8, "joy": 0.2}},
            {"scores": {"neutral": 0.3, "joy": 0.9}},
        ],
        [3, 1],
        threshold=0.5,
    )

    assert result["segmented_mean_neutral_score"] == pytest.approx(0.675)
    assert result["segmented_mean_joy_score"] == pytest.approx(0.375)
    assert result["segmented_max_neutral_score"] == 0.8
    assert result["segmented_max_joy_score"] == 0.9
    assert result["segments_above_threshold_neutral"] == 1
    assert result["segments_above_threshold_joy"] == 1
    assert result["segmented_mean_emotion"] == "neutral"
    assert result["segmented_max_emotion"] == "joy"
