from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import json
import math
import pandas as pd
import sys
from sentiment_analyzer.analyzer import SentimentAnalyzer
from sentiment_analyzer import cli

@patch("sentiment_analyzer.backends.transformers.pipeline")
def test_analyzer_dataframe(mock_pipeline):
    # Mock pipeline return values
    mock_pipe_instance = MagicMock()
    # Mock the return (which can be a list of lists of dicts when top_k=1 is specified)
    mock_pipe_instance.return_value = [[{"label": "joy", "score": 0.99}]]
    mock_pipeline.return_value = mock_pipe_instance
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(backend_model="mock-model", device="cpu")
    
    # Input DataFrame
    df = pd.DataFrame({
        "review_text": ["This game is absolutely awesome!"],
        "voted_up": [True]
    })
    
    # Run analysis
    result_df = analyzer.analyze_dataframe(df, text_column="review_text", batch_size=1)
    
    # Verify outputs
    assert "emotion" in result_df.columns
    assert "emotion_score" in result_df.columns
    assert result_df.loc[0, "emotion"] == "joy"
    assert result_df.loc[0, "emotion_score"] == 0.99
    
    # Verify agreement evaluation
    metrics = SentimentAnalyzer.evaluate_agreement(result_df)
    assert metrics["agreement_rate"] == 1.0
    assert metrics["total_count"] == 1

@patch("sentiment_analyzer.backends.transformers.pipeline")
def test_analyzer_min_words_filter(mock_pipeline):
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [[{"label": "joy", "score": 0.99}]]
    mock_pipeline.return_value = mock_pipe_instance
    
    # Initialize analyzer with min_words = 5
    analyzer = SentimentAnalyzer(backend_model="mock-model", device="cpu", min_words=5)
    
    # Input DataFrame: one short review (4 words), one long review (7 words)
    df = pd.DataFrame({
        "review_text": [
            "This is very short.",         # 4 words
            "This review has enough words to analyze." # 7 words
        ],
        "voted_up": [True, True]
    })
    
    # Run analysis
    result_df = analyzer.analyze_dataframe(df, text_column="review_text", batch_size=1)
    
    # Verify outputs
    assert pd.isna(result_df.loc[0, "emotion"])
    assert pd.isna(result_df.loc[0, "emotion_score"])
    assert result_df.loc[0, "analysis_status"] == "skipped_short"
    
    assert result_df.loc[1, "emotion"] == "joy"
    assert result_df.loc[1, "emotion_score"] == 0.99
    
    # Verify agreement evaluation ignores the skipped review
    metrics = SentimentAnalyzer.evaluate_agreement(result_df)
    assert metrics["agreement_rate"] == 1.0
    assert metrics["total_count"] == 1

@patch("sentiment_analyzer.backends.transformers.pipeline")
def test_analyzer_heuristic_flags_disagreement_without_changing_prediction(
    mock_pipeline,
):
    mock_pipe_instance = MagicMock()
    # Mock pipeline returning "contempt" for a review
    mock_pipe_instance.return_value = [[{"label": "contempt", "score": 0.48}]]
    mock_pipeline.return_value = mock_pipe_instance
    
    analyzer = SentimentAnalyzer(backend_model="mock-model", device="cpu")
    
    # Input DataFrame with a review classified as contempt but voted_up = True
    df = pd.DataFrame({
        "review_text": ["simplesmente o melhor jogo ja feito"],
        "voted_up": [1]
    })
    
    # Run analysis passing voted_up_column
    result_df = analyzer.analyze_dataframe(df, text_column="review_text", voted_up_column="voted_up", batch_size=1)
    
    # The model output remains intact; voted_up only raises an audit flag.
    assert result_df.loc[0, "emotion"] == "contempt"
    assert result_df.loc[0, "emotion_score"] == 0.48
    assert result_df.loc[0, "emotion_contempt_score"] == 0.48
    assert pd.isna(result_df.loc[0, "emotion_joy_score"])
    assert result_df.loc[0, "emotion_adjustment"] == "suggested"
    assert bool(result_df.loc[0, "adjustment_suggested"])


def test_voted_up_normalization_accepts_common_csv_representations():
    positive_values = [True, 1, 1.0, "true", "TRUE", " 1 "]
    negative_values = [False, 0, 0.0, "false", "FALSE", " 0 "]
    unavailable_values = [None, float("nan"), "", "unknown", 2]

    assert all(
        SentimentAnalyzer.normalize_voted_up(value) is True
        for value in positive_values
    )
    assert all(
        SentimentAnalyzer.normalize_voted_up(value) is False
        for value in negative_values
    )
    assert all(
        SentimentAnalyzer.normalize_voted_up(value) is None
        for value in unavailable_values
    )


def test_agreement_normalizes_numeric_and_text_recommendations():
    frame = pd.DataFrame(
        {
            "emotion": ["joy", "anger", "love", "sadness", "neutral"],
            "voted_up": [1, 0, "true", "false", "unknown"],
        }
    )

    metrics = SentimentAnalyzer.evaluate_agreement(frame)

    assert metrics["total_count"] == 4
    assert metrics["agreement_rate"] == 1.0
    assert metrics["positive_recommendation_count"] == 2
    assert metrics["negative_recommendation_count"] == 2


@patch("sentiment_analyzer.backends.transformers.pipeline")
def test_analyzer_preserves_all_multilabel_probabilities(mock_pipeline):
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [
        [
            {"label": "joy", "score": 0.81},
            {"label": "gratitude", "score": 0.72},
            {"label": "sadness", "score": 0.11},
        ]
    ]
    mock_pipeline.return_value = mock_pipe_instance

    analyzer = SentimentAnalyzer(backend_model="mock-model", device="cpu")
    result_df = analyzer.analyze_dataframe(
        pd.DataFrame({"review_text": ["Great game"]}),
        batch_size=1,
    )

    assert result_df.loc[0, "emotion"] == "joy"
    assert result_df.loc[0, "emotion_score"] == 0.81
    assert result_df.loc[0, "top_emotion_raw"] == "joy"
    assert result_df.loc[0, "emotion_joy_score"] == 0.81
    assert result_df.loc[0, "emotion_gratitude_score"] == 0.72
    assert result_df.loc[0, "emotion_sadness_score"] == 0.11
    assert result_df.loc[0, "analysis_status"] == "analyzed"


@patch("sentiment_analyzer.backends.transformers.pipeline")
def test_chinese_text_uses_cjk_units_for_minimum_length(mock_pipeline):
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [
        [{"label": "joy", "score": 0.9}]
    ]
    mock_pipeline.return_value = mock_pipe_instance

    analyzer = SentimentAnalyzer(
        backend_model="mock-model",
        device="cpu",
        min_words=5,
    )
    result_df = analyzer.analyze_dataframe(
        pd.DataFrame({"review_text": ["这个游戏真的很好玩"]}),
        batch_size=1,
    )

    assert result_df.loc[0, "analysis_status"] == "analyzed"
    assert result_df.loc[0, "emotion"] == "joy"


@patch("sentiment_analyzer.backends.transformers.pipeline")
def test_emoji_and_punctuation_only_text_reaches_model(mock_pipeline):
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [
        [{"label": "joy", "score": 0.9}]
    ]
    mock_pipeline.return_value = mock_pipe_instance

    analyzer = SentimentAnalyzer(backend_model="mock-model", device="cpu")
    result_df = analyzer.analyze_dataframe(
        pd.DataFrame({"review_text": ["😂 ❤️ !!!"]}),
        batch_size=1,
    )

    assert result_df.loc[0, "analysis_status"] == "analyzed"
    mock_pipe_instance.assert_called_once()
    assert mock_pipe_instance.call_args.args[0] == ["😂 ❤️ !!!"]


@patch("sentiment_analyzer.backends.transformers.pipeline")
def test_missing_text_is_not_converted_to_literal_nan(mock_pipeline):
    analyzer = SentimentAnalyzer(backend_model="mock-model", device="cpu")
    result_df = analyzer.analyze_dataframe(
        pd.DataFrame({"review_text": [None, float("nan")]}),
        batch_size=2,
    )

    assert result_df["emotion"].isna().all()
    assert result_df["emotion_score"].isna().all()
    assert result_df["analysis_status"].tolist() == [
        "empty_text",
        "empty_text",
    ]
    mock_pipeline.return_value.assert_not_called()


def test_analysis_year_parses_iso_and_day_first_dates_without_warning():
    frame = pd.DataFrame(
        {
            "date_created": [
                "2020-12-10",
                "2021-07-26T14:30:00Z",
                "27/07/2022",
                "invalid",
                None,
            ]
        }
    )

    result = SentimentAnalyzer.add_analysis_year(frame)

    assert result["analysis_year"].tolist()[:3] == [2020, 2021, 2022]
    assert result["analysis_year"].isna().tolist() == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_emotion_summary_uses_all_valid_reviews_and_independent_thresholds():
    frame = pd.DataFrame(
        {
            "country_code": ["BR", "BR", "US", "US"],
            "date_created": [
                "01/01/2024",
                "02/01/2024",
                "01/01/2024",
                "02/01/2024",
            ],
            "emotion": ["joy", "sadness", "joy", None],
            "analysis_status": [
                "analyzed",
                "analyzed",
                "analyzed",
                "inference_failed",
            ],
            "emotion_joy_score": [0.9, 0.6, 0.8, math.nan],
            "emotion_sadness_score": [0.7, 0.8, 0.2, math.nan],
        }
    )

    summary = SentimentAnalyzer.summarize_emotions(frame, threshold=0.5)
    br_joy = summary[
        (summary["scope"] == "country_year")
        & (summary["country_code"] == "BR")
        & (summary["analysis_year"] == 2024)
        & (summary["emotion"] == "joy")
    ].iloc[0]
    br_sadness = summary[
        (summary["scope"] == "country_year")
        & (summary["country_code"] == "BR")
        & (summary["analysis_year"] == 2024)
        & (summary["emotion"] == "sadness")
    ].iloc[0]
    us_joy = summary[
        (summary["scope"] == "country")
        & (summary["country_code"] == "US")
        & (summary["emotion"] == "joy")
    ].iloc[0]

    assert br_joy["valid_reviews"] == 2
    assert br_joy["emotion_count"] == 2
    assert br_joy["emotion_percent"] == 100.0
    assert br_joy["mean_score"] == 0.75
    assert br_sadness["emotion_count"] == 2
    assert br_sadness["emotion_percent"] == 100.0
    assert us_joy["valid_reviews"] == 1
    assert us_joy["mean_score"] == 0.8


def test_cli_writes_analyzed_data_summary_and_model_provenance(
    tmp_path,
    monkeypatch,
):
    class FakePipeline:
        def __init__(self):
            self.model = SimpleNamespace(
                config=SimpleNamespace(
                    _name_or_path="mock/model",
                    _commit_hash="revision-1",
                    problem_type="multi_label_classification",
                    num_labels=2,
                    id2label={0: "joy", 1: "sadness"},
                )
            )
            self.tokenizer = SimpleNamespace(model_max_length=192)

        def __call__(self, texts, **_kwargs):
            prediction = [
                {"label": "joy", "score": 0.8},
                {"label": "sadness", "score": 0.6},
            ]
            if isinstance(texts, list):
                return [prediction for _ in texts]
            return [prediction]

    input_path = tmp_path / "reviews.csv"
    output_path = tmp_path / "analyzed.csv"
    pd.DataFrame(
        {
            "country_code": ["BR", "US"],
            "date_created": ["01/01/2024", "02/01/2024"],
            "review_text": ["Muito bom", "Great game"],
            "voted_up": [True, True],
        }
    ).to_csv(input_path, index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sentiment-analyzer",
            str(input_path),
            "--output-path",
            str(output_path),
            "--skip-report",
        ],
    )
    with patch(
        "sentiment_analyzer.backends.transformers.pipeline",
        return_value=FakePipeline(),
    ):
        cli.main()

    analyzed = pd.read_csv(output_path)
    run_dir = next((tmp_path / "outputs").glob("run_*"))
    summary_frame = pd.read_csv(run_dir / "emotion_summary.csv")
    execution_summary = json.loads((run_dir / "summary.json").read_text())

    assert analyzed["analysis_status"].tolist() == ["analyzed", "analyzed"]
    assert analyzed["emotion_joy_score"].tolist() == [0.8, 0.8]
    assert {
        "overall",
        "country",
        "country_year",
    }.issubset(set(summary_frame["scope"]))
    assert execution_summary["emotion_threshold"] == 0.5
    assert execution_summary["model_metadata"]["revision"] == "revision-1"
