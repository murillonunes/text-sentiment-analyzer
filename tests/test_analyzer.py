from unittest.mock import MagicMock, patch
import pandas as pd
from sentiment_analyzer.analyzer import SentimentAnalyzer

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
    assert result_df.loc[0, "emotion"] == "skipped_short"
    assert result_df.loc[0, "emotion_score"] == 0.0
    
    assert result_df.loc[1, "emotion"] == "joy"
    assert result_df.loc[1, "emotion_score"] == 0.99
    
    # Verify agreement evaluation ignores the skipped review
    metrics = SentimentAnalyzer.evaluate_agreement(result_df)
    assert metrics["agreement_rate"] == 1.0
    assert metrics["total_count"] == 1
