import pandas as pd
from typing import Dict, Any, List
from tqdm import tqdm

from sentiment_analyzer.preprocessor import Preprocessor
from sentiment_analyzer.backends.transformers import TransformersBackend

class SentimentAnalyzer:
    """
    Main class orchestrating the text preprocessing and emotion analysis pipeline.
    """
    
    def __init__(self, backend_model: str = "tabularisai/multilingual-emotion-classification", device: str = None, preprocess_config: Dict[str, Any] = None, min_words: int = 0):
        """
        Initialize the analyzer.
        
        Args:
            backend_model: Name of the Hugging Face model.
            device: 'cuda', 'cpu', or None (auto-detect).
            preprocess_config: Config dictionary for the preprocessor.
            min_words: Minimum number of words required in a text to be analyzed.
        """
        self.preprocessor = Preprocessor(**(preprocess_config or {}))
        self.backend = TransformersBackend(model_name=backend_model, device=device)
        self.min_words = min_words
        
    def analyze_list(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Analyze a simple list of texts.
        
        Returns:
            List of dicts containing 'label' and 'score'.
        """
        cleaned_texts = [self.preprocessor.clean(t) for t in texts]
        
        results = [{"label": "skipped_short", "score": 0.0} for _ in texts]
        valid_indices = []
        valid_texts = []
        
        for idx, text in enumerate(cleaned_texts):
            if len(text.split()) >= self.min_words:
                valid_indices.append(idx)
                valid_texts.append(text)
                
        if valid_texts:
            valid_results = self.backend.predict_batch(valid_texts, batch_size=batch_size)
            for idx, res in zip(valid_indices, valid_results):
                results[idx] = res
                
        return results
        
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "review_text", batch_size: int = 32) -> pd.DataFrame:
        """
        Analyze a pandas DataFrame containing a column of text.
        Adds 'emotion' and 'emotion_score' columns to a copy of the DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
            
        df = df.copy()
        
        # Extract and clean texts
        texts = df[text_column].astype(str).tolist()
        cleaned_texts = []
        for text in tqdm(texts, desc="Preprocessing texts"):
            cleaned_texts.append(self.preprocessor.clean(text))
            
        # Filter texts based on word count
        num_texts = len(cleaned_texts)
        results = [{"label": "skipped_short", "score": 0.0} for _ in range(num_texts)]
        valid_indices = []
        valid_texts = []
        
        for idx, text in enumerate(cleaned_texts):
            word_count = len(text.split())
            if word_count >= self.min_words:
                valid_indices.append(idx)
                valid_texts.append(text)
                
        # Analyze emotions in batches
        num_valid = len(valid_texts)
        if num_valid > 0:
            valid_results = []
            with tqdm(total=num_valid, desc="Analyzing emotions") as pbar:
                for i in range(0, num_valid, batch_size):
                    batch = valid_texts[i:i+batch_size]
                    batch_res = self.backend.predict_batch(batch, batch_size=batch_size)
                    valid_results.extend(batch_res)
                    pbar.update(len(batch))
            
            for idx, res in zip(valid_indices, valid_results):
                results[idx] = res
                
        df["emotion"] = [r["label"] for r in results]
        df["emotion_score"] = [r["score"] for r in results]
        
        return df
        
    @staticmethod
    def evaluate_agreement(df: pd.DataFrame, emotion_col: str = "emotion", voted_up_col: str = "voted_up") -> Dict[str, Any]:
        """
        Evaluates the agreement rate between predicted emotion (mapped to polarity) 
        and the Steam recommendation (voted_up).
        Returns a dictionary with agreement statistics.
        """
        if voted_up_col not in df.columns or emotion_col not in df.columns:
            return {}
            
        # Filter for rows that have valid emotion and recommendation, excluding skipped rows
        valid_df = df[
            df[voted_up_col].notna() & 
            df[emotion_col].notna() & 
            (df[emotion_col] != "skipped_short")
        ].copy()
        total = len(valid_df)
        if total == 0:
            return {"total_count": 0}
            
        # Map predicted emotions to positive/negative/neutral polarities
        emotion_to_polarity = {
            "joy": "positive",
            "love": "positive",
            "optimism": "positive",
            "gratitude": "positive",
            "thankfulness": "positive",
            "surprise": "positive",
            
            "sadness": "negative",
            "anger": "negative",
            "fear": "negative",
            "disgust": "negative",
            "frustration": "negative",
            "contempt": "negative",
            "pessimism": "negative",
            
            "neutral": "neutral",
            "positive": "positive",
            "negative": "negative"
        }
        
        valid_df["predicted_polarity"] = valid_df[emotion_col].map(emotion_to_polarity).fillna("neutral")
        
        # Map voted_up (True -> positive, False -> negative)
        valid_df["expected_sentiment"] = valid_df[voted_up_col].map({True: "positive", False: "negative"})
        
        # Calculate agreement rate (excluding neutral from matching expectations unless they match)
        correct = (valid_df["predicted_polarity"] == valid_df["expected_sentiment"]).sum()
        agreement_rate = correct / total
        
        # Calculate breakdown for positive recommendations (voted_up = True)
        pos_df = valid_df[valid_df["expected_sentiment"] == "positive"]
        pos_total = len(pos_df)
        pos_agreement = (pos_df["predicted_polarity"] == "positive").sum() / pos_total if pos_total > 0 else 0.0
        
        # Calculate breakdown for negative recommendations (voted_up = False)
        neg_df = valid_df[valid_df["expected_sentiment"] == "negative"]
        neg_total = len(neg_df)
        neg_agreement = (neg_df["predicted_polarity"] == "negative").sum() / neg_total if neg_total > 0 else 0.0
        
        emotion_counts = valid_df[emotion_col].value_counts().to_dict()
        
        return {
            "total_count": total,
            "agreement_rate": float(agreement_rate),
            "positive_recommendation_count": pos_total,
            "positive_recommendation_agreement": float(pos_agreement),
            "negative_recommendation_count": neg_total,
            "negative_recommendation_agreement": float(neg_agreement),
            "emotion_counts": {str(k): int(v) for k, v in emotion_counts.items()}
        }
