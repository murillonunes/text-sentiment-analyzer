import pandas as pd
import re
from typing import Dict, Any, List
from tqdm import tqdm

from sentiment_analyzer.preprocessor import Preprocessor
from sentiment_analyzer.backends.transformers import TransformersBackend

DEFAULT_EMOTION_LABELS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "frustration",
    "gratitude",
    "joy",
    "love",
    "neutral",
    "sadness",
    "surprise",
)


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

    @staticmethod
    def _text_unit_count(text: str) -> int:
        """
        Counts whitespace-delimited terms plus CJK characters.

        This keeps ``--min-words`` usable for Chinese text, which commonly has
        no spaces between lexical units.
        """
        without_cjk = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff]", " ", text)
        whitespace_terms = len(without_cjk.split())
        cjk_characters = len(
            re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text)
        )
        return whitespace_terms + cjk_characters

    @staticmethod
    def _pending_result(status: str) -> Dict[str, Any]:
        return {
            "label": None,
            "score": None,
            "scores": {},
            "status": status,
        }

    @staticmethod
    def normalize_voted_up(value: Any) -> bool | None:
        """Normalizes common Steam recommendation representations."""
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            return None

        if value == 1 or value == 1.0:
            return True
        if value == 0 or value == 0.0:
            return False

        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return None
        
    def analyze_list(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Analyze a simple list of texts.
        
        Returns:
            List of dicts containing 'label' and 'score'.
        """
        cleaned_texts = [self.preprocessor.clean(t) for t in texts]
        
        results = [self._pending_result("skipped_short") for _ in texts]
        valid_indices = []
        valid_texts = []
        
        for idx, text in enumerate(cleaned_texts):
            if not text:
                results[idx] = self._pending_result("empty_text")
            elif self._text_unit_count(text) >= self.min_words:
                valid_indices.append(idx)
                valid_texts.append(text)
                
        if valid_texts:
            valid_results = self.backend.predict_batch(valid_texts, batch_size=batch_size)
            for idx, res in zip(valid_indices, valid_results):
                results[idx] = res
                
        return results
        
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "review_text", voted_up_column: str = None, batch_size: int = 32) -> pd.DataFrame:
        """
        Analyze a pandas DataFrame containing a column of text.
        Adds 'emotion', 'emotion_score', and 'emotion_adjustment' columns to a copy of the DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
            
        df = df.copy()
        
        # Extract and clean texts
        texts = df[text_column].tolist()
        cleaned_texts = []
        for text in tqdm(texts, desc="Preprocessing texts"):
            cleaned_texts.append(self.preprocessor.clean(text))
            
        # Filter texts based on word count
        num_texts = len(cleaned_texts)
        results = [
            self._pending_result("skipped_short") for _ in range(num_texts)
        ]
        valid_indices = []
        valid_texts = []
        
        for idx, text in enumerate(cleaned_texts):
            if not text:
                results[idx] = self._pending_result("empty_text")
            elif self._text_unit_count(text) >= self.min_words:
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
                
        df["emotion"] = [result["label"] for result in results]
        df["emotion_score"] = pd.to_numeric(
            [result["score"] for result in results],
            errors="coerce",
        )
        df["analysis_status"] = [result["status"] for result in results]
        df["top_emotion_raw"] = df["emotion"]
        df["top_emotion_score_raw"] = df["emotion_score"]

        observed_labels = {
            label
            for result in results
            for label in result.get("scores", {})
        }
        for label in sorted(set(DEFAULT_EMOTION_LABELS) | observed_labels):
            column = f"emotion_{label}_score"
            df[column] = pd.to_numeric(
                [
                    result.get("scores", {}).get(label)
                    for result in results
                ],
                errors="coerce",
            )
        
        df["emotion_adjustment"] = "original"
        df["adjustment_suggested"] = False
        
        if voted_up_column and voted_up_column in df.columns:
            negative_emotions_to_flag = {
                "contempt",
                "anger",
                "frustration",
                "disgust",
            }
            suggested_count = 0
            for idx, row in df.iterrows():
                voted_val = self.normalize_voted_up(row[voted_up_column])
                if voted_val is True:
                    orig_emo = row["emotion"]
                    if orig_emo in negative_emotions_to_flag:
                        df.at[idx, "emotion_adjustment"] = "suggested"
                        df.at[idx, "adjustment_suggested"] = True
                        suggested_count += 1
            if suggested_count > 0:
                print(
                    "Flagged heuristic disagreement on "
                    f"{suggested_count} reviews without changing model predictions."
                )
        
        return df

    @staticmethod
    def add_analysis_year(df: pd.DataFrame) -> pd.DataFrame:
        """Adds a nullable analysis year from supported review date columns."""
        result = df.copy()
        if "analysis_year" in result.columns:
            return result
        if "date_created" in result.columns:
            raw_dates = result["date_created"].astype("string").str.strip()
            iso_dates = raw_dates.str.match(
                r"^\d{4}-\d{2}-\d{2}(?:$|[T\s])",
                na=False,
            )
            years = pd.Series(pd.NA, index=result.index, dtype="Int64")
            years.loc[iso_dates] = pd.to_datetime(
                raw_dates.loc[iso_dates],
                errors="coerce",
                format="ISO8601",
                utc=True,
            ).dt.year.astype("Int64")
            years.loc[~iso_dates] = pd.to_datetime(
                raw_dates.loc[~iso_dates],
                errors="coerce",
                dayfirst=True,
                utc=True,
            ).dt.year.astype("Int64")
            result["analysis_year"] = years
            return result
        elif "timestamp_created" in result.columns:
            dates = pd.to_datetime(
                result["timestamp_created"],
                errors="coerce",
                unit="s",
                utc=True,
            )
        else:
            return result
        result["analysis_year"] = dates.dt.year.astype("Int64")
        return result

    @staticmethod
    def summarize_emotions(
        df: pd.DataFrame,
        *,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Builds auditable multi-label metrics overall and by country/year.

        Percentages use all successfully analyzed reviews in the group as the
        denominator. Because labels are independent, percentages can sum above
        100 percent.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        score_columns = [
            column
            for column in df.columns
            if column.startswith("emotion_")
            and column.endswith("_score")
            and column != "emotion_score"
        ]
        if not score_columns:
            return pd.DataFrame(
                columns=[
                    "scope",
                    "country_code",
                    "analysis_year",
                    "emotion",
                    "valid_reviews",
                    "emotion_count",
                    "emotion_percent",
                    "mean_score",
                    "rank",
                    "threshold",
                ]
            )

        working = SentimentAnalyzer.add_analysis_year(df)
        if "analysis_status" in working.columns:
            working = working[working["analysis_status"] == "analyzed"]
        else:
            working = working[working["emotion"].notna()]

        groupings = [("overall", [])]
        if "country_code" in working.columns:
            groupings.append(("country", ["country_code"]))
            if "analysis_year" in working.columns:
                groupings.append(
                    ("country_year", ["country_code", "analysis_year"])
                )

        rows: List[Dict[str, Any]] = []
        for scope, group_columns in groupings:
            if group_columns:
                grouper = (
                    group_columns[0]
                    if len(group_columns) == 1
                    else group_columns
                )
                groups = working.groupby(grouper, dropna=False, sort=True)
            else:
                groups = [((), working)]

            for key, group in groups:
                key_values = key if isinstance(key, tuple) else (key,)
                dimensions = dict(zip(group_columns, key_values))
                valid_reviews = len(group)
                group_rows = []
                for column in score_columns:
                    emotion = column[len("emotion_") : -len("_score")]
                    scores = pd.to_numeric(group[column], errors="coerce")
                    count = int((scores >= threshold).sum())
                    group_rows.append(
                        {
                            "scope": scope,
                            "country_code": dimensions.get("country_code"),
                            "analysis_year": dimensions.get("analysis_year"),
                            "emotion": emotion,
                            "valid_reviews": valid_reviews,
                            "emotion_count": count,
                            "emotion_percent": (
                                round(count / valid_reviews * 100, 2)
                                if valid_reviews
                                else 0.0
                            ),
                            "mean_score": (
                                round(float(scores.mean()), 6)
                                if scores.notna().any()
                                else None
                            ),
                            "threshold": threshold,
                        }
                    )
                group_rows.sort(
                    key=lambda row: (
                        -row["emotion_count"],
                        -(
                            row["mean_score"]
                            if row["mean_score"] is not None
                            else -1.0
                        ),
                        row["emotion"],
                    )
                )
                for rank, row in enumerate(group_rows, start=1):
                    row["rank"] = rank
                    rows.append(row)

        return pd.DataFrame(rows)
        
    @staticmethod
    def evaluate_agreement(df: pd.DataFrame, emotion_col: str = "emotion", voted_up_col: str = "voted_up") -> Dict[str, Any]:
        """
        Evaluates the agreement rate between predicted emotion (mapped to polarity) 
        and the Steam recommendation (voted_up).
        Returns a dictionary with agreement statistics.
        """
        if voted_up_col not in df.columns or emotion_col not in df.columns:
            return {}
            
        # Normalize recommendation values before filtering. CSV readers often
        # represent the same Steam field as bool, integer, float, or text.
        valid_df = df[
            df[emotion_col].notna() &
            (df[emotion_col] != "skipped_short")
        ].copy()
        valid_df["normalized_voted_up"] = valid_df[voted_up_col].map(
            SentimentAnalyzer.normalize_voted_up
        )
        valid_df = valid_df[valid_df["normalized_voted_up"].notna()]
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
        
        valid_df["expected_sentiment"] = valid_df[
            "normalized_voted_up"
        ].map(lambda value: "positive" if value else "negative")
        
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
