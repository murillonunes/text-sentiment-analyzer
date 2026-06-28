from typing import Dict, List, Union
import torch
from transformers import pipeline

from sentiment_analyzer.backends.base import BaseSentimentBackend

class TransformersBackend(BaseSentimentBackend):
    """
    Emotion and general classification backend using Hugging Face Transformers pipeline.
    """
    
    def __init__(self, model_name: str = "tabularisai/multilingual-emotion-classification", device: str = None):
        """
        Initialize the backend.
        
        Args:
            model_name: Hugging Face model identifier.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.pipeline = None
        
    def _init_pipeline(self):
        """Lazy load the pipeline to avoid importing and loading model until needed."""
        if self.pipeline is None:
            # Map device name to pipeline device argument:
            # 0 or more for CUDA index, -1 for CPU
            device_idx = 0 if self.device == "cuda" else -1
            
            # Create classification pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=device_idx,
                top_k=1
            )
            
    def _normalize_label(self, label: str) -> str:
        """
        Normalize label to lowercase and strip whitespace.
        """
        return label.strip().lower()
            
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict label for a single text.
        """
        self._init_pipeline()
        if not text or not isinstance(text, str) or not text.strip():
            return {"label": "neutral", "score": 0.0}
            
        # Run pipeline
        try:
            results = self.pipeline(text)
            # If top_k=1 is used, results is [[{'label': '...', 'score': ...}]]
            res = results[0][0] if isinstance(results[0], list) else results[0]
            return {
                "label": self._normalize_label(res["label"]),
                "score": float(res["score"])
            }
        except Exception as e:
            try:
                # Truncate text to a reasonable length and try again
                truncated_text = text[:500]
                results = self.pipeline(truncated_text)
                res = results[0][0] if isinstance(results[0], list) else results[0]
                return {
                    "label": self._normalize_label(res["label"]),
                    "score": float(res["score"])
                }
            except Exception:
                return {"label": "neutral", "score": 0.0}
                
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Union[str, float]]]:
        """
        Predict labels for a batch of texts.
        """
        self._init_pipeline()
        
        # Filter and track indices of valid texts
        results = [{"label": "neutral", "score": 0.0} for _ in texts]
        valid_indices = []
        valid_texts = []
        
        for idx, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_indices.append(idx)
                # Truncate to 1000 characters to be safe.
                valid_texts.append(text[:1000])
                
        if not valid_texts:
            return results
            
        try:
            # Run batch inference
            batch_results = self.pipeline(valid_texts, batch_size=batch_size, truncation=True)
            for idx, res in zip(valid_indices, batch_results):
                # If top_k=1 is used, res is [{'label': '...', 'score': ...}]
                single_res = res[0] if isinstance(res, list) else res
                results[idx] = {
                    "label": self._normalize_label(single_res["label"]),
                    "score": float(single_res["score"])
                }
        except Exception as e:
            # Fallback to single predictions if batch fails
            for idx in valid_indices:
                results[idx] = self.predict(texts[idx])
                
        return results
