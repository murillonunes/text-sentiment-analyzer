from typing import Any, Dict, List
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
                top_k=None,
            )
            
    def _normalize_label(self, label: str) -> str:
        """
        Normalize label to lowercase and strip whitespace.
        """
        return label.strip().lower()

    def metadata(self) -> Dict[str, Any]:
        """Returns the loaded model identity and classification configuration."""
        self._init_pipeline()
        model = getattr(self.pipeline, "model", None)
        config = getattr(model, "config", None)
        tokenizer = getattr(self.pipeline, "tokenizer", None)
        return {
            "requested_model": self.model_name,
            "resolved_model": getattr(config, "_name_or_path", self.model_name),
            "revision": getattr(config, "_commit_hash", None),
            "problem_type": getattr(config, "problem_type", None),
            "num_labels": getattr(config, "num_labels", None),
            "id2label": getattr(config, "id2label", None),
            "tokenizer_max_length": getattr(
                tokenizer,
                "model_max_length",
                None,
            ),
            "device": self.device,
        }
            
    def _parse_prediction(self, raw_result: Any) -> Dict[str, Any]:
        """Normalizes a pipeline response while preserving every label score."""
        values = raw_result
        while (
            isinstance(values, list)
            and len(values) == 1
            and isinstance(values[0], list)
        ):
            values = values[0]
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            raise ValueError("Unexpected model prediction format")

        scores = {
            self._normalize_label(str(item["label"])): float(item["score"])
            for item in values
            if isinstance(item, dict)
            and "label" in item
            and "score" in item
        }
        if not scores:
            raise ValueError("Model prediction did not contain label scores")
        label, score = max(scores.items(), key=lambda item: item[1])
        return {
            "label": label,
            "score": score,
            "scores": scores,
            "status": "analyzed",
        }

    @staticmethod
    def _failed_prediction() -> Dict[str, Any]:
        return {
            "label": None,
            "score": None,
            "scores": {},
            "status": "inference_failed",
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict label for a single text.
        """
        self._init_pipeline()
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "label": None,
                "score": None,
                "scores": {},
                "status": "empty_text",
            }

        try:
            results = self.pipeline(text, truncation=True)
            return self._parse_prediction(results)
        except Exception:
            return self._failed_prediction()
                
    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Predict labels for a batch of texts.
        """
        self._init_pipeline()
        
        # Filter and track indices of valid texts
        results = [self._failed_prediction() for _ in texts]
        valid_indices = []
        valid_texts = []
        
        for idx, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_indices.append(idx)
                valid_texts.append(text)
            else:
                results[idx] = {
                    "label": None,
                    "score": None,
                    "scores": {},
                    "status": "empty_text",
                }
                
        if not valid_texts:
            return results
            
        try:
            # Run batch inference
            batch_results = self.pipeline(
                valid_texts,
                batch_size=batch_size,
                truncation=True,
            )
            for idx, raw_result in zip(valid_indices, batch_results):
                results[idx] = self._parse_prediction(raw_result)
        except Exception:
            # Fallback to single predictions if batch fails
            for idx in valid_indices:
                results[idx] = self.predict(texts[idx])
                
        return results
