from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseSentimentBackend(ABC):
    """
    Base class for sentiment analyzer backends.
    """
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict label for a single text.
        
        Returns:
            Dict containing:
                - 'label': predicted class (str)
                - 'score': confidence score (float, usually between 0 and 1)
                - 'scores': mapping containing every returned label score
                - 'status': analysis outcome
        """
        pass
        
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict labels for a batch of texts.
        
        Returns:
            List of prediction dictionaries with raw label scores and status.
        """
        pass
