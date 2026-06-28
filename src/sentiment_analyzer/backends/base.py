from abc import ABC, abstractmethod
from typing import Dict, List, Union

class BaseSentimentBackend(ABC):
    """
    Base class for sentiment analyzer backends.
    """
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict label for a single text.
        
        Returns:
            Dict containing:
                - 'label': predicted class (str)
                - 'score': confidence score (float, usually between 0 and 1)
        """
        pass
        
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict labels for a batch of texts.
        
        Returns:
            List of dicts containing 'label' and 'score' keys.
        """
        pass
