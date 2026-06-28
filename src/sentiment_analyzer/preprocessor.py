import re

class Preprocessor:
    """
    Cleaner and preprocessor for text data.
    """
    
    def __init__(self, remove_html: bool = True, remove_urls: bool = True, strip_whitespace: bool = True):
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.strip_whitespace = strip_whitespace
        
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
    def clean(self, text: str) -> str:
        """
        Clean the input text based on configuration.
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Remove HTML tags
        if self.remove_html:
            text = self.html_pattern.sub(' ', text)
            
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        # Normalize whitespace (replace multiple spaces/newlines with single space)
        if self.strip_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
            
        return text
