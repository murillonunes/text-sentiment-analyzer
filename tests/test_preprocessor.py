from sentiment_analyzer.preprocessor import Preprocessor

def test_preprocessor_basic_cleaning():
    preprocessor = Preprocessor()
    
    # Test HTML removal
    assert preprocessor.clean("<p>Hello World</p>") == "Hello World"
    
    # Test URL removal
    assert preprocessor.clean("Check this out: https://example.com/foo") == "Check this out:"
    
    # Test whitespace normalization
    assert preprocessor.clean("Hello   \n  World  ") == "Hello World"
    
    # Test combined cleaning
    assert preprocessor.clean("  <p>Hello  http://test.com  </p>  ") == "Hello"
