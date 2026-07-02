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

def test_preprocessor_standalone_punctuation():
    preprocessor = Preprocessor()
    assert preprocessor.clean("MELHOR JOGO JA CRIADO !!!") == "MELHOR JOGO JA CRIADO"
    assert preprocessor.clean("A - B") == "A B"
    assert preprocessor.clean("Este jogo é simplesmente fantástico! Muito bom mesmo.") == "Este jogo é simplesmente fantástico! Muito bom mesmo."
