from utils.content_parser_analyzer import ContentParserAnalyzer

# Sample text containing various elements to parse
sample_text = """
Here's a code example from https://example.com:

```python
def greet(name):
    # This obviously works perfectly
    return f"Hello, {name}!"  # Always returns greeting
```
test@example.com
Contact our support team at support@example.com.
Studies show that this approach is highly effective.
This is probably the best solution, but maybe there are alternatives.
"""

def test_code_analysis():
    """Test code extraction and analysis capabilities"""
    parser = ContentParserAnalyzer(sample_text)
    result = (parser
        .extract_python_code_block()
        .detect_code_language()
        .detect_bias()
        .get())
    
    print("=== Code Analysis ===")
    print("Extracted Code:", result)
    print("Analysis Details:", parser.get_counts())

    assert "def greet(name):" in result, "Code extraction failed"
    assert "python" in parser.get_counts().get("code_language", ["unknown"]), "Code language detection failed"

def test_text_processing():
    """Test text normalization, stopword removal, and summarization"""
    parser = ContentParserAnalyzer(sample_text)
    result = (parser
        .normalize_text()
        .remove_stopwords()
        .summarize_text(50)
        .get())

    print("\n=== Text Processing ===")
    print("Processed Text:", result)
    print("Processing Details:", parser.get_counts())

    assert "code example" in result.lower(), "Text normalization failed"
    assert len(result) <= 53, "Summarization failed"

def test_nlp_analysis():
    """Test sentiment analysis, vagueness detection, and hedging detection"""
    parser = ContentParserAnalyzer(sample_text)
    result = (parser
              .extract_text_without_code_blocks()
              .normalize_text()
        .analyze_sentiment()
        .detect_hedging()
        .detect_vagueness()
        .get())

    print("\n=== NLP Analysis ===")
    print("Analyzed Text:", result)
    print("NLP Analysis Details:", parser.get_counts())

    assert "sentiment" in parser.get_counts(), "Sentiment analysis missing"
    assert "hedging" in parser.get_counts(), "Hedging detection missing"
    assert "vagueness" in parser.get_counts(), "Vagueness detection missing"

def test_metadata_extraction():
    """Test URL and email extraction"""
    parser = ContentParserAnalyzer(sample_text)
    parser.extract_urls().reset_remaining().extract_emails()
    
    result = parser.metadata_extractor.get()  

    print("\n=== Metadata Extraction ===")
    print("Extracted Metadata:", result)
    print("Extraction Details:", parser.get_counts())
    

    assert "urls" in result and any("https://example.com" in url for url in result["urls"]), "URL extraction failed"
    assert "emails" in result and any("support@example.com" in email for email in result["emails"]), "Email extraction failed"

def test_logical_analysis():
    """Test logical analysis features such as fallacy detection and coherence"""
    logical_text = "If we allow A, then B will definitely happen. Everyone knows that experts always agree."
    parser = ContentParserAnalyzer(logical_text)
    result = (parser
        .detect_fallacies()
        .evaluate_coherence()
        .get())

    print("\n=== Logical Analysis ===")
    print("Logical Analysis Text:", result)
    print("Logical Analysis Details:", parser.get_counts())

    assert "fallacies" in parser.get_counts(), "Fallacy detection missing"
    assert "coherence" in parser.get_counts(), "Coherence evaluation missing"

def test_full_pipeline():
    """Test a full pipeline involving multiple chained methods"""
    parser = ContentParserAnalyzer(sample_text)
    result = (parser
        .extract_python_code_block()
        .detect_code_language()
        .reset_remaining()
        .remove_code_blocks()
        .normalize_text()
        .analyze_sentiment()
        .extract_urls()
        .reset_remaining()
        .extract_emails()
        .reset_remaining()
        .summarize_text(60)
        .get())

    print("\n=== Full Pipeline Test ===")
    print("Final Processed Output:", result)
    print("Final Analysis Details:", parser.get_counts())
    
        
    
    
    #assert "python" in parser.get_counts().get("code_language", []), "Code language detection failed"
    assert "https://example.com" in parser.metadata_extractor.metadata.get("urls", []), "URL extraction failed"
    assert "support@example.com" in parser.metadata_extractor.metadata.get("emails", []), "Email extraction failed"
    assert len(result) <= 63, "Summarization failed"

### Test CodeParser functionality through ContentParserAnalyzer

sample_text_code_parser = """
        Here’s a Python function:

        ```python
        def greatings(name):
            return f"Hello, {name}!"
        ```
        
        Here's another function:
        
        ```python
        def another_func(name):
            return f"Hello world!"
        ```

        A JSON example:

        ```json
        { "name": "John", "age": 30 }
        ```

        A SQL query:

        ```sql
        SELECT * FROM users WHERE age > 21;
        ```
        """
def test_extract_code_blocks():
    """Test extraction of code blocks using ContentParserAnalyzer"""
    parser = ContentParserAnalyzer(sample_text_code_parser)
    
    # First extract Python code and check language
    result = (parser
        .extract_python_code_block()
        .detect_code_language()
        .get())
    print("\n=== Python Code Extraction ===")
    print("Extracted Python Code:", result)
    print("Language Detection:", parser.get_counts())
    
    parser = ContentParserAnalyzer(sample_text_code_parser)
    result = (parser
        .detect_code_language()
        .extract_python_code_block()
        .strip_code_markers()
        .get())
    print("\n=== Python Code Extraction1 ===")
    print("Extracted Python Code:", result)
    print("Language Detection ALL:", parser.get_counts())
    
    parser = ContentParserAnalyzer(sample_text_code_parser)
    result = (parser
        .extract_python_code_block()
        .reset_remaining()
        .extract_python_code_block()
        .detect_code_language()
        .get())
    print("\n=== Python Code Extraction2 ===")
    print("Extracted Python Code:", result)
    print("Language Detection python:", parser.get_counts())
    
    parser = ContentParserAnalyzer(sample_text_code_parser)
    result = (parser
        .extract_json_block()
        .detect_code_language()
        .get())
    print("\n=== JSON Code Extraction3 ===")
    print("Extracted JSON Code:", result)
    print("Language Detection json:", parser.get_counts())
    
    parser = ContentParserAnalyzer(sample_text_code_parser)
    result = (parser
        .reset()
        .extract_code_from_any_language()
        .get())
    print("\n=== All Code Blocks ===")
    print("All Code Blocks:", result)
    print("Code Block Count:", parser.get_counts())

if __name__ == "__main__":
    #generic test
    test_code_analysis()
    test_text_processing()
    test_nlp_analysis()
    test_metadata_extraction()
    test_logical_analysis()
    test_full_pipeline()
    
    #code parser test
    test_extract_code_blocks()
    # test_detect_code_language()
    # test_remove_code_blocks()
    # test_extract_text_without_code()
    # test_full_code_parser_pipeline()
