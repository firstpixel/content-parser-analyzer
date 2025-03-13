import langdetect
from typing import List, Dict, Union, Any

try:
    from polyglot.detect import Detector
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False
    print("Warning: polyglot not available. Some language detection features will be limited.")

from utils.code_parser import CodeParser
from utils.text_processor import TextProcessor
from utils.nlp_analyzer import NLPAnalyzer
from utils.logical_analyzer import LogicalAnalyzer
from utils.metadata_extractor import MetadataExtractor

class ContentParserAnalyzer:
    """A powerful AI-driven content parser that combines multiple specialized parsers,
    maintaining three distinct states:
      - original_text: The full, unmodified text.
      - extracted_text: Cumulative content extracted over time.
      - remaining_text: The text left after extractions.
    """

    def __init__(self, text: str):
        self.original_text = text      # Unmodified full text
        self.text = text               # Working state: current text to process
        self.remaining_text = text     # Text that remains unextracted
        self.extracted_text = ""       # Accumulates all extraction results

        # Initialize specialized parsers with the current text
        self.code_parser = CodeParser(text)
        self.text_processor = TextProcessor(text)
        self.nlp_analyzer = NLPAnalyzer(text)
        self.logical_analyzer = LogicalAnalyzer(text)
        self.metadata_extractor = MetadataExtractor(text)
        self._sync_parsers()

    def _sync_parsers(self):
        """Synchronize the working state (text), remaining_text, and extracted_text across all sub-parsers."""
        parsers = [
            self.code_parser,
            self.text_processor,
            self.nlp_analyzer,
            self.logical_analyzer,
            self.metadata_extractor
        ]
        for parser in parsers:
            parser.text = self.text
            parser.remaining_text = self.remaining_text
            parser.extracted_text = self.extracted_text

    def reset(self):
        """
        Reset the working state to the original text.
        This sets both self.text and self.remaining_text to the original_text.
        (Note: The cumulative extracted_text is preserved.)
        """
        self.text = self.original_text
        self.remaining_text = self.original_text
        self._sync_parsers()
        return self

    def reset_remaining(self):
        """
        Reset the working state to the current remaining_text.
        """
        self.text = self.remaining_text
        self._sync_parsers()
        return self

    def get(self) -> Union[str, Dict[str, Any]]:
        """
        Retrieve the cumulative extracted content.
        """
        return self.extracted_text

    def get_current_state(self) -> Union[str, List[str], Dict[str, Any]]:
        """
        Retrieve the current working state (self.text).
        """
        return self.text

    def get_remaining(self) -> str:
        """
        Retrieve the text that remains after extractions.
        """
        return self.remaining_text

    def get_original(self) -> str:
        """
        Retrieve the original, unmodified text.
        """
        return self.original_text

    def get_counts(self) -> Dict[str, Any]:
        """
        Aggregate counts from all sub-parsers.
        """
        counts = {}
        for parser in [self.code_parser, self.text_processor, self.nlp_analyzer, self.logical_analyzer, self.metadata_extractor]:
            counts.update(parser.get_counts())
        return counts

    def get_inverse(self) -> str:
        """
        Retrieve the text that was removed during extractions.
        Computed by removing the current working text from the original.
        """
        return self.original_text.replace(self.text, "").strip()

    def detect_language(self):
        """Detect the primary language of text with fallback options."""
        try:
            if POLYGLOT_AVAILABLE:
                from polyglot.detect import Detector
                detector = Detector(self.text)
                self.counts["language"] = detector.language.code
            else:
                self.counts["language"] = langdetect.detect(self.text)
        except Exception as e:
            self.counts["language"] = "unknown"
            self.counts["language_error"] = str(e)
        return self

    def __getattr__(self, name):
        """
        Delegate method calls to the first sub-parser that defines the method.
        After the method call, update the overall state (text, remaining_text, extracted_text)
        and synchronize the sub-parsers.
        """
        for parser in [self.code_parser, self.text_processor, self.nlp_analyzer, self.logical_analyzer, self.metadata_extractor]:
            if hasattr(parser, name):
                method = getattr(parser, name)
                def wrapper(*args, **kwargs):
                    result = method(*args, **kwargs)
                    # Update overall state from the invoked parser
                    self.text = parser.text
                    self.remaining_text = parser.remaining_text
                    self.extracted_text = parser.extracted_text
                    self._sync_parsers()
                    return self
                return wrapper
        raise AttributeError(f"'ContentParser' has no attribute '{name}'")

# Main demonstration block for quick testing
if __name__ == "__main__":
    sample_text = """
    This is a sample string.
    
    It contains a Python code block:
    ```python
    print("Hello, world!")
    ```
    
    And a factual claim: studies show that exercise is beneficial.
    """
    parser = ContentParserAnalyzer(sample_text)
    result = (parser.extract_python_code_block()
                .normalize_text()
                .analyze_sentiment()
                .assign_credibility_score()
                .evaluate_coherence()  # New evaluation added
                .get())
    print("Cumulative Extracted Content:", result)
    print("Current Working State:", parser.get_current_state())
    print("Analysis Counts:", parser.get_counts())
