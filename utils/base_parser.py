from typing import Union, List, Dict, Any
import re
from bs4 import BeautifulSoup
import Levenshtein

class BaseParser:
    def __init__(self, text: str):
        # Original, unmodified text
        self.original_text = text
        # Working state that changes over time
        self.text = text
        # The text that has not yet been extracted
        self.remaining_text = text
        # Accumulates all extracted segments
        self.extracted_text = ""
        self.should_update_state = True

        self.counts = {}
        self.metadata = {}

    def get(self) -> Union[str, List[str], Dict[str, Any]]:
        """
        Retrieve the cumulative extracted content.
        This method now returns the extracted_text which accumulates all extraction results.
        """
        return self.extracted_text if self.extracted_text else ""

    def get_current_state(self) -> Union[str, List[str], Dict[str, Any]]:
        """
        Retrieve the current working state of text.
        Returns the current self.text in its structured format.
        """
        if isinstance(self.text, dict):
            return self.text
        elif self.text:
            return self.text
        else:
            return ""

    def get_remaining(self) -> str:
        """
        Retrieve the current remaining text (i.e. the text that has not yet been extracted).
        """
        return self.remaining_text

    def _update_extraction_result(self, extracted_segment: str):
        """
        Update the parser state after an extraction.
        
        - Appends the extracted_segment to the cumulative extracted_text.
        - Removes one occurrence of the extracted_segment from remaining_text.
        - Updates self.text (the working state) to the new remaining_text.
        """
        # Append the extracted segment to extracted_text (with a newline separator if needed)
        if self.extracted_text and self.should_update_state:
            self.extracted_text += "\n" + extracted_segment
        else:
            self.extracted_text = extracted_segment

        # Remove one occurrence of the extracted_segment from remaining_text
        self.remaining_text = self.remaining_text.replace(extracted_segment, "", 1)
        # Update working state to the remaining text
        self.text = self.extracted_text
        
    def _override_extracted_text(self, new_text: str):
        """
        Overrides extracted_text with new_text instead of appending.
        This is useful for operations like summarization where we modify extracted content.
        """
        self.extracted_text = new_text
        self.text = new_text  # Ensure self.text also reflects the override


    def reset(self):
        """
        Resets the working state to the original text.
        This method sets both self.text and self.remaining_text to the original_text.
        Note: extracted_text is preserved (accumulative).
        """
        self.should_update_state = True
        self.text = self.original_text
        self.remaining_text = self.original_text
        return self

    def reset_remaining(self):
        """
        Resets the working state to the current remaining_text.
        """
        self.should_update_state = True
        self.text = self.remaining_text
        return self

    def get_counts(self) -> Dict[str, Any]:
        """Get the current counts dictionary."""
        return self.counts

    def __getattr__(self, name):
        """Support method chaining for any method."""
        if hasattr(super(), name):
            method = getattr(super(), name)
            def wrapper(*args, **kwargs):
                result = method(*args, **kwargs)
                return self if result is None else result
            return wrapper
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def remove_special_characters(self):
        """Remove all special characters from text - common across parsers."""
        self.text = re.sub(r"[^a-zA-Z0-9\s]", "", self.text)
        return self

    def strip_html_tags(self):
        """Remove HTML tags from text - used by multiple parsers."""
        self.text = BeautifulSoup(self.text, "html.parser").get_text()
        return self

    def count_text_elements(self):
        """Count basic text elements - common functionality."""
        if isinstance(self.text, str):
            self.counts["word_count"] = len(self.text.split())
            self.counts["character_count"] = len(self.text)
        return self

    def chunk_text(self, chunk_size: int = 100):
        """Split text into chunks - used by multiple parsers."""
        if isinstance(self.text, str):
            self.text = [self.text[i:i+chunk_size] for i in range(0, len(self.text), chunk_size)]
        return self

    def calculate_levenshtein(self, other_text: str) -> int:
        """Optimized Levenshtein distance calculation using the Levenshtein library."""
        return Levenshtein.distance(self.text, other_text)
