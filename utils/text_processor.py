import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from utils.base_parser import BaseParser

class TextProcessor(BaseParser):
    def normalize_text(self):
        """Normalize text by first expanding contractions, then converting to lowercase."""
        self.expand_contractions()  
        self.text = self.text.strip().lower()
        self._override_extracted_text(self.text)
        return self
    def split_sentences(self):
        """Split text into sentences."""
        self.text = sent_tokenize(self.text)
        self._override_extracted_text(self.text)
        return self
    
    def split_words(self):
        """Split text into words."""
        self.text = word_tokenize(self.text)
        self._override_extracted_text(self.text)
        return self

    def remove_stopwords(self):
        """Remove common stopwords from text."""
        stop_words = set(stopwords.words("english"))
        
        if isinstance(self.text, list):  
            self.text = [word for word in self.text if word.lower() not in stop_words]
        elif isinstance(self.text, str):  
            self.text = " ".join(word for word in self.text.split() if word.lower() not in stop_words)
        self._override_extracted_text(self.text)
        return self
    
    def extract_plain_text(self):
        """
        Extracts plain text by removing:
        - Markdown formatting
        - Code blocks
        - HTML tags
        - Special characters
        - Extra whitespace
        """
        # Remove markdown-style code blocks
        self.text = re.sub(r"```.*?\n(.*?)\n```", "", self.text, flags=re.DOTALL)
        self.text = re.sub(r"```.*?```", "", self.text, flags=re.DOTALL)
        # Remove inline Markdown formatting (e.g., **bold**, *italic*, `inline code`)
        self.text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", self.text)
        self.text = re.sub(r"`(.*?)`", r"\1", self.text)
        
        # Remove HTML tags
        self.text = BeautifulSoup(self.text, "html.parser").get_text(separator=" ")
        
        # Remove non-alphanumeric characters (except spaces)
        self.text = re.sub(r"[^a-zA-Z0-9\s]", "", self.text)
        
        # Normalize spaces
        self.text = " ".join(self.text.split())
        self._override_extracted_text(self.text)
        return self

    def stem_text(self):
        """Apply word stemming to text."""
        ps = PorterStemmer()
        self.text = " ".join(ps.stem(word) for word in self.text.split())
        self._override_extracted_text(self.text)
        return self

    def lemmatize_text(self):
        """Apply word lemmatization to text."""
        lemmatizer = WordNetLemmatizer()
        self.text = " ".join(lemmatizer.lemmatize(word) for word in self.text.split())
        self._override_extracted_text(self.text)
        return self

    def expand_contractions(self):
        """Expand common contractions in text while preserving punctuation."""
        contractions = {
            "can't": "cannot", "won't": "will not", "i'm": "i am",
            "here's": "here is", "it's": "it is", "you're": "you are",
            "that's": "that is", "didn't": "did not", "doesn't": "does not"
        }

        # Use regex to match contractions more accurately
        pattern = re.compile(r"\b(" + "|".join(contractions.keys()) + r")\b", re.IGNORECASE)

        self.text = pattern.sub(lambda match: contractions[match.group(0).lower()], self.text)
        self._override_extracted_text(self.text)
        return self

    def categorize_response_style(self):
        """Categorize the style of the response."""
        self.counts["response_style"] = "formal"  # Placeholder implementation
        return self

    def compare_with_expert(self, expert_text: str):
        """Compare text similarity with expert text."""
        self.counts["expert_similarity"] = (
            SequenceMatcher(None, self.text, expert_text).ratio() 
            if isinstance(self.text, str) else 0.0
        )
        return self

    def summarize_text(self, max_length: int = 100):
        """Summarizes text and updates it properly."""
        if len(self.text) > max_length:
            summarized = self.text[:max_length - 3] + "..."  # Ensure total length does not exceed max_length
        else:
            summarized = self.text  

        self._override_extracted_text(summarized)
        return self


    def measure_response_depth(self):
        """Measure the depth/complexity of the response."""
        self.counts["response_depth"] = len(self.text.split())
        return self

    def analyze_tone_formality(self):
        """Analyze the formality of the text tone."""
        informal_words = {"gonna", "wanna", "kinda"}
        self.counts["tone_formality"] = (
            "informal" if any(word in self.text.split() for word in informal_words) else "formal"
        )
        return self
