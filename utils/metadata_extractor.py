import re
import json
from bs4 import BeautifulSoup
from typing import Union, List, Dict, Any
from collections import Counter
from utils.base_parser import BaseParser

class MetadataExtractor(BaseParser):
    def __init__(self, text: str):
        super().__init__(text)
        # Keep the original text as provided
        self.original_text = text

    def get(self) -> Dict[str, Any]:
        """Retrieve extracted metadata in a structured format."""
        return self.metadata if self.metadata else {}

    def extract_urls(self):
        """Extract URLs from the working text, store them in metadata, and update extraction state."""
        urls = re.findall(r"https?://[a-zA-Z0-9.-]+(?:/[^\s]*)?", self.text)
        if urls:
            self.metadata['urls'] = urls
            self.counts['urls_found'] = len(urls)
            # Join the found URLs into a single string to update extraction state
            extracted = "\n".join(urls)
            self._update_extraction_result(extracted)
        return self

    def extract_emails(self):
        """Extract email addresses from the working text, store them in metadata, and update extraction state."""
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", self.text)
        if emails:
            self.metadata['emails'] = emails
            self.counts['emails_found'] = len(emails)
            extracted = "\n".join(emails)
            self._update_extraction_result(extracted)
        return self

    def extract_dates(self):
        """Extract dates from the working text, store them in metadata, and update extraction state."""
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", self.text)
        if dates:
            self.metadata['dates'] = dates
            self.counts['dates_found'] = len(dates)
            extracted = "\n".join(dates)
            self._update_extraction_result(extracted)
        return self

    def extract_keywords(self):
        """Extract keywords from the working text, store them in metadata, and update extraction state."""
        words = re.findall(r"\b\w+\b", self.text)
        keywords = Counter(words).most_common(5)
        if keywords:
            # Format keywords as a string for extraction update (e.g., join pairs)
            formatted = ", ".join(f"{word}:{count}" for word, count in keywords)
            self.metadata['keywords'] = keywords
            self.counts['keywords_found'] = len(keywords)
            self._update_extraction_result(formatted)
        return self

    def extract_hyperlinks(self):
        """Extract hyperlinks from HTML text in the working state, store them in metadata, and update extraction state."""
        soup = BeautifulSoup(self.text, "html.parser")
        hyperlinks = [a["href"] for a in soup.find_all("a", href=True)]
        if hyperlinks:
            self.metadata['hyperlinks'] = hyperlinks
            self.counts['hyperlinks_found'] = len(hyperlinks)
            extracted = "\n".join(hyperlinks)
            self._update_extraction_result(extracted)
        return self

    def extract_numbers(self):
        """Extract numbers from the working text, store them in metadata, and update extraction state."""
        numbers = re.findall(r"\d+(?:\.\d+)?", self.text)
        if numbers:
            self.metadata['numbers'] = numbers
            self.counts['numbers_found'] = len(numbers)
            extracted = " ".join(numbers)
            self._update_extraction_result(extracted)
        return self

    def extract_key_value_pairs(self):
        """Extract key-value pairs from the working text, store them in metadata, and update extraction state."""
        pairs = re.findall(r"(\w+):\s*([^\n]+)", self.text)
        if pairs:
            self.metadata['key_value_pairs'] = dict(pairs)
            self.counts['key_value_pairs_found'] = len(pairs)
            # Format key-value pairs as "key:value" strings
            formatted = ", ".join(f"{k}:{v}" for k, v in pairs)
            self._update_extraction_result(formatted)
        return self

    def extract_markdown_headings(self):
        """Extract markdown headings from the working text, store them in metadata, and update extraction state."""
        markdown_headings = re.findall(r"#{1,6}\s+(.+)", self.text)
        if markdown_headings:
            self.metadata['markdown_headings'] = markdown_headings
            self.counts['markdown_headings_found'] = len(markdown_headings)
            extracted = "\n".join(markdown_headings)
            self._update_extraction_result(extracted)
        return self

    def extract_hashtags(self):
        """Extract hashtags from the working text, store them in metadata, and update extraction state."""
        hashtags = re.findall(r"#\w+", self.text)
        if hashtags:
            self.metadata['hashtags'] = hashtags
            self.counts['hashtags_found'] = len(hashtags)
            extracted = "\n".join(hashtags)
            self._update_extraction_result(extracted)
        return self

    def extract_html_metadata(self):
        """Extract HTML metadata from the working text, store it in metadata, and update extraction state."""
        soup = BeautifulSoup(self.text, "html.parser")
        html_metadata = {meta.get("name", meta.get("property")): meta.get("content")
                         for meta in soup.find_all("meta")}
        if html_metadata:
            self.metadata['html_metadata'] = html_metadata
            self.counts['html_metadata_found'] = len(html_metadata)
            # Convert dictionary to a string representation for extraction update
            extracted = ", ".join(f"{k}:{v}" for k, v in html_metadata.items())
            self._update_extraction_result(extracted)
        return self

    def extract_open_graph_metadata(self):
        """Extract Open Graph metadata from the working text, store it in metadata, and update extraction state."""
        soup = BeautifulSoup(self.text, "html.parser")
        open_graph_metadata = {meta.get("property"): meta.get("content")
                               for meta in soup.find_all("meta", property=re.compile(r"^og:"))}
        if open_graph_metadata:
            self.metadata['open_graph_metadata'] = open_graph_metadata
            self.counts['open_graph_metadata_found'] = len(open_graph_metadata)
            extracted = ", ".join(f"{k}:{v}" for k, v in open_graph_metadata.items())
            self._update_extraction_result(extracted)
        return self

    def extract_json_ld(self):
        """Extract JSON-LD metadata from the working text, store it in metadata, and update extraction state."""
        soup = BeautifulSoup(self.text, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")
        json_ld = []
        for script in scripts:
            try:
                json_ld.append(json.loads(script.string))
            except Exception:
                continue
        if json_ld:
            self.metadata['json_ld'] = json_ld
            self.counts['json_ld_found'] = len(json_ld)
            # Convert the list of JSON objects into a string representation
            extracted = " ".join(str(item) for item in json_ld)
            self._update_extraction_result(extracted)
        return self
