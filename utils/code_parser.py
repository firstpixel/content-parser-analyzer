import re
from typing import Dict, List, Union
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.util import ClassNotFound
from utils.base_parser import BaseParser

class CodeParser(BaseParser):
    """Parser class for handling code-related operations."""
    
    def _extract_code(self, language: str = None):
        """Helper method to extract code blocks with proper removal from remaining text."""
        pattern = r'```(\w*)\s*\n(.*?)\n\s*```'
        matches = list(re.finditer(pattern, self.text, re.DOTALL))
        
        code_blocks = {}
        
        for match in matches:
            original_block = match.group(0)  # Get complete original block
            lang = match.group(1).lower().strip()
            code = match.group(2).strip()
            
            if language:  # If specific language requested
                if lang == language.lower():
                    self._update_extraction_result(original_block)
                    code_blocks.setdefault(lang, []).append(code)
                    break  # Stop after first match for specific language
            else:  # If no specific language requested
                self._update_extraction_result(original_block)
                code_blocks.setdefault(lang or 'unknown', []).append(code)

        if code_blocks:
            self.metadata['code_blocks'] = code_blocks
            self.counts['code_blocks_by_language'] = {
                lang: len(blocks) for lang, blocks in code_blocks.items()
            }
            
        return self

    def extract_code_from_any_language(self):
        """Extract all code blocks regardless of language."""
        return self._extract_code()

    def detect_code_language(self):
        """
        Detect programming language in all code blocks or just extracted ones.
        Updates counts with detailed language statistics.
        """
        # If we have extracted text, analyze that first
        if self.extracted_text:
            text_to_analyze = self.extracted_text
        else:
            # Otherwise analyze all text
            text_to_analyze = self.text

        pattern = r'```(\w*)\s*\n(.*?)\n\s*```'
        matches = re.finditer(pattern, text_to_analyze, re.DOTALL)
        
        language_counts = {}
        code_blocks = []
        
        for match in matches:
            lang, code = match.group(1).lower().strip(), match.group(2).strip()
            if not lang:
                try:
                    lexer = guess_lexer(code)
                    lang = lexer.name.lower()
                except ClassNotFound:
                    lang = "unknown"
            
            language_counts[lang] = language_counts.get(lang, 0) + 1
            code_blocks.append(code)

        # Update counts with detailed information
        self.counts.update({
            "code_language": list(language_counts.keys()),
            "code_blocks_total": len(code_blocks),
            "code_blocks_by_language": language_counts
        })
        
        return self

    # Language-specific extractors
    def extract_python_code_block(self):
        """Extract Python code blocks."""
        return self._extract_code("python")

    def extract_json_block(self):
        """Extract JSON code blocks."""
        return self._extract_code("json")

    def extract_sql_code_block(self):
        """Extract SQL code blocks."""
        return self._extract_code("sql")

    def extract_shell_script_block(self):
        """Extract shell script blocks."""
        return self._extract_code("sh")

    def extract_css_block(self):
        """Extract CSS code blocks."""
        return self._extract_code("css")

    def extract_html_block(self):
        """Extract HTML code blocks."""
        return self._extract_code("html")

    def extract_yaml_block(self):
        """Extract YAML code blocks."""
        return self._extract_code("yaml")

    # Code block removal methods (non-extraction, so left unchanged)
    def remove_code_blocks(self):
        """Remove all code blocks from text."""
        self.text = re.sub(r"```.*?```", "", self.text, flags=re.DOTALL)
        return self

    def extract_text_without_code_blocks(self):
        """Extract text content excluding code blocks."""
        self.text = re.sub(r"```.*?```", "", self.text, flags=re.DOTALL)
        self.text = re.sub(r"`[^`]+`", "", self.text)  # Remove inline code
        return self

    def get_code_blocks(self) -> Dict[str, List[str]]:
        """Get all extracted code blocks organized by language."""
        return self.metadata.get('code_blocks', {})

    def get(self) -> Union[str, Dict[str, List[str]]]:
        """
        Retrieve either the extracted text or code blocks depending on context.
        Returns:
            - Dictionary of code blocks by language if available
            - Extracted text otherwise
        """
        if self.metadata.get('code_blocks'):
            return self.metadata['code_blocks']
        return self.extracted_text

    def strip_code_markers(self):
        """Remove markdown code block syntax from extracted text."""
        if self.extracted_text:
            # Remove ```language and ``` markers
            pattern = r'```\w*\s*\n(.*?)\n\s*```'
            matches = re.finditer(pattern, self.extracted_text, re.DOTALL)
            stripped_blocks = []
            
            for match in matches:
                code = match.group(1).strip()
                stripped_blocks.append(code)
            
            if stripped_blocks:
                self._override_extracted_text('\n\n'.join(stripped_blocks))
        
        return self
