# ContentParserAnalyzer: The Ultimate Tool for LLM Input & Output Analysis! 🚀
The ContentParserAnalyzer is a powerful yet simple-to-use utility designed to analyze, refine, and manipulate both inputs and outputs of large language models (LLMs). Whether you're cleaning raw text, extracting code, detecting sentiment, or validating logical consistency, this tool ensures structured, high-quality processing for any AI-driven application. Its modular design allows you to chain operations seamlessly, making it perfect for pre-processing prompts, post-processing responses, and enhancing AI-generated content. With built-in bias detection, factual validation, metadata extraction, and sentiment analysis, it empowers you to optimize and control the behavior of LLMs effortlessly. Enhance, filter, and structure your AI interactions with this next-generation LLM optimization toolkit! 🚀🔥

## Key Features

### 1. Text Processing
- Normalization and cleaning
- Sentence and word tokenization
- Stopword removal
- Text summarization
- Stemming and lemmatization
- Contraction expansion
- Plain text extraction from markdown/HTML

### 2. Code Analysis
- Multi-language code block extraction (Python, JSON, SQL, Shell, CSS, HTML, YAML)
- Language detection
- Code block removal
- Syntax validation

### 3. NLP Analysis
- Sentiment analysis
- Language detection
- Emotional tone detection
- Sarcasm and humor detection
- Vagueness analysis
- Hedging detection
- Bias detection
- Harmful content detection
- Policy violation checks
- Named entity recognition
- Credibility scoring
- Hallucination detection

### 4. Logical Analysis
- Logical flow analysis
- Contradiction detection
- Fallacy detection
- Factual claim identification
- Coherence evaluation
- Argument extraction
- Socratic method detection
- Missing perspective identification
- Irrelevant content detection
- Misinterpretation analysis

### 5. Metadata Extraction
- URLs
- Emails
- Dates
- Keywords
- Hyperlinks
- Numbers
- Key-value pairs
- Markdown headings
- Hashtags
- HTML metadata
- Open Graph metadata
- JSON-LD

## Available methods

### Methods in ContentParserAnalyzer

---

***reset()*** - Resets text to the original state while preserving extracted content.

***reset_remaining()*** - Resets text to the remaining unprocessed text.

***get()*** - Retrieves the cumulative extracted content.

***get_current_state()*** - Retrieves the current working text state.

***get_remaining()*** - Retrieves the remaining unprocessed text.

***get_original()*** - Retrieves the original, unmodified text.

***get_counts()*** - Aggregates data counts from all sub-parsers.

***detect_language()*** - Detects the primary language of the text.


<br><br>

### Methods in BaseParser

---

***remove_special_characters()*** - Removes all special characters from the text.

***strip_html_tags()*** - Removes HTML tags from the text.

***count_text_elements()*** - Counts words and characters in the text.

***chunk_text(chunk_size: int = 100)*** - Splits the text into chunks of a specified size.

***calculate_levenshtein(other_text: str)*** - Computes the Levenshtein distance between two texts.

<br><br>

### Methods in CodeParser

---

***_extract_code(language: str = None)*** - Extracts code blocks, optionally filtering by language.

***extract_code_from_any_language()*** - Extracts all code blocks regardless of language.

***detect_code_language()*** - Detects programming languages in extracted code blocks.

***extract_python_code_block()*** - Extracts Python code blocks.

***extract_json_block()*** - Extracts JSON code blocks.

***extract_sql_code_block()*** - Extracts SQL code blocks.

***extract_shell_script_block()*** - Extracts shell script blocks.

***extract_css_block()*** - Extracts CSS code blocks.

***extract_html_block()*** - Extracts HTML code blocks.

***extract_yaml_block()*** - Extracts YAML code blocks.

***remove_code_blocks()*** - Removes all code blocks from the text.

***extract_text_without_code_blocks()*** - Extracts plain text, removing code blocks.

***get_code_blocks()*** - Retrieves extracted code blocks organized by language.

***get()*** - Returns either extracted text or code blocks, depending on context.

***strip_code_markers()*** - Removes markdown syntax markers from extracted code.

<br><br>

### Methods in TextProcessor

---

***normalize_text()*** - Expands contractions and converts text to lowercase.

***split_sentences()*** - Splits text into sentences.

***split_words()*** - Splits text into words.

***remove_stopwords()*** - Removes common stopwords from text.

***extract_plain_text()*** - Removes markdown, code blocks, HTML tags, special characters, and extra whitespace.

***stem_text()*** - Applies stemming to reduce words to their root form.

***lemmatize_text()*** - Applies lemmatization for word normalization.

***expand_contractions()*** - Expands common contractions while maintaining punctuation.

***categorize_response_style()*** – Categorizes text as formal or informal.

***compare_with_expert(expert_text: str)*** – Computes text similarity with expert-written content.

***summarize_text(max_length: int = 100)*** – Summarizes text to a maximum length.

***measure_response_depth()*** – Measures the complexity of the text.

***analyze_tone_formality()*** – Analyzes the formality of the text tone.


<br><br>

### Methods in LogicalAnalyzer

---

***analyze_logical_flow()*** – Evaluates logical flow using transition words and sentence structure.

***detect_contradictions()*** – Identifies contradictions within the text.

***detect_fallacies()*** – Detects common logical fallacies such as ad hominem or false dichotomy.

***detect_factual_claims()*** – Identifies factual claims in the text.

***evaluate_coherence()*** – Measures coherence by analyzing sentence relationships and consistency.

***extract_premises_conclusion()*** – Extracts logical premises and conclusions from the text.

***extract_arguments()*** – Identifies arguments along with supporting evidence.

***detect_socratic_method()*** – Detects questioning techniques used in the Socratic method.

***detect_missing_perspectives()*** – Identifies missing viewpoints in discussions.

***detect_irrelevant_content()*** – Finds content that is unrelated to the main topic.

***detect_misinterpretation()*** – Detects potential misinterpretations or ambiguities.

***measure_prompt_coverage()*** – Assesses how well the text covers expected key points.


### Methods in MetadataExtractor

---

***extract_urls()*** – Extracts URLs from text and stores them in metadata.

***extract_emails()*** – Extracts email addresses from text and updates metadata.

***extract_dates()*** – Extracts date patterns (YYYY-MM-DD) from text.

***extract_keywords()*** – Extracts the most frequent words as keywords.

***extract_hyperlinks()*** – Extracts hyperlinks from HTML content.

***extract_numbers()*** – Extracts numerical values from text.

***extract_key_value_pairs()*** – Extracts key-value pairs in the format key: value.

***extract_markdown_headings()*** – Extracts headings from Markdown-formatted text.

***extract_hashtags()*** – Extracts hashtags (e.g., #example) from text.

***extract_html_metadata()*** – Extracts metadata from HTML <meta> tags.

***extract_open_graph_metadata()*** – Extracts Open Graph metadata from HTML.

***extract_json_ld()*** – Extracts JSON-LD structured data from HTML.


### Methods in NLPAnalyzer

---

***analyze_sentiment()*** – Analyzes sentiment polarity (positive, neutral, or negative).

***detect_language()*** – Detects the primary language of the text.

***detect_emotional_tone()*** – Identifies the dominant emotion in the text.

***detect_sarcasm_humor()*** – Detects sarcasm and humor based on sentiment contrast and keywords.

***detect_vagueness()*** – Identifies vague or imprecise language.

***detect_hedging()*** – Detects hedging language that reduces commitment.

***detect_bias()*** – Detects biased language and statements.

***detect_harmful_content()*** – Identifies potentially harmful content based on keyword heuristics.

***detect_policy_violations()*** – Detects text that may violate content policies.

***extract_named_entities()*** – Extracts named entities (e.g., people, places, organizations).

***assign_credibility_score()*** – Assigns a credibility score based on bias and factual claims.

***detect_hallucinations()*** – Detects inconsistencies and unsupported claims in the text.


## Installation

### Prerequisites

### macOS (using Homebrew)
```bash
# Install system dependencies
brew install python3
brew install icu4c
brew install cld2
brew install libtool

# For polyglot language support
brew install libicu
```

### Windows
1. Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. Download and install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. For ICU support (needed by polyglot):
   - Download the latest ICU binaries from [icu.unicode.org](https://icu.unicode.org/download)
   - Add the ICU bin directory to your system PATH

## Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

## Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download required spaCy model
python -m spacy download en_core_web_sm
```

## Verification

```python
# Test the installation
python -c "from utils.content_parser_analyzer import ContentParserAnalyzer; print('Setup successful!')"
```

## Troubleshooting

If you encounter issues with polyglot installation:
- macOS: Make sure ICU is installed correctly: `brew install icu4c`
- Windows: Verify Visual C++ Build Tools are installed and ICU is in your PATH

For other dependencies:
- Make sure your virtual environment is activated
- Try installing problematic packages individually
- Check system requirements for your OS version

## Usage Examples

### Basic Usage

```python
from utils.content_parser_analyzer import ContentParserAnalyzer

# Initialize parser with text
text = """
Here's a code example from https://example.com:

```python
def greet(name):
    return f"Hello, {name}!"
```

Contact us at support@example.com
"""

parser = ContentParserAnalyzer(text)
```

### Code Extraction and Analysis

```python
# Extract and analyze code blocks
result = (parser
    .extract_python_code_block()
    .detect_code_language()
    .get())

print("Extracted Code:", result)
print("Analysis Details:", parser.get_counts())
```

### Text Processing and Analysis

```python
# Process and analyze text
result = (parser
    .normalize_text()
    .remove_stopwords()
    .analyze_sentiment()
    .detect_emotional_tone()
    .get())

print("Processed Text:", result)
print("Analysis Details:", parser.get_counts())
```

### Metadata Extraction

```python
# Extract metadata
parser.extract_urls().reset_remaining().extract_emails()
metadata = parser.metadata_extractor.get()

print("URLs:", metadata.get("urls", []))
print("Emails:", metadata.get("emails", []))
```

### Logical Analysis

```python
# Analyze logical structure
result = (parser
    .detect_fallacies()
    .evaluate_coherence()
    .detect_contradictions()
    .get())

print("Logical Analysis:", parser.get_counts())
```

### Full Pipeline Example

```python
# Complex analysis pipeline
result = (parser
    .extract_python_code_block()      # Extract code blocks
    .detect_code_language()           # Detect programming language
    .reset_remaining()                # Reset to work with remaining text
    .remove_code_blocks()             # Remove any remaining code blocks
    .normalize_text()                 # Normalize the text
    .analyze_sentiment()              # Analyze sentiment
    .detect_harmful_content()         # Check for harmful content
    .detect_hallucinations()          # Check for AI hallucinations
    .extract_named_entities()         # Extract named entities
    .get())

print("Final Output:", result)
print("Analysis Results:", parser.get_counts())
```

## State Management

The ContentParserAnalyzer maintains three distinct states:
- `original_text`: The unmodified input text
- `extracted_text`: Cumulative content extracted over time
- `remaining_text`: Text that remains after extractions

Use these methods to manage state:
- `reset()`: Reset to original text
- `reset_remaining()`: Reset to current remaining text
- `get()`: Get current extracted content
- `get_current_state()`: Get current working state
- `get_remaining()`: Get remaining unprocessed text
- `get_counts()`: Get analysis results and counts

## Use Cases

1. **LLM Input Preprocessing**
   - Clean and normalize user inputs
   - Extract relevant components
   - Validate content safety
   - Check for harmful/biased content

2. **LLM Response Analysis**
   - Validate output quality
   - Detect hallucinations
   - Check coherence and logic
   - Extract structured data
   - Analyze sentiment and tone

3. **Content Moderation**
   - Detect harmful content
   - Check policy violations
   - Analyze bias and vagueness
   - Validate credibility

4. **Code Analysis**
   - Extract code snippets
   - Detect programming languages
   - Separate code from text
   - Analyze code context

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
