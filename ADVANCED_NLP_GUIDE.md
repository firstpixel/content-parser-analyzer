# Advanced NLP Analyzer - Complete Guide

## Overview

The **Advanced NLP Analyzer** is a powerful text-to-numerical-data transformation system that implements temporal orthogonal functions for comprehensive text analysis. It extracts **44+ independent parameters** across the **RCGE-PAVU framework** and supports **multi-scale temporal windowing** for tracking how text characteristics evolve.

## Key Capabilities

### 1. Temporal Orthogonal Functions
- **Multi-scale windowing**: Analyzes text at 100, 200, 300, and 600 token windows plus full text
- **Temporal trends**: Tracks how parameters change across windows (increasing, decreasing, volatility)
- **Sliding windows**: 50% overlap for continuous analysis
- **Complete coverage**: Every part of the text is analyzed multiple times at different scales

### 2. RCGE-PAVU Framework (40+ Parameters)

All parameters are normalized to [0, 1] range for easy comparison and validation.

#### R - Reasoning/Logic Structure (5 parameters)
1. **Logical Coherence** - Internal consistency via sentence similarity
2. **Causal Density** - Number of cause-effect relationships
3. **Argumentation Entropy** - Balance of claims vs evidence
4. **Contradiction Ratio** - Percentage of internal contradictions
5. **Inferential Depth** - Average reasoning complexity

#### C - Constraints/Context Integrity (5 parameters)
1. **Domain Consistency** - Vocabulary coherence within topic
2. **Referential Stability** - Entity persistence tracking
3. **Temporal Consistency** - Verb tense coherence
4. **Modality Balance** - Fact vs possibility statements
5. **Precision Index** - Specificity vs ambiguity

#### G - Goals/Intent & Direction (5 parameters)
1. **Goal Clarity** - Clarity of stated intent
2. **Focus Retention** - Topic drift measurement
3. **Persuasiveness** - Rhetorical strength
4. **Commitment** - Modal certainty level
5. **Teleology** - Purpose-driven language

#### E - Emotion/Expressive Content (5 parameters)
1. **Emotional Valence** - Positive/negative emotion
2. **Arousal** - Emotional intensity
3. **Empathy Score** - Perspective-taking tone
4. **Emotional Volatility** - Sentiment change rate
5. **Symbolic Resonance** - Metaphor density

#### P - Pragmatic/Contextual Use (5 parameters)
1. **Speech Act Ratio** - Distribution of assertive/directive/expressive acts
2. **Dialogue Coherence** - Question-answer quality
3. **Pragmatic Truth** - Informativeness vs filler
4. **Social Tone** - Politeness, dominance, cooperation
5. **Engagement Index** - Direct audience addressing

#### A - Aesthetic/Stylistic (5 parameters)
1. **Rhythm Variance** - Sentence length variation
2. **Lexical Diversity** - Type-token ratio
3. **Imagery Density** - Descriptive richness
4. **Symmetry Index** - Structural balance
5. **Surprise/Novelty** - Information gain

#### V - Veracity/Factual Dimension (5 parameters)
1. **Factual Density** - Claims per sentence
2. **Fact Precision** - Specificity of claims
3. **Evidence Linkage** - Citation presence
4. **Truth Confidence** - Verification potential
5. **Source Diversity** - Unique source count

#### U - Uncertainty/Ambiguity (5 parameters)
1. **Ambiguity Entropy** - Word sense ambiguity
2. **Vagueness** - Fuzzy quantifier usage
3. **Cognitive Dissonance** - Sentiment/logic mismatch
4. **Hypothetical Load** - Counterfactual statements
5. **Certainty Oscillation** - Certainty variance

### 3. Advanced NLP Features (4 features)

1. **Named Entity Recognition (NER)**
   - Extracts entities by type (PERSON, ORG, GPE, DATE, etc.)
   - Provides entity counts and density metrics
   - Compatible with spaCy models

2. **Relationship Extraction**
   - Extracts subject-verb-object triples
   - Counts relationships per sentence
   - Enables relationship graphs

3. **Word-Sense Disambiguation**
   - Analyzes word specificity using context
   - Measures ambiguity levels
   - Uses POS tagging and dependencies

4. **Information Extraction**
   - Extracts dates, numbers, emails, URLs
   - Structured data extraction
   - Entity-specific information

## Usage Examples

### Basic Usage

```python
from utils.content_parser_analyzer import ContentParserAnalyzer

text = """
Your text here...
"""

parser = ContentParserAnalyzer(text)

# Run complete analysis
results = parser.advanced_nlp_analyzer.run_complete_analysis()

# Access full text analysis
full = results['full_text_analysis']
print(f"Logical Coherence: {full['logical_coherence']:.3f}")
print(f"Emotional Valence: {full['emotional_valence']:.3f}")
```

### Accessing Individual Parameters

```python
analyzer = parser.advanced_nlp_analyzer

# Reasoning parameters
coherence = analyzer.analyze_logical_coherence(text)
causal = analyzer.analyze_causal_density(text)

# Emotion parameters
valence = analyzer.analyze_emotional_valence(text)
arousal = analyzer.analyze_arousal(text)

# Veracity parameters
factual = analyzer.analyze_factual_density(text)
precision = analyzer.analyze_fact_precision(text)
```

### Temporal Window Analysis

```python
# Run temporal analysis
results = parser.advanced_nlp_analyzer.analyze_temporal_windows()

# Access window analyses
for window_key, windows in results['window_analyses'].items():
    print(f"{window_key}: {len(windows)} windows")
    for window in windows:
        print(f"  Coherence: {window['logical_coherence']:.3f}")

# Access temporal trends
for window_key, trends in results['temporal_trends'].items():
    valence_trend = trends['emotional_valence']
    print(f"Valence: mean={valence_trend['mean']:.3f}, trend={valence_trend['trend']}")
```

### JSON Export

```python
import json

results = parser.advanced_nlp_analyzer.run_complete_analysis()

# Export to JSON
json_output = json.dumps(results, indent=2, default=str)

# Save to file
with open('analysis.json', 'w') as f:
    f.write(json_output)
```

## Output Structure

```python
{
    'full_text_analysis': {
        'window_id': 'full',
        'logical_coherence': 0.xxx,
        'causal_density': 0.xxx,
        # ... all 44+ parameters
        'named_entities': {...},
        'relationships': {...},
        'word_sense_disambiguation': {...},
        'information_extraction': {...}
    },
    'window_analyses': {
        'window_100': [...],  # List of window results
        'window_200': [...],
        'window_300': [...],
        'window_600': [...]
    },
    'temporal_trends': {
        'window_100': {
            'logical_coherence': {
                'mean': 0.xxx,
                'std': 0.xxx,
                'min': 0.xxx,
                'max': 0.xxx,
                'trend': 'increasing' | 'decreasing',
                'volatility': 0.xxx
            },
            # ... for all parameters
        },
        # ... for all window sizes
    },
    'metadata': {
        'total_tokens': xxx,
        'window_sizes': [100, 200, 300, 600],
        'total_windows': {...}
    }
}
```

## Use Cases

### 1. Text Quality Analysis
Compare multiple texts by their orthogonal parameters to identify:
- Most coherent text (logical_coherence)
- Most persuasive text (persuasiveness)
- Most factual text (factual_density)
- Most engaging text (engagement_index)

### 2. AI-Generated Content Detection
Analyze patterns in AI vs human text:
- Temporal consistency (more uniform in AI)
- Emotional volatility (less in AI)
- Lexical diversity (often different patterns)
- Relationship density (structural differences)

### 3. Content Moderation
Identify problematic content:
- High contradiction_ratio (misleading)
- Low truth_confidence (unreliable)
- High cognitive_dissonance (manipulative)
- Low pragmatic_truth (spam/filler)

### 4. Writing Quality Assessment
Evaluate writing across dimensions:
- Clarity (precision_index, goal_clarity)
- Style (lexical_diversity, rhythm_variance)
- Logic (logical_coherence, inferential_depth)
- Engagement (engagement_index, empathy_score)

### 5. Temporal Analysis
Track how text evolves:
- Introduction vs conclusion differences
- Topic drift detection
- Emotional arc tracking
- Argumentation structure evolution

## Technical Details

### Dependencies
- spaCy (for NER and parsing)
- NLTK (for tokenization)
- scikit-learn (for TF-IDF and similarity)
- TextBlob (for sentiment)
- NumPy (for numerical operations)

### Performance
- Full analysis: ~1-2 seconds for 1000 words
- Window analysis: Scales linearly with text length
- Memory efficient: Processes windows incrementally

### Limitations
- Requires spaCy model for some features (graceful fallback)
- English-optimized (can be extended to other languages)
- Best for texts > 50 tokens (shorter texts have limited windows)

## Advanced Features

### Custom Window Sizes
```python
analyzer = parser.advanced_nlp_analyzer
analyzer.window_sizes = [50, 150, 250, 500]  # Custom sizes
results = analyzer.analyze_temporal_windows()
```

### Parameter Filtering
```python
# Extract only specific parameter families
full = results['full_text_analysis']
reasoning_params = {k: v for k, v in full.items() 
                   if k in ['logical_coherence', 'causal_density', 
                           'argumentation_entropy', 'contradiction_ratio', 
                           'inferential_depth']}
```

### Comparative Analysis
```python
# Compare two texts
text1_results = analyzer1.run_complete_analysis()
text2_results = analyzer2.run_complete_analysis()

# Calculate differences
for param in text1_results['full_text_analysis']:
    if isinstance(text1_results['full_text_analysis'][param], (int, float)):
        diff = text2_results['full_text_analysis'][param] - text1_results['full_text_analysis'][param]
        print(f"{param}: {diff:+.3f}")
```

## Best Practices

1. **Use full analysis for comprehensive insights**
   ```python
   results = parser.advanced_nlp_analyzer.run_complete_analysis()
   ```

2. **Focus on relevant parameters for your use case**
   - Quality assessment: Reasoning + Constraints + Veracity
   - Engagement: Emotion + Pragmatic + Aesthetic
   - Credibility: Veracity + Uncertainty + Reasoning

3. **Compare across windows for temporal patterns**
   - Look for trends (increasing/decreasing)
   - Identify volatility (unstable writing)
   - Find peaks (climactic moments)

4. **Export to JSON for reproducibility**
   - All numerical data
   - Easy comparison
   - Version control friendly

5. **Normalize before comparison**
   - All parameters already in [0, 1]
   - Can directly compare across texts
   - Can aggregate into composite scores

## Validation

All parameters are designed to be:
- **Orthogonal**: Independent dimensions
- **Normalized**: [0, 1] range for comparability
- **Numerical**: Quantifiable and reproducible
- **Interpretable**: Clear semantic meaning
- **Validatable**: Can be verified against ground truth

## Examples

See the following files for complete examples:
- `example_advanced_nlp.py` - Comprehensive demonstration
- `test_advanced_nlp_analyzer.py` - Unit tests for each parameter
- `test_integration_advanced.py` - Integration tests

## Support

For issues, questions, or contributions, see the main README.md file.
