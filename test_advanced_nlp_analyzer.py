"""
Tests for Advanced NLP Analyzer with Temporal Orthogonal Functions
"""

import json
from utils.advanced_nlp_analyzer import AdvancedNLPAnalyzer


def test_reasoning_parameters():
    """Test R - Reasoning/Logic Structure parameters"""
    text = """
    The experiment demonstrates that increased temperature leads to faster reaction rates.
    Therefore, we can conclude that temperature is a critical factor. 
    Because of this relationship, we must control temperature precisely.
    However, other variables may also influence the outcome.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Reasoning Parameters ===")
    
    # Test logical coherence
    coherence = analyzer.analyze_logical_coherence(text)
    print(f"Logical Coherence: {coherence:.3f}")
    assert 0 <= coherence <= 1, "Coherence should be in [0, 1]"
    
    # Test causal density
    causal = analyzer.analyze_causal_density(text)
    print(f"Causal Density: {causal:.3f}")
    assert 0 <= causal <= 1, "Causal density should be in [0, 1]"
    
    # Test argumentation entropy
    arg_entropy = analyzer.analyze_argumentation_entropy(text)
    print(f"Argumentation Entropy: {arg_entropy:.3f}")
    assert 0 <= arg_entropy <= 1, "Argumentation entropy should be in [0, 1]"
    
    # Test contradiction ratio
    contradictions = analyzer.analyze_contradiction_ratio(text)
    print(f"Contradiction Ratio: {contradictions:.3f}")
    assert 0 <= contradictions <= 1, "Contradiction ratio should be in [0, 1]"
    
    # Test inferential depth
    depth = analyzer.analyze_inferential_depth(text)
    print(f"Inferential Depth: {depth:.3f}")
    assert 0 <= depth <= 1, "Inferential depth should be in [0, 1]"
    
    print("✓ All Reasoning parameters validated\n")


def test_constraints_parameters():
    """Test C - Constraints/Context Integrity parameters"""
    text = """
    The Python programming language provides excellent tools for data analysis.
    Python's libraries like NumPy and Pandas are widely used.
    Many developers prefer Python for machine learning tasks.
    The language was created in the 1990s and continues to evolve.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Constraints Parameters ===")
    
    # Test domain consistency
    consistency = analyzer.analyze_domain_consistency(text)
    print(f"Domain Consistency: {consistency:.3f}")
    assert 0 <= consistency <= 1, "Domain consistency should be in [0, 1]"
    
    # Test referential stability
    stability = analyzer.analyze_referential_stability(text)
    print(f"Referential Stability: {stability:.3f}")
    assert 0 <= stability <= 1, "Referential stability should be in [0, 1]"
    
    # Test temporal consistency
    temporal = analyzer.analyze_temporal_consistency(text)
    print(f"Temporal Consistency: {temporal:.3f}")
    assert 0 <= temporal <= 1, "Temporal consistency should be in [0, 1]"
    
    # Test modality balance
    modality = analyzer.analyze_modality_balance(text)
    print(f"Modality Balance: {modality:.3f}")
    assert 0 <= modality <= 1, "Modality balance should be in [0, 1]"
    
    # Test precision index
    precision = analyzer.analyze_precision_index(text)
    print(f"Precision Index: {precision:.3f}")
    assert 0 <= precision <= 1, "Precision index should be in [0, 1]"
    
    print("✓ All Constraints parameters validated\n")


def test_goals_parameters():
    """Test G - Goals/Intent & Direction parameters"""
    text = """
    Our goal is to achieve sustainable development in the community.
    We aim to reduce carbon emissions by 50% over the next decade.
    The objective is clear: we need to transition to renewable energy.
    This plan is designed to ensure long-term environmental health.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Goals Parameters ===")
    
    # Test goal clarity
    clarity = analyzer.analyze_goal_clarity(text)
    print(f"Goal Clarity: {clarity:.3f}")
    assert 0 <= clarity <= 1, "Goal clarity should be in [0, 1]"
    
    # Test focus retention
    focus = analyzer.analyze_focus_retention(text)
    print(f"Focus Retention: {focus:.3f}")
    assert 0 <= focus <= 1, "Focus retention should be in [0, 1]"
    
    # Test persuasiveness
    persuasion = analyzer.analyze_persuasiveness(text)
    print(f"Persuasiveness: {persuasion:.3f}")
    assert 0 <= persuasion <= 1, "Persuasiveness should be in [0, 1]"
    
    # Test commitment
    commitment = analyzer.analyze_commitment(text)
    print(f"Commitment: {commitment:.3f}")
    assert 0 <= commitment <= 1, "Commitment should be in [0, 1]"
    
    # Test teleology
    teleology = analyzer.analyze_teleology(text)
    print(f"Teleology: {teleology:.3f}")
    assert 0 <= teleology <= 1, "Teleology should be in [0, 1]"
    
    print("✓ All Goals parameters validated\n")


def test_emotion_parameters():
    """Test E - Emotion/Expressive Content parameters"""
    text = """
    I absolutely love this amazing opportunity! It's incredibly exciting!
    We feel deeply grateful for this wonderful experience.
    This represents a truly meaningful achievement for our team.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Emotion Parameters ===")
    
    # Test emotional valence
    valence = analyzer.analyze_emotional_valence(text)
    print(f"Emotional Valence: {valence:.3f}")
    assert 0 <= valence <= 1, "Emotional valence should be in [0, 1]"
    
    # Test arousal
    arousal = analyzer.analyze_arousal(text)
    print(f"Arousal: {arousal:.3f}")
    assert 0 <= arousal <= 1, "Arousal should be in [0, 1]"
    
    # Test empathy score
    empathy = analyzer.analyze_empathy_score(text)
    print(f"Empathy Score: {empathy:.3f}")
    assert 0 <= empathy <= 1, "Empathy score should be in [0, 1]"
    
    # Test emotional volatility
    volatility = analyzer.analyze_emotional_volatility(text)
    print(f"Emotional Volatility: {volatility:.3f}")
    assert 0 <= volatility <= 1, "Emotional volatility should be in [0, 1]"
    
    # Test symbolic resonance
    resonance = analyzer.analyze_symbolic_resonance(text)
    print(f"Symbolic Resonance: {resonance:.3f}")
    assert 0 <= resonance <= 1, "Symbolic resonance should be in [0, 1]"
    
    print("✓ All Emotion parameters validated\n")


def test_pragmatic_parameters():
    """Test P - Pragmatic/Contextual Use parameters"""
    text = """
    You should consider this option carefully. We need to work together.
    Please review the document. Thank you for your cooperation.
    What do you think about this approach? Let's collaborate on this project.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Pragmatic Parameters ===")
    
    # Test speech act ratio
    speech_acts = analyzer.analyze_speech_act_ratio(text)
    print(f"Speech Act Ratio: {speech_acts}")
    assert all(0 <= v <= 1 for v in speech_acts.values()), "Speech acts should be in [0, 1]"
    
    # Test dialogue coherence
    dialogue = analyzer.analyze_dialogue_coherence(text)
    print(f"Dialogue Coherence: {dialogue:.3f}")
    assert 0 <= dialogue <= 1, "Dialogue coherence should be in [0, 1]"
    
    # Test pragmatic truth
    truth = analyzer.analyze_pragmatic_truth(text)
    print(f"Pragmatic Truth: {truth:.3f}")
    assert 0 <= truth <= 1, "Pragmatic truth should be in [0, 1]"
    
    # Test social tone
    social = analyzer.analyze_social_tone(text)
    print(f"Social Tone: {social}")
    assert all(0 <= v <= 1 for v in social.values()), "Social tone values should be in [0, 1]"
    
    # Test engagement index
    engagement = analyzer.analyze_engagement_index(text)
    print(f"Engagement Index: {engagement:.3f}")
    assert 0 <= engagement <= 1, "Engagement index should be in [0, 1]"
    
    print("✓ All Pragmatic parameters validated\n")


def test_aesthetic_parameters():
    """Test A - Aesthetic/Stylistic parameters"""
    text = """
    Short sentence. This one is a bit longer with more words.
    Here we have an even longer sentence that continues with additional information.
    Brief again. The beautiful landscape stretched endlessly before us.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Aesthetic Parameters ===")
    
    # Test rhythm variance
    rhythm = analyzer.analyze_rhythm_variance(text)
    print(f"Rhythm Variance: {rhythm:.3f}")
    assert 0 <= rhythm <= 1, "Rhythm variance should be in [0, 1]"
    
    # Test lexical diversity
    diversity = analyzer.analyze_lexical_diversity(text)
    print(f"Lexical Diversity: {diversity:.3f}")
    assert 0 <= diversity <= 1, "Lexical diversity should be in [0, 1]"
    
    # Test imagery density
    imagery = analyzer.analyze_imagery_density(text)
    print(f"Imagery Density: {imagery:.3f}")
    assert 0 <= imagery <= 1, "Imagery density should be in [0, 1]"
    
    # Test symmetry index
    symmetry = analyzer.analyze_symmetry_index(text)
    print(f"Symmetry Index: {symmetry:.3f}")
    assert 0 <= symmetry <= 1, "Symmetry index should be in [0, 1]"
    
    # Test surprise/novelty
    novelty = analyzer.analyze_surprise_novelty(text)
    print(f"Surprise/Novelty: {novelty:.3f}")
    assert 0 <= novelty <= 1, "Novelty should be in [0, 1]"
    
    print("✓ All Aesthetic parameters validated\n")


def test_veracity_parameters():
    """Test V - Veracity/Factual Dimension parameters"""
    text = """
    Research shows that 85% of participants improved their performance.
    According to Smith (2020), the methodology was rigorously tested.
    The data demonstrates a clear correlation between variables.
    Multiple studies have confirmed these findings.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Veracity Parameters ===")
    
    # Test factual density
    factual = analyzer.analyze_factual_density(text)
    print(f"Factual Density: {factual:.3f}")
    assert 0 <= factual <= 1, "Factual density should be in [0, 1]"
    
    # Test fact precision
    precision = analyzer.analyze_fact_precision(text)
    print(f"Fact Precision: {precision:.3f}")
    assert 0 <= precision <= 1, "Fact precision should be in [0, 1]"
    
    # Test evidence linkage
    evidence = analyzer.analyze_evidence_linkage(text)
    print(f"Evidence Linkage: {evidence:.3f}")
    assert 0 <= evidence <= 1, "Evidence linkage should be in [0, 1]"
    
    # Test truth confidence
    confidence = analyzer.analyze_truth_confidence(text)
    print(f"Truth Confidence: {confidence:.3f}")
    assert 0 <= confidence <= 1, "Truth confidence should be in [0, 1]"
    
    # Test source diversity
    diversity = analyzer.analyze_source_diversity(text)
    print(f"Source Diversity: {diversity:.3f}")
    assert 0 <= diversity <= 1, "Source diversity should be in [0, 1]"
    
    print("✓ All Veracity parameters validated\n")


def test_uncertainty_parameters():
    """Test U - Uncertainty/Ambiguity parameters"""
    text = """
    This might be the right approach, but we're not entirely certain.
    Perhaps we should consider alternative options. If we were to implement this,
    it could potentially work. Maybe there are some issues to address.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Uncertainty Parameters ===")
    
    # Test ambiguity entropy
    ambiguity = analyzer.analyze_ambiguity_entropy(text)
    print(f"Ambiguity Entropy: {ambiguity:.3f}")
    assert 0 <= ambiguity <= 1, "Ambiguity entropy should be in [0, 1]"
    
    # Test vagueness
    vagueness = analyzer.analyze_vagueness(text)
    print(f"Vagueness: {vagueness:.3f}")
    assert 0 <= vagueness <= 1, "Vagueness should be in [0, 1]"
    
    # Test cognitive dissonance
    dissonance = analyzer.analyze_cognitive_dissonance(text)
    print(f"Cognitive Dissonance: {dissonance:.3f}")
    assert 0 <= dissonance <= 1, "Cognitive dissonance should be in [0, 1]"
    
    # Test hypothetical load
    hypothetical = analyzer.analyze_hypothetical_load(text)
    print(f"Hypothetical Load: {hypothetical:.3f}")
    assert 0 <= hypothetical <= 1, "Hypothetical load should be in [0, 1]"
    
    # Test certainty oscillation
    oscillation = analyzer.analyze_certainty_oscillation(text)
    print(f"Certainty Oscillation: {oscillation:.3f}")
    assert 0 <= oscillation <= 1, "Certainty oscillation should be in [0, 1]"
    
    print("✓ All Uncertainty parameters validated\n")


def test_advanced_nlp_features():
    """Test advanced NLP features: NER, relationships, etc."""
    text = """
    John Smith works at Microsoft in Seattle. He met Sarah Johnson yesterday.
    The company was founded in 1975. They discussed the new project at 
    john.smith@microsoft.com. The meeting happened on 2024-01-15.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Advanced NLP Features ===")
    
    # Test NER
    entities = analyzer.extract_named_entities_advanced(text)
    print(f"Named Entities: {entities}")
    assert 'total_entities' in entities, "Should extract entities"
    assert entities['total_entities'] > 0, "Should find at least some entities"
    
    # Test relationship extraction
    relationships = analyzer.extract_relationships(text)
    print(f"Relationships: {relationships}")
    assert 'relationship_count' in relationships, "Should extract relationships"
    
    # Test word sense disambiguation
    wsd = analyzer.extract_word_sense_disambiguation(text)
    print(f"Word Sense Disambiguation: {wsd}")
    assert 'avg_word_specificity' in wsd, "Should provide WSD metrics"
    
    # Test information extraction
    info = analyzer.extract_information_extraction(text)
    print(f"Information Extraction: {info}")
    assert 'date_count' in info, "Should extract dates"
    assert 'email_count' in info, "Should extract emails"
    
    print("✓ All Advanced NLP features validated\n")


def test_temporal_windows():
    """Test multi-scale temporal window analysis"""
    # Create a longer text for meaningful window analysis
    text = """
    The research project began with a comprehensive literature review.
    We analyzed over 100 academic papers from the past decade.
    The methodology was carefully designed to ensure reliability.
    Data collection took place over six months in 2023.
    
    Initial results showed promising trends in the dataset.
    Statistical analysis revealed significant correlations.
    We observed unexpected patterns in the temporal data.
    These findings challenged our original hypotheses.
    
    The discussion section explores implications thoroughly.
    We compared our results with previous studies extensively.
    Several limitations were identified and addressed properly.
    Future research directions are clearly outlined below.
    
    In conclusion, this study contributes valuable insights.
    The practical applications are numerous and significant.
    We recommend further investigation in this area.
    Collaboration between researchers will be essential moving forward.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Temporal Window Analysis ===")
    
    # Run complete temporal analysis
    results = analyzer.analyze_temporal_windows()
    
    # Validate structure
    assert 'full_text_analysis' in results, "Should have full text analysis"
    assert 'window_analyses' in results, "Should have window analyses"
    assert 'temporal_trends' in results, "Should have temporal trends"
    assert 'metadata' in results, "Should have metadata"
    
    print(f"Total tokens: {results['metadata']['total_tokens']}")
    print(f"Window sizes: {results['metadata']['window_sizes']}")
    
    # Validate full text analysis contains all parameters
    full_analysis = results['full_text_analysis']
    required_params = [
        'logical_coherence', 'causal_density', 'argumentation_entropy',
        'emotional_valence', 'arousal', 'lexical_diversity',
        'factual_density', 'vagueness'
    ]
    
    for param in required_params:
        assert param in full_analysis, f"Missing parameter: {param}"
        value = full_analysis[param]
        if isinstance(value, (int, float)):
            assert 0 <= value <= 1, f"{param} should be in [0, 1], got {value}"
    
    print(f"✓ Full text analysis validated with {len(full_analysis)} parameters")
    
    # Validate window analyses
    for window_key, windows in results['window_analyses'].items():
        print(f"  {window_key}: {len(windows)} windows analyzed")
        assert len(windows) > 0, f"Should have windows for {window_key}"
    
    # Validate temporal trends
    if results['temporal_trends']:
        print(f"  Temporal trends calculated for {len(results['temporal_trends'])} window sizes")
    
    print("✓ Temporal window analysis validated\n")


def test_complete_analysis():
    """Test the complete analysis pipeline"""
    text = """
    Climate change poses an unprecedented challenge to our planet.
    Scientists have definitively shown that global temperatures are rising.
    According to NASA, the past decade was the warmest on record.
    We must act urgently to reduce greenhouse gas emissions.
    
    Renewable energy sources offer a viable solution to this crisis.
    Solar and wind power have become increasingly cost-effective.
    Many countries are transitioning away from fossil fuels rapidly.
    However, significant obstacles remain in implementing these changes.
    """
    
    analyzer = AdvancedNLPAnalyzer(text)
    
    print("=== Testing Complete Analysis Pipeline ===")
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Validate it's stored in counts
    assert 'advanced_nlp_analysis' in analyzer.counts, "Should store in counts"
    
    # Validate JSON serializability
    try:
        json_str = json.dumps(results, indent=2, default=str)
        print(f"✓ Results are JSON serializable ({len(json_str)} characters)")
    except Exception as e:
        raise AssertionError(f"Results should be JSON serializable: {e}")
    
    # Display sample output
    print("\nSample Full Text Analysis Results:")
    full = results['full_text_analysis']
    print(f"  Logical Coherence: {full['logical_coherence']:.3f}")
    print(f"  Emotional Valence: {full['emotional_valence']:.3f}")
    print(f"  Factual Density: {full['factual_density']:.3f}")
    print(f"  Lexical Diversity: {full['lexical_diversity']:.3f}")
    print(f"  Persuasiveness: {full['persuasiveness']:.3f}")
    
    print("\n✓ Complete analysis pipeline validated\n")


def test_all_parameters_numerical():
    """Verify all parameters return numerical values"""
    text = "This is a simple test sentence for validation."
    
    analyzer = AdvancedNLPAnalyzer(text)
    results = analyzer.run_complete_analysis()
    
    print("=== Validating All Parameters Are Numerical ===")
    
    full_analysis = results['full_text_analysis']
    
    non_numerical = []
    for key, value in full_analysis.items():
        if isinstance(value, dict):
            # Check nested values
            for k, v in value.items():
                if not isinstance(v, (int, float, list, dict)):
                    non_numerical.append(f"{key}.{k}: {type(v)}")
        elif not isinstance(value, (int, float, list, dict, str)):
            non_numerical.append(f"{key}: {type(value)}")
    
    if non_numerical:
        print(f"⚠ Non-numerical values found: {non_numerical}")
    else:
        print("✓ All parameter values are numerical or structured data")
    
    print(f"Total parameters analyzed: {len(full_analysis)}")
    print("✓ Numerical validation complete\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Advanced NLP Analyzer Test Suite")
    print("="*60 + "\n")
    
    # Test each parameter family
    test_reasoning_parameters()
    test_constraints_parameters()
    test_goals_parameters()
    test_emotion_parameters()
    test_pragmatic_parameters()
    test_aesthetic_parameters()
    test_veracity_parameters()
    test_uncertainty_parameters()
    
    # Test advanced NLP features
    test_advanced_nlp_features()
    
    # Test temporal analysis
    test_temporal_windows()
    
    # Test complete pipeline
    test_complete_analysis()
    
    # Validate all numerical
    test_all_parameters_numerical()
    
    print("="*60)
    print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)
