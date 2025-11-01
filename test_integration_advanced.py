"""
Integration test for Advanced NLP Analyzer through ContentParserAnalyzer
"""

from utils.content_parser_analyzer import ContentParserAnalyzer
import json


def test_integration_basic():
    """Test basic integration through ContentParserAnalyzer"""
    text = """
    The research demonstrates that climate change is accelerating globally.
    Scientists have conclusively proven that human activities contribute significantly.
    We must act urgently to mitigate these effects and protect future generations.
    Renewable energy solutions offer a promising path forward.
    """
    
    parser = ContentParserAnalyzer(text)
    
    print("=== Testing Integration: Basic Access ===")
    
    # Access advanced analyzer directly
    results = parser.advanced_nlp_analyzer.run_complete_analysis()
    
    print(f"✓ Analysis completed successfully")
    print(f"  Total tokens: {results['metadata']['total_tokens']}")
    print(f"  Window sizes: {results['metadata']['window_sizes']}")
    
    # Check full text analysis
    full = results['full_text_analysis']
    print(f"\nKey Metrics:")
    print(f"  Logical Coherence: {full['logical_coherence']:.3f}")
    print(f"  Emotional Valence: {full['emotional_valence']:.3f}")
    print(f"  Factual Density: {full['factual_density']:.3f}")
    print(f"  Persuasiveness: {full['persuasiveness']:.3f}")
    print(f"  Lexical Diversity: {full['lexical_diversity']:.3f}")
    
    # Verify counts integration
    counts = parser.get_counts()
    assert 'advanced_nlp_analysis' in counts, "Advanced analysis should be in counts"
    
    print("\n✓ Integration test passed\n")


def test_integration_temporal_windows():
    """Test temporal window analysis integration"""
    text = """
    The scientific method requires careful observation and measurement.
    Researchers collect data systematically over extended periods of time.
    Analysis of this data reveals patterns and correlations in the findings.
    Statistical methods help validate the significance of observed results.
    
    Peer review ensures quality and rigor in published research papers.
    Collaboration between institutions strengthens the research community worldwide.
    Innovation emerges from combining diverse perspectives and methodologies together.
    Future directions point toward interdisciplinary approaches and novel techniques.
    """
    
    parser = ContentParserAnalyzer(text)
    
    print("=== Testing Integration: Temporal Windows ===")
    
    # Run temporal analysis
    results = parser.advanced_nlp_analyzer.analyze_temporal_windows()
    
    print(f"✓ Temporal analysis completed")
    print(f"  Total tokens: {results['metadata']['total_tokens']}")
    
    # Check window analyses
    for window_key in results['window_analyses']:
        windows = results['window_analyses'][window_key]
        print(f"  {window_key}: {len(windows)} windows")
    
    # Check temporal trends
    if results['temporal_trends']:
        print(f"\n✓ Temporal trends calculated:")
        for window_key, trends in results['temporal_trends'].items():
            num_params = len(trends)
            print(f"  {window_key}: {num_params} parameters tracked")
    
    print("\n✓ Temporal integration test passed\n")


def test_integration_all_parameters():
    """Test that all RCGE-PAVU parameters are accessible"""
    text = "This is a test sentence for parameter validation."
    
    parser = ContentParserAnalyzer(text)
    analyzer = parser.advanced_nlp_analyzer
    
    print("=== Testing Integration: All RCGE-PAVU Parameters ===")
    
    # R - Reasoning
    print("Testing R - Reasoning parameters...")
    assert hasattr(analyzer, 'analyze_logical_coherence')
    assert hasattr(analyzer, 'analyze_causal_density')
    assert hasattr(analyzer, 'analyze_argumentation_entropy')
    assert hasattr(analyzer, 'analyze_contradiction_ratio')
    assert hasattr(analyzer, 'analyze_inferential_depth')
    print("  ✓ 5 Reasoning parameters")
    
    # C - Constraints
    print("Testing C - Constraints parameters...")
    assert hasattr(analyzer, 'analyze_domain_consistency')
    assert hasattr(analyzer, 'analyze_referential_stability')
    assert hasattr(analyzer, 'analyze_temporal_consistency')
    assert hasattr(analyzer, 'analyze_modality_balance')
    assert hasattr(analyzer, 'analyze_precision_index')
    print("  ✓ 5 Constraints parameters")
    
    # G - Goals
    print("Testing G - Goals parameters...")
    assert hasattr(analyzer, 'analyze_goal_clarity')
    assert hasattr(analyzer, 'analyze_focus_retention')
    assert hasattr(analyzer, 'analyze_persuasiveness')
    assert hasattr(analyzer, 'analyze_commitment')
    assert hasattr(analyzer, 'analyze_teleology')
    print("  ✓ 5 Goals parameters")
    
    # E - Emotion
    print("Testing E - Emotion parameters...")
    assert hasattr(analyzer, 'analyze_emotional_valence')
    assert hasattr(analyzer, 'analyze_arousal')
    assert hasattr(analyzer, 'analyze_empathy_score')
    assert hasattr(analyzer, 'analyze_emotional_volatility')
    assert hasattr(analyzer, 'analyze_symbolic_resonance')
    print("  ✓ 5 Emotion parameters")
    
    # P - Pragmatic
    print("Testing P - Pragmatic parameters...")
    assert hasattr(analyzer, 'analyze_speech_act_ratio')
    assert hasattr(analyzer, 'analyze_dialogue_coherence')
    assert hasattr(analyzer, 'analyze_pragmatic_truth')
    assert hasattr(analyzer, 'analyze_social_tone')
    assert hasattr(analyzer, 'analyze_engagement_index')
    print("  ✓ 5 Pragmatic parameters")
    
    # A - Aesthetic
    print("Testing A - Aesthetic parameters...")
    assert hasattr(analyzer, 'analyze_rhythm_variance')
    assert hasattr(analyzer, 'analyze_lexical_diversity')
    assert hasattr(analyzer, 'analyze_imagery_density')
    assert hasattr(analyzer, 'analyze_symmetry_index')
    assert hasattr(analyzer, 'analyze_surprise_novelty')
    print("  ✓ 5 Aesthetic parameters")
    
    # V - Veracity
    print("Testing V - Veracity parameters...")
    assert hasattr(analyzer, 'analyze_factual_density')
    assert hasattr(analyzer, 'analyze_fact_precision')
    assert hasattr(analyzer, 'analyze_evidence_linkage')
    assert hasattr(analyzer, 'analyze_truth_confidence')
    assert hasattr(analyzer, 'analyze_source_diversity')
    print("  ✓ 5 Veracity parameters")
    
    # U - Uncertainty
    print("Testing U - Uncertainty parameters...")
    assert hasattr(analyzer, 'analyze_ambiguity_entropy')
    assert hasattr(analyzer, 'analyze_vagueness')
    assert hasattr(analyzer, 'analyze_cognitive_dissonance')
    assert hasattr(analyzer, 'analyze_hypothetical_load')
    assert hasattr(analyzer, 'analyze_certainty_oscillation')
    print("  ✓ 5 Uncertainty parameters")
    
    # Advanced NLP
    print("Testing Advanced NLP features...")
    assert hasattr(analyzer, 'extract_named_entities_advanced')
    assert hasattr(analyzer, 'extract_relationships')
    assert hasattr(analyzer, 'extract_word_sense_disambiguation')
    assert hasattr(analyzer, 'extract_information_extraction')
    print("  ✓ 4 Advanced NLP features")
    
    print("\n✓ Total: 44 orthogonal parameters + complete temporal analysis")
    print("✓ All parameters accessible through integration\n")


def test_integration_json_output():
    """Test JSON serialization of complete analysis"""
    text = """
    Machine learning algorithms have revolutionized data analysis.
    Neural networks can identify patterns humans might miss entirely.
    However, interpretability remains a significant challenge in practice.
    """
    
    parser = ContentParserAnalyzer(text)
    
    print("=== Testing Integration: JSON Output ===")
    
    # Run analysis
    results = parser.advanced_nlp_analyzer.run_complete_analysis()
    
    # Serialize to JSON
    try:
        json_output = json.dumps(results, indent=2, default=str)
        print(f"✓ Results successfully serialized to JSON")
        print(f"  JSON size: {len(json_output)} characters")
        
        # Parse back
        parsed = json.loads(json_output)
        assert 'full_text_analysis' in parsed
        assert 'window_analyses' in parsed
        assert 'temporal_trends' in parsed
        
        print(f"✓ JSON successfully parsed back")
        print(f"  Contains all required sections")
        
    except Exception as e:
        raise AssertionError(f"JSON serialization failed: {e}")
    
    print("\n✓ JSON integration test passed\n")


def test_example_usage():
    """Example usage demonstrating the power of the analyzer"""
    text = """
    Climate change represents one of the most pressing challenges of our time.
    Scientific consensus overwhelmingly confirms that human activities are the primary driver.
    Rising global temperatures threaten ecosystems, economies, and human societies worldwide.
    
    We must transition rapidly to renewable energy sources like solar and wind power.
    Individual actions matter, but systemic change requires coordinated policy efforts.
    International cooperation is essential to achieve meaningful emissions reductions.
    
    The technology exists to address this crisis effectively and economically.
    What we lack is not capability, but collective will and political courage.
    Future generations will judge us by the actions we take today.
    """
    
    print("="*60)
    print("Example: Comprehensive Text Analysis")
    print("="*60)
    
    parser = ContentParserAnalyzer(text)
    results = parser.advanced_nlp_analyzer.run_complete_analysis()
    
    full = results['full_text_analysis']
    
    print("\n📊 RCGE-PAVU Analysis Summary:")
    print("\n🧠 R - Reasoning/Logic:")
    print(f"  Logical Coherence:     {full['logical_coherence']:.3f}")
    print(f"  Causal Density:        {full['causal_density']:.3f}")
    print(f"  Inferential Depth:     {full['inferential_depth']:.3f}")
    
    print("\n🔒 C - Constraints/Context:")
    print(f"  Domain Consistency:    {full['domain_consistency']:.3f}")
    print(f"  Temporal Consistency:  {full['temporal_consistency']:.3f}")
    print(f"  Precision Index:       {full['precision_index']:.3f}")
    
    print("\n🎯 G - Goals/Intent:")
    print(f"  Goal Clarity:          {full['goal_clarity']:.3f}")
    print(f"  Persuasiveness:        {full['persuasiveness']:.3f}")
    print(f"  Commitment:            {full['commitment']:.3f}")
    
    print("\n💭 E - Emotion/Expression:")
    print(f"  Emotional Valence:     {full['emotional_valence']:.3f}")
    print(f"  Arousal:               {full['arousal']:.3f}")
    print(f"  Empathy Score:         {full['empathy_score']:.3f}")
    
    print("\n💬 P - Pragmatic/Context:")
    print(f"  Engagement Index:      {full['engagement_index']:.3f}")
    print(f"  Pragmatic Truth:       {full['pragmatic_truth']:.3f}")
    
    print("\n🎨 A - Aesthetic/Style:")
    print(f"  Lexical Diversity:     {full['lexical_diversity']:.3f}")
    print(f"  Rhythm Variance:       {full['rhythm_variance']:.3f}")
    print(f"  Surprise/Novelty:      {full['surprise_novelty']:.3f}")
    
    print("\n✅ V - Veracity/Facts:")
    print(f"  Factual Density:       {full['factual_density']:.3f}")
    print(f"  Truth Confidence:      {full['truth_confidence']:.3f}")
    print(f"  Evidence Linkage:      {full['evidence_linkage']:.3f}")
    
    print("\n❓ U - Uncertainty/Ambiguity:")
    print(f"  Vagueness:             {full['vagueness']:.3f}")
    print(f"  Hypothetical Load:     {full['hypothetical_load']:.3f}")
    print(f"  Certainty Oscillation: {full['certainty_oscillation']:.3f}")
    
    # Show NER
    ner = full['named_entities']
    if ner['total_entities'] > 0:
        print(f"\n🏷️  Named Entities: {ner['total_entities']} found")
        for entity_type, count in ner['entity_counts'].items():
            print(f"  {entity_type}: {count}")
    
    # Show relationships
    rels = full['relationships']
    print(f"\n🔗 Relationships: {rels['relationship_count']} extracted")
    
    # Temporal windows
    print(f"\n⏱️  Temporal Analysis:")
    print(f"  Total tokens: {results['metadata']['total_tokens']}")
    print(f"  Window sizes: {results['metadata']['window_sizes']}")
    for window_key, windows in results['window_analyses'].items():
        print(f"  {window_key}: {len(windows)} windows analyzed")
    
    print("\n" + "="*60)
    print("✓ Complete orthogonal text-to-numerical transformation")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Advanced NLP Analyzer Integration Test Suite")
    print("="*60 + "\n")
    
    test_integration_basic()
    test_integration_temporal_windows()
    test_integration_all_parameters()
    test_integration_json_output()
    test_example_usage()
    
    print("="*60)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("="*60)
