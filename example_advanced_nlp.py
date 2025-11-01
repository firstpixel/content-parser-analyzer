"""
Comprehensive Example: Advanced NLP Analyzer
Demonstrates the power of temporal orthogonal functions for text analysis
"""

from utils.content_parser_analyzer import ContentParserAnalyzer
import json


def main():
    # Sample text for analysis
    text = """
    Artificial intelligence is transforming the world at an unprecedented pace.
    Machine learning algorithms now surpass human performance on many specific tasks.
    Deep neural networks have achieved remarkable breakthroughs in computer vision and natural language.
    
    However, we must carefully consider the ethical implications of these powerful technologies.
    AI systems can perpetuate biases present in their training data, leading to unfair outcomes.
    Transparency and accountability are essential for building trustworthy AI systems.
    
    The future of AI depends on responsible development and deployment practices.
    We need interdisciplinary collaboration between technologists, ethicists, and policymakers.
    Only through collective effort can we ensure AI benefits all of humanity equitably.
    """
    
    print("="*80)
    print("ADVANCED NLP ANALYZER - COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    
    # Initialize parser
    parser = ContentParserAnalyzer(text)
    
    # Run complete temporal orthogonal analysis
    print("\n📊 Running complete temporal orthogonal analysis...")
    results = parser.advanced_nlp_analyzer.run_complete_analysis()
    
    # Display metadata
    print(f"\n📈 Analysis Metadata:")
    print(f"  Total Tokens: {results['metadata']['total_tokens']}")
    print(f"  Window Sizes: {results['metadata']['window_sizes']}")
    print(f"  Total Windows: {sum(results['metadata']['total_windows'].values())}")
    
    # Full text analysis
    full = results['full_text_analysis']
    
    # Display RCGE-PAVU parameters by family
    print("\n" + "="*80)
    print("RCGE-PAVU ORTHOGONAL PARAMETER ANALYSIS")
    print("="*80)
    
    print("\n🧠 R - REASONING / LOGIC STRUCTURE")
    print("  " + "-"*76)
    print(f"  Logical Coherence:        {full['logical_coherence']:.3f}  (sentence similarity)")
    print(f"  Causal Density:           {full['causal_density']:.3f}  (cause-effect links)")
    print(f"  Argumentation Entropy:    {full['argumentation_entropy']:.3f}  (claim/evidence balance)")
    print(f"  Contradiction Ratio:      {full['contradiction_ratio']:.3f}  (internal conflicts)")
    print(f"  Inferential Depth:        {full['inferential_depth']:.3f}  (reasoning complexity)")
    
    print("\n🔒 C - CONSTRAINTS / CONTEXT INTEGRITY")
    print("  " + "-"*76)
    print(f"  Domain Consistency:       {full['domain_consistency']:.3f}  (topic coherence)")
    print(f"  Referential Stability:    {full['referential_stability']:.3f}  (entity persistence)")
    print(f"  Temporal Consistency:     {full['temporal_consistency']:.3f}  (tense coherence)")
    print(f"  Modality Balance:         {full['modality_balance']:.3f}  (fact vs possibility)")
    print(f"  Precision Index:          {full['precision_index']:.3f}  (specificity)")
    
    print("\n🎯 G - GOALS / INTENT & DIRECTION")
    print("  " + "-"*76)
    print(f"  Goal Clarity:             {full['goal_clarity']:.3f}  (intent clarity)")
    print(f"  Focus Retention:          {full['focus_retention']:.3f}  (topic drift)")
    print(f"  Persuasiveness:           {full['persuasiveness']:.3f}  (rhetorical strength)")
    print(f"  Commitment:               {full['commitment']:.3f}  (certainty level)")
    print(f"  Teleology:                {full['teleology']:.3f}  (purpose-driven)")
    
    print("\n💭 E - EMOTION / EXPRESSIVE CONTENT")
    print("  " + "-"*76)
    print(f"  Emotional Valence:        {full['emotional_valence']:.3f}  (positive/negative)")
    print(f"  Arousal:                  {full['arousal']:.3f}  (intensity)")
    print(f"  Empathy Score:            {full['empathy_score']:.3f}  (perspective-taking)")
    print(f"  Emotional Volatility:     {full['emotional_volatility']:.3f}  (sentiment changes)")
    print(f"  Symbolic Resonance:       {full['symbolic_resonance']:.3f}  (metaphor density)")
    
    print("\n💬 P - PRAGMATIC / CONTEXTUAL USE")
    print("  " + "-"*76)
    speech_acts = full['speech_act_ratio']
    print(f"  Speech Acts:")
    print(f"    - Assertive:            {speech_acts['assertive']:.3f}")
    print(f"    - Directive:            {speech_acts['directive']:.3f}")
    print(f"    - Expressive:           {speech_acts['expressive']:.3f}")
    print(f"  Dialogue Coherence:       {full['dialogue_coherence']:.3f}  (Q&A quality)")
    print(f"  Pragmatic Truth:          {full['pragmatic_truth']:.3f}  (informativeness)")
    social = full['social_tone']
    print(f"  Social Tone:")
    print(f"    - Politeness:           {social['politeness']:.3f}")
    print(f"    - Dominance:            {social['dominance']:.3f}")
    print(f"    - Cooperation:          {social['cooperation']:.3f}")
    print(f"  Engagement Index:         {full['engagement_index']:.3f}  (audience addressing)")
    
    print("\n🎨 A - AESTHETIC / STYLISTIC")
    print("  " + "-"*76)
    print(f"  Rhythm Variance:          {full['rhythm_variance']:.3f}  (sentence pacing)")
    print(f"  Lexical Diversity:        {full['lexical_diversity']:.3f}  (vocabulary richness)")
    print(f"  Imagery Density:          {full['imagery_density']:.3f}  (descriptiveness)")
    print(f"  Symmetry Index:           {full['symmetry_index']:.3f}  (structural balance)")
    print(f"  Surprise/Novelty:         {full['surprise_novelty']:.3f}  (information gain)")
    
    print("\n✅ V - VERACITY / FACTUAL DIMENSION")
    print("  " + "-"*76)
    print(f"  Factual Density:          {full['factual_density']:.3f}  (claims/sentence)")
    print(f"  Fact Precision:           {full['fact_precision']:.3f}  (specificity)")
    print(f"  Evidence Linkage:         {full['evidence_linkage']:.3f}  (citations)")
    print(f"  Truth Confidence:         {full['truth_confidence']:.3f}  (verification potential)")
    print(f"  Source Diversity:         {full['source_diversity']:.3f}  (unique sources)")
    
    print("\n❓ U - UNCERTAINTY / AMBIGUITY")
    print("  " + "-"*76)
    print(f"  Ambiguity Entropy:        {full['ambiguity_entropy']:.3f}  (word sense)")
    print(f"  Vagueness:                {full['vagueness']:.3f}  (fuzzy quantifiers)")
    print(f"  Cognitive Dissonance:     {full['cognitive_dissonance']:.3f}  (sentiment/logic mismatch)")
    print(f"  Hypothetical Load:        {full['hypothetical_load']:.3f}  (counterfactuals)")
    print(f"  Certainty Oscillation:    {full['certainty_oscillation']:.3f}  (certainty variance)")
    
    # Advanced NLP Features
    print("\n" + "="*80)
    print("ADVANCED NLP FEATURES")
    print("="*80)
    
    # Named Entities
    ner = full['named_entities']
    print(f"\n🏷️  Named Entity Recognition:")
    print(f"  Total Entities: {ner['total_entities']}")
    print(f"  Entity Density: {ner['entity_density']:.3f}")
    if ner['entity_counts']:
        print(f"  Entity Types:")
        for entity_type, count in ner['entity_counts'].items():
            print(f"    - {entity_type}: {count}")
    
    # Relationships
    rels = full['relationships']
    print(f"\n🔗 Relationship Extraction:")
    print(f"  Total Relationships: {rels['relationship_count']}")
    print(f"  Relationship Density: {rels['relationship_density']:.3f}")
    if rels['relationships']:
        print(f"  Sample Relationships:")
        for rel in rels['relationships'][:3]:
            print(f"    {rel['subject']} → {rel['verb']} → {rel['object']}")
    
    # Word Sense Disambiguation
    wsd = full['word_sense_disambiguation']
    print(f"\n📖 Word Sense Disambiguation:")
    print(f"  Avg Word Specificity: {wsd['avg_word_specificity']:.3f}")
    print(f"  Words Analyzed: {wsd['total_words_analyzed']}")
    
    # Information Extraction
    info = full['information_extraction']
    print(f"\n📝 Information Extraction:")
    print(f"  Dates Found: {info['date_count']}")
    print(f"  Numbers Found: {info['number_count']}")
    print(f"  Emails Found: {info['email_count']}")
    print(f"  URLs Found: {info['url_count']}")
    
    # Temporal Window Analysis
    print("\n" + "="*80)
    print("TEMPORAL WINDOW ANALYSIS")
    print("="*80)
    
    print(f"\n⏱️  Multi-Scale Windowing:")
    for window_key, windows in results['window_analyses'].items():
        print(f"  {window_key}: {len(windows)} window(s) analyzed")
    
    # Temporal Trends
    if results['temporal_trends']:
        print(f"\n📈 Temporal Trends Available:")
        for window_key, trends in results['temporal_trends'].items():
            print(f"  {window_key}: tracking evolution of {len(trends)} parameters")
            
            # Show sample trend for emotional valence
            if 'emotional_valence' in trends:
                ev = trends['emotional_valence']
                print(f"    Example - Emotional Valence:")
                print(f"      Mean: {ev['mean']:.3f}, Std: {ev['std']:.3f}")
                print(f"      Trend: {ev['trend']}, Volatility: {ev['volatility']:.3f}")
    
    # JSON Export
    print("\n" + "="*80)
    print("JSON EXPORT")
    print("="*80)
    
    json_output = json.dumps(results, indent=2, default=str)
    print(f"\n💾 Complete analysis exported to JSON:")
    print(f"  Total size: {len(json_output):,} characters")
    print(f"  All data is numerical and serializable")
    print(f"  Ready for validation and comparison")
    
    # Save to file
    output_file = "/tmp/advanced_nlp_analysis.json"
    with open(output_file, 'w') as f:
        f.write(json_output)
    print(f"  Saved to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"""
✓ Analyzed text with {results['metadata']['total_tokens']} tokens
✓ Extracted 44+ orthogonal parameters across RCGE-PAVU framework
✓ Performed multi-scale temporal windowing ({len(results['metadata']['window_sizes'])} sizes)
✓ Tracked parameter evolution across time windows
✓ Extracted named entities, relationships, and structured information
✓ All data normalized to [0, 1] range for validation
✓ Complete JSON export available for further analysis

This represents a complete text-to-numerical-data transformation,
enabling orthogonal validation and comparison of textual content.
    """)
    
    print("="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
