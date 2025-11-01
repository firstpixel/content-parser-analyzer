"""
Advanced NLP Analyzer with Temporal Orthogonal Functions

This module implements comprehensive text-to-numerical-data analysis with:
- Multi-scale temporal windowing (100, 200, 300, 600 tokens + full text)
- RCGE-PAVU orthogonal parameter families
- Named Entity Recognition (NER)
- Relationship extraction
- Word-sense disambiguation
- Information extraction

All outputs are numerical/JSON format for validation and comparison.
"""

import spacy
import numpy as np
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import re
from typing import Dict, List, Any, Tuple
from utils.base_parser import BaseParser


class AdvancedNLPAnalyzer(BaseParser):
    """
    Advanced NLP Analyzer implementing temporal orthogonal functions for comprehensive
    text-to-numerical-data transformation.
    """
    
    def __init__(self, text: str):
        super().__init__(text)
        self.nlp = None
        self.window_sizes = [100, 200, 300, 600]
        self.temporal_analysis = {}
        
    def _load_spacy_model(self):
        """Lazy load spacy model to avoid loading if not needed."""
        if self.nlp is None:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except Exception as e:
                self.counts["spacy_error"] = str(e)
                self.nlp = None
        return self.nlp is not None
    
    def _tokenize_by_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def _create_windows(self, tokens: List[str], window_size: int) -> List[List[str]]:
        """Create sliding windows of specified size from tokens."""
        if window_size >= len(tokens):
            return [tokens]
        windows = []
        for i in range(0, len(tokens) - window_size + 1, window_size // 2):  # 50% overlap
            windows.append(tokens[i:i + window_size])
        return windows
    
    def _normalize_score(self, value: float, min_val: float = 0, max_val: float = 1) -> float:
        """Normalize a score to [0, 1] range."""
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    # ========== R - Reasoning / Logic Structure ==========
    
    def analyze_logical_coherence(self, text: str) -> float:
        """
        Measure internal logical consistency using sentence-to-sentence similarity.
        Returns a score in [0, 1] where higher means more coherent.
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0
        
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(sentences)
            similarities = []
            
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
                similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.5
        except:
            return 0.5
    
    def analyze_causal_density(self, text: str) -> float:
        """
        Count cause-effect relationships using causal markers.
        Returns normalized density.
        """
        causal_markers = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'leads to', 'causes', 'due to', 'since',
            'so that', 'in order to', 'results in'
        ]
        
        text_lower = text.lower()
        count = sum(text_lower.count(marker) for marker in causal_markers)
        words = len(word_tokenize(text))
        
        # Normalize by text length
        return self._normalize_score(count / max(words / 100, 1), 0, 5)
    
    def analyze_argumentation_entropy(self, text: str) -> float:
        """
        Measure balance of claims vs evidence using Toulmin model indicators.
        Returns entropy score [0, 1].
        """
        claims = ['claim', 'argue', 'assert', 'maintain', 'contend', 'believe']
        evidence = ['evidence', 'proof', 'data', 'study', 'research', 'shows', 'demonstrates']
        
        text_lower = text.lower()
        claim_count = sum(text_lower.count(word) for word in claims)
        evidence_count = sum(text_lower.count(word) for word in evidence)
        
        total = claim_count + evidence_count
        if total == 0:
            return 0.5
        
        # Shannon entropy of claim/evidence distribution
        p_claim = claim_count / total
        p_evidence = evidence_count / total
        
        entropy = 0
        if p_claim > 0:
            entropy -= p_claim * np.log2(p_claim)
        if p_evidence > 0:
            entropy -= p_evidence * np.log2(p_evidence)
        
        return float(entropy)  # Max entropy is 1 for binary distribution
    
    def analyze_contradiction_ratio(self, text: str) -> float:
        """
        Detect internal contradictions using sentiment and negation analysis.
        Returns ratio [0, 1] where higher means more contradictions.
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0
        
        contradictions = 0
        total_pairs = 0
        
        for i in range(len(sentences)):
            for j in range(i + 1, min(i + 5, len(sentences))):  # Check nearby sentences
                sent1 = TextBlob(sentences[i])
                sent2 = TextBlob(sentences[j])
                
                # Check sentiment opposition
                if sent1.sentiment.polarity * sent2.sentiment.polarity < -0.3:
                    contradictions += 1
                
                total_pairs += 1
        
        return self._normalize_score(contradictions / max(total_pairs, 1), 0, 0.3)
    
    def analyze_inferential_depth(self, text: str) -> float:
        """
        Measure average reasoning depth using subordinate clauses and conjunctions.
        Returns normalized depth score.
        """
        sentences = sent_tokenize(text)
        depths = []
        
        for sentence in sentences:
            # Count subordinating conjunctions and relative pronouns
            depth_markers = ['if', 'when', 'because', 'although', 'unless', 'while',
                           'that', 'which', 'who', 'where', 'whether']
            words = word_tokenize(sentence.lower())
            depth = sum(1 for word in words if word in depth_markers)
            depths.append(depth)
        
        avg_depth = np.mean(depths) if depths else 0
        return self._normalize_score(avg_depth, 0, 3)
    
    # ========== C - Constraints / Context Integrity ==========
    
    def analyze_domain_consistency(self, text: str, domain_keywords: List[str] = None) -> float:
        """
        Measure vocabulary consistency within topic bounds.
        Returns consistency score [0, 1].
        """
        if not domain_keywords:
            # Extract top keywords as domain
            words = word_tokenize(text.lower())
            word_freq = Counter(words)
            domain_keywords = [word for word, _ in word_freq.most_common(10)]
        
        sentences = sent_tokenize(text)
        consistency_scores = []
        
        for sentence in sentences:
            words = set(word_tokenize(sentence.lower()))
            overlap = len(words.intersection(set(domain_keywords)))
            consistency_scores.append(overlap / max(len(words), 1))
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.5
    
    def analyze_referential_stability(self, text: str) -> float:
        """
        Track entity persistence using NER.
        Returns stability score [0, 1].
        """
        if not self._load_spacy_model():
            return 0.5
        
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        
        if not entities:
            return 0.5
        
        # Measure entity repetition (stability)
        entity_counts = Counter(entities)
        avg_repetition = np.mean(list(entity_counts.values()))
        
        return self._normalize_score(avg_repetition, 1, 5)
    
    def analyze_temporal_consistency(self, text: str) -> float:
        """
        Analyze verb tense coherence.
        Returns consistency score [0, 1].
        """
        if not self._load_spacy_model():
            # Fallback: simple pattern matching
            past_markers = len(re.findall(r'\b\w+ed\b', text))
            present_markers = len(re.findall(r'\b\w+s\b', text))
            future_markers = text.lower().count('will') + text.lower().count('shall')
            
            total = past_markers + present_markers + future_markers
            if total == 0:
                return 0.5
            
            # Entropy of tense distribution
            tenses = [past_markers, present_markers, future_markers]
            probs = [t / total for t in tenses if t > 0]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            
            # Lower entropy = more consistent
            return 1.0 - self._normalize_score(entropy, 0, 1.58)  # log2(3) max entropy
        
        doc = self.nlp(text)
        tenses = [token.tag_ for token in doc if token.pos_ == 'VERB']
        
        if not tenses:
            return 0.5
        
        tense_counts = Counter(tenses)
        dominant_tense_ratio = max(tense_counts.values()) / len(tenses)
        
        return float(dominant_tense_ratio)
    
    def analyze_modality_balance(self, text: str) -> float:
        """
        Measure balance between fact and possibility statements.
        Returns balance score [0, 1].
        """
        modal_verbs = ['can', 'could', 'may', 'might', 'must', 'should', 'would', 'shall', 'will']
        
        words = word_tokenize(text.lower())
        modal_count = sum(1 for word in words if word in modal_verbs)
        
        # Balance is when modals are ~10-20% of verbs
        verb_count = sum(1 for word in words if word.endswith('s') or word.endswith('ed') or word.endswith('ing'))
        
        if verb_count == 0:
            return 0.5
        
        modal_ratio = modal_count / verb_count
        # Optimal balance around 0.15
        balance = 1.0 - abs(modal_ratio - 0.15) / 0.15
        
        return max(0.0, min(1.0, balance))
    
    def analyze_precision_index(self, text: str) -> float:
        """
        Measure specificity vs ambiguity using lexical density.
        Returns precision score [0, 1].
        """
        words = word_tokenize(text)
        unique_words = set(word.lower() for word in words)
        
        # Type-token ratio as precision measure
        lexical_density = len(unique_words) / max(len(words), 1)
        
        # Adjust for vague quantifiers
        vague_words = ['some', 'many', 'few', 'several', 'various', 'about', 'approximately']
        vague_count = sum(text.lower().count(word) for word in vague_words)
        vague_penalty = self._normalize_score(vague_count / max(len(words) / 100, 1), 0, 5)
        
        precision = lexical_density * (1 - 0.3 * vague_penalty)
        
        return max(0.0, min(1.0, precision))
    
    # ========== G - Goals / Intent & Direction ==========
    
    def analyze_goal_clarity(self, text: str) -> float:
        """
        Measure clarity of stated intent.
        Returns clarity score [0, 1].
        """
        goal_markers = ['goal', 'aim', 'objective', 'purpose', 'intend', 'plan', 'want', 'need']
        
        text_lower = text.lower()
        goal_count = sum(text_lower.count(marker) for marker in goal_markers)
        
        sentences = sent_tokenize(text)
        goal_sentences = sum(1 for s in sentences if any(marker in s.lower() for marker in goal_markers))
        
        # Clarity based on explicit goal statements
        clarity = goal_sentences / max(len(sentences), 1)
        
        return self._normalize_score(clarity, 0, 0.3)
    
    def analyze_focus_retention(self, text: str) -> float:
        """
        Measure topic drift over time using moving window cosine similarity.
        Returns retention score [0, 1] where higher means less drift.
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 1.0
        
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(sentences)
            
            # Compare first quarter with last quarter
            first_quarter = int(len(sentences) * 0.25)
            last_quarter = int(len(sentences) * 0.75)
            
            first_vec = vectors[:first_quarter].mean(axis=0)
            last_vec = vectors[last_quarter:].mean(axis=0)
            
            similarity = cosine_similarity(first_vec, last_vec)[0][0]
            
            return float(similarity)
        except:
            return 0.5
    
    def analyze_persuasiveness(self, text: str) -> float:
        """
        Measure rhetorical strength using appeal indicators.
        Returns persuasiveness score [0, 1].
        """
        # Rhetorical appeals
        ethos_markers = ['expert', 'research', 'study', 'proven', 'demonstrated']
        pathos_markers = ['feel', 'believe', 'imagine', 'consider', 'think']
        logos_markers = ['therefore', 'thus', 'evidence', 'shows', 'indicates']
        
        text_lower = text.lower()
        
        ethos = sum(text_lower.count(marker) for marker in ethos_markers)
        pathos = sum(text_lower.count(marker) for marker in pathos_markers)
        logos = sum(text_lower.count(marker) for marker in logos_markers)
        
        total_appeals = ethos + pathos + logos
        words = len(word_tokenize(text))
        
        persuasiveness = total_appeals / max(words / 100, 1)
        
        return self._normalize_score(persuasiveness, 0, 5)
    
    def analyze_commitment(self, text: str) -> float:
        """
        Measure modal certainty (opposite of hedging).
        Returns commitment score [0, 1].
        """
        hedging_words = ['may', 'might', 'could', 'possibly', 'perhaps', 'maybe']
        certain_words = ['will', 'must', 'definitely', 'certainly', 'absolutely', 'clearly']
        
        text_lower = text.lower()
        
        hedge_count = sum(text_lower.count(word) for word in hedging_words)
        certain_count = sum(text_lower.count(word) for word in certain_words)
        
        total = hedge_count + certain_count
        if total == 0:
            return 0.5
        
        commitment = certain_count / total
        
        return float(commitment)
    
    def analyze_teleology(self, text: str) -> float:
        """
        Measure purpose-driven phrasing.
        Returns teleology score [0, 1].
        """
        purpose_markers = [
            'to achieve', 'in order to', 'for the purpose of', 'aimed at',
            'designed to', 'intended to', 'so that', 'to ensure'
        ]
        
        text_lower = text.lower()
        purpose_count = sum(text_lower.count(marker) for marker in purpose_markers)
        
        sentences = sent_tokenize(text)
        teleology = purpose_count / max(len(sentences), 1)
        
        return self._normalize_score(teleology, 0, 0.5)
    
    # ========== E - Emotion / Expressive Content ==========
    
    def analyze_emotional_valence(self, text: str) -> float:
        """
        Measure positive/negative emotion.
        Returns valence score [-1, 1] normalized to [0, 1].
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Convert from [-1, 1] to [0, 1]
        return (polarity + 1) / 2
    
    def analyze_arousal(self, text: str) -> float:
        """
        Measure emotional intensity using exclamations and amplifiers.
        Returns arousal score [0, 1].
        """
        exclamations = text.count('!')
        caps_words = sum(1 for word in word_tokenize(text) if word.isupper() and len(word) > 1)
        
        amplifiers = ['very', 'extremely', 'incredibly', 'highly', 'absolutely']
        amplifier_count = sum(text.lower().count(amp) for amp in amplifiers)
        
        words = len(word_tokenize(text))
        
        arousal = (exclamations + caps_words + amplifier_count) / max(words / 50, 1)
        
        return self._normalize_score(arousal, 0, 5)
    
    def analyze_empathy_score(self, text: str) -> float:
        """
        Measure perspective-taking tone using pronouns and sentiment.
        Returns empathy score [0, 1].
        """
        perspective_words = ['you', 'your', 'we', 'our', 'us', 'understand', 'feel']
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        perspective_count = sum(1 for word in words if word in perspective_words)
        
        # Combine with positive sentiment
        blob = TextBlob(text)
        sentiment = max(blob.sentiment.polarity, 0)  # Only positive contributes to empathy
        
        empathy = (perspective_count / max(len(words) / 50, 1)) * (1 + sentiment) / 2
        
        return self._normalize_score(empathy, 0, 3)
    
    def analyze_emotional_volatility(self, text: str) -> float:
        """
        Measure sentiment change rate across windows.
        Returns volatility score [0, 1].
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0
        
        sentiments = [TextBlob(sent).sentiment.polarity for sent in sentences]
        
        # Calculate standard deviation of sentiment changes
        changes = [abs(sentiments[i+1] - sentiments[i]) for i in range(len(sentiments)-1)]
        
        volatility = np.std(changes) if changes else 0
        
        return self._normalize_score(volatility, 0, 0.5)
    
    def analyze_symbolic_resonance(self, text: str) -> float:
        """
        Measure metaphor density using figurative language indicators.
        Returns resonance score [0, 1].
        """
        metaphor_markers = ['like', 'as if', 'as though', 'metaphorically', 'symbolically']
        figurative = ['represents', 'symbolizes', 'embodies', 'reflects']
        
        text_lower = text.lower()
        
        metaphor_count = sum(text_lower.count(marker) for marker in metaphor_markers)
        figurative_count = sum(text_lower.count(word) for word in figurative)
        
        words = len(word_tokenize(text))
        
        resonance = (metaphor_count + figurative_count) / max(words / 100, 1)
        
        return self._normalize_score(resonance, 0, 3)
    
    # ========== P - Pragmatic / Contextual Use ==========
    
    def analyze_speech_act_ratio(self, text: str) -> Dict[str, float]:
        """
        Classify sentences by speech act type.
        Returns distribution of assertive, directive, expressive acts.
        """
        sentences = sent_tokenize(text)
        
        assertive = 0  # Statements
        directive = 0  # Commands, requests
        expressive = 0  # Emotions, opinions
        
        directive_markers = ['please', 'must', 'should', 'need to', 'have to', 'let us']
        expressive_markers = ['feel', 'think', 'believe', 'hope', 'wish', 'unfortunately', 'fortunately']
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            
            if sentence.endswith('?'):
                continue  # Questions are separate
            elif any(marker in sent_lower for marker in directive_markers):
                directive += 1
            elif any(marker in sent_lower for marker in expressive_markers):
                expressive += 1
            else:
                assertive += 1
        
        total = max(len(sentences), 1)
        
        return {
            'assertive': assertive / total,
            'directive': directive / total,
            'expressive': expressive / total
        }
    
    def analyze_dialogue_coherence(self, text: str) -> float:
        """
        Measure question-answer quality using adjacency pair detection.
        Returns coherence score [0, 1].
        """
        sentences = sent_tokenize(text)
        
        qa_pairs = 0
        questions = 0
        
        for i in range(len(sentences) - 1):
            if sentences[i].endswith('?'):
                questions += 1
                # Check if next sentence is an answer
                if not sentences[i + 1].endswith('?'):
                    qa_pairs += 1
        
        if questions == 0:
            return 0.5  # No questions = neutral coherence
        
        return qa_pairs / questions
    
    def analyze_pragmatic_truth(self, text: str) -> float:
        """
        Measure informativeness vs filler content.
        Returns truth score [0, 1].
        """
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally']
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        filler_count = sum(text_lower.count(filler) for filler in filler_words)
        
        # Informativeness is inverse of filler ratio
        informativeness = 1.0 - self._normalize_score(filler_count / max(len(words), 1), 0, 0.1)
        
        return informativeness
    
    def analyze_social_tone(self, text: str) -> Dict[str, float]:
        """
        Analyze politeness, dominance, cooperation.
        Returns tone scores.
        """
        politeness_markers = ['please', 'thank', 'sorry', 'excuse', 'pardon', 'kindly']
        dominance_markers = ['must', 'will', 'shall', 'demand', 'require', 'insist']
        cooperation_markers = ['we', 'us', 'our', 'together', 'collaborate', 'team']
        
        text_lower = text.lower()
        words = len(word_tokenize(text))
        
        politeness = sum(text_lower.count(marker) for marker in politeness_markers)
        dominance = sum(text_lower.count(marker) for marker in dominance_markers)
        cooperation = sum(text_lower.count(marker) for marker in cooperation_markers)
        
        return {
            'politeness': self._normalize_score(politeness / max(words / 100, 1), 0, 3),
            'dominance': self._normalize_score(dominance / max(words / 100, 1), 0, 3),
            'cooperation': self._normalize_score(cooperation / max(words / 100, 1), 0, 3)
        }
    
    def analyze_engagement_index(self, text: str) -> float:
        """
        Measure direct audience addressing.
        Returns engagement score [0, 1].
        """
        engagement_words = ['you', 'your', 'we', 'us', 'our']
        
        words = word_tokenize(text.lower())
        engagement_count = sum(1 for word in words if word in engagement_words)
        
        engagement = engagement_count / max(len(words), 1)
        
        return self._normalize_score(engagement, 0, 0.2)
    
    # ========== A - Aesthetic / Stylistic ==========
    
    def analyze_rhythm_variance(self, text: str) -> float:
        """
        Measure pacing via sentence length variation.
        Returns variance score [0, 1].
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        
        variance = np.std(lengths) if lengths else 0
        
        return self._normalize_score(variance, 0, 10)
    
    def analyze_lexical_diversity(self, text: str) -> float:
        """
        Calculate type-token ratio.
        Returns diversity score [0, 1].
        """
        words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
        
        if not words:
            return 0.0
        
        unique_words = set(words)
        
        return len(unique_words) / len(words)
    
    def analyze_imagery_density(self, text: str) -> float:
        """
        Measure descriptive richness via adjective/noun ratio.
        Returns density score [0, 1].
        """
        if not self._load_spacy_model():
            # Fallback: simple heuristic
            words = word_tokenize(text)
            adj_markers = [w for w in words if w.endswith('ly') or w.endswith('ful') or w.endswith('ous')]
            return self._normalize_score(len(adj_markers) / max(len(words), 1), 0, 0.3)
        
        doc = self.nlp(text)
        
        adjectives = sum(1 for token in doc if token.pos_ == 'ADJ')
        nouns = sum(1 for token in doc if token.pos_ == 'NOUN')
        
        if nouns == 0:
            return 0.0
        
        density = adjectives / nouns
        
        return self._normalize_score(density, 0, 1)
    
    def analyze_symmetry_index(self, text: str) -> float:
        """
        Detect structural balance using sentence pattern similarity.
        Returns symmetry score [0, 1].
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return 0.0
        
        # Compare sentence structures (length patterns)
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        
        # Check for patterns (e.g., similar lengths)
        length_variance = np.std(lengths)
        
        # Lower variance = more symmetry
        symmetry = 1.0 - self._normalize_score(length_variance, 0, 10)
        
        return symmetry
    
    def analyze_surprise_novelty(self, text: str) -> float:
        """
        Measure information gain using word frequency surprise.
        Returns novelty score [0, 1].
        """
        words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
        
        if not words:
            return 0.0
        
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calculate entropy (higher entropy = more novelty)
        entropy = 0
        for count in word_freq.values():
            p = count / total_words
            entropy -= p * np.log2(p)
        
        # Normalize by max possible entropy
        max_entropy = np.log2(total_words)
        
        return entropy / max_entropy if max_entropy > 0 else 0.5
    
    # ========== V - Veracity / Factual Dimension ==========
    
    def analyze_factual_density(self, text: str) -> float:
        """
        Count factual claims per sentence.
        Returns density score [0, 1].
        """
        sentences = sent_tokenize(text)
        
        fact_markers = ['is', 'are', 'was', 'were', 'has', 'have', 'shows', 'indicates', 'proves']
        
        factual_sentences = sum(1 for sent in sentences if any(marker in sent.lower().split() for marker in fact_markers))
        
        density = factual_sentences / max(len(sentences), 1)
        
        return density
    
    def analyze_fact_precision(self, text: str) -> float:
        """
        Measure specificity of claims using numbers and proper nouns.
        Returns precision score [0, 1].
        """
        # Count numbers (specific data points)
        numbers = len(re.findall(r'\b\d+\.?\d*\b', text))
        
        # Count proper nouns if spacy available
        proper_nouns = 0
        if self._load_spacy_model():
            doc = self.nlp(text)
            proper_nouns = sum(1 for token in doc if token.pos_ == 'PROPN')
        
        words = len(word_tokenize(text))
        
        precision = (numbers + proper_nouns) / max(words / 50, 1)
        
        return self._normalize_score(precision, 0, 5)
    
    def analyze_evidence_linkage(self, text: str) -> float:
        """
        Detect citations and references.
        Returns linkage score [0, 1].
        """
        citation_patterns = [
            r'\([A-Z][a-z]+,?\s+\d{4}\)',  # (Author, 2020)
            r'\[\d+\]',  # [1]
            r'according to',
            r'cited in',
            r'referenced by'
        ]
        
        citations = sum(len(re.findall(pattern, text)) for pattern in citation_patterns)
        
        sentences = sent_tokenize(text)
        
        linkage = citations / max(len(sentences), 1)
        
        return self._normalize_score(linkage, 0, 0.5)
    
    def analyze_truth_confidence(self, text: str) -> float:
        """
        Estimate factual verification potential.
        Returns confidence score [0, 1].
        """
        # High confidence markers
        confidence_markers = ['proven', 'demonstrated', 'verified', 'confirmed', 'established']
        
        # Low confidence markers
        uncertainty_markers = ['allegedly', 'supposedly', 'rumored', 'unconfirmed', 'disputed']
        
        text_lower = text.lower()
        
        confident = sum(text_lower.count(marker) for marker in confidence_markers)
        uncertain = sum(text_lower.count(marker) for marker in uncertainty_markers)
        
        if confident + uncertain == 0:
            return 0.5
        
        confidence = confident / (confident + uncertain)
        
        return confidence
    
    def analyze_source_diversity(self, text: str) -> float:
        """
        Count unique source references.
        Returns diversity score [0, 1].
        """
        source_markers = ['according to', 'states', 'claims', 'reports', 'says']
        
        sentences = sent_tokenize(text)
        
        sourced_sentences = sum(1 for sent in sentences if any(marker in sent.lower() for marker in source_markers))
        
        diversity = sourced_sentences / max(len(sentences), 1)
        
        return diversity
    
    # ========== U - Uncertainty / Ambiguity ==========
    
    def analyze_ambiguity_entropy(self, text: str) -> float:
        """
        Measure word sense entropy (polysemy density).
        Returns entropy score [0, 1].
        """
        # Words with high ambiguity
        ambiguous_words = ['run', 'set', 'go', 'take', 'get', 'make', 'put', 'give']
        
        words = word_tokenize(text.lower())
        ambiguous_count = sum(1 for word in words if word in ambiguous_words)
        
        ambiguity = ambiguous_count / max(len(words), 1)
        
        return self._normalize_score(ambiguity, 0, 0.1)
    
    def analyze_vagueness(self, text: str) -> float:
        """
        Detect fuzzy quantifiers.
        Returns vagueness score [0, 1].
        """
        vague_quantifiers = ['some', 'many', 'few', 'several', 'often', 'sometimes', 'maybe', 'approximately']
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        vague_count = sum(1 for word in words if word in vague_quantifiers)
        
        vagueness = vague_count / max(len(words), 1)
        
        return self._normalize_score(vagueness, 0, 0.1)
    
    def analyze_cognitive_dissonance(self, text: str) -> float:
        """
        Detect mismatch between sentiment and logical content.
        Returns dissonance score [0, 1].
        """
        sentences = sent_tokenize(text)
        
        dissonance_cases = 0
        
        for sentence in sentences:
            blob = TextBlob(sentence)
            sentiment = blob.sentiment.polarity
            
            # Negative logical markers
            negative_logic = ['however', 'but', 'although', 'despite', 'unfortunately']
            has_negative_logic = any(marker in sentence.lower() for marker in negative_logic)
            
            # Dissonance: positive sentiment with negative logic or vice versa
            if (sentiment > 0.3 and has_negative_logic) or (sentiment < -0.3 and not has_negative_logic):
                dissonance_cases += 1
        
        dissonance = dissonance_cases / max(len(sentences), 1)
        
        return self._normalize_score(dissonance, 0, 0.3)
    
    def analyze_hypothetical_load(self, text: str) -> float:
        """
        Measure counterfactual and conditional statements.
        Returns load score [0, 1].
        """
        hypothetical_markers = ['if', 'were', 'could', 'would', 'might', 'suppose', 'imagine', 'what if']
        
        text_lower = text.lower()
        
        hypothetical_count = sum(text_lower.count(marker) for marker in hypothetical_markers)
        
        sentences = sent_tokenize(text)
        
        load = hypothetical_count / max(len(sentences), 1)
        
        return self._normalize_score(load, 0, 1)
    
    def analyze_certainty_oscillation(self, text: str) -> float:
        """
        Measure variance of certainty over time.
        Returns oscillation score [0, 1].
        """
        sentences = sent_tokenize(text)
        
        certain_markers = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously']
        uncertain_markers = ['maybe', 'perhaps', 'possibly', 'might', 'could']
        
        certainty_scores = []
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            certain = sum(sent_lower.count(marker) for marker in certain_markers)
            uncertain = sum(sent_lower.count(marker) for marker in uncertain_markers)
            
            if certain + uncertain > 0:
                score = certain / (certain + uncertain)
            else:
                score = 0.5
            
            certainty_scores.append(score)
        
        if len(certainty_scores) < 2:
            return 0.0
        
        oscillation = np.std(certainty_scores)
        
        return self._normalize_score(oscillation, 0, 0.5)
    
    # ========== Named Entity Recognition & Relationship Extraction ==========
    
    def extract_named_entities_advanced(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities with counts and types.
        Returns structured entity data.
        """
        if not self._load_spacy_model():
            return {'entities': {}, 'entity_count': 0}
        
        doc = self.nlp(text)
        
        entities_by_type = defaultdict(list)
        
        for ent in doc.ents:
            entities_by_type[ent.label_].append(ent.text)
        
        entity_counts = {label: len(ents) for label, ents in entities_by_type.items()}
        
        return {
            'entities_by_type': dict(entities_by_type),
            'entity_counts': entity_counts,
            'total_entities': sum(entity_counts.values()),
            'entity_density': sum(entity_counts.values()) / max(len(word_tokenize(text)), 1)
        }
    
    def extract_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract subject-verb-object relationships.
        Returns relationship data.
        """
        if not self._load_spacy_model():
            return {'relationships': [], 'relationship_count': 0}
        
        doc = self.nlp(text)
        relationships = []
        
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == 'VERB':
                    subject = None
                    obj = None
                    
                    for child in token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            subject = child.text
                        elif child.dep_ in ['dobj', 'pobj']:
                            obj = child.text
                    
                    if subject and obj:
                        relationships.append({
                            'subject': subject,
                            'verb': token.text,
                            'object': obj
                        })
        
        return {
            'relationships': relationships,
            'relationship_count': len(relationships),
            'relationship_density': len(relationships) / max(len(list(doc.sents)), 1)
        }
    
    def extract_word_sense_disambiguation(self, text: str) -> Dict[str, float]:
        """
        Analyze word sense ambiguity using context.
        Returns disambiguation metrics.
        """
        if not self._load_spacy_model():
            return {'avg_word_specificity': 0.5}
        
        doc = self.nlp(text)
        
        # Use POS tags and dependency parsing as proxy for word sense
        word_specificities = []
        
        for token in doc:
            if token.is_alpha and not token.is_stop:
                # More specific POS tags and dependencies indicate clearer sense
                specificity = 0.5
                
                if token.pos_ in ['PROPN', 'NUM']:
                    specificity = 1.0
                elif token.pos_ in ['NOUN', 'VERB']:
                    specificity = 0.7
                else:
                    specificity = 0.5
                
                word_specificities.append(specificity)
        
        return {
            'avg_word_specificity': float(np.mean(word_specificities)) if word_specificities else 0.5,
            'total_words_analyzed': len(word_specificities)
        }
    
    def extract_information_extraction(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information: dates, numbers, entities, facts.
        Returns extracted information.
        """
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b', text)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        
        # Extract URLs - simplified pattern for security
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        
        entities = {}
        if self._load_spacy_model():
            doc = self.nlp(text)
            entities = {ent.label_: [e.text for e in doc.ents if e.label_ == ent.label_] 
                       for ent in doc.ents}
        
        return {
            'dates': dates,
            'date_count': len(dates),
            'numbers': numbers,
            'number_count': len(numbers),
            'emails': emails,
            'email_count': len(emails),
            'urls': urls,
            'url_count': len(urls),
            'entities': entities
        }
    
    # ========== Temporal Analysis ==========
    
    def analyze_temporal_windows(self) -> Dict[str, Any]:
        """
        Perform multi-scale temporal window analysis.
        Returns complete temporal analysis with all parameters across window sizes.
        """
        text = self.text
        tokens = self._tokenize_by_words(text)
        
        # Full text analysis
        full_analysis = self._analyze_single_window(text, "full")
        
        # Window-based analysis
        window_analyses = {}
        
        for window_size in self.window_sizes:
            if len(tokens) < window_size:
                # Text is shorter than window, analyze as single window
                window_analyses[f"window_{window_size}"] = [full_analysis]
            else:
                windows = self._create_windows(tokens, window_size)
                window_results = []
                
                for i, window_tokens in enumerate(windows):
                    window_text = ' '.join(window_tokens)
                    window_result = self._analyze_single_window(window_text, f"window_{i}")
                    window_results.append(window_result)
                
                window_analyses[f"window_{window_size}"] = window_results
        
        # Calculate temporal trends
        temporal_trends = self._calculate_temporal_trends(window_analyses)
        
        return {
            'full_text_analysis': full_analysis,
            'window_analyses': window_analyses,
            'temporal_trends': temporal_trends,
            'metadata': {
                'total_tokens': len(tokens),
                'window_sizes': self.window_sizes,
                'total_windows': {f"window_{ws}": len(window_analyses.get(f"window_{ws}", [])) 
                                for ws in self.window_sizes}
            }
        }
    
    def _analyze_single_window(self, text: str, window_id: str) -> Dict[str, Any]:
        """
        Analyze a single text window with all orthogonal parameters.
        """
        return {
            'window_id': window_id,
            # R - Reasoning
            'logical_coherence': self.analyze_logical_coherence(text),
            'causal_density': self.analyze_causal_density(text),
            'argumentation_entropy': self.analyze_argumentation_entropy(text),
            'contradiction_ratio': self.analyze_contradiction_ratio(text),
            'inferential_depth': self.analyze_inferential_depth(text),
            
            # C - Constraints
            'domain_consistency': self.analyze_domain_consistency(text),
            'referential_stability': self.analyze_referential_stability(text),
            'temporal_consistency': self.analyze_temporal_consistency(text),
            'modality_balance': self.analyze_modality_balance(text),
            'precision_index': self.analyze_precision_index(text),
            
            # G - Goals
            'goal_clarity': self.analyze_goal_clarity(text),
            'focus_retention': self.analyze_focus_retention(text),
            'persuasiveness': self.analyze_persuasiveness(text),
            'commitment': self.analyze_commitment(text),
            'teleology': self.analyze_teleology(text),
            
            # E - Emotion
            'emotional_valence': self.analyze_emotional_valence(text),
            'arousal': self.analyze_arousal(text),
            'empathy_score': self.analyze_empathy_score(text),
            'emotional_volatility': self.analyze_emotional_volatility(text),
            'symbolic_resonance': self.analyze_symbolic_resonance(text),
            
            # P - Pragmatic
            'speech_act_ratio': self.analyze_speech_act_ratio(text),
            'dialogue_coherence': self.analyze_dialogue_coherence(text),
            'pragmatic_truth': self.analyze_pragmatic_truth(text),
            'social_tone': self.analyze_social_tone(text),
            'engagement_index': self.analyze_engagement_index(text),
            
            # A - Aesthetic
            'rhythm_variance': self.analyze_rhythm_variance(text),
            'lexical_diversity': self.analyze_lexical_diversity(text),
            'imagery_density': self.analyze_imagery_density(text),
            'symmetry_index': self.analyze_symmetry_index(text),
            'surprise_novelty': self.analyze_surprise_novelty(text),
            
            # V - Veracity
            'factual_density': self.analyze_factual_density(text),
            'fact_precision': self.analyze_fact_precision(text),
            'evidence_linkage': self.analyze_evidence_linkage(text),
            'truth_confidence': self.analyze_truth_confidence(text),
            'source_diversity': self.analyze_source_diversity(text),
            
            # U - Uncertainty
            'ambiguity_entropy': self.analyze_ambiguity_entropy(text),
            'vagueness': self.analyze_vagueness(text),
            'cognitive_dissonance': self.analyze_cognitive_dissonance(text),
            'hypothetical_load': self.analyze_hypothetical_load(text),
            'certainty_oscillation': self.analyze_certainty_oscillation(text),
            
            # Advanced NLP
            'named_entities': self.extract_named_entities_advanced(text),
            'relationships': self.extract_relationships(text),
            'word_sense_disambiguation': self.extract_word_sense_disambiguation(text),
            'information_extraction': self.extract_information_extraction(text)
        }
    
    def _calculate_temporal_trends(self, window_analyses: Dict) -> Dict[str, Any]:
        """
        Calculate trends and patterns across temporal windows.
        """
        trends = {}
        
        for window_size_key, windows in window_analyses.items():
            if not windows or len(windows) < 2:
                continue
            
            # Extract time series for each parameter
            param_series = defaultdict(list)
            
            for window in windows:
                for param, value in window.items():
                    if isinstance(value, (int, float)):
                        param_series[param].append(value)
            
            # Calculate trends
            window_trends = {}
            for param, series in param_series.items():
                if len(series) >= 2:
                    # Calculate basic statistics
                    window_trends[param] = {
                        'mean': float(np.mean(series)),
                        'std': float(np.std(series)),
                        'min': float(np.min(series)),
                        'max': float(np.max(series)),
                        'trend': 'increasing' if series[-1] > series[0] else 'decreasing',
                        'volatility': float(np.std(np.diff(series))) if len(series) > 1 else 0.0
                    }
            
            trends[window_size_key] = window_trends
        
        return trends
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete temporal orthogonal analysis on the text.
        Returns comprehensive analysis results.
        """
        temporal_results = self.analyze_temporal_windows()
        
        # Store in counts for compatibility
        self.counts['advanced_nlp_analysis'] = temporal_results
        
        return temporal_results
