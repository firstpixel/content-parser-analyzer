from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.base_parser import BaseParser
import re

class LogicalAnalyzer(BaseParser):
    def analyze_logical_flow(self):
        """Analyze logical flow using transition words and sentence structure."""
        transitions = {
            'sequence': ['first', 'second', 'then', 'next', 'finally'],
            'cause_effect': ['therefore', 'because', 'consequently', 'thus', 'as a result'],
            'contrast': ['however', 'but', 'although', 'conversely', 'nevertheless'],
            'addition': ['moreover', 'furthermore', 'additionally', 'also', 'besides']
        }
        
        sentences = sent_tokenize(self.text)
        flow_score = 0
        transition_count = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for category, markers in transitions.items():
                if any(marker in words for marker in markers):
                    transition_count += 1
                    
        flow_score = (transition_count / len(sentences)) * 100 if sentences else 0
        self.counts["logical_flow"] = {
            'score': flow_score,
            'transition_count': transition_count,
            'sentence_count': len(sentences)
        }
        return self

    def detect_contradictions(self):
        contradiction_phrases = [("yes", "no"), ("always", "never"), ("possible", "impossible")]
        self.counts["contradictions_found"] = sum(
            1 for pair in contradiction_phrases if pair[0] in self.text and pair[1] in self.text
        )
        return self
    
    def detect_fallacies(self):
        """Detect common logical fallacies in the text."""
        fallacy_patterns = {
            'ad_hominem': [r'\b(stupid|idiot|foolish|dumb)\b'],
            'appeal_to_authority': [r'(expert|authority|research|study).{0,30}(says|claims|proves)'],
            'hasty_generalization': [r'\b(all|every|always|never)\b.{0,30}\b(are|is|do|does)\b'],
            'false_dichotomy': [r'(either|or|neither|nor).{0,50}(nothing in between|no other)'],
            'slippery_slope': [r'\b(if|when)\b.{0,50}\b(then)\b.{0,50}\b(eventually|ultimately|finally|definitely)\b'],
            'circular_reasoning': [r'(\w+).{0,30}because.{0,30}\1']
        }
        
        fallacies_found = {}
        for fallacy, patterns in fallacy_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.finditer(pattern, self.text, re.IGNORECASE)
                matches.extend([m.group(0) for m in found])
            if matches:
                fallacies_found[fallacy] = matches
        
        self.counts["fallacies"] = fallacies_found
        return self

    def detect_factual_claims(self):
        fact_phrases = ["according to", "studies show", "research indicates"]
        self.counts["factual_claims"] = sum(1 for phrase in fact_phrases if phrase in self.text.lower())
        return self

    def evaluate_coherence(self):
        """Evaluate text coherence using topic consistency and flow."""
        sentences = sent_tokenize(self.text)
        coherence_score = 0
        
        # Check sentence relationships
        for i in range(len(sentences)-1):
            current = TextBlob(sentences[i])
            next_sent = TextBlob(sentences[i+1])
            
            # Compare sentiment consistency
            sent_consistency = abs(current.sentiment.polarity - next_sent.sentiment.polarity) < 0.5
            coherence_score += 1 if sent_consistency else 0
            
            # Check for common terms
            common_terms = set(current.words) & set(next_sent.words)
            coherence_score += len(common_terms) * 0.5
        
        max_possible_score = (len(sentences) - 1) * 5  # Arbitrary max score per sentence pair
        normalized_score = (coherence_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        self.counts["coherence"] = {
            'score': min(100, normalized_score),
            'sentence_count': len(sentences)
        }
        return self

    def extract_premises_conclusion(self):
        """Extract logical premises and conclusion from argumentative text."""
        premises = []
        conclusion = None
        
        sentences = sent_tokenize(self.text)
        conclusion_markers = ['therefore', 'thus', 'consequently', 'hence', 'so']
        premise_markers = ['because', 'since', 'as', 'given that']
        
        for sentence in sentences:
            lower_sent = sentence.lower()
            
            # Check for conclusion
            if any(marker in lower_sent for marker in conclusion_markers):
                conclusion = sentence
                continue
                
            # Check for premises
            if any(marker in lower_sent for marker in premise_markers):
                premises.append(sentence)
        
        self.text = {
            'premises': premises,
            'conclusion': conclusion
        }
        return self

    def extract_arguments(self):
        """Extract arguments and their supporting evidence."""
        sentences = sent_tokenize(self.text)
        arguments = []
        
        current_argument = None
        supporting_evidence = []
        
        for sentence in sentences:
            lower_sent = sentence.lower()
            
            # Identify argument statements
            if any(phrase in lower_sent for phrase in ['argue', 'claim', 'propose', 'suggest']):
                if current_argument:
                    arguments.append({
                        'argument': current_argument,
                        'evidence': supporting_evidence
                    })
                current_argument = sentence
                supporting_evidence = []
                
            # Identify supporting evidence
            elif any(phrase in lower_sent for phrase in ['because', 'since', 'evidence', 'example']):
                if current_argument:
                    supporting_evidence.append(sentence)
        
        # Add last argument if exists
        if current_argument:
            arguments.append({
                'argument': current_argument,
                'evidence': supporting_evidence
            })
        
        self.text = arguments
        return self

    def detect_socratic_method(self):
        """Detect Socratic method patterns in the text."""
        question_patterns = {
            'clarification': r'what do you mean by|could you explain|how would you define',
            'assumption': r'why do you assume|what are you assuming|is it always true that',
            'evidence': r'what are your reasons|why do you think|what evidence is there',
            'viewpoint': r'what alternative|how else could we|is there another way',
            'implication': r'what would follow|what are the consequences|how would that affect',
            'question': r'why is that important|what is the significance|how does this relate'
        }
        
        matches = {}
        for category, pattern in question_patterns.items():
            found = re.finditer(pattern, self.text.lower())
            matches[category] = [m.group(0) for m in found]
        
        self.counts["socratic_method"] = {
            'detected': any(bool(v) for v in matches.values()),
            'patterns': matches
        }
        return self

    def detect_missing_perspectives(self):
        """Detect potential missing viewpoints or perspectives."""
        common_perspectives = {
            'temporal': ['past', 'present', 'future'],
            'scope': ['individual', 'group', 'society', 'global'],
            'stakeholders': ['user', 'provider', 'regulator', 'competitor'],
            'aspects': ['technical', 'economic', 'social', 'environmental']
        }
        
        found_perspectives = {category: [] for category in common_perspectives}
        missing_perspectives = {category: [] for category in common_perspectives}
        
        for category, terms in common_perspectives.items():
            for term in terms:
                if re.search(r'\b' + term + r'\b', self.text.lower()):
                    found_perspectives[category].append(term)
                else:
                    missing_perspectives[category].append(term)
        
        self.counts["perspective_analysis"] = {
            'found': found_perspectives,
            'missing': missing_perspectives
        }
        return self

    def detect_irrelevant_content(self):
        """Detect content that's irrelevant to the main topic."""
        # Split into sentences
        sentences = sent_tokenize(self.text)
        if len(sentences) < 2:
            self.counts["irrelevant_content"] = {"score": 0, "irrelevant_sentences": []}
            return self
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Calculate average similarity for each sentence
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # Identify irrelevant sentences (below threshold)
        threshold = np.mean(avg_similarities) - np.std(avg_similarities)
        irrelevant_indices = np.where(avg_similarities < threshold)[0]
        
        self.counts["irrelevant_content"] = {
            "score": float(np.mean(avg_similarities)),
            "irrelevant_sentences": [sentences[i] for i in irrelevant_indices]
        }
        return self

    def detect_misinterpretation(self):
        """Detect potential misinterpretations or misunderstandings."""
        misinterpretation_indicators = {
            'contradictory_statements': [],
            'unclear_references': [],
            'ambiguous_terms': []
        }
        
        sentences = sent_tokenize(self.text)
        
        # Check for contradictory statements
        for i, sent1 in enumerate(sentences):
            blob1 = TextBlob(sent1)
            for sent2 in sentences[i+1:]:
                blob2 = TextBlob(sent2)
                if blob1.sentiment.polarity * blob2.sentiment.polarity < 0:
                    if any(word in sent2 for word in blob1.noun_phrases):
                        misinterpretation_indicators['contradictory_statements'].append(
                            (sent1, sent2)
                        )
        
        # Check for unclear references
        pronouns = r'\b(it|this|that|these|those|they)\b'
        for sentence in sentences:
            if re.search(pronouns, sentence.lower()):
                if not any(noun in sentence for noun in TextBlob(sentence).noun_phrases):
                    misinterpretation_indicators['unclear_references'].append(sentence)
        
        # Check for ambiguous terms
        ambiguous_patterns = r'\b(thing|stuff|something|anything|everything|it)\b'
        for sentence in sentences:
            if re.search(ambiguous_patterns, sentence.lower()):
                misinterpretation_indicators['ambiguous_terms'].append(sentence)
        
        self.counts["misinterpretation"] = misinterpretation_indicators
        return self

    def measure_prompt_coverage(self):
        """Measure how well the text covers expected points or requirements."""
        # This is a placeholder implementation
        # For real implementation, we need:
        # 1. Original prompt/requirements
        # 2. Expected key points
        # 3. Scoring mechanism
        
        coverage_metrics = {
            'total_points_covered': 0,
            'missing_points': [],
            'coverage_score': 0.0,
            'suggestions': []
        }
        
        self.counts["prompt_coverage"] = coverage_metrics
        return self
