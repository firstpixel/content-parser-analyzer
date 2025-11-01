"""
Minimal yet functional Advanced NLP Analyzer implementing requested features.
This is a lightweight implementation intended to satisfy determinism, config, schema and tests.
Heavy NLP features are optional and guarded by try/except.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import Counter

CAUSAL_WORDS = {"because", "since", "therefore", "thus", "hence", "if", "then", "as", "so"}
MODAL_WORDS = {"can", "could", "might", "may", "should", "would", "will", "must", "shall"}
POSITIVE_WORDS = {"good", "great", "positive", "beneficial", "effective", "success", "improve", "support", "confident"}
NEGATIVE_WORDS = {"bad", "poor", "negative", "harmful", "fail", "issue", "risk", "doubt", "problem"}
VAGUE_TERMS = {"maybe", "perhaps", "some", "many", "various", "multiple", "often", "sometimes", "generally"}
HEDGE_TERMS = {"might", "could", "possibly", "seems", "appears", "suggests", "approximately", "around", "roughly"}
EMPATHY_WORDS = {"understand", "feel", "together", "share", "support", "care", "appreciate", "encourage"}
IMAGERY_WORDS = {"bright", "dark", "colorful", "vivid", "spark", "glow", "shadow", "fragrant", "melody", "texture"}
PERSUASIVE_WORDS = {"must", "need", "should", "required", "critical", "essential", "vital", "important"}
PURPOSE_WORDS = {"to", "ensure", "enable", "achieve", "in", "order", "so", "that"}
AMBIGUOUS_WORDS = {"bank", "charge", "fair", "right", "left", "light", "sound", "match", "pitch", "scale"}
HYPOTHETICAL_TERMS = {"if", "would", "could", "should", "suppose", "imagine", "assuming"}
SOURCE_WORDS = {"according", "cited", "reported", "source", "study", "research", "data"}
ENGAGEMENT_PRONOUNS = {"you", "your", "yours"}
INCLUSIVE_PRONOUNS = {"we", "our", "ours", "together"}
STOPWORDS = {"the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for", "with"}
CONTENT_WORD_THRESHOLD = 4

try:
    from .seed import centralize_seed
except ImportError:
    import random
    import numpy as np

    def centralize_seed(seed: int | None = None) -> int | None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return seed

import random
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None

# light tokenizer
import re

def simple_tokenize(text: str):
    return [t for t in re.findall(r"\w+", text.lower())]

@dataclass
class AnalyzerConfig:
    seed: int = 42
    windows: List[int] = field(default_factory=lambda: [100, 200, 300, 600])
    stride_fraction: float = 0.5
    light_mode: bool = True
    orthogonalize: bool = False
    factual_adapter: bool = False
    results_dir: str = "results"

class AdvancedNLPAnalyzer:
    def __init__(self, text: str, config: Optional[AnalyzerConfig] = None):
        self.text = text or ""
        self.config = config or AnalyzerConfig()
        self.seed = centralize_seed(self.config.seed)
        self.counts: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.tokens = simple_tokenize(self.text)
        self.sentences = self._split_sentences(self.text)
        self.domain_keywords = {tok for tok in self.tokens if len(tok) >= CONTENT_WORD_THRESHOLD}
        self.window_sizes = sorted({w for w in self.config.windows if w > 0})
        os.makedirs(self.config.results_dir, exist_ok=True)

        # placeholder for model versions
        self.model_versions = {
            'python': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        }

    def _version_hash(self, commit_sha: str = "UNKNOWN"):
        payload = json.dumps({
            'models': self.model_versions,
            'schema_version': '1.0',
            'commit_sha': commit_sha
        }, sort_keys=True).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _window_slices(self):
        n = len(self.tokens)
        if n == 0:
            return {}
        window_sizes = sorted({min(max(1, w), n) for w in self.window_sizes} | {n})
        slices: Dict[str, List[tuple]] = {}
        for size in window_sizes:
            stride = max(1, int(size * self.config.stride_fraction))
            ranges = []
            for start in range(0, max(1, n - size) + 1, stride):
                end = min(n, start + size)
                ranges.append((start, end))
                if end == n:
                    break
            if not ranges:
                ranges = [(0, n)]
            key = f"window_{size if size < n else 'full'}"
            slices[key] = ranges
        return slices

    def _compute_simple_metrics(self, token_list: List[str]) -> Dict[str, float]:
        if not token_list:
            return {'lexical_diversity': 0.0, 'avg_word_length': 0.0, 'vowel_ratio': 0.0}
        length = len(token_list)
        uniq = len(set(token_list))
        lexical_diversity = self._round(uniq / length)
        avg_word_len = sum(len(t) for t in token_list) / length
        avg_word_length = self._round(min(avg_word_len / 12.0, 1.0))
        vowels = sum(sum(1 for ch in t if ch in 'aeiou') for t in token_list)
        vowel_ratio = self._round(vowels / (length * 3))
        return {
            'lexical_diversity': lexical_diversity,
            'avg_word_length': avg_word_length,
            'vowel_ratio': vowel_ratio,
        }

    def _flatten_metrics(self, nested: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for family, values in nested.items():
            if family == 'advanced_features':
                flat.update(values)
                continue
            for key, value in values.items():
                new_key = key if family != 'core_metrics' else f"core_{key}"
                if new_key in flat:
                    new_key = f"{family}_{key}"
                flat[new_key] = value
        if 'lexical_diversity' not in flat and 'core_lexical_diversity' in flat:
            flat['lexical_diversity'] = flat['core_lexical_diversity']
        return flat

    def _aggregate_windows(self, slices: Dict[str, List[Any]]):
        window_entries: Dict[str, List[Dict[str, Any]]] = {}
        window_matrices: Dict[str, List[List[float]]] = {}
        for name, ranges in slices.items():
            entries, matrices = [], []
            for start, end in ranges:
                window_tokens = self.tokens[start:end]
                window_text = " ".join(window_tokens)
                metrics = self._flatten_metrics(self._compute_family_metrics(window_text))
                metrics['window_start'] = start
                metrics['window_end'] = end
                entries.append(metrics)
                matrices.append([
                    metrics.get('lexical_diversity', 0.0),
                    metrics.get('core_avg_word_length', metrics.get('avg_word_length', 0.0)),
                    metrics.get('core_vowel_ratio', metrics.get('vowel_ratio', 0.0))
                ])
            window_entries[name] = entries
            window_matrices[name] = matrices
        return window_entries, window_matrices

    def _build_temporal_trends(self, window_entries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        tracked = ['logical_coherence', 'emotional_valence', 'persuasiveness', 'factual_density', 'lexical_diversity']
        trends: Dict[str, Dict[str, Any]] = {}
        for name, entries in window_entries.items():
            metrics: Dict[str, Any] = {}
            for metric in tracked:
                values = [entry.get(metric) for entry in entries if isinstance(entry.get(metric), (int, float))]
                if not values:
                    continue
                metric_trend = {
                    'mean': self._round(float(np.mean(values))),
                    'std': self._round(float(np.std(values))) if len(values) > 1 else 0.0,
                    'min': self._round(min(values)),
                    'max': self._round(max(values)),
                    'trend': 'increasing' if values[-1] > values[0] else ('decreasing' if values[-1] < values[0] else 'stable'),
                    'volatility': self._round(float(np.std(np.diff(values)))) if len(values) > 2 else 0.0,
                }
                metrics[metric] = metric_trend
            if metrics:
                trends[name] = metrics
        return trends

    def _build_metadata(self, window_entries: Dict[str, List[Dict[str, Any]]], start_time: float, commit_sha: str) -> Dict[str, Any]:
        window_sizes = {}
        for key in window_entries:
            try:
                size_part = key.split('_', 1)[1]
                window_sizes[key] = len(window_entries[key])
            except Exception:
                window_sizes[key] = len(window_entries[key])
        return {
            'seed': self.seed,
            'version_hash': self._version_hash(commit_sha),
            'schema_version': '1.0',
            'analysis_time_seconds': round(time.time() - start_time, 3),
            'total_tokens': len(self.tokens),
            'window_sizes': list(window_sizes.keys()),
            'total_windows': window_sizes,
        }

    def run_complete_analysis(self, commit_sha: str = "UNKNOWN") -> Dict[str, Any]:
        start_time = time.time()
        slices = self._window_slices()
        window_entries, window_matrices = self._aggregate_windows(slices)
        full_metrics = self._flatten_metrics(self._compute_family_metrics())
        temporal_trends = self._build_temporal_trends(window_entries)
        all_rows = [row for rows in window_matrices.values() for row in rows if any(row)]
        orthogonal = self._orthogonalize_matrix(all_rows) if self.config.orthogonalize and all_rows else None
        attributions = {'top_tokens': sorted(self.domain_keywords, key=lambda t: (-len(t), t))[:10]}
        metadata = self._build_metadata(window_entries, start_time, commit_sha)
        output = {
            'full_text_analysis': full_metrics,
            'window_analyses': window_entries,
            'temporal_trends': temporal_trends,
            'metadata': metadata,
            'explainability': attributions,
        }
        if orthogonal:
            output['orthogonalization'] = orthogonal
        out_path = os.path.join(self.config.results_dir, f"analysis_{self.seed}.json")
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(output, fh, indent=2)
        attr_path = os.path.join(self.config.results_dir, f"attributions_{self.seed}.json")
        with open(attr_path, 'w', encoding='utf-8') as fh:
            json.dump(attributions, fh, indent=2)
        self.counts['advanced_nlp_analysis'] = full_metrics
        self.counts['advanced_nlp_metadata'] = metadata
        self.metadata = output
        return output

    def analyze_temporal_windows(self) -> Dict[str, Any]:
        return self.run_complete_analysis(commit_sha="TEMPORAL_ANALYSIS")

    def analyze_logical_coherence(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 1.0 if sentences else 0.0
        overlaps = []
        for i in range(len(sentences) - 1):
            s1 = set(simple_tokenize(sentences[i]))
            s2 = set(simple_tokenize(sentences[i + 1]))
            if not s1 or not s2:
                overlaps.append(0.0)
            else:
                overlaps.append(len(s1 & s2) / len(s1 | s2))
        return self._round(float(np.mean(overlaps)))

    def analyze_causal_density(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        causal = sum(1 for token in tokens if token in CAUSAL_WORDS)
        return self._safe_ratio(causal, len(tokens))

    def analyze_argumentation_entropy(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        premises = sum(1 for token in tokens if token in {"because", "since", "as"})
        conclusions = sum(1 for token in tokens if token in {"therefore", "thus", "hence", "so"})
        total = premises + conclusions
        if total == 0:
            return 0.0
        balance = 1 - abs(premises - conclusions) / total
        return self._round(balance)

    def analyze_contradiction_ratio(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        contradictions = sum(1 for token in tokens if token in {"however", "but", "although", "nevertheless"})
        return self._safe_ratio(contradictions, len(tokens))

    def analyze_inferential_depth(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        depth_markers = sum(1 for token in tokens if token in {"if", "unless", "therefore", "implies"})
        return self._safe_ratio(depth_markers, len(tokens))

    def analyze_domain_consistency(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        if not tokens:
            return 0.0
        hits = sum(1 for token in tokens if token in self.domain_keywords)
        return self._safe_ratio(hits, len(tokens))

    def analyze_referential_stability(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        pronouns = sum(1 for token in tokens if token in {"he", "she", "they", "it", "this", "that"})
        return self._round(1.0 - self._safe_ratio(pronouns, len(tokens), scale=2.5))

    def analyze_temporal_consistency(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        if not tokens:
            return 0.0
        past = sum(1 for token in tokens if token.endswith('ed'))
        present = sum(1 for token in tokens if token.endswith('ing'))
        total = past + present
        if total == 0:
            return 0.0
        balance = 1 - abs(past - present) / total
        return self._round(balance)

    def analyze_modality_balance(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        if not tokens:
            return 0.0
        ratio = sum(1 for token in tokens if token in MODAL_WORDS) / len(tokens)
        return self._round(1 - min(1.0, abs(ratio - 0.2) / 0.2))

    def analyze_precision_index(self, text: Optional[str] = None) -> float:
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text or self.text)
        tokens = self._get_tokens(text)
        precise_terms = sum(1 for token in tokens if token in {"exactly", "precisely", "specifically", "accurately"})
        return self._safe_ratio(len(numbers) + precise_terms, max(1, len(tokens)))

    def analyze_goal_clarity(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        clarity_terms = sum(1 for token in tokens if token in {"goal", "objective", "aim", "target", "purpose", "ensure"})
        return self._safe_ratio(clarity_terms, len(tokens))

    def analyze_focus_retention(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        if not tokens:
            return 0.0
        first_sentence_tokens = set(self._get_tokens(self._split_sentences(text)[0])) if self._split_sentences(text) else set()
        aligned = sum(1 for token in tokens if token in first_sentence_tokens)
        return self._safe_ratio(aligned, len(tokens))

    def analyze_persuasiveness(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        persuasive_terms = sum(1 for token in tokens if token in PERSUASIVE_WORDS)
        return self._safe_ratio(persuasive_terms, len(tokens))

    def analyze_commitment(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        hedges = sum(1 for token in tokens if token in HEDGE_TERMS)
        return self._round(1.0 - self._safe_ratio(hedges, len(tokens), scale=3.0))

    def analyze_teleology(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        sequences = sum(1 for i in range(len(tokens) - 1) if tokens[i] == "to" and tokens[i + 1] in PURPOSE_WORDS)
        return self._safe_ratio(sequences, len(tokens))

    def analyze_emotional_valence(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
        negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
        total = positive + negative
        if total == 0:
            return 0.5
        score = (positive - negative + total) / (2 * total)
        return self._round(score)

    def analyze_arousal(self, text: Optional[str] = None) -> float:
        text_segment = text or self.text
        markers = text_segment.count('!') + sum(1 for token in self._get_tokens(text) if token in {"exciting", "thrilling", "urgent", "intense"})
        return self._safe_ratio(markers, len(self._get_tokens(text)) or len(text_segment))

    def analyze_empathy_score(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        empathy_terms = sum(1 for token in tokens if token in EMPATHY_WORDS or token in INCLUSIVE_PRONOUNS)
        return self._safe_ratio(empathy_terms, len(tokens))

    def analyze_emotional_volatility(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        polarities = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
        return self._round(float(np.std(polarities)))

    def analyze_symbolic_resonance(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        imagery_terms = sum(1 for token in tokens if token in IMAGERY_WORDS)
        return self._safe_ratio(imagery_terms, len(tokens))

    def analyze_speech_act_ratio(self, text: Optional[str] = None) -> Dict[str, float]:
        tokens = self._get_tokens(text)
        if not tokens:
            return {'assertive': 0.0, 'directive': 0.0, 'expressive': 0.0}
        assertive = sum(1 for token in tokens if token in {"state", "explain", "report", "describe"})
        directive = sum(1 for token in tokens if token in {"ask", "request", "should", "must", "please"})
        expressive = sum(1 for token in tokens if token in {"thanks", "sorry", "congrats", "love", "feel"})
        total = max(1, assertive + directive + expressive)
        return {
            'assertive': self._round(assertive / total),
            'directive': self._round(directive / total),
            'expressive': self._round(expressive / total),
        }

    def analyze_dialogue_coherence(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        questions = [i for i, sentence in enumerate(sentences) if sentence.strip().endswith('?')]
        if not questions:
            return 0.0
        followups = sum(1 for idx in questions if idx + 1 < len(sentences) and not sentences[idx + 1].strip().endswith('?'))
        return self._round(followups / len(questions))

    def analyze_pragmatic_truth(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        informative = sum(1 for token in tokens if token not in STOPWORDS and len(token) > 3)
        return self._safe_ratio(informative, len(tokens))

    def analyze_social_tone(self, text: Optional[str] = None) -> Dict[str, float]:
        tokens = self._get_tokens(text)
        if not tokens:
            return {'politeness': 0.0, 'cooperation': 0.0, 'dominance': 0.0}
        politeness = sum(1 for token in tokens if token in {"please", "thank", "appreciate", "welcome"})
        cooperation = sum(1 for token in tokens if token in {"together", "share", "collaborate", "support"})
        dominance = sum(1 for token in tokens if token in {"must", "need", "command", "insist"})
        total = max(1, politeness + cooperation + dominance)
        return {
            'politeness': self._round(politeness / total),
            'cooperation': self._round(cooperation / total),
            'dominance': self._round(dominance / total),
        }

    def analyze_engagement_index(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        engagement = sum(1 for token in tokens if token in ENGAGEMENT_PRONOUNS)
        return self._safe_ratio(engagement, len(tokens))

    def analyze_rhythm_variance(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0
        lengths = [max(1, len(simple_tokenize(sentence))) for sentence in sentences]
        mean_length = float(np.mean(lengths))
        if mean_length == 0:
            return 0.0
        return self._round(float(np.std(lengths) / mean_length))

    def analyze_lexical_diversity(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        return self._round(len(set(tokens)) / max(1, len(tokens)))

    def analyze_imagery_density(self, text: Optional[str] = None) -> float:
        return self.analyze_symbolic_resonance(text)

    def analyze_symmetry_index(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        halves = np.array_split([len(simple_tokenize(s)) for s in sentences], 2)
        diff = abs(halves[0].mean() - halves[1].mean())
        baseline = max(halves[0].mean(), halves[1].mean(), 1)
        return self._round(1 - min(1.0, diff / baseline))

    def analyze_surprise_novelty(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        if not tokens:
            return 0.0
        token_counts = Counter(tokens)
        hapax = sum(1 for _, count in token_counts.items() if count == 1)
        return self._safe_ratio(hapax, len(tokens))

    def analyze_factual_density(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0
        factual_markers = sum(1 for sentence in sentences if any(marker in sentence.lower() for marker in ["according to", "study", "research", "data", "evidence"]))
        return self._safe_ratio(factual_markers, len(sentences))

    def analyze_fact_precision(self, text: Optional[str] = None) -> float:
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text or self.text)
        tokens = self._get_tokens(text)
        proper_nouns = sum(1 for token in tokens if token.istitle())
        return self._safe_ratio(len(numbers) + proper_nouns, max(1, len(tokens)))

    def analyze_evidence_linkage(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        citations = sum(1 for token in tokens if token in {"according", "cited", "reported", "study", "research"})
        return self._safe_ratio(citations, len(tokens))

    def analyze_truth_confidence(self, text: Optional[str] = None) -> float:
        return self._round(1.0 - self.analyze_vagueness(text))

    def analyze_source_diversity(self, text: Optional[str] = None) -> float:
        lower = (text or self.text).lower()
        sources = re.findall(r"\b(according to|reported by|source|study|research)\s+([A-Z][a-zA-Z]+)", lower)
        unique = len({match[1] for match in sources})
        return self._safe_ratio(unique, len(self._split_sentences(text)) or len(self.tokens))

    def analyze_ambiguity_entropy(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        ambiguous = sum(1 for token in tokens if token in AMBIGUOUS_WORDS)
        return self._safe_ratio(ambiguous, len(tokens))

    def analyze_vagueness(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        vague_terms = sum(1 for token in tokens if token in VAGUE_TERMS)
        return self._safe_ratio(vague_terms, len(tokens))

    def analyze_cognitive_dissonance(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
        flips = sum(1 for i in range(len(sentiments) - 1) if sentiments[i] * sentiments[i + 1] < 0)
        return self._safe_ratio(flips, len(sentiments) - 1)

    def analyze_hypothetical_load(self, text: Optional[str] = None) -> float:
        tokens = self._get_tokens(text)
        hypothetical = sum(1 for token in tokens if token in HYPOTHETICAL_TERMS)
        return self._safe_ratio(hypothetical, len(tokens))

    def analyze_certainty_oscillation(self, text: Optional[str] = None) -> float:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        certainty_series = []
        for sentence in sentences:
            stokens = simple_tokenize(sentence)
            certain = sum(1 for token in stokens if token in {"must", "will", "always", "never", "definitely"})
            uncertain = sum(1 for token in stokens if token in HEDGE_TERMS or token in VAGUE_TERMS)
            total = max(1, certain + uncertain)
            certainty_series.append(certain / total)
        return self._round(float(np.std(certainty_series)))

    def extract_named_entities_advanced(self, text: Optional[str] = None) -> Dict[str, Any]:
        segment = text or self.text
        nlp = _ensure_spacy_model()
        if nlp:
            doc = nlp(segment)
            counts = Counter(ent.label_ for ent in doc.ents)
            total = sum(counts.values())
            density = self._safe_ratio(total, len(self._get_tokens(segment)))
            return {
                'total_entities': int(total),
                'entity_counts': dict(counts),
                'entity_density': density,
            }
        matches = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", segment)
        counts = Counter("PROPER" for _ in matches)
        density = self._safe_ratio(len(matches), len(self._get_tokens(segment)))
        return {
            'total_entities': len(matches),
            'entity_counts': dict(counts),
            'entity_density': density,
        }

    def extract_relationships(self, text: Optional[str] = None) -> Dict[str, Any]:
        segment = text or self.text
        sentences = self._split_sentences(segment)
        patterns = [
            r'(?P<subject>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+(?P<verb>works at|met|founded|leads|joined|manages)\s+(?P<object>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)',
            r'(?P<subject>[A-Z][a-zA-Z]+)\s+(?P<verb>collaborated with|partnered with|supports)\s+(?P<object>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)',
        ]
        relationships = []
        for sentence in sentences:
            for pattern in patterns:
                for match in re.finditer(pattern, sentence):
                    relationships.append({
                        'subject': match.group('subject'),
                        'verb': match.group('verb'),
                        'object': match.group('object'),
                        'sentence': sentence.strip(),
                    })
        return {
            'relationship_count': len(relationships),
            'relationships': relationships[:25],
        }

    def extract_word_sense_disambiguation(self, text: Optional[str] = None) -> Dict[str, Any]:
        tokens = self._get_tokens(text)
        ambiguous = [token for token in tokens if token in AMBIGUOUS_WORDS]
        specificity = self._round(1.0 - self._safe_ratio(len(ambiguous), len(tokens)))
        return {
            'ambiguous_terms': len(ambiguous),
            'unique_ambiguous_terms': len(set(ambiguous)),
            'avg_word_specificity': specificity,
        }

    def extract_information_extraction(self, text: Optional[str] = None) -> Dict[str, Any]:
        segment = text or self.text
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", segment)
        urls = re.findall(r"https?://[^\s]+", segment)
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", segment)
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", segment)
        return {
            'email_count': len(emails),
            'url_count': len(urls),
            'date_count': len(dates),
            'number_count': len(numbers),
        }

    def get_counts(self) -> Dict[str, Any]:
        return self.counts

# Backwards-compatible wrapper expected by repo
class ContentParserAnalyzer:
    def __init__(self, text: str, config: Optional[AnalyzerConfig] = None):
        self.text = text
        self.advanced_nlp_analyzer = AdvancedNLPAnalyzer(text, config=config)