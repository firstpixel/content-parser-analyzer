import langdetect
import text2emotion as te
import spacy
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from utils.base_parser import BaseParser
import re

class NLPAnalyzer(BaseParser):
    def analyze_sentiment(self):
        """Analyze sentiment polarity of text."""  # Implemented
        if hasattr(self, "code_parser"):  
            clean_text = self.code_parser.remove_code_blocks().text  
        else:
            clean_text = self.text  # Fallback in case `code_parser` isn't available

        self.counts["sentiment"] = TextBlob(clean_text).sentiment.polarity
        return self

    def detect_language(self):
        """Detect the primary language of text."""  # Implemented
        self.counts["language"] = langdetect.detect(self.text)
        return self

    def detect_emotional_tone(self):
        """Detect emotional tone of text."""  # Implemented
        emotions = te.get_emotion(self.text)
        self.counts["dominant_emotion"] = max(emotions, key=emotions.get)
        return self

    def detect_sarcasm_humor(self):
        """Detect sarcasm and humor using sentiment contrast and patterns."""  # Implemented
        sentences = sent_tokenize(self.text)
        sarcasm_indicators = {
            'sentiment_contrast': 0,
            'exaggeration': 0,
            'rhetorical_questions': 0,
            'ironic_interjections': 0,
            'laughter_indicators': 0
        }

        # Expanded sets of keywords and phrases for improved detection.
        exaggeration_keywords = [
            'absolutely', 'definitely', 'totally', 'obviously',
            'unbelievably', 'incredibly', 'ridiculously'
        ]
        rhetorical_question_keywords = [
            'really', 'seriously', 'right', 'indeed', 'eh'
        ]
        ironic_interjections = [
            'oh, really', 'yeah, right', 'sure, why not', 'of course', 'as if'
        ]
        laughter_keywords = [
            'lol', 'haha', 'hehe', 'lmao', 'rofl'
        ]

        prev_sentiment = None
        for sentence in sentences:
            # Calculate sentiment for the current sentence.
            sentiment = TextBlob(sentence).sentiment.polarity

            # Check for sentiment contrast with the previous sentence.
            if prev_sentiment is not None and abs(sentiment - prev_sentiment) > 0.5:
                sarcasm_indicators['sentiment_contrast'] += 1
            prev_sentiment = sentiment

            # Check for exaggeration using an expanded set of keywords.
            for word in exaggeration_keywords:
                if re.search(r'\b' + re.escape(word) + r'\b', sentence.lower()):
                    sarcasm_indicators['exaggeration'] += 1

            # Check for rhetorical questions by punctuation and key phrases.
            if '?' in sentence:
                for word in rhetorical_question_keywords:
                    if re.search(r'\b' + re.escape(word) + r'\b', sentence.lower()):
                        sarcasm_indicators['rhetorical_questions'] += 1
                        break  # Only count once per sentence.
            # Additional check: multiple punctuation marks (e.g., "?!", "!!!").
            if re.search(r'[!?]{2,}', sentence):
                sarcasm_indicators['rhetorical_questions'] += 1

            # Check for ironic interjections.
            for phrase in ironic_interjections:
                if phrase in sentence.lower():
                    sarcasm_indicators['ironic_interjections'] += 1

            # Check for laughter indicators.
            for laugh in laughter_keywords:
                if re.search(r'\b' + re.escape(laugh) + r'\b', sentence.lower()):
                    sarcasm_indicators['laughter_indicators'] += 1

        self.counts["sarcasm_humor"] = sarcasm_indicators
        return self

    def detect_vagueness(self):
        """Detect vague language and imprecise statements."""  # Implemented
        # Define a comprehensive dictionary of vague words and phrases.
        vague_words = {
            'quantifiers': ['some', 'many', 'few', 'several', 'various', 'numerous'],
            'modifiers': ['probably', 'maybe', 'possibly', 'kind of', 'sort of', 'somewhat'],
            'generalizations': ['always', 'never', 'everyone', 'nobody', 'everything', 'all', 'none'],
            'approximators': ['about', 'approximately', 'roughly', 'around', 'circa'],
            'uncertain_modifiers': ['likely', 'presumably', 'apparently', 'arguably', 'reportedly'],
            'vague_expressions': ['in a way', 'to some extent', 'in some cases', 'in certain situations']
        }

        # Normalize the text for case-insensitive matching.
        text_lower = self.text.lower() if isinstance(self.text, str) else ""
        word_list = text_lower.split()
        word_count = len(word_list)
        vagueness_score = 0
        vague_instances = {category: {} for category in vague_words}

        # Iterate over each category and term.
        for category, terms in vague_words.items():
            for term in terms:
                # If the term contains spaces, count it as a phrase.
                if " " in term:
                    count = len(re.findall(re.escape(term), text_lower))
                else:
                    # Use word boundaries for single words.
                    count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                if count > 0:
                    vague_instances[category][term] = count
                    vagueness_score += count

        # Calculate vagueness as a percentage of the total word count.
        vagueness_percentage = (vagueness_score / word_count * 100) if word_count > 0 else 0

        # Store detailed vagueness analysis in self.counts.
        self.counts["vagueness"] = {
            'score': vagueness_percentage,
            'total_vague_words': vagueness_score,
            'word_count': word_count,
            'instances': vague_instances
        }
        return self

    def detect_hedging(self):
        """Detect hedging language that reduces commitment."""  # Implemented
        hedging_patterns = {
            'epistemic': ['might', 'may', 'could', 'presumably', 'probably'],
            'approximators': ['approximately', 'roughly', 'about', 'around', 'estimate'],
            'shields': ['i think', 'it seems', 'it appears', 'suggests that', 'indicates that'],
            'limiters': ['somewhat', 'partly', 'relatively', 'generally', 'usually']
        }
        
        hedging_counts = {category: 0 for category in hedging_patterns}
        for category, patterns in hedging_patterns.items():
            for pattern in patterns:
                count = len(re.findall(r'\b' + pattern + r'\b', self.text.lower()))
                hedging_counts[category] += count
        
        total_hedges = sum(hedging_counts.values())
        word_count = len(self.text.split())
        
        self.counts["hedging"] = {
            'score': (total_hedges / word_count * 100) if word_count > 0 else 0,
            'breakdown': hedging_counts
        }
        return self

    def detect_bias(self):
        """Detect biased language and statements."""  # Implemented
        bias_keywords = ["obviously", "clearly", "everyone knows"]
        self.counts["bias_detected"] = sum(1 for word in bias_keywords if word in self.text.lower())
        return self

    def detect_harmful_content(self):
        """
        Detect potentially harmful content using a keyword-based heuristic.

        This method scans the text for a set of predefined harmful words and phrases
        that are commonly associated with hate speech, incitement to violence, or other
        harmful content. Each category of terms is assigned a weight that reflects its severity.
        The method calculates a severity score based on the frequency of these terms and records
        the detected terms with their counts.

        Returns:
            self: to allow method chaining.
        """
        # Ensure we are working with lowercase text for case-insensitive matching.
        text_lower = self.text.lower() if isinstance(self.text, str) else ""

        # Define a comprehensive dictionary of harmful terms categorized by type.
        harmful_terms = {
            "hate_speech": [
                "hate", "hateful", "bigot", "bigotry", "disgusting", "despicable", "detestable"
            ],
            "racial_slurs": [
                "nigger", "chink", "spic", "kike", "faggot", "retard", "tranny", "cunt", "whore"
            ],
            "violent_terms": [
                "kill", "murder", "slaughter", "exterminate", "assassinate", "massacre", "execute"
            ],
            "extremist_terms": [
                "terrorist", "jihadi", "white supremacist", "neo-nazi", "extremist", "radical"
            ],
            "explicit_violence": [
                "bomb", "explode", "assault", "shoot", "stab", "attack"
            ]
        }

        # Dictionary to accumulate detected terms and their counts.
        found_terms = {}
        total_score = 0

        # Define weights for each category to reflect severity.
        category_weights = {
            "hate_speech": 1,
            "racial_slurs": 3,
            "violent_terms": 2,
            "extremist_terms": 2,
            "explicit_violence": 2
        }

        # Scan text for each harmful term and calculate weighted score.
        for category, terms in harmful_terms.items():
            for term in terms:
                # Use regex word boundaries to match exact words.
                matches = re.findall(r'\b' + re.escape(term) + r'\b', text_lower)
                if matches:
                    count = len(matches)
                    if category not in found_terms:
                        found_terms[category] = {}
                    found_terms[category][term] = count
                    total_score += count * category_weights.get(category, 1)

        # Record the computed score and details in the counts dictionary.
        self.counts["harmful_content"] = {
            "score": total_score,
            "found_terms": found_terms,
            "description": "Score is computed as a weighted sum of occurrences for detected harmful terms."
        }
        return self

    def detect_policy_violations(self):
        """
        Detect potential content policy violations using a heuristic keyword-based approach.

        This method scans the text for a wide range of terms and phrases associated with
        policy violations such as harassment, explicit sexual content, illegal activities,
        self-harm promotion, and spam. For each category, a weighted severity score is computed
        based on the frequency of occurrence of the respective terms. The results, including
        a detailed breakdown of detected terms and their counts, are stored in the counts
        dictionary under "policy_violations".

        Returns:
            self: to allow method chaining.
        """
        # Normalize text to lowercase to ensure case-insensitive matching.
        text_lower = self.text.lower() if isinstance(self.text, str) else ""

        # Define a comprehensive dictionary of policy-violating terms categorized by type.
        policy_violation_terms = {
            "harassment": [
                "insult", "idiot", "moron", "loser", "bastard", "asshole", "stupid", "dumb",
                "scum", "trash", "sucker", "fuck you", "fuck off"
            ],
            "explicit_content": [
                "porn", "sex", "xxx", "adult content", "explicit", "nude", "naked", "graphic sexual",
                "erotic", "nsfw", "hardcore", "smut"
            ],
            "illegal_activities": [
                "drug trafficking", "smuggle", "illegal arms", "money laundering", "fraud", "scam",
                "counterfeit", "piracy", "hacking", "cybercrime", "extortion"
            ],
            "self_harm": [
                "self harm", "suicide", "cutting", "self-injury", "kill myself", "self destruct",
                "self-destruction", "overdose"
            ],
            "spam_and_malware": [
                "click here", "subscribe", "buy now", "free offer", "limited time", "risk free",
                "act now", "guaranteed", "win money", "prize", "winner", "cheap", "discount"
            ]
        }

        # Initialize dictionary to accumulate detected terms and total weighted score.
        found_terms = {}
        total_score = 0

        # Define severity weights for each category.
        category_weights = {
            "harassment": 2,
            "explicit_content": 2,
            "illegal_activities": 3,
            "self_harm": 2,
            "spam_and_malware": 1,
        }


        # Scan text for each term and accumulate weighted counts.
        for category, terms in policy_violation_terms.items():
            for term in terms:
                # Use regex word boundaries to match whole words or phrases.
                matches = re.findall(r'\b' + re.escape(term) + r'\b', text_lower)
                if matches:
                    count = len(matches)
                    if category not in found_terms:
                        found_terms[category] = {}
                    found_terms[category][term] = count
                    total_score += count * category_weights.get(category, 1)

        # Store detailed results and the weighted score in the counts dictionary.
        self.counts["policy_violations"] = {
            "score": total_score,
            "found_terms": found_terms,
            "description": "Weighted score computed from the frequency of potential policy-violating keywords."
        }
        return self


    # def detect_misinformation(self):
    #     """Detect potential misinformation."""  # Not Implemented
    #     self.counts["misinformation"] = "not_implemented"
    #     return self

    # def detect_plagiarism(self):
    #     """Detect potential plagiarism."""  # Not Implemented
    #     self.counts["plagiarism"] = "not_checked"
    #     return self

    def extract_named_entities(self):
        """Extract named entities using spaCy."""  # Implemented
        try:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(self.text)
            entities = {}
            
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            self.text = entities
        except Exception as e:
            self.text = {"error": "Spacy model not available: " + str(e)}
        return self

    def assign_credibility_score(self):
        """Assign credibility score based on multiple factors."""  # Implemented
        score = 100
        score -= self.counts.get("bias_detected", 0) * 5
        score -= self.counts.get("hallucinations_detected", 0) * 15
        score += self.counts.get("factual_claims", 0) * 5
        self.counts["credibility_score"] = max(0, min(100, round(score)))
        return self
    
    def detect_hallucinations(self):
        """Detect potential hallucinations using fact checking and consistency."""
        hallucination_indicators = {
            'inconsistencies': [],
            'unsupported_claims': [],
            'confidence_score': 0.0
        }
        
        sentences = sent_tokenize(self.text)
        facts = set()
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in ['is', 'are', 'was', 'were']):
                facts.add(sentence)
        
        for fact1 in facts:
            blob1 = TextBlob(fact1)
            for fact2 in facts:
                if fact1 != fact2:
                    blob2 = TextBlob(fact2)
                    if blob1.sentiment.polarity * blob2.sentiment.polarity < 0:
                        hallucination_indicators['inconsistencies'].append((fact1, fact2))
        
        definitive_markers = ['definitely', 'absolutely', 'certainly', 'always', 'never']
        for sentence in sentences:
            if any(marker in sentence.lower() for marker in definitive_markers):
                hallucination_indicators['unsupported_claims'].append(sentence)
        
        confidence_score = 100
        confidence_score -= len(hallucination_indicators['inconsistencies']) * 10
        confidence_score -= len(hallucination_indicators['unsupported_claims']) * 5
        hallucination_indicators['confidence_score'] = max(0, confidence_score)
        
        self.counts["hallucinations"] = hallucination_indicators
        return self
