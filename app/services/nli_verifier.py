import re
import math
from typing import List, Tuple, Optional, Set
from app.schemas.payload import EvidenceMatch, NLIScores, ClaimEvaluation
from app.core.config import Settings, get_settings

class NLIVerifier:
    """
    Evaluates (claim, evidence) pairs using Natural Language Inference (NLI).
    Computes entailment, neutral, and contradiction probabilities.
    Supports DeBERTa cross-encoders with a calibrated heuristic fallback.
    """

    NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "cannot", "hardly", "barely", "refused", "denied"}
    
    ANTONYM_PAIRS = [
        ("increase", "decrease"), ("rise", "fall"), ("grow", "shrink"),
        ("true", "false"), ("correct", "incorrect"), ("valid", "invalid"),
        ("alive", "dead"), ("success", "failure"), ("win", "lose"),
        ("before", "after"), ("above", "below"), ("first", "last")
    ]

    WORD_NUMBERS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "first": "1st", "second": "2nd", "third": "3rd"
    }

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "in", "on", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "to", "from", "up",
        "down", "of", "and", "or", "but", "while", "that", "this", "these", "those"
    }

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None
        self._cross_encoder_available = False
        self._init_model()

    def _init_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            device = self.settings.DEVICE
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
            self._model_name = self.settings.NLI_MODEL_NAME

            # Load cross-encoder if weights are available locally
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name, local_files_only=True).to(self._device)
            self._model.eval()
            self._cross_encoder_available = True
        except Exception:
            self._cross_encoder_available = False

    def verify_claim(
        self,
        claim_id: int,
        claim: str,
        evidence_matches: List[EvidenceMatch]
    ) -> ClaimEvaluation:
        if not evidence_matches:
            return ClaimEvaluation(
                claim_id=claim_id,
                claim=claim,
                verdict="NEUTRAL",
                confidence=0.90,
                is_grounded=False,
                evidence=None,
                all_evidence=[],
                scores=NLIScores(entailment=0.05, neutral=0.90, contradiction=0.05),
                reasoning="No relevant evidence was found in the retrieved context documents."
            )

        best_evidence = evidence_matches[0]
        
        # Check neural cross-encoder if loaded
        if self._cross_encoder_available and self._model is not None:
            scores, best_ev = self._evaluate_neural(claim, evidence_matches)
        else:
            scores, best_ev = self._evaluate_heuristic(claim, evidence_matches)

        # Classify verdict according to calibrated thresholds
        if scores.contradiction >= self.settings.CONTRADICTION_THRESHOLD:
            verdict = "CONTRADICTION"
            confidence = scores.contradiction
            is_grounded = False
            reasoning = f"Contradiction detected against evidence snippet: '{best_ev.text}'"
        elif scores.entailment >= self.settings.ENTAILMENT_THRESHOLD:
            verdict = "ENTAILMENT"
            confidence = scores.entailment
            is_grounded = True
            reasoning = f"Directly grounded and supported by evidence from doc '{best_ev.doc_id}'."
        else:
            verdict = "NEUTRAL"
            confidence = scores.neutral
            is_grounded = False
            reasoning = "Evidence does not sufficiently entail or disprove this claim (unsupported/hallucinated)."

        return ClaimEvaluation(
            claim_id=claim_id,
            claim=claim,
            verdict=verdict,
            confidence=round(confidence, 4),
            is_grounded=is_grounded,
            evidence=best_ev,
            all_evidence=evidence_matches,
            scores=scores,
            reasoning=reasoning
        )

    def _extract_tokens_and_numbers(self, text: str) -> Tuple[Set[str], Set[str], Set[str]]:
        # Normalize hyphens so "8-day" becomes "8" and "day"
        normalized = text.replace("-", " ")
        raw_words = re.findall(r'\b[a-zA-Z0-9\.]+\b', normalized)
        
        tokens = set()
        numbers = set()
        
        for w in raw_words:
            w_lower = w.lower()
            tokens.add(w_lower)
            # Check digits
            if re.match(r'^\d+(?:\.\d+)?$', w):
                numbers.add(w)
            elif w_lower in self.WORD_NUMBERS:
                mapped = self.WORD_NUMBERS[w_lower]
                tokens.add(mapped)
                if mapped.isdigit():
                    numbers.add(mapped)
        
        content_tokens = {t for t in tokens if t not in self.STOP_WORDS}
        return tokens, content_tokens, numbers

    def _extract_entities(self, text: str) -> Set[str]:
        words = text.split()
        ents = set()
        months_days = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
        common_starters = {"the", "a", "an", "this", "that", "these", "those", "in", "on", "at", "for", "from", "after", "before", "while", "when", "as", "four", "all", "here"}
        
        for idx, w in enumerate(words):
            clean = re.sub(r'^[^\w]+|[^\w]+$', '', w)
            if not clean or len(clean) <= 1:
                continue
            if idx == 0 and clean.lower() in common_starters:
                continue
            if clean[0].isupper() and clean.lower() not in months_days and clean.lower() not in self.STOP_WORDS:
                ents.add(clean.lower())
        return ents

    def _evaluate_heuristic(
        self,
        claim: str,
        evidence_matches: List[EvidenceMatch]
    ) -> Tuple[NLIScores, EvidenceMatch]:
        best_ev = evidence_matches[0]
        max_entailment = 0.0
        max_contradiction = 0.0

        claim_tokens, claim_content, claim_numbers = self._extract_tokens_and_numbers(claim)
        claim_entities = self._extract_entities(claim)
        claim_has_negation = bool(claim_tokens & self.NEGATION_WORDS)

        for ev in evidence_matches:
            ev_tokens, ev_content, ev_numbers = self._extract_tokens_and_numbers(ev.text)
            ev_entities = self._extract_entities(ev.text)
            ev_has_negation = bool(ev_tokens & self.NEGATION_WORDS)

            # Overlap metrics
            intersection = claim_tokens & ev_tokens
            content_intersection = claim_content & ev_content
            overlap_ratio = len(content_intersection) / max(1, len(claim_content))
            
            contradiction_signals = 0

            # 1. Direct Antonym Contradiction
            for w1, w2 in self.ANTONYM_PAIRS:
                w1_in_claim = any(t.startswith(w1) for t in claim_tokens)
                w2_in_ev = any(t.startswith(w2) for t in ev_tokens)
                w2_in_claim = any(t.startswith(w2) for t in claim_tokens)
                w1_in_ev = any(t.startswith(w1) for t in ev_tokens)
                if (w1_in_claim and w2_in_ev) or (w2_in_claim and w1_in_ev):
                    contradiction_signals += 3

            # 2. Entity Substitution Contradiction
            # e.g., "Michael Collins was the first person to step onto the Moon" vs "Commander Neil Armstrong became the first human to walk on the lunar surface"
            unmatched_claim_ent = {e for e in claim_entities if e not in ev_entities and e not in ev_tokens}
            unmatched_ev_ent = {e for e in ev_entities if e not in claim_entities and e not in claim_tokens}
            if unmatched_claim_ent and unmatched_ev_ent and len(content_intersection) >= 2:
                # If both have conflicting named entities for the same action/predicate
                contradiction_signals += 3

            # 3. Number Mismatch Contradiction
            # If claim asserts numbers that are conflicting with evidence
            # ONLY when claim_numbers is NOT a subset of ev_numbers
            if claim_numbers and ev_numbers and not claim_numbers.issubset(ev_numbers):
                unmatched_nums = claim_numbers - ev_numbers
                if unmatched_nums and len(content_intersection) >= 3:
                    contradiction_signals += 2

            # 4. Negation polarity mismatch on identical actions
            action_words = claim_content - claim_entities - claim_numbers
            ev_action_words = ev_content - ev_entities - ev_numbers
            action_intersection = action_words & ev_action_words

            if (claim_has_negation != ev_has_negation) and (len(action_intersection) >= 2):
                # Only flag negation contradiction if the actual action/verb is shared
                contradiction_signals += 2

            # Scoring
            if contradiction_signals > 0 and (ev.similarity_score > 0.25 or len(content_intersection) >= 2):
                con_score = min(0.95, 0.45 + 0.20 * contradiction_signals)
                ent_score = max(0.02, 1.0 - con_score - 0.05)
            else:
                con_score = 0.05
                # Boost entailment when content overlap or semantic similarity is high
                sim = ev.similarity_score
                ent_score = min(0.95, (overlap_ratio * 0.55) + (sim * 0.45))
                if overlap_ratio >= 0.6:
                    ent_score = max(ent_score, 0.70)
                if overlap_ratio >= 0.8:
                    ent_score = max(ent_score, 0.85)

            neu_score = max(0.05, 1.0 - (ent_score + con_score))
            total = ent_score + neu_score + con_score
            ent_norm = ent_score / total
            con_norm = con_score / total

            if ent_norm > max_entailment:
                max_entailment = ent_norm
                if con_norm < 0.35:
                    best_ev = ev
            if con_norm > max_contradiction:
                max_contradiction = con_norm
                if con_norm > 0.45:
                    best_ev = ev

        final_ent = max_entailment
        final_con = max_contradiction
        final_neu = max(0.05, 1.0 - (final_ent + final_con))
        norm_sum = final_ent + final_con + final_neu

        return NLIScores(
            entailment=round(final_ent / norm_sum, 4),
            neutral=round(final_neu / norm_sum, 4),
            contradiction=round(final_con / norm_sum, 4)
        ), best_ev

    def _evaluate_neural(
        self,
        claim: str,
        evidence_matches: List[EvidenceMatch]
    ) -> Tuple[NLIScores, EvidenceMatch]:
        import torch
        best_ev = evidence_matches[0]
        max_ent = 0.0
        max_con = 0.0

        pairs = [(ev.text, claim) for ev in evidence_matches]
        inputs = self._tokenizer(pairs, padding=True, truncation=True, max_length=self.settings.MAX_SEQUENCE_LENGTH, return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for idx, p in enumerate(probs):
            id2label = getattr(self._model.config, "id2label", {0: "contradiction", 1: "neutral", 2: "entailment"})
            label_map = {v.lower(): k for k, v in id2label.items()}
            
            c_prob = float(p[label_map.get("contradiction", 0)])
            n_prob = float(p[label_map.get("neutral", 1)])
            e_prob = float(p[label_map.get("entailment", 2)])

            if e_prob > max_ent:
                max_ent = e_prob
                if c_prob < 0.4:
                    best_ev = evidence_matches[idx]
            if c_prob > max_con:
                max_con = c_prob
                if c_prob > 0.5:
                    best_ev = evidence_matches[idx]

        final_neu = max(0.05, 1.0 - (max_ent + max_con))
        norm_sum = max_ent + max_con + final_neu
        return NLIScores(
            entailment=round(max_ent / norm_sum, 4),
            neutral=round(final_neu / norm_sum, 4),
            contradiction=round(max_con / norm_sum, 4)
        ), best_ev
