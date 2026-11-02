import re
from typing import List

class AtomicClaimExtractor:
    """
    Extracts atomic, verifiable factual claims from generated LLM text.
    Handles sentence boundary tokenization, filtering conversational filler,
    and decomposing compound sentences into standalone propositions.
    """
    
    # Common conversational fillers to discard if they appear standalone
    DISCARD_PATTERNS = [
        r"^(sure|certainly|of course|here is|here are|hope this helps|in summary|to summarize|as an ai)[\.,!:]*$",
        r"^(let me know if you need|feel free to ask|thank you)[\.,!:]*$"
    ]
    
    # Conjunctions that often introduce new propositions
    CLAUSE_SPLIT_REGEX = re.compile(r";\s*|\s+(?:furthermore|moreover|additionally|however|nevertheless|in addition),\s*", re.IGNORECASE)

    def __init__(self):
        # Regex for sentence boundaries respecting common honorifics/abbreviations
        self._sentence_regex = re.compile(
            r'(?:(?<=[\.\?\!])|(?<=[\.\?\!][\"\'\)]))\s+'
        )

    def extract_claims(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        # Step 1: Normalize whitespace and linebreaks
        normalized = text.strip()
        paragraphs = [p.strip() for p in normalized.splitlines() if p.strip()]
        
        raw_sentences: List[str] = []
        for p in paragraphs:
            # Check for markdown bullet points or numbering
            cleaned_p = re.sub(r'^\s*[-*•\d+\.]+\s*', '', p)
            sentences = self._sentence_regex.split(cleaned_p)
            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    raw_sentences.append(s_clean)
        
        # Step 2: Clause decomposition for complex compound sentences
        decomposed: List[str] = []
        for sentence in raw_sentences:
            # Sub-split on strong clause delimiters (e.g. semicolons, furthermore)
            parts = self.CLAUSE_SPLIT_REGEX.split(sentence)
            for part in parts:
                cleaned_part = part.strip()
                if len(cleaned_part) > 3 and not self._is_filler(cleaned_part):
                    decomposed.append(cleaned_part)
        
        # Step 3: If decomposition resulted in empty list, fallback to raw sentence
        if not decomposed and raw_sentences:
            return raw_sentences
            
        return decomposed if decomposed else [text.strip()]

    def _is_filler(self, text: str) -> bool:
        lowered = text.lower().strip()
        for pattern in self.DISCARD_PATTERNS:
            if re.match(pattern, lowered):
                return True
        return False
