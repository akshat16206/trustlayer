import re
import math
from typing import List, Dict, Any, Optional
from app.schemas.payload import ContextDocument, EvidenceMatch
from app.core.config import Settings, get_settings

class ContextMatcher:
    """
    Chunks source context documents into granular evidence sentences and
    performs semantic vector similarity search to find the most relevant
    evidence for each atomic claim.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.sentence_regex = re.compile(
            r'(?:(?<=[\.\?\!])|(?<=[\.\?\!][\"\'\)]))\s+'
        )
        self._sbert_model = None
        self._sbert_available = False
        self._init_encoder()

    def _init_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert_model = SentenceTransformer(self.settings.EMBEDDING_MODEL_NAME)
            self._sbert_available = True
        except Exception:
            self._sbert_available = False

    def chunk_document(self, doc: ContextDocument) -> List[Dict[str, Any]]:
        """Splits document text into sentence-level passages with metadata, including multi-sentence context windows."""
        text = doc.text.strip()
        if not text:
            return []
        
        raw_sentences = [s.strip() for s in self.sentence_regex.split(text) if len(s.strip()) > 5]
        if not raw_sentences:
            return [{"doc_id": doc.doc_id, "text": text, "source_url": doc.source_url}]
        
        chunks = []
        # 1. Add individual sentences
        for s in raw_sentences:
            chunks.append({
                "doc_id": doc.doc_id,
                "text": s,
                "source_url": doc.source_url
            })
        
        # 2. Add 2-sentence sliding windows (preserves pronouns like 'they', 'the crew', 'their')
        for i in range(len(raw_sentences) - 1):
            window = f"{raw_sentences[i]} {raw_sentences[i+1]}"
            chunks.append({
                "doc_id": doc.doc_id,
                "text": window,
                "source_url": doc.source_url
            })

        # 3. Add leading context anchor if available (e.g. intro sentence paired with conclusion)
        if len(raw_sentences) >= 3:
            anchor = raw_sentences[0]
            for s in raw_sentences[2:]:
                chunks.append({
                    "doc_id": doc.doc_id,
                    "text": f"{anchor} {s}",
                    "source_url": doc.source_url
                })

        return chunks

    def retrieve_evidence(
        self,
        claim: str,
        contexts: List[ContextDocument],
        top_k: Optional[int] = None
    ) -> List[EvidenceMatch]:
        """
        Finds the top-K most relevant evidence sentences for a given claim.
        """
        k = top_k or self.settings.TOP_K_EVIDENCE
        if not contexts or not claim:
            return []
        
        # Flatten all context documents into sentence chunks
        all_chunks: List[Dict[str, Any]] = []
        for doc in contexts:
            all_chunks.extend(self.chunk_document(doc))
            
        if not all_chunks:
            return []

        chunk_texts = [c["text"] for c in all_chunks]

        # Use neural SentenceTransformers if available
        if self._sbert_available and self._sbert_model is not None:
            return self._retrieve_neural(claim, all_chunks, chunk_texts, k)
        else:
            return self._retrieve_tfidf(claim, all_chunks, chunk_texts, k)

    def _retrieve_neural(
        self,
        claim: str,
        all_chunks: List[Dict[str, Any]],
        chunk_texts: List[str],
        top_k: int
    ) -> List[EvidenceMatch]:
        try:
            import numpy as np
            from sentence_transformers import util

            claim_emb = self._sbert_model.encode(claim, convert_to_tensor=True)
            chunk_embs = self._sbert_model.encode(chunk_texts, convert_to_tensor=True)
            
            cos_scores = util.cos_sim(claim_emb, chunk_embs)[0].cpu().numpy()
            
            top_indices = np.argsort(-cos_scores)[:top_k]
            matches = []
            for idx in top_indices:
                score = float(cos_scores[idx])
                chunk = all_chunks[idx]
                matches.append(EvidenceMatch(
                    doc_id=chunk["doc_id"],
                    text=chunk["text"],
                    similarity_score=round(max(0.0, min(1.0, score)), 4),
                    source_url=chunk["source_url"]
                ))
            return matches
        except Exception:
            return self._retrieve_tfidf(claim, all_chunks, chunk_texts, top_k)

    def _retrieve_tfidf(
        self,
        claim: str,
        all_chunks: List[Dict[str, Any]],
        chunk_texts: List[str],
        top_k: int
    ) -> List[EvidenceMatch]:
        """Fast fallback using TF-IDF cosine similarity from scikit-learn."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(chunk_texts + [claim])
            
            chunk_vectors = tfidf_matrix[:-1]
            claim_vector = tfidf_matrix[-1:]
            
            sims = cosine_similarity(claim_vector, chunk_vectors).flatten()
            
            top_indices = np.argsort(-sims)[:top_k]
            matches = []
            for idx in top_indices:
                score = float(sims[idx])
                chunk = all_chunks[idx]
                matches.append(EvidenceMatch(
                    doc_id=chunk["doc_id"],
                    text=chunk["text"],
                    similarity_score=round(max(0.0, min(1.0, score)), 4),
                    source_url=chunk["source_url"]
                ))
            return matches
        except Exception:
            # Simple lexical Jaccard overlap fallback
            return self._retrieve_jaccard(claim, all_chunks, top_k)

    def _retrieve_jaccard(
        self,
        claim: str,
        all_chunks: List[Dict[str, Any]],
        top_k: int
    ) -> List[EvidenceMatch]:
        claim_words = set(re.findall(r'\w+', claim.lower()))
        scored_chunks = []
        for chunk in all_chunks:
            chunk_words = set(re.findall(r'\w+', chunk["text"].lower()))
            if not chunk_words:
                sim = 0.0
            else:
                sim = len(claim_words & chunk_words) / float(len(claim_words | chunk_words))
            scored_chunks.append((sim, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        matches = []
        for sim, chunk in scored_chunks[:top_k]:
            matches.append(EvidenceMatch(
                doc_id=chunk["doc_id"],
                text=chunk["text"],
                similarity_score=round(sim, 4),
                source_url=chunk["source_url"]
            ))
        return matches
