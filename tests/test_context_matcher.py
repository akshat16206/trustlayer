import pytest
from app.schemas.payload import ContextDocument
from app.services.context_matcher import ContextMatcher

def test_document_chunking():
    matcher = ContextMatcher()
    doc = ContextDocument(
        doc_id="doc_space",
        text="The solar system consists of eight planets. Jupiter is the largest planet in our solar system."
    )
    chunks = matcher.chunk_document(doc)
    assert len(chunks) >= 2
    assert chunks[0]["doc_id"] == "doc_space"
    assert "eight planets" in chunks[0]["text"]
    assert "Jupiter" in chunks[1]["text"]

def test_evidence_retrieval_ranking():
    matcher = ContextMatcher()
    docs = [
        ContextDocument(
            doc_id="doc_planets",
            text="Jupiter is the largest planet in the solar system. Mars is often called the Red Planet."
        ),
        ContextDocument(
            doc_id="doc_history",
            text="The French Revolution began in 1789."
        )
    ]
    
    claim = "Jupiter has the largest planetary mass in our solar system."
    matches = matcher.retrieve_evidence(claim=claim, contexts=docs, top_k=2)
    
    assert len(matches) > 0
    assert matches[0].doc_id == "doc_planets"
    assert "Jupiter" in matches[0].text
    assert matches[0].similarity_score > 0.0
