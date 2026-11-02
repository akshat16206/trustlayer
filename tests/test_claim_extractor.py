import pytest
from app.services.claim_extractor import AtomicClaimExtractor

def test_extract_simple_sentences():
    extractor = AtomicClaimExtractor()
    text = "The Earth revolves around the Sun. The Moon revolves around the Earth."
    claims = extractor.extract_claims(text)
    assert len(claims) == 2
    assert "The Earth revolves around the Sun." in claims
    assert "The Moon revolves around the Earth." in claims

def test_extract_compound_sentences_with_semicolon():
    extractor = AtomicClaimExtractor()
    text = "Quantum computing relies on qubits; classical computers use bits."
    claims = extractor.extract_claims(text)
    assert len(claims) >= 2
    assert any("Quantum computing relies on qubits" in c for c in claims)
    assert any("classical computers use bits" in c for c in claims)

def test_extract_markdown_bullet_points():
    extractor = AtomicClaimExtractor()
    text = """
    Here are the facts:
    * Python was created by Guido van Rossum.
    * Rust was designed by Graydon Hoare.
    """
    claims = extractor.extract_claims(text)
    assert any("Python was created by Guido van Rossum" in c for c in claims)
    assert any("Rust was designed by Graydon Hoare" in c for c in claims)

def test_empty_or_whitespace_text():
    extractor = AtomicClaimExtractor()
    assert extractor.extract_claims("") == []
    assert extractor.extract_claims("   \n  ") == []
