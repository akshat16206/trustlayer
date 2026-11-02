import pytest
from app.schemas.payload import EvidenceMatch
from app.services.nli_verifier import NLIVerifier

def test_nli_entailment():
    verifier = NLIVerifier()
    evidence = [
        EvidenceMatch(
            doc_id="doc_1",
            text="The Earth revolves around the Sun every 365.25 days.",
            similarity_score=0.85
        )
    ]
    claim = "The Earth revolves around the Sun."
    res = verifier.verify_claim(claim_id=1, claim=claim, evidence_matches=evidence)
    
    assert res.verdict == "ENTAILMENT"
    assert res.is_grounded is True
    assert res.scores.entailment > res.scores.contradiction

def test_nli_contradiction():
    verifier = NLIVerifier()
    evidence = [
        EvidenceMatch(
            doc_id="doc_1",
            text="Company revenue increased by 25 percent in 2023.",
            similarity_score=0.75
        )
    ]
    claim = "Company revenue decreased by 25 percent in 2023."
    res = verifier.verify_claim(claim_id=2, claim=claim, evidence_matches=evidence)
    
    assert res.verdict == "CONTRADICTION"
    assert res.is_grounded is False
    assert res.scores.contradiction >= 0.5

def test_nli_unsupported_neutral():
    verifier = NLIVerifier()
    # When no evidence is found
    res = verifier.verify_claim(claim_id=3, claim="Alien life was detected on Europa.", evidence_matches=[])
    assert res.verdict == "NEUTRAL"
    assert res.is_grounded is False
