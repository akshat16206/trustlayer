import pytest
from app.schemas.payload import ClaimEvaluation, NLIScores, EvidenceMatch
from app.services.aggregator import ScoreAggregator

def test_aggregator_pass():
    aggregator = ScoreAggregator()
    evals = [
        ClaimEvaluation(
            claim_id=1,
            claim="Fact A",
            verdict="ENTAILMENT",
            confidence=0.95,
            is_grounded=True,
            scores=NLIScores(entailment=0.95, neutral=0.03, contradiction=0.02)
        ),
        ClaimEvaluation(
            claim_id=2,
            claim="Fact B",
            verdict="ENTAILMENT",
            confidence=0.90,
            is_grounded=True,
            scores=NLIScores(entailment=0.90, neutral=0.08, contradiction=0.02)
        )
    ]
    response = aggregator.aggregate(
        request_id="req_test_pass",
        generated_text="Fact A. Fact B.",
        evaluations=evals
    )
    assert response.action == "PASS"
    assert response.status == "PASSED"
    assert response.trust_score >= 0.8
    assert len(response.verified_claims) == 2
    assert len(response.flagged_claims) == 0

def test_aggregator_block_on_contradiction():
    aggregator = ScoreAggregator()
    evals = [
        ClaimEvaluation(
            claim_id=1,
            claim="Fact True",
            verdict="ENTAILMENT",
            confidence=0.95,
            is_grounded=True,
            scores=NLIScores(entailment=0.95, neutral=0.03, contradiction=0.02)
        ),
        ClaimEvaluation(
            claim_id=2,
            claim="Fact False",
            verdict="CONTRADICTION",
            confidence=0.88,
            is_grounded=False,
            scores=NLIScores(entailment=0.02, neutral=0.10, contradiction=0.88)
        )
    ]
    response = aggregator.aggregate(
        request_id="req_test_block",
        generated_text="Fact True. Fact False.",
        evaluations=evals
    )
    assert response.action == "BLOCK"
    assert response.status == "BLOCKED"
    assert response.trust_score < 0.5
    assert len(response.flagged_claims) == 1

def test_aggregator_filtering():
    aggregator = ScoreAggregator()
    evals = [
        ClaimEvaluation(
            claim_id=1,
            claim="Keep this factual claim.",
            verdict="ENTAILMENT",
            confidence=0.95,
            is_grounded=True,
            scores=NLIScores(entailment=0.95, neutral=0.03, contradiction=0.02)
        ),
        ClaimEvaluation(
            claim_id=2,
            claim="Remove this hallucinated claim.",
            verdict="NEUTRAL",
            confidence=0.80,
            is_grounded=False,
            scores=NLIScores(entailment=0.10, neutral=0.80, contradiction=0.10)
        )
    ]
    response = aggregator.aggregate(
        request_id="req_test_filter",
        generated_text="Keep this factual claim. Remove this hallucinated claim.",
        evaluations=evals,
        filter_hallucinations=True
    )
    assert "Keep this factual claim." in response.filtered_text
    assert "Remove this hallucinated claim." not in response.filtered_text
