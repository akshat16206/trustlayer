from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class ContextDocument(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for the source document")
    text: str = Field(..., description="Raw text content of the document or chunk")
    source_url: Optional[str] = Field(None, description="Optional URL or citation reference")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata key-values")

class EvidenceMatch(BaseModel):
    doc_id: str
    text: str
    similarity_score: float
    source_url: Optional[str] = None

class NLIScores(BaseModel):
    entailment: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    contradiction: float = Field(..., ge=0.0, le=1.0)

class ClaimEvaluation(BaseModel):
    claim_id: int
    claim: str
    verdict: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    confidence: float
    is_grounded: bool
    evidence: Optional[EvidenceMatch] = None
    all_evidence: List[EvidenceMatch] = Field(default_factory=list)
    scores: NLIScores
    reasoning: Optional[str] = None

class LatencyBreakdown(BaseModel):
    extraction_ms: float = 0.0
    retrieval_ms: float = 0.0
    nli_ms: float = 0.0
    aggregation_ms: float = 0.0
    total_ms: float = 0.0

class TrustRequest(BaseModel):
    request_id: str = Field(..., description="Unique request trace ID")
    generated_text: str = Field(..., min_length=1, description="LLM output to be verified")
    retrieved_contexts: List[ContextDocument] = Field(..., description="Source passages from RAG")
    strictness: Optional[Literal["STRICT", "BALANCED", "LENIENT"]] = Field(None, description="Policy strictness override")
    filter_hallucinations: bool = Field(False, description="If True, returns filtered_text with unsupported claims stripped")
    min_trust_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum acceptable trust score")

class TrustResponse(BaseModel):
    request_id: str
    status: str = Field(..., description="PASSED, FLAGGED, or BLOCKED")
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated score between 0.0 and 1.0")
    action: Literal["PASS", "FLAG", "BLOCK"] = "PASS"
    verified_claims: List[Dict[str, Any]] = Field(default_factory=list)
    flagged_claims: List[Dict[str, Any]] = Field(default_factory=list)
    claim_evaluations: List[ClaimEvaluation] = Field(default_factory=list)
    highlighted_html: Optional[str] = None
    filtered_text: Optional[str] = None
    summary: Optional[str] = None
    latency: Optional[LatencyBreakdown] = None