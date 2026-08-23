from pydantic import BaseModel, Field
from typing import List, Optional

class ContextDocument(BaseModel):
    doc_id: str
    text: str
    source_url: Optional[str] = None

class TrustRequest(BaseModel):
    request_id: str
    generated_text: str = Field(..., description="LLM output to be verified")
    retrieved_contexts: List[ContextDocument] = Field(..., description="Source passages from RAG")

class TrustResponse(BaseModel):
    request_id: str
    status: str
    trust_score: float
    verified_claims: List[dict]
    flagged_claims: List[dict]