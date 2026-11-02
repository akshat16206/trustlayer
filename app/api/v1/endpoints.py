from fastapi import APIRouter, HTTPException, status
from typing import List
import asyncio
from app.schemas.payload import TrustRequest, TrustResponse
from app.services.pipeline import TrustLayerPipeline
from app.core.config import get_settings

router = APIRouter(tags=["Evaluation"])
pipeline = TrustLayerPipeline()

@router.get("/health", status_code=status.HTTP_200_OK)
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "models": {
            "embedding": settings.EMBEDDING_MODEL_NAME,
            "nli": settings.NLI_MODEL_NAME,
            "device": settings.DEVICE
        }
    }

@router.get("/config", status_code=status.HTTP_200_OK)
def get_pipeline_config():
    settings = get_settings()
    return {
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "nli_model": settings.NLI_MODEL_NAME,
        "top_k_evidence": settings.TOP_K_EVIDENCE,
        "contradiction_threshold": settings.CONTRADICTION_THRESHOLD,
        "entailment_threshold": settings.ENTAILMENT_THRESHOLD,
        "min_trust_score_pass": settings.MIN_TRUST_SCORE_PASS,
        "default_policy": settings.DEFAULT_POLICY
    }

@router.post("/evaluate", response_model=TrustResponse, status_code=status.HTTP_200_OK)
async def evaluate_rag_output(request: TrustRequest):
    """
    Evaluates a generated LLM text against retrieved source context documents.
    Extracts atomic claims, searches source evidence, performs NLI cross-checking,
    and returns calibrated trust scores with visual highlighting markup.
    """
    try:
        response = await pipeline.process_payload(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

@router.post("/evaluate/batch", response_model=List[TrustResponse], status_code=status.HTTP_200_OK)
async def evaluate_batch_rag_outputs(requests: List[TrustRequest]):
    """
    Evaluates a batch of LLM responses against their respective contexts.
    """
    try:
        tasks = [pipeline.process_payload(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        return list(responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")