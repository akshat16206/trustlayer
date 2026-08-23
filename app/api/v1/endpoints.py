# app/api/v1/endpoints.py
from fastapi import APIRouter, HTTPException, status
from app.schemas.payload import TrustRequest, TrustResponse
from app.services.pipeline import TrustLayerPipeline

router = APIRouter()
pipeline = TrustLayerPipeline()

@router.post("/evaluate", response_model=TrustResponse, status_code=status.HTTP_200_OK)
async def evaluate_rag_output(request: TrustRequest):
    try:
        response = await pipeline.process_payload(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))