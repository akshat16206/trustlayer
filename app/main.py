from fastapi import FastAPI, HTTPException, status
from app.schemas.payload import TrustRequest, TrustResponse
from app.services.pipeline import TrustLayerPipeline

app = FastAPI(title="TrustLayer API", version="0.1.0")
pipeline = TrustLayerPipeline()

@app.get("/")
def health_check():
    return {"status": "ok", "service": "TrustLayer"}

@app.post("/evaluate", response_model=TrustResponse, status_code=status.HTTP_200_OK)
async def evaluate_rag_output(request: TrustRequest):
    try:
        response = await pipeline.process_payload(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))