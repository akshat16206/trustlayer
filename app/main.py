import os
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from app.schemas.payload import TrustRequest, TrustResponse
from app.services.pipeline import TrustLayerPipeline
from app.api.v1.endpoints import router as api_v1_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=f"{settings.APP_NAME} API",
    description="Real-time Hallucination Firewall & NLI Guardrail for RAG systems.",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

pipeline = TrustLayerPipeline()

# Mount API routes
app.include_router(api_v1_router, prefix="/api/v1")

# Static UI directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root(request: Request):
    accept_header = request.headers.get("accept", "")
    index_file = os.path.join(static_dir, "index.html")
    # If a browser is requesting the web page
    if "text/html" in accept_header and os.path.exists(index_file):
        return FileResponse(index_file)
    # Default API response for JSON clients & test suites
    return {"status": "ok", "service": "TrustLayer"}

@app.get("/playground")
def playground():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Playground static UI not found"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TrustLayer"}

@app.post("/evaluate", response_model=TrustResponse, status_code=status.HTTP_200_OK)
async def evaluate_rag_output(request: TrustRequest):
    """
    Top-level evaluate endpoint for backward compatibility with existing tests and clients.
    """
    try:
        response = await pipeline.process_payload(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))