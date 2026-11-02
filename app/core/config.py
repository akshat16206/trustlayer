from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "TrustLayer"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = False
    
    # Model configurations
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"
    DEVICE: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Context Matching parameters
    TOP_K_EVIDENCE: int = 3
    SIMILARITY_THRESHOLD: float = 0.20
    CHUNK_BY_SENTENCE: bool = True
    
    # Verification & Decision thresholds
    CONTRADICTION_THRESHOLD: float = 0.50
    ENTAILMENT_THRESHOLD: float = 0.55
    MIN_TRUST_SCORE_PASS: float = 0.60
    BLOCK_ON_CONTRADICTION: bool = True
    DEFAULT_POLICY: Literal["STRICT", "BALANCED", "LENIENT"] = "BALANCED"
    
    # Performance & Inference
    BATCH_SIZE: int = 16
    MAX_SEQUENCE_LENGTH: int = 512
    ENABLE_CACHING: bool = True
    
    # Fallback to heuristic engine if neural model is not present
    FALLBACK_TO_HEURISTIC: bool = True

    model_config = SettingsConfigDict(
        env_prefix="TRUSTLAYER_",
        case_sensitive=False
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
