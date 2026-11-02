import time
import asyncio
from typing import Optional
from app.schemas.payload import TrustRequest, TrustResponse, LatencyBreakdown
from app.services.claim_extractor import AtomicClaimExtractor
from app.services.context_matcher import ContextMatcher
from app.services.nli_verifier import NLIVerifier
from app.services.aggregator import ScoreAggregator
from app.core.config import Settings, get_settings

class TrustLayerPipeline:
    """
    Orchestrates the 4-stage Hallucination Firewall pipeline:
    1. Input Ingestion & Claim Extraction
    2. Context Matching & Evidence Retrieval
    3. Natural Language Inference (NLI) Verification
    4. Trust Score Aggregation & Policy Guardrail Enforcement
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.extractor = AtomicClaimExtractor()
        self.matcher = ContextMatcher(self.settings)
        self.verifier = NLIVerifier(self.settings)
        self.aggregator = ScoreAggregator(self.settings)

    async def process_payload(self, payload: TrustRequest) -> TrustResponse:
        t0 = time.perf_counter()
        
        # Stage 1: Atomic Claim Extraction
        t_ext_start = time.perf_counter()
        claims = await asyncio.to_thread(self.extractor.extract_claims, payload.generated_text)
        t_ext_end = time.perf_counter()
        
        # Stage 2: Context Matching (Evidence Retrieval)
        t_ret_start = time.perf_counter()
        # Retrieve evidence for each claim concurrently or in thread pool
        def retrieve_all_evidence():
            results = []
            for claim in claims:
                matches = self.matcher.retrieve_evidence(
                    claim=claim,
                    contexts=payload.retrieved_contexts,
                    top_k=self.settings.TOP_K_EVIDENCE
                )
                results.append(matches)
            return results

        claim_evidence_list = await asyncio.to_thread(retrieve_all_evidence)
        t_ret_end = time.perf_counter()
        
        # Stage 3: NLI Claim Verification
        t_nli_start = time.perf_counter()
        def verify_all_claims():
            evaluations = []
            for idx, (claim, evidence_matches) in enumerate(zip(claims, claim_evidence_list)):
                eval_res = self.verifier.verify_claim(
                    claim_id=idx + 1,
                    claim=claim,
                    evidence_matches=evidence_matches
                )
                evaluations.append(eval_res)
            return evaluations

        evaluations = await asyncio.to_thread(verify_all_claims)
        t_nli_end = time.perf_counter()
        
        # Stage 4: Score Aggregation & Policy Guardrail
        t_agg_start = time.perf_counter()
        latency = LatencyBreakdown(
            extraction_ms=round((t_ext_end - t_ext_start) * 1000, 2),
            retrieval_ms=round((t_ret_end - t_ret_start) * 1000, 2),
            nli_ms=round((t_nli_end - t_nli_start) * 1000, 2),
            aggregation_ms=0.0,
            total_ms=0.0
        )
        
        response = self.aggregator.aggregate(
            request_id=payload.request_id,
            generated_text=payload.generated_text,
            evaluations=evaluations,
            latency=latency,
            strictness_override=payload.strictness,
            filter_hallucinations=payload.filter_hallucinations,
            min_trust_score_override=payload.min_trust_score
        )
        t_agg_end = time.perf_counter()
        
        total_time_ms = round((t_agg_end - t0) * 1000, 2)
        latency.aggregation_ms = round((t_agg_end - t_agg_start) * 1000, 2)
        latency.total_ms = total_time_ms
        response.latency = latency
        
        return response