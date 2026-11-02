import re
from typing import List, Dict, Any, Tuple, Optional, Literal
from app.schemas.payload import ClaimEvaluation, TrustResponse, LatencyBreakdown
from app.core.config import Settings, get_settings

class ScoreAggregator:
    """
    Aggregates granular claim-level evaluations into an overall Trust Score,
    evaluates policy guardrails (PASS, FLAG, BLOCK), formats visual highlights,
    and produces sanitized / filtered safe output.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def aggregate(
        self,
        request_id: str,
        generated_text: str,
        evaluations: List[ClaimEvaluation],
        latency: Optional[LatencyBreakdown] = None,
        strictness_override: Optional[Literal["STRICT", "BALANCED", "LENIENT"]] = None,
        filter_hallucinations: bool = False,
        min_trust_score_override: Optional[float] = None
    ) -> TrustResponse:
        policy = strictness_override or self.settings.DEFAULT_POLICY
        min_score = min_trust_score_override if min_trust_score_override is not None else self.settings.MIN_TRUST_SCORE_PASS

        if not evaluations:
            # Fallback for empty claims
            return TrustResponse(
                request_id=request_id,
                status="PASSED",
                trust_score=1.0,
                action="PASS",
                verified_claims=[],
                flagged_claims=[],
                claim_evaluations=[],
                highlighted_html=generated_text,
                filtered_text=generated_text,
                summary="No verifiable factual claims detected in the text.",
                latency=latency
            )

        num_claims = len(evaluations)
        total_entailment = sum(e.scores.entailment for e in evaluations)
        avg_entailment = total_entailment / float(num_claims)
        
        max_contradiction = max(e.scores.contradiction for e in evaluations)
        has_contradiction = any(e.verdict == "CONTRADICTION" for e in evaluations)
        has_neutral = any(e.verdict == "NEUTRAL" for e in evaluations)

        # Contradictions incur severe penalties on trust
        penalty_factor = (1.0 - (max_contradiction ** 1.5))
        calibrated_score = max(0.0, min(1.0, avg_entailment * penalty_factor))
        trust_score = round(calibrated_score, 4)

        # Policy guardrail determination
        if policy == "STRICT":
            con_thresh = 0.40
            min_pass = 0.80
        elif policy == "LENIENT":
            con_thresh = 0.70
            min_pass = 0.55
        else:  # BALANCED
            con_thresh = self.settings.CONTRADICTION_THRESHOLD
            min_pass = min_score

        if (has_contradiction and self.settings.BLOCK_ON_CONTRADICTION) or (max_contradiction >= con_thresh and self.settings.BLOCK_ON_CONTRADICTION):
            action = "BLOCK"
            status = "BLOCKED"
        elif has_neutral or any(not e.is_grounded for e in evaluations):
            action = "FLAG"
            status = "FLAGGED"
        elif trust_score < min_pass and not all(e.is_grounded for e in evaluations):
            action = "FLAG"
            status = "FLAGGED"
        else:
            action = "PASS"
            status = "PASSED"

        # Organize verified vs flagged claims
        verified_claims: List[Dict[str, Any]] = []
        flagged_claims: List[Dict[str, Any]] = []

        for e in evaluations:
            claim_dict = {
                "claim_id": e.claim_id,
                "claim": e.claim,
                "verdict": e.verdict,
                "confidence": e.confidence,
                "is_grounded": e.is_grounded,
                "evidence": e.evidence.text if e.evidence else None,
                "evidence_doc_id": e.evidence.doc_id if e.evidence else None,
                "scores": e.scores.model_dump()
            }
            if e.verdict == "ENTAILMENT":
                verified_claims.append(claim_dict)
            else:
                flagged_claims.append(claim_dict)

        # Generate HTML highlights and sanitized text
        highlighted_html, filtered_text = self._build_markup_and_filtered(generated_text, evaluations)

        # Executive summary
        summary = self._build_summary(num_claims, verified_claims, flagged_claims, action, trust_score)

        return TrustResponse(
            request_id=request_id,
            status=status,
            trust_score=trust_score,
            action=action,
            verified_claims=verified_claims,
            flagged_claims=flagged_claims,
            claim_evaluations=evaluations,
            highlighted_html=highlighted_html,
            filtered_text=filtered_text if filter_hallucinations else None,
            summary=summary,
            latency=latency
        )

    def _build_markup_and_filtered(
        self,
        original_text: str,
        evaluations: List[ClaimEvaluation]
    ) -> Tuple[str, str]:
        """
        Creates color-coded HTML markup and a sanitized output with hallucinated claims stripped.
        """
        html = original_text
        valid_sentences = []

        for e in evaluations:
            claim_raw = e.claim.strip()
            if not claim_raw:
                continue

            escaped = re.escape(claim_raw)
            if e.verdict == "ENTAILMENT":
                badge_class = "claim-entailed"
                css_style = "background-color: #d1fae5; color: #065f46; border-radius: 4px; padding: 2px 4px; border-bottom: 2px solid #10b981;"
                valid_sentences.append(claim_raw)
            elif e.verdict == "CONTRADICTION":
                badge_class = "claim-contradiction"
                css_style = "background-color: #fee2e2; color: #991b1b; border-radius: 4px; padding: 2px 4px; border-bottom: 2px solid #ef4444;"
            else:  # NEUTRAL
                badge_class = "claim-neutral"
                css_style = "background-color: #fef3c7; color: #92400e; border-radius: 4px; padding: 2px 4px; border-bottom: 2px solid #f59e0b;"

            evidence_hint = f"Evidence: {e.evidence.text[:80]}..." if e.evidence else "No evidence found"
            replacement = (
                f'<span class="{badge_class}" style="{css_style}" '
                f'data-verdict="{e.verdict}" title="{e.verdict} (Score: {e.confidence:.2f}) - {evidence_hint}">'
                f'{claim_raw}</span>'
            )
            # Replace first instance of this claim in text
            html = re.sub(escaped, replacement, html, count=1)

        filtered_text = " ".join(valid_sentences) if valid_sentences else ""
        return html, filtered_text

    def _build_summary(
        self,
        total: int,
        verified: List[Dict[str, Any]],
        flagged: List[Dict[str, Any]],
        action: str,
        score: float
    ) -> str:
        pct = int(score * 100)
        if action == "BLOCK":
            return f"Guardrail Alert: Response BLOCKED. High-risk contradictions detected against source evidence ({len(flagged)}/{total} claims flagged). Overall trust score: {pct}%."
        elif action == "FLAG":
            return f"Guardrail Warning: Response FLAGGED. {len(verified)}/{total} claims verified; {len(flagged)} claims lack direct evidence or showed discrepancies. Overall trust score: {pct}%."
        else:
            return f"Verification Passed: All {total} factual claims were successfully verified and grounded in source documents. Overall trust score: {pct}%."
