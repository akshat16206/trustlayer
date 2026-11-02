# TrustLayer: Real-Time Hallucination Firewall & Truth Guardrail

**TrustLayer** is an enterprise-grade "hallucination firewall" designed to validate LLM outputs against source context documents before reaching end users—functioning as a **"grammar checker for truth."** It intercepts model responses in Retrieval-Augmented Generation (RAG) pipelines, decomposes statements into atomic testable claims, retrieves granular evidence passages via vector similarity, and verifies factual entailment using cross-encoder Natural Language Inference (NLI).

---

## 🏛️ Core Architecture & Flow

```
                           ┌───────────────────────────┐
                           │   LLM Response & Source   │
                           │     Context Documents     │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │ 1. Atomic Claim Extractor │
                           │ (Sentence/Clause Parsing) │
                           └─────────────┬─────────────┘
                                         │ Claims [c₁, c₂, ...]
                                         ▼
                           ┌───────────────────────────┐
                           │ 2. Context Matcher        │
                           │  (Dense Vector Cosine Sim │
                           │   & Granular Evidence)    │
                           └─────────────┬─────────────┘
                                         │ (Claim, Evidence) Pairs
                                         ▼
                           ┌───────────────────────────┐
                           │ 3. NLI Claim Verifier     │
                           │ (DeBERTa-v3 Cross-Encoder │
                           │  Entail / Neutral / Cont) │
                           └─────────────┬─────────────┘
                                         │ P(Ent), P(Neu), P(Con)
                                         ▼
                           ┌───────────────────────────┐
                           │ 4. Score Aggregator &     │
                           │    Policy Guardrail       │
                           │ (PASS / FLAG / BLOCK,     │
                           │  Visual Markup & Filter)  │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │   Calibrated Trust Score  │
                           │     & Guardrail Action    │
                           └───────────────────────────┘
```

1. **Input Ingestion & Validation**: Accepts generated LLM text and retrieved context passages with configurable strictness policies (`STRICT`, `BALANCED`, `LENIENT`).
2. **Atomic Claim Extraction**: Decomposes compound sentences and structured markdown into isolated factual propositions.
3. **Context Matching**: Segments context documents into granular evidence sentences and performs dense vector similarity search to pair each claim with candidate evidence.
4. **NLI Claim Verification**: Evaluates each (claim, evidence) pair using a cross-encoder model (such as DeBERTa-v3 or RoBERTa) to calculate entailment, neutral, and contradiction probabilities.
5. **Score Aggregation & Policy Guardrail**: Aggregates claim probabilities into a calibrated document-level Trust Score ($0.0 \to 1.0$), triggering guardrail policies (`PASS`, `FLAG`, `BLOCK`), generating color-coded interactive highlights, and optionally outputting sanitized text with hallucinations filtered out.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Open Interactive Playground
Visit **[http://localhost:8000](http://localhost:8000)** (or `http://localhost:8000/playground`) in your browser to test claims, adjust sensitivity thresholds, and inspect evidence matches in real-time.

Interactive API Swagger documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 📡 API Endpoints

### `POST /api/v1/evaluate` (or `/evaluate`)
Evaluates a single LLM response against retrieved context documents.

**Request Payload:**
```json
{
  "request_id": "req_001",
  "generated_text": "Apollo 11 landed humans on the Moon in July 1969.",
  "retrieved_contexts": [
    {
      "doc_id": "doc_nasa",
      "text": "The Apollo 11 mission landed humans on the Moon on July 20, 1969."
    }
  ],
  "strictness": "BALANCED",
  "filter_hallucinations": true
}
```

**Response:**
```json
{
  "request_id": "req_001",
  "status": "PASSED",
  "trust_score": 0.95,
  "action": "PASS",
  "verified_claims": [
    {
      "claim_id": 1,
      "claim": "Apollo 11 landed humans on the Moon in July 1969.",
      "verdict": "ENTAILMENT",
      "confidence": 0.95,
      "is_grounded": true,
      "evidence": "The Apollo 11 mission landed humans on the Moon on July 20, 1969."
    }
  ],
  "flagged_claims": [],
  "highlighted_html": "<span class=\"claim-entailed\" ...>Apollo 11 landed humans on the Moon in July 1969.</span>",
  "filtered_text": "Apollo 11 landed humans on the Moon in July 1969.",
  "summary": "Verification Passed: All 1 factual claims were successfully verified and grounded in source documents. Overall trust score: 95%."
}
```

### `POST /api/v1/evaluate/batch`
Evaluates a list of verification payloads concurrently.

### `GET /api/v1/config`
Inspects active runtime model parameters, thresholds, and guardrail settings.

### `GET /health`
Returns health check status and model configuration.

---

## 🧪 Testing & Benchmarks

### Run Automated Unit & Integration Tests
```bash
pytest -v tests/
```

### Run Hallucination Detection Benchmark
```bash
python training/benchmark_eval.py
```

---

## 🧠 Model Training & Fine-Tuning

To fine-tune a custom cross-encoder model on domain-specific claim-evidence pairs (such as FEVER, SNLI, or custom enterprise datasets):

```bash
python training/train_cross_encoder.py \
  --model cross-encoder/nli-deberta-v3-small \
  --output ./models/trustlayer-nli-checkpoint \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5
```

---

## ⚙️ Configuration Options

Configuration can be customized via environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `TRUSTLAYER_EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer for context retrieval |
| `TRUSTLAYER_NLI_MODEL_NAME` | `cross-encoder/nli-deberta-v3-small` | NLI cross-encoder model |
| `TRUSTLAYER_DEVICE` | `auto` | Execution device (`cpu`, `cuda`, `mps`) |
| `TRUSTLAYER_TOP_K_EVIDENCE` | `3` | Evidence sentences retrieved per claim |
| `TRUSTLAYER_CONTRADICTION_THRESHOLD` | `0.50` | Contradiction sensitivity trigger |
| `TRUSTLAYER_ENTAILMENT_THRESHOLD` | `0.55` | Minimum entailment probability |
| `TRUSTLAYER_BLOCK_ON_CONTRADICTION` | `True` | Automatically block response if contradiction detected |
| `TRUSTLAYER_DEFAULT_POLICY` | `BALANCED` | Guardrail policy (`STRICT`, `BALANCED`, `LENIENT`) |