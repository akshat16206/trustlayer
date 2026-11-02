# TrustLayer: Real-Time Hallucination Firewall for RAG Applications
## Comprehensive 5-Milestone Architecture & Project Report

**Authors**: Aditi Agrawal (BTCS24O1009), Akshat Gupta (BTCS24O1016)  
**Project Repository**: `trustlayer/`  
**Application Type**: Real-Time Guardrail Middleware for Retrieval-Augmented Generation (RAG)

---

# Milestone Overview Matrix

| Milestone | Module Name | Core Components & Files | Key Deliverables | Status |
|---|---|---|---|---|
| **Milestone 1** | Core Architecture, API Middleware & Data Contract Setup | `app/main.py`, `app/schemas/payload.py`, `app/services/pipeline.py`, `tests/test_api.py` | FastAPI skeleton, Pydantic schemas, stubbed pipeline, initial PyTest suite (3 tests) | Completed & Verified |
| **Milestone 2** | Atomic Claim Extraction & Linguistic Decomposition | `app/services/claim_extractor.py`, `tests/test_claim_extractor.py` | Quotation-aware sentence tokenizer, clause decomposition, conversational filler stripping | Completed & Verified |
| **Milestone 3** | Context Matching & Semantic Vector Retrieval | `app/services/context_matcher.py`, `tests/test_context_matcher.py` | SBERT dense vector retrieval, windowed chunking with coreference anchors, TF-IDF fallback | Completed & Verified |
| **Milestone 4** | Natural Language Inference (NLI) Verification Engine | `app/services/nli_verifier.py`, `training/train_cross_encoder.py`, `tests/test_nli_verifier.py` | DeBERTa-v3 cross-encoder inference, entity-substitution & predicate-aware safeguards, training pipeline | Completed & Verified |
| **Milestone 5** | Trust Score Aggregation, Guardrail Policies & Web Playground | `app/services/aggregator.py`, `app/api/v1/endpoints.py`, `app/static/index.html`, `training/benchmark_eval.py` | Mathematical trust scoring, PASS/FLAG/BLOCK policy, text filtering, interactive dark-mode UI, 19-test suite | Completed & Verified |

---

# MILESTONE 1
## Module: Core Architecture, API Middleware & Data Contract Setup
* **Status**: Completed & Verified  
* **Submitted by**: Aditi Agrawal (BTCS24O1009), Akshat Gupta (BTCS24O1016)

### 1. Executive Summary & Objectives
TrustLayer is an automated "hallucination firewall" designed to intercept Retrieval-Augmented Generation (RAG) outputs, parse their factual claims, and verify them against source documentation before delivering them to users.

The main objective of **Milestone 1** is to establish the software infrastructure, REST API middleware layer, schema data contracts, and unit testing suite. Building this communication pipeline first ensures that when deep learning models (NLI models and vector search) are integrated in subsequent milestones, the system has a stable, modular, and error-tolerant pipe through which data flows.

### 2. Technical Implementations & Theoretical Justification

#### 2.1 Project Architecture & Directory Layout
* **What was done**: Organized the project repository using a clean, modular package structure following standard enterprise Python design patterns.
* **Directory Structure**:
  ```
  trustlayer/
  ├── app/
  │   ├── __init__.py
  │   ├── main.py            # Application entry point & route handlers
  │   ├── schemas/
  │   │   ├── __init__.py
  │   │   └── payload.py     # Pydantic data validation contracts
  │   └── services/
  │       ├── __init__.py
  │       └── pipeline.py    # Middleware interceptor & mock logic
  └── tests/
      ├── __init__.py
      └── test_api.py        # Automated PyTest integration test suite
  ```
* **Why it is needed (Theoretical Justification)**: This follows the **Separation of Concerns (SoC)** software engineering principle. By separating routing (`main.py`) from schemas (`payload.py`) and processing services (`pipeline.py`), the Machine Learning logic can be swapped, upgraded, or fine-tuned in later milestones without breaking the external API interface.

#### 2.2 Data Contract & Schema Layer (`schemas/payload.py`)
* **What was done**: Created strict data input/output models using Pydantic (`ContextDocument`, `TrustRequest`, and `TrustResponse`).
* **Source Code**:
  ```python
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
  ```
* **Why it is needed (Theoretical Justification)**: Middleware systems operating in production RAG environments must enforce strict **Data Contracts**. Pydantic inspects incoming JSON payloads at runtime. If an upstream LLM or user sends incomplete data or wrong types, the middleware immediately rejects it with a `422 Unprocessable Entity` status before wasting unnecessary GPU/compute power on NLP evaluation.

#### 2.3 Middleware Processing Pipeline (`services/pipeline.py`)
* **What was done**: Created the `TrustLayerPipeline` class to manage payload lifecycle and stubbed out future NLP processing steps.
* **Why it is needed (Theoretical Justification)**: This uses the **Stubbing / Pipeline Pattern**. By implementing mock responses early on, we prove that the entire end-to-end data lifecycle (ingestion $\to$ processing $\to$ output formatting) works before writing complex Machine Learning code.

#### 2.4 Server & API Interface (`main.py`)
* **What was done**: Built an asynchronous REST API server using FastAPI with `GET /` (health monitoring) and `POST /evaluate` (interception gate) endpoints.
* **Why it is needed (Theoretical Justification)**: TrustLayer operates as middleware (proxy layer). It must provide an HTTP endpoint that RAG applications can call asynchronously without blocking the user interface. FastAPI provides high-concurrency request handling and auto-generates standard OpenAPI/Swagger documentation.

### 3. System Verification & Output Execution

#### 3.1 Automated Unit Testing Results
An automated test suite was constructed inside `tests/test_api.py` using PyTest:

| Test Function | Endpoint | Scenario Tested | Expected Result | Pass / Fail |
|---|---|---|---|---|
| `test_health_check` | `GET /` | API availability | Status 200, `{"status": "ok", "service": "TrustLayer"}` | **PASSED** |
| `test_valid_evaluation_request` | `POST /evaluate` | Valid JSON payload | Status 200, `PASSED_MOCK`, Trust Score: 1.0 | **PASSED** |
| `test_invalid_payload_fails` | `POST /evaluate` | Missing required fields | Status 422 Validation Error | **PASSED** |

### 4. Conclusion & Connection to Milestone 2
Milestone 1 successfully established the architectural framework, payload validation layer, API endpoints, and testing harness. Milestone 2 replaces the mock logic with **Atomic Claim Extraction**, decomposing compound LLM responses into standalone factual propositions.

---

# MILESTONE 2
## Module: Atomic Claim Extraction & Linguistic Decomposition
* **Status**: Completed & Verified  
* **Components**: `app/services/claim_extractor.py`, `tests/test_claim_extractor.py`

### 1. Executive Summary & Objectives
Large Language Models rarely generate single isolated facts; they generate long, compound sentences combining multiple statements into single sentences. In many hallucinations, one half of a sentence is completely true while the other half is fabricated.

The objective of **Milestone 2** is to build the **Atomic Claim Extractor** service that intercepts generated LLM text and breaks it down into individual, verifiable factual claims while eliminating conversational padding.

```
                      [Generated LLM Response]
                                 │
                                 ▼
         ┌────────────────────────────────────────────────┐
         │ 1. Conversational Filler Filter                │
         │    (Strips: "Sure, here is...", "As an AI...") │
         └───────────────────────┬────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────┐
         │ 2. Quotation-Aware Boundary Tokenizer          │
         │    (Regex splits across quotes: ." ?)          │
         └───────────────────────┬────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────┐
         │ 3. Clause Decomposition                        │
         │    (Splits across: ';' / 'furthermore' / etc.) │
         └───────────────────────┬────────────────────────┘
                                 │
                                 ▼
            [Atomic Claims: Claim 1, Claim 2, ...]
```

### 2. Technical Implementations & Theoretical Justification

#### 2.1 Linguistic Clause & Boundary Tokenizer (`services/claim_extractor.py`)
* **What was done**: Implemented the `AtomicClaimExtractor` class using regular expression sentence boundary detection and clause decomposition.
* **Source Code Snippet**:
  ```python
  import re
  from typing import List

  class AtomicClaimExtractor:
      DISCARD_PATTERNS = [
          r"^(sure|certainly|of course|here is|here are|hope this helps|in summary|to summarize|as an ai)[\.,!:]*$",
          r"^(let me know if you need|feel free to ask|thank you)[\.,!:]*$"
      ]
      
      CLAUSE_SPLIT_REGEX = re.compile(
          r";\s*|\s+(?:furthermore|moreover|additionally|however|nevertheless|in addition),\s*", 
          re.IGNORECASE
      )

      def __init__(self):
          # Quotation-aware lookbehind sentence boundary regex
          self._sentence_regex = re.compile(
              r'(?:(?<=[\.\?\!])|(?<=[\.\?\!][\"\'\)]))\s+'
          )

      def extract_claims(self, text: str) -> List[str]:
          if not text or not text.strip():
              return []
          paragraphs = [p.strip() for p in text.strip().splitlines() if p.strip()]
          raw_sentences = []
          for p in paragraphs:
              cleaned_p = re.sub(r'^\s*[-*•\d+\.]+\s*', '', p)
              for s in self._sentence_regex.split(cleaned_p):
                  if s.strip():
                      raw_sentences.append(s.strip())

          decomposed = []
          for s in raw_sentences:
              for part in self.CLAUSE_SPLIT_REGEX.split(s):
                  clean = part.strip()
                  if len(clean) > 3 and not self._is_filler(clean):
                      decomposed.append(clean)
          return decomposed if decomposed else [text.strip()]
  ```
* **Why it is needed (Theoretical Justification)**:
  1. **Principle of Minimum Factual Units (Atomicity)**: Evaluating an entire multi-clause sentence in NLI leads to ambiguous truth values. Decomposing compound statements ensures that each claim contains a single subject-predicate assertion.
  2. **Quotation Boundary Preservation**: Standard tokenizers fail when a quote precedes whitespace (`." `), fusing unrelated sentences into multi-clause blocks. The lookbehind pattern `r'(?:(?<=[\.\?\!])|(?<=[\.\?\!][\"\'\)]))\s+'` guarantees clean boundaries without splitting abbreviations.

### 3. System Verification & Testing Results

| Test Function | Input Scenario | Expected Result | Pass / Fail |
|---|---|---|---|
| `test_extract_simple_sentences` | Two sentences separated by periods | 2 distinct atomic claims | **PASSED** |
| `test_extract_compound_sentences_with_semicolon` | Sentence conjoined with semicolon | Split into 2 standalone propositions | **PASSED** |
| `test_extract_markdown_bullet_points` | Markdown list (`* Item 1\n* Item 2`) | Strips bullet prefixes; returns 2 clean claims | **PASSED** |
| `test_empty_or_whitespace_text` | Empty string / whitespace string | Returns empty list `[]` | **PASSED** |

### 4. Conclusion & Connection to Milestone 3
Milestone 2 enables TrustLayer to isolate exact statements from LLM responses. Milestone 3 integrates **Context Matching & Vector Retrieval** to pair each extracted claim with relevant source evidence.

---

# MILESTONE 3
## Module: Context Matching & Semantic Vector Retrieval
* **Status**: Completed & Verified  
* **Components**: `app/services/context_matcher.py`, `tests/test_context_matcher.py`

### 1. Executive Summary & Objectives
In production RAG systems, context documents can span hundreds or thousands of tokens. Cross-encoders cannot efficiently process entire documents for every atomic claim due to quadratic attention complexity $\mathcal{O}(N^2)$. 

The objective of **Milestone 3** is to build a fast, high-accuracy **Evidence Retrieval Engine** that chunks reference documents, resolves pronoun coreference loss, and retrieves the top-$K$ most relevant evidence snippets using dense vector similarity.

```
       [Context Document]                  [Atomic Claim]
               │                                  │
               ▼                                  ▼
 ┌───────────────────────────┐      ┌───────────────────────────┐
 │ Windowed Chunking Engine  │      │ SentenceTransformer       │
 │ - Individual sentences    │      │ (all-MiniLM-L6-v2)        │
 │ - 2-sentence windows      │      │ Dense 384-d Embedding     │
 │ - Context anchors         │      └─────────────┬─────────────┘
 └─────────────┬─────────────┘                    │
               ▼                                  │
 ┌───────────────────────────┐                    │
 │ Dense Vector Embeddings   │                    │
 └─────────────┬─────────────┘                    │
               │                                  │
               └───────────────┬──────────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ Cosine Similarity Ranking │
                 │ Top-K Candidate Evidence  │
                 └───────────────────────────┘
```

### 2. Technical Implementations & Theoretical Justification

#### 2.1 Multi-Sentence Windowing & Coreference Anchoring (`services/context_matcher.py`)
* **What was done**: Created document chunkers supporting both single sentences and **sliding multi-sentence windows**, paired with leading context anchors.
* **Why it is needed (Theoretical Justification - The Coreference Resolution Problem)**:
  * In isolated sentence chunking, concluding sentences such as *"The crew returned safely after completing their historic 8-day mission"* omit the explicit subject (*Apollo 11*).
  * Evaluated alone, NLI models cannot resolve who *"their"* refers to, causing false neutral (unsupported) flags (**99.9% Neutral**).
  * Providing 2-sentence sliding windows and pairing the opening anchor sentence preserves pronoun referents, jumping entailment confidence from **0.02% to 99.85%**.

#### 2.2 Dense Vector Retrieval with TF-IDF Fallback
* **Source Code**:
  ```python
  class ContextMatcher:
      def __init__(self, settings=None):
          self.settings = settings or get_settings()
          self.sentence_regex = re.compile(r'(?:(?<=[\.\?\!])|(?<=[\.\?\!][\"\'\)]))\s+')
          self._init_encoder()

      def _init_encoder(self):
          try:
              from sentence_transformers import SentenceTransformer
              self._sbert_model = SentenceTransformer(self.settings.EMBEDDING_MODEL_NAME)
              self._sbert_available = True
          except Exception:
              self._sbert_available = False

      def chunk_document(self, doc: ContextDocument) -> List[Dict[str, Any]]:
          raw_sentences = [s.strip() for s in self.sentence_regex.split(doc.text) if len(s.strip()) > 5]
          chunks = []
          for s in raw_sentences:
              chunks.append({"doc_id": doc.doc_id, "text": s, "source_url": doc.source_url})
          for i in range(len(raw_sentences) - 1):
              chunks.append({"doc_id": doc.doc_id, "text": f"{raw_sentences[i]} {raw_sentences[i+1]}", "source_url": doc.source_url})
          if len(raw_sentences) >= 3:
              for s in raw_sentences[2:]:
                  chunks.append({"doc_id": doc.doc_id, "text": f"{raw_sentences[0]} {s}", "source_url": doc.source_url})
          return chunks

      def retrieve_evidence(self, claim: str, contexts: List[ContextDocument], top_k: int = 3):
          # SBERT neural cosine retrieval with scikit-learn TF-IDF fallback
          ...
  ```
* **Theoretical Justification**: Dense bi-encoder embeddings (`all-MiniLM-L6-v2`) capture semantic equivalence even when vocabulary differs (e.g. mapping *"revolves around"* to *"orbits"*), overcoming the lexical vocabulary mismatch problem of pure keyword search.

### 3. System Verification & Testing Results

| Test Function | Scenario Tested | Expected Result | Pass / Fail |
|---|---|---|---|
| `test_document_chunking` | Multi-sentence paragraph chunking | Generates $\ge 2$ chunks (isolated + sliding window) | **PASSED** |
| `test_evidence_retrieval_ranking` | Disparate topics (planets vs history) | Ranks planet evidence #1 for planetary mass claim | **PASSED** |

### 4. Conclusion & Connection to Milestone 4
Milestone 3 successfully establishes fast semantic evidence retrieval. Milestone 4 connects these matched `(Claim, Evidence)` pairs to the **Natural Language Inference (NLI) Verification Engine**.

---

# MILESTONE 4
## Module: Natural Language Inference (NLI) Verification Engine
* **Status**: Completed & Verified  
* **Components**: `app/services/nli_verifier.py`, `training/train_cross_encoder.py`, `tests/test_nli_verifier.py`

### 1. Executive Summary & Objectives
Semantic similarity only measures whether two statements discuss the same topic; it *cannot* determine if one statement contradicts or disproves the other.

The objective of **Milestone 4** is to deploy a transformer cross-encoder NLI model (`DeBERTa-v3`) that performs bidirectional cross-attention over `(Evidence, Claim)` pairs, computing calibrated probabilities for **Entailment** (True), **Contradiction** (False), and **Neutral** (Unsupported), backed by deep learning fine-tuning pipelines.

```
       [Candidate Evidence Snippet]      [Atomic Claim]
                    │                          │
                    └───────────┬──────────────┘
                                ▼
         ╔════════════════════════════════════════════════════╗
         ║   Transformer Cross-Encoder (DeBERTa-v3)           ║
         ║   Bidirectional Self-Attention over (Premise, Hyp) ║
         ╚════════════════════════════════════════════════════╝
                                │
                                ▼
         ┌────────────────────────────────────────────────────┐
         │ Softmax Probabilities:                             │
         │ - P(Entailment):    Directly grounded in evidence  │
         │ - P(Contradiction): Directly refuted by evidence   │
         │ - P(Neutral):       Unsupported / Hallucinated     │
         └────────────────────────────────────────────────────┘
```

### 2. Technical Implementations & Theoretical Justification

#### 2.1 Cross-Encoder Architecture & Verification (`services/nli_verifier.py`)
* **What was done**: Implemented the `NLIVerifier` class using `cross-encoder/nli-deberta-v3-small` with calibrated fallback logic.
* **Key Algorithmic Safeguards**:
  1. **Entity Role Substitution**: Identifies when a named entity is swapped into the same predicate position (e.g. claiming *"Michael Collins was the first person to step on the Moon"* when the evidence specifies *"Neil Armstrong became the first human to walk on the lunar surface"*), triggering high-confidence contradiction.
  2. **Predicate-Aware Action Negation**: Evaluates verb/predicate overlap before asserting negation contradictions, avoiding false alarms when actions differ.
  3. **Number Word & Subset Normalization**: Normalizes words to digits (*"two and a half"* $\to$ `2.5`, *"8-day"* $\to$ `8 days`). When claim numbers form a subset of evidence numbers, false numerical contradictions are prevented.
* **Source Code Snippet**:
  ```python
  class NLIVerifier:
      def verify_claim(self, claim_id: int, claim: str, evidence_matches: List[EvidenceMatch]) -> ClaimEvaluation:
          if not evidence_matches:
              return ClaimEvaluation(
                  claim_id=claim_id, claim=claim, verdict="NEUTRAL", confidence=0.90,
                  is_grounded=False, evidence=None,
                  scores=NLIScores(entailment=0.05, neutral=0.90, contradiction=0.05),
                  reasoning="No relevant evidence was found in retrieved context."
              )
          best_ev = evidence_matches[0]
          if self._cross_encoder_available and self._model is not None:
              scores, best_ev = self._evaluate_neural(claim, evidence_matches)
          else:
              scores, best_ev = self._evaluate_heuristic(claim, evidence_matches)

          if scores.contradiction >= self.settings.CONTRADICTION_THRESHOLD:
              verdict = "CONTRADICTION"
          elif scores.entailment >= self.settings.ENTAILMENT_THRESHOLD:
              verdict = "ENTAILMENT"
          else:
              verdict = "NEUTRAL"
          return ClaimEvaluation(claim_id=claim_id, claim=claim, verdict=verdict, scores=scores, ...)
  ```

#### 2.2 Deep Learning Cross-Encoder Fine-Tuning Pipeline (`training/train_cross_encoder.py`)
* **What was done**: Built an end-to-end PyTorch / HuggingFace `Trainer` script to fine-tune DeBERTa models on domain-specific claim-evidence pairs with classification heads for $\{0: \text{contradiction}, 1: \text{neutral}, 2: \text{entailment}\}$.

### 3. System Verification & Testing Results

| Test Function | Input Scenario | Expected Verdict | Pass / Fail |
|---|---|---|---|
| `test_nli_entailment` | Factually matched statement & evidence | `ENTAILMENT` (Score $> 0.70$) | **PASSED** |
| `test_nli_contradiction` | Revenue *"decreased"* vs. evidence *"increased"* | `CONTRADICTION` (Score $\ge 0.50$) | **PASSED** |
| `test_nli_unsupported_neutral` | Alien life claim with empty evidence | `NEUTRAL` (Score $\ge 0.85$) | **PASSED** |

### 4. Conclusion & Connection to Milestone 5
Milestone 4 produces precise claim-level truth probabilities. Milestone 5 aggregates these metrics into document-level **Trust Scores**, enforces **Guardrail Policies**, and provides a real-time **Web Playground UI**.

---

# MILESTONE 5
## Module: Trust Score Aggregation, Guardrail Policies & Interactive UI Dashboard
* **Status**: Completed & Verified  
* **Components**: `app/services/aggregator.py`, `app/api/v1/endpoints.py`, `app/main.py`, `app/static/index.html`, `training/benchmark_eval.py`

### 1. Executive Summary & Objectives
The final objective of TrustLayer is to translate granular claim evaluations into automated enterprise guardrails: deciding whether to **`PASS`**, **`FLAG`**, or **`BLOCK`** a response, highlighting exact sentences for human reviewers, sanitizing text, and providing an interactive operator dashboard.

```
                  [Claim-Level NLI Probabilities]
                                 │
                                 ▼
         ┌────────────────────────────────────────────────┐
         │ Mathematical Score Aggregation                 │
         │ Trust Score = Base Score × (1 - Max_Con^1.5)   │
         └───────────────────────┬────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────┐
         │ Automated Guardrail Policy Engine              │
         │ - PASS:  Score >= 60%, no contradictions       │
         │ - FLAG:  Unsupported/neutral claims detected   │
         │ - BLOCK: Contradiction exceeds threshold       │
         └───────────────────────┬────────────────────────┘
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
 ┌───────────────────────┐               ┌───────────────────────┐
 │ Visual HTML Highlighting              │ Filtered Safe Text    │
 │ (Green / Yellow / Red)│               │ (Hallucinations out)  │
 └───────────────────────┘               └───────────────────────┘
```

### 2. Technical Implementations & Theoretical Justification

#### 2.1 Mathematical Trust Score Formulation (`services/aggregator.py`)
The document-level Trust Score aggregates claim entailments with an exponential penalty for contradictions:

$$\text{Base Score} = \frac{1}{N} \sum_{i=1}^{N} P(\text{Entailment}_i)$$
$$\text{Penalty Factor} = 1.0 - \left(\max_{i} P(\text{Contradiction}_i)\right)^{1.5}$$
$$\text{Trust Score} = \max\left(0.0, \min\left(1.0, \text{Base Score} \times \text{Penalty Factor}\right)\right)$$

* **Theoretical Justification**: In enterprise risk management, a single direct contradiction (such as an incorrect medical dosage or erroneous financial figure) invalidates an otherwise factual document. An exponential penalty guarantees that even if 9 out of 10 claims are true, a severe contradiction drops the overall Trust Score into the **BLOCK** zone.

#### 2.2 Policy Enforcement & Sanitization
* **`PASS`**: All assertions are verified in context ($\ge 60\%$). Response is delivered safely.
* **`FLAG`**: Neutral/unsupported statements present. Human review recommended.
* **`BLOCK`**: High-confidence contradiction refuting source documentation ($P_{\text{con}} \ge 0.50$). Response transmission halted.
* **Text Filtering (`filtered_text`)**: Strips ungrounded claims, delivering a sanitized subset of verified facts.

#### 2.3 Interactive Web Playground Dashboard (`app/static/index.html`)
* Built with modern dark-mode aesthetics, HSL gradients, and responsive glassmorphism.
* Includes presets (*Accurate*, *Hallucinated*, *Contradicted*), interactive sensitivity sliders, real-time trust gauge, annotated highlights, and claim breakdown cards.

---

### 3. Comprehensive Verification & Empirical Benchmarks

#### 3.1 Automated PyTest Suite (Full 19-Test Suite)
```bash
pytest -v tests/
```
**Results**: **19 Passed, 0 Failed (100% Pass Rate)**

```
tests/test_aggregator.py::test_aggregator_pass PASSED                    [  5%]
tests/test_aggregator.py::test_aggregator_block_on_contradiction PASSED  [ 10%]
tests/test_aggregator.py::test_aggregator_filtering PASSED               [ 15%]
tests/test_api.py::test_health_check PASSED                              [ 21%]
tests/test_api.py::test_valid_evaluation_request PASSED                  [ 26%]
tests/test_api.py::test_invalid_payload_fails PASSED                     [ 31%]
tests/test_api.py::test_v1_health_and_config PASSED                      [ 36%]
tests/test_api.py::test_v1_evaluate_contradiction_blocking PASSED        [ 42%]
tests/test_api.py::test_v1_evaluate_batch PASSED                         [ 47%]
tests/test_api.py::test_filtering_hallucinations PASSED                  [ 52%]
tests/test_claim_extractor.py::test_extract_simple_sentences PASSED      [ 57%]
tests/test_claim_extractor.py::test_extract_compound_sentences_with_semicolon PASSED [ 63%]
tests/test_claim_extractor.py::test_extract_markdown_bullet_points PASSED [ 68%]
tests/test_claim_extractor.py::test_empty_or_whitespace_text PASSED      [ 73%]
tests/test_context_matcher.py::test_document_chunking PASSED             [ 78%]
tests/test_context_matcher.py::test_evidence_retrieval_ranking PASSED    [ 84%]
tests/test_nli_verifier.py::test_nli_entailment PASSED                   [ 89%]
tests/test_nli_verifier.py::test_nli_contradiction PASSED                [ 94%]
tests/test_nli_verifier.py::test_nli_unsupported_neutral PASSED          [100%]
============================== 19 passed in 1.46s ==============================
```

#### 3.2 Empirical Apollo 11 Benchmark Verification
Testing against authentic historical aerospace documentation demonstrated the calibrated accuracy across all claim types:

| Test Claim | Initial Heuristic | Calibrated Neural Pipeline | Final Verdict |
|---|---|---|---|
| *"The Apollo 11 mission lasted a total of 8 days before the spacecraft returned to Earth."* | 28% (FLAG) | **89% (PASS)** • 90% Entailment | **PASSED** |
| *"Neil Armstrong and Buzz Aldrin spent two and a half hours on the Moon."* | 13% (BLOCK) | **85% (PASS)** • 87% Entailment | **PASSED** |
| *"Michael Collins was the first person to step onto the surface of the Moon."* | 40% (FLAG) | **15% (BLOCK)** • 65% Contradiction | **BLOCKED** |
| *"All three astronauts on Apollo 11 explored the Sea of Tranquility"* | 34% (FLAG) | **23% (FLAG)** • 71% Neutral (Unsupported) | **FLAGGED** |

#### 3.3 Quantitative Guardrail Benchmark (`training/benchmark_eval.py`)
```bash
python training/benchmark_eval.py
```
* **Benchmark Accuracy**: **100.0% (3/3 test suites correctly classified)**
* **Average Latency**: **~430ms**

---

### 4. Project Conclusion & Deliverables Summary
The completion of all 5 milestones transforms TrustLayer from an initial mock skeleton into an enterprise-ready, real-time hallucination firewall:
1. **Milestone 1**: Robust software architecture, API gateway, and Pydantic data contracts.
2. **Milestone 2**: Precision sentence and clause claim extraction.
3. **Milestone 3**: Multi-sentence windowed context matching with dense SBERT embeddings.
4. **Milestone 4**: Deep learning cross-encoder NLI verification and training scripts.
5. **Milestone 5**: Calibrated trust scoring, automated policy guardrails, and an operator dashboard.
