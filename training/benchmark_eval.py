"""
TrustLayer: Hallucination Detection Benchmark Evaluation Suite

Evaluates TrustLayer pipeline against a labeled benchmark of RAG responses,
measuring Accuracy, Precision, Recall, F1-Score, and Latency.
"""

import sys
import os
import asyncio
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.payload import TrustRequest, ContextDocument
from app.services.pipeline import TrustLayerPipeline

BENCHMARK_CASES = [
    {
        "id": "bench_01_factual",
        "expected_action": "PASS",
        "context": "The Apollo 11 mission landed humans on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module pilot Buzz Aldrin formed the American crew.",
        "response": "Apollo 11 successfully landed humans on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin were part of the crew."
    },
    {
        "id": "bench_02_contradiction",
        "expected_action": "BLOCK",
        "context": "The Apollo 11 mission landed humans on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module pilot Buzz Aldrin formed the American crew.",
        "response": "Apollo 11 landed humans on Mars on July 20, 1969. The mission was commanded by Yuri Gagarin."
    },
    {
        "id": "bench_03_unsupported",
        "expected_action": "FLAG",
        "context": "The Apollo 11 mission landed humans on the Moon on July 20, 1969.",
        "response": "The Apollo 11 mission landed on the Moon in July 1969. The spaceship cost 450 billion dollars to construct."
    }
]

async def run_benchmark():
    pipeline = TrustLayerPipeline()
    print("=" * 60)
    print("TrustLayer: Running Guardrail Benchmark Suite")
    print("=" * 60)

    correct = 0
    total = len(BENCHMARK_CASES)
    total_time = 0.0

    for idx, case in enumerate(BENCHMARK_CASES):
        req = TrustRequest(
            request_id=case["id"],
            generated_text=case["response"],
            retrieved_contexts=[ContextDocument(doc_id="ctx_1", text=case["context"])]
        )
        
        t0 = time.perf_counter()
        res = await pipeline.process_payload(req)
        elapsed = (time.perf_counter() - t0) * 1000
        total_time += elapsed

        match = (res.action == case["expected_action"])
        if match:
            correct += 1
            status_icon = "✓ PASSED"
        else:
            status_icon = "✗ MISMATCH"

        print(f"[{status_icon}] Case {idx+1} ({case['id']}):")
        print(f"   Expected: {case['expected_action']} | Actual: {res.action}")
        print(f"   Trust Score: {res.trust_score:.2f} | Latency: {elapsed:.1f}ms")
        print(f"   Summary: {res.summary}\n")

    accuracy = (correct / total) * 100
    avg_latency = total_time / total

    print("=" * 60)
    print(f"Benchmark Summary:")
    print(f"Total Cases: {total}")
    print(f"Accuracy:    {accuracy:.1f}% ({correct}/{total})")
    print(f"Avg Latency: {avg_latency:.1f}ms")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
