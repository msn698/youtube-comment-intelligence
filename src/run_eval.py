"""Batch eval that works without ollama for routing/retrieval; tries LLM if available."""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent
import monitoring
import rag_pipeline as rag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def retrieval_eval():
    rag.build_index()
    queries = ["battery life", "camera", "price"]
    modes = ["semantic", "lexical", "hybrid", "metadata"]
    rows = []
    for q in queries:
        for m in modes:
            t0 = time.time()
            if m == "metadata":
                hits = rag.retrieve(q, k=6, mode=m, sentiment="negative")
            else:
                hits = rag.retrieve(q, k=6, mode=m)
            rows.append({
                "query": q,
                "mode": m,
                "latency_s": round(time.time() - t0, 4),
                "n_hits": len(hits),
            })
    return rows


def route_eval():
    tests = [
        ("summarize what people say about the screen", "summarize"),
        ("how positive are the comments overall?", "sentiment"),
        ("did anyone mention overheating?", "qa"),
        ("what do people think about the price?", "qa"),
    ]
    rows = []
    for q, expected in tests:
        got = agent.route(q)
        rows.append({"question": q, "expected": expected, "got": got, "ok": got == expected})
    return rows


def main():
    summary = {
        "ollama_available": ollama_up(),
        "routing": route_eval(),
        "retrieval": retrieval_eval(),
    }

    if summary["ollama_available"]:
        test_qs = [
            "what do people think about the battery?",
            "summarize opinions on the camera",
            "how negative are the comments?",
        ]
        summary["agent"] = monitoring.evaluate(agent.handle, test_qs, retrieval_mode="hybrid")
    else:
        summary["agent_note"] = "ollama was not running - agent LLM eval skipped; routing + retrieval still measured"

    out = os.path.join(ROOT, "eval_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
