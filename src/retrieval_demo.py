"""Compare retrieval modes on a few fixed queries - saves real examples for the report."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_pipeline as rag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "stats", "retrieval_examples.json")

QUERIES = [
    "battery life",
    "camera quality",
    "price too expensive",
]


def main():
    rag.build_index()
    results = []
    for q in QUERIES:
        entry = {"query": q, "modes": {}}
        for mode in ["semantic", "lexical", "hybrid"]:
            hits = rag.retrieve(q, k=3, mode=mode)
            entry["modes"][mode] = [
                {"text": doc[:220], "sentiment": meta.get("sentiment"), "topic": meta.get("topic")}
                for doc, meta in hits
            ]
        # metadata-filtered example
        hits = rag.retrieve(q, k=3, mode="metadata", sentiment="negative")
        entry["modes"]["metadata_negative"] = [
            {"text": doc[:220], "sentiment": meta.get("sentiment")}
            for doc, meta in hits
        ]
        results.append(entry)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
