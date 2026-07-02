# mlflow logging - one run per query, plus a batch eval helper

import json
import os
import time

import mlflow

# mlflow 3.x blocks file store unless this is set
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL_JSON = os.path.join(ROOT, "eval_results.json")

mlflow.set_tracking_uri("file:" + os.path.join(ROOT, "mlruns").replace("\\", "/"))
mlflow.set_experiment("youtube-rag")


def logged_query(query, handler, **kwargs):
    with mlflow.start_run(run_name="query"):
        mlflow.log_param("query", query[:250])
        t0 = time.time()
        out = handler(query, **kwargs)
        latency = time.time() - t0

        mlflow.log_param("intent", out.get("intent", "?"))
        if out.get("retrieval"):
            mlflow.log_param("retrieval", out["retrieval"])
        mlflow.log_metric("latency_s", latency)
        mlflow.log_metric("n_sources", len(out.get("sources", [])))
        mlflow.log_metric("answer_len", len(out.get("answer", "")))
    return out


def evaluate(handler, test_questions, retrieval_mode="hybrid"):
    rows = []
    latencies, src_counts = [], []

    with mlflow.start_run(run_name="eval"):
        for q in test_questions:
            t0 = time.time()
            out = handler(q, retrieval_mode=retrieval_mode)
            lat = time.time() - t0
            latencies.append(lat)
            n_src = len(out.get("sources", []))
            src_counts.append(n_src)
            rows.append({
                "question": q,
                "intent": out.get("intent"),
                "retrieval": out.get("retrieval"),
                "latency_s": round(lat, 3),
                "n_sources": n_src,
                "answer_preview": (out.get("answer") or "")[:200],
            })

        summary = {
            "n_questions": len(test_questions),
            "retrieval_mode": retrieval_mode,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3),
            "avg_sources": round(sum(src_counts) / len(src_counts), 2),
            "questions": rows,
        }

        mlflow.log_metric("n_questions", summary["n_questions"])
        mlflow.log_metric("avg_latency_s", summary["avg_latency_s"])
        mlflow.log_metric("avg_sources", summary["avg_sources"])
        mlflow.log_param("retrieval_mode", retrieval_mode)

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("saved", EVAL_JSON)
    return summary
