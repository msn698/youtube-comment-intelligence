# routes user questions to qa / summarize / sentiment tools
# DSPy classifier from lab 7, keyword fallback if ollama is down

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
import rag_pipeline as rag

lm = dspy.LM(
    "ollama/llama3.2:3b",
    api_base="http://localhost:11434",
    api_key=None,
    temperature=0,
    cache=False,
)
dspy.configure(lm=lm)


class IntentSignature(dspy.Signature):
    """Pick one intent for a YouTube comment question: qa, summarize, or sentiment.
    qa = specific question about what people said
    summarize = wants an overview
    sentiment = wants positive/negative breakdown"""

    question: str = dspy.InputField()
    intent: str = dspy.OutputField()


_classifier = dspy.Predict(IntentSignature)


def route(query):
    try:
        result = _classifier(question=query)
        label = result.intent.strip().lower().split()[0]
        if label in ("qa", "summarize", "sentiment"):
            return label
    except Exception:
        pass

    ql = query.lower()
    if any(w in ql for w in ["summar", "overview", "overall", "what are people saying", "take on", "general"]):
        return "summarize"
    if any(w in ql for w in ["positive", "negative", "sentiment", "percent", "how many", "breakdown"]):
        return "sentiment"
    return "qa"


def sentiment_stats():
    df = rag.load_comments()
    if "sentiment" not in df.columns:
        return "run notebook 4 first - no sentiment column yet"
    counts = df["sentiment"].value_counts()
    total = len(df)
    lines = [f"  {s}: {n:,} ({100 * n / total:.1f}%)" for s, n in counts.items()]
    return f"sentiment over {total:,} comments:\n" + "\n".join(lines)


def handle(query, retrieval_mode="hybrid"):
    intent = route(query)

    if intent == "summarize":
        text, hits = rag.summarize(query, mode=retrieval_mode)
        return {"intent": intent, "answer": text, "sources": hits, "retrieval": retrieval_mode}

    if intent == "sentiment":
        return {"intent": intent, "answer": sentiment_stats(), "sources": [], "retrieval": None}

    text, hits = rag.answer_question(query, mode=retrieval_mode)
    return {"intent": intent, "answer": text, "sources": hits, "retrieval": retrieval_mode}
