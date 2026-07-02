"""
Pull real numbers from the csvs for the report.
Run from final/:  python src/collect_stats.py
Writes stats/project_stats.json
"""

import ast
import json
import os
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "stats")
os.makedirs(OUT, exist_ok=True)


def load(name):
    return pd.read_csv(os.path.join(DATA, name))


def top_entities(series, n=15):
    c = Counter()
    for val in series.dropna():
        try:
            ents = ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else None
            if ents is None and isinstance(val, str):
                ents = [val]
            if isinstance(ents, list):
                for e in ents:
                    if isinstance(e, tuple) and len(e) >= 1:
                        c[str(e[0])] += 1
                    elif isinstance(e, str) and e.strip():
                        c[e.strip()] += 1
        except (ValueError, SyntaxError):
            if isinstance(val, str) and val.strip():
                c[val.strip()] += 1
    return c.most_common(n)


def main():
    raw = load("raw_comments.csv")
    clean = load("clean_comments.csv")
    meta = load("comments_meta.csv")
    sent = load("comments_sentiment.csv")
    topics = load("comments_topics.csv")

    stats = {
        "topic_domain": "iPhone 16 review (YouTube comments)",
        "dataset": {
            "raw_comments": int(len(raw)),
            "after_cleaning": int(len(clean)),
            "unique_videos": int(raw["video_id"].nunique()),
            "top_level_comments": int((~raw["is_reply"]).sum()),
            "replies": int(raw["is_reply"].sum()),
            "avg_likes": round(float(raw["like_count"].mean()), 2),
            "median_likes": int(raw["like_count"].median()),
        },
        "preprocessing": {
            "dropped_on_clean": int(len(raw) - len(clean)),
            "has_clean_text": int(clean["clean_text"].notna().sum()),
            "has_lemmas": int(clean["lemmas"].notna().sum()) if "lemmas" in clean.columns else None,
        },
        "sentiment": {
            "counts": {k: int(v) for k, v in sent["sentiment"].value_counts().items()},
            "pct": {k: round(100 * v / len(sent), 1) for k, v in sent["sentiment"].value_counts().items()},
            "mean_vader": round(float(sent["vader_score"].mean()), 4) if "vader_score" in sent.columns else None,
            "mean_textblob": round(float(sent["textblob_polarity"].mean()), 4) if "textblob_polarity" in sent.columns else None,
        },
        "topics": {
            "n_topics_bertopic": int(topics[topics["topic"] != -1]["topic"].nunique()),
            "outlier_topic_minus1": int((topics["topic"] == -1).sum()),
            "top_topics_by_count": {int(k): int(v) for k, v in topics[topics["topic"] != -1]["topic"].value_counts().head(10).items()},
        },
        "ner_keywords": {
            "comments_with_entities": int(meta["entities"].notna().sum()) if "entities" in meta.columns else None,
            "top_entities": top_entities(meta["entities"]) if "entities" in meta.columns else [],
            "sample_keywords": meta["keywords"].dropna().head(5).tolist() if "keywords" in meta.columns else [],
        },
    }

    if "sentiment" in topics.columns:
        stats["topics_per_sentiment"] = {}
        for s in ["positive", "negative", "neutral"]:
            sub = topics[topics["sentiment"] == s]
            stats["topics_per_sentiment"][s] = {
                "n_comments": int(len(sub)),
                "n_distinct_topics": int(sub[sub["topic"] != -1]["topic"].nunique()),
            }

    # sample comments for report (real text, short)
    samples = []
    for label in ["positive", "negative", "neutral"]:
        row = sent[sent["sentiment"] == label].head(1)
        if len(row):
            r = row.iloc[0]
            samples.append({
                "sentiment": label,
                "text": str(r.get("text", ""))[:280],
                "clean_text": str(r.get("clean_text", ""))[:280],
                "keywords": str(r.get("keywords", ""))[:120],
            })
    stats["sample_comments"] = samples

    path = os.path.join(OUT, "project_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("wrote", path)
    return stats


if __name__ == "__main__":
    main()
