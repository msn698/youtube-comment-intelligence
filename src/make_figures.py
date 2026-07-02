"""Save matplotlib figures from real csvs into screenshots/ for the report."""
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "screenshots")
os.makedirs(OUT, exist_ok=True)


def sentiment_chart():
    df = pd.read_csv(os.path.join(DATA, "comments_sentiment.csv"))
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color=["#6c757d", "#28a745", "#dc3545"])
    ax.set_title("Sentiment labels (VADER) - iPhone 16 YouTube comments")
    ax.set_ylabel("comment count")
    ax.set_xlabel("sentiment")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 150, f"{v:,}", ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "sentiment_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def topic_chart():
    df = pd.read_csv(os.path.join(DATA, "comments_topics.csv"))
    top = df[df["topic"] != -1]["topic"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    top.sort_values().plot(kind="barh", ax=ax, color="#4c78a8")
    ax.set_title("Top 10 BERTopic clusters (by comment count)")
    ax.set_xlabel("comments")
    fig.tight_layout()
    path = os.path.join(OUT, "top_topics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def vader_hist():
    df = pd.read_csv(os.path.join(DATA, "comments_sentiment.csv"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["vader_score"], bins=40, color="#72b7b2", edgecolor="white")
    ax.axvline(df["vader_score"].mean(), color="red", linestyle="--", label=f"mean={df['vader_score'].mean():.3f}")
    ax.set_title("VADER compound score distribution")
    ax.set_xlabel("compound score")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT, "vader_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    sentiment_chart()
    topic_chart()
    vader_hist()
