# RAG helpers - notebooks + dashboard both import from here
# Lab 7 style: embed -> chroma -> retrieve -> ollama chains

import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")

EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

_embedder = None
_collection = None
_chain_qa = None
_chain_summ = None

# tf-idf index for lexical search - built once on first lexical/hybrid call
_tfidf = None
_tfidf_matrix = None
_tfidf_docs = None
_tfidf_meta = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def load_comments():
    for name in ["comments_topics.csv", "comments_sentiment.csv",
                 "comments_meta.csv", "clean_comments.csv"]:
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            return pd.read_csv(path).dropna(subset=["clean_text"]).reset_index(drop=True)
    raise FileNotFoundError("no processed csv found - run notebooks 1-4 first")


def build_index(rebuild=False, batch_size=512):
    import chromadb
    global _collection

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection("comments")
        except Exception:
            pass

    _collection = client.get_or_create_collection("comments")

    if _collection.count() > 0 and not rebuild:
        print(f"reusing chroma index ({_collection.count()} docs)")
        return _collection

    df = load_comments()
    embedder = get_embedder()
    texts = df["clean_text"].astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        sub = df.iloc[start:start + batch_size]
        embs = embedder.encode(chunk, show_progress_bar=False).tolist()

        metas = []
        for _, row in sub.iterrows():
            metas.append({
                "video_id": str(row.get("video_id", "")),
                "sentiment": str(row.get("sentiment", "")),
                "topic": int(row["topic"]) if "topic" in row and pd.notna(row.get("topic")) else -1,
                "likes": int(row["like_count"]) if pd.notna(row.get("like_count", 0)) else 0,
                "keywords": str(row.get("keywords", "")),
            })

        ids = [f"c{start + i}" for i in range(len(chunk))]
        _collection.add(ids=ids, documents=chunk, embeddings=embs, metadatas=metas)
        print(f"  indexed {start + len(chunk)}/{len(texts)}")

    print("done,", _collection.count(), "in chroma")
    return _collection


def get_collection():
    global _collection
    if _collection is None:
        build_index()
    return _collection


def _build_tfidf():
    """lazy tf-idf over clean_text for lexical retrieval"""
    global _tfidf, _tfidf_matrix, _tfidf_docs, _tfidf_meta
    if _tfidf is not None:
        return

    from sklearn.feature_extraction.text import TfidfVectorizer

    df = load_comments()
    _tfidf_docs = df["clean_text"].astype(str).tolist()
    _tfidf_meta = [_row_meta(row) for _, row in df.iterrows()]

    _tfidf = TfidfVectorizer(stop_words="english", max_features=25000, ngram_range=(1, 2))
    _tfidf_matrix = _tfidf.fit_transform(_tfidf_docs)


def _row_meta(row):
    return {
        "video_id": str(row.get("video_id", "")),
        "sentiment": str(row.get("sentiment", "")),
        "topic": int(row["topic"]) if "topic" in row and pd.notna(row.get("topic")) else -1,
        "likes": int(row["like_count"]) if pd.notna(row.get("like_count", 0)) else 0,
        "keywords": str(row.get("keywords", "")),
    }


def _apply_metadata_filter(indices, sentiment=None, topic=None):
    if sentiment is None and topic is None:
        return indices
    out = []
    for i in indices:
        m = _tfidf_meta[i]
        if sentiment and m.get("sentiment") != sentiment:
            continue
        if topic is not None and m.get("topic") != topic:
            continue
        out.append(i)
    return out


def retrieve_semantic(query, k=6, sentiment=None, topic=None):
    col = get_collection()
    q_emb = get_embedder().encode([query]).tolist()
    where_parts = []
    if sentiment:
        where_parts.append({"sentiment": sentiment})
    if topic is not None:
        where_parts.append({"topic": topic})
    where = None
    if len(where_parts) == 1:
        where = where_parts[0]
    elif len(where_parts) > 1:
        where = {"$and": where_parts}

    res = col.query(query_embeddings=q_emb, n_results=k, where=where)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def retrieve_lexical(query, k=6, sentiment=None, topic=None):
    _build_tfidf()
    q_vec = _tfidf.transform([query])
    scores = (_tfidf_matrix @ q_vec.T).toarray().ravel()

    # grab a bigger pool if we're filtering by metadata
    pool = min(len(scores), max(k * 20, 200))
    top_idx = np.argsort(scores)[::-1][:pool]
    top_idx = _apply_metadata_filter(top_idx, sentiment=sentiment, topic=topic)
    top_idx = top_idx[:k]

    hits = []
    for i in top_idx:
        if scores[i] <= 0:
            break
        hits.append((_tfidf_docs[i], _tfidf_meta[i]))
    return hits


def retrieve_hybrid(query, k=6, sentiment=None, topic=None, alpha=0.6):
    """alpha=weight on semantic score; rest is lexical"""
    _build_tfidf()
    col = get_collection()

    # semantic side
    q_emb = get_embedder().encode([query]).tolist()
    where_parts = []
    if sentiment:
        where_parts.append({"sentiment": sentiment})
    if topic is not None:
        where_parts.append({"topic": topic})
    where = where_parts[0] if len(where_parts) == 1 else ({"$and": where_parts} if where_parts else None)

    sem_k = min(k * 5, 50)
    sem = col.query(query_embeddings=q_emb, n_results=sem_k, where=where)
    sem_docs = sem["documents"][0]
    sem_dist = sem["distances"][0]
    sem_meta = sem["metadatas"][0]

    # lexical side
    q_vec = _tfidf.transform([query])
    lex_scores = (_tfidf_matrix @ q_vec.T).toarray().ravel()
    lex_pool = np.argsort(lex_scores)[::-1][:sem_k * 2]
    lex_pool = _apply_metadata_filter(lex_pool, sentiment=sentiment, topic=topic)

    # merge by doc text
    combined = {}

    if sem_dist:
        max_d = max(sem_dist) or 1.0
        for doc, meta, dist in zip(sem_docs, sem_meta, sem_dist):
            sem_score = 1.0 - (dist / max_d)
            combined[doc] = {"meta": meta, "score": alpha * sem_score}

    if len(lex_pool):
        max_lex = max(lex_scores[i] for i in lex_pool) or 1.0
        for i in lex_pool:
            doc = _tfidf_docs[i]
            lex_score = lex_scores[i] / max_lex
            if doc in combined:
                combined[doc]["score"] += (1 - alpha) * lex_score
            else:
                combined[doc] = {"meta": _tfidf_meta[i], "score": (1 - alpha) * lex_score}

    ranked = sorted(combined.items(), key=lambda x: (x[1]["score"], len(x[0].split())), reverse=True)
    return [(doc, info["meta"]) for doc, info in ranked[:k]]


MIN_COMMENT_WORDS = 4
MIN_COMMENT_CHARS = 30


def _retrieval_query(user_q):
    """strip filler so retrieval focuses on the actual topic"""
    q = user_q.lower()
    noise = (
        r"\b(what|whats|what's|the|take|on|people|say|says|said|think|"
        r"about|opinion|opinions|overall|how|is|are|do|does|did|"
        r"anyone|mention|tell|me|give|summary|summarize)\b"
    )
    q = re.sub(noise, " ", q)
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q if len(q) > 2 else user_q


def _good_comment(doc):
    t = doc.strip()
    words = t.split()
    if len(t) < MIN_COMMENT_CHARS or len(words) < MIN_COMMENT_WORDS:
        return False
    # "camera" / "the camera" / "battery life?" alone aren't useful
    if len(words) <= 3 and len(t) < 50:
        return False
    return True


def _pick_hits(hits, k):
    """drop junk + near-duplicates, keep substantive comments"""
    seen = set()
    good = []
    for doc, meta in hits:
        key = doc.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        if _good_comment(doc):
            good.append((doc, meta))
    if len(good) < max(3, k // 2):
        # relax a bit if the pool is thin
        seen.clear()
        for doc, meta in hits:
            key = doc.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            if len(doc.strip()) >= 20 and len(doc.split()) >= 3:
                good.append((doc, meta))
    return good[:k]


def _fetch_hits(query, k, mode, sentiment=None, topic=None):
    search_q = _retrieval_query(query)
    pool = max(k * 5, 40)
    hits = retrieve(search_q, k=pool, mode=mode, sentiment=sentiment, topic=topic)
    picked = _pick_hits(hits, k)
    if len(picked) < max(3, k // 2) and search_q != query:
        extra = retrieve(query, k=pool, mode=mode, sentiment=sentiment, topic=topic)
        picked = _pick_hits(hits + extra, k)
    return picked


def retrieve(query, k=6, mode="semantic", sentiment=None, topic=None):
    """mode: semantic | lexical | metadata | hybrid"""
    mode = (mode or "semantic").lower()
    if mode == "lexical":
        return retrieve_lexical(query, k=k, sentiment=sentiment, topic=topic)
    if mode == "metadata":
        return retrieve_semantic(query, k=k, sentiment=sentiment, topic=topic)
    if mode == "hybrid":
        return retrieve_hybrid(query, k=k, sentiment=sentiment, topic=topic)
    return retrieve_semantic(query, k=k, sentiment=sentiment, topic=topic)


def _build_chains():
    global _chain_qa, _chain_summ
    if _chain_qa is not None:
        return

    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)
    parser = StrOutputParser()

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You answer questions using YouTube comments about iPhone 16 reviews.\n"
            "Use ONLY the comments below. Give a direct, useful answer in 2-4 sentences.\n"
            "State actual opinions (praise, complaints, comparisons). "
            "If comments disagree, say that. "
            "Do not describe writing style or how often words appear.\n"
            "If nothing relevant, say you did not find comments on that topic.\n\n"
            "Comments:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    summ_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You summarize what YouTube commenters think about iPhone 16 reviews.\n"
            "Use ONLY the comments below. Write 3-5 sentences covering:\n"
            "- what people praise or complain about\n"
            "- specific details (camera, battery, price, display, etc.)\n"
            "- whether views sound mostly positive, negative, or mixed\n\n"
            "Only discuss the topic in the question — don't mention unrelated features.\n"
            "Do NOT comment on tone, punctuation, usernames, or how often a word is used.\n"
            "Do NOT say things like 'users mention the word camera' — state their actual views.\n\n"
            "Comments:\n{context}\n\n"
            "Topic: {question}\n"
            "Summary:"
        ),
    )

    _chain_qa = qa_prompt | llm | parser
    _chain_summ = summ_prompt | llm | parser


def _fmt(hits):
    lines = []
    for doc, meta in hits:
        s = meta.get("sentiment", "?")
        lines.append(f"- [{s}] {doc}")
    return "\n".join(lines)


def answer_question(query, k=6, sentiment=None, mode="hybrid"):
    _build_chains()
    hits = _fetch_hits(query, k=k, mode=mode, sentiment=sentiment)
    answer = _chain_qa.invoke({"context": _fmt(hits), "question": query})
    return answer, hits


def summarize(query, k=10, sentiment=None, mode="hybrid"):
    _build_chains()
    hits = _fetch_hits(query, k=k, mode=mode, sentiment=sentiment)
    summary = _chain_summ.invoke({"context": _fmt(hits), "question": query})
    return summary, hits


if __name__ == "__main__":
    build_index()
    q = "battery life"
    for m in ["semantic", "lexical", "hybrid"]:
        hits = retrieve(q, k=3, mode=m)
        print(m, "->", len(hits), "hits")
