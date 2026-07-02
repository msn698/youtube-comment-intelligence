# streamlit dashboard for the YT comment intelligence project
# run from final/:  streamlit run app/dashboard.py

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import agent
import rag_pipeline as rag


st.set_page_config(page_title="YouTube Comment Intelligence", layout="wide")
st.title("YouTube Comment Intelligence Engine")
st.caption("CSCI370 - iPhone 16 review comments scraped from YouTube")


@st.cache_data
def load_data():
    return rag.load_comments()


df = load_data()

tab_overview, tab_ask = st.tabs(["Overview", "Ask"])

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Comments", f"{len(df):,}")
    c2.metric("Videos", df["video_id"].nunique() if "video_id" in df.columns else "-")
    c3.metric("Replies", int(df["is_reply"].sum()) if "is_reply" in df.columns else "-")

    if "sentiment" in df.columns:
        st.subheader("Sentiment")
        st.bar_chart(df["sentiment"].value_counts())

    if "topic" in df.columns:
        st.subheader("Top BERTopic clusters")
        top = df[df["topic"] != -1]["topic"].value_counts().head(10)
        st.bar_chart(top)

    cols = [c for c in ["text", "sentiment", "keywords"] if c in df.columns]
    st.subheader("Sample rows")
    st.dataframe(df[cols].head(40), use_container_width=True)


with tab_ask:
    st.subheader("Ask the comments")
    mode = st.selectbox(
        "Retrieval mode",
        ["hybrid", "semantic", "lexical", "metadata"],
        help="hybrid = semantic + tf-idf; metadata = filter + semantic",
    )
    sentiment_filter = st.selectbox("Optional sentiment filter", ["(none)", "positive", "negative", "neutral"])
    query = st.text_input("Question", placeholder="what do people say about the camera?")

    if st.button("Run") and query:
        sent = None if sentiment_filter == "(none)" else sentiment_filter
        with st.spinner("..."):
            # route through agent; pass sentiment filter via rag when needed
            intent = agent.route(query)
            if intent == "sentiment":
                out = agent.handle(query, retrieval_mode=mode)
            elif intent == "summarize":
                ans, hits = rag.summarize(query, mode=mode, sentiment=sent)
                out = {"intent": intent, "answer": ans, "sources": hits, "retrieval": mode}
            else:
                ans, hits = rag.answer_question(query, mode=mode, sentiment=sent)
                out = {"intent": intent, "answer": ans, "sources": hits, "retrieval": mode}

        st.markdown(f"**Routed to:** `{out['intent']}`  |  **Retrieval:** `{out.get('retrieval', '-')}`")
        st.write(out["answer"])
        if out.get("sources"):
            with st.expander("Retrieved comments"):
                for doc, meta in out["sources"]:
                    st.markdown(f"- *({meta.get('sentiment', '?')})* {doc[:300]}")
