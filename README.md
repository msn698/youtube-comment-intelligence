# YouTube Comment Intelligence Engine

CSCI370 Spring 2026 project. We scrape YouTube comments on **iPhone 16 reviews**, clean them,
run NER/sentiment/topic modeling, and put everything behind a local RAG + agent stack with a
Streamlit dashboard. LLM is Ollama (`llama3.2:3b`) - same setup as Lab 7.

## What's in here

```
final/
  data/           csv outputs from the pipeline (+ contractions/acronyms lists)
  notebooks/      7 stages, run in order
  src/            rag_pipeline.py, agent.py, monitoring.py + eval/stats helpers
  app/            streamlit dashboard
  stats/          dataset/retrieval stats from src/collect_stats.py
  eval_results.json   output of src/run_eval.py
  example.env     template for the .env notebook 1 reads its API key from
```

These get created locally when you run things (gitignored, so not in the repo):

```
  venv/           virtual environment from the setup step
  .env            your API key, copied from example.env
  chroma_db/      vector index - notebook 6 builds it (~50 MB)
  mlruns/         mlflow logs - notebook 7 and src/run_eval.py write here
  screenshots/    report figures from src/make_figures.py
```

## Setup

```bash
cd final
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Ollama for stages 6-7 and the dashboard ask tab:

```bash
ollama pull llama3.2:3b
ollama serve
```

## Pipeline

| # | Notebook | Output |
|---|----------|--------|
| 1 | `notebooks/1_scrape.ipynb` | `data/raw_comments_{0,1}.csv` |
| 2 | `notebooks/2_preprocess.ipynb` | `data/clean_comments.csv` |
| 3 | `notebooks/3_ner_keywords.ipynb` | `data/comments_meta.csv` |
| 4 | `notebooks/4_sentiment.ipynb` | `data/comments_sentiment.csv` |
| 5 | `notebooks/5_topic_modeling.ipynb` | `data/comments_topics.csv` |
| 6 | `notebooks/6_rag.ipynb` | `chroma_db/` |
| 7 | `notebooks/7_agent_mlflow.ipynb` | `mlruns/`, `eval_results.json` |

All the csv outputs are already in `data/`, so any stage can be run on its own. Notebook 1 is only
needed to re-scrape from scratch - it wants a YouTube Data API key: copy `example.env` to `.env`
and fill in `YOUTUBE_API_KEY`.
Notebook 6 builds `chroma_db/` on first run; the dashboard and notebook 7 need it.

## Retrieval modes (`src/rag_pipeline.py`)

- **semantic** - chroma + sentence embeddings
- **lexical** - tf-idf cosine over comment text
- **metadata** - semantic search with sentiment/topic filters
- **hybrid** - weighted mix of semantic + lexical (default for QA)

## Dashboard

```bash
streamlit run app/dashboard.py
```

## MLflow

```bash
mlflow ui
# http://localhost:5000
```
