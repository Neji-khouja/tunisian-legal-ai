---
title: Tunisian Legal RAG
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Tunisian Legal RAG

Question answering over Tunisian legal codes (penal, civil procedure, labour, obligations and contracts, personal status,
commerce and others, 14 PDFs in `data/`). Ask in French, Arabic or English; the answer is written in your language and
every claim cites the article it comes from. If the question is vague the assistant asks for details, and if it is outside
the corpus it declines.

## How it works

The main system is an agent built with LangGraph (`agent/agent_graph.py`):

```
question -> condense -> analyze --+--> clarify (vague question, asks the user)
                                  +--> refuse  (out of scope)
                                  '--> retrieve -> grade -+-> answer -> verify -> finalize
                                          ^               |              |
                                          '-- reformulate-'              '-> repair (re-checked)
```

- **condense**: rewrites a follow-up question so it stands alone.
- **analyze**: detects language, intent (answer, clarify, refuse) and which legal codes are concerned.
- **retrieve**: the query is normalized into French legal wording once (`agent/normalize.py`), then hybrid search runs
  (below).
- **grade**: checks whether the retrieved articles are enough. If not, **reformulate** writes new queries, at most twice.
- **answer**: writes short claims, each citing article ids from the evidence.
- **verify / repair**: each claim is checked against the article it cites; unsupported claims are repaired or dropped.

Retrieval (`retriever.py`, `reranker.py`):

1. Semantic search (Solon embeddings in ChromaDB) and BM25, per query and per detected code.
2. Direct lookup when the question names an article ("article 243").
3. Pools merged, then reranked with a multilingual cross-encoder; the top 6 articles go to the model.
4. Articles cross-referenced by the evidence are added.

Indexing: each PDF is cut into one chunk per article (`splitter.py`). The chunk text starts with the code and heading path,
and very long articles are split into sub-chunks. The index is `db_v1_2/`. `config.py` has every switch;
`INDEX_V12 = False` goes back to `db_v1_1/`.

There is also a simpler one-pass pipeline (`pipeline.ask_question`: rewrite, retrieve, rerank, answer) used as the
baseline in evaluation.

| Part | Choice |
|---|---|
| Embeddings | `OrdalieTech/Solon-embeddings-large-0.1` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Vector store | ChromaDB (local files) + BM25 |
| LLM | `openai/gpt-oss-120b` through Groq (`LLM_MODEL` in `config.py`) |
| PDF parsing | PyMuPDF |
| Interfaces | Streamlit (`ui/`), FastAPI (`api/`) |

## Setup

```bash
conda create -n rag-app python=3.11
conda activate rag-app
pip install -r requirements.txt        # requirements.lock.txt is the full lock of the dev machine
```

Put your Groq key in a `.env` file at the root (never commit it):

```
GROQ_API_KEY=...
```

The PDFs are in `data/`. To rebuild the index (about 10 minutes, needs the embedding model downloaded once):

```bash
python -c "import pipeline; pipeline.build_vectorstore()"
```

## Run

```bash
ui
un.cmd                         # Streamlit chat (Windows); RAG_UI_DEMO=1 for canned answers without models
uvicorn api.api:app --port 8000    # HTTP API
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Quelle est la peine pour le vol ?\"}"
```

The API response has `answer` and `sources` (id, source, article). Set `HF_HUB_OFFLINE=1` once the models are cached.

Every question and answer the web interface handles is written to a SQLite file, `logs/interactions.db` (`RAG_LOG_DB`
changes the path): time, session id, question, status, answer, claims, cited article ids, whether verification failed,
seconds, and whether the owner's or the visitor's key was used. API keys are never stored. The interface tells visitors
that questions are recorded.

## Deploy on Hugging Face Spaces (free)

1. Create a Space with SDK **Docker** (CPU basic). The YAML block at the top of this file is its configuration.
2. Push this repository to the Space (`git remote add space https://huggingface.co/spaces/<user>/<name>`,
   `git push space <branch>:main`). The index `db_v1_2/` is stored with git LFS (`.gitattributes`), which Spaces require
   for files over 10 MB: run `git lfs install` once.
3. In the Space settings, add these secrets:
   - `GROQ_API_KEY`: your key (shared by all visitors, so also set `UI_OWNER_DAILY_CAP`, for example `30`, the number of
     questions per day on your key; visitors can paste their own key in the sidebar).
   - `HF_LOG_REPO` and `HF_TOKEN`: optional, a private dataset repo (`user/rag-logs`) and a write token. Free Spaces lose
     their disk at each restart, so the log database is copied to that repo under a name unique to each start, at most
     every 5 minutes. `ui/store.py: load_all(folder)` merges the downloaded copies.
4. The Space builds the image (about 15 minutes the first time: it downloads the two models) and then serves
   `https://<user>-<name>.hf.space`. A free Space sleeps after a couple of idle days and wakes on the next visit.

## Evaluation

The exam is `eval/answer_set.json` (100 questions: answer, clarify, refuse). It is split into two stages by
`eval/stages.json`, and `eval/exam_hashes.txt` fingerprints both so a run on a changed exam is refused.

```bash
python eval/preflight.py                                              # checks before a run
python eval/run_answer_eval.py --stage 1 --system v2 --dry-run        # plan, no call
python eval/run_answer_eval.py --stage 1 --system v2                  # run + judge
python eval/run_answer_eval.py --summarize                            # tables from saved runs
```

- Systems: `v2` (agent), `v1.1` (baseline pipeline), `v1` (original, run from its own git worktree at the
  `v1-baseline-runnable` tag).
- **Judge**: a second model grades each answer against reference points written by a lawyer and against the text of the
  cited articles. It reports whether the points are covered, whether the answer contradicts them, and which statements
  the cited articles do not support. It must differ from the generator: set `--judge-provider` (groq or gemini) and
  `--judge-model`, or `JUDGE_MODEL`. `eval/judge_check.py` tests a judge on canned cases before an exam run, and answers
  can be saved first and judged later with `--judge-only`.
- Article text for the judge comes straight from the PDF (`eval/pdf_truth.py`), not from the index. The same text is
  used to check that every number in an answer appears in the cited article.
- `eval/exam_hashes.txt` (and `noisy_set_hashes.txt`) store the SHA-256 of the exam files. A run is refused if the exam
  changed, so all results come from the same questions. After editing the exam, rerun `eval/make_stages.py`.
- `eval/noisy_set.json` holds the same questions with typos and informal wording (`eval/noisy_vs_clean.py` compares).
- Results are in `results/`; each file records the model, reranker, temperature and git commit/tags of the run.

Git tags mark the frozen states used as references: `v1-baseline` and `v1-baseline-runnable` (original pipeline, the
second with two config id fixes so it runs; the official v1 numbers), `v1.1-baseline` (retrieval fixes),
`agent-prompts-frozen` and `agent-retrieval-frozen` (agent prompts and retrieval at the time of evaluation) and
`v2-candidate`. Check out a tag in a separate worktree to reproduce it.

Unit tests (no model, no network):

```bash
python test_v21_retrieval.py
python agent/test_graph_logic.py
python agent/test_normalize_switch.py
python eval/test_run_harness.py
python ui/tests/test_ui.py
```

## Layout

```
config.py         models, paths, retrieval switches
loaders.py        PDF reading
preprocessing.py  header / watermark cleanup
splitter.py       one chunk per article
translate.py      query rewriting and translation (fast pipeline)
retriever.py      semantic + BM25 + direct lookup + fusion
reranker.py       cross-encoder
pipeline.py       index build and the one-pass pipeline
agent/            the LangGraph agent (nodes, prompts, tests)
api/              FastAPI service
ui/               Streamlit interface and the SQLite log
eval/             exam, harness, PDF ground truth, tests
results/          saved runs
data/             legal PDFs
db_v1_2/          current index (db_v1_1/ is the previous one)
Dockerfile        image for Hugging Face Spaces
```

## Chunk ids

Every article has a stable id: `<pdf stem>/s<NN>/<article label>`, for example `PenalFr/s02/243` or
`ProcedurepenaleFr/s01/13-bis`. `s<NN>` numbers the texts inside a PDF in reading order (several PDFs bundle a code with
decrees; a new section starts when article numbering restarts). Sub-chunks of a long article are `<id>#k` and keep the
article id as `parent_id`. A chunk starts only at a real heading (`Article N`, `Art. N`, `Article premier`); in-text
references do not split.

## License

MIT
