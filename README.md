---
title: Tunisian Legal RAG
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
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
3. Pools merged, then reranked with a multilingual cross-encoder; the best articles go to the model: 8 in
   the agent (`AGENT_TOP_K`), 6 in the one-pass pipeline (`RERANK_TOP_K`).
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
| Interfaces | Gradio (`app.py`), Streamlit (`ui/app.py`), FastAPI (`api/`) |

## Setup

```bash
conda create -n rag-app python=3.11
conda activate rag-app
pip install -r requirements.txt gradio==5.23.1 torch   # requirements.lock.txt is the full lock of the dev machine
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
python app.py                      # Gradio web chat on http://127.0.0.1:7860 (RAG_UI_DEMO=1: canned answers, no models)
ui
un.cmd                         # the older Streamlit chat (Windows)
uvicorn api.api:app --port 8000    # HTTP API
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Quelle est la peine pour le vol ?\"}"
```

Set `API_KEY` to require an `X-API-Key` header on `/ask`; without it the endpoint is open, so don't expose it that way.
Questions are limited to 1000 characters. The API response has `answer` and `sources` (id, source, article). Set
`HF_HUB_OFFLINE=1` once the models are cached.

The Gradio page has a sidebar with the visitor's past conversations (kept in their browser), a choice between two modes
(**Agent**: the full graph above, with verification; **Rapide**: the one-pass pipeline on the question as written, no memory
and no verification, far fewer tokens), a token meter (input and output tokens of every question, summed over all model calls), and a "Comment ça
marche" tab with the agent diagram.

Every question and answer the web interface handles is written to a SQLite file, `logs/interactions.db` (`RAG_LOG_DB`
changes the path): time, session id, question, status, answer, claims, cited article ids, whether verification failed,
mode, input and output tokens, seconds, and whether the owner's or the visitor's key was used. API keys are never stored.
The interface tells visitors that questions are recorded.

## Deploy on a Hugging Face Space (Gradio)

The YAML block at the top of this file configures the Space (`sdk: gradio`, entry point `app.py`).

1. Create a Space with SDK **Gradio** (blank template). Free accounts may be limited to ZeroGPU hardware; the app runs on CPU
   and only defines one unused `@spaces.GPU` function, because ZeroGPU Spaces refuse to start without one.
2. Add your SSH public key at https://huggingface.co/settings/keys, then push this repository (the index `db_v1_2/` is in
   git LFS, which Spaces require for files over 10 MB: run `git lfs install` once):

       git remote add space git@hf.co:spaces/<user>/<name>
       git push space <branch>:main

3. In the Space settings, add these secrets:
   - `GROQ_API_KEY`: your key (shared by all visitors; `UI_OWNER_DAILY_CAP` limits the questions per day on it, 30 by
     default, `0` removes the limit; visitors can paste their own key in the page).
   - Keeping the log (a Space loses its disk at each restart), one of:
     - `RAG_LOG_BACKUP_DIR`: a folder of a private Storage Bucket mounted read-write in the Space, for example
       `/data/logs` (no token needed). The database is copied there after every question, as one file per start. Do not
       point `RAG_LOG_DB` at the bucket: the mount takes whole-file writes, not SQLite's in-place ones.
     - `HF_LOG_REPO` and `HF_TOKEN`: a private dataset repo (`user/rag-logs`) and a write token; copied at most every
       5 minutes.

     `ui/store.py: load_all(folder)` merges the downloaded copies.
4. The Space installs `requirements.txt` and starts; the first question waits for the models to load (a minute or two).

The `Dockerfile` (Streamlit interface) is for self-hosting on a VM or a container service.

## Security notes

- Secrets live only in the `.env` file (git-ignored, excluded from the Docker image) or in the Space's secrets. A scan of
  the working tree and all 163 commits found no API key or token.
- The interface escapes the HTML in user and model text before showing it, caps questions at 1000 characters, and never stores
  or logs an API key. The container runs as a non-root user.
- The owner's Groq key is capped per day (`UI_OWNER_DAILY_CAP`), and visitors can use their own key instead.
- The log database holds visitors' questions: keep the log dataset private, and don't enter personal data (the page says so).
- Commits carry the author's email address, and `results/*run_log*` contain a local Windows path. Both become public if the
  repository is public.

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
app.py            Gradio web interface (Space entry point)
ui/               shared chat logic, page style, Streamlit interface, SQLite log
eval/             exam, harness, PDF ground truth, tests
results/          saved runs
data/             legal PDFs
db_v1_2/          current index (db_v1_1/ is the previous one)
Dockerfile        image for self-hosting (Streamlit)
```

## Chunk ids

Every article has a stable id: `<pdf stem>/s<NN>/<article label>`, for example `PenalFr/s02/243` or
`ProcedurepenaleFr/s01/13-bis`. `s<NN>` numbers the texts inside a PDF in reading order (several PDFs bundle a code with
decrees; a new section starts when article numbering restarts). Sub-chunks of a long article are `<id>#k` and keep the
article id as `parent_id`. A chunk starts only at a real heading (`Article N`, `Art. N`, `Article premier`); in-text
references do not split.

## License

MIT
