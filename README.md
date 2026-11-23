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
"fast" mode in the UI and as the baseline in evaluation.

| Part | Choice |
|---|---|
| Embeddings | `OrdalieTech/Solon-embeddings-large-0.1` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Vector store | ChromaDB (local files) + BM25 |
| LLM | `openai/gpt-oss-120b` through Groq (`LLM_MODEL` in `config.py`) |
| PDF parsing | PyMuPDF |
| Interfaces | Streamlit (`ui/`, `app.py`), FastAPI (`api/`) |

## Setup

```bash
conda create -n rag-app python=3.11
conda activate rag-app
pip install -r requirements.lock.txt
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
ui\run.cmd                         # Streamlit chat (Windows), verified agent or fast mode
uvicorn api.api:app --port 8000    # HTTP API
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Quelle est la peine pour le vol ?\"}"
```

The response has `answer` and `sources` (id, source, article). Set `HF_HUB_OFFLINE=1` once the models are cached.

## Evaluation

The exam is `eval/answer_set.json` (100 questions: answer, clarify, refuse). It is split into two stages by
`eval/stages.json`, and `eval/exam_hashes.txt` fingerprints both so a run on a changed exam is refused.

```bash
python eval/preflight.py                                              # checks before a run
python eval/run_answer_eval.py --stage 1 --system v2 --dry-run        # plan, no call
python eval/run_answer_eval.py --stage 1 --system v2                  # run + judge
python eval/run_answer_eval.py --summarize                            # tables from saved runs
```

- Systems: `v2` (agent), `v1.1` (baseline pipeline), `v1` (original, needs its own worktree).
- Answers are judged by a different model than the generator (`--judge-model` or `JUDGE_MODEL`), against the text of the
  cited article taken straight from the PDF (`eval/pdf_truth.py`), not from the index.
- `eval/noisy_set.json` holds the same questions with typos and informal wording (`eval/noisy_vs_clean.py` compares).
- Results are in `results/`; each file records the model, reranker and temperature used.

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
ui/, app.py       Streamlit interfaces
eval/             exam, harness, PDF ground truth, tests
results/          saved runs
data/             legal PDFs
db_v1_2/          current index (db_v1_1/ is the previous one)
```

## Chunk ids

Every article has a stable id: `<pdf stem>/s<NN>/<article label>`, for example `PenalFr/s02/243` or
`ProcedurepenaleFr/s01/13-bis`. `s<NN>` numbers the texts inside a PDF in reading order (several PDFs bundle a code with
decrees; a new section starts when article numbering restarts). Sub-chunks of a long article are `<id>#k` and keep the
article id as `parent_id`. A chunk starts only at a real heading (`Article N`, `Art. N`, `Article premier`); in-text
references do not split.

## License

MIT
