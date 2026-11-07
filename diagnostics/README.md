# diagnostics/ (branch agent-v2.1)

## v2 baseline

System fingerprint at the start of agent-v2.1 (branched from agent-v2 at `872e07a`), as printed by
`python eval/preflight.py` (sha256 over agent/*.py except test_*/run_*, config.py, splitter.py, retriever.py,
reranker.py, pipeline.py, index path and chunk count):

    v2 baseline: dc8fd664c66230b19fa88aad2195be1ab8b67c5be5540a010ba091c4537ec5ca   (index db_v1_1, 6031 chunks)

It equals the fingerprint of the pre-registration addendum (agent/LEARNING.md).

## Files

| File | What it is |
|---|---|
| `retrieval_diagnostic.py` | Retrieval funnel of the v1.1 path (existing functions imported, not modified), cross-references in the index and in the articles cited by saved v2 answers, chunk structure. No LLM call. `--md-only` rewrites the .md from the .json without opening the index. |
| `translate_skipped.py` | One call of `translate.translate_to_french` per Arabic/English answerable item (47; the only Groq use, key GROQ_API_KEY, exit 75 on a daily quota). Resumable. |
| `translations.json` | Its output: set, item id, language, query, French query. Used by `retrieval_diagnostic.py`. |
| `funnel_core.py` | Pure helpers (ranks, funnel classes, summaries, Hit@k, reference patterns). |
| `test_funnel.py` | Funnel arithmetic and reference patterns on a stub index. |
| `chroma_open_check.py` | Which sqlite tables one Chroma open changes, measured on a temporary copy of db_v1_1. |
| `retrieval_funnel.json` | Per-item records, skipped items, summaries, cross-reference and chunk statistics. |
| `chroma_open_check.json` | Output of chroma_open_check.py. |
| `RETRIEVAL_DIAGNOSTIC.md` | Tables. |

## Commands (Windows cmd, from the repository root)

```bat
conda activate rag-app
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python diagnostics\test_funnel.py
python diagnostics\translate_skipped.py
rem open a byte copy of db_v1_1 so the real index is never written (one Chroma open adds an acquire_write row)
xcopy /e /i /q db_v1_1 %TEMP%\db_v1_1_copy
python diagnostics\retrieval_diagnostic.py --db %TEMP%\db_v1_1_copy
python diagnostics\chroma_open_check.py
python diagnostics\retrieval_diagnostic.py --md-only
```

## Notes on the run

- Only the result files and eval/ exam files are read; result rows are used for ids and metrics only (gold_hit,
  evidence_ids, cited), never for answer text.
- Opening the index through Chroma (as every run of the harness and the UI does) appends one row to its internal
  `acquire_write` table; the other tables are unchanged (`chroma_open_check.json`). The first retrieval run opened
  db_v1_1 once; the second (all 110 items) opened a byte copy, and db_v1_1's files hashed the same before and after.
- Arabic/English items: the translation is not deterministic, so the reproduced Hit@6 can differ from the saved v1.1
  row (2 of 47 translated items; listed in RETRIEVAL_DIAGNOSTIC.md).
- `chroma_open_check.py` leaves its temporary copy in %TEMP% when Chroma still holds the files open (Windows).
