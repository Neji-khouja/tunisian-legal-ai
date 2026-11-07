# diagnostics/ (branch agent-v2.1)

## v2 baseline

System fingerprint at the start of agent-v2.1 (branched from agent-v2 at `872e07a`), as printed by
`python eval/preflight.py` (sha256 over agent/*.py except test_*/run_*, config.py, splitter.py, retriever.py,
reranker.py, pipeline.py, index path and chunk count):

    v2 baseline: dc8fd664c66230b19fa88aad2195be1ab8b67c5be5540a010ba091c4537ec5ca   (index db_v1_1, 6031 chunks)

It equals the fingerprint of the pre-registration addendum (agent/LEARNING.md).

## v2.1-dev wave 1

System fingerprint after wave 1 (headings forward, index db_v1_2 with context and sub-chunks, strict direct lookup +
union, cross-reference expansion, evidence headers; BM25 tokenizer unchanged), as printed by `python eval/preflight.py`:

    v2.1-dev wave 1: d9ab1df83cb3c23dd0a53fa11241c5381b9dd0d703b3e1809dd4e783f54aa944   (index db_v1_2, 6112 entries / 6031 articles)

All v2.1 switches are in config.py; INDEX_V12 = False and the other switches False give the v2 retrieval on db_v1_1.

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
| `build_v12.py` | Builds db_v1_2 (pipeline.build_index_v12; refuses db_v1_1) and appends the build facts to `index_v12_build.json`. |
| `compare_v12.py` | Offline comparison on the 110 dev items: db_v1_1 with the switches off vs db_v1_2 with them on, BM25 tokenizer variants, evidence window 6/8/10. `--v11` takes a byte copy of db_v1_1. |
| `compare_v12.json`, `COMPARISON_V12.md` | Its output (development data, not evidence of generalisation). |

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
python test_v21_retrieval.py
python diagnostics\build_v12.py
python diagnostics\compare_v12.py --v11 %TEMP%\db_v1_1_copy
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
