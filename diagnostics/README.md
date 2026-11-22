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

## v2.1-dev wave 1b

Wave 1 plus the "Contexte :" header mode (config.EVIDENCE_CONTEXT_HEADER = "deepest") and the cross-reference token cap
(config.XREF_TOKEN_CAP = 300); db_v1_2 unchanged:

    v2.1-dev wave 1b: cea3ca90b893f58e4b1c575ed6aab87085846145310caf0c7c9f2f3540bbbd0d   (index db_v1_2, 6112 entries / 6031 articles)

## v2.1-dev wave 2 integrated (V3 off)

Wave 1b plus the wave 2 query normalization V3 in the agent behind `config.NORMALIZE_QUERY` (default `"off"`;
`"v3"` normalizes the initial query of a turn in agent/retrieve_g.py, RRF of the two pools, reranked on the
normalized query; the reformulate step's queries are never normalized). With the switch off the agent's states are
identical to commit cc46906 (agent/test_normalize_switch.py); db_v1_2 unchanged:

    v2.1-dev wave 2 integrated (V3 off): 54b31ddd94262bbd4aa65a3b5a95e3fe47f9176ce9434f7dbc66bc1ca09b362d   (index db_v1_2, 6112 entries / 6031 articles)

Fresh-question protocol: `fresh/` (questions_template.json, LABELLING_GUIDE.md, validate_fresh.py,
near_dup_calibration.json, FRESH_PREREG_DRAFT.md, power_estimate.py) and `eval/retrieval_only.py --abc`
(configs A / B / C, LLM calls only with --allow-llm, caches in fresh/cache/).

## v2.1-dev V3 default on (owner decision)

`config.NORMALIZE_QUERY` default changed to `"v3"` by the owner's decision. This is NOT a validated result: the
pre-registered wave 2 rule selected none (V3 failed conditions ii and iii, diagnostics/WAVE2_VARIANTS.md), the
dev gain (Hit@6 85 vs 75 on 110 items) is on the items V3 was chosen on, and the fresh-question test
(fresh/FRESH_PREREG_DRAFT.md) has not been run. Each turn makes one extra Groq call (about 600 tokens).
`NORMALIZE_QUERY = "off"` restores the previous behaviour.

    v2.1-dev V3 default on: ab0fad94d66d0bc4eb03c9b6e87e368caa346dcc98befa80759c82a965746eb1   (index db_v1_2, 6112 entries / 6031 articles)

## v2.1-dev after cleanup (V3 default on)

Same behaviour as the previous entry; only the file set of the fingerprint changed (23 -> 18 files: dev-phase scripts
removed from agent/: eval_loop, eval_oracle, eval_plan_vs_raw, code_accuracy, loop_core):

    v2.1-dev after cleanup: 3053e132c20710a6bb265ef520061ecacd6997c35b0b1703cbee1405a0e17e87   (index db_v1_2, 6112 entries / 6031 articles)

## v2.1-dev after comment trim

Comments and module docstrings shortened in 54 files (no code change: identical syntax trees; prompt hashes
unchanged). The fingerprint hashes file bytes, so it changed:

    v2.1-dev after comment trim: 12c8095daec4b660e7777f295c47869a8e1278f88a8954785d8de83ffd83c88b   (index db_v1_2, 6112 entries / 6031 articles)

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
| `evidence_budget.py`, `evidence_budget.json`, `EVIDENCE_BUDGET.md` | "Contexte :" header tokens for full / deepest / none, cross-reference token caps 300 / 600 / unlimited, and the cap chosen by the rule fixed before the run. |
| `token_estimate.py`, `token_estimate.json`, `TOKEN_ESTIMATE.md` | ESTIMATE of the extra v2 input tokens of the header and capped expansion (evidence blocks rebuilt with the agent's formatting functions, rate fitted on saved v2 rows). |
| `dev_items.json` | The 110 development items (`eval/retrieval_only.py --export-dev`). |
| `retrieval_only_dev.json`, `RETRIEVAL_ONLY_DEV.md` | `eval/retrieval_only.py` on them: db_v1_1 baseline vs db_v1_2 v1.2, paired. |
| `WAVE2_PREREG.md` | Wave 2 pre-registration (committed before any normalization call): prompt, hash, overlap check, variants V0-V3, decision rule, scope. |
| `wave2_core.py` | Pure helpers of wave 2: RRF of two pools, fallback of a failed normalization, the decision rule. |
| `normalize_queries.py` | One `agent/normalize.py` call per development item (Groq, resumable, exit 75 on a daily quota); `--check` reruns the no-LLM checks. |
| `normalized_queries.json` | Its output: id, input query, legal_query, status, prompt hash, tokens, timestamp, flags. |
| `wave2_variants.json`, `WAVE2_VARIANTS.md` | `eval/retrieval_only.py --variants V0 V1 V2 V3` on db_v1_2 (development data, not evidence of generalisation). |

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
xcopy /e /i /q db_v1_2 %TEMP%\db_v1_2_copy
python diagnostics\evidence_budget.py --v12 %TEMP%\db_v1_2_copy
python diagnostics\token_estimate.py --v11 %TEMP%\db_v1_1_copy --v12 %TEMP%\db_v1_2_copy
python eval\test_retrieval_only.py
python eval\retrieval_only.py --export-dev diagnostics\dev_items.json
python eval\retrieval_only.py --items diagnostics\dev_items.json --index-a db_v1_1 --config-a baseline --index-b db_v1_2 --config-b v1.2 --out diagnostics\retrieval_only_dev.json --md diagnostics\RETRIEVAL_ONLY_DEV.md
rem wave 2
python eval\prompt_overlap.py --prompt agent/normalize.py:NORMALIZE_PROMPT
python diagnostics\normalize_queries.py
python eval\test_wave2.py
python eval\retrieval_only.py --items diagnostics\dev_items.json --index-a db_v1_2 --config-a v1.2 --variants V0 V1 V2 V3 --normalized diagnostics\normalized_queries.json --out diagnostics\wave2_variants.json --md diagnostics\WAVE2_VARIANTS.md
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
