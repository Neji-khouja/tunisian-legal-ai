# Wave 2 pre-registration: query normalization (branch agent-v2.1)

Committed BEFORE any normalization call. Nothing below is changed after the first call; if the result is negative it
is reported as it is and the prompt is not tuned on these items.

## Scope

These are the 110 development items (`diagnostics/dev_items.json`: Stage 1 and Stage 2 answerable exam items,
answerable noisy-set items), already used in waves 1 and 1b. A positive result is a candidate only, to be confirmed
on fresh questions. Nothing is integrated into the agent graph in this wave.

## Normalizer

`agent/normalize.py` (not imported by the agent graph; being in agent/ it enters the system fingerprint, which
becomes b06de532f2811e61171258112e3f76d7bc426d7bb2d81e88d55f81f5512637a3 with db_v1_2; no graph file changed).
One prompt version only, `NORMALIZE_PROMPT`:

    sha256 (utf-8) = 1c27f89224093fc0578e2f3dae7d0465fe3bd91bdb000e7d3b23f0d8068e866c

Model `openai/gpt-oss-120b` (Groq, single key GROQ_API_KEY), temperature 0, reasoning effort "low", JSON mode
(`response_format = json_object`), no client-side retries, at most 2 attempts per call (a second attempt after an
API error or an invalid output; a per-minute 429 waits the stated time, at most 120 s). Daily-quota 429: stop, exit 75.

The prompt (verbatim):

```text
You turn a user's question into ONE search query for a database of TUNISIAN legal codes, which are written in French.

The question is already in French. It may be misspelled, informal, use wrong or everyday words, or describe a
situation without naming the legal concept.

Write one precise French search query that uses the vocabulary of the Tunisian legal codes:
- When the user only describes a situation, name the offence, institution or legal concept it falls under.
- Correct the spelling.
- Keep every number, date, duration and amount exactly as the user wrote it.
- NEVER write an article number that the user did not write.
- NEVER add facts that are not in the question.
- At most 40 words.
- Do not answer the question.

Output only a JSON object, with no other text: {"legal_query": "..."}

Examples (invented, illustration only):
Question: "l'archi qui a dessiné nos plans et l'entrepreneur ont fini les travaux, et 3 ans aprés le plafond s'est fissuré, on peut se retourner contre eux ?"
Output: {"legal_query": "garantie de l'architecte et de l'entrepreneur pour les vices de construction de l'édifice apparus 3 ans après la fin des travaux"}
Question: "un temoin a raconté des mensonges a l'audiance pour m'enfoncer, il risque quoi ?"
Output: {"legal_query": "sanction du faux témoignage à l'audience"}
Question: "le gars d'à coté a planté des arbres colés au mur entre nos jardins, il a le droit ?"
Output: {"legal_query": "distance légale des plantations d'arbres par rapport à la limite séparative des fonds"}
```

### Overlap check

`python eval/prompt_overlap.py --prompt agent/normalize.py:NORMALIZE_PROMPT` (new option; writes nothing). The
example questions and outputs are compared with the 510 questions of every eval/*.json file (exam, noisy, gate,
candidates, demo, plumbing, questions.json), the development items and the cached French translations: content-word
Jaccard as in the existing check, and the content words they share (word, number of corpus questions containing it).

| Example | Max Jaccard (closest corpus question) | Pairs >= 0.4 | Shared content words (word, questions) |
|---|---|---|---|
| question 1 (architect / construction defects) | 0.08 (translations ac40) | 0 | ans 7, apre 16, contre 13, eux 1, fini 4, nos 3 |
| question 2 (false testimony) | 0.12 (answer_set_demo demo1) | 0 | risque 25 |
| question 3 (tree planting distance) | 0.09 (questions q016) | 0 | arbre 2, droit 17, entre 4, nos 3 |
| output 1 | 0.07 (translations n_ac07) | 0 | ans 7, apre 16, fin 14 |
| output 2 | 0.17 (questions q008) | 0 | sanction 3 |
| output 3 | 0.10 (questions q021) | 0 | arbre 2, legale 1 |
| overall | 0.17 | 0 | |

("arbre": both corpus questions are the out-of-scope "feuilles des arbres en automne" question.)

## Input queries

Per item, the French query that `eval/retrieval_only.py` uses: the question for French items, the saved translation
(`diagnostics/translations.json`) for Arabic/English items. One call per item (110), saved to
`diagnostics/normalized_queries.json` (id, input query, legal_query, status, prompt hash, tokens, timestamp).
Resume-safe: an item already saved is never called again.

Checks, no LLM: a number in legal_query that is not in the input query is flagged (the item is still used); an empty
or invalid output (not a JSON object with a non-empty string "legal_query", after the attempts) or an API error is a
FAILURE: the item falls back to its input query, so for it V1 = V0 and V2 = V3 = V0. Failures are reported.

## Variants

Index db_v1_2 (opened from a temporary byte copy), current v2.1 defaults (preset "v1.2" of eval/retrieval_only.py:
strict direct lookup + union, cross-reference expansion with cap 300, deepest header, sub-chunk scoring, BM25
whitespace). fq = input query, nq = legal_query (or fq on a failure). In every variant the user's question decides
the direct article lookup numbers and, for Arabic/English items, the original-language dense list, as in V0.

| Variant | Pool | Reranked on |
|---|---|---|
| V0 | `pipeline.step_retrieve(question, fq)` (the current retrieval; baseline) | fq |
| V1 | `pipeline.step_retrieve(question, nq)` (normalized query only) | nq |
| V2 | RRF (k = 60) of the V0 pool and the V1 pool, cut to the larger pool size | fq |
| V3 | the V2 pool | nq |

RRF ties keep the order of first appearance (V0 pool first). The whole pool is reranked (cross-encoder); Hit@k = the
gold article (source stem + article label) at reranked rank <= k. Funnel classes as in eval/retrieval_only.py (V2/V3
"fused" list: RRF of the two untruncated fused lists). Implementation: `diagnostics/wave2_core.py` (fusion, fallback,
rule, committed here) and eval/retrieval_only.py (extended after this commit; its existing modes unchanged).

Extra LLM tokens: V0 = 0; V1, V2, V3 = the normalization call (the same tokens for the three).

## Decision rule

Over the 110 development items, Hit@6:

1. The candidate is the variant among V1, V2, V3 with the highest total Hit@6. Ties: fewer extra tokens, then the
   lower-numbered variant.
2. The candidate is adopted only if:
   - (i) its Hit@6 on the noisy set (set N) is at least +3 items over V0 (net: hits minus V0 hits);
   - (ii) it loses at most 1 item versus V0 on Stage 1 plus Stage 2 combined (lost = hit in V0, miss in the variant);
   - (iii) it loses at most 1 item versus V0 in each language (fr, ar, en), counted the same way.
3. Otherwise none is adopted. The runner-up is not considered.

`wave2_core.decide` implements exactly this.

## Reported

For V0..V3: Hit@6/8/10 with 95% bootstrap CIs (5000 resamples, seed 20261022) per set, language and noise type;
funnel classes; paired differences versus V0 with CIs, gained/lost item ids, exact sign-test p-values; each condition
of the rule with its numbers and the selected variant (or "none"); total tokens; flagged outputs; 12 random rows
(seed 20261022) of input query / legal_query. Development data, not evidence of generalisation.
