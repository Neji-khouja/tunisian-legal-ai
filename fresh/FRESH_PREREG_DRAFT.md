# Fresh-question test: pre-registration (DRAFT)

**Status: DRAFT.** Neji finalises this file and commits it (renamed `fresh/FRESH_PREREG.md`) **before any fresh
question is run** through any configuration. Until then nothing here is binding. Items marked `[TO FILL]` or
`[TO DECIDE]` must be completed in the final version.

| Placeholder | Value |
|---|---|
| Date of the final pre-registration | `[TO FILL: YYYY-MM-DD]` |
| Commit of the final pre-registration | `[TO FILL]` |
| System fingerprint (`python eval/preflight.py`), db_v1_2, `NORMALIZE_QUERY = "off"` | `[TO FILL]` |
| System fingerprint, db_v1_1 (config A) | `[TO FILL]` |
| sha256 of the locked question file, as committed (`git show <commit>:fresh/questions.json \| sha256sum`) | `[TO FILL]` |
| Number of questions in the locked file, per language and type | `[TO FILL]` |

## 1. Why

Everything decided so far (waves 1, 1b, 2) used the same 110 development items. The wave 2 normalization (V3) was
chosen as the best of three variants on those items, so its development effect is likely optimistic. This test
measures A, B and C once on new questions that no part of the system has seen.

## 2. Questions

- Collected from real people, typed by them, never edited (`fresh/LABELLING_GUIDE.md`, rule 4), anonymised.
- Labelled before any run by a helper with legal knowledge who does not see any system output.
- `python fresh/validate_fresh.py fresh/questions.json` reports 0 errors. Every near-duplicate warning (cosine
  >= 0.68 against the exam, noisy set, development items, their translations and the prompt examples, or within
  the file) is resolved by excluding the question; the excluded ids and the reason are listed in the final
  pre-registration, before the lock. The threshold comes from `fresh/near_dup_calibration.json`.
- Target 150 questions, at least 30 per language (fr, en, ar) when possible; types clear / messy / vague /
  out_of_scope.
- The file is locked by committing it; its sha256 goes in the table above. No question is added, removed or
  re-labelled after the lock. A labelling error found later is reported and analysed only in a sensitivity
  analysis; the primary analysis uses the locked labels.
- `out_of_scope` questions have no gold article and are not in Hit@k; their number is reported.

## 3. Configurations

| Config | Index | Switches | Command |
|---|---|---|---|
| A | db_v1_1 | preset `baseline`: no strict direct lookup, no union, no cross-reference expansion, no header, no sub-chunk scoring, no normalization (the v2 retrieval) | `eval/retrieval_only.py --abc` |
| B | db_v1_2 | preset `v1.2`: strict direct lookup + union, cross-reference expansion with cap 300, deepest header, sub-chunk scoring (wave 1 defaults) | same |
| C | db_v1_2 | B + `NORMALIZE_QUERY = "v3"`: pools of the French query and of the normalized query fused by RRF, reranked on the normalized query (diagnostics/WAVE2_PREREG.md, prompt sha256 1c27f892…) | same |

Retrieval-only measurement, the same for the three configs: the French query (the question for French questions,
one cached faithful translation for Arabic/English ones), no Analyze code hint, the whole pool reranked, gold =
source + article label of the gold chunk ids. Indexes opened from temporary byte copies only.

LLM calls: one translation per Arabic/English question and one normalization per question, made once with
`--allow-llm` and cached in `fresh/cache/` (key GROQ_API_KEY, daily quota: stop with exit 75 and resume). The
caches are committed with the results. A failed translation excludes the question from all three configs
(reported). A failed, empty or invalid normalization is not excluded: C falls back to B for it, and it counts in
the failure condition of the C rule.

Command (after the lock):

```bat
python eval\retrieval_only.py --abc --items fresh\questions.json --allow-llm --cache-dir fresh\cache --out fresh\abc_results.json --md fresh\ABC_RESULTS.md
```

## 4. Outcomes

Primary: paired Hit@6 for B - A, C - B and C - A: mean difference with a 95% percentile bootstrap CI (5000
resamples, seed 20261022), exact two-sided sign test on the discordant questions, gained and lost question ids,
per language and per type (net Hit@6 per group).

Secondary: Hit@8 and Hit@10 (same statistics), funnel classes, gold in evidence, normalization failures and
flagged outputs (numbers not in the input).

## 5. Decision rules

Net loss of a language = (hits of the new config) - (hits of the old config) over the questions of that language;
"a net loss of 3 items or more" means this number is <= -3.

1. **B** is adopted if the paired Hit@6 difference B - A is >= 0 **and** no language has a net loss of 3 items or
   more (B versus A).
2. **C** is adopted only if the lower bound of the 95% bootstrap CI of C - B for Hit@6 is above 0, **and** no
   language has a net loss of 3 items or more (C versus B), **and** at most 2% of the normalizations fail or are
   empty (over the questions analysed in C).
3. Otherwise v2.1 ships as B.
4. If B fails rule 1, v2.1 ships as A (the current v2 retrieval, unchanged). C is considered only if B passes
   rule 1. B's adoption rests on a design argument (mechanical fixes: stable headings, strict direct lookup,
   cross-references, sub-chunks), not on a proven gain: under no real effect B passes rule 1 about 40-48% of the
   time (section 6).

`eval/retrieval_only.py --abc` computes every condition (table "Draft decision rules"); the final file states
the decision in words.

## 6. Power (ROUGH ESTIMATE)

`fresh/power_estimate.py` (`fresh/power_estimate.json`): a simulation under assumed effects, NOT a property of the
test. Paired outcomes with the development discordance rate (C vs B: 22/110 = 0.20, from V3 vs V0; B vs A:
19/110), questions split evenly over the three languages, 2000 simulated studies (1000 bootstrap resamples each).
The development effects come from the items the variants were chosen on and are probably optimistic.

Probability that rule 2 (C) passes, without the failure condition:

| Assumed C - B effect | n = 100 | n = 150 | n = 200 |
|---|---|---|---|
| development effect (+16 / -6 per 110, net +9.1 pp) | 0.51 | 0.72 | 0.83 |
| half of it (net +4.5 pp) | 0.15 | 0.23 | 0.28 |
| no effect | 0.02 | 0.02 | 0.02 |

Smallest net C - B difference with 80% power for "CI lower bound > 0" (normal approximation, discordance 0.20):
12.1 pp (n = 100), 10.0 pp (n = 150), 8.7 pp (n = 200).

Probability that rule 1 (B) passes:

| Assumed B - A effect | n = 100 | n = 150 | n = 200 |
|---|---|---|---|
| development effect (+12 / -7 per 110, net +4.5 pp) | 0.82 | 0.82 | 0.86 |
| no effect | 0.48 | 0.41 | 0.40 |
| a loss (net -4.5 pp) | 0.13 | 0.07 | 0.03 |

## 7. Secondary: end-to-end subset

At least 40 questions, drawn from the locked file before any run (stratified by language and type,
`random.Random(20261022)`; the ids listed in the final pre-registration), answered end to end by v2 (agent-v2,
fingerprint dc8fd664…, db_v1_1) and by v2.1 as shipped by section 5. Reported per system: tokens (usage
callback, normalization included), LLM calls, latency (wall seconds, retry waits apart), and a descriptive
correctness check by the same judge as before (gemini-3.1-flash-lite). Descriptive only: **no claim of
an accuracy gain** is made from this subset, and no decision depends on it.

## 8. Not done

No prompt, switch or threshold is tuned on the fresh questions. No second run with a changed configuration
replaces the first; any later run is reported as exploratory. Results are reported whatever they are.
