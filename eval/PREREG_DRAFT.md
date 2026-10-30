# Pre-registration (DRAFT) — final answer-level evaluation

Status: draft for review. Numbers marked `[Neji: ...]` are to be fixed before any final run.
Once fixed, this file is committed and tagged before the run; nothing below changes after the run starts.

## Systems and data

- **v2**: agent graph (`agent/agent_graph.py`): Analyze, retrieve, Grade/Reformulate, Answer, Verify/Repair.
- **v1.1**: baseline pipeline on the rebuilt index.
- **v1**: original pipeline at tag `v1-baseline-runnable`, run from its own worktree.
- Generator: `openai/gpt-oss-120b`, temperature 0, for all systems.
- Judge: a different model, `[Neji: judge model id]`, temperature 0, with the prompt in `eval/run_answer_eval.py`.
- Questions: `eval/answer_set.json`, frozen at commit `[Neji: hash]`.
  - It holds `[Neji: n]` items: answer / clarify / refuse = `[Neji: n / n / n]`.
  - It is disjoint from the dev set, the gate set and every question used during development.

## Hypotheses

**H1. Correctness.** v2 answers more questions correctly than v1.1 and v1.
- Primary metric: share of `expected = answer` items judged "correct".
- Supported if the paired difference v2 − v1.1 is ≥ `[Neji: ...]` and its 95% CI lower bound is > 0.
- The same test is applied to v2 − v1.

**H2. Hallucination.** v2 hallucinates less than v1.1 and v1.
- Primary metric: answer-level hallucination rate. An answer counts as hallucinated if any of these holds:
  - the judge finds an unsupported statement;
  - a cited id is not in the evidence;
  - a number is missing from the cited PDF articles.
- Supported if the paired difference v2 − v1.1 is ≤ `−[Neji: ...]` and its 95% CI upper bound is < 0.
- Secondary (v2 only, same run): numeric miss on draft claims versus final claims. This is the effect of verification.

**H3. Refusal and clarification.** v2 asks for clarification on ambiguous items and refuses out-of-scope items without over-blocking answerable ones.
- Primary metric: gate accuracy on `clarify` and `refuse` items, with a target of ≥ `[Neji: ...]`.
- Guardrail: on `answer` items, false clarification + false refusal must be ≤ `[Neji: ...]`.
- v1 and v1.1 have no gate, so their gate accuracy is reported for completeness but not tested.

**H4. Latency.** v2's extra steps cost a bounded amount of time.
- Primary metric: median seconds per question with the rate-limit waits subtracted (`seconds − waited_seconds`).
- Target: v2 ≤ `[Neji: ...]` s.
- Also reported, never mixed into latency: 429 responses and client retry waits.

## Analysis

- **Differences between systems:** paired bootstrap over questions, 5000 resamples, seed 20261022, 95% percentile CI.
- **Single-system rates:** 95% Wilson intervals.
- **Breakdowns:** every metric is also reported by language (fr / en / ar), with no test per language.
- **Judge audit:** a human check of `[Neji: n]` items from `--export-audit`. If agreement is below `[Neji: ...]`, H1 and H2 are reported as judge-dependent.

## Rules

1. One run per system. No tuning, no prompt change, no re-run to improve a number after the run starts.
2. Degraded items (a node fell back, `status = "degraded"`) are re-run, not scored. Items still degraded after `[Neji: n]` re-runs are listed and excluded.
3. Errors and rate-limit stops are resumed, not restarted. Completed items are kept.
4. Frozen before the run: prompt hashes, codes.py, index, answer set, judge model, thresholds. `eval/preflight.py` must show no FAIL except items marked as accepted here: `[Neji: ...]`.
5. The run uses `--min-gap [Neji: s]` between questions.
6. Every deviation from this plan is reported with its reason.
