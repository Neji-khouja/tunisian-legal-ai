# Learning log: Agentic Legal Researcher

My own notes. Plain words, so I can explain every step in an interview.
Last updated: 23 Oct 2026.

---

## 0. Where I am right now

| Part | Status |
|---|---|
| Baseline RAG v1.1 (retrieval fixes) | Done, tagged `v1.1-baseline`, pushed |
| Agent branch `agent-v2` | Created from the v1.1 tag, pushed |
| Analyze node (understands the question) | Works on my dev set, held-out set still to do |
| Agent graph (Analyze, Clarify, Refuse, stub) | Works, with the follow-up question loop |
| Plan node (writes French search queries) | 20/20 format checks, retrieval test still to do |
| Retrieve, Grade, Answer, Verify | Not started |
| 100-question answer-level evaluation | Not started |

---

## 1. The project in one paragraph

I have a Tunisian legal RAG: 14 legal codes, questions in Arabic, French or English.
I am turning it into an **agentic legal researcher**. The agent should:
split a question into smaller searches, search across the codes, check that every article
it cites really exists and really supports the claim, ask a follow-up when the question is
unclear, and refuse questions that are not about Tunisian law.

The story I will show has three columns: **v1 (original) → v1.1 (better retrieval) → v2 (agent)**,
with accuracy and hallucination numbers for each.

---

## 2. The plan in four phases

1. **Design** the agent on paper (diagrams, data).
2. **Fix the baseline** first, so I measure the agent against the best version of the old system.
3. **Learn LangGraph** with a tiny toy, then build the agent nodes one by one.
4. **Evaluate** on about 100 questions: accuracy, fake citations, unsupported claims.

I write the agent code myself to learn. Claude explains, gives syntax, and reviews.

---

## 3. Phase 1: Design (done)

**What I did:** drew three diagrams in draw.io and wrote one data file.
- L1: who talks to the system (user, app, models, database).
- L2: the flow of the agent from question to answer.
- L3: the data that moves between the boxes (`docs/L3_data.md`).

**The agent flow:**
Analyze → (Clarify | Refuse | Plan) → Retrieve → Grade (retry max 2) → Answer →
Existence check → Support check (repair max 1) → Final answer.

**Why design first:** I did not understand the old code, and starting to code the agent
without a map would have made me lost. The diagrams are the map.

**Limits I decided:** at most 2 retrieval retries, at most 1 repair.
Every loop needs a limit, or the agent can run forever.

**Still to update:** the article ID format in `L3_data.md` (see section 4, chunk IDs).

---

## 4. Phase 2: Fix the baseline

### Why fix it before building the agent

If I build the agent on a weak baseline, I cannot tell if the agent helped or if I just
fixed bugs. So I improved the baseline as far as I reasonably could, then froze it.

### What I found wrong in v1 (the diagnosis)

- The v1 tag did not even run: two model names no longer worked. The reranker also failed
  silently, so the system ran without reranking and nobody noticed.
- The old query rewrite **hurt**: using the raw question got 57.5% Hit@6, rewrite plus
  sub-queries only 46.0%.
- Of 243 generated queries, 63 had an invented article number, 155 had filler words,
  17 contained "jurisprudence".
- 16% of chunk boundaries were wrong: a mention like "see article 5" was treated as the
  start of a new article, cutting real articles into fragments.
- Several PDFs contain more than one law, so "article 12" alone is ambiguous.
- A synonym table in the code used the same topics as my evaluation questions
  (theft, divorce, dismissal...). That is **leakage**: the system was partly given the answers.
- My gold labels were written from memory and had errors (for example an abrogated article).
- The reranker took about 13 s per query on CPU.

### How I measure (so the numbers mean something)

- A question counts as found only if **both the code (source) and the article number** match.
  Matching the number alone is wrong, because article 12 exists in many codes.
- Temperature 0 everywhere, so a rerun gives the same answer.
- The questions are split into **dev** (35, for tuning) and **test** (52, used once at the end).
  Seed 20261022.
- Labels were reviewed by me, not only by the AI.

**Metrics in plain words:**
- **Hit@6:** the right article is somewhere in the top 6 results.
- **Hit@1:** the right article is the very first result.
- **Recall:** how many of the right articles were found.
- **MRR:** how high the first right article ranks (1.0 is perfect).

### The seven improvement steps (dev set, 35 questions)

| Step | What I changed | Hit@6 | Hit@1 | Recall | MRR | Kept? |
|---|---|---|---|---|---|---|
| Start | v1 with corrected labels | 57.1 | 34.3 | 25.4 | 0.403 | |
| 1 | Chunks start only at real article headings, stable IDs | 71.4 | 25.7 | 32.0 | 0.403 | Yes |
| 2 | Search with the raw question first (translate only Arabic/English) | 88.6 | 54.3 | 53.5 | 0.657 | Yes |
| 3 | Remove the leaked synonym table | 85.7 | 45.7 | 51.3 | 0.593 | Yes |
| 4 | Reranker fix, pool of 20, int8, shorter max length | 85.7 | 57.1 | 52.7 | 0.651 | Yes |
| 5 | Reciprocal rank fusion before the reranker | 88.6 | 57.1 | 54.1 | 0.659 | Yes |
| 6 | Add context text to each chunk before embedding | 82.9 | 62.9 | 52.3 | 0.689 | **No** |
| 7 | Try bge-m3 instead of Solon embeddings | 82.9 | 54.3 | 51.1 | 0.643 | **No** |

**What each step means:**
1. **Chunk fix.** Articles now start only at real headings. Wrong fragments in the top 6
   went from 18 to 0. Chunk IDs look like `PenalFr/s02/226-ter` (source, section, article),
   so an ID never collides. See `docs/CHUNK_IDS.md`.
2. **Raw first.** The biggest jump (+17 points of Hit@6). The old LLM rewrite was adding
   noise; the plain question is usually the best query.
3. **Synonym table removed.** The score dropped on purpose. The table was cheating,
   so 85.7% is the honest number. A drop caused by removing leakage is progress.
4. **Reranker.** Same quality, about 4.5 times faster.
5. **Fusion.** Combines the semantic and keyword rankings in a fairer way than before.
6 and 7. **Rejected.** They helped one metric and hurt another. I tried them, measured,
   and kept the simpler setup. Showing what did not work makes the result more credible.

### Final result: run once on the test set (52 questions)

| | Hit@6 | Hit@1 | Recall | MRR | Time per question |
|---|---|---|---|---|---|
| v1 | 57.7 | 32.7 | 34.4 | 0.405 | about 18 s plus 3 LLM calls |
| **v1.1** | **80.8** | **59.6** | **50.4** | **0.687** | **about 3 s** |

Retrieval is far better, much faster, and it needs no LLM call before searching.
Full write-up: `docs/baseline_v1.1_report.md`. Tag: `v1.1-baseline`.

**Caveat to remember:** these are retrieval numbers (is the right article in the results).
They do not yet say whether the final answer is correct or whether citations are real.
That is what the agent evaluation will measure.

---

## 5. Phase 3a: Learning LangGraph with a toy

**Why a toy:** I wanted to understand graphs, routing and loops on something tiny, before
mixing them with a legal system.

**What I built (`toy_graph/`):**
- `toy.py`: classify a text as question, statement or empty, then route to the right reply.
- `toy_loop.py`: a node that runs twice, then the router sends it to the end.

**What I learned (and the errors that taught me):**
- A node returns **only the fields it changes**. Other fields keep their value.
- `invoke` returns the final state.
- `add_conditional_edges` needs a real **node name** as its source. A router function
  is not a node. I got this wrong first.
- A router must return the `END` constant, not the string `"END"`.
- A node with no edge going in is unreachable.
- Every path must end at END.
- A loop limit belongs in my own state (a counter). Do not rely on LangGraph's safety limit.
- "Ends with a question mark" is a bad rule for detecting a question
  ("Tell me the penalty for theft" has none). That is why Analyze uses an LLM.

---

## 6. Phase 3b: Analyze node

**Job:** read the question and return: language, intent, which codes apply, and a follow-up
question if needed.

**Intent has three values:**
- **clear:** one area of law applies. The answer can list the rules and their variants.
  "Simple theft is punished by X, aggravated theft by Y" is a clear question.
- **ambiguous:** several different areas of law could apply, so a search would be a guess
  ("Can I break my contract?" could be employment, rent or sale).
  Or a missing fact decides **which rule** applies.
- **out_of_scope:** not about Tunisian law, or not covered by the 14 codes.

**How it works:** Groq `openai/gpt-oss-120b` with a Pydantic schema and
`with_structured_output(Analysis, method="json_schema")`. Temperature 0.

**Problems and fixes:**
| Problem | Fix |
|---|---|
| 400 error "Tool choice is required" | Use `method="json_schema"` |
| The model asked "which country?" | Tell it that all questions are about Tunisia |
| The prompt examples were my test questions | Replaced them (**prompt leakage**). Use a separate held-out set |
| "Tell me the penalty for theft" came back ambiguous | The model reasoned "it depends on the type of theft". Almost every legal answer depends on facts, so I narrowed the definition of ambiguous to the **area of law**, not the details |
| A garbled line in my prompt (I pasted a note instead of a rule) | Rewrite it as a clear rule |
| Output showed `target_code` | Field name must be `target_codes` everywhere |

**Scores:**
| Date | Change | Dev score |
|---|---|---|
| Oct 2026 | First prompt with Tunisian context | 9/10 |
| | Narrower "ambiguous" + "if unsure choose clear" | (fill in) |
| | Dev set with 3-language groups and consistency check | (fill in) |
| | Held-out set, run once | (fill in) |

**Is English better than other languages?** It might be. French questions match the French
codes directly, English and Arabic go through a translation step, the Solon embedding model
is French-oriented, and LLMs are usually weakest in Arabic. I must **measure** it: the same
30 questions in FR, EN and AR, and report Hit@6 and answer accuracy per language.

---

## 7. Phase 3c: The agent graph and the follow-up loop

**Graph:** START → analyze → (clarify | refuse | plan_stub) → END.
The three branches are chosen by `route_after_analyze`. The plan node is still a stub.

**Safety:** if Analyze crashes, the question continues as "clear" (fallback from the L3 file).

**Follow-up question loop:**
- A **checkpointer** (`MemorySaver`) remembers the state between calls that share a `thread_id`.
  Without it, every call starts from zero and "it is my job contract" means nothing.
- The Clarify node saves a running **transcript** of the conversation and a counter.
- The user's reply goes back through Analyze, which sees the whole transcript.
- The agent asks at most **2** follow-up questions. After that it does its best with what
  it knows. This is a design choice: asking forever is annoying.
- The language of the first question is passed on, so the follow-up stays in the same language.
- After an answer or a refusal, the counter and transcript are reset.

**Bugs I hit:**
- I had **two functions named `clarify`**. Python silently keeps the last one, so the good version
  never ran: no memory, no counter, and a confusing answer in the wrong language.
  Lesson: Python does not warn about redefined functions. Search for `def name` when behaviour is strange.
- My test inputs were poor ("what do not understand?"). A test is only as good as its inputs.

**Test scenarios to keep:**
1. Vague question, then a real answer → ends in `answer` with the right code.
2. Vague question, then "I don't know" twice → the limit stops the follow-ups.
3. Vague question, then something off-topic → `refuse`.

---

## 8. Phase 3d: Plan node

**Job:** turn one question into 1 to 4 short **French** search queries, each tagged with a code.

**Rules in the prompt:** French only; one query for a simple question, up to four for a
mixed one; never invent an article number (the baseline had 63 of 243 with one);
no filler words; code must be one of the codes found by Analyze.

**Tested with plain code, no LLM judge** (`plan_tests.json`, `run_plan_tests.py`):
number of queries in range, no digits (unless the user named an article), no Arabic script,
no banned filler words, code allowed, max 12 words, **no English words**.

**Results:**
| Date | Change | Score |
|---|---|---|
| Oct 2026 | First prompt, 8 tests | 8/8 |
| | 12 harder tests added (typos, Tunisian dialect, one-word question, long story, empty codes, named article in Arabic, three-code question) | 20/20 |
| | Added the English-word check | 19/20: it caught "award" in an arbitration query |
| | Added "French legal vocabulary only" to the prompt | (fill in) |

**Lessons:**
- The checks said 20/20 but one query contained an English word. Passing means "well-formed",
  not "good". I also read the queries by hand.
- A new check should fail first, then pass after the fix. That proves the check works.
- Some queries are very short ("vol"). Whether they retrieve well is an open question.

**The experiment that decides if Plan is useful:** on the 35 dev questions, compare
(A) the raw question, (B) the Plan queries only, (C) raw question plus Plan queries.
Report Hit@6 for each. The old rewrite hurt, so Plan must prove it does not.

---

## 9. Git state

- `v1-baseline`: unrunnable original. `v1-baseline-runnable`: only two model IDs fixed.
- `baseline-v1.1` and tag `v1.1-baseline`: the fixed baseline (final commit `bef876f`).
- `agent-v2`: branch for the agent, created from `v1.1-baseline`. Agent files live in `agent/`.
- `toy_graph/` is git-ignored, it is my scratch space.
- `.env` is git-ignored. Never commit API keys.

---

## 10. Check-yourself questions (rewrite in my own words)

1. What does a node return, and what happens to the fields it does not return?
   It returns a dict with only the fields to change; the others keep their value.
2. What does `invoke` return? The final state.
3. What happens if a router returns a name that is not a node? LangGraph raises an error.
   (I tried it on purpose and wrote down the message.)
4. What does `thread_id` do? It identifies one conversation, so the checkpointer can
   restore its state on the next call.
5. Why does `MemorySaver` lose everything when the program stops? It keeps state in RAM only.
   In production I would use a database-backed saver.
6. Why is "the answer depends on the facts" not enough to call a question ambiguous?
   Because almost every legal answer does. The question is whether one area of law applies.
7. Why did removing the synonym table lower my score, and why is that good news?
   The table copied the topics of my test questions, so part of the score was cheating.

---

## 11. What is next

1. Finish the Plan fix: add the French-only rule, rerun, record the score.
2. Finish Analyze: language-group consistency check, then the held-out set once.
3. Build the Retrieve node on top of the v1.1 retrieval (`hybrid_retrieve` in `retriever.py`).
4. Run the experiment: raw vs Plan vs raw plus Plan on the 35 dev questions.
5. Grade node: drop irrelevant articles, retry max 2.
6. Answer node: structured claims, each with the article IDs that support it.
7. Verifier: existence check in plain code, support check with an LLM, repair max 1.
8. Trace wrapper (one JSONL line per node) and a Streamlit panel showing the reasoning.
9. Build the ~100-question answer-level test set: normal, ambiguous, unanswerable, trap
   questions (such as "What does Article 9999 say?"), in three languages.
10. Measure v1, v1.1 and v2: accuracy, fake-citation rate, unsupported-claim rate, refusal quality,
    and compare languages.

**Open questions:**
- Groq free tier is 200k tokens per day on this model. The evaluation will need more:
  a paid tier, a cheaper model for simple nodes, or Ollama for Analyze and Grade.
  Put every LLM call behind one function with the model name in config.
- What to do when evidence is weak and retries are used up. Proposed: refuse.
- Update the article ID format in `L3_data.md` to the new chunk IDs.
## Experiment: code hint (G) and Grade -> Reformulate loop (dev only, 2026-10-27)

Expectation (written before running): loop beats A2 by >= 3 points Hit@6, no concept worse.
Result: met on dev.
  A2 71.4 | G 80.0 | loop from A2 80.0 | loop from G 85.7 | A1 oracle 88.6  (Hit@6)
  Loop made 0 questions worse (10 retried from G, 4 improved). 1.74 LLM calls/question.
Caveats:
  - Gains come from 2 concepts (frapper; validite du contrat in AR/EN). Evidence is thin.
  - Plan variants also beat A2 on dev and then lost on test. Dev is not proof.
  - Code descriptions were fixed after looking at dev failures, so hint accuracy (29/30) is optimistic.
  - The AR/EN contract-validity wins may be a translation weakness, not agentic reasoning (untested).
  - Analyze cache used by G had no prompt hash (UNVERIFIED, prompt is frozen).
Status: UNCONFIRMED. Test split already seen twice. Confirmation = the fresh 100-question set.


## Corpus audit against the original PDFs (2026-10-29)
- Independent extraction (pypdf) vs the index (PyMuPDF). 14 sources verified. Fidelity: 141/148 sampled chunks match,
  7 mismatches (3 footnote digits, 3 CommerceFr chunks contain the page footer "Imprimerie Officielle de la
  République Tuinsienne" (typo in source, so clean_text missed it), 1 footnote text).
- 9 articles have no chunk of their own because the splitter misses headings broken across lines
  ("Article \n335.-"): COC 1273, 882; DIPFR 10; dtreelFr 335; societeFr 193, 194, 212, 221; travailFr 48. Their text sits
  inside the previous article's chunk (0.17% of articles). They cannot be used as gold and may be cited under the
  neighbouring article number.
- The index stores the base number in `article`: 112 cases in 10 sources share a base number with a -bis/-ter chunk.
  Labels are now derived from the chunk id.
- Mojibake ("quelquâ€™un") exists in no index chunk and no PDF extraction (checked 3 more indexes). Likely emitted by the
  model once. check_answer now flags it.
- DECISION: splitter heading repair is PINNED (new index = new baseline = rerun of dev numbers; v1.1 and v2 must
  stay on the same index for a fair comparison). Backlog item "v1.2".

## Answer node and verifier
- Answer = claims with cited chunk ids (check_answer: ids exist, no empty claim, no mojibake). ANSWER_CHARS = VERIFY_CHARS = 3000.
- Verifier calibration (ground truth by construction, French verbatim sentences, n=24 per run, seed 20261022):
  run 1: false alarms 0/24, number_changed 23/24, wrong_article 24/24, added_condition 24/24.
  run 2 (after label change and 3000 limit; sample changed, 4 articles in common): 0/24, 24/24, 24/24, 24/24.
  NOTE: no thresholds were written before run 1. Any threshold written afterwards is post hoc. No verifier threshold is
  stated (not recorded; no claim made). The evaluation thresholds are in the PRE-REGISTRATION ADDENDUM at the end of
  this file (written 2026-11-01, after verifier runs 1 and 2).
- Limits: blatant mutations only; French claims only (Arabic/English claims against French articles are untested);
  same model family writes and checks claims.
- PDF numeric check: 60 claims, 0 numbers missing after the implicit-reference rule ("l'article précédent").
- hit_fr answer verified by hand against the PDF: art. 218 supports the 4 claims on penalties and the attempt.
- Overlapping provisions: art. 113 (one month of effective work) vs art. 123 (six months of continuous service) for paid
  leave. Both are valid sources; the answer set must accept either. Which one prevails: not recorded; no claim made.
  (Recorded: eval/questions.json lists art. 113 as gold in 3 items; art. 123 is in no gold list of eval/questions.json or
  eval/answer_set.json.)
- Clarification path works (Arabic ambiguous question -> follow-up -> answer cites 13-bis). Retrieval after a
  clarification receives the labelled transcript (User:/Agent asked:); n=1, no visible harm.
- Run-to-run noise: French penal questions identical over 3 runs; the Arabic clarification wording and the 5th/6th
  evidence chunk changed between two identical runs at temperature 0. Noise on AR/EN not yet measured.

## Incidents and withdrawn claims
- Dev A1 of 77.1 vs official 88.6: the 77.1 run and the environment it ran in are not recorded in the repo; no claim
  made. Recorded: 88.6 (Hit@6, dev) was reproduced on 2026-10-23 at commit 19a58b6 (docs/baseline_v1.1_report.md §9,
  results/repro_agent_v2_dev.txt); results/plan_vs_raw_dev.txt (committed in e331e72) shows A1 at 88.6; the rag-app
  environment was pinned in requirements.lock.txt by commit fda965a (2026-10-23 22:08). Conclusions built on the 77.1
  run were discarded.
- Detached HEAD incident (git reflog, 2026-10-26): at 17:05 HEAD was checked out to commit 0dba643 (detached); at 17:07
  commit a63037f ("Freeze Analyze/Plan prompts and code descriptions (variant D)") was made there. Its only change,
  `retrieve_for_plan(sqs, soft=True)` in agent/agent_graph.py, was therefore not on agent-v2. At 19:22 HEAD returned to
  agent-v2 and a63037f was cherry-picked as 56a6612; branch rescue/detached-a63037f keeps the original commit. Tag
  agent-prompts-frozen (e331e72) contains 56a6612.
- Test split run twice (first run discarded because a variant was skipped). The test split is considered seen.
- Withdrawn hypothesis: the wording of the Arabic hypothesis is not recorded; no claim made. Recorded reason: v1.1 Hit@6
  dev -> test by language was Arabic 75.0 -> 76.9 (n 8 -> 13), English 100.0 -> 83.3 (n 8 -> 12), French 89.5 -> 81.5
  (n 19 -> 27) (results/repro_agent_v2_dev.txt, results/final_v1_1_test.txt): on the test split English lost most, not
  Arabic.

## Pre-registration: final evaluation (100 questions, written BEFORE running)
Set: ~70 answerable, ~15 ambiguous, ~15 out of scope; systems v1, v1.1, v2; one run; no tuning after the run.
Primary metrics: correct (judge: all reference points covered, no contradiction) and answer-level hallucination.
With ~70 answerable questions each rate has a 95% interval of about +-10-12 points, so only large gaps are detectable.
- H1: v2 correct rate >= v1.1 + 10 points. [edit]
- H2: v2 hallucination rate <= half of v1.1's. [edit]
- H3: ambiguity recall >= 70%, false clarification on answerable questions <= 5%. [edit]
- H4: refusal recall on out-of-scope >= 90%. [edit]
Rule: if the paired 95% interval of a difference includes 0, report "no detectable difference". Failed hypotheses are
reported as failed. Human audit of 30 answers (including Arabic and English) to check the judge.

## Backlog (pinned)
- v1.2: splitter heading repair + footer typo (new index, rerun dev numbers).
- Verifier calibration for Arabic/English claims.
- Query for retrieval after a clarification: user turns only.
- Ambiguity gate set (separate from the 100 questions); "Can I break my contract?" is still classified clear.

## PRE-REGISTRATION ADDENDUM (written 2026-11-01, before any exam item was run)
System under test: v2-candidate. Fingerprint (sha256 over agent/*.py except test_*/run_*,
config.py, splitter.py, retriever.py, reranker.py, pipeline.py, plus index path and chunk
count): dc8fd664c66230b19fa88aad2195be1ab8b67c5be5540a010ba091c4537ec5ca (index db_v1_1, 6031 chunks).
Exam: eval/answer_set.json sha256 61939392c495ee52c528575dd2c286d4e35fbedf780cca58ed67baac777a1c6e;
eval/stages.json sha256 621dd54387ae48abdd05c97d3b6fe946cf555a9700b41dd9b9be4a0949a7401b.
Judge: gemini-3.1-flash-lite, thinking minimal, chosen by the rule in eval/judge_check.py
(>=5/6 successes, median latency <30 s, both canned verdicts correct).
Design: Stage 1 (50 items) is primary; Stage 2 (50) is confirmation. No pooling across
stages or versions. Paired bootstrap CIs over questions (5000 resamples, seed 20261022).
No agent change between the start of Stage 1 and its end. If the agent changes after
Stage 1, the new version is tested on Stage 2 only and reported as a separate version.
Pre-registered thresholds:
- Human audit of 30 answers: judge/human agreement >= 80%, otherwise judge results are
  reported as unreliable.
- Ambiguous items: v2 asks for clarification on >= 60% of the 10 unscripted items;
  over-clarification (asking on answerable items) <= 10%.
- Out-of-scope items: >= 80% "no substantive answer" (explicit refusal or "not found").
- Latency: separate warm benchmark of 10 questions, not taken from the batch runs.
Reporting rules: persistent errors count as system failures (a sensitivity line treats
them as incorrect). Degraded items are rerun, not scored. A difference whose CI includes
zero is reported as "no detectable difference". Known weaknesses are reported, not fixed:
support is not applicability (recel, periode d'essai art. 370), a date read as an article
number, ambiguity gate recall 8/10, verifier calibrated only on blatant French mutations,
few Arabic items in Stage 1.

## v2.1: the agent keeps 8 articles instead of 6 (2026-11-24, config.AGENT_TOP_K)
On the deployed app the agent answered "Quelle est la peine pour le vol simple ?" without article 264, the article that
gives the penalty. Article 264 says "tous autres vols", never "vol simple", so the reranker places it low and a limit of 6
can cut it, at the first search or when a reformulation's results are merged in.
Measured on the real retrieval functions with Groq disabled, normalized queries and reformulations written by hand:
article 264 is in the evidence after the first search in 8/10 cases with 6 and 9/10 with 8, after one reformulation in
16/20 and 18/20, after two in 18/20 and 20/20.
Dev split, French questions with gold articles (17), first search, normalization off: gold recall 0.739 with 6 and 0.759
with 8 (1 question better, 0 worse), evidence about 980 tokens and 1290. This is a small gain on few questions.
Not measured: the final answers. The saved exam runs describe the agent with 6; by the pre-registration above, the new
version is to be tested on Stage 2 only. `AGENT_TOP_K = 6` gives back exactly the old retrieval (agent/test_normalize_switch.py).

## v2.1: answer style, targeted repair, articles sent once to the verifier (2026-11-24)
Three switches in config.py, each with the old value given back exactly (tests compare with the old prompts and graph).
`ANSWER_STYLE = "natural"`: a short plain explanation (the rule, then the main conditions), details only on request; used by
the agent and by the one-pass pipeline. The law's date and the amendment or repeal of an article are not written by the
model: ui/laws.py reads them from the PDFs and the article text.
`REPAIR_MODE = "targeted"`: in the saved exam runs 22 of 24 repairs had at least one confirmed sentence, yet the repair
rewrote everything (no final sentence was identical to a draft sentence, 67 draft sentences became 47). Now only the rejected
sentences are rewritten and checked again; if all are rejected, the whole answer is rewritten as before.
`VERIFY_TEXT_ONCE = True`: 674 -> 490 tokens of claims and articles per check on the saved drafts.
Not measured: the effect on the answers (no model call was made). The saved exam runs describe the old behaviour.

## v2.1: short check on the real model and the fixes it led to (2026-11-24/25, results/short_check)
10 dev questions (6 used for the fixes, 4 fresh), answers read by hand against the articles: an indication, not an exam.
Final agent 8 right and 2 partly right, old agent (all switches back) 8 and 2, one-pass natural prompt 5 right, 4 partly, 1 wrong.
Tokens per question (cached tokens of analyze not counted twice): 11,840, 11,572 and 1,824. Gold articles in evidence: 19/30,
19/30, 15/30. Results vary between runs: the same 6 questions cost 8,668, 10,792 and 12,937 in three runs of the new agent.
- The verifier rejected "vol simple" because article 264 does not use these words: 6 times of 6 with the v2 prompt, 0 of 6 with
  the new one (VERIFY_QUESTION). Corrupted claims still flagged: 72/72 with a first version of the prompt, 30/30 with the final one.
- Less reasoning in grade was tried and removed: it called good evidence insufficient (one question 8,420 -> 20,461 tokens).
- Sending each article once to the verifier was tried and switched off: fewer correct claims accepted in a small test (5/8, 6/8).
- A linked article cut by XREF_TOKEN_CAP = 300 lost the condition of its penalty and produced a false rule; the cap is now 600.
- Known weaknesses left: in Arabic the model writes "wife" for "femme" and no verifier version sees it (2 of 3 accepted); rules
  of private international law still appear in answers about ordinary cases; grade costs 18% of the tokens and its extra
  searches helped in 1 of the 6 runs that had them.

## v2.1: the answer step judges the articles, no separate grade step (2026-11-25, config.ANSWER_FIRST)
The grade call (about 2,000 tokens) is gone: the answer call first says whether the articles answer, then writes the answer. If
they do not, the search is rewritten and run once more (MAX_EXTRA_SEARCHES = 1); if the question turns out to be unclear the user
is asked, once; if nothing answers, the agent says so instead of describing the closest article.
Real model, 6 of the 10 dev questions (results/short_check/run4_answer_first.jsonl): 7,991 tokens per question against 12,636 for the
previous version and 11,433 for the old agent on the same 6; 5 right and 1 partly right, as before, and the assault answer is now
complete. One run per question, and costs vary between runs. All six were judged "answers"; the marriage question in Arabic was
incomplete and still judged so. The new-search path was seen once, on a question outside the corpus (speeding): judged not found,
searched again, said plainly. The post-search clarification was only tested with stubs.

## v2.1: a stabbing question, what was fixed and what is still off (2026-11-25, results/short_check/run5-7)
On: every step of the agent waits for the provider's per-minute limit (llm_retry), so an answer is no longer shown as not verified
because the check was refused; seen on the real model (a follow-up right after a question: 95 s, verified).
Off until tested on other questions: LIST_CASES (the cases listed before writing, 8 sentences: the stabbing answer becomes
complete, 218, 219 and 208) and FAST_JUDGED (Rapide with the agent's search and judging step). Rapide's error, a penalty for a
death given for any injury, came from its search: without article 218 the model stretched article 208, in free text (4 runs of 4) and
with the judging step alone (2 of 2); with the agent's search the answer was right (1 run), at about 5,000 tokens instead of 1,700.
Rule fixed before any run: the 10 unseen dev questions of results/short_check/heldout_set.json, each system's answers read without
knowing which system wrote them, graded right, partly right or wrong against the gold articles. A switch is turned on only if it has at
least as many right answers and no more wrong ones than without it.

