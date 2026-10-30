"""Stability check: the full agent graph run 3 times on 4 dev questions (calls Groq, loads the models).

    python eval/noise_check.py            -> prints the report, saves results/noise_check.json
    python eval/noise_check.py --extra    -> the same 4 questions + 2 Arabic + 2 English, saves results/noise_check_extra.json
    python eval/noise_check.py [--extra] --list   -> only prints the selected questions (no model, no LLM)

Questions: the first dev question (eval/questions.json order) of each of the first 4 concepts met.
--extra adds, in the same order, the first Arabic question of each further concept until 2 are taken, then the
first English question of each further concept until 2 are taken (concepts never repeat). An extra question is
taken only if the Analyze intent recorded for it in results/plan_vs_raw_dev.json (dev run, frozen prompts) is
"clear": q055 was recorded "ambiguous" and would stop at clarify. Extra questions run 2 times each by default
(--extra-runs), the first 4 questions 3 times.
Each run uses a fresh thread_id, so no state is shared between runs. Transient errors (connection, timeout, 429)
left after the client's retries are retried 3 times (waits 10/30/60 s) on a new thread_id.
"""
import sys
import json
import time
import argparse
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
from evaluate import load_questions

RUNS = 3
EXTRA_RUNS = 2
N_QUESTIONS = 4
EXTRA = (("ar", 2), ("en", 2))
WAITS = (10, 30, 60)
INTENTS = ROOT / "results" / "plan_vs_raw_dev.json"
OUT = ROOT / "results" / "noise_check.json"
OUT_EXTRA = ROOT / "results" / "noise_check_extra.json"


def recorded_intents() -> dict:
    """{question id: Analyze intent} from the dev run of eval_plan_vs_raw.py (notes)."""
    if not INTENTS.exists():
        return {}
    return {n["id"]: n.get("intent") for n in json.loads(INTENTS.read_text(encoding="utf-8")).get("notes", [])}


def pick_questions(extra: bool = False) -> list[dict]:
    dev = load_questions("dev")
    intents = recorded_intents()
    seen, out = set(), []
    for q in dev:
        if q["concept_id"] not in seen:
            seen.add(q["concept_id"])
            out.append(q)
        if len(out) == N_QUESTIONS:
            break
    if extra:
        for lang, n in EXTRA:
            taken = 0
            for q in dev:
                if taken == n:
                    break
                if q["language"] == lang and q["concept_id"] not in seen and intents.get(q["id"]) == "clear":
                    seen.add(q["concept_id"])
                    out.append(q | {"extra": True})
                    taken += 1
    return out


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def transient(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}"
    return any(k in s for k in ("RateLimit", "429", "APIConnection", "Connect", "Timeout", "RemoteProtocol"))


def invoke_retry(app, question: str, thread: str):
    for attempt in range(len(WAITS) + 1):
        try:
            return app.invoke({"question": question}, {"configurable": {"thread_id": f"{thread}-a{attempt}"}})
        except Exception as e:
            if not transient(e) or attempt == len(WAITS):
                raise
            print(f"  transient error ({type(e).__name__}); retry in {WAITS[attempt]} s", file=sys.stderr, flush=True)
            time.sleep(WAITS[attempt])


def temperature_lines() -> list[str]:
    """Where the shared llm's temperature is set (read-only quote)."""
    lines = (ROOT / "agent" / "analyze.py").read_text(encoding="utf-8").splitlines()
    return [f"agent/analyze.py:{i}: {l.strip()}" for i, l in enumerate(lines, 1) if "temperature" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true", help="add 2 Arabic and 2 English dev questions (other concepts)")
    ap.add_argument("--extra-runs", type=int, default=EXTRA_RUNS, help="runs per extra question (default 2)")
    ap.add_argument("--list", action="store_true", help="print the selected questions and stop (no model, no LLM)")
    args = ap.parse_args()
    out_path = OUT_EXTRA if args.extra else OUT
    questions = pick_questions(args.extra)
    n_runs = lambda q: args.extra_runs if q.get("extra") else RUNS
    if args.list:
        intents = recorded_intents()
        for q in questions:
            print(f"{q['id']:<6}{q['language']:<4}{q['concept_id']:<34}runs {n_runs(q)}  recorded intent "
                  f"{intents.get(q['id'])!s:<10}{q['question']}")
        print(f"{len(questions)} questions, {sum(map(n_runs, questions))} runs; would save {out_path.relative_to(ROOT)}")
        return
    from agent_graph import app
    print("Shared llm temperature (quoted, read-only):")
    for l in temperature_lines():
        print("  " + l)
    data = {"runs_per_question": RUNS, "runs_per_extra_question": args.extra_runs if args.extra else None,
            "temperature_lines": temperature_lines(), "questions": []}
    stopped = None
    for q in questions:
        runs = []
        for k in range(1, n_runs(q) + 1):
            print(f"  {q['id']} run {k}/{n_runs(q)}...", file=sys.stderr, flush=True)
            t0 = time.time()
            try:
                s = invoke_retry(app, q["question"], f"noise-{q['id']}-{k}")
            except Exception as e:
                stopped = f"{q['id']} run {k}: {type(e).__name__}: {e}"
                break
            runs.append({"status": s.get("status"), "intent": s.get("intent"), "retries": s.get("retries"),
                         "tried": s.get("tried"), "evidence_ids": list(s.get("evidence") or {}),
                         "cited": s.get("cited"), "claims": s.get("claims"), "caveat": s.get("caveat"),
                         "degraded": s.get("degraded"), "seconds": round(time.time() - t0, 1)})
        data["questions"].append({"id": q["id"], "concept_id": q["concept_id"], "language": q["language"],
                                  "question": q["question"], "extra": bool(q.get("extra")), "runs": runs})
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")   # saved after each question
        if stopped:
            break

    print(f"\n{'id':<6}{'lang':<5}{'runs':>5}  {'same 6 evidence ids':<20}{'mean Jaccard cited':>19}{'retries':>10}  question")
    jac_all = []
    for q in data["questions"]:
        rs = q["runs"]
        same_ev = len({tuple(r["evidence_ids"]) for r in rs}) == 1 if rs else None
        pairs = [jaccard(a["cited"] or [], b["cited"] or []) for a, b in combinations(rs, 2)]
        mj = sum(pairs) / len(pairs) if pairs else None
        if mj is not None:
            jac_all.append(mj)
        retr = [r["retries"] for r in rs if r["retries"] is not None]
        q["summary"] = {"same_evidence_ids": same_ev, "mean_pairwise_jaccard_cited": mj,
                        "retries_min": min(retr) if retr else None, "retries_max": max(retr) if retr else None}
        print(f"{q['id']:<6}{q['language']:<5}{len(rs):>5}  {str(same_ev):<20}"
              f"{(f'{mj:.3f}' if mj is not None else '-'):>19}{(f'{min(retr)}-{max(retr)}' if retr else '-'):>10}  {q['question']}")
        for k, r in enumerate(rs, 1):
            print(f"      run {k}: status={r['status']} retries={r['retries']} degraded={r['degraded']} {r['seconds']}s"
                  f"\n             evidence={r['evidence_ids']}\n             cited={r['cited']}\n             tried={r['tried']}")
    data["overall_mean_jaccard_cited"] = sum(jac_all) / len(jac_all) if jac_all else None
    print(f"\nOverall mean pairwise Jaccard of cited sets: "
          f"{data['overall_mean_jaccard_cited']:.3f}" if jac_all else "\nOverall mean Jaccard: -")
    print("(Jaccard of two empty cited sets is counted as 1.0)")
    if stopped:
        data["stopped"] = stopped
        print(f"\nSTOPPED: {stopped}")
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {out_path.relative_to(ROOT)}")
    sys.exit(1 if stopped else 0)


if __name__ == "__main__":
    main()
