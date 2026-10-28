"""Stability check: the full agent graph run 3 times on 4 dev questions (calls Groq, loads the models).

    python eval/noise_check.py            -> prints the report, saves results/noise_check.json

Questions: the first dev question (eval/questions.json order) of each of the first 4 concepts met.
Each run uses a fresh thread_id, so no state is shared between runs.
"""
import sys
import json
import time
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
from evaluate import load_questions
from agent_graph import app

RUNS = 3
N_QUESTIONS = 4
OUT = ROOT / "results" / "noise_check.json"


def pick_questions() -> list[dict]:
    seen, out = set(), []
    for q in load_questions("dev"):
        if q["concept_id"] not in seen:
            seen.add(q["concept_id"])
            out.append(q)
        if len(out) == N_QUESTIONS:
            break
    return out


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def temperature_lines() -> list[str]:
    """Where the shared llm's temperature is set (read-only quote)."""
    lines = (ROOT / "agent" / "analyze.py").read_text(encoding="utf-8").splitlines()
    return [f"agent/analyze.py:{i}: {l.strip()}" for i, l in enumerate(lines, 1) if "temperature" in l]


def main():
    print("Shared llm temperature (quoted, read-only):")
    for l in temperature_lines():
        print("  " + l)
    questions = pick_questions()
    data = {"runs_per_question": RUNS, "temperature_lines": temperature_lines(), "questions": []}
    stopped = None
    for q in questions:
        runs = []
        for k in range(1, RUNS + 1):
            print(f"  {q['id']} run {k}/{RUNS}...", file=sys.stderr, flush=True)
            t0 = time.time()
            try:
                s = app.invoke({"question": q["question"]}, {"configurable": {"thread_id": f"noise-{q['id']}-{k}"}})
            except Exception as e:
                stopped = f"{q['id']} run {k}: {type(e).__name__}: {e}"
                break
            runs.append({"status": s.get("status"), "intent": s.get("intent"), "retries": s.get("retries"),
                         "tried": s.get("tried"), "evidence_ids": list(s.get("evidence") or {}),
                         "cited": s.get("cited"), "claims": s.get("claims"), "caveat": s.get("caveat"),
                         "degraded": s.get("degraded"), "seconds": round(time.time() - t0, 1)})
        data["questions"].append({"id": q["id"], "concept_id": q["concept_id"], "language": q["language"],
                                  "question": q["question"], "runs": runs})
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")   # saved after each question
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
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT)}")
    sys.exit(1 if stopped else 0)


if __name__ == "__main__":
    main()
