"""Read-only checks before the final answer-level evaluation. No Groq call, no tag, no worktree is created.

    python eval/preflight.py [--v1-worktree ../RAG_PDF_v1] [--judge-provider groq|gemini] [--judge-model ID]
"""
import os
import sys
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import harness_core as hc
EXPECTED_HASHES = {"analyze": "0bbb0226bdc9", "plan": "04c1c55cf119", "codes": "7f49e7a7c8c1",
                   "combined": "e4c6a65fda60", "verify": "6568dc934538", "grade": "4a25ce5f5b52",
                   "reformulate": "5d689216e8cc"}
TYPES = {"answer": "answerable", "clarify": "ambiguous", "refuse": "out_of_scope"}
V1_TAG = "v1-baseline-runnable"
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))


def sha(text: str) -> str:            # as agent/eval_plan_vs_raw.py
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1-worktree", default=str(ROOT.parent / "RAG_PDF_v1"),
                    help="v1 worktree to check (default: ../RAG_PDF_v1, a suggestion; run_answer_eval.py has no default)")
    ap.add_argument("--judge-provider", choices=["groq", "gemini"], default=hc.env_judge_provider())
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    args = ap.parse_args()

    jm = args.judge_model
    record("judge model", bool(jm), ("set" if jm else "unset") + " (--judge-model or JUDGE_MODEL in the process "
                                    "environment, as run_answer_eval.py reads it)")
    record("judge model differs from the agent model", bool(jm) and jm != "openai/gpt-oss-120b",
           f"provider {args.judge_provider}; model {jm or 'unset'}")
    if args.judge_provider == "gemini":
        try:
            import langchain_google_genai  # noqa: F401
            record("langchain-google-genai importable", True)
        except ImportError:
            record("langchain-google-genai importable", False, "pip install langchain-google-genai (see requirements.lock.txt)")

    # 2. uncommitted Python files in agent/ and eval/
    dirty = [l for l in git("status", "--porcelain", "--", "agent", "eval").splitlines()
             if l[3:].endswith(".py") and "__pycache__" not in l]
    record("no uncommitted agent/*.py, eval/*.py", not dirty, ", ".join(dirty) if dirty else "clean")

    # 3. frozen prompt hashes (importing the modules loads no model and calls no network)
    sys.path.insert(0, str(ROOT / "agent"))
    import analyze, plan, grade, reformulate, verify
    got = {"analyze": sha(analyze.SYSTEM_PROMPT), "plan": sha(plan.PLAN_PROMPT),
           "codes": sha((ROOT / "agent" / "codes.py").read_text(encoding="utf-8")),
           "verify": sha(verify.SYSTEM_PROMPT), "grade": sha(grade.GRADE_PROMPT),
           "reformulate": sha(reformulate.REFORMULATE_PROMPT)}
    got["combined"] = sha(got["analyze"] + got["plan"] + got["codes"])
    for k, want in EXPECTED_HASHES.items():
        record(f"hash {k}", got[k] == want, f"{got[k]} (expected {want})")
    import condense
    print(f"INFO  hash condense: {sha(condense.CONDENSE_PROMPT)} (printed only, not checked)")

    # 4. answer set (eval/check_answer_set.py rules: duplicate ids, gold articles in the index and the PDF text)
    sys.path.insert(0, str(ROOT / "eval"))
    import check_answer_set as cas
    entries = json.loads(cas.SET_PATH.read_text(encoding="utf-8"))
    pairs, _ = cas.corpus_index()
    errors, warnings = cas.check(entries, pairs)
    counts = Counter(TYPES.get(e.get("expected"), e.get("expected")) for e in entries if isinstance(e, dict))
    record("eval/answer_set.json counts", bool(entries),
           f"{len(entries)} items {dict(counts) or '{}'}" + ("" if entries else " (empty)"))
    record("eval/answer_set.json valid (check_answer_set.py)", not errors and pairs is not None,
           f"{len(errors)} error(s), {len(warnings)} warning(s)" + ("" if pairs is not None else "; index missing"))
    for msg in errors[:20]:
        print("        -", msg)

    # 4b. exam fingerprint and stages (eval/harness_core.py, as run_answer_eval.py checks them)
    _, diffs = hc.exam_check(cas.SET_PATH, ROOT / "eval" / "stages.json", ROOT / "eval" / "exam_hashes.txt")
    record("eval/exam_hashes.txt matches answer_set.json and stages.json", not diffs, "; ".join(diffs) or "match")
    st = json.loads((ROOT / "eval" / "stages.json").read_text(encoding="utf-8"))
    s1, s2, all_ids = set(st["stage1"]), set(st["stage2"]), {e["id"] for e in entries}
    record("stages disjoint and covering every exam item", not (s1 & s2) and (s1 | s2) == all_ids,
           f"stage 1 {len(s1)}, stage 2 {len(s2)}, overlap {len(s1 & s2)}, exam {len(all_ids)}, "
           f"missing {sorted(all_ids - s1 - s2)}, unknown {sorted((s1 | s2) - all_ids)}")

    # 5. v1 worktree, as eval/run_answer_eval.py V1Process tests it
    wt = Path(args.v1_worktree)
    listed = wt.resolve().as_posix().lower() in git("worktree", "list", "--porcelain").replace("\\", "/").lower()
    ok = (wt / "pipeline.py").exists() and (wt / "db").exists()
    record("v1 worktree", ok and listed,
           f"{os.path.relpath(wt, ROOT)}: pipeline.py {'yes' if (wt / 'pipeline.py').exists() else 'no'}, "
           f"db/ {'yes' if (wt / 'db').exists() else 'no'}, git worktree {'yes' if listed else 'no'}")
    record(f"tag {V1_TAG} exists", V1_TAG in git("tag", "-l").split())

    # 6. INFO: system fingerprint (stored in every row; a result file of another fingerprint is refused) and judge
    sys.path.insert(0, str(ROOT))
    from config import DB_DIR
    import run_answer_eval as rae
    fp = hc.system_fingerprint(ROOT, DB_DIR)
    print(f"INFO  GOOGLE_API_KEY: {'set' if os.environ.get('GOOGLE_API_KEY') else 'unset'} (after config.py loads .env, as "
          f"in a harness run; needed for --judge-provider gemini)")
    print(f"INFO  system fingerprint: {fp['sha256']} ({fp['files']} files: agent/*.py except test_*/run_*, "
          f"{', '.join(hc.FINGERPRINT_FILES)}; index {fp['index']['path']}, {fp['index']['chunks']} chunks)")
    think = rae.gemini_thinking(jm or "") if args.judge_provider == "gemini" else None
    print(f"INFO  judge: provider {args.judge_provider}, model {jm or 'unset'}, thinking "
          f"{'auto (default): starts as ' + str(think) + ', low if minimal is not supported' if think is not None else 'n/a (groq)'}; prompts {rae.JUDGE_PROMPT_SHA} (answer), "
          f"{rae.GATE_PROMPT_SHA} (clarify/refuse)")

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS)} checks, {n_fail} FAIL")
    wt_rel = os.path.relpath(wt, ROOT)
    jp, jmod = args.judge_provider, jm or "JUDGE_MODEL_ID"      # no angle brackets: cmd.exe reads < as a redirection
    judge = f"--judge-provider {jp} --judge-model {jmod}"
    print("\nNext commands for the USER (not run by this script):")
    if not ok:
        print(f"""  git worktree add {wt_rel} {V1_TAG}
  mklink /J {wt_rel}\\db db                            # Windows: the v1 index; or copy db/""")
    print(f"""  git tag FINAL_EVAL_TAG HEAD                         # after PREREG is final and every check passes
  python eval/judge_check.py {judge} --list
  python eval/judge_check.py {judge}
      # 6 judge calls; use this judge if it meets the selection rule, else a second Groq model (same two commands)
  python eval/canary.py                               # Analyze drift check (5 Groq calls), once per run day
  python eval/run_batch.py --stage 1 {judge} --v1-worktree {wt_rel} --dry-run
  python eval/run_batch.py --stage 1 {judge} --v1-worktree {wt_rel} --min-gap 3
      # one batch: 10 questions x v1, v1.1, v2 (one process each); repeat until "nothing left to run"
      # exit code 75 = daily quota or budget reached: re-run the same command (next day) to continue
  python eval/run_answer_eval.py --stage 1 --system all --judge-only {judge}   # if judge_pending rows
  python eval/run_answer_eval.py --summarize""")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
