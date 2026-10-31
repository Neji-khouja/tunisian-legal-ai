"""Canary: run ONLY the Analyze step (agent/analyze.py, one Groq call each) on 5 fixed dev questions and append
intent / language / target_codes with the date to results/canary_log.json, to see whether the model drifts over days.

    python eval/canary.py --dry-run      # list the questions, no call
    python eval/canary.py                # 5 Analyze calls
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
OUT = ROOT / "results" / "canary_log.json"
QUESTIONS = [   # dev questions of eval/questions.json (concepts in the dev split of eval/split.json): 2 fr, 2 en, 1 ar
    ("q010", "fr", "Comment est puni l'incendie volontaire ?"),
    ("q026", "fr", "Quelles sont les conditions du mariage ?"),
    ("q078", "en", "How long can police custody last?"),
    ("q082", "en", "What makes a contract valid?"),
    ("q057", "ar", "ما هي مدة الإجازة السنوية المدفوعة؟"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"canary: {len(QUESTIONS)} dev questions, Analyze only")
    for qid, lang, q in QUESTIONS:
        print(f"  {qid} ({lang}) {q}")
    if args.dry_run:
        return
    from analyze import analyze, SYSTEM_PROMPT
    import hashlib
    run = {"date": datetime.now().isoformat(timespec="seconds"),
           "analyze_prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12], "results": []}
    for qid, lang, q in QUESTIONS:
        try:
            a = analyze(q)
            r = {"id": qid, "intent": a.intent, "language": a.language, "target_codes": a.target_codes}
        except Exception as e:
            r = {"id": qid, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        run["results"].append(r)
        print(f"  {qid}: {r}")
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    log.append(run)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"appended to {OUT.relative_to(ROOT).as_posix()} ({len(log)} runs)")


if __name__ == "__main__":
    main()
