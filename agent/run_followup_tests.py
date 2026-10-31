"""Manual check of follow-up handling (calls Groq and loads the models). Run it yourself:

    python agent/run_followup_tests.py [--min-gap 3] [--only fr-drill,en-switch]

6 scripted dialogues (2 fr, 2 en, 2 ar), each = a first question, then a follow-up on the SAME thread_id.
Prints per dialogue: the follow-up, the standalone rewrite, is_followup, the cited article ids and the final text.
No scoring: read the output. Saves results/followup_tests.json after every dialogue; an error stops the run
(completed dialogues are kept in the file).
"""
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
OUT = ROOT / "results" / "followup_tests.json"

DIALOGUES = [   # (name, kind, first question, follow-up)
    ("fr-drill", "drill-down (travail)",
     "Quelle est la durée maximale de la période d'essai ?", "Et pour les cadres ?"),
    ("fr-pronoun", "follow-up with a pronoun (penal)",
     "Qu'est-ce que le recel d'objets volés ?", "Et quelle est sa peine ?"),
    ("en-drill", "drill-down (statut_personnel)",
     "Who inherits when a person dies?", "And if there are no children?"),
    ("en-switch", "topic switch: the follow-up must stay unchanged",
     "What are the formalities to create a limited liability company?",
     "Can a judgment given abroad be enforced in Tunisia?"),
    ("ar-drill", "drill-down (nationalite)",
     "ما هي شروط اكتساب الجنسية التونسية بالتجنس؟", "وبالنسبة للقصّر؟"),
    ("ar-lang", "follow-up that changes language (ar -> fr)",
     "ما هي مدة عطلة الأمومة؟", "Et pour le père ?"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-gap", type=float, default=3, metavar="SECONDS", help="sleep between graph runs (default 3)")
    ap.add_argument("--only", help="comma-separated dialogue names")
    args = ap.parse_args()
    from agent_graph import app
    from condense import PROMPT_SHA1
    todo = [d for d in DIALOGUES if not args.only or d[0] in args.only.split(",")]
    data = {"condense_prompt_sha1": PROMPT_SHA1, "min_gap": args.min_gap, "dialogues": []}
    print(f"condense prompt sha1 {PROMPT_SHA1}; {len(todo)} dialogues, 2 graph runs each; min gap {args.min_gap} s")
    first = True
    for name, kind, q1, q2 in todo:
        cfg = {"configurable": {"thread_id": f"followup-{name}-{time.time_ns()}"}}
        rec = {"name": name, "kind": kind, "turns": []}
        print(f"\n=== [{name}] {kind}")
        try:
            for q in (q1, q2):
                if not first and args.min_gap:
                    time.sleep(args.min_gap)
                first = False
                s = app.invoke({"question": q}, cfg)
                rec["turns"].append({"message": q, "standalone": s.get("question"),
                                     "condensed_from": s.get("condensed_from"), "is_followup": s.get("is_followup"),
                                     "status": s.get("status"), "cited": s.get("cited") or [],
                                     "text": s.get("text"), "degraded": s.get("degraded")})
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        data["dialogues"].append(rec)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        for k, t in enumerate(rec["turns"], 1):
            print(f"  turn {k}: {t['message']!r} -> status {t['status']}, degraded {t['degraded']}")
            if k == 2:
                print(f"    standalone rewrite: {t['standalone']!r}")
                print(f"    is_followup: {t['is_followup']}")
            print(f"    cited: {t['cited']}")
            print("    text: " + (t["text"] or "").replace("\n", "\n          "))
        if "error" in rec:
            print(f"\nSTOPPED on {name}: {rec['error']}\nSaved {OUT.relative_to(ROOT).as_posix()}")
            sys.exit(1)
    print(f"\nSaved {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
