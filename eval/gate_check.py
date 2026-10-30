"""Gate check: the Analyze step alone on eval/gate_set.json, scored against the pre-registered eval/gate_rule.txt.

    python eval/gate_check.py --dry-run               # lists the questions and readings, no model, no LLM call
    python eval/gate_check.py [--repeat 2] [--min-gap 3]

Calls agent/analyze.analyze(question, None), the entry point analyze_node uses on a first turn (no retrieval, no
other node): one Groq call per question and repeat. Transient errors (connection, timeout, 429) left after the
client's retries are retried 3 times (waits 10/30/60 s); a failure after that stops the run (partial results saved).
Saves results/gate_check.json: per question, intent / language / target_codes / clarification question per repeat.

Rule (eval/gate_rule.txt), applied to EACH repeat separately; the overall verdict is PASS only if every repeat passes:
  ambiguity recall >= 7/10; at most 1 of the 6 clear questions flagged "ambiguous";
  all 4 out_of_scope questions refused (intent out_of_scope); language correct on all 20.
"""
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
SET = ROOT / "eval" / "gate_set.json"
RULE = ROOT / "eval" / "gate_rule.txt"
OUT = ROOT / "results" / "gate_check.json"
CLASSES = ("ambiguous", "clear", "out_of_scope")
WAITS = (10, 30, 60)


def transient(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}"
    return any(k in s for k in ("RateLimit", "429", "APIConnection", "Connect", "Timeout", "RemoteProtocol"))


def call_retry(fn, question):
    for attempt in range(len(WAITS) + 1):
        try:
            return fn(question, None)
        except Exception as e:
            if not transient(e) or attempt == len(WAITS):
                raise
            print(f"  transient error ({type(e).__name__}); retry in {WAITS[attempt]} s", file=sys.stderr, flush=True)
            time.sleep(WAITS[attempt])


def score(items, k) -> dict:
    """Rule quantities for repeat k (0-based); questions without a result for this repeat count as errors."""
    conf = {g: {p: 0 for p in CLASSES + ("error",)} for g in CLASSES}
    lang_err = []
    for it in items:
        r = it["repeats"][k] if k < len(it["repeats"]) else {"intent": "error", "language": None}
        pred = r.get("intent") if r.get("intent") in CLASSES else "error"
        conf[it["gold"]][pred] += 1
        if r.get("language") != it["language"]:
            lang_err.append(it["id"])
    n = {g: sum(conf[g].values()) for g in CLASSES}
    s = {"confusion": conf, "ambiguity_recall": conf["ambiguous"]["ambiguous"], "n_ambiguous": n["ambiguous"],
         "over_clarification": conf["clear"]["ambiguous"], "n_clear": n["clear"],
         "refused": conf["out_of_scope"]["out_of_scope"], "n_out_of_scope": n["out_of_scope"],
         "language_errors": lang_err}
    s["checks"] = {"ambiguity recall >= 7/10": s["ambiguity_recall"] >= 7,
                   "at most 1 clear flagged ambiguous": s["over_clarification"] <= 1,
                   "all out_of_scope refused": s["refused"] == n["out_of_scope"],
                   "language correct on all": not lang_err}
    s["pass"] = all(s["checks"].values())
    return s


def report(items, repeat):
    print("\nRule (eval/gate_rule.txt):\n" + RULE.read_text(encoding="utf-8"))
    verdicts = []
    for k in range(repeat):
        s = score(items, k)
        verdicts.append(s)
        print(f"=== repeat {k + 1}/{repeat}: confusion (rows gold, columns predicted)")
        print(f"  {'':<14}" + "".join(f"{p:>14}" for p in CLASSES + ("error",)))
        for g in CLASSES:
            print(f"  {g:<14}" + "".join(f"{s['confusion'][g][p]:>14}" for p in CLASSES + ("error",)))
        print(f"  ambiguity recall {s['ambiguity_recall']}/{s['n_ambiguous']}; clear flagged ambiguous "
              f"{s['over_clarification']}/{s['n_clear']}; out_of_scope refused {s['refused']}/{s['n_out_of_scope']}; "
              f"language errors {len(s['language_errors'])} {s['language_errors']}")
        for name, ok in s["checks"].items():
            print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    dis = [it["id"] for it in items
           if len({(r.get("intent"), r.get("language")) for r in it["repeats"]}) > 1]
    print(f"\nDisagreements between repeats (intent or language): {len(dis)} {dis}")
    overall = all(v["pass"] for v in verdicts) and all(len(it["repeats"]) == repeat for it in items)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'} (every repeat must pass)")
    return verdicts, dis, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--min-gap", type=float, default=3, metavar="SECONDS", help="sleep between Groq calls (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="list the questions, no model, no LLM call")
    args = ap.parse_args()
    gate = json.loads(SET.read_text(encoding="utf-8"))
    counts = {g: sum(it["gold"] == g for it in gate) for g in CLASSES}
    print(f"{SET.relative_to(ROOT).as_posix()}: {len(gate)} questions {counts}; repeat {args.repeat}; "
          f"Groq calls: {len(gate) * args.repeat}; min gap {args.min_gap} s")
    if args.dry_run:
        for it in gate:
            print(f"\n{it['id']}  {it['language']}  {it['gold']:<13} {it['question']}")
            for r in it.get("readings", []):
                print(f"      reading: {r['reading']}  -> {r['code']}")
            if it.get("where"):
                print(f"      expected: {it['code']} ({it['where']})")
            if it.get("corpus_check"):
                print(f"      corpus check: {it['corpus_check']}")
        print("\n--dry-run: nothing called. Rule (eval/gate_rule.txt):\n" + RULE.read_text(encoding="utf-8"))
        return

    import analyze as an
    sha = hashlib.sha1(an.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    print(f"analyze SYSTEM_PROMPT sha1 {sha}")
    items = [{k: it[k] for k in ("id", "question", "language", "gold")} | {"repeats": []} for it in gate]
    data = {"analyze_prompt_sha1": sha, "repeat": args.repeat, "min_gap": args.min_gap, "items": items}
    stopped, first = None, True
    for k in range(args.repeat):
        for it in items:
            if not first and args.min_gap:
                time.sleep(args.min_gap)
            first = False
            print(f"  repeat {k + 1} {it['id']}...", file=sys.stderr, flush=True)
            try:
                r = call_retry(an.analyze, it["question"])
            except Exception as e:
                stopped = f"repeat {k + 1} {it['id']}: {type(e).__name__}: {str(e)[:300]}"
                break
            it["repeats"].append({"intent": r.intent, "language": r.language, "target_codes": r.target_codes,
                                  "clarification_question": r.clarification_question})
            OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        if stopped:
            break
    print(f"\n{'id':<5}{'lang':<5}{'gold':<14}repeats (intent/language)")
    for it in items:
        print(f"{it['id']:<5}{it['language']:<5}{it['gold']:<14}"
              + "  ".join(f"{r['intent']}/{r['language']}" for r in it["repeats"]) + f"   {it['question']}")
    verdicts, dis, overall = report(items, args.repeat)
    data |= {"per_repeat": verdicts, "disagreements": dis, "overall_pass": overall}
    if stopped:
        data["stopped"] = stopped
        print(f"\nSTOPPED: {stopped}")
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT).as_posix()}")
    sys.exit(1 if stopped else 0)


if __name__ == "__main__":
    main()
