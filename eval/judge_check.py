"""Check a judge model before the exam runs: same judge prompt, schema, thinking setting and retries as
eval/run_answer_eval.py, on 2 canned cases x 3 repeats.

    python eval/judge_check.py --judge-provider gemini --judge-model gemini-3.8-flash --list   # models the key can use
    python eval/judge_check.py --judge-provider gemini --judge-model gemini-3.8-flash           # 6 judge calls
    python eval/judge_check.py --judge-provider groq --judge-model <id> --dry-run               # no call

Defaults as in the harness: --judge-provider groq, --judge-model from env JUDGE_MODEL, --judge-thinking minimal.
Canned cases: one question built from PenalFr art. 264 (verbatim PDF text, chunk PenalFr/s02/264; not in
eval/answer_set.json), a correct answer (expected verdict "correct") and a clearly wrong one (expected "incorrect").
--list prints model names only (Gemini: models supporting generateContent; Groq: the models list), never a key.
"""
import os
import sys
import time
import argparse
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import run_answer_eval as rae

CID = "PenalFr/s02/264"
ARTICLE = ("Article 264.- La peine est de cinq ans d’emprisonnement et de \ncent vingt dinars d’amende, pour tous autres "
           "vols et soustractions \ncommis hors les cas prévus aux articles 260 à 263 du présent code. \nLa tentative est "
           "punissable.")
ITEM = {"id": "judge-check", "question": "Quelle peine le code pénal prévoit-il pour un vol simple, commis hors les cas "
                                         "aggravés des articles 260 à 263 ?",
        "language": "fr", "expected": "answer", "gold": [{"source": "PenalFr.pdf", "article": "264"}],
        "reference_points": ["cinq ans d'emprisonnement", "cent vingt dinars d'amende"]}
CASES = [("correct answer", "correct",
          "Le vol simple est puni de cinq ans d'emprisonnement et de cent vingt dinars d'amende (article 264 du code "
          "pénal) ; la tentative est punissable."),
         ("wrong answer", "incorrect",
          "Le vol simple est puni d'une amende de cinquante dinars seulement, sans emprisonnement (article 264 du code "
          "pénal) ; la tentative n'est pas punissable.")]
REPEATS = 3
RULE = ("Selection rule: the first candidate with >= 5/6 successes, median latency < 30 s and both canned verdicts "
        "correct; otherwise a second Groq model as judge.")


def run_cases(provider: str, model: str, repeats: int = REPEATS, clock=time.perf_counter) -> list[dict]:
    """Each call goes through rae.judge (prompt, schema, thinking, invoke_judge retries, as in a harness run)."""
    out = []
    for r in range(1, repeats + 1):
        for name, want, text in CASES:
            t0 = clock()
            try:
                j = rae.judge(ITEM, {"text": text, "cited": [CID], "_evidence_text": {CID: ARTICLE}}, model, provider)
                res = {"ok": "verdict" in j, "verdict": j.get("verdict"), "error": j.get("error")}
            except Exception as e:
                res = {"ok": False, "verdict": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            res |= {"case": name, "expected": want, "repeat": r, "latency": round(clock() - t0, 1)}
            out.append(res)
            print(f"  repeat {r} {name:<15} {'ok   ' if res['ok'] else 'error'} {res['latency']:6.1f} s  verdict "
                  f"{res['verdict']!s:<10} (expected {want})" + (f"  {res['error']}" if res["error"] else ""))
    return out


def summary(results: list[dict]) -> dict:
    ok = [r for r in results if r["ok"]]
    med = statistics.median(r["latency"] for r in results) if results else None
    verdicts_ok = all(any(r["case"] == name and r["ok"] for r in results) for name, _, _ in CASES) and \
        all(r["verdict"] == r["expected"] for r in ok)
    passes = len(ok) >= 5 and med is not None and med < 30 and verdicts_ok
    return {"successes": len(ok), "calls": len(results), "median_latency": med, "verdicts_as_expected": verdicts_ok,
            "passes": passes}


def list_models(provider: str) -> list[str]:
    if provider == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        return sorted(m.name for m in client.models.list() if "generateContent" in (m.supported_actions or []))
    from groq import Groq
    return sorted(m.id for m in Groq().models.list().data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judge-provider", choices=["groq", "gemini"], default="groq")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    ap.add_argument("--judge-thinking", choices=["minimal", "default"], default="minimal")
    ap.add_argument("--list", action="store_true", help="print the model names the key can use")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rae.JUDGE_THINKING = args.judge_thinking
    import config  # noqa: F401  (load_dotenv: GOOGLE_API_KEY / GROQ_API_KEY for the calls)
    if args.list:
        for name in list_models(args.judge_provider):
            print(name)
        return
    think = rae.judge_thinking(args)
    print(f"judge {args.judge_provider} {args.judge_model or 'NOT SET'}; prompt sha {rae.JUDGE_PROMPT_SHA}; thinking "
          f"{think if think is not None else 'n/a (groq)'}; {len(CASES)} cases x {REPEATS} repeats; article {CID}")
    print(f"question: {ITEM['question']}\nreference points: {ITEM['reference_points']}")
    for name, want, text in CASES:
        print(f"  {name} (expected {want}): {text}")
    print(RULE)
    if args.dry_run:
        print("--dry-run: no call made.")
        return
    if not args.judge_model:
        sys.exit("No judge model: set JUDGE_MODEL or pass --judge-model.")
    s = summary(run_cases(args.judge_provider, args.judge_model))
    med = "-" if s["median_latency"] is None else f"{s['median_latency']:.1f} s"
    print(f"\nsuccesses {s['successes']}/{s['calls']}; median latency {med}; both verdicts as expected: "
          f"{'yes' if s['verdicts_as_expected'] else 'no'}; meets the rule: {'yes' if s['passes'] else 'no'}")
    sys.exit(0 if s["passes"] else 1)


if __name__ == "__main__":
    main()
