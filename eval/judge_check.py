"""Check a judge model before the exam runs: same judge prompt, schema, thinking setting and retries as
eval/run_answer_eval.py, on 2 canned cases x 3 repeats.

    python eval/judge_check.py --judge-provider gemini --judge-model gemini-3.8-flash --list   # models the key can use
    python eval/judge_check.py --judge-provider gemini --judge-model gemini-3.8-flash           # 6 judge calls
    python eval/judge_check.py --candidates "gemini:gemini-3.8-flash,gemini:gemini-3.7-flash,groq:openai/gpt-oss-20b"
    python eval/judge_check.py --judge-provider groq --judge-model JUDGE_MODEL_ID --dry-run       # no call

Defaults as in the harness: --judge-provider groq, --judge-model from env JUDGE_MODEL, --judge-thinking auto (Gemini 3+:
minimal, then low for the rest of the run if a 400 says minimal is not supported; groq: n/a).
Canned cases: one question built from PenalFr art. 264 (verbatim PDF text, chunk PenalFr/s02/264; not in
eval/answer_set.json), a correct answer (expected verdict "correct") and a clearly wrong one (expected "incorrect").
--min-gap seconds (default 4) between calls, so the check itself does not hit per-minute limits. After the results:
the first full text of each distinct error type (429s: quota metric, limit, retry delay, per-minute or per-day when the
text says). --candidates "provider:model,...": in this order, stop at the first that meets the rule. Every attempt
(candidate, effective thinking, date/time, results, error texts) is appended to --log (results/judge_check_log.json).
--list prints model names only (Gemini: models supporting generateContent; Groq: the models list), never a key.
"""
import os
import re
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import harness_core as hc
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
LOG = ROOT / "results" / "judge_check_log.json"
EARLIER = ["gemini-3.8-flash minimal: 0/6, 400 + 429",
           "qwen/qwen3.8-27b: 4/6, json errors, one 'partial' verdict on the wrong canned answer"]
PER_MINUTE = re.compile(r"per[ _-]?minute|PerMinute|\bTPM\b|\bRPM\b", re.I)


def run_cases(provider: str, model: str, repeats: int = REPEATS, clock=time.perf_counter, min_gap: float = 0,
              sleep=time.sleep) -> list[dict]:
    """Each call goes through rae.judge (prompt, schema, thinking, retries, as in a harness run); min_gap seconds
    between calls."""
    out = []
    for r in range(1, repeats + 1):
        for name, want, text in CASES:
            if out and min_gap:
                sleep(min_gap)
            t0 = clock()
            try:
                j = rae.judge(ITEM, {"text": text, "cited": [CID], "_evidence_text": {CID: ARTICLE}}, model, provider)
                res = {"ok": "verdict" in j, "verdict": j.get("verdict"), "error": j.get("error"),
                       "error_type": None if "verdict" in j else "judgement error", "error_text": j.get("error")}
            except Exception as e:
                full = hc.full_message(e)
                code = re.search(r"\b(400|401|403|404|429|500|503|504)\b", full)
                res = {"ok": False, "verdict": None, "error": f"{type(e).__name__}: {str(e)[:200]}",
                       "error_type": type(e).__name__ + (f" {code.group(1)}" if code else ""), "error_text": full[:800]}
            res |= {"case": name, "expected": want, "repeat": r, "latency": round(clock() - t0, 1),
                    "thinking": rae.effective_thinking(provider, model)}
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


def describe_429(text: str) -> str:
    """Quota metric, limit, retry delay and period of a rate-limit text, as far as the text states them."""
    period = "per-day" if hc.is_daily_quota(text) else "per-minute" if PER_MINUTE.search(text) else "period not stated"
    metric = re.search(r"metric:\s*([\w./-]+)", text) or re.search(r"quotaId\W+([\w-]+)", text)
    limit = re.search(r"\blimit:\s*(\d+)|\bLimit (\d+)|quotaValue\W+(\d+)", text)
    retry = hc.parse_retry_after(text)
    lim = next(g for g in limit.groups() if g) if limit else "not stated"
    delay = f"{retry:g} s" if retry is not None else "not stated"
    return (f"429 looks {period}; quota metric {metric.group(1) if metric else 'not stated'}; limit {lim}; "
            f"retry delay {delay}")


def error_texts(results: list[dict]) -> dict:
    """error type -> the first full error text of that type (up to 800 characters)."""
    out = {}
    for r in results:
        if r.get("error_type") and r["error_type"] not in out:
            out[r["error_type"]] = r["error_text"] or ""
    return out


def print_errors(errors: dict):
    for kind, text in errors.items():
        print(f"\nfirst error of type {kind}:\n  {text}")
        if hc.is_rate_limit(text):
            print(f"  -> {describe_429(text)}")


def append_log(path: Path, entry: dict):
    path = Path(path)
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    log.setdefault("note", EARLIER)
    log.setdefault("attempts", []).append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def check_candidate(provider: str, model: str, args, sleep=time.sleep, clock=time.perf_counter) -> dict:
    rae.set_thinking(args.judge_thinking)                 # auto starts again at minimal for every candidate
    started = datetime.now().isoformat(timespec="seconds")
    print(f"\n=== {provider}:{model}; thinking requested {args.judge_thinking if provider == 'gemini' else 'n/a'}, "
          f"starts as {rae.effective_thinking(provider, model) if provider == 'gemini' else 'n/a (groq)'}")
    results = run_cases(provider, model, clock=clock, min_gap=args.min_gap, sleep=sleep)
    s, errors = summary(results), error_texts(results)
    eff = rae.effective_thinking(provider, model)
    med = "-" if s["median_latency"] is None else f"{s['median_latency']:.1f} s"
    print(f"\nsuccesses {s['successes']}/{s['calls']}; median latency {med}; both verdicts as expected: "
          f"{'yes' if s['verdicts_as_expected'] else 'no'}; effective thinking "
          f"{eff if eff is not None else 'n/a (groq)'}; meets the rule: {'yes' if s['passes'] else 'no'}")
    print_errors(errors)
    entry = {"candidate": f"{provider}:{model}", "provider": provider, "model": model,
             "thinking_requested": args.judge_thinking if provider == "gemini" else None, "thinking_effective": eff,
             "time": started, "min_gap": args.min_gap, "prompt_sha": rae.JUDGE_PROMPT_SHA, "summary": s,
             "results": results, "errors": errors}
    append_log(args.log, entry)
    return entry


def list_models(provider: str) -> list[str]:
    if provider == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        return sorted(m.name for m in client.models.list() if "generateContent" in (m.supported_actions or []))
    from groq import Groq
    return sorted(m.id for m in Groq().models.list().data)


def main(argv=None, sleep=time.sleep, clock=time.perf_counter):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judge-provider", choices=["groq", "gemini"], default="groq")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    ap.add_argument("--judge-thinking", choices=["minimal", "low", "medium", "high", "default", "auto"], default="auto")
    ap.add_argument("--candidates", help='"provider:model,provider:model,..." checked in this order')
    ap.add_argument("--min-gap", type=float, default=4, metavar="SECONDS", help="sleep between judge calls")
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--list", action="store_true", help="print the model names the key can use")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rae.set_thinking(args.judge_thinking)
    import config  # noqa: F401  (load_dotenv: GOOGLE_API_KEY / GROQ_API_KEY for the calls)
    if args.list:
        for name in list_models(args.judge_provider):
            print(name)
        return
    if args.candidates:
        cands = [tuple(c.strip().split(":", 1)) for c in args.candidates.split(",") if c.strip()]
        bad = [c for c in cands if len(c) != 2 or c[0] not in ("groq", "gemini")]
        if bad:
            sys.exit(f"--candidates: use provider:model with provider groq or gemini ({bad})")
    else:
        cands = [(args.judge_provider, args.judge_model)]
    names = ", ".join(f"{p}:{m or 'NOT SET'}" for p, m in cands)
    print(f"candidates (in order): {names}; prompt sha "
          f"{rae.JUDGE_PROMPT_SHA}; thinking {args.judge_thinking} (Gemini; n/a for groq): starts as "
          f"{[rae.effective_thinking(p, m or '') for p, m in cands]}; {len(CASES)} cases x {REPEATS} repeats; "
          f"article {CID}; --min-gap {args.min_gap:g} s; log {args.log}")
    print(f"question: {ITEM['question']}\nreference points: {ITEM['reference_points']}")
    for name, want, text in CASES:
        print(f"  {name} (expected {want}): {text}")
    print(RULE)
    if args.dry_run:
        print("--dry-run: no call made.")
        return
    if not all(m for _, m in cands):
        sys.exit("No judge model: set JUDGE_MODEL, pass --judge-model, or --candidates.")
    for p, m in cands:
        if check_candidate(p, m, args, sleep=sleep, clock=clock)["summary"]["passes"]:
            print(f"\nSELECTED: {p}:{m} (first candidate meeting the rule)")
            return
    print("\nno candidate meets the rule.")
    sys.exit(1)


if __name__ == "__main__":
    main()
