"""French queries for the answerable items the retrieval diagnostic skipped (Arabic, or detected as English).

    python diagnostics/translate_skipped.py    -> diagnostics/translations.json

One call of the existing translate.translate_to_french (v1.1's step_query for these items) per item, single key
GROQ_API_KEY (config.py). Resumable: items already in translations.json are not translated again. The function
swallows its exceptions and returns the question unchanged; that case is detected (warning captured) and the item is
not saved. A daily-quota 429 (eval/harness_core.py rules) stops the run with exit code 75.
"""
import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(Path(__file__).resolve().parent)]
OUT = ROOT / "diagnostics" / "translations.json"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def todo_items() -> list:
    from retrieval_diagnostic import items_by_set
    from retriever import detect_arabic, detect_english
    out = []
    for s, its in items_by_set().items():
        for it in its:
            if detect_arabic(it["question"]) or detect_english(it["question"]):
                out.append((s, it))
    return out


def main() -> int:
    import harness_core as hc
    hc.install_groq_patches()
    import translate
    from retriever import detect_arabic
    cap = _Capture()
    logging.getLogger(translate.__name__).addHandler(cap)
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"model": None, "items": []}
    have = {(x["set"], x["id"]) for x in done["items"]}
    from config import LLM_MODEL, TRANSLATE_REASONING_EFFORT
    done["model"], done["reasoning_effort"] = LLM_MODEL, TRANSLATE_REASONING_EFFORT
    done["function"] = "translate.translate_to_french (pipeline.step_query for Arabic/English questions)"
    items = todo_items()
    failed = []
    for s, it in items:
        if (s, it["id"]) in have:
            continue
        cap.msgs.clear()
        q = it["question"]
        fr = translate.translate_to_french(q)
        if hc.QUOTA_SEEN["msg"] or any(hc.is_daily_quota(m) for m in cap.msgs):
            OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"daily quota reached at {s} {it['id']}; {len(done['items'])} saved", file=sys.stderr)
            return 75
        if cap.msgs or fr == q:
            failed.append({"set": s, "id": it["id"], "warning": (cap.msgs or ["translation returned the question"])[0][:300]})
            print(f"  {s} {it['id']}: FAILED ({failed[-1]['warning'][:120]})", flush=True)
            continue
        done["items"].append({"set": s, "id": it["id"], "language": it.get("language"),
                              "detected": "arabic" if detect_arabic(q) else "english",
                              "query": q, "french_query": fr})
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {s} {it['id']}: ok", flush=True)
    print(f"{len(done['items'])} of {len(items)} items translated; failed this run: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
