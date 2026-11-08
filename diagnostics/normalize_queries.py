"""Wave 2 STEP 2: one normalization call (agent/normalize.py) per development item.

    python diagnostics/normalize_queries.py            -> diagnostics/normalized_queries.json
    python diagnostics/normalize_queries.py --check    # the no-LLM checks on the saved file only

Input query = the French query of eval/retrieval_only.py (the question for French items, the saved translation of
diagnostics/translations.json for Arabic/English items). Single key GROQ_API_KEY (config.py), temperature 0, at most
2 attempts per call, no client retries. Resume-safe: an item already saved (whatever its status) is not called
again. A daily-quota 429 (eval/harness_core.py rules) saves and stops with exit code 75. Checks (no LLM): a number
in legal_query that is not in the input query is flagged; an empty or invalid output (status != ok) is a failure:
the item falls back to the input query (diagnostics/wave2_core.query_for).
"""
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "agent"), str(Path(__file__).resolve().parent)]
OUT = ROOT / "diagnostics" / "normalized_queries.json"
ITEMS = ROOT / "diagnostics" / "dev_items.json"
TRANSLATIONS = ROOT / "diagnostics" / "translations.json"
MIN_INTERVAL_S = 7.0            # pacing under the per-minute token limit (about 8 calls a minute)


def input_queries(items_path=ITEMS, translations_path=TRANSLATIONS) -> list:
    """[(item, French input query)] in item order; an item without one raises (all 110 have one)."""
    import retrieval_only as ro
    items, tr = ro.load_items(items_path), ro.load_translations(translations_path)
    out = []
    for it in items:
        fq, _, why = ro.french_query(it, tr)
        if fq is None:
            raise SystemExit(f"{it['id']}: {why}")
        out.append((it, fq))
    return out


def flags(entry) -> dict:
    import normalize as N
    f = {"failure": entry.get("status") != "ok"}
    if entry.get("legal_query"):
        f["new_numbers"] = N.new_numbers(entry["input_query"], entry["legal_query"])
        f["over_40_words"] = len(entry["legal_query"].split()) > N.MAX_WORDS
    return f


def load(path=OUT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {"items": []}


def save(done, path=OUT):
    Path(path).write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")


def totals(done) -> dict:
    es = done["items"]
    tok = {k: sum(e["tokens"][k] for e in es) for k in ("input_tokens", "output_tokens", "total_tokens")}
    return {"items": len(es), "ok": sum(e["status"] == "ok" for e in es),
            "failures": [e["id"] for e in es if e["status"] != "ok"],
            "new_numbers": {e["id"]: e["flags"]["new_numbers"] for e in es if e["flags"].get("new_numbers")},
            "over_40_words": [e["id"] for e in es if e["flags"].get("over_40_words")],
            "attempts_2": [e["id"] for e in es if e["attempts"] > 1], "tokens": tok,
            "prompt_sha256": sorted({e["prompt_sha256"] for e in es})}


def run(pairs, done, call, fatal, path=OUT, sleep=time.sleep, clock=time.monotonic, interval=MIN_INTERVAL_S) -> int:
    """Calls call(query) for every item not yet in done; saves after each. -> 0, or 75 on a daily quota."""
    import normalize as N
    have = {e["id"] for e in done["items"]}
    last = None
    for it, fq in pairs:
        if it["id"] in have:
            continue
        if last is not None and clock() - last < interval:
            sleep(interval - (clock() - last))
        last = clock()
        try:
            r = call(fq)
        except Exception as e:
            if fatal(e):
                save(done, path)
                print(f"daily quota reached at {it['id']}; {len(done['items'])} saved", file=sys.stderr)
                return 75
            raise
        e = {"id": it["id"], "set": it.get("set"), "language": it.get("language"), "input_query": fq,
             "legal_query": r["legal_query"], "status": r["status"], "error": r["error"], "attempts": r["attempts"],
             "prompt_sha256": N.PROMPT_SHA256, "model": N.MODEL, "temperature": N.TEMPERATURE,
             "reasoning_effort": N.REASONING_EFFORT, "tokens": r["tokens"],
             "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if r["status"] != "ok":
            e["raw"] = (r.get("raw") or "")[:500]
        e["flags"] = flags(e)
        done["items"].append(e)
        save(done, path)
        print(f"  {it['id']}: {r['status']}" + (f" ({r['error'][:100]})" if r["error"] else "")
              + (f"  NEW NUMBERS {e['flags']['new_numbers']}" if e["flags"].get("new_numbers") else ""), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    done = load()
    if not a.check:
        import harness_core as hc
        hc.install_groq_patches()
        import normalize as N
        fatal = lambda e: bool(hc.QUOTA_SEEN["msg"]) or hc.is_daily_quota(hc.full_message(e))
        code = run(input_queries(), done, lambda q: N.normalize(q, fatal=fatal), fatal)
        if code:
            return code
    for e in done["items"]:
        e["flags"] = flags(e)
    save(done)
    print(json.dumps(totals(done), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
