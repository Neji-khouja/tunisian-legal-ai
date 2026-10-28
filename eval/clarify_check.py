"""Clarification check: one ambiguous question, then the user's reply on the same thread (calls Groq, loads the models).

    python eval/clarify_check.py        -> prints the report, saves results/clarify_check.json

Turn 1 = the Arabic dev question q055 "ما هي مدة الاحتفاظ؟". The graph is stopped right after Analyze; if Analyze
does not say "ambiguous", the cases of agent/analyze_tests.json that expect "ambiguous" are tried in file order
(one Analyze call each) and the first one Analyze classifies "ambiguous" is used, with its own reply.
Turn 2 = the reply, same thread_id. Nothing in the agent is changed: the query strings given to retrieve_g and to
the reranker are recorded by wrapping those functions in this process only.
Connection errors, timeouts and 429 that reach this script are retried 3 times (waits 10/30/60 s); a run that was
interrupted mid-graph is resumed from its checkpoint, not restarted.
"""
import sys
import json
import time
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
import agent_graph
import retrieve as retrieve_mod
import retrieve_g as retrieve_g_mod
from agent_graph import app

OUT = ROOT / "results" / "clarify_check.json"
FIRST = ("q055", "ما هي مدة الاحتفاظ؟", "أقصد الاحتفاظ بالمشتبه فيه لدى الضابطة العدلية")
REPLIES = {                                      # replies for the fallback questions of agent/analyze_tests.json
    "Can I break my contract?": "It is my employment contract, I want to resign",
    "Quel est le délai de préavis ?": "Je parle du préavis de démission d'un salarié en contrat à durée indéterminée.",
}
WAITS = (10, 30, 60)
HTTP = {"200": 0, "429": 0, "other": 0}
LOG = []                                          # recorded calls: (turn, function, args)
TURN = {"n": 0}


class _CountGroq(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "api.groq.com" in msg:
            HTTP["200" if " 200 " in msg else "429" if " 429 " in msg else "other"] += 1


logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpx").addHandler(_CountGroq())
logging.getLogger("httpx").propagate = False


def _show(a):
    if isinstance(a, str) or (isinstance(a, list) and all(isinstance(x, str) for x in a)):
        return a
    return f"<{len(a)} documents>" if isinstance(a, list) else repr(a)


def _wrap(module, name, label):
    original = getattr(module, name)

    def wrapper(*args, **kw):
        LOG.append({"turn": TURN["n"], "function": label, "args": [_show(a) for a in args]})
        return original(*args, **kw)
    setattr(module, name, wrapper)


_wrap(agent_graph, "retrieve_g", "retrieve_g(question, target_codes)")
_wrap(retrieve_g_mod, "step_query", "retrieve_g: step_query(question)")
_wrap(retrieve_g_mod, "step_rerank", "retrieve_g: step_rerank(french_query, candidates)")
_wrap(retrieve_mod, "step_rerank", "retrieve_for_plan: step_rerank(text, candidates)")


def transient(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}"
    return any(k in s for k in ("RateLimit", "429", "APIConnection", "Connect", "Timeout", "RemoteProtocol"))


def invoke(inp, cfg, **kw):
    """app.invoke with 3 retries on transient errors; after a failure the run resumes from its checkpoint."""
    for attempt in range(len(WAITS) + 1):
        try:
            pending = attempt > 0 and app.get_state(cfg).next
            return app.invoke(None if pending else inp, cfg, **kw)
        except Exception as e:
            if not transient(e) or attempt == len(WAITS):
                raise
            print(f"  transient error ({type(e).__name__}: {str(e)[:120]}); retry in {WAITS[attempt]} s", flush=True)
            time.sleep(WAITS[attempt])


def analyze_only(question: str, thread: str) -> dict:
    """Turn 1 up to Analyze (interrupt after it). Retries when Analyze itself failed (the node catches the error)."""
    for attempt in range(len(WAITS) + 1):
        cfg = {"configurable": {"thread_id": f"{thread}-{attempt}"}}
        s = invoke({"question": question}, cfg, interrupt_after=["analyze"])
        failed = [d for d in s.get("degraded") or [] if d.startswith("analyze_failed")]
        if not failed:
            return {"cfg": cfg, "state": s}
        if attempt == len(WAITS):
            raise RuntimeError(f"Analyze failed {len(WAITS) + 1} times: {failed}")
        print(f"  Analyze failed ({failed}); retry in {WAITS[attempt]} s", flush=True)
        time.sleep(WAITS[attempt])


def main():
    t0 = time.time()
    TURN["n"] = 1
    tried, chosen = [], None
    cands = [FIRST] + [(None, c["question"], REPLIES.get(c["question"]))
                       for c in json.loads((ROOT / "agent" / "analyze_tests.json").read_text(encoding="utf-8"))
                       if c["expected_intent"] == "ambiguous"]
    for qid, question, reply in cands:
        r = analyze_only(question, f"clarify-{len(tried)}")
        intent = r["state"].get("intent")
        tried.append({"id": qid, "question": question, "intent": intent})
        print(f"Turn 1 Analyze: {question!r} -> intent {intent!r}")
        if intent == "ambiguous":
            chosen = (qid, question, reply, r["cfg"])
            break
    if not chosen:
        print("No candidate is classified 'ambiguous' by Analyze now; stopping.")
        OUT.write_text(json.dumps({"tried": tried, "http": HTTP}, ensure_ascii=False, indent=1), encoding="utf-8")
        sys.exit(1)
    qid, question, reply, cfg = chosen
    if qid != FIRST[0]:
        print(f"q055 not flagged ambiguous; using analyze_tests.json case {question!r} with reply {reply!r}")
    s1 = invoke(None, cfg)                                  # resume: route_after_analyze -> clarify -> END
    print(f"Turn 1 final: status={s1.get('status')!r} clarify_count={s1.get('clarify_count')}"
          f"\n  text (clarification question): {s1.get('text')!r}\n  original_question stored: {s1.get('original_question')!r}")
    http1 = dict(HTTP)

    TURN["n"] = 2
    s = invoke({"question": reply}, cfg)
    http2 = {k: HTTP[k] - http1[k] for k in HTTP}
    ev = list(s.get("evidence") or {})
    calls2 = [c for c in LOG if c["turn"] == 2]
    print(f"\nTurn 2 reply: {reply!r}   (same thread_id {cfg['configurable']['thread_id']!r})")
    print(f"  status: {s.get('status')!r}")
    print(f"  full_question: {s.get('full_question')!r}")
    print(f"  intent: {s.get('intent')!r}")
    print(f"  retries: {s.get('retries')!r}")
    print(f"  tried: {s.get('tried')!r}")
    print(f"  first 6 evidence ids: {ev[:6]}")
    print(f"  degraded: {s.get('degraded')!r}")
    print(f"  text:\n{s.get('text')}")
    print("\n  calls recorded on turn 2 (exact strings):")
    for c in calls2:
        print(f"    {c['function']}: {c['args'][0]!r}" + (f"   target_codes={c['args'][1]!r}"
                                                         if c["function"].startswith("retrieve_g(") else ""))
    if qid:
        gold = next((q for q in json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
                     if q.get("id") == qid), None)
        if gold:
            gids = [c for g in gold.get("gold", []) for c in g.get("chunk_ids", [])]
            print(f"\n  {qid} gold chunk ids (eval/questions.json): {gids}; in the 6 evidence ids: "
                  f"{sorted(set(gids) & set(ev[:6])) or 'none'}")
    print(f"\nGroq HTTP responses: turn 1 {http1}, turn 2 {http2}; total {HTTP}; {time.time() - t0:.1f} s")
    OUT.write_text(json.dumps({"tried_turn1": tried, "question": question, "reply": reply,
                               "turn1": {"status": s1.get("status"), "text": s1.get("text"),
                                         "original_question": s1.get("original_question")},
                               "turn2": {k: s.get(k) for k in ("status", "full_question", "intent", "target_codes",
                                                               "retries", "tried", "degraded", "text", "cited",
                                                               "claims", "caveat")} | {"evidence_ids": ev},
                               "calls_turn2": calls2, "http": {"turn1": http1, "turn2": http2},
                               "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
