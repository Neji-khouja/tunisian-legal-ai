"""Offline checks of the verify loop in agent_graph (no Groq, no models).

    python agent/test_graph_logic.py

Importing agent_graph builds the graph but loads no model and calls no network (models load lazily).
"""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_graph as g

EVIDENCE = {cid: {"source": "PenalFr.pdf", "article": cid.split("/")[-1], "law_type": "penal", "text": "..."}
            for cid in ["PenalFr/s02/218", "PenalFr/s02/219", "PenalFr/s02/220", "PenalFr/s02/221"]}
CLAIMS = [{"text": "Claim one.", "cited_ids": ["PenalFr/s02/218"]},
          {"text": "Claim two.", "cited_ids": ["PenalFr/s02/219"]},
          {"text": "Claim three.", "cited_ids": ["PenalFr/s02/220", "PenalFr/s02/218"]}]


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    assert cond, name


def final_state(bad, lang):
    return {"language": lang, "claims": CLAIMS, "caveat": "", "bad": bad, "evidence": EVIDENCE}


# 1-2. routers
check("route_after_verify bad [2] repairs 0 -> repair", g.route_after_verify({"bad": [2], "repairs": 0}) == "repair")
check("route_after_verify bad [2] repairs 1 -> finalize", g.route_after_verify({"bad": [2], "repairs": 1}) == "finalize")
check("route_after_verify bad [] repairs 0 -> finalize", g.route_after_verify({"bad": [], "repairs": 0}) == "finalize")
check("route_after_answer no claims -> end", g.route_after_answer({"claims": []}) == "end")
check("route_after_answer one claim -> verify", g.route_after_answer({"claims": CLAIMS[:1]}) == "verify")

# 3-4. one claim dropped
for lang in ("fr", "en", "ar"):
    r = g.finalize_node(final_state([2], lang))
    note = g.DROPPED[lang].format(n=1)
    check(f"finalize {lang} bad [2]: 2 claims kept", [c["text"] for c in r["claims"]] == ["Claim one.", "Claim three."])
    check(f"finalize {lang} bad [2]: text ends with note {note!r}", r["text"].endswith(note))
    check(f"finalize {lang} bad [2]: dropped == 1", r["dropped"] == 1)
    check(f"finalize {lang} bad [2]: cited only kept claims {r['cited']}",
          r["cited"] == ["PenalFr/s02/218", "PenalFr/s02/220"])
    check(f"finalize {lang} bad [2]: status done", r["status"] == "done")

# 5. every claim dropped
for lang in ("fr", "en", "ar"):
    r = g.finalize_node(final_state([1, 2, 3], lang))
    expected = g.NO_SUPPORT[lang] + " [PenalFr art. 218, PenalFr art. 219, PenalFr art. 220]"
    check(f"finalize {lang} all bad: text {r['text']!r}", r["text"] == expected)
    check(f"finalize {lang} all bad: claims [] and dropped 3", r["claims"] == [] and r["dropped"] == 3)

# 6. nothing dropped
r = g.finalize_node(final_state([], "fr"))
check("finalize bad []: 3 claims kept", len(r["claims"]) == 3)
check("finalize bad []: no note", g.DROPPED["fr"].format(n=0) not in r["text"] and "retirée" not in r["text"])
check("finalize bad []: dropped 0", r["dropped"] == 0)

# 7. feedback_from
fb = g.feedback_from({"claims": CLAIMS, "verdicts": [
    {"claim_index": 1, "verdict": "supported", "problem": ""},
    {"claim_index": 2, "verdict": "partly", "problem": "x"},
    {"claim_index": 3, "verdict": "unsupported", "problem": "y"},
    {"claim_index": 9, "verdict": "unsupported", "problem": "z"}]})
print("feedback_from text:\n" + fb)
lines = fb.splitlines()
check("feedback_from: exactly 2 lines", len(lines) == 2)
check("feedback_from: line 1 = claim 2, partly, x", all(s in lines[0] for s in ("Claim two.", "partly", "x")))
check("feedback_from: line 2 = claim 3, unsupported, y", all(s in lines[1] for s in ("Claim three.", "unsupported", "y")))
check("feedback_from: index 9 ignored", "z" not in fb)

# 8. verify_node falls back visibly
def boom(*a, **k):
    raise RuntimeError("down")
real_verify, g.verify = g.verify, boom
try:
    r = g.verify_node({"claims": CLAIMS, "caveat": "", "evidence": EVIDENCE, "degraded": []})
finally:
    g.verify = real_verify
print("verify_node with failing verify:", r)
check("verify_node failure: bad [] and verdicts []", r["bad"] == [] and r["verdicts"] == [])
check("verify_node failure: degraded has verify_failed: RuntimeError", "verify_failed: RuntimeError" in r["degraded"])

# 9. fresh_state shares nothing
a, b = g.fresh_state(), g.fresh_state()
a["claims"].append("x"); a["evidence"]["k"] = 1; a["tried"].append("q")
check("fresh_state: two calls share no list/dict", b["claims"] == [] and b["evidence"] == {} and b["tried"] == [])
check("fresh_state: no object identity shared",
      all(a[k] is not b[k] for k in a if isinstance(a[k], (list, dict))))

# Task 3. stale state from a finished answer
stale = {"question": "Quelle est la peine ?", "language": "fr", "status": "done", "intent": "clear",
         "clarification_question": "Quel code ?", "original_question": None, "clarify_count": 0,
         "claims": CLAIMS, "caveat": "c", "evidence": EVIDENCE, "verdicts": [{"claim_index": 1}],
         "bad": [2], "repairs": 1, "dropped": 2, "draft_claims": CLAIMS, "draft_text": "old",
         "retries": 2, "tried": ["q"], "degraded": ["x"]}
r = g.refuse(stale)
print("refuse(stale) returns:", r)
check("refuse returns only status/text/clarify_count/original_question",
      set(r) == {"status", "text", "clarify_count", "original_question"})
r = g.clarify(stale)
print("clarify(stale) returns:", r)
check("clarify returns only status/text/original_question/clarify_count",
      set(r) == {"status", "text", "original_question", "clarify_count"})
r = g.clarify(stale | {"clarification_question": None, "language": "ar"})
check(f"clarify fallback in the language: {r['text']!r}", r["text"] == g.CLARIFY_FALLBACK["ar"])

real_analyze = g.analyze
g.analyze = lambda q, hint=None: SimpleNamespace(language="fr", intent="clear", target_codes=["penal"],
                                                 clarification_question=None)
try:
    r = g.analyze_node(stale)
finally:
    g.analyze = real_analyze
print("analyze_node(stale) return keys:", sorted(r))
for k in ("evidence", "claims", "verdicts", "bad", "repairs", "dropped", "draft_claims", "draft_text",
          "retries", "tried"):
    check(f"analyze_node resets {k} -> {r[k]!r}", r[k] == g.fresh_state()[k])
check("analyze_node does not return original_question or clarify_count",
      "original_question" not in r and "clarify_count" not in r)
print("all checks passed")
