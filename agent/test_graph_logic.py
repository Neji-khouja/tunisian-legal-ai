"""Offline checks of the verify loop in agent_graph (no Groq, no models).

    python agent/test_graph_logic.py
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
    return {"language": lang, "claims": CLAIMS, "caveat": "", "bad": bad, "evidence": EVIDENCE, "full_question": "Q"}


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
r = g.finalize_node(final_state([1, 2, 3], "fr") | {"caveat": "Les articles ne couvrent pas tout."})
check("finalize all bad: caveat cleared", r.get("caveat") == "")

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

# 8b. first_verdicts: set by the first verify, not overwritten by the verify after a repair
from verify import Verification, ClaimVerdict
passes = iter([[("supported", ""), ("unsupported", "x"), ("supported", "")],
               [("supported", ""), ("supported", ""), ("supported", "")]])
g.verify = lambda ans, ev: Verification(verdicts=[ClaimVerdict(claim_index=i, verdict=v, problem=p)
                                                  for i, (v, p) in enumerate(next(passes), 1)])
try:
    s = {"claims": CLAIMS, "caveat": "", "evidence": EVIDENCE, "degraded": [], "repairs": 0}
    out1 = g.verify_node(s)
    s.update(out1)
    s["repairs"] = 1                                     # as repair_node would set
    out2 = g.verify_node(s)
    s.update(out2)                                       # LangGraph merges node returns the same way
finally:
    g.verify = real_verify
check("first verify: first_verdicts == verdicts, bad [2]", out1.get("first_verdicts") == out1["verdicts"] and out1["bad"] == [2])
check("verify after a repair: no first_verdicts key in its return", "first_verdicts" not in out2)
check("after the repair: state first_verdicts still has claim 2 unsupported",
      [v["verdict"] for v in s["first_verdicts"]] == ["supported", "unsupported", "supported"]
      and [v["verdict"] for v in s["verdicts"]] == ["supported"] * 3)

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
         "retries": 2, "tried": ["q"], "degraded": ["x"], "text": "old answer",
         "first_verdicts": [{"claim_index": 1, "verdict": "partly", "problem": "p"}]}
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
    r_clar = g.analyze_node(stale | {"status": "clarify", "question": "Le code pénal.",
                                     "original_question": "User: Quelle est la peine ?\nAgent asked: Quel code ?"})
finally:
    g.analyze = real_analyze
check(f"analyze_node reads status 'clarify' before the reset: full_question {r_clar['full_question']!r}",
      r_clar["full_question"].endswith("\nUser's answer: Le code pénal.") and r_clar["status"] == "")
print("analyze_node(stale) return keys:", sorted(r))
for k in ("evidence", "claims", "verdicts", "bad", "repairs", "dropped", "draft_claims", "draft_text",
          "retries", "tried", "text", "status", "first_verdicts"):
    check(f"analyze_node resets {k} -> {r[k]!r}", r[k] == g.fresh_state()[k])
check("analyze_node does not return original_question or clarify_count",
      "original_question" not in r and "clarify_count" not in r)

# Follow-ups. condense_node with a stubbed structured llm (no Groq)
import condense as cmod
from condense import Standalone
from langchain_core.documents import Document


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = [], result, error

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.result


H1 = [{"q": "Quelle est la peine pour le vol ?", "a": "Cinq ans d'emprisonnement [PenalFr art. 264]."}]
STANDALONE = "Quelle est la peine pour le vol commis par un mineur ?"
real_llm = cmod.structured_llm
cmod.structured_llm = fake = FakeLLM(Standalone(question=STANDALONE, is_followup=True))
try:
    r = g.condense_node({"question": "Et pour les mineurs ?", "status": "done"})          # no history key at all
    check("condense: no history key -> 0 llm calls, question unchanged", not fake.calls and "question" not in r)
    r = g.condense_node({"question": "Et pour les mineurs ?", "status": "done", "history": []})
    check("condense: empty history -> 0 llm calls, question unchanged", not fake.calls and "question" not in r)
    r = g.condense_node({"question": "Le code pénal.", "status": "clarify", "history": H1, "degraded": ["x"]})
    check("condense: previous status clarify -> 0 llm calls, question and degraded untouched",
          not fake.calls and "question" not in r and "degraded" not in r)
    r = g.condense_node({"question": "Et pour les mineurs ?", "status": "done", "history": H1, "degraded": ["old"]})
    human = fake.calls[-1][1][1]
    print("condense human message:\n" + human)
    check("condense: history present -> 1 llm call", len(fake.calls) == 1)
    check(f"condense: question replaced -> {r['question']!r}", r["question"] == STANDALONE)
    check("condense: condensed_from keeps the raw message, is_followup True",
          r["condensed_from"] == "Et pour les mineurs ?" and r["is_followup"] is True)
    check("condense: degraded reset for the new question", r["degraded"] == [])
    check("condense: prompt has the earlier turn and the new message",
          H1[0]["q"] in human and H1[0]["a"] in human and human.endswith("New message: Et pour les mineurs ?"))
    g.condense_node({"question": "x", "status": "done", "history": [{"q": f"q{i}", "a": f"a{i}"} for i in range(3)]})
    human = fake.calls[-1][1][1]
    check("condense: only the last 2 turns are sent", "q0" not in human and "q1" in human and "q2" in human)
    cmod.structured_llm = FakeLLM(error=ValueError("bad json"))
    r = g.condense_node({"question": "Et pour les mineurs ?", "status": "done", "history": H1, "degraded": []})
    print("condense_node with failing llm:", r)
    check("condense failure: raw message kept (no question key)", "question" not in r and r["condensed_from"] == "")
    check("condense failure: degraded has condense_failed: ValueError", r["degraded"] == ["condense_failed: ValueError"])
finally:
    cmod.structured_llm = real_llm
g.analyze = lambda q, hint=None: SimpleNamespace(language="fr", intent="clear", target_codes=["penal"],
                                                 clarification_question=None)
try:
    r = g.analyze_node(stale | {"status": "", "degraded": ["condense_failed: ValueError"], "history": H1})
finally:
    g.analyze = real_analyze
check("analyze_node keeps condense_failed from condense_node", r["degraded"] == ["condense_failed: ValueError"])
check("fresh_state and analyze_node do not reset history", "history" not in g.fresh_state() and "history" not in r)

# history: finalize and the no-evidence answer append, refuse and clarify do not, at most 2 turns
r = g.finalize_node(final_state([], "fr"))                                              # no history key
check("finalize appends {q: full_question, a: text}", r["history"] == [{"q": "Q", "a": r["text"][:600]}])
two = [{"q": "q1", "a": "a1"}, {"q": "q2", "a": "a2"}]
r = g.finalize_node(final_state([1, 2, 3], "fr") | {"history": two})
check("finalize (all dropped) appends and keeps the last 2 turns", [t["q"] for t in r["history"]] == ["q2", "Q"])
r = g.answer_node({"evidence": {}, "language": "en", "full_question": "Q3", "degraded": [], "history": two})
check("answer_node without evidence appends", r["history"] == [two[1], {"q": "Q3", "a": g.NO_ARTICLE["en"]}])
check("answer text cut to 600 chars", len(g.remember({"full_question": "Q", "history": []}, "x" * 1000)[0]["a"]) == 600)
check("refuse and clarify do not touch history", "history" not in g.refuse(stale) and "history" not in g.clarify(stale))

# whole graph, every model stubbed: a follow-up is condensed, history stays at 2, a clarification reply is not
seen = []
real = {k: getattr(g, k) for k in ("analyze", "retrieve_g", "grade", "make_answer", "verify", "bad_claims")}
intents = iter(["clear", "clear", "clear", "ambiguous", "clear"])
g.analyze = lambda q, hint=None: (seen.append(("analyze", q)), SimpleNamespace(
    language="fr", intent=next(intents), target_codes=["penal"], clarification_question="Quel code ?"))[1]
g.retrieve_g = lambda q, codes, info=None: [Document(page_content="...", metadata={"chunk_id": "PenalFr/s02/264",
                                 "source": "PenalFr.pdf", "article": "264", "law_type": "penal"})]
g.grade = lambda q, ev: SimpleNamespace(enough=True, missing="")
g.make_answer = lambda q, lang, ev, feedback="": g.Answer(claims=[g.Claim(text="x", cited_ids=["PenalFr/s02/264"])], caveat="")
g.verify = lambda ans, ev: SimpleNamespace(verdicts=[])
g.bad_claims = lambda ans, v: []
cmod.structured_llm = fake = FakeLLM(Standalone(question=STANDALONE, is_followup=True))
try:
    cfg = {"configurable": {"thread_id": "followup-offline"}}
    states = [g.app.invoke({"question": q}, cfg) for q in
              ["Quelle est la peine pour le vol ?", "Et pour les mineurs ?", "Et la tentative ?",
               "Quelle est la peine ?", "Le code pénal."]]
finally:
    cmod.structured_llm = real_llm
    for k, v in real.items():
        setattr(g, k, v)
print("graph run, Analyze inputs:", [q for _, q in seen])
check("graph turn 1: empty history -> 0 condense calls", states[0]["condensed_from"] == "")
check("graph turn 2: Analyze gets the standalone question", seen[1] == ("analyze", STANDALONE)
      and states[1]["condensed_from"] == "Et pour les mineurs ?")
check("graph: history never exceeds 2 turns", [len(s["history"]) for s in states] == [1, 2, 2, 2, 2])
check("graph turn 4 clarify: history not appended", states[3]["status"] == "clarify"
      and states[3]["history"] == states[2]["history"])
check("graph turn 5 (reply to the clarification): no condense call, merged by analyze_node",
      len(fake.calls) == 3 and seen[4][1].endswith("User's answer: Le code pénal."))

# the evaluation harness starts a new thread for every item, so history is empty and condense makes no call
line = next(l for l in (Path(__file__).resolve().parent.parent / "eval" / "run_answer_eval.py")
            .read_text(encoding="utf-8").splitlines() if '"thread_id":' in l)
print("eval/run_answer_eval.py:", line.strip())
check("run_answer_eval: thread_id per item (id + time_ns)", "item['id']" in line and "time_ns()" in line)

# targeted repair (config.REPAIR_MODE)
import config as C
from answer import Answer, Claim
real_make = g.make_answer
N = lambda text, cid="PenalFr/s02/221": {"text": text, "cited_ids": [cid]}
m, rc = g.targeted(CLAIMS, {1, 3}, [N("new one"), N("new three"), N("too many")])
check("targeted: replacements take the rejected places, in order, no more than rejected",
      [c["text"] for c in m] == ["new one", "Claim two.", "new three"] and rc == [1, 3])
m, rc = g.targeted(CLAIMS, {1, 3}, [N("claim  TWO."), N("new")])
check("targeted: a replacement repeating a confirmed claim is left out", [c["text"] for c in m] == ["new", "Claim two."] and rc == [1])


def repair_run(verdicts, replacements, repair_mode="targeted"):
    """draft CLAIMS -> verify -> repair -> verify -> finalize with stubs. verdicts: one list per verify call."""
    calls, checks = [], []

    def make(q, lang, ev, feedback="", keep=()):
        calls.append({"feedback": feedback, "keep": list(keep)})
        return Answer(claims=[Claim(**c) for c in replacements], caveat="")

    def ver(ans, ev):
        checks.append([c.text for c in ans.claims])
        vs = verdicts[len(checks) - 1]
        return Verification(verdicts=[ClaimVerdict(claim_index=i, verdict=v, problem="" if v == "supported" else "p")
                                      for i, v in enumerate(vs, 1)])
    old_mode, C.REPAIR_MODE = C.REPAIR_MODE, repair_mode
    g.make_answer, g.verify = make, ver
    try:
        st = g._fresh() | {"claims": CLAIMS, "draft_claims": CLAIMS, "caveat": "c0", "evidence": EVIDENCE, "degraded": [],
                           "language": "fr", "full_question": "Q", "repairs": 0}
        st |= g.verify_node(st)
        if g.route_after_verify(st) == "repair":
            st |= g.repair_node(st)
            if g.route_after_repair(st) == "verify":
                st |= g.verify_node(st)
        st |= g.finalize_node(st)
    finally:
        C.REPAIR_MODE, g.make_answer, g.verify = old_mode, real_make, real_verify
    return st, calls, checks


S, U = "supported", "unsupported"
check("default repair mode is targeted", C.REPAIR_MODE == "targeted")
st, calls, checks = repair_run([[S, U, S], [S]], [N("Claim two, fixed.")])
check("targeted: the model gets the rejected claim's problem and the confirmed claims",
      "Claim two." in calls[0]["feedback"] and "Claim one." not in calls[0]["feedback"] and calls[0]["keep"] == ["Claim one.", "Claim three."])
check("targeted: the second check sees only the replacement", checks == [["Claim one.", "Claim two.", "Claim three."], ["Claim two, fixed."]])
check("targeted: order kept, confirmed sentences untouched, caveat kept, nothing dropped",
      [c["text"] for c in st["claims"]] == ["Claim one.", "Claim two, fixed.", "Claim three."] and st["dropped"] == 0 and st["caveat"] == "c0")
check("targeted: verdicts numbered over the whole answer, first verdicts kept",
      [v["claim_index"] for v in st["verdicts"]] == [1, 2, 3] and [v["verdict"] for v in st["first_verdicts"]] == [S, U, S])
st, calls, checks = repair_run([[S, U, S], [U]], [N("still wrong")])
check("targeted: a replacement rejected again is dropped, the confirmed sentences stay",
      [c["text"] for c in st["claims"]] == ["Claim one.", "Claim three."] and st["dropped"] == 1)
st, calls, checks = repair_run([[U, S, U], [S, U]], [N("one, fixed"), N("three, wrong")])
check("targeted: two rejected, one fixed and one dropped", [c["text"] for c in st["claims"]] == ["one, fixed", "Claim two."] and st["dropped"] == 1)
st, calls, checks = repair_run([[S, U, S]], [N("Claim one.")])
check("targeted: nothing usable from the repair: no second check, the rejected sentence is dropped and counted",
      len(checks) == 1 and [c["text"] for c in st["claims"]] == ["Claim one.", "Claim three."] and st["dropped"] == 1
      and st["text"].endswith(g.DROPPED["fr"].format(n=1)))
st, calls, checks = repair_run([[U, U, U], [S]], [N("whole new answer")])
check("every sentence rejected: a whole new answer, checked in full",
      calls[0]["keep"] == [] and checks[1] == ["whole new answer"] and [c["text"] for c in st["claims"]] == ["whole new answer"] and st["dropped"] == 0)
st, calls, checks = repair_run([[S, U, S], [S, S]], [N("a"), N("b")], "full")
check("REPAIR_MODE full: the whole answer is rewritten and re-checked (v2)",
      calls[0]["keep"] == [] and checks[1] == ["a", "b"] and [c["text"] for c in st["claims"]] == ["a", "b"] and "recheck" not in st)
prompts = []
sentinel = type("L", (), {"with_structured_output": lambda self, *a, **k: self,
                          "invoke": lambda self, ms: (prompts.append(ms), Answer(claims=[Claim(text="x", cited_ids=[])]))[1]})()
import answer as A
keep_llm, A.llm = A.llm, sentinel
try:
    A.make_answer("Q", "fr", EVIDENCE)
    A.make_answer("Q", "fr", EVIDENCE, feedback="f", keep=["k"])
finally:
    A.llm = keep_llm
check("the repair call starts with exactly the first call's prompt (reusable by the prompt cache)",
      prompts[1][0] == prompts[0][0] and prompts[1][1][1].startswith(prompts[0][1][1]) and "Write ONLY their replacements" in prompts[1][1][1])

# answer style: the prompt sent follows config.ANSWER_STYLE, the evaluated prompts are unchanged
import hashlib
import config as C
import answer as A
import pipeline
sent = []


class FakeLLM:
    def with_structured_output(self, *a, **k):
        return self

    def invoke(self, messages):
        sent.append(messages)
        return A.Answer(claims=[A.Claim(text="x", cited_ids=["[PenalFr/s02/218]"])])


style, real_llm, A.llm = C.ANSWER_STYLE, A.llm, FakeLLM()
try:
    check("default answer style is natural", style == "natural")
    for value, want in (("natural", A.NATURAL_PROMPT), ("v2", A.SYSTEM_PROMPT)):
        C.ANSWER_STYLE = value
        a = A.make_answer("Q", "fr", EVIDENCE)
        check(f"ANSWER_STYLE {value}: its prompt is sent, ids cleaned", sent[-1][0][1] == want and a.claims[0].cited_ids == ["PenalFr/s02/218"])
    for value, marker in (("natural", "HOW TO WRITE"), ("v2", "REQUIRED RESPONSE STRUCTURE")):
        C.ANSWER_STYLE = value
        p = pipeline._NATURAL_PROMPT if value == "natural" else pipeline._PROMPT
        check(f"one-pass prompt {value}", marker in p.format(context="c", input="q"))
finally:
    C.ANSWER_STYLE, A.llm = style, real_llm
check("evaluated answer prompt (v2) unchanged", hashlib.sha256(A.SYSTEM_PROMPT.encode("utf-8")).hexdigest().startswith("5dbea27a8fcdf8c6"))
check("evaluated one-pass prompt (v1.1) unchanged", hashlib.sha256(pipeline._SYSTEM_PROMPT.encode("utf-8")).hexdigest().startswith("d2acda8ce54721ef"))
print("all checks passed")
