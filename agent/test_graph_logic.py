"""Offline checks of the verify loop in agent_graph (no Groq, no models).

    python agent/test_graph_logic.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
os.environ["GROQ_API_KEY"] = "offline-test"          # set before config's load_dotenv: no test can reach the provider
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_graph as g
import config
LOW_EFFORT_STEPS, config.LOW_EFFORT_STEPS = config.LOW_EFFORT_STEPS, ()     # the fakes below replace the steps' default models
ANSWER_FIRST, config.ANSWER_FIRST = config.ANSWER_FIRST, False     # the first tests are those of the v2 flow (grade step)

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
g.verify = lambda ans, ev, *q: Verification(verdicts=[ClaimVerdict(claim_index=i, verdict=v, problem=p)
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

    def ver(ans, ev, *q):
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

# the verifier's articles (config.VERIFY_TEXT_ONCE)
import verify as V
ans = Answer(claims=[Claim(text="c1", cited_ids=["A/s1/1", "A/s1/2"]), Claim(text="c2", cited_ids=["A/s1/1", "Z/s1/9"])])
ev = {"A/s1/1": {"text": "TEXTE UN"}, "A/s1/2": {"text": "TEXTE DEUX"}}
check("default: the text is repeated under each claim", C.VERIFY_TEXT_ONCE is False)
twice = V._claims_block(ans, ev)
C.VERIFY_TEXT_ONCE = True
once = V._claims_block(ans, ev)
C.VERIFY_TEXT_ONCE = False
check("verify: an article's text is sent once and referred to afterwards; every citation still listed",
      once.count("TEXTE UN") == 1 and "(text given above, under claim 1)" in once and once.count("Cited [") == 4 and "(not in evidence)" in once)
check("verify: False gives the v2 block (text repeated under each claim)", twice.count("TEXTE UN") == 2 and "given above" not in twice)

# an answer taken from the interface's memory (ui/backend.py remember): the next question sees it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))
from backend import Backend
cfg = {"configurable": {"thread_id": "remember-test"}}
Backend.remember(SimpleNamespace(graph=g.app), "remember-test", {"full_question": "Quelle peine pour le vol ?", "text": "Cinq ans.", "language": "fr"})
vals = g.app.get_state(cfg).values
check("remember: the answer is in the conversation's history and nothing is pending",
      vals["history"] == [{"q": "Quelle peine pour le vol ?", "a": "Cinq ans."}] and g.app.get_state(cfg).next == () and vals["status"] == "done")
asked = []
real_condense, g.condense = g.condense, lambda history, q: (asked.append(history), SimpleNamespace(question=q + " (vol)", is_followup=True))[1]
real_an, g.analyze = g.analyze, lambda q, hint=None: SimpleNamespace(language="fr", intent="out_of_scope", target_codes=[], clarification_question=None)
try:
    out = g.app.invoke({"question": "et la tentative ?"}, cfg)
finally:
    g.condense, g.analyze = real_condense, real_an
check("remember: a follow-up is condensed with that answer", asked and asked[0][0]["a"] == "Cinq ans." and out["question"] == "et la tentative ? (vol)")
Backend.remember(SimpleNamespace(graph=None), "x", {"mode": "fast"})

# the verifier's question (config.VERIFY_QUESTION)
got = []
vllm, V.llm = V.llm, type("L", (), {"with_structured_output": lambda self, *a, **k: self,
                                    "invoke": lambda self, ms: (got.append(ms), Verification(verdicts=[]))[1]})()
try:
    V.verify(ans, ev)
    V.verify(ans, ev, "Quelle peine ?")
finally:
    V.llm = vllm
check("verify without a question: the evaluated prompt and block", got[0][0][1] == V.SYSTEM_PROMPT and got[0][1][1] == V._claims_block(ans, ev))
check("verify with the question: it is shown, with its own prompt",
      got[1][0][1] == V.QUESTION_PROMPT and got[1][1][1].startswith("Question: Quelle peine ?") and C.VERIFY_QUESTION)
seen_q = []
g.verify = lambda a_, e_, *q: (seen_q.append(q), Verification(verdicts=[]))[1]
try:
    base = {"claims": CLAIMS, "caveat": "", "evidence": EVIDENCE, "degraded": [], "repairs": 0, "full_question": "Q ?"}
    g.verify_node(base)
    C.VERIFY_QUESTION = False
    g.verify_node(base)
finally:
    g.verify, C.VERIFY_QUESTION = real_verify, True
check("verify_node passes the question only when the switch is on", seen_q == [("Q ?",), ()])

# less reasoning in some steps (config.LOW_EFFORT_STEPS)
import analyze as AN
import grade as GR
check("low effort: condense and reformulate by default; the deciding steps keep the model's default",
      set(LOW_EFFORT_STEPS) == {"condense", "reformulate"} and AN.llm_low.reasoning_effort == "low"
      and AN.llm_low.model_name == AN.llm.model_name and AN.llm_low.temperature == AN.llm.temperature)
check("LOW_EFFORT_STEPS () gives the v2 models", AN.for_step("grade", GR.structured_llm, GR.Grade) is GR.structured_llm)
C.LOW_EFFORT_STEPS = ("grade",)
check("a listed step gets its own model, another does not",
      AN.for_step("grade", GR.structured_llm, GR.Grade) is not GR.structured_llm and AN.for_step("analyze", AN.structured_llm, AN.Analysis) is AN.structured_llm)
C.LOW_EFFORT_STEPS = ()

# answer first (config.ANSWER_FIRST): the whole graph with stubs
import answer as ANS
C.ANSWER_FIRST = True
check("answer first is the default, one extra search", ANSWER_FIRST is True and C.MAX_EXTRA_SEARCHES == 1)
for st_name in ("natural", "v2"):
    C.ANSWER_STYLE, keep_style = st_name, C.ANSWER_STYLE
    pr = ANS.judged_prompt()
    C.ANSWER_STYLE = keep_style
    check(f"judged prompt ({st_name}): rule 6 replaced, the judgement added, the rest unchanged",
          ANS.NOT_FOUND_RULE in pr and "describing what the most related article" not in pr and pr.endswith(ANS.JUDGE_PROMPT) and "1. Use ONLY" in pr)
check("the judgement is written before the claims", list(ANS.Judged.model_fields)[0] == "fit" and list(ANS.Judged.model_fields).index("claims") > 2)
J = ANS.Judged
CL = [Claim(text="Règle.", cited_ids=["PenalFr/s02/218"])]
OK_J, NO_J = J(fit="answers", claims=CL), J(fit="not_found", missing="sanction du vol")
ASK_J = J(fit="unclear", clarification_question="Quel contrat ?")
REAL = {k: getattr(g, k) for k in ("analyze", "retrieve_g", "retrieve_for_plan", "judge_answer", "make_answer", "reformulate", "verify", "to_evidence", "condense", "grade")}


def run_first(judgements, turns=("Q ?",), evidence=EVIDENCE, intent="clear"):
    """Runs the compiled graph; returns the states and what each step received."""
    log, it = [], iter(judgements)
    g.analyze = lambda q, hint=None: SimpleNamespace(language="fr", intent=intent, target_codes=[], clarification_question="Quoi ?")
    g.retrieve_g = lambda q, codes, info=None: (log.append(("search", q)), ["d1"])[1]
    g.retrieve_for_plan = lambda subs, k, f, soft=True: (log.append(("more", subs)), ["d2"])[1]
    g._fuse, keep_fuse = (lambda lists: [d for l in lists for d in l]), g._fuse
    g.to_evidence = lambda docs: dict(evidence)
    g.judge_answer = lambda q, lang, ev: (log.append(("judge", q)), next(it))[1]
    g.make_answer = lambda q, lang, ev, feedback="", keep=(): (log.append(("forced", q)), Answer(claims=CL))[1]
    g.reformulate = lambda q, lang, tried, ev, missing: (log.append(("reformulate", missing)), SimpleNamespace(sub_queries=[SimpleNamespace(text="x", code=None)]))[1]
    g.verify = lambda a_, e_, *q: Verification(verdicts=[ClaimVerdict(claim_index=i, verdict="supported") for i in range(1, len(a_.claims) + 1)])
    g.condense = lambda history, q: SimpleNamespace(question=q, is_followup=False)
    g.grade = lambda *a: (_ for _ in ()).throw(AssertionError("grade called"))
    out, cfg = [], {"configurable": {"thread_id": f"af-{len(log)}-{id(log)}"}}
    try:
        for t in turns:
            out.append(dict(g.app.invoke({"question": t}, cfg)))
    finally:
        g._fuse = keep_fuse
        for k, v in REAL.items():
            setattr(g, k, v)
    return out, log


kinds = lambda log: [x[0] for x in log]
o, log = run_first([OK_J])
check("articles fit: one search, one answer, no grade, no new search", kinds(log) == ["search", "judge"] and o[0]["status"] == "done"
      and o[0]["fit"] == "answers" and len(o[0]["claims"]) == 1 and o[0]["retries"] == 0)
o, log = run_first([J(fit="partly", claims=CL, caveat="reste")])
check("partly: answered with its remark, no new search", kinds(log) == ["search", "judge"] and o[0]["caveat"] == "reste")
o, log = run_first([NO_J, OK_J])
check("articles do not fit: the missing rule goes to the rewriting, new search, then the answer",
      kinds(log) == ["search", "judge", "reformulate", "more", "judge"] and log[2][1] == "sanction du vol" and o[0]["status"] == "done"
      and len(o[0]["claims"]) == 1 and o[0]["retries"] == 1)
o, log = run_first([NO_J, NO_J])
check("still nothing: said plainly, no claim, remembered for the conversation",
      kinds(log) == ["search", "judge", "reformulate", "more", "judge"] and o[0]["text"] == g.NOT_FOUND["fr"] and o[0]["claims"] == []
      and o[0]["status"] == "done" and o[0]["history"][-1]["a"] == g.NOT_FOUND["fr"])
o, log = run_first([ASK_J, OK_J], turns=("Puis-je annuler ?", "Un bail."))
check("results show the question is unclear: the user is asked, the reply is merged, searched and answered",
      o[0]["status"] == "clarify" and o[0]["text"] == "Quel contrat ?" and kinds(log) == ["search", "judge", "search", "judge"]
      and "Puis-je annuler ?" in log[2][1] and "Un bail." in log[2][1] and o[1]["status"] == "done" and len(o[1]["claims"]) == 1)
o, log = run_first([ASK_J, ASK_J], turns=("Puis-je annuler ?", "Un bail."))
check("never asked twice after a search: the reply is answered with what was found",
      o[1]["status"] == "done" and kinds(log)[-2:] == ["judge", "forced"] and len(o[1]["claims"]) == 1)
o, log = run_first([J(fit="unclear"), ], turns=("Q ?",))
check("unclear without a question: answered with what was found", o[0]["status"] == "done" and kinds(log)[-1] == "forced")
o, log = run_first([J(fit="answers"), OK_J])
check("'answers' without any claim counts as not found", kinds(log) == ["search", "judge", "reformulate", "more", "judge"] and o[0]["claims"])
o, log = run_first([J(fit="not_found", claims=CL), NO_J])
check("claims written although nothing fits are ignored", o[0]["claims"] == [])
o, log = run_first([], evidence={})
check("no article at all: new search without a model call to judge, then said plainly",
      kinds(log) == ["search", "reformulate", "more"] and o[0]["text"] == g.NOT_FOUND["fr"])
o, log = run_first([OK_J, OK_J], turns=("Q1 ?", "Q2 ?"))
check("a new question starts clean", o[1]["retries"] == 0 and o[1]["clarified"] is False and o[1]["status"] == "done")
o, log = run_first([], intent="out_of_scope")
check("out of scope is still refused before any search", o[0]["status"] == "refuse" and log == [])
C.ANSWER_FIRST = False
check("ANSWER_FIRST off: the grade step is back", g.route_after_retrieve({}) == "grade" and g.route_after_more({"retries": 1}) == "grade"
      and g.route_after_answer({"claims": CLAIMS[:1], "fit": "not_found"}) == "verify" and "fit" not in g._fresh())
C.ANSWER_FIRST = True

# the steps wait for the provider's per-minute limit instead of failing
import analyze as AN2
class Busy:
    """Refuses `n` times with a per-minute 429, then answers."""
    def __init__(self, n, result, msg="429 Rate limit reached on tokens per minute (TPM). Please try again in 10ms."):
        self.n, self.result, self.msg, self.calls = n, result, msg, 0
    def with_structured_output(self, *a, **k):
        return self
    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.n:
            raise RuntimeError(self.msg)
        return self.result
for name, mod, attr, call, result in (
        ("verify", V, "llm", lambda: V.verify(ans, ev, "Q ?"), Verification(verdicts=[])),
        ("answer", ANS, "llm", lambda: ANS.make_answer("Q", "fr", EVIDENCE), Answer(claims=CL)),
        ("answer (judging)", ANS, "llm", lambda: ANS.judge_answer("Q", "fr", EVIDENCE), OK_J),
        ("analyze", AN2, "structured_llm", lambda: AN2.analyze("Q"), "analysis")):
    busy = Busy(1, result)
    keep, _ = getattr(mod, attr), setattr(mod, attr, busy)
    try:
        got_it = call()
    finally:
        setattr(mod, attr, keep)
    check(f"{name}: waits after a per-minute refusal and tries again", got_it is result and busy.calls == 2)
busy = Busy(99, None, "401 invalid api key")
keep, V.llm = V.llm, busy
try:
    V.verify(ans, ev, "Q ?")
    failed = False
except RuntimeError:
    failed = True
finally:
    V.llm = keep
check("another error is not retried", failed and busy.calls == 1)
eight = [Claim(text=f"Cas {i}.", cited_ids=["PenalFr/s02/218"]) for i in range(8)]
check("unproven switches are off by default", C.LIST_CASES is False and C.FAST_JUDGED is False)
check("cases not listed: 6 sentences, no cases field, no cases prompt", "cases" not in J.model_fields and "list in `cases`" not in ANS.judged_prompt()
      and "never more than 6" in ANS.judged_prompt())
C.LIST_CASES = True
o8, _ = run_first([ANS.JudgedCases(fit="answers", cases=["a"], claims=eight)])
check("up to 8 sentences go through the whole graph (answer, check, finalize)", len(o8[0]["claims"]) == 8 and o8[0]["status"] == "done")
check("the judged prompt allows 8, the evaluated schema still 6", "up to 8" in ANS.judged_prompt() and "never more than 6" not in ANS.judged_prompt()
      and ANS.Answer.model_fields["claims"].metadata[-1].max_length == 6)
check("the cases are listed before the claims", list(ANS.JudgedCases.model_fields).index("cases") < list(ANS.JudgedCases.model_fields).index("claims")
      and "list in `cases`" in ANS.judged_prompt())
C.LIST_CASES = False

# Rapide with the judging step (ui/backend.py, config.FAST_JUDGED)
import pipeline as PL
DOC = Document(page_content="Article 264.- La peine", metadata={"chunk_id": "PenalFr/s02/264", "source": "PenalFr.pdf", "article": "264", "law_type": "penal"})
keep_pl = {"ask_question": PL.ask_question}
keep_g = {k: getattr(g, k) for k in ("retrieve_g", "to_evidence")}


def rapide(judged, docs=(DOC,), judged_mode=True, q="Quelle peine ?"):
    g.retrieve_g = lambda q_, codes, info=None: list(docs)
    g.to_evidence = lambda ds: {d.metadata["chunk_id"]: {"source": "PenalFr.pdf", "article": "264", "law_type": "penal", "text": d.page_content} for d in ds}
    PL.ask_question = lambda *a: ("texte libre", [DOC])
    keep_j, ANS.judge_answer = ANS.judge_answer, lambda q_, l, e: judged
    C.FAST_JUDGED, keep_mode = judged_mode, C.FAST_JUDGED
    me = SimpleNamespace(_vs=object())
    me._fast_judged = lambda *a: Backend._fast_judged(me, *a)
    try:
        return list(Backend.stream_fast(me, q, "t"))
    finally:
        ANS.judge_answer, C.FAST_JUDGED = keep_j, keep_mode
        for k, v in keep_pl.items():
            setattr(PL, k, v)
        for k, v in keep_g.items():
            setattr(g, k, v)


CL2 = [Claim(text="Règle.", cited_ids=["PenalFr/s02/264"])]
ev_ = rapide(J(fit="answers", claims=CL2, caveat="c"))
fin = ev_[-1][1]
check("Rapide: one search, one judged answer with its citations, no check", [e[1] for e in ev_[:-1]] == ["retrieve", "answer"]
      and fin["claims"][0]["cited_ids"] == ["PenalFr/s02/264"] and fin["caveat"] == "c" and fin["mode"] == "fast")
check("Rapide: articles that do not answer -> said plainly, nothing written", rapide(J(fit="not_found", claims=CL2))[-1][1]["claims"] == []
      and rapide(J(fit="not_found"))[-1][1]["text"] == g.NOT_FOUND["fr"])
check("Rapide: no article found -> no model call", rapide(None, docs=())[-1][1]["text"] == g.NOT_FOUND["fr"])
check("Rapide: unclear -> the question to the user", rapide(J(fit="unclear", clarification_question="Quel contrat ?"))[-1][1]["text"] == "Quel contrat ?")
check("Rapide: English question -> English message", rapide(J(fit="not_found"), q="What is the punishment for stabbing?")[-1][1]["text"] == g.NOT_FOUND["en"])
check("FAST_JUDGED off: the free text of the one-pass pipeline", rapide(None, judged_mode=False)[-1][1]["text"] == "texte libre")

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
blk = lambda q: A._user_block(q, "fr", EVIDENCE)
check("details asked (fr, en, ar): the request is added; otherwise the block is the v2 one",
      all(A.DETAILS_REQUEST in blk(q) for q in ("Donnez plus de détails sur le vol", "Can you explain the penalty?", "أريد تفاصيل أكثر"))
      and A.DETAILS_REQUEST not in blk("Quelle est la peine pour le vol ?"))
C.ANSWER_STYLE = "v2"
check("v2 style: no details request", A.DETAILS_REQUEST not in blk("plus de détails"))
C.ANSWER_STYLE = style
check("evaluated answer prompt (v2) unchanged", hashlib.sha256(A.SYSTEM_PROMPT.encode("utf-8")).hexdigest().startswith("5dbea27a8fcdf8c6"))
check("evaluated one-pass prompt (v1.1) unchanged", hashlib.sha256(pipeline._SYSTEM_PROMPT.encode("utf-8")).hexdigest().startswith("d2acda8ce54721ef"))
print("all checks passed")
