from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from analyze import analyze
from langgraph.checkpoint.memory import MemorySaver
from retrieve import retrieve_for_plan
from retriever import _fuse
from retrieve_g import retrieve_g
from grade import grade
from reformulate import reformulate
from answer import make_answer, check_answer, render
from answer import Answer, Claim
from verify import verify, bad_claims
from condense import condense

def fresh_state() -> dict:
    return {"evidence": {}, "docs": [], "claims": [], "caveat": "", "cited": [], "problems": [],
            "verdicts": [], "bad": [], "repairs": 0, "dropped": 0, "draft_claims": [], "draft_text": "",
            "retries": 0, "tried": [], "missing": "", "enough": False, "sub_queries": [],
            "text": "", "status": "", "first_verdicts": []}

def _fresh() -> dict:
    """fresh_state, plus (config.NORMALIZE_QUERY == "v3" only) an empty normalization record for the new turn."""
    import config as C
    return fresh_state() | ({"normalize": {}} if getattr(C, "NORMALIZE_QUERY", "off") == "v3" else {})

REFUSE = {"fr": "Cette question sort du cadre des codes tunisiens que je couvre.",
          "en": "This is outside the Tunisian legal codes I cover.",
          "ar": "هذا السؤال خارج نطاق القوانين التونسية التي أغطيها."}
NO_ARTICLE = {"fr": "Aucun article pertinent n'a été trouvé dans les codes que je couvre.",
              "en": "No relevant article was found in the codes I cover.",
              "ar": "لم يتم العثور على مادة ذات صلة في القوانين التي أغطيها."}
CLARIFY_FALLBACK = {"fr": "Pouvez-vous préciser votre question ?",
                    "en": "Could you give more details?",
                    "ar": "هل يمكنك توضيح سؤالك أكثر؟"}
class AgentState(TypedDict):
    question: str
    language: str
    intent: str
    target_codes: list
    clarification_question: Optional[str]
    status: str
    text: str
    original_question: Optional[str]
    clarify_count: int
    sub_queries: list
    evidence: dict
    docs: list
    retries: int
    tried: list[str]
    missing: str
    enough: bool
    problems: list[str]
    cited: list[str]
    full_question: str
    degraded: list[str]
    claims: list[dict]
    caveat: str
    repairs: int
    verdicts: list[dict]
    bad: list[int]
    dropped: int
    draft_claims: list[dict]
    draft_text: str
    first_verdicts: list[dict]
    history: list[dict]          # last 2 answered turns {"q": standalone question, "a": answer text[:600]}
    condensed_from: str          # the raw message when condense rewrote it, else ""
    is_followup: bool
    normalize: dict              # config.NORMALIZE_QUERY == "v3" only: the normalization record of the turn

def remember(state, text) -> list[dict]:
    return (state.get("history", []) + [{"q": state["full_question"], "a": text[:600]}])[-2:]

def condense_node(state: AgentState) -> dict:
    if state.get("status") == "clarify":            # a reply to a clarification: analyze_node merges it
        return {"condensed_from": "", "is_followup": False}
    out = {"condensed_from": "", "is_followup": False, "degraded": []}     # new question: reset
    history = state.get("history", [])
    if not history:
        return out
    try:
        s = condense(history[-2:], state["question"])
    except Exception as e:                          # visible fallback: keep the raw message
        return out | {"degraded": [f"condense_failed: {type(e).__name__}"]}
    return out | {"question": s.question, "condensed_from": state["question"], "is_followup": s.is_followup}

def analyze_node(state: AgentState) -> dict:
    if state.get("status") == "clarify" and state.get("original_question"):
        full_question = state["original_question"] + f"\nUser's answer: {state['question']}"
        degraded = list(state.get("degraded") or [])      # same question, keep earlier events
    else:
        full_question = state["question"]
        degraded = list(state.get("degraded") or [])      # new question: condense_node reset it
    try:
        hint = state.get("language") if state.get("clarify_count", 0) > 0 else None
        r = analyze(full_question, hint)
        return {
            **_fresh(),
            "language": r.language,
            "intent": r.intent,
            "target_codes": r.target_codes,
            "clarification_question": r.clarification_question,
            "full_question": full_question,
            "degraded": degraded,
        }
    except Exception as e:
        print("Analyze failed:", e)
        return {
            **_fresh(),
            "language": "en",
            "intent": "clear",
            "target_codes": [],
            "clarification_question": None,
            "full_question": full_question,
            "degraded": degraded + [f"analyze_failed: {type(e).__name__}"],
        }

def clarify(state: AgentState) -> dict:
    prev = state.get("original_question") or ""
    transcript = prev + f"\nUser: {state['question']}\nAgent asked: {state['clarification_question']}"
    count = state.get("clarify_count", 0)
    return {
        "status": "clarify",
        "text": state["clarification_question"] or CLARIFY_FALLBACK.get(state.get("language"), CLARIFY_FALLBACK["fr"]),
        "original_question": transcript.strip(),
        "clarify_count": count + 1,
    }

def to_evidence(docs) -> dict:
    """v2.1: the documents plus the articles they reference (pipeline.evidence_docs, config.XREF_EXPAND), each text
    with its labelled header lines (pipeline.evidence_text: "Article cité par l'article N", "Contexte : ...")."""
    from pipeline import evidence_docs, evidence_text
    return {
        d.metadata["chunk_id"]: {
            "source": d.metadata.get("source"),
            "article": d.metadata.get("article"),
            "law_type": d.metadata.get("law_type"),
            "text": evidence_text(d),
        }
        for d in evidence_docs(docs)
    }

def retrieve_node(state: AgentState) -> dict:
    import config as C
    info = {}                                   # filled only when config.NORMALIZE_QUERY == "v3"
    if getattr(C, "NORMALIZE_QUERY", "off") == "v3":
        docs = retrieve_g(state["full_question"], state["target_codes"], info)
    else:                                       # off: the call of v2.1-dev wave 1b, unchanged
        docs = retrieve_g(state["full_question"], state["target_codes"])
    evidence = to_evidence(docs)
    out = {"docs": docs, "evidence": evidence, "retries": 0, "tried": [], "missing": "",
           "status": "answer", "clarify_count": 0, "original_question": None}
    if info:                                    # V3: the record, and a visible event when it fell back
        out["normalize"] = info
        if info.get("fell_back"):
            out["degraded"] = list(state.get("degraded") or []) + [f"normalize_failed: {info.get('status')}"]
    return out

def grade_node(state: AgentState) -> dict:
    g = grade(state["full_question"], state["evidence"])
    return {"enough": g.enough, "missing": g.missing}

def reformulate_node(state: AgentState) -> dict:
    r = reformulate(state["full_question"], state["language"], state["tried"], state["evidence"], state["missing"])
    return {"sub_queries": [(s.text, s.code) for s in r.sub_queries],
            "tried": state["tried"] + [s.text for s in r.sub_queries]}

def retrieve_more_node(state: AgentState) -> dict:
    extra = retrieve_for_plan(state["sub_queries"], 6, True, soft=True)
    docs = _fuse([state["docs"], extra])[:6]
    return {"docs": docs, "evidence": to_evidence(docs), "retries": state["retries"] + 1}

def answer_node(state: AgentState) -> dict:
    if not state["evidence"]:                 # nothing retrieved: do not ask the LLM to answer from nothing
        text = NO_ARTICLE.get(state.get("language"), NO_ARTICLE["fr"])
        return {"text": text,
                "problems": [], "cited": [], "status": "done", "degraded": state.get("degraded", []),
                "claims": [], "caveat": "", "history": remember(state, text)}
    ans = make_answer(state["full_question"], state["language"], state["evidence"])
    problems = check_answer(ans, state["evidence"])
    cited = sorted({i for c in ans.claims for i in c.cited_ids})
    text = render(ans, state["evidence"], state["language"])
    claims = [c.model_dump() for c in ans.claims]
    return {"text": text,
            "problems": problems, "cited": cited, "status": "done", "degraded": state.get("degraded", []),
            "claims": claims, "caveat": ans.caveat,
            "draft_claims": claims, "draft_text": text,
            "repairs": 0, "verdicts": [], "bad": [], "dropped": 0}

def route_after_analyze(state: AgentState) -> str:
    if state["intent"] == "out_of_scope":
        return "refuse"
    if state["intent"] == "ambiguous" and state.get("clarify_count", 0) < 2:
        return "clarify"
    return "retrieve"

def route_after_grade(state: AgentState) -> str:
    return "answer" if state["enough"] or state["retries"] >= 2 else "reformulate"

def route_after_more(state: AgentState) -> str:
    return "answer" if state["retries"] >= 2 else "grade"   # no grade after the last reformulation (as loop_core)

def refuse(state):
    return {"status": "refuse", "text": REFUSE.get(state.get("language"), REFUSE["fr"]),
            "clarify_count": 0, "original_question": None}


NO_SUPPORT = {"fr": "Je n'ai pas trouvé de réponse que les articles retrouvés permettent de vérifier.",
              "en": "I could not find an answer that the retrieved articles let me verify.",
              "ar": "لم أجد إجابة يمكن التحقق منها من المواد المسترجعة."}
DROPPED = {"fr": "{n} affirmation(s) retirée(s) car non confirmée(s) par les articles cités.",
           "en": "{n} claim(s) removed because the cited articles do not confirm them.",
           "ar": "تم حذف {n} من الادعاءات لأن المواد المستشهد بها لا تؤكدها."}

def verify_node(state):
    ans = Answer(claims=[Claim(**c) for c in state["claims"]], caveat=state["caveat"])
    try:
        v = verify(ans, state["evidence"])
    except Exception as e:                       # visible fallback, never silent
        return {"verdicts": [], "bad": [],
                "degraded": state["degraded"] + [f"verify_failed: {type(e).__name__}"]}
    out = {"verdicts": [x.model_dump() for x in v.verdicts], "bad": bad_claims(ans, v)}
    if state["repairs"] == 0:                    # first pass: kept, a repair's verify does not overwrite it
        out["first_verdicts"] = out["verdicts"]
    return out

def feedback_from(state):
    lines = []
    for v in state["verdicts"]:
        i = v["claim_index"]
        if v["verdict"] != "supported" and 1 <= i <= len(state["claims"]):
            lines.append(f'- "{state["claims"][i-1]["text"]}" -> {v["verdict"]}: {v["problem"]}')
    return "\n".join(lines)

def repair_node(state):
    ans = make_answer(state["full_question"], state["language"], state["evidence"],
                      feedback=feedback_from(state))
    return {"claims": [c.model_dump() for c in ans.claims], "caveat": ans.caveat,
            "problems": check_answer(ans, state["evidence"]), "repairs": state["repairs"] + 1}

def finalize_node(state):
    lang = state["language"]
    claims = [Claim(**c) for c in state["claims"]]
    bad = set(state["bad"])
    kept = [c for i, c in enumerate(claims, 1) if i not in bad]
    if not kept:
        seen = ", ".join(cid.split("/")[0] + " art. " + cid.split("/")[-1]
                         for cid in list(state["evidence"])[:3])
        text = NO_SUPPORT.get(lang, NO_SUPPORT["fr"]) + f" [{seen}]"
        return {"text": text, "claims": [], "caveat": "", "cited": [], "dropped": len(bad), "status": "done",
                "history": remember(state, text)}
    text = render(Answer(claims=kept, caveat=state["caveat"]), state["evidence"], lang)
    if bad:
        text += "\n\n" + DROPPED.get(lang, DROPPED["fr"]).format(n=len(bad))
    return {"text": text, "claims": [c.model_dump() for c in kept],
            "cited": sorted({i for c in kept for i in c.cited_ids}),
            "dropped": len(bad), "status": "done", "history": remember(state, text)}

def route_after_answer(state):
    return "verify" if state["claims"] else "end"

def route_after_verify(state):
    return "repair" if state["bad"] and state["repairs"] == 0 else "finalize"





builder = StateGraph(AgentState)

builder.add_node("condense", condense_node)
builder.add_node("analyze", analyze_node)
builder.add_node("clarify", clarify)
builder.add_node("refuse", refuse)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("reformulate", reformulate_node)
builder.add_node("retrieve_more", retrieve_more_node)
builder.add_node("answer", answer_node)
builder.add_node("verify", verify_node)
builder.add_node("repair", repair_node)
builder.add_node("finalize", finalize_node)


builder.add_edge(START, "condense")
builder.add_edge("condense", "analyze")
builder.add_conditional_edges("answer", route_after_answer, {"verify": "verify", "end": END})
builder.add_conditional_edges("verify", route_after_verify, {"repair": "repair", "finalize": "finalize"})
builder.add_conditional_edges("analyze", route_after_analyze,
                              {"refuse": "refuse", "clarify": "clarify", "retrieve": "retrieve"})
builder.add_edge("clarify", END)
builder.add_edge("refuse", END)
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route_after_grade, {"answer": "answer", "reformulate": "reformulate"})
builder.add_edge("reformulate", "retrieve_more")
builder.add_conditional_edges("retrieve_more", route_after_more, {"answer": "answer", "grade": "grade"})
builder.add_edge("repair", "verify")
builder.add_edge("finalize", END)

app = builder.compile(checkpointer=MemorySaver())
def run(thread, turns):
    cfg = {"configurable": {"thread_id": thread}}
    for q in turns:
        r = app.invoke({"question": q}, cfg)
        print(f"[{thread}] {q!r} -> {r['status']}")
        if r.get("condensed_from"):
            print("   standalone:", r["question"])
        print("   text:", r["text"])
        if r.get("sub_queries"):
            print("   queries:", r["sub_queries"])
        print(f"   repairs: {r.get('repairs')}, dropped: {r.get('dropped')}, degraded: {r.get('degraded')}")

if __name__ == "__main__":
    run("clear", ["What is the penalty for theft?"])
    run("ambig", ["ما هي مدة الاحتفاظ؟", "أقصد الاحتفاظ بالمشتبه فيه لدى الضابطة العدلية"])
    run("off",   ["What is the capital of France?"])
    run("gate-miss", ["Can I break my contract?"])