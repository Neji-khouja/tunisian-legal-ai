"""Offline tests of the UI: rendering helpers on stub outputs, error messages, key policy, per-session key, and the app
run end to end in demo mode (streamlit.testing). No network, no Groq: the key test answers through an httpx mock.

    python ui/tests/test_ui.py
"""
import io
import os
import sys
import logging
import threading
import contextlib
from pathlib import Path

UI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UI))
import texts as T
import render as R
import keys
import stub

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or not detail else f": {detail}"))


def test_views():
    v = R.view_v2(stub.final_state("answer", "q"))
    check("answer: claims with citations, code name and article label from the chunk id",
          v["kind"] == "answer" and len(v["claims"]) == 2
          and [c["label"] for c in v["claims"][1]["citations"]] == ["Code pénal, art. 264", "Code pénal, art. 264-bis"])
    check("answer: article text under each citation, dropped count, caveat, not marked unverified",
          v["claims"][0]["citations"][0]["text"] == stub.PLACEHOLDER_ARTICLE and v["dropped"] == 1 and v["caveat"]
          and not v["unverified"])
    v = R.view_v2(stub.final_state("clarify", "q"))
    check("clarify: the question back is the assistant turn", v["kind"] == "clarify" and v["text"].startswith("[Exemple]"))
    v = R.view_v2(stub.final_state("refuse", "q"))
    check("refuse: refusal message", v["kind"] == "refuse" and "sort du cadre" in v["text"])
    v = R.view_v2(stub.final_state("degraded", "q"))
    check("degraded verify: 'Réponse non vérifiée' banner and a rate-limit note", v["unverified"] and T.RATE_LIMIT_IN_STEP in v["notes"])
    v = R.view_v2(stub.final_state("arabic", "q"))
    check("Arabic: claim right to left, French article text left to right",
          v["claims"][0]["rtl"] and not v["claims"][0]["citations"][0]["rtl"])
    v = R.view_v2({"status": "done", "claims": [], "text": "Aucun article pertinent n'a été trouvé.", "degraded": ["analyze_failed: X"]})
    check("no article: message, analyze fallback note", v["kind"] == "message" and T.FALLBACK_NOTES["analyze_failed"] in v["notes"])
    v = R.view_v2({"status": "done", "question": "Quelle peine pour le vol ?", "condensed_from": "et pour le vol ?",
                   "claims": [{"text": "x", "cited_ids": ["Missing/s1/9"]}], "evidence": {}})
    check("follow-up: rewritten question shown; a citation missing from the evidence is labelled, text unavailable",
          v["rewritten"] == "Quelle peine pour le vol ?" and v["claims"][0]["citations"][0]["label"] == "Missing, art. 9"
          and not v["claims"][0]["citations"][0]["available"])
    v = R.view_v11(*stub.DemoBackend(0).run_v11("vol"))
    check("v1.1: same layout without verification fields, with the 'not verified' note",
          v["kind"] == "fast" and v["note"] == T.FAST_NOTE and len(v["sources"]) == 2 and "claims" not in v and "unverified" not in v)
    v = R.view_v11(*stub.DemoBackend(0).run_v11("ما هي عقوبة السرقة؟"))
    check("v1.1 Arabic: right to left", v["rtl"])
    check("RTL detection", R.is_rtl("ما هي عقوبة السرقة؟") and not R.is_rtl("Code pénal, art. 264") and not R.is_rtl(""))
    check("claim number isolated, so it stays at the start of an Arabic line",
          R.to_html("هذه إجابة", number=1).startswith('<div dir="rtl"') and "<bdi>1.</bdi>" in R.to_html("هذه إجابة", number=1))
    check("HTML escaped (no markup from model or user text)", "&lt;script&gt;" in R.to_html("<script>alert(1)</script>")
          and "<script>" not in R.to_html("<script>"))
    check("unknown node name shown as is, known ones in French", R.node_label("verify") == T.NODE_LABELS["verify"] and R.node_label("x") == "x")


def test_errors():
    daily = RuntimeError("Error code: 429 - Rate limit reached on tokens per day (TPD). Please try again in 7m12.5s.")
    minute = RuntimeError("Error code: 429 - Rate limit reached on tokens per minute (TPM). Please try again in 1.5s.")
    auth = RuntimeError("Error code: 401 - {'error': {'message': 'Invalid API Key', 'code': 'invalid_api_key'}}")
    import httpx
    import groq
    conn = groq.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:9"))
    check("daily quota, per-minute limit, bad key, unreachable service, other",
          [R.classify_error(e) for e in (daily, minute, auth, conn, ValueError("x"))] == ["quota_daily", "rate_limit", "auth", "network", "other"])
    try:
        try:
            raise ValueError("SECRET question text")
        except ValueError as inner:
            raise RuntimeError("SECRET model output") from inner
    except RuntimeError as e:
        detail = R.technical_detail(e)
    check("console detail: types and code locations, never the messages", "RuntimeError" in detail and "ValueError" in detail
          and "SECRET" not in detail and "test_ui.py" in detail)
    v = R.view_error("quota_daily")
    check("friendly French message", v["kind"] == "error" and v["text"] == T.ERRORS["quota_daily"])


def test_policy():
    days = iter(["d1"] * 4 + ["d2"] * 3)
    p = keys.KeyPolicy(True, 2, lambda: True, today=lambda: next(days))
    got = [p.resolve(None) for _ in range(3)]
    check("owner key: cap of 2 per day, then refused", got == [(None, None), (None, None), (None, "owner_cap")], got)
    check("a personal key passes the cap", p.resolve(" gsk_user ") == ("gsk_user", None))
    check("the cap resets the next day", p.resolve(None) == (None, None))
    p = keys.KeyPolicy(False, 0, lambda: True)
    check("owner key off and no personal key: refused", p.resolve("") == (None, "no_key"))
    p = keys.KeyPolicy(True, 0, lambda: False)
    check("no owner key configured: refused, no cap shown", p.resolve(None) == (None, "no_key") and p.remaining() is None)
    os.environ["UI_OWNER_KEY"], os.environ["UI_OWNER_DAILY_CAP"] = "off", "5"
    p = keys.KeyPolicy.from_env(owner_present=lambda: True)
    check("policy from the environment", not p.owner_usable() and p.daily_cap == 5)
    del os.environ["UI_OWNER_KEY"], os.environ["UI_OWNER_DAILY_CAP"]


def test_session_key():
    """The real chain of the agent (ChatGroq inside a LangGraph node, streamed) against an httpx mock: each thread's
    requests carry its own key; without a session key, the client's (owner's) key."""
    import httpx
    from typing import TypedDict
    from langchain_groq import ChatGroq
    from langgraph.graph import StateGraph, START, END
    keys.install()
    seen, lock = [], threading.Lock()
    body = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def handler(request):
        with lock:
            seen.append((request.headers.get("x-tag"), request.headers["authorization"]))
        return httpx.Response(200, json=body)
    llm = ChatGroq(model="m", api_key="owner-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    class S(TypedDict):
        tag: str

    def node(s):
        for _ in range(3):
            llm.invoke("ping", extra_headers={"x-tag": s["tag"]})
        return {}
    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile()
    barrier = threading.Barrier(4)

    def worker(tag, key):
        barrier.wait()
        with keys.session_key(key):
            for _ in graph.stream({"tag": tag}, stream_mode="updates"):
                pass
    threads = [threading.Thread(target=worker, args=(f"t{i}", None if i == 3 else f"user-key-{i}")) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    want = {f"t{i}": ("Bearer owner-key" if i == 3 else f"Bearer user-key-{i}") for i in range(4)}
    check("per-session key: 4 concurrent sessions, each request signed with its session's key (owner's when none)",
          len(seen) == 12 and all(auth == want[tag] for tag, auth in seen), seen[:4])
    llm.invoke("ping", extra_headers={"x-tag": "after"})
    check("after the block, the owner's key again", seen[-1] == ("after", "Bearer owner-key"))
    keys.install()
    import groq
    check("install is idempotent", groq.Groq._ui_session_key)


def test_app_demo():
    from streamlit.testing.v1 import AppTest
    os.environ["RAG_UI_DEMO"], os.environ["RAG_UI_DEMO_DELAY"] = "1", "0"
    secret = "Puis-je rompre mon contrat SECRET-QUESTION ?"
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    logging.getLogger().addHandler(handler)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        at = AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()
        texts = lambda: " ".join(x.value for x in at.markdown) + " ".join(x.value for x in at.error) + \
            " ".join(x.value for x in at.warning) + " ".join(x.value for x in at.info) + " ".join(x.value for x in at.caption)
        check("app starts in demo mode: disclaimer and demo banner, no exception",
              not at.exception and T.DISCLAIMER_FR in texts() and T.DEMO_BANNER in texts())
        at.chat_input[0].set_value(secret).run()
        check("clarification shown as an assistant turn", not at.exception and "S&#x27;agit-il" in texts()
              and T.CLARIFY_HINT in texts())
        at.chat_input[0].set_value("Un contrat de travail").run()
        check("reply on the same conversation -> verified answer with citations", not at.exception
              and "Première affirmation" in texts() and any(e.label == "Code pénal, art. 264" for e in at.expander))
        for question, expect in (("Quelle sera la météo demain ?", "sort du cadre"), ("Exemple de panne", "Réponse non vérifiée"),
                                 ("ما هي عقوبة السرقة؟", 'dir="rtl"'), ("Exemple de quota", T.ERRORS["quota_daily"]),
                                 ("Exemple d'erreur", T.ERRORS["other"])):
            at.chat_input[0].set_value(question).run()
            check(f"case {expect[:24]!r}", not at.exception and expect in texts(), texts()[-300:])
        check("no traceback or raw JSON shown", "Traceback" not in texts() and "cited_ids" not in texts() and "{'" not in texts())
        at.sidebar.radio[0].set_value(T.MODE_FAST).run()
        at.chat_input[0].set_value("vol").run()
        check("mode Rapide: answer with the 'not verified' note and sources", not at.exception and T.FAST_NOTE in texts())
        n = len(at.session_state["messages"])
        thread = at.session_state["thread_id"]
        next(b for b in at.sidebar.button if b.label == T.NEW_CONVERSATION).click().run()
        check("Nouvelle conversation: history cleared, new thread id", n > 0 and at.session_state["messages"] == []
              and at.session_state["thread_id"] != thread)
    logging.getLogger().removeHandler(handler)
    check("privacy: the question appears in no log or console output", "SECRET-QUESTION" not in log_buf.getvalue() + out.getvalue())
    check("console gets the technical detail of the unexpected error, as a type and a location",
          "RuntimeError at" in log_buf.getvalue() and "demo failure" not in log_buf.getvalue(), log_buf.getvalue()[-300:])
    del os.environ["RAG_UI_DEMO"], os.environ["RAG_UI_DEMO_DELAY"]


def test_policy_in_app():
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_resource.clear()                       # the policy is cached per process, as in the app
    os.environ["RAG_UI_DEMO"], os.environ["RAG_UI_DEMO_DELAY"], os.environ["UI_OWNER_DAILY_CAP"] = "1", "0", "1"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        at = AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()
        at.chat_input[0].set_value("vol").run()
        at.chat_input[0].set_value("vol").run()
        capped = any(T.ERRORS["owner_cap"] in e.value for e in at.error)
        at.sidebar.text_input[0].set_value("gsk_personal").run()
        at.chat_input[0].set_value("vol").run()
        errors_after = sum(T.ERRORS["owner_cap"] in e.value for e in at.error)
    check("owner cap reached in the app: friendly message; a personal key then answers", capped and errors_after == 1)
    for k in ("RAG_UI_DEMO", "RAG_UI_DEMO_DELAY", "UI_OWNER_DAILY_CAP"):
        del os.environ[k]


if __name__ == "__main__":
    for t in (test_views, test_errors, test_policy, test_session_key, test_app_demo, test_policy_in_app):
        t()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
