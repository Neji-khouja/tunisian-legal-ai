"""Offline tests of the UI: rendering helpers on stub outputs, error messages, key policy, per-session key, the SQLite log,
and the app run end to end in demo mode (streamlit.testing). No network, no Groq: the key test answers through an httpx mock.

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
    import tempfile
    logdb = Path(tempfile.mkdtemp()) / "log.db"
    os.environ["RAG_LOG_DB"] = str(logdb)
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
        n = len(at.session_state["messages"])
        thread = at.session_state["thread_id"]
        next(b for b in at.sidebar.button if b.label == T.NEW_CONVERSATION).click().run()
        check("Nouvelle conversation: history cleared, new thread id", n > 0 and at.session_state["messages"] == []
              and at.session_state["thread_id"] != thread)
    logging.getLogger().removeHandler(handler)
    check("privacy: the question appears in no log or console output", "SECRET-QUESTION" not in log_buf.getvalue() + out.getvalue())
    check("console gets the technical detail of the unexpected error, as a type and a location",
          "RuntimeError at" in log_buf.getvalue() and "demo failure" not in log_buf.getvalue(), log_buf.getvalue()[-300:])
    import store
    rows = store.Store(logdb, upload=False).rows()
    by_q = {r["question"]: r for r in rows}
    check("every question of the session is in the SQLite log, in order, with its session id",
          len(rows) == 7 and rows[0]["question"] == secret and len({r["session"] for r in rows}) == 1, [r["question"] for r in rows])
    check("log: statuses clarify / done / refuse / error and the cited articles",
          by_q[secret]["status"] == "clarify" and by_q["Un contrat de travail"]["status"] == "done"
          and "PenalFr/s02/264" in by_q["Un contrat de travail"]["cited"] and by_q["Quelle sera la météo demain ?"]["status"] == "refuse"
          and by_q["Exemple de quota"]["error"] == "quota_daily", {k: (r["status"], r["error"]) for k, r in by_q.items()})
    check("log: a verification failure is flagged and the turn counter follows the session",
          by_q["Exemple de panne"]["unverified"] == 1 and [r["turn"] for r in rows[:3]] == [1, 2, 3])
    for k in ("RAG_UI_DEMO", "RAG_UI_DEMO_DELAY", "RAG_LOG_DB"):
        del os.environ[k]


def test_store():
    import tempfile
    import sqlite3
    import store
    tmp = Path(tempfile.mkdtemp())
    sent = []
    t = [0.0]
    st1 = store.Store(tmp / "a.db", upload=lambda p: sent.append((p.name, sqlite3.connect(p).execute("select count(*) from interactions").fetchone()[0])),
                      clock=lambda: t[0])
    state = {"status": "done", "language": "fr", "text": "t", "claims": [{"text": "c", "cited_ids": ["A/s1/1", "B/s1/2", "A/s1/1"]}],
             "dropped": 2, "degraded": ["verify_failed: X", "analyze_failed: Y"]}
    t[0] = 1000.0
    st1.log("s1", 1, "question un", state, None, 1.234, "owner")
    import time
    for _ in range(50):
        if sent:
            break
        time.sleep(0.05)
    r = st1.rows()[0]
    check("log row: question, answer text, claims JSON, cited ids deduplicated, dropped, unverified, seconds, key source",
          r["question"] == "question un" and r["answer"] == "t" and r["cited"] == '["A/s1/1", "B/s1/2"]' and r["dropped"] == 2
          and r["unverified"] == 1 and r["seconds"] == 1.23 and r["key_source"] == "owner" and '"cited_ids"' in r["claims"])
    check("sync: a consistent copy named with this start's id is uploaded once", len(sent) == 1 and sent[0][0].startswith("interactions-") and sent[0][1] == 1, sent)
    t[0] = 1100.0
    st1.log("s1", 2, "question deux", None, "quota_daily", 0.5, "personal")
    time.sleep(0.2)
    check("sync is throttled (not again within 5 minutes)", len(sent) == 1)
    check("an error row has status error and no answer", st1.rows()[1]["status"] == "error" and st1.rows()[1]["error"] == "quota_daily"
          and st1.rows()[1]["answer"] == "")
    old = tmp / "old.db"
    con = sqlite3.connect(old)
    con.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, turn INTEGER, question TEXT, status TEXT, language TEXT, answer TEXT, claims TEXT, cited TEXT, dropped INTEGER, unverified INTEGER, degraded TEXT, error TEXT, seconds REAL, key_source TEXT)")
    con.commit()
    con.close()
    check("a database from before the mode and token columns is upgraded in place",
          {"mode", "tokens_in", "tokens_out"} <= set(store.Store(old, upload=False).log("s", 1, "q", state, None, 1, "owner", "fast", (3, 4)) or store.Store(old, upload=False).rows()[0]))
    check("no API key column or value anywhere in the schema", "key" not in " ".join(store.COLUMNS).replace("key_source", ""))
    bad = store.Store(tmp / "b.db", upload=lambda p: 1 / 0)
    bad.log("s", 1, "q", state, None, 0.1, "owner")
    bad.sync()
    check("a failing upload never raises", len(bad.rows()) == 1)
    st2 = store.Store(tmp / "interactions-x.db", upload=False)
    st2.log("s9", 1, "autre", state, None, 0.1, "owner")
    merged = store.load_all(tmp)
    check("load_all merges the backups of several starts, oldest first", len(merged) == 1 and merged[0]["question"] == "autre")
    st3 = store.Store(tmp / "interactions-y.db", upload=False)
    st3.log("s10", 1, "encore", state, None, 0.1, "owner")
    check("load_all with two files", sorted(r["question"] for r in store.load_all(tmp)) == ["autre", "encore"])


def test_chat():
    import tempfile
    import chat
    import store
    import stub
    tmp = Path(tempfile.mkdtemp())
    st = store.Store(tmp / "c.db", upload=False)
    be = stub.DemoBackend(0)
    pol = keys.KeyPolicy(True, 0, lambda: True)

    def run(q, thread="t1", turn=1, key=""):
        ev = list(chat.ask(q, thread, turn, key, be, pol, st))
        return [e for e in ev if e[0] == "step"], ev[-1][1]["markdown"]
    steps, md = run("Que risque-t-on en cas de vol simple ?")
    check("answer: steps streamed in French, then one Markdown answer with numbered claims and details per citation",
          [x[1] for x in steps][:2] == [T.NODE_LABELS["condense"], T.NODE_LABELS["analyze"]] and "<bdi>1.</bdi>" in md
          and "<details><summary>Code pénal, art. 264</summary>" in md and "<script" not in md)
    steps, md = run("Puis-je rompre mon contrat ?", "t2")
    check("clarify: question back plus the hint", "S&#x27;agit-il" in md and T.CLARIFY_HINT in md)
    _, md = run("Quelle sera la météo demain ?", "t3")
    check("refuse: refusal text", "sort du cadre" in md)
    _, md = run("ما هي عقوبة السرقة؟", "t4")
    check("Arabic claim is right to left", 'dir="rtl"' in md)
    _, md = run("Exemple de panne", "t5")
    check("failed verification: warning shown", "⚠️" in md and T.UNVERIFIED_BANNER.strip("*")[:20] in md)
    _, md = run("Exemple de quota", "t6")
    check("quota error: friendly message, nothing raised", T.ERRORS["quota_daily"] in md)
    _, md = run("Exemple d'erreur", "t7")
    check("unexpected error: generic message, no traceback", T.ERRORS["other"] in md and "Traceback" not in md and "demo failure" not in md)
    capped = keys.KeyPolicy(True, 1, lambda: True)
    list(chat.ask("vol", "c1", 1, "", be, capped, st))
    out = list(chat.ask("vol", "c1", 2, "", be, capped, st))
    check("owner cap reached: friendly message", T.ERRORS["owner_cap"] in out[-1][1]["markdown"])
    out = list(chat.ask("vol", "c1", 3, "gsk_personal", be, capped, st))
    check("a personal key passes the cap", T.ERRORS["owner_cap"] not in out[-1][1]["markdown"])
    rows = st.rows()
    check("every question logged with its turn, status and key source",
          [r["question"] for r in rows][:2] == ["Que risque-t-on en cas de vol simple ?", "Puis-je rompre mon contrat ?"]
          and rows[1]["status"] == "clarify" and rows[5]["error"] == "quota_daily" and rows[-1]["key_source"] == "personal"
          and rows[-2]["error"] == "owner_cap", [(r["status"], r["error"], r["key_source"]) for r in rows])
    out = list(chat.ask("vol", "t10", 1, "", be, pol, st))[-1][1]
    check("token meter: usage line with the total, input and output tokens", out["tokens_in"] == 4200 and out["tokens_out"] == 650
          and "4 850 tokens (entrée 4200, sortie 650)" in out["markdown"] and "mode agent" in out["markdown"], out["markdown"][-120:])
    out = list(chat.ask("Quelle est la peine du vol ?", "t11", 1, "", be, pol, st, "fast"))
    fast = out[-1][1]
    check("fast mode: two steps, one answer, the articles given to the model, 1080 tokens",
          [e[1] for e in out if e[0] == "step"] == [T.NODE_LABELS["retrieve"], T.NODE_LABELS["answer"]]
          and "Réponse rapide" in fast["markdown"] and T.SOURCES_TITLE in fast["markdown"]
          and "<details><summary>Code pénal, art. 264</summary>" in fast["markdown"] and "1 080 tokens" in fast["markdown"]
          and "mode rapide" in fast["markdown"])
    r = st.rows()[-1]
    check("log keeps the mode, the tokens and the articles of a fast answer",
          r["mode"] == "fast" and r["tokens_in"] == 900 and r["tokens_out"] == 180 and "PenalFr/s02/264" in r["cited"], r)
    out = list(chat.ask("erreur", "t12", 1, "", be, pol, st))[-1][1]["markdown"]
    check("an error row has no usage line", "tokens" not in out)
    class Broken:
        def log(self, *a, **k):
            raise RuntimeError("disk full")
    out = list(chat.ask("vol", "c9", 1, "", be, pol, Broken()))
    check("a failing log never hides the answer", out[-1][0] == "done" and "<bdi>1.</bdi>" in out[-1][1]["markdown"])


def test_policy_in_app():
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_resource.clear()                       # the policy is cached per process, as in the app
    import tempfile
    os.environ["RAG_LOG_DB"] = str(Path(tempfile.mkdtemp()) / "log.db")
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
    for k in ("RAG_UI_DEMO", "RAG_UI_DEMO_DELAY", "UI_OWNER_DAILY_CAP", "RAG_LOG_DB"):
        del os.environ[k]


if __name__ == "__main__":
    for t in (test_views, test_errors, test_policy, test_session_key, test_store, test_chat, test_app_demo, test_policy_in_app):
        t()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
