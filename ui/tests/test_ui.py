"""Offline tests of the UI: rendering helpers on stub outputs, error messages, key policy, per-session key, the SQLite log,
and the app run end to end in demo mode (streamlit.testing). No network, no Groq: the key test answers through an httpx mock.

    python ui/tests/test_ui.py
"""
import io
import os
import re
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
EMOJI = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0xFE0F, 0xFE0F), (0xFF0B, 0xFF0B))) + "]")
FAST_ANSWER = """**Réponse directe :**
La peine pour le vol simple est de cinq ans d’emprisonnement et de cent vingt dinars d’amende.

**Articles applicables :**
- **Article 264**, Code pénal : « La peine est de cinq ans <b>x</b> »

**Source(s) :**
- *PenalFr.pdf* (Code pénal tunisien)

**Explication :**
L’article 264 prévoit la sanction des vols autres que ceux des articles 260 à 263."""


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
          {"mode", "tokens_in", "tokens_out", "tokens_cached", "tokens_reasoning"} <= set(store.Store(old, upload=False).log("s", 1, "q", state, None, 1, "owner", "fast", {"in": 3, "out": 4}) or store.Store(old, upload=False).rows()[0]))
    check("no API key column or value anywhere in the schema", "key" not in " ".join(store.COLUMNS).replace("key_source", ""))
    bad = store.Store(tmp / "b.db", upload=lambda p: 1 / 0)
    bad.log("s", 1, "q", state, None, 0.1, "owner")
    bad.sync()
    check("a failing upload never raises", len(bad.rows()) == 1)
    os.environ["RAG_LOG_BACKUP_DIR"] = str(tmp / "bucket" / "logs")
    st4 = store.Store(tmp / "c.db")
    del os.environ["RAG_LOG_BACKUP_DIR"]
    for i in (1, 2):
        st4.log("b", i, f"q{i}", state, None, 0.1, "owner")
        st4.sync()
    files = list((tmp / "bucket" / "logs").glob("interactions-*.db"))
    check("bucket folder: one readable copy per start, refreshed at every question, read back by load_all",
          len(files) == 1 and st4.every == 0 and [r["question"] for r in store.load_all(tmp / "bucket" / "logs")] == ["q1", "q2"], files)
    st2 = store.Store(tmp / "interactions-x.db", upload=False)
    st2.log("s9", 1, "autre", state, None, 0.1, "owner")
    merged = store.load_all(tmp)
    check("load_all merges the backups of several starts, oldest first", len(merged) == 1 and merged[0]["question"] == "autre")
    st3 = store.Store(tmp / "interactions-y.db", upload=False)
    st3.log("s10", 1, "encore", state, None, 0.1, "owner")
    check("load_all with two files", sorted(r["question"] for r in store.load_all(tmp)) == ["autre", "encore"])


def test_laws():
    import laws as L
    import chat
    try:
        import fitz
        import functools
        norm = lambda x: re.sub(r"\s*-\s*", "-", re.sub(r"n°\s*", "n° ", re.sub(r"\s+", " ", x))).replace("’", "'").lower()
        pdf = functools.lru_cache(None)(lambda stem: norm("".join(x.get_text() for x in fitz.open(UI.parent / "data" / f"{stem}.pdf"))))
        miss = [stem for stem, a in L.ACTS.items() if norm(L.act(*a)) not in pdf(stem)]
        check("every code's act, with its date, is written in its PDF", not miss, miss)
        miss = [k for k, v in L.SECTIONS.items() if norm(v) not in pdf(k.split("/")[0])]
        check("every annexed act named is written in its PDF", not miss, miss)
        check("the act promulgates or recasts the code (the words are next to it in the PDF)", not [
            stem for stem, a in L.ACTS.items() if stem not in ("PenalFr", "tun_RecAcFreeLi")
            and not re.search(re.escape(norm(L.act(*a))) + r".{0,60}(promulgation|publication|refonte)", pdf(stem))])
    except ImportError:
        print("skip PDF checks (PyMuPDF missing)")
    check("the 14 codes have a name in the three languages", all(set(L.NAMES[x]) == set(L.ACTS) for x in L.NAMES) and len(L.ACTS) == 14)
    check("a code's article, an annexed law's, the promulgating act's, an unidentified annex's",
          [R.citation_label(x) for x in ("PenalFr/s02/264", "Statutpersonnel/s07/3", "PenalFr/s00/1", "EnfantFr/s04/224")]
          == ["Code pénal, art. 264", "Loi n° 98-94 du 9 novembre 1998, art. 3", "Décret du 9 juillet 1913, art. 1",
              "Texte annexé au Code de la protection de l'enfant, art. 224"])
    T1 = "Contexte : Section II\nArticle 260 (Modifié par la loi n° 89-23 du \n27 février 1989).- Est puni"
    T2 = "Article 6 (Abrogé par l’article 7 de la loi n°2010-39 du 26 juillet 2010, portant unification de l’âge).-"
    T3 = "Art. 232 (Abrogée par le décret du 26 mai 1949 et ajouté par la loi n° 64-34 du 2 juillet 1964).- Sera"
    T4 = "Article 264.- La peine est de cinq ans (article 5) hors les cas prévus"
    check("amendment read from the article, not from its body", [L.note(x) for x in (T1, T4)] == ["modifié par la loi n° 89-23 du 27 février 1989", ""])
    check("repealed, and repealed then replaced", [L.repealed(x) for x in (T1, T2, T3, T4)] == [False, True, False, False]
          and L.note(T2) == "abrogé par l'article 7 de la loi n° 2010-39 du 26 juillet 2010")
    one_fr = L.sentence(["PenalFr/s02/260"], {"PenalFr/s02/260": T1})
    check("one article: law, date and amendment", one_fr == "Cette règle figure à l'article 260 du Code pénal (décret du 9 juillet 1913). "
          "Cet article a été modifié par la loi n° 89-23 du 27 février 1989.", one_fr)
    check("several articles and laws, grouped; no amendment list",
          L.sentence(["PenalFr/s02/260", "travailFr/s01/14", "PenalFr/s02/264"], {"PenalFr/s02/260": T1})
          == "Ces règles figurent aux articles 260 et 264 du Code pénal (décret du 9 juillet 1913) et à l'article 14 du Code du travail (loi n° 66-27 du 30 avril 1966).")
    check("the same sentence in English and Arabic, amendment included",
          L.sentence(["PenalFr/s02/260"], {"PenalFr/s02/260": T1}, "en") == "This rule is in article 260 of the Penal Code (Decree of 9 July 1913). "
          "This article was amended by Law No. 89-23 of 27 February 1989."
          and "المجلة الجزائية" in L.sentence(["PenalFr/s02/260"], {}, "ar"))
    check("an annexed law named in the answer's language", "article 3 of Law No. 98-94 of 9 November 1998" in L.sentence(["Statutpersonnel/s07/3"], {}, "en")
          and "de la loi n° 98-94 du 9 novembre 1998" in L.sentence(["Statutpersonnel/s07/3"], {}))
    st = {"status": "done", "language": "fr", "claims": [{"text": "Règle.", "cited_ids": ["CommerceFr/s01/6"]}],
          "evidence": {"CommerceFr/s01/6": {"source": "CommerceFr.pdf", "article": "6", "text": T2}}}
    md = chat.view_markdown(R.view_v2(st))
    check("a repealed article is flagged above the sources and in its label",
          md.index("Attention : Code de commerce, art. 6") < md.index("src-title") and "(abrogé par l" in md and "a été abrogé" not in md)
    many = chat.view_markdown(R.view_v2(st | {"claims": [{"text": f"Phrase {i}.", "cited_ids": ["CommerceFr/s01/6"]} for i in range(5)]}))
    check("up to 4 sentences: a paragraph; more: a list", "<ul" not in md and many.count("<li>") == 5)


def test_cache():
    import chat
    import store
    import tempfile
    st = store.Store(Path(tempfile.mkdtemp()) / "k.db", upload=False)
    be, pol, cache = stub.DemoBackend(0), keys.KeyPolicy(True, 2, lambda: True), chat.AnswerCache(size=2)
    seen = []
    be.remember = lambda t, s: seen.append((t, s["claims"][0]["text"]))
    run = lambda q, t, turn=1, mode="agent": list(chat.ask(q, t, turn, "", be, pol, st, mode, cache))
    ok_state = stub.final_state("answer", "x") | {"dropped": 0}
    be.stream_v2 = lambda q, t: iter([("usage", {"in": 10, "out": 5}), ("final", dict(ok_state, question=q, full_question=q))])
    first = run("Quelle peine pour le vol ?", "a")
    again = run("  quelle peine pour le VOL", "b")
    check("same first question (case, spaces, final punctuation ignored): same answer, 0 token, no step",
          again[-1][1]["tokens_in"] == 0 and T.FROM_CACHE in again[-1][1]["markdown"] and len(again) == 1
          and again[-1][1]["markdown"].split("</div>", 1)[1].strip() == first[-1][1]["markdown"].split("<div class=\"usage\">")[0].strip())
    check("an answer from memory says so first, before the answer", again[-1][1]["markdown"].startswith('<div class="memo">' + T.FROM_CACHE))
    check("the conversation is told the answer, for follow-up questions", [x[0] for x in seen] == ["b"])
    check("a reused answer does not use the owner's daily cap, and is logged as 'cache' with 0 token",
          pol.remaining() == 1 and st.rows()[-1]["key_source"] == "cache" and st.rows()[-1]["tokens_in"] == 0 and st.rows()[-1]["answer"] == st.rows()[0]["answer"])
    check("the other mode and later turns are computed", T.FROM_CACHE not in run("Quelle peine pour le vol ?", "c", mode="fast")[-1][1]["markdown"]
          and cache.get("Quelle peine pour le vol ?", "agent") is not None)
    pol2 = keys.KeyPolicy(True, 0, lambda: True)
    later = list(chat.ask("Quelle peine pour le vol ?", "a", 2, "", be, pol2, st, "agent", cache))
    check("a question repeated inside its conversation is answered again", T.FROM_CACHE not in later[-1][1]["markdown"])
    c = chat.AnswerCache()
    bad = [ok_state | {"dropped": 1}, ok_state | {"degraded": ["verify_failed: X"]}, stub.final_state("clarify", "q"), stub.final_state("refuse", "q"),
           ok_state | {"claims": []}, ok_state | {"is_followup": True}]
    check("not kept: dropped sentences, a failed step, a clarification, a refusal, no answer, a follow-up", not any(c.put(f"q{i}", "agent", b) for i, b in enumerate(bad)))
    check("not kept: a fast answer that rests on no article", not c.put("q", "fast", {"status": "done", "mode": "fast", "text": "Je ne sais pas.", "sources": []}))
    for i in range(4):
        cache.put(f"question {i}", "agent", ok_state)
    check("the oldest answers leave when it is full", cache.get("question 0", "agent") is None and cache.get("question 3", "agent") is not None)
    check("only the cited articles are kept", set(cache.get("question 3", "agent")["evidence"]) == {"PenalFr/s02/264", "PenalFr/s02/264-bis"})


def test_quota():
    import chat
    import store
    import quota
    import tempfile
    now = [1_000_000.0]
    b = quota.TokenBudget(200_000, clock=lambda: now[0])
    b.add(None, 12_000)
    b.add("gsk_personal", 3_000)
    check("tokens left: each key has its own count", (b.left(None), b.left("gsk_personal"), b.left("gsk_other")) == (188_000, 197_000, 200_000))
    check("a personal key is kept only as a hash", not any("gsk_" in k for k in b._used))
    now[0] += quota.DAY + 1
    check("tokens older than 24 hours no longer count", b.left(None) == 200_000)
    b.add(None, 500_000)
    check("never below zero", b.left(None) == 0)
    now[0] += quota.DAY + 1
    msg = "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199990, Requested 900."
    check("the provider's refusal gives the real figure", b.learn(None, msg) and b.left(None) == 10 and not b.learn(None, "429 tokens per minute (TPM): Limit 8000, Used 7900"))
    check("cached tokens do not count", quota.counted(4200, 650, 1500) == 3350)
    from datetime import datetime, timezone
    c = quota.TokenBudget(100_000)
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session": "s", "turn": 1, "key_source": "owner", "tokens_in": 9000, "tokens_out": 1000, "tokens_cached": 0}
    c.seed([row, dict(row), dict(row, turn=2, key_source="personal"), dict(row, turn=3, ts="2020-01-01T00:00:00+00:00"), dict(row, turn=4, ts=None)])
    check("restored from the log: the app's key only, last 24 hours, each row once", c.used(None) == 10_000)
    st = store.Store(Path(tempfile.mkdtemp()) / "q.db", upload=False)
    be, pol, cache, bud = stub.DemoBackend(0), keys.KeyPolicy(True, 0, lambda: True), chat.AnswerCache(), quota.TokenBudget(200_000)
    ok_state = stub.final_state("answer", "x") | {"dropped": 0}
    be.stream_v2 = lambda q, t: iter([("usage", {"in": 4200, "out": 650, "cached": 1500}), ("final", dict(ok_state, question=q, full_question=q))])
    list(chat.ask("Q ?", "a", 1, "", be, pol, st, "agent", cache, bud))
    list(chat.ask("Q ?", "b", 1, "", be, pol, st, "agent", cache, bud))
    check("a question lowers the app's tokens left; an answer from memory does not", bud.used(None) == 3350)
    list(chat.ask("Q2 ?", "c", 1, "gsk_mine", be, pol, st, "agent", cache, bud))
    check("a personal key lowers its own count only", bud.used("gsk_mine") == 3350 and bud.used(None) == 3350)
    def refuse(q, t):
        raise stub.DemoRateLimit(msg)
        yield
    be.stream_v2 = refuse
    list(chat.ask("Q3 ?", "d", 1, "", be, pol, st, "agent", cache, bud))
    check("a daily refusal updates the meter", bud.left(None) == 10)
    none = keys.KeyPolicy(True, 0, lambda: False)
    seen = []
    be.stream_v2 = lambda q, t: seen.append(q) or iter([("final", ok_state)])
    out = list(chat.ask("Question jamais posée ?", "e", 1, "", be, none, st, "agent", cache, bud))[-1][1]["markdown"]
    check("no key anywhere: a new question is refused and the model is not called", T.ERRORS["no_key"] in out and not seen)


def test_not_found():
    import chat
    ev = {f"PenalFr/s02/{i}": {"source": "PenalFr.pdf", "article": str(i), "text": f"Article {i}.- texte"} for i in range(1, 9)}
    md = chat.view_markdown(R.view_v2({"status": "done", "language": "fr", "claims": [], "text": "Je n'ai pas trouvé.", "evidence": ev}))
    check("nothing fits: the message, no source, the articles that were read in a closed block",
          md.startswith("<div") and "Je n" in md and "src-title" not in md and T.WORDS["fr"]["others"].format(n=6) in md)


def test_fast_layout():
    import chat
    src = [{"id": f"PenalFr/s02/{a}", "source": "PenalFr.pdf", "article": a, "text": f"texte {a}"}
           for a in ("274", "263", "262", "260", "261", "264")]
    v = R.view_fast({"text": FAST_ANSWER, "sources": src})
    check("fast answer cut at its section labels", set(v["sections"]) == {"direct", "articles", "sources", "explanation"}
          and v["sections"]["direct"].startswith("La peine pour le vol simple"), v["sections"])
    md = chat.view_markdown(v)
    check("fast answer: the answer, its explanation, the articles, then the sources; no 'Réponse directe', no file name",
          md.index("cinq ans d’emprisonnement et") < md.index("L’article 264 prévoit") < md.index("**Articles applicables**") < md.index("src-title")
          and "Réponse directe" not in md and "PenalFr.pdf" not in md)
    check("fast answer: Markdown kept, HTML escaped", "**Article 264**" in md and "&lt;b&gt;" in md and "<b>x" not in md)
    check("fast sources: the article the answer rests on, not those it only refers to; the others in one closed block",
          [x["id"].split("/")[-1] for x in v["sources"]] == ["264"] and len(v["others"]) == 5
          and T.WORDS["fr"]["others"].format(n=5) in md, [x["id"].split("/")[-1] for x in v["sources"]])
    v = R.view_fast({"text": "Texte libre, sans rubrique.", "sources": src[:2]})
    md = chat.view_markdown(v)
    check("fast answer naming no article: shown whole, nothing presented as a source, the articles in the closed block",
          "Texte libre" in md and not v["sources"] and len(v["others"]) == 2 and "src-title" not in md)
    v = R.view_fast({"language": "en", "sources": src, "text": "**Direct Answer:** I don't have information about the weather.\n\n"
                                                              "**Applicable Articles:** None.\n\n**Source(s):** None."})
    md = chat.view_markdown(v)
    check("fast non-answer in English: no source, no empty list of articles, English labels",
          "weather" in md and "None" not in md and "src-title" not in md and T.WORDS["en"]["others"].format(n=6) in md, md[:200])
    v = R.view_fast({"sources": src, "text": "**Réponse directe :** Dix ans (article 263).\n\n**Articles applicables :**\n"
                                            "- **Article 263** – Code pénal : dix ans.\n- Art. 264 : cinq ans, hors les articles 260 à 262."})
    check("fast sources: the direct answer's article and each listed article",
          sorted(x["id"].split("/")[-1] for x in v["sources"]) == ["263", "264"])
    v = R.view_fast({"text": "**الإجابة المباشرة:** العقوبة خمس سنوات حسب الفصل 264.", "sources": src})
    md = chat.view_markdown(v)
    check("fast answer in Arabic: right to left, Arabic word for the sources, cited article found",
          'dir="rtl"' in md and T.WORDS["ar"]["sources"] in md and [x["id"].split("/")[-1] for x in v["sources"]] == ["264"])
    st = {"status": "done", "question": "Quelle peine ?", "claims": [{"text": "c <i>", "cited_ids": ["A/s1/1"]}], "evidence": {}}
    v = R.view_v2(st | {"condensed_from": "Quelle peine ? "})
    check("'Question comprise comme' is hidden when the question was not rewritten", v["rewritten"] is None)
    md = chat.view_markdown(R.view_v2(st | {"condensed_from": "et ça ?", "question": "Q <script>", "caveat": "x <script>", "language": "ar"}))
    check("rewritten question, claim and remark escaped; the remark's word follows the answer's language",
          "<script>" not in md and "<i>" not in md and T.WORDS["ar"]["note"] in md)


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
          [x[1] for x in steps][:2] == [T.NODE_LABELS["condense"], T.NODE_LABELS["analyze"]] and "Première affirmation, citée par un article.<sup>[1]</sup> [Exemple] Seconde affirmation" in md
          and "<sup>[1]</sup><sup>[2]</sup>" in md and "<summary>[1] Code pénal, art. 264</summary>" in md and "<script" not in md)
    check("answer first, then the remark, then the sources (each article once)",
          md.index("Seconde affirmation") < md.index("Ces règles figurent") < md.index("Remarque") < md.index("src-title") < md.index("<summary>[1]")
          and md.count("<summary>") == 2)
    steps, md = run("Puis-je rompre mon contrat ?", "t2")
    check("clarify: question back plus the hint", "S&#x27;agit-il" in md and T.CLARIFY_HINT in md)
    _, md = run("Quelle sera la météo demain ?", "t3")
    check("refuse: refusal text", "sort du cadre" in md)
    _, md = run("ما هي عقوبة السرقة؟", "t4")
    check("Arabic claim is right to left", 'dir="rtl"' in md)
    _, md = run("Exemple de panne", "t5")
    check("failed verification: warning shown", "Réponse non vérifiée" in md and 'class="warn"' in md)
    _, md = run("Exemple de quota", "t6")
    check("quota error: friendly message, nothing raised", T.ERRORS["quota_daily"] in md)
    _, md = run("Exemple d'erreur", "t7")
    check("unexpected error: generic message, no traceback", T.ERRORS["other"] in md and "Traceback" not in md and "demo failure" not in md)
    capped = keys.KeyPolicy(True, 1, lambda: True)
    list(chat.ask("vol", "c1", 1, "", be, capped, st))
    out = list(chat.ask("vol", "c1", 2, "", be, capped, st))
    check("no emoji in the interface texts, the styles or the code of the page", not EMOJI.search("".join(
        (UI / f).read_text(encoding="utf-8") for f in ("texts.py", "chat.py", "render.py", "style.css", "app.py", "../app.py"))))
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
          and "4 850 tokens (entrée 4 200, dont 1 500 en cache, sortie 650, dont 300 de réflexion)" in out["markdown"] and out["tokens_cached"] == 1500 and "mode agent" in out["markdown"], out["markdown"][-120:])
    out = list(chat.ask("Quelle est la peine du vol ?", "t11", 1, "", be, pol, st, "fast"))
    fast = out[-1][1]
    check("fast mode: two steps, one answer, the articles given to the model, 1080 tokens",
          [e[1] for e in out if e[0] == "step"] == [T.NODE_LABELS["retrieve"], T.NODE_LABELS["answer"]]
          and "Réponse courte" in fast["markdown"] and "Sources" in fast["markdown"]
          and "<summary>Code pénal, art. 264</summary>" in fast["markdown"] and "1 080 tokens" in fast["markdown"]
          and "mode rapide" in fast["markdown"])
    r = st.rows()[-1]
    check("log keeps the mode, the tokens and the articles of a fast answer",
          r["tokens_reasoning"] == 60 and r["tokens_cached"] == 0 and r["mode"] == "fast" and r["tokens_in"] == 900 and r["tokens_out"] == 180 and "PenalFr/s02/264" in r["cited"], r)
    out = list(chat.ask("erreur", "t12", 1, "", be, pol, st))[-1][1]["markdown"]
    check("an error row has no usage line", "tokens" not in out)
    class Broken:
        def log(self, *a, **k):
            raise RuntimeError("disk full")
    out = list(chat.ask("vol", "c9", 1, "", be, pol, Broken()))
    check("a failing log never hides the answer", out[-1][0] == "done" and "Première affirmation" in out[-1][1]["markdown"])


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
    for t in (test_views, test_errors, test_policy, test_session_key, test_store, test_laws, test_cache, test_quota, test_not_found, test_fast_layout, test_chat, test_app_demo, test_policy_in_app):
        t()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
