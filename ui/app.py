"""Streamlit interface (French): one chat, answers with cited articles, every question logged to SQLite."""
import os
import sys
import time
import uuid
import logging
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts as T
import render as R
from store import Store
from stub import DemoBackend

LOG = logging.getLogger("rag_ui")
DEMO = os.environ.get("RAG_UI_DEMO", "").strip().lower() in ("1", "true", "yes", "on") or "--demo" in sys.argv


@st.cache_resource(show_spinner="Chargement des modèles et de l'index…")
def get_backend():
    if DEMO:
        return DemoBackend()
    from backend import Backend
    return Backend()


@st.cache_resource
def get_policy():
    import keys
    if DEMO:
        return keys.KeyPolicy(True, int(os.environ.get("UI_OWNER_DAILY_CAP", "0") or 0), lambda: True)
    return keys.KeyPolicy.from_env()


@st.cache_resource
def get_store():
    return Store()


def new_conversation():
    get_backend().forget(st.session_state.thread_id)
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def html(text: str, number: int | None = None):
    st.markdown(R.to_html(text, number), unsafe_allow_html=True)


def show(view: dict):
    if view["kind"] == "error":
        st.error(view["text"])
        return
    if view.get("rewritten"):
        st.caption(T.REWRITTEN.format(text=view["rewritten"]))
    if view.get("unverified"):
        st.warning(T.UNVERIFIED_BANNER)
    for note in view.get("notes") or []:
        st.caption(note)
    if view["kind"] == "answer":
        for i, claim in enumerate(view["claims"], 1):
            html(claim["text"], number=i)
            for c in claim["citations"]:
                with st.expander(c["label"]):
                    html(c["text"])
    else:
        html(view["text"])
        if view["kind"] == "clarify":
            st.caption(T.CLARIFY_HINT)
    if view.get("dropped"):
        st.caption(T.DROPPED.format(n=view["dropped"]))
    if view.get("caveat"):
        st.info(T.CAVEAT.format(text=view["caveat"]))


def sidebar(policy):
    with st.sidebar:
        if st.button(T.NEW_CONVERSATION, use_container_width=True):
            new_conversation()
        st.text_input(T.KEY_LABEL, type="password", key="personal_key", help=T.KEY_HELP)
        if st.session_state.personal_key.strip():
            st.caption(T.KEY_PERSONAL_ACTIVE)
            st.button(T.KEY_FORGET, on_click=lambda: st.session_state.update(personal_key=""))
        elif policy.owner_usable():
            left = policy.remaining()
            if left is not None:
                st.caption(T.KEY_OWNER_REMAINING.format(n=left))
        else:
            st.caption(T.KEY_NONE)


def answer(question: str, policy) -> dict:
    """Runs one question, logs it, and returns the view to display. Never raises."""
    import keys
    started, final, error = time.time(), None, None
    key, refused = policy.resolve(st.session_state.personal_key)
    key_source = "personal" if key else "owner"
    if refused:
        error, view = refused, R.view_error(refused)
    else:
        try:
            with keys.session_key(key):
                with st.status(T.STEPS_TITLE, expanded=True) as status:
                    try:
                        for kind, value in get_backend().stream_v2(question, st.session_state.thread_id):
                            if kind == "step":
                                st.write(f"• {R.node_label(value)}")
                            else:
                                final = value
                    except Exception:
                        status.update(label=T.STEPS_STOPPED, state="error", expanded=False)
                        raise
                    status.update(label=T.STEPS_DONE, state="complete", expanded=False)
            view = R.view_v2(final or {})
        except Exception as e:                      # friendly message; technical detail to the console only
            LOG.error("question failed: %s", R.technical_detail(e))
            error = R.classify_error(e)
            view = R.view_error(error)
    try:
        turn = sum(m["role"] == "user" for m in st.session_state.messages)
        get_store().log(st.session_state.thread_id, turn, question, final, error, time.time() - started, key_source)
    except Exception as e:                          # a logging problem must never hide the answer
        LOG.error("logging failed: %s", type(e).__name__)
    return view


def main():
    st.set_page_config(page_title=T.TITLE, page_icon="⚖️", initial_sidebar_state="collapsed")
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("thread_id", str(uuid.uuid4()))
    ss.setdefault("personal_key", "")
    policy = get_policy()
    sidebar(policy)
    st.title(T.TITLE)
    st.write(T.SUBTITLE)
    if DEMO:
        st.warning(T.DEMO_BANNER)
    st.caption(T.DISCLAIMER_FR)
    st.caption(T.LOG_NOTICE)
    with st.expander(T.DISCLAIMER_OTHER_TITLE):
        html(T.DISCLAIMER_AR)
        st.markdown(T.DISCLAIMER_EN)
    for m in ss.messages:
        with st.chat_message(m["role"]):
            html(m["text"]) if m["role"] == "user" else show(m["view"])
    question = st.chat_input(T.INPUT_PLACEHOLDER)
    if question:
        ss.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            html(question)
        with st.chat_message("assistant"):
            view = answer(question, policy)
            show(view)
        ss.messages.append({"role": "assistant", "view": view})


main()
