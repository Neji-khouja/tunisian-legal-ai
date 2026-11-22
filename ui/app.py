"""Interface Streamlit (français) du chercheur juridique : modes Rapide (v1.1) et Vérifié (v2)."""
import os
import sys
import uuid
import logging
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts as T
import render as R
from stub import DemoBackend, DEMO_EXAMPLES

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


def init_session():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("thread_id", str(uuid.uuid4()))
    ss.setdefault("personal_key", "")
    ss.setdefault("pending", None)


def new_conversation():
    get_backend().forget(st.session_state.thread_id)
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def html(text: str, number: int | None = None):
    st.markdown(R.to_html(text, number), unsafe_allow_html=True)


def show(view: dict):
    kind = view["kind"]
    if kind == "error":
        st.error(view["text"])
        return
    if view.get("rewritten"):
        st.caption(T.REWRITTEN.format(text=view["rewritten"]))
    if kind == "fast":
        if view["rtl"]:
            html(view["text"])
        else:
            st.markdown(view["text"])
        st.caption(view["note"])
        if view["sources"]:
            st.markdown(f"**{T.SOURCES_TITLE}**")
            for c in view["sources"]:
                with st.expander(c["label"]):
                    html(c["text"])
        return
    if view.get("unverified"):
        st.error(T.UNVERIFIED_BANNER)
    for note in view.get("notes") or []:
        st.warning(note)
    if kind in ("clarify", "refuse", "message"):
        html(view["text"])
        if kind == "clarify":
            st.caption(T.CLARIFY_HINT)
    else:
        for i, claim in enumerate(view["claims"], 1):
            html(claim["text"], number=i)
            for c in claim["citations"]:
                with st.expander(c["label"]):
                    html(c["text"])
    if view.get("dropped"):
        st.caption(T.DROPPED.format(n=view["dropped"]))
    if view.get("caveat"):
        st.info(T.CAVEAT.format(text=view["caveat"]))


def sidebar(policy):
    with st.sidebar:
        mode = st.radio("Mode", [T.MODE_VERIFIED, T.MODE_FAST], index=0, key="mode")
        st.markdown(T.MODE_HELP[mode])
        if st.button(T.NEW_CONVERSATION, use_container_width=True):
            new_conversation()
        st.divider()
        st.markdown(f"**{T.KEY_TITLE}**")
        st.text_input(T.KEY_LABEL, type="password", key="personal_key", help=T.KEY_HELP)
        if st.session_state.personal_key.strip():
            st.caption(T.KEY_PERSONAL_ACTIVE)
            st.button(T.KEY_FORGET, on_click=lambda: st.session_state.update(personal_key=""))
        elif policy.owner_usable():
            st.caption(T.KEY_OWNER_ACTIVE)
            left = policy.remaining()
            if left is not None:
                st.caption(T.KEY_OWNER_REMAINING.format(n=left))
        else:
            st.caption(T.KEY_NONE)
        if DEMO:
            st.divider()
            st.markdown(f"**{T.DEMO_EXAMPLES_TITLE}**")
            for label, question in DEMO_EXAMPLES.items():
                if st.button(label, use_container_width=True):
                    st.session_state.pending = question
    return mode


def answer(question: str, mode: str, policy) -> dict:
    """Runs one question and returns the view to display. Never raises."""
    import keys
    key, refused = policy.resolve(st.session_state.personal_key)
    if refused:
        return R.view_error(refused)
    backend = get_backend()
    try:
        with keys.session_key(key):
            if mode == T.MODE_FAST:
                with st.spinner(T.FAST_SPINNER):
                    return R.view_v11(*backend.run_v11(question))
            with st.status(T.STEPS_TITLE, expanded=True) as status:
                final = None
                try:
                    for kind, value in backend.stream_v2(question, st.session_state.thread_id):
                        if kind == "step":
                            st.write(f"• {R.node_label(value)}")
                        else:
                            final = value
                except Exception:
                    status.update(label=T.STEPS_STOPPED, state="error", expanded=False)
                    raise
                status.update(label=T.STEPS_DONE, state="complete", expanded=False)
            return R.view_v2(final or {})
    except Exception as e:                                  # friendly message; technical detail to the console only
        LOG.error("question failed: %s", R.technical_detail(e))
        return R.view_error(R.classify_error(e))


def main():
    st.set_page_config(page_title=T.TITLE, page_icon="⚖️")
    init_session()
    policy = get_policy()
    mode = sidebar(policy)
    st.title(T.TITLE)
    if DEMO:
        st.warning(T.DEMO_BANNER)
    st.info(T.DISCLAIMER_FR)
    with st.expander(T.DISCLAIMER_OTHER_TITLE):
        html(T.DISCLAIMER_AR)
        st.markdown(T.DISCLAIMER_EN)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                html(m["text"])
            else:
                show(m["view"])
    question = st.chat_input(T.INPUT_PLACEHOLDER) or st.session_state.pending
    st.session_state.pending = None
    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            html(question)
        with st.chat_message("assistant"):
            view = answer(question, mode, policy)
            show(view)
        st.session_state.messages.append({"role": "assistant", "view": view})


main()
