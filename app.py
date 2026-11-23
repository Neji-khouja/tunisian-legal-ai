"""Gradio web interface (Hugging Face Space entry point). Local: python app.py   (RAG_UI_DEMO=1: canned answers, no models).

Left sidebar: conversations (kept in the visitor's browser), mode, token meter, personal key. Tabs: chat, how it works.
Every question and answer is also written to the SQLite log (ui/store.py).
"""
import os
import sys
import time
import copy
import uuid
import threading
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ui"))
import texts as T
import chat
import keys
from store import Store
from stub import DemoBackend

try:
    import spaces                                   # present on ZeroGPU Spaces, which refuse to start without one @spaces.GPU function

    @spaces.GPU
    def _gpu_not_used():
        return 0
except ImportError:
    pass

DEMO = os.environ.get("RAG_UI_DEMO", "").strip().lower() in ("1", "true", "yes", "on")
STORE = Store()
if DEMO:
    POLICY = keys.KeyPolicy(True, int(os.environ.get("UI_OWNER_DAILY_CAP", "0") or 0), lambda: True)
else:
    POLICY = keys.KeyPolicy.from_env()
_BACKEND = {}
_LOCK = threading.Lock()
MAX_CONVERSATIONS = 40                              # kept in the browser, oldest dropped
EXAMPLES = ["Quelle est la peine pour le vol simple ?",
            "Quelle est la durée de la période d'essai dans un contrat de travail ?",
            "ما هي شروط صحة العقد؟",
            "What are the conditions for a valid marriage contract?"]
CSS = """
footer {display: none !important}
.disclaimer {font-size: 0.8em; opacity: 0.75}
.tokens {font-size: 0.85em}
"""


def backend():
    with _LOCK:
        if "b" not in _BACKEND:
            if DEMO:
                _BACKEND["b"] = DemoBackend()
            else:
                from backend import Backend
                _BACKEND["b"] = Backend()
        return _BACKEND["b"]


# ---- conversations (a list of dicts kept in the browser: id, title, mode, ts, messages, tokens)

def find(convs, conv_id):
    return next((c for c in convs if c["id"] == conv_id), None)


def history_choices(convs):
    return [(c["title"], c["id"]) for c in sorted(convs, key=lambda c: -c["ts"])]


def history_update(convs, conv_id):
    return gr.update(choices=history_choices(convs), value=conv_id if find(convs, conv_id) else None)


def meter(convs, conv_id):
    here = (find(convs, conv_id) or {}).get("tokens", 0)
    device = sum(c.get("tokens", 0) for c in convs)
    f = lambda n: f"{n:,}".replace(",", " ")
    return T.TOKENS_METER.format(conv=f(here), device=f(device))


def respond(question, messages, conv_id, convs, mode, personal_key):
    question = (question or "").strip()
    messages = messages or []
    convs = copy.deepcopy(convs or [])
    if not question:
        yield messages, "", conv_id, convs, history_update(convs, conv_id), meter(convs, conv_id)
        return
    conv = find(convs, conv_id)
    if conv is None:
        conv = {"id": str(uuid.uuid4()), "title": question[:48], "mode": mode, "ts": time.time(), "messages": [], "tokens": 0}
        convs.append(conv)
    conv_id = conv["id"]
    turn = sum(m["role"] == "user" for m in messages) + 1
    messages = messages + [{"role": "user", "content": question}, {"role": "assistant", "content": "…"}]
    conv["messages"], conv["ts"] = messages, time.time()
    yield messages, "", conv_id, convs, history_update(convs, conv_id), meter(convs, conv_id)
    steps = []
    for kind, value in chat.ask(question, conv_id, turn, personal_key or "", backend(), POLICY, STORE, mode):
        if kind == "step":
            steps.append(value)
            messages[-1] = {"role": "assistant", "content": f"*{' → '.join(steps)}*"}
        else:
            messages[-1] = {"role": "assistant", "content": value["markdown"]}
            conv["tokens"] += value["tokens_in"] + value["tokens_out"]
        conv["messages"] = messages
        yield messages, "", conv_id, convs, history_update(convs, conv_id), meter(convs, conv_id)
    convs.sort(key=lambda c: -c["ts"])
    del convs[MAX_CONVERSATIONS:]
    yield messages, "", conv_id, convs, history_update(convs, conv_id), meter(convs, conv_id)


def open_conversation(conv_id, convs):
    conv = find(convs or [], conv_id)
    if conv is None:
        return [], None, gr.update(), meter(convs or [], None)
    return conv["messages"], conv_id, conv.get("mode", "agent"), meter(convs, conv_id)


def new_conversation(convs):
    return [], None, history_update(convs or [], None), meter(convs or [], None)


def delete_conversation(conv_id, convs):
    convs = [c for c in (convs or []) if c["id"] != conv_id]
    if conv_id and "b" in _BACKEND:
        _BACKEND["b"].forget(conv_id)
    return [], None, convs, history_update(convs, None), meter(convs, None)


def start(convs):
    convs = convs or []
    return history_update(convs, None), meter(convs, None), key_status("")


def key_status(personal_key):
    if (personal_key or "").strip():
        return T.KEY_PERSONAL_ACTIVE
    if POLICY.owner_usable():
        left = POLICY.remaining()
        return T.KEY_OWNER_REMAINING.format(n=left) if left is not None else ""
    return T.KEY_NONE


def mode_help(mode):
    return T.MODE_HELP.get(mode, "")


with gr.Blocks(title=T.TITLE, theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CSS) as demo:
    convs = gr.BrowserState([], storage_key="tunisian_legal_rag_conversations")
    conv_id = gr.State(None)

    with gr.Sidebar(open=True, width=330):
        gr.Markdown(f"## ⚖️ {T.TITLE}")
        new_btn = gr.Button(T.NEW_CHAT, variant="primary")
        history = gr.Radio(choices=[], value=None, label=T.HISTORY_LABEL, interactive=True)
        delete_btn = gr.Button(T.DELETE_CHAT, size="sm", variant="secondary")
        mode = gr.Radio(T.MODE_CHOICES, value="agent", label="Mode")
        mode_note = gr.Markdown(mode_help("agent"))
        token_meter = gr.Markdown(meter([], None), elem_classes="tokens")
        with gr.Accordion(T.KEY_LABEL, open=False):
            key_box = gr.Textbox(type="password", label=T.KEY_LABEL, info=T.KEY_HELP)
            key_note = gr.Markdown(key_status(""))

    with gr.Tabs():
        with gr.Tab(T.TAB_CHAT):
            if DEMO:
                gr.Markdown(f"**{T.DEMO_BANNER}**")
            chatbot = gr.Chatbot(type="messages", height="64vh", show_label=False, show_copy_button=True,
                                 placeholder=f"### {T.TITLE}\n{T.SUBTITLE}")
            with gr.Row():
                box = gr.Textbox(placeholder=T.INPUT_PLACEHOLDER, max_length=1000, show_label=False, autofocus=True,
                                 scale=8, container=False)
                send = gr.Button("Envoyer", variant="primary", scale=1, min_width=90)
            gr.Examples(EXAMPLES, inputs=box, label="Exemples")
            gr.Markdown(f"{T.DISCLAIMER_FR}\n\n{T.LOG_NOTICE}", elem_classes="disclaimer")
            with gr.Accordion(T.DISCLAIMER_OTHER_TITLE, open=False):
                gr.Markdown(f'<div dir="rtl">{T.DISCLAIMER_AR}</div>\n\n{T.DISCLAIMER_EN}', elem_classes="disclaimer")
        with gr.Tab(T.TAB_HOW):
            gr.Markdown(T.ABOUT_INTRO)
            gr.Markdown(T.ABOUT_MODES)
            gr.Markdown(T.ABOUT_RETRIEVAL)
            gr.Markdown(T.ABOUT_AGENT)
            gr.Image(str(ROOT / "agent" / "agent_flow_v2.png"), show_label=False, interactive=False,
                     show_download_button=False, show_fullscreen_button=True)
            gr.Markdown(T.ABOUT_TOKENS)
            gr.Markdown(T.ABOUT_DATA)

    ins = [box, chatbot, conv_id, convs, mode, key_box]
    outs = [chatbot, box, conv_id, convs, history, token_meter]
    box.submit(respond, ins, outs, api_name="ask")
    send.click(respond, ins, outs, api_name=False)
    new_btn.click(new_conversation, [convs], [chatbot, conv_id, history, token_meter], api_name=False)
    history.input(open_conversation, [history, convs], [chatbot, conv_id, mode, token_meter], api_name=False)
    delete_btn.click(delete_conversation, [conv_id, convs], [chatbot, conv_id, convs, history, token_meter], api_name=False)
    mode.change(mode_help, [mode], [mode_note], api_name=False)
    key_box.change(key_status, [key_box], [key_note], api_name=False)
    demo.load(start, [convs], [history, token_meter, key_note], api_name=False)

demo.queue(default_concurrency_limit=2)

if not DEMO:
    threading.Thread(target=backend, daemon=True).start()          # load models and index while the page opens

if __name__ == "__main__":
    demo.launch()
