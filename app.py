"""Gradio web interface (Hugging Face Space entry point). Local: python app.py   (RAG_UI_DEMO=1: canned answers, no models)."""
import os
import sys
import uuid
import threading
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "ui"))
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


def backend():
    with _LOCK:
        if "b" not in _BACKEND:
            if DEMO:
                _BACKEND["b"] = DemoBackend()
            else:
                from backend import Backend
                _BACKEND["b"] = Backend()
        return _BACKEND["b"]


def respond(question, history, thread_id, personal_key):
    question = (question or "").strip()
    history = history or []
    if not question:
        yield history, "", thread_id
        return
    thread_id = thread_id or str(uuid.uuid4())
    turn = sum(m["role"] == "user" for m in history) + 1
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": "…"}]
    yield history, "", thread_id
    steps = []
    for kind, value in chat.ask(question, thread_id, turn, personal_key or "", backend(), POLICY, STORE):
        if kind == "step":
            steps.append(value)
            content = f"*{' → '.join(steps)}*"
        else:
            content = value
        history[-1] = {"role": "assistant", "content": content}
        yield history, "", thread_id


def new_conversation(thread_id):
    if thread_id:
        backend().forget(thread_id)
    return [], None


def key_status(personal_key):
    if (personal_key or "").strip():
        return T.KEY_PERSONAL_ACTIVE
    if POLICY.owner_usable():
        left = POLICY.remaining()
        return T.KEY_OWNER_REMAINING.format(n=left) if left is not None else ""
    return T.KEY_NONE


with gr.Blocks(title=T.TITLE) as demo:
    gr.Markdown(f"# {T.TITLE}\n{T.SUBTITLE}")
    if DEMO:
        gr.Markdown(f"**{T.DEMO_BANNER}**")
    gr.Markdown(f"<small>{T.DISCLAIMER_FR}<br>{T.LOG_NOTICE}</small>")
    with gr.Accordion(T.DISCLAIMER_OTHER_TITLE, open=False):
        gr.Markdown(f'<div dir="rtl">{T.DISCLAIMER_AR}</div>\n\n{T.DISCLAIMER_EN}')
    chatbot = gr.Chatbot(type="messages", height=480, show_label=False, label="Conversation")
    box = gr.Textbox(placeholder=T.INPUT_PLACEHOLDER, max_length=1000, show_label=False, autofocus=True)
    thread = gr.State()
    with gr.Row():
        new_btn = gr.Button(T.NEW_CONVERSATION)
    with gr.Accordion(T.KEY_LABEL, open=False):
        key_box = gr.Textbox(type="password", label=T.KEY_LABEL, info=T.KEY_HELP)
        key_note = gr.Markdown(key_status(""))
    box.submit(respond, [box, chatbot, thread, key_box], [chatbot, box, thread], api_name="ask")
    new_btn.click(new_conversation, [thread], [chatbot, thread], api_name=False)
    key_box.change(key_status, [key_box], [key_note], api_name=False)
    demo.load(lambda: key_status(""), None, [key_note], api_name=False)

demo.queue(default_concurrency_limit=2)

if not DEMO:
    threading.Thread(target=backend, daemon=True).start()          # load models and index while the page opens

if __name__ == "__main__":
    demo.launch()
