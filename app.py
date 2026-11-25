"""Gradio web interface (Hugging Face Space entry point). Local: python app.py   (RAG_UI_DEMO=1: canned answers, no models).

Left sidebar: mode, conversations (kept in the visitor's browser), token meter, personal key. Tabs: chat, how it works.
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
import quota
from store import Store, load_all
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
BUDGET = quota.TokenBudget(int(os.environ.get("UI_TOKENS_PER_DAY", "200000") or 200000))
try:                                                # after a restart: what the app's key used in the last 24 hours
    BUDGET.seed(STORE.rows() + (load_all(os.environ["RAG_LOG_BACKUP_DIR"]) if os.environ.get("RAG_LOG_BACKUP_DIR") else []))
except Exception as e:
    print("token budget not restored:", type(e).__name__)
_BACKEND = {}
_LOCK = threading.Lock()
CACHE = None if os.environ.get("UI_ANSWER_CACHE", "on").strip().lower() in ("off", "0", "false", "no") else chat.AnswerCache()
PAINT = 0.4                                         # seconds
MAX_CONVERSATIONS = 40                              # kept in the browser, oldest dropped
# Gradio encrypts browser storage with this. It only protects a visitor's own list on their own machine, and it must
# stay the same: with the default (random at each start) every restart would erase everyone's conversations.
BROWSER_SECRET = "tunisian-legal-rag-v1"
EXAMPLES = ["Quelle est la peine pour le vol simple ?",
            "Quelle est la durée de la période d'essai dans un contrat de travail ?",
            "ما هي شروط صحة العقد؟",
            "What are the conditions for a valid marriage contract?"]
# on a phone the open sidebar covers the page: start closed
CLOSE_ON_PHONE = """() => { if (window.innerWidth < 768) document.querySelector('.sidebar.open .toggle-button')?.click(); }"""
CSS = (ROOT / "ui" / "style.css").read_text(encoding="utf-8")
THEME = gr.themes.Base(primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"], radius_size="md")


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


def meter(convs, conv_id):
    f = lambda n: f"{n:,}".replace(",", " ")
    rows = [(T.TOKENS_CONV, (find(convs, conv_id) or {}).get("tokens", 0)),
            (T.TOKENS_DEVICE, sum(c.get("tokens", 0) for c in convs))]
    cached = sum(c.get("cached", 0) for c in convs)
    if cached:
        rows.append((T.TOKENS_HIT, cached))
    return ('<div class="meter"><div class="meter-title">' + T.TOKENS_TITLE + "</div>"
            + "".join(f'<div class="meter-row"><span>{k}</span><b>{f(v)}</b></div>' for k, v in rows) + "</div>")


def quota_card(personal_key):
    """Tokens left on the key that would be used for the next question, and whose key it is."""
    f = lambda n: f"{n:,}".replace(",", " ")
    personal = bool((personal_key or "").strip())
    if not personal and not POLICY.owner_usable():
        return f'<div class="meter"><div class="meter-title">{T.QUOTA_TITLE}</div><div class="meter-sub warn-text">{T.QUOTA_NONE}</div></div>'
    left, limit = BUDGET.left(personal_key), BUDGET.limit
    share = round(100 * left / limit)
    level = "low" if share < 15 else "mid" if share < 40 else "ok"
    lines = [T.QUOTA_PERSONAL if personal else T.QUOTA_OWNER, T.QUOTA_ABOUT.format(agent=left // 11000, fast=left // 2000)]
    asked = None if personal else POLICY.remaining()
    if asked is not None:
        lines.append(T.QUOTA_QUESTIONS.format(n=asked))
    lines.append(T.QUOTA_HINT)
    return (f'<div class="meter"><div class="meter-title">{T.QUOTA_TITLE}</div>'
            f'<div class="meter-big"><b>{f(left)}</b> / {f(limit)}</div><div class="bar {level}"><i style="width:{share}%"></i></div>'
            + "".join(f'<div class="meter-sub">{x}</div>' for x in lines) + "</div>")


def side(convs, conv_id):
    """The sidebar for a list of conversations: (history, empty note, delete button, token meter)."""
    here = find(convs, conv_id)
    items = [(c["title"], c["id"]) for c in sorted(convs, key=lambda c: -c["ts"])]
    return (gr.update(choices=items, value=conv_id if here else None, visible=bool(items)),
            gr.update(visible=not items), gr.update(visible=here is not None), meter(convs, conv_id))


def page(messages):
    """(chatbot, welcome block): an empty conversation shows the welcome block instead of an empty chat area."""
    return gr.update(value=messages, visible=bool(messages)), gr.update(visible=not messages)


def respond(question, messages, conv_id, convs, mode, personal_key):
    question = (question or "").strip()
    messages = messages or []
    convs = copy.deepcopy(convs or [])
    if not question:
        yield (*page(messages), gr.skip(), conv_id, convs, *side(convs, conv_id), gr.skip())
        return
    conv = find(convs, conv_id)
    if conv is None:
        conv = {"id": str(uuid.uuid4()), "title": question[:60], "mode": mode, "ts": time.time(), "messages": [], "tokens": 0}
        convs.append(conv)
    conv_id = conv["id"]
    turn = sum(m["role"] == "user" for m in messages) + 1
    messages = messages + [{"role": "user", "content": question}, {"role": "assistant", "content": "…"}]
    conv.update(messages=messages, ts=time.time(), mode=mode)
    yield (*page(messages), "", conv_id, convs, *side(convs, conv_id), gr.skip())
    time.sleep(PAINT)                               # an instant result sent right behind this one was lost by the page
    steps = []
    for kind, value in chat.ask(question, conv_id, turn, personal_key or "", backend(), POLICY, STORE, mode, CACHE, BUDGET):
        if kind == "step":
            steps.append(value)
            messages[-1] = {"role": "assistant", "content": f'<div class="steps">{" → ".join(steps[-3:])}…</div>'}
        else:
            messages[-1] = {"role": "assistant", "content": value["markdown"]}
            conv["tokens"] += value["tokens_in"] + value["tokens_out"]
            conv["cached"] = conv.get("cached", 0) + value["tokens_cached"]
            convs.sort(key=lambda c: -c["ts"])
            del convs[MAX_CONVERSATIONS:]
        yield (*page(messages), gr.skip(), conv_id, convs, *side(convs, conv_id),      # skip: keep what is being typed
               quota_card(personal_key) if kind == "done" else gr.skip())


def open_conversation(chosen, convs):
    convs = convs or []
    conv = find(convs, chosen)
    if conv is None:
        return (*page([]), None, gr.update(), *side(convs, None))
    return (*page(conv["messages"]), conv["id"], conv.get("mode", "agent"), *side(convs, conv["id"]))


def new_conversation(convs):
    return (*page([]), None, *side(convs or [], None))


def delete_conversation(conv_id, convs):
    convs = [c for c in (convs or []) if c["id"] != conv_id]
    if conv_id and "b" in _BACKEND:
        _BACKEND["b"].forget(conv_id)
    return (*page([]), None, convs, *side(convs, None))


def start(convs):
    return (*side(convs or [], None), key_status(""), quota_card(""))


def key_status(personal_key):
    if (personal_key or "").strip():
        return T.KEY_PERSONAL_ACTIVE
    if POLICY.owner_usable():
        left = POLICY.remaining()
        return T.KEY_OWNER_REMAINING.format(n=left) if left is not None else ""
    return T.KEY_NONE


def mode_help(mode):
    return T.MODE_HELP.get(mode, "")


with gr.Blocks(title=T.TITLE, theme=THEME, head=f"<style>{CSS}</style>") as demo:
    convs = gr.BrowserState([], storage_key="tunisian_legal_rag_conversations", secret=BROWSER_SECRET)
    conv_id = gr.State(None)

    with gr.Sidebar(open=True, width=300, elem_id="side"):
        gr.HTML(f'<div class="brand"><div class="brand-name">{T.TITLE}</div><div class="brand-sub">{T.BRAND_SUB}</div></div>', padding=False)
        new_btn = gr.Button(T.NEW_CHAT, elem_id="new-chat")
        gr.HTML(f'<div class="side-label">{T.MODE_LABEL}</div>', padding=False)
        mode = gr.Radio(T.MODE_CHOICES, value="agent", show_label=False, container=False, elem_id="mode")
        mode_note = gr.Markdown(mode_help("agent"), elem_classes="side-note")
        gr.HTML(f'<div class="side-label">{T.HISTORY_LABEL}</div>', padding=False)
        history_empty = gr.Markdown(T.HISTORY_EMPTY, elem_classes="side-note")
        history = gr.Radio(choices=[], value=None, show_label=False, container=False, visible=False, elem_id="history")
        delete_btn = gr.Button(T.DELETE_CHAT, size="sm", visible=False, elem_id="delete-chat")
        quota_meter = gr.HTML(quota_card(""), padding=False)
        token_meter = gr.HTML(meter([], None), padding=False)
        with gr.Accordion(T.KEY_LABEL, open=False, elem_id="key"):
            key_box = gr.Textbox(type="password", show_label=False, container=False, info=T.KEY_HELP)
            key_note = gr.Markdown(key_status(""), elem_classes="side-note")

    with gr.Tabs(elem_id="tabs"):
        with gr.Tab(T.TAB_CHAT):
            if DEMO:
                gr.HTML(f'<div class="warn">{T.DEMO_BANNER}</div>', padding=False)
            with gr.Column(elem_id="welcome") as welcome:
                gr.HTML(f'<div class="hello"><h1>{T.TITLE}</h1><p>{T.SUBTITLE}</p></div>')
                with gr.Row(elem_id="examples"):
                    example_btns = [gr.Button(e, elem_classes="example", min_width=240) for e in EXAMPLES]
            chatbot = gr.Chatbot(type="messages", height="60vh", show_label=False, show_copy_button=True, visible=False,
                                 elem_id="chat", group_consecutive_messages=False)
            with gr.Row(elem_id="ask"):
                box = gr.Textbox(placeholder=T.INPUT_PLACEHOLDER, max_length=1000, show_label=False, autofocus=True,
                                 scale=8, container=False)
                send = gr.Button(T.SEND, variant="primary", scale=1, min_width=96, elem_id="send")
            gr.Markdown(f"{T.DISCLAIMER_FR}\n\n{T.LOG_NOTICE}", elem_classes="disclaimer")
            with gr.Accordion(T.DISCLAIMER_OTHER_TITLE, open=True, elem_id="other-languages"):
                gr.Markdown(f'<div dir="rtl">{T.DISCLAIMER_AR}</div>\n\n{T.DISCLAIMER_EN}', elem_classes="disclaimer")
        with gr.Tab(T.TAB_HOW):
            with gr.Column(elem_id="about"):
                gr.Markdown(T.ABOUT_INTRO)
                gr.Markdown(T.ABOUT_MODES)
                with gr.Tabs(elem_id="about-modes"):
                    for tab, text, svg in ((T.TAB_AGENT, T.ABOUT_AGENT, "flow_agent.svg"), (T.TAB_FAST, T.ABOUT_FAST, "flow_rapide.svg")):
                        with gr.Tab(tab):
                            gr.Markdown(text)
                            gr.HTML('<div class="flow-wrap">' + (ROOT / "ui" / svg).read_text(encoding="utf-8") + "</div>", padding=False)
                gr.Markdown(T.ABOUT_TOKENS)
                gr.Markdown(T.ABOUT_DATA)

    rest = [history, history_empty, delete_btn, token_meter]
    ins, outs = [box, chatbot, conv_id, convs, mode, key_box], [chatbot, welcome, box, conv_id, convs, *rest, quota_meter]
    box.submit(respond, ins, outs, api_name="ask")
    send.click(respond, ins, outs, api_name=False)
    for b in example_btns:
        b.click(respond, [b, *ins[1:]], outs, api_name=False)
    new_btn.click(new_conversation, [convs], [chatbot, welcome, conv_id, *rest], api_name=False)
    history.input(open_conversation, [history, convs], [chatbot, welcome, conv_id, mode, *rest], api_name=False)
    delete_btn.click(delete_conversation, [conv_id, convs], [chatbot, welcome, conv_id, convs, *rest], api_name=False)
    mode.change(mode_help, [mode], [mode_note], api_name=False)
    key_box.change(lambda k: (key_status(k), quota_card(k)), [key_box], [key_note, quota_meter], api_name=False)
    demo.load(start, [convs], [*rest, key_note, quota_meter], api_name=False)
    demo.load(None, None, None, js=CLOSE_ON_PHONE)

demo.queue(default_concurrency_limit=2)

if not DEMO:
    threading.Thread(target=backend, daemon=True).start()          # load models and index while the page opens

if __name__ == "__main__":
    demo.launch()
