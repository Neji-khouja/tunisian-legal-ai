"""Gradio web interface (Hugging Face Space entry point). Local: python app.py   (RAG_UI_DEMO=1: canned answers, no models).

Left sidebar: new conversation, mode, conversations (kept in the visitor's browser), tokens left, personal key.
Tabs: chat, how it works. Every question and answer is also written to the SQLite log (ui/store.py).
API: /answer, for programs such as a voice assistant, refused unless the secret API_KEY is set and sent (see README).
"""
import os
import sys
import hmac
import time
import copy
import uuid
import threading
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ui"))
import texts as T
import chat
import keys
import quota
import voice
from store import Store, load_all
from stub import DemoBackend

try:
    import spaces                                   # present on ZeroGPU Spaces, which refuse to start without one @spaces.GPU function

    @spaces.GPU
    def _gpu_not_used():
        return 0
except ImportError:
    pass

env = lambda name, default="": os.environ.get(name, default).strip()
DEMO = env("RAG_UI_DEMO").lower() in ("1", "true", "yes", "on")
STORE = Store()
if DEMO:
    POLICY = keys.KeyPolicy(True, int(env("UI_OWNER_DAILY_CAP", "0") or 0), lambda: True)
else:
    POLICY = keys.KeyPolicy.from_env()
BUDGET = quota.TokenBudget(int(env("UI_TOKENS_PER_DAY", "200000") or 200000))
try:                                                # after a restart: what the app's key used in the last 24 hours
    BUDGET.seed(STORE.rows() + (load_all(env("RAG_LOG_BACKUP_DIR")) if env("RAG_LOG_BACKUP_DIR") else []))
except Exception as e:
    print("token budget not restored:", type(e).__name__)
VISITORS = quota.VisitorLimit(int(env("UI_VISITOR_HOURLY_CAP", "10") or 0))
API_KEY = env("API_KEY")                            # the API is closed while this secret is not set
API_POLICY = keys.KeyPolicy(POLICY.owner_enabled, int(env("API_DAILY_CAP", "0") or 0), POLICY._owner_present)
API_TURNS: dict = {}                                # conversation id -> questions asked through the API
_BACKEND = {}
_LOCK = threading.Lock()
CACHE = None if env("UI_ANSWER_CACHE", "on").lower() in ("off", "0", "false", "no") else chat.AnswerCache()
NEEDS_KEY = ("no_key", "owner_cap", "quota_daily", "visitor_cap")   # errors after which the key box opens
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
CLOSE_ON_PHONE = """() => {
  if (window.innerWidth >= 768) return;
  let tries = 0;                                    // the sidebar may be drawn after this runs: wait for it (3 s at most)
  const t = setInterval(() => {
    const b = document.querySelector('.sidebar.open .toggle-button');
    if (b) { b.click(); clearInterval(t); } else if (++tries > 30) clearInterval(t);
  }, 100);
}"""
CSS = (ROOT / "ui" / "style.css").read_text(encoding="utf-8")
THEME = gr.themes.Base(primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"], radius_size="md")
fmt = lambda n: f"{n:,}".replace(",", " ")          # a narrow space that never breaks a number


def backend():
    with _LOCK:
        if "b" not in _BACKEND:
            if DEMO:
                _BACKEND["b"] = DemoBackend()
            else:
                from backend import Backend
                _BACKEND["b"] = Backend()
        return _BACKEND["b"]


def visitor_of(request) -> str:
    """Who is asking, for the hourly limit: the forwarded address chain and the connection address."""
    if request is None:
        return "?"
    host = getattr(getattr(request, "client", None), "host", "") or ""
    return f"{request.headers.get('x-forwarded-for', '')}|{host}"


# ---- conversations (a list of dicts kept in the browser: id, title, mode, ts, messages, tokens)

def find(convs, conv_id):
    return next((c for c in convs if c["id"] == conv_id), None)


def used_line(convs, conv_id):
    here, total = (find(convs, conv_id) or {}).get("tokens", 0), sum(c.get("tokens", 0) for c in convs)
    return f'<div class="used">{T.USED_LINE.format(conv=fmt(here), device=fmt(total))}</div>' if total else ""


def owner_empty() -> bool:
    return POLICY.remaining() == 0 or BUDGET.left(None) == 0


def quota_card(personal_key):
    """Whose key the next question uses, and what is left on it."""
    personal = bool((personal_key or "").strip())
    if not personal and (not POLICY.owner_usable() or owner_empty()):
        text = T.QUOTA_NONE if not POLICY.owner_usable() else T.QUOTA_EMPTY
        return f'<div class="meter"><div class="meter-head"><span>{T.QUOTA_OWNER}</span></div><div class="meter-sub warn-text">{text}</div></div>'
    left, limit = BUDGET.left(personal_key), BUDGET.limit
    share = round(100 * left / limit)
    level = "low" if share < 15 else "mid" if share < 40 else "ok"
    lines = [T.QUOTA_ABOUT.format(agent=left // 10000, fast=left // 2000)]
    asked = None if personal else POLICY.remaining()
    if asked is not None:
        lines.insert(0, T.QUOTA_QUESTIONS.format(n=asked))
    return (f'<div class="meter"><div class="meter-head"><span>{T.QUOTA_PERSONAL if personal else T.QUOTA_OWNER}</span>'
            f'</div><div class="meter-num">{T.QUOTA_LEFT.format(n=f"<b>{fmt(left)}</b>")}</div><div class="bar {level}"><i style="width:{share}%"></i></div>'
            + "".join(f'<div class="meter-sub">{x}</div>' for x in lines) + "</div>")


def side(convs, conv_id):
    """The sidebar for a list of conversations: (history, empty note, delete button, tokens used)."""
    here = find(convs, conv_id)
    items = [(c["title"], c["id"]) for c in sorted(convs, key=lambda c: -c["ts"])]
    return (gr.update(choices=items, value=conv_id if here else None, visible=bool(items)),
            gr.update(visible=not items), gr.update(visible=here is not None), used_line(convs, conv_id))


def page(messages):
    """(chatbot, welcome block): an empty conversation shows the welcome block instead of an empty chat area."""
    return gr.update(value=messages, visible=bool(messages)), gr.update(visible=not messages)


def respond(question, messages, conv_id, convs, mode, personal_key, request: gr.Request):
    question = (question or "").strip()[:1000]
    messages = messages or []
    convs = copy.deepcopy(convs or [])
    if not question:
        yield (*page(messages), gr.skip(), conv_id, convs, *side(convs, conv_id), gr.skip(), gr.skip())
        return
    conv = find(convs, conv_id)
    if conv is None:
        conv = {"id": str(uuid.uuid4()), "title": question[:60], "mode": mode, "ts": time.time(), "messages": [], "tokens": 0}
        convs.append(conv)
    conv_id = conv["id"]
    turn = sum(m["role"] == "user" for m in messages) + 1
    messages = messages + [{"role": "user", "content": question}, {"role": "assistant", "content": "…"}]
    conv.update(messages=messages, ts=time.time(), mode=mode)
    yield (*page(messages), "", conv_id, convs, *side(convs, conv_id), gr.skip(), gr.skip())
    time.sleep(PAINT)                               # an instant result sent right behind this one was lost by the page
    steps = []
    for kind, value in chat.ask(question, conv_id, turn, personal_key or "", backend(), POLICY, STORE, mode, CACHE, BUDGET,
                                VISITORS, visitor_of(request)):
        done = kind == "done"
        if not done:
            steps.append(value)
            messages[-1] = {"role": "assistant", "content": f'<div class="steps">{" → ".join(steps[-3:])}…</div>'}
        else:
            messages[-1] = {"role": "assistant", "content": value["markdown"]}
            conv["tokens"] += value["tokens_in"] + value["tokens_out"]
            conv["cached"] = conv.get("cached", 0) + value["tokens_cached"]
            convs.sort(key=lambda c: -c["ts"])
            del convs[MAX_CONVERSATIONS:]
        yield (*page(messages), gr.skip(), conv_id, convs, *side(convs, conv_id),      # skip: keep what is being typed
               quota_card(personal_key) if done else gr.skip(),
               gr.update(open=True) if done and value["error"] in NEEDS_KEY else gr.skip())


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
    return (*side(convs or [], None), key_status(""), quota_card(""), gr.update(open=not POLICY.owner_usable() or owner_empty()))


def key_status(personal_key):
    if (personal_key or "").strip():
        return T.KEY_PERSONAL_ACTIVE
    return f"{T.KEY_HELP} {T.KEY_GET}"


def mode_help(mode):
    return T.MODE_HELP.get(mode, "")


STEP_NAMES = {label: name for name, label in T.NODE_LABELS.items()}     # chat.ask gives the label: the API also sends the name


def api_answer(question: str, mode: str = "agent", conversation_id: str = "", groq_key: str = "",
               request: gr.Request = None) -> Iterator[dict[str, Any]]:
    """For programs (a voice assistant): send the header X-API-Key with the Space secret API_KEY.
    mode: "agent" or "fast". conversation_id: send back the one you received to ask a follow-up question.
    groq_key: optional, the caller's own Groq key. While it works, sends {"status": "working", "step", "label"} for each
    step (step: retrieve, answer, verify...; label: what the chat page shows). The last message is the answer: status,
    speech (plain text to read aloud), answer, sources, warnings, language, conversation_id, tokens."""
    given = (request.headers.get("x-api-key", "") if request is not None else "") or ""
    if not API_KEY:
        yield {"status": "error", "error": "api_disabled", "speech": "", "answer": ""}
        return
    if not hmac.compare_digest(given.encode("utf-8"), API_KEY.encode("utf-8")):
        yield {"status": "error", "error": "unauthorized", "speech": "", "answer": ""}
        return
    question = (question or "").strip()
    if not question or len(question) > 1000:
        yield {"status": "error", "error": "question must have 1 to 1000 characters", "speech": "", "answer": ""}
        return
    mode = "fast" if mode == "fast" else "agent"
    cid = conversation_id.strip()[:64] if conversation_id and conversation_id.strip() else str(uuid.uuid4())
    turn = API_TURNS.get(cid, 0) + 1
    API_TURNS[cid] = turn
    while len(API_TURNS) > 2000:
        API_TURNS.pop(next(iter(API_TURNS)))
    done = {}
    for kind, value in chat.ask(question, "api-" + cid, turn, groq_key or "", backend(), API_POLICY, STORE, mode, CACHE, BUDGET):
        if kind == "done":
            done = value
        else:
            yield {"status": "working", "step": STEP_NAMES.get(value, value), "label": value}
    yield voice.reply(done, cid, mode)


with gr.Blocks(title=T.TITLE) as demo:
    convs = gr.BrowserState([], storage_key="tunisian_legal_rag_conversations", secret=BROWSER_SECRET)
    conv_id = gr.State(None)

    with gr.Sidebar(open=True, width=290, elem_id="side"):
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
        token_meter = gr.HTML("", padding=False)
        with gr.Accordion(T.KEY_LABEL, open=False, elem_id="key") as key_acc:
            key_box = gr.Textbox(type="password", show_label=False, container=False, placeholder="gsk_…", max_length=200)
            key_note = gr.Markdown(key_status(""), elem_classes="side-note")

    with gr.Tabs(elem_id="tabs"):
        with gr.Tab(T.TAB_CHAT):
            if DEMO:
                gr.HTML(f'<div class="warn">{T.DEMO_BANNER}</div>', padding=False)
            with gr.Column(elem_id="welcome") as welcome:
                gr.HTML(f'<div class="hello"><h1>{T.TITLE}</h1><p>{T.SUBTITLE}</p></div>')
                with gr.Row(elem_id="examples"):
                    example_btns = [gr.Button(e, elem_classes="example", min_width=220) for e in EXAMPLES]
            chatbot = gr.Chatbot(height="66vh", show_label=False, buttons=["copy"], visible=False,
                                 elem_id="chat", group_consecutive_messages=False)
            with gr.Row(elem_id="ask"):
                box = gr.Textbox(placeholder=T.INPUT_PLACEHOLDER, max_length=1000, show_label=False, autofocus=True,
                                 scale=8, container=False)
                send = gr.Button(T.SEND, variant="primary", scale=1, min_width=96, elem_id="send")
            gr.HTML(f'<div class="foot"><p>{T.FOOT}</p><p dir="auto">{T.FOOT_OTHER}</p></div>', padding=False)
        with gr.Tab(T.TAB_HOW):
            with gr.Column(elem_id="about"):
                gr.Markdown(T.ABOUT_INTRO)
                gr.Markdown(T.ABOUT_STORY, elem_id="story")
                gr.Markdown(T.ABOUT_MODES)
                # a switch, not nested gr.Tabs: Gradio measures those while hidden and can move the second one into a menu
                about_mode = gr.Radio([(T.TAB_AGENT, "agent"), (T.TAB_FAST, "fast")], value="agent", show_label=False,
                                      container=False, elem_id="about-modes")
                about = []
                for text, svg, shown in ((T.ABOUT_AGENT, "flow_agent.svg", True), (T.ABOUT_FAST, "flow_rapide.svg", False)):
                    with gr.Column(visible=shown) as part:
                        gr.Markdown(text)
                        gr.HTML('<div class="flow-wrap">' + (ROOT / "ui" / svg).read_text(encoding="utf-8") + "</div>", padding=False)
                    about.append(part)
                gr.Markdown(T.ABOUT_TOKENS)
                gr.Markdown(T.ABOUT_DATA)
                gr.Markdown(f"### Avertissement\n\n{T.DISCLAIMER_FR}\n\n{T.LOG_NOTICE}\n\n<div dir=\"rtl\">{T.DISCLAIMER_AR}</div>\n\n{T.DISCLAIMER_EN}")

    rest = [history, history_empty, delete_btn, token_meter]
    ins, outs = [box, chatbot, conv_id, convs, mode, key_box], [chatbot, welcome, box, conv_id, convs, *rest, quota_meter, key_acc]
    # the page's own events are private: hidden from the API and not callable by API clients; programs use /answer
    box.submit(respond, ins, outs, api_visibility="private")
    send.click(respond, ins, outs, api_visibility="private")
    for b in example_btns:
        b.click(respond, [b, *ins[1:]], outs, api_visibility="private")
    new_btn.click(new_conversation, [convs], [chatbot, welcome, conv_id, *rest], api_visibility="private")
    history.input(open_conversation, [history, convs], [chatbot, welcome, conv_id, mode, *rest], api_visibility="private")
    delete_btn.click(delete_conversation, [conv_id, convs], [chatbot, welcome, conv_id, convs, *rest], api_visibility="private")
    mode.change(mode_help, [mode], [mode_note], api_visibility="private")
    about_mode.change(lambda m: (gr.update(visible=m == "agent"), gr.update(visible=m == "fast")), [about_mode], about, api_visibility="private")
    key_box.change(lambda k: (key_status(k), quota_card(k)), [key_box], [key_note, quota_meter], api_visibility="private")
    demo.load(start, [convs], [*rest, key_note, quota_meter, key_acc], api_visibility="private")
    demo.load(None, None, None, js=CLOSE_ON_PHONE, api_visibility="private")
    gr.api(api_answer, api_name="answer")

demo.queue(default_concurrency_limit=2)

if not DEMO:
    threading.Thread(target=backend, daemon=True).start()          # load models and index while the page opens

if __name__ == "__main__":
    demo.launch(theme=THEME, head=f"<style>{CSS}</style>")     # Gradio 6: theme and styles are given to launch()
