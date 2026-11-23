"""Framework-free chat logic shared by the web interface: runs one question (in one worker thread, so the session's Groq
key stays set for the whole run), logs it, and turns the result into Markdown."""
import html
import time
import queue
import logging
import threading

import texts as T
import render as R

LOG = logging.getLogger("rag_ui")


def view_markdown(view: dict) -> str:
    """A view (render.view_v2 / view_error) as Markdown with a little HTML (details for the cited articles)."""
    if view["kind"] == "error":
        return f"**{view['text']}**"
    out = []
    if view.get("rewritten"):
        out.append(f"*{T.REWRITTEN.format(text=view['rewritten'])}*")
    if view.get("unverified"):
        out.append(f"⚠️ **{T.UNVERIFIED_BANNER.strip('*')}**")
    out += [f"*{n}*" for n in view.get("notes") or []]
    if view["kind"] == "answer":
        for i, claim in enumerate(view["claims"], 1):
            block = [R.to_html(claim["text"], number=i)]
            for c in claim["citations"]:
                block.append(f"<details><summary>{html.escape(c['label'])}</summary>\n\n{R.to_html(c['text'])}\n\n</details>")
            out.append("\n".join(block))
    else:
        out.append(R.to_html(view["text"]))
        if view["kind"] == "clarify":
            out.append(f"*{T.CLARIFY_HINT}*")
    if view.get("dropped"):
        out.append(f"*{T.DROPPED.format(n=view['dropped'])}*")
    if view.get("caveat"):
        out.append(f"> {T.CAVEAT.format(text=view['caveat'])}")
    return "\n\n".join(out)


def ask(question: str, thread_id: str, turn: int, personal_key: str, backend, policy, store):
    """Generator of ("step", label) while the agent works, then ("done", markdown). Never raises; logs the question."""
    import keys
    started, final, error = time.time(), None, None
    key, refused = policy.resolve(personal_key)
    key_source = "personal" if key else "owner"
    if refused:
        error = refused
    else:
        events: queue.Queue = queue.Queue()

        def work():
            try:
                with keys.session_key(key):
                    for kind, value in backend.stream_v2(question, thread_id):
                        events.put((kind, value))
            except Exception as e:                  # technical detail to the console only
                LOG.error("question failed: %s", R.technical_detail(e))
                events.put(("error", R.classify_error(e)))
            events.put(("end", None))
        threading.Thread(target=work, daemon=True).start()
        while True:
            kind, value = events.get()
            if kind == "end":
                break
            if kind == "step":
                yield "step", R.node_label(value)
            elif kind == "final":
                final = value
            elif kind == "error":
                error = value
    view = R.view_error(error) if error else R.view_v2(final or {})
    try:
        store.log(thread_id, turn, question, final, error, time.time() - started, key_source)
    except Exception as e:                          # a logging problem must never hide the answer
        LOG.error("logging failed: %s", type(e).__name__)
    yield "done", view_markdown(view)
