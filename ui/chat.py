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


def _esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def _source(label: str, text: str) -> str:
    return f'<details class="src"><summary>{html.escape(label)}</summary>\n\n{R.to_html(text)}\n\n</details>'


def view_markdown(view: dict) -> str:
    """A view (render.view_v2 / view_fast / view_error) as Markdown with a little HTML: notices, the answer, then its
    sources. Everything written by the user or the model is escaped."""
    if view["kind"] == "error":
        return f"**{view['text']}**"
    word = T.WORDS.get(view.get("language"), T.WORDS["fr"])
    out = []
    if view.get("rewritten"):
        out.append(f'<div class="note">{_esc(T.REWRITTEN.format(text=view["rewritten"]))}</div>')
    if view.get("unverified"):
        out.append(f'<div class="warn">{_esc(T.UNVERIFIED_BANNER.replace("*", ""))}</div>')
    out += [f'<div class="note">{_esc(n)}</div>' for n in view.get("notes") or []]
    sources = []
    if view["kind"] == "fast":
        sec = view["sections"]
        parts = [sec.get("intro"), sec.get("direct"), sec.get("explanation")]
        if sec.get("articles"):
            parts.append(f"**{word['articles']}**\n\n{sec['articles']}")
        body = "\n\n".join(p for p in parts if p)
        out.append(R.md_block(body or view["text"]))
        sources = [_source(c["label"], c["text"]) for c in view["sources"]]
        if view["others"]:
            rest = "\n".join(_source(c["label"], c["text"]) for c in view["others"])
            sources.append(f'<details class="src"><summary>{_esc(T.OTHER_SOURCES.format(n=len(view["others"])))}</summary>\n\n{rest}\n\n</details>')
    elif view["kind"] == "answer":
        order = {}                                   # citation id -> number, in order of first use
        for claim in view["claims"]:
            for c in claim["citations"]:
                order.setdefault(c["id"], (len(order) + 1, c))
        for claim in view["claims"]:
            refs = "".join(f"<sup>[{order[c['id']][0]}]</sup>" for c in claim["citations"])
            rtl = R.is_rtl(claim["text"])
            out.append(f'<div class="claim" dir="{"rtl" if rtl else "ltr"}" style="text-align:{"right" if rtl else "left"}">'
                       f'{_esc(claim["text"])} {refs}</div>')
        sources = [_source(f"[{n}] {c['label']}", c["text"]) for n, c in order.values()]
    else:
        out.append(R.to_html(view["text"]))
        if view["kind"] == "clarify":
            out.append(f'<div class="note">{_esc(T.CLARIFY_HINT)}</div>')
    if view.get("caveat"):
        rtl = R.is_rtl(view["caveat"])
        out.append(f'<div class="note" dir="{"rtl" if rtl else "ltr"}">{word["note"]} : {_esc(view["caveat"])}</div>')
    if view.get("dropped"):
        out.append(f'<div class="note">{_esc(T.DROPPED.format(n=view["dropped"]))}</div>')
    if sources:
        out.append(f'<div class="src-title">{word["sources"]}</div>')
        out += sources
    return "\n\n".join(out)


def usage_line(tokens_in: int, tokens_out: int, seconds: float, mode: str) -> str:
    f = lambda n: f"{n:,}".replace(",", " ")
    text = T.TOKENS_LINE.format(total=f(tokens_in + tokens_out), tin=f(tokens_in), tout=f(tokens_out), sec=round(seconds),
                                mode=T.MODE_NAMES.get(mode, mode))
    return f'<div class="usage">{text}</div>'


def ask(question: str, thread_id: str, turn: int, personal_key: str, backend, policy, store, mode: str = "agent"):
    """Generator of ("step", label) while the backend works, then ("done", {"markdown", "tokens_in", "tokens_out"}).
    Never raises; logs the question. mode: "agent" (stream_v2) or "fast" (stream_fast)."""
    import keys
    from langchain_core.callbacks import get_usage_metadata_callback
    mode = "fast" if mode == "fast" else "agent"
    started, final, error, used = time.time(), None, None, [0, 0]
    key, refused = policy.resolve(personal_key)
    key_source = "personal" if key else "owner"
    if refused:
        error = refused
    else:
        events: queue.Queue = queue.Queue()

        def work():
            try:
                with keys.session_key(key), get_usage_metadata_callback() as cb:
                    try:
                        run = backend.stream_fast if mode == "fast" else backend.stream_v2
                        for kind, value in run(question, thread_id):
                            events.put((kind, value))
                    finally:                        # calls made before a failure were paid for too
                        for u in cb.usage_metadata.values():
                            events.put(("usage", (u.get("input_tokens", 0), u.get("output_tokens", 0))))
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
            elif kind == "usage":
                used[0] += value[0]
                used[1] += value[1]
            elif kind == "error":
                error = value
    seconds = time.time() - started
    view = R.view_error(error) if error else (R.view_fast(final or {}) if mode == "fast" else R.view_v2(final or {}))
    try:
        store.log(thread_id, turn, question, final, error, seconds, key_source, mode, tuple(used))
    except Exception as e:                          # a logging problem must never hide the answer
        LOG.error("logging failed: %s", type(e).__name__)
    md = view_markdown(view)
    if not error and (used[0] or used[1]):
        md += "\n\n" + usage_line(used[0], used[1], seconds, mode)
    yield "done", {"markdown": md, "tokens_in": used[0], "tokens_out": used[1]}
