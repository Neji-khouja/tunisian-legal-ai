"""Framework-free chat logic shared by the web interface: runs one question (in one worker thread, so the session's Groq
key stays set for the whole run), logs it, and turns the result into Markdown."""
import html
import time
import queue
import logging
import threading
import unicodedata

import laws
import quota
import texts as T
import render as R

LOG = logging.getLogger("rag_ui")


def _esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def _source(label: str, c: dict) -> str:
    label += f" ({c['note']})" if c.get("note") else ""
    return f'<details class="src"><summary>{html.escape(label)}</summary>\n\n{R.to_html(c["text"])}\n\n</details>'


def _law(cited: list[dict], lang: str) -> list[str]:
    """The sentence naming the law and its date, and a warning when a cited article is marked as repealed."""
    if not cited:
        return []
    texts = {c["id"]: c["text"] for c in cited}
    rtl = lang == "ar"
    out = [f'<div class="law" dir="{"rtl" if rtl else "ltr"}" style="text-align:{"right" if rtl else "left"}">'
           f'{_esc(laws.sentence([c["id"] for c in cited], texts, lang))}</div>']
    out += [f'<div class="warn">{_esc(T.REPEALED.format(label=c["label"], note=c["note"]))}</div>' for c in cited if c.get("repealed")]
    return out


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
    sources, others = [], None
    if view["kind"] == "fast":
        sec = view["sections"]
        parts = [sec.get("intro"), sec.get("direct"), sec.get("explanation")]
        if any(ch.isdigit() for ch in sec.get("articles", "")):         # not "None." / "Aucun."
            parts.append(f"**{word['articles']}**\n\n{sec['articles']}")
        body = "\n\n".join(p for p in parts if p)
        out.append(R.md_block(body or view["text"]))
        out += _law(view["sources"], view["language"])
        sources = [_source(c["label"], c) for c in view["sources"]]
        if view["others"]:
            rest = "\n".join(_source(c["label"], c) for c in view["others"])
            others = f'<details class="src"><summary>{_esc(word["others"].format(n=len(view["others"])))}</summary>\n\n{rest}\n\n</details>'
    elif view["kind"] == "answer":
        order = {}                                   # citation id -> number, in order of first use
        for claim in view["claims"]:
            for c in claim["citations"]:
                order.setdefault(c["id"], (len(order) + 1, c))
        rtl = R.is_rtl(" ".join(c["text"] for c in view["claims"]))
        attrs = f'dir="{"rtl" if rtl else "ltr"}" style="text-align:{"right" if rtl else "left"}"'
        said = [_esc(c["text"]) + "".join(f"<sup>[{order[x['id']][0]}]</sup>" for x in c["citations"]) for c in view["claims"]]
        if len(said) <= 4:                           # a short paragraph; a longer answer (details asked) as a list
            out.append(f'<div class="claim" {attrs}>{" ".join(said)}</div>')
        else:
            out.append(f'<ul class="claim" {attrs}>' + "".join(f"<li>{x}</li>" for x in said) + "</ul>")
        out += _law([c for _, c in order.values()], view.get("language") or "fr")
        sources = [_source(f"[{n}] {c['label']}", c) for n, c in order.values()]
    else:
        out.append(R.to_html(view["text"]))
        if view["kind"] == "clarify":
            out.append(f'<div class="note">{_esc(T.CLARIFY_HINT)}</div>')
        if view.get("others"):
            rest = "\n".join(_source(c["label"], c) for c in view["others"])
            others = f'<details class="src"><summary>{_esc(word["others"].format(n=len(view["others"])))}</summary>\n\n{rest}\n\n</details>'
    if view.get("caveat"):
        rtl = R.is_rtl(view["caveat"])
        out.append(f'<div class="note" dir="{"rtl" if rtl else "ltr"}">{word["note"]} : {_esc(view["caveat"])}</div>')
    if view.get("dropped"):
        out.append(f'<div class="note">{_esc(T.DROPPED.format(n=view["dropped"]))}</div>')
    if sources:
        out.append(f'<div class="src-title">{word["sources"]}</div>')
        out += sources
    if others:
        out.append(others)
    return "\n\n".join(out)


def usage_line(used: dict, seconds: float, mode: str) -> str:
    f = lambda n: f"{n:,}".replace(",", " ")
    text = T.TOKENS_LINE.format(
        total=f(used["in"] + used["out"]), sec=round(seconds), mode=T.MODE_NAMES.get(mode, mode),
        tin=f(used["in"]) + (T.TOKENS_CACHED.format(n=f(used["cached"])) if used["cached"] else ""),
        tout=f(used["out"]) + (T.TOKENS_REASONING.format(n=f(used["reasoning"])) if used["reasoning"] else ""))
    return f'<div class="usage">{text}</div>'


class AnswerCache:
    """Answers to first questions, kept in memory (gone at a restart, so never older than the running code). Only complete
    ones: nothing degraded, nothing dropped. The same question in a new conversation then costs no token."""
    KEEP = ("status", "mode", "language", "question", "full_question", "text", "claims", "caveat", "sources", "dropped", "degraded")

    def __init__(self, size: int = 300):
        self.size, self._d, self._lock = size, {}, threading.Lock()

    @staticmethod
    def key(question: str, mode: str) -> tuple:
        q = unicodedata.normalize("NFKC", question).lower().replace("’", "'")
        return mode, " ".join(q.split()).strip(" ?!.؟")

    def get(self, question: str, mode: str) -> dict | None:
        with self._lock:
            return self._d.get(self.key(question, mode))

    def put(self, question: str, mode: str, state: dict) -> bool:
        complete = (state.get("status") == "done" and not state.get("degraded") and not state.get("dropped")
                    and not state.get("is_followup") and not state.get("original_question"))
        if mode == "fast":
            view = R.view_fast(state)
            complete = complete and bool(view["sources"])
        else:
            complete = complete and bool(state.get("claims"))
        if not complete:
            return False
        kept = {k: state[k] for k in self.KEEP if k in state}
        cited = {i for c in state.get("claims") or [] for i in c.get("cited_ids") or []}
        kept["evidence"] = {i: e for i, e in (state.get("evidence") or {}).items() if i in cited}
        with self._lock:
            self._d[self.key(question, mode)] = kept
            while len(self._d) > self.size:
                self._d.pop(next(iter(self._d)))
        return True


def ask(question: str, thread_id: str, turn: int, personal_key: str, backend, policy, store, mode: str = "agent",
        cache: AnswerCache | None = None, budget: "quota.TokenBudget | None" = None):
    """Generator of ("step", label) while the backend works, then ("done", {"markdown", "tokens_in", "tokens_out",
    "tokens_cached"}). Never raises; logs the question. mode: "agent" (stream_v2) or "fast" (stream_fast). cache: used for
    the first question of a conversation only. budget: the tokens left on the key used."""
    import keys
    from langchain_core.callbacks import get_usage_metadata_callback
    mode = "fast" if mode == "fast" else "agent"
    started, final, error = time.time(), None, None
    used = {"in": 0, "out": 0, "cached": 0, "reasoning": 0}
    known = cache.get(question, mode) if cache and turn == 1 else None
    key, refused = (None, None) if known else policy.resolve(personal_key)
    key_source = "cache" if known else "personal" if key else "owner"
    if known:
        final = known
        try:
            backend.remember(thread_id, known)      # so that a follow-up question knows this answer
        except Exception as e:
            LOG.error("remember failed: %s", type(e).__name__)
    elif refused:
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
                            events.put(("usage", {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
                                                  "cached": (u.get("input_token_details") or {}).get("cache_read") or 0,
                                                  "reasoning": (u.get("output_token_details") or {}).get("reasoning") or 0}))
            except Exception as e:                  # technical detail to the console only
                LOG.error("question failed: %s", R.technical_detail(e))
                if budget:
                    budget.learn(key, R.hc.full_message(e))
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
                for k in used:
                    used[k] += value.get(k, 0)
            elif kind == "error":
                error = value
    seconds = time.time() - started
    if budget and not known:
        budget.add(key, quota.counted(used["in"], used["out"], used["cached"]))
    view = R.view_error(error) if error else (R.view_fast(final or {}) if mode == "fast" else R.view_v2(final or {}))
    try:
        store.log(thread_id, turn, question, final, error, seconds, key_source, mode, used)
    except Exception as e:                          # a logging problem must never hide the answer
        LOG.error("logging failed: %s", type(e).__name__)
    md = view_markdown(view)
    if known:                                       # said first, so that nobody takes it for a new model call
        md = f'<div class="memo">{T.FROM_CACHE}</div>\n\n' + md
    elif not error and (used["in"] or used["out"]):
        md += "\n\n" + usage_line(used, seconds, mode)
    if cache and not known and turn == 1 and final and not error:
        cache.put(question, mode, final)
    yield "done", {"markdown": md, "tokens_in": used["in"], "tokens_out": used["out"], "tokens_cached": used["cached"]}
