"""What the API sends back: a short plain text to read aloud, the full answer as plain text, and the sources. No HTML,
no Markdown, no [1] marks: a voice assistant reads `speech` as it is."""
import re
import html

import laws
import texts as T

STATUS = {"answer": "answer", "fast": "answer", "clarify": "clarify", "refuse": "refuse", "message": "not_found", "error": "error"}
SPOKEN_CLAIMS = 4                                    # sentences read aloud; the full answer has them all


def plain(text: str) -> str:
    """Text without HTML tags, Markdown marks, [n] references or repeated spaces."""
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"(\*\*|__|\*|`|^#+\s*|^\s*[-•]\s+)", "", t, flags=re.M)
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", t)).strip()


def _sources(view: dict) -> list[dict]:
    seen, out = set(), []
    cits = [c for claim in view.get("claims") or [] for c in claim["citations"]] + list(view.get("sources") or [])
    for c in cits:
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append({"id": c["id"], "label": c["label"], "note": c.get("note") or "", "repealed": bool(c.get("repealed"))})
    return out


def reply(done: dict, conversation_id: str, mode: str) -> dict:
    """done: the last value of chat.ask (it carries the view, the error and the tokens)."""
    view, lang = done["view"], (done["view"].get("language") or "fr")
    status = STATUS.get(view["kind"], "answer")
    sources = _sources(view)
    texts = {c["id"]: c["text"] for claim in view.get("claims") or [] for c in claim["citations"]}
    texts.update({c["id"]: c["text"] for c in view.get("sources") or []})
    law = laws.sentence([s["id"] for s in sources], texts, lang) if sources and status == "answer" else ""
    caveat = plain(view.get("caveat") or "")
    if view["kind"] == "answer":
        said = [plain(c["text"]) for c in view["claims"]]
        speech = " ".join(said[:SPOKEN_CLAIMS] + [x for x in (caveat, law) if x])
        full = "\n".join(f"- {s} ({', '.join(c['label'] for c in claim['citations'])})" for s, claim in zip(said, view["claims"]))
        full += "".join(f"\n{x}" for x in (caveat, law) if x)
    elif view["kind"] == "fast":
        sec = view.get("sections") or {}
        body = plain("\n".join(x for x in (sec.get("intro"), sec.get("direct"), sec.get("explanation")) if x) or view["text"])
        speech, full = " ".join(x for x in (body.replace("\n", " "), law) if x), "\n".join(x for x in (body, law) if x)
    else:
        speech = full = plain(view["text"])
    warnings = [T.REPEALED.format(label=s["label"], note=s["note"]) for s in sources if s["repealed"]]
    if view.get("unverified"):
        warnings.append(plain(T.UNVERIFIED_BANNER))
    return {"status": status, "speech": speech, "answer": full, "sources": sources, "warnings": warnings,
            "language": lang, "mode": mode, "conversation_id": conversation_id,
            "error": done.get("error"), "from_memory": bool(done.get("from_memory")),
            "tokens": int(done.get("tokens_in", 0)) + int(done.get("tokens_out", 0))}
