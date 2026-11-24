"""Pure helpers: turn an agent state (v2) or a pipeline result (v1.1) into a view the app displays, and an exception into
a friendly French message. No Streamlit, no model, no network: unit-tested with stub outputs."""
import re
import sys
import html
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import harness_core as hc
import laws
import texts as T

ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LETTER = re.compile(r"[^\W\d_]")


def is_rtl(text: str) -> bool:
    """True when most letters of the text are Arabic."""
    letters = LETTER.findall(text or "")
    return bool(letters) and sum(bool(ARABIC.match(c)) for c in letters) / len(letters) > 0.5


def to_html(text: str, number: int | None = None) -> str:
    """Escaped text in a block with the right direction (Arabic: right to left). A claim number is isolated (<bdi>) so
    that it stays at the start of the line in both directions."""
    rtl = is_rtl(text)
    body = html.escape(text or "").replace("\n", "<br>")
    if number is not None:
        body = f"<bdi>{number}.</bdi> {body}"
    return f'<div dir="{"rtl" if rtl else "ltr"}" style="text-align:{"right" if rtl else "left"}">{body}</div>'


def md_block(text: str) -> str:
    """Model text that is Markdown (the fast mode): HTML escaped, Markdown kept, right direction."""
    rtl = is_rtl(text)
    body = html.escape(text or "", quote=False)
    return f'<div dir="{"rtl" if rtl else "ltr"}" style="text-align:{"right" if rtl else "left"}">\n\n{body}\n\n</div>'


# section labels required by pipeline._SYSTEM_PROMPT, in the three languages (the model may keep one or all of them)
SECTIONS = {"direct": r"direct answer|réponse directe|الإجابة المباشرة", "articles": r"applicable articles|articles applicables|الفصول المنطبقة",
            "sources": r"sources?\s*\(s\)|sources?|المصادر|المصدر", "explanation": r"explanation|explication|شرح|الشرح"}
_HEAD = re.compile(r"^\s*(?:#+\s*)?\**\s*((?:%s)(?:\s*/\s*(?:%s))*)\s*:?\s*\**\s*:?\s*(.*)$"
                   % (("|".join(SECTIONS.values()),) * 2), re.I)


def split_sections(text: str) -> dict:
    """The fast mode's text cut at its section labels: {"direct", "articles", "sources", "explanation", "intro"}."""
    out, current = {"intro": []}, "intro"
    for line in (text or "").splitlines():
        m = _HEAD.match(line)
        if m and (":" in line or line.strip().startswith(("*", "#"))):
            current = next(k for k, pat in SECTIONS.items() if re.match(rf"(?:{pat})", m.group(1), re.I))
            out.setdefault(current, [])
            line = m.group(2)
        out[current].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items() if "\n".join(v).strip()}


_MENTION = re.compile(r"(?:\barticles?|\bart\.?|الفصول|الفصل|المواد|المادة)[\s*]*((?:\d+(?:-\w+)?[\s*]*(?:,|;|et|à|and|to|و|إلى)?[\s*]*)+)", re.I)


def mentioned(text: str) -> set:
    """Article labels written after an article marker in a text ("articles 260, 261 et 264" -> 260, 261, 264)."""
    return {n for m in _MENTION.finditer(text or "") for n in re.findall(r"\d+(?:-\w+)?", m.group(1))}


_LEAD = re.compile(r"^[\s*•-]*(?:articles?|art\.?|الفصل|المادة)[\s*]*(\d+(?:-\w+)?)", re.I | re.M)


def relied_on(sections: dict, text: str) -> set:
    """Article labels a fast answer rests on: named in the direct answer, or heading an item of its list of articles
    (not the ones a quoted article merely refers to). Without any, every article the text names."""
    return mentioned(sections.get("direct", "")) | set(_LEAD.findall(sections.get("articles", ""))) or mentioned(text)


def citation_label(cid: str) -> str:
    """'Code pénal, art. 220-bis': the law from the id's section (an annexed law is not the code), the article label from
    the id, as in agent/answer.py."""
    return f"{laws.law(cid)[0]}, {T.ARTICLE} {cid.split('/')[-1]}"


def citation(cid: str, evidence: dict) -> dict:
    """note: what the article says of its own amendment; repealed: it says it is repealed."""
    text = (evidence.get(cid) or {}).get("text")
    return {"id": cid, "label": citation_label(cid), "text": text or T.CITATION_UNAVAILABLE, "available": bool(text),
            "rtl": is_rtl(text or ""), "note": laws.note(text or ""), "repealed": laws.repealed(text or "")}


def _degraded_notes(degraded: list) -> list[str]:
    notes = []
    for d in degraded:
        kind = str(d).split(":")[0]
        if kind in T.FALLBACK_NOTES and T.FALLBACK_NOTES[kind] not in notes:
            notes.append(T.FALLBACK_NOTES[kind])
    if any("RateLimit" in str(d) or "429" in str(d) for d in degraded):
        notes.append(T.RATE_LIMIT_IN_STEP)
    return notes


def view_v2(state: dict) -> dict:
    """Agent final state -> view. kind: clarify | refuse | answer | message."""
    status = state.get("status")
    evidence = state.get("evidence") or {}
    degraded = list(state.get("degraded") or [])
    claims = state.get("claims") or []
    view = {"mode": "v2", "unverified": any(str(d).startswith("verify_failed") for d in degraded),
            "notes": _degraded_notes(degraded), "language": state.get("language") or "fr", "rewritten": None, "dropped": 0, "caveat": None, "claims": [],
            "text": state.get("text") or ""}
    raw, asked = (state.get("condensed_from") or "").strip(), (state.get("question") or "").strip()
    if raw and asked and raw != asked:               # the agent sets condensed_from on every later turn
        view["rewritten"] = asked
    if status == "clarify":
        return view | {"kind": "clarify", "unverified": False}
    if status == "refuse":
        return view | {"kind": "refuse", "unverified": False}
    view["dropped"] = int(state.get("dropped") or 0)
    view["caveat"] = state.get("caveat") or None
    if not claims:
        return view | {"kind": "message"}
    view["claims"] = [{"text": c.get("text", ""), "rtl": is_rtl(c.get("text", "")),
                       "citations": [citation(cid, evidence) for cid in c.get("cited_ids") or []]} for c in claims]
    return view | {"kind": "answer"}


def view_fast(state: dict) -> dict:
    """Result of the one-pass pipeline -> view: the model's text and the articles it was given."""
    sources = []
    for src in state.get("sources") or []:
        sources.append(citation(src["id"], {src["id"]: src}))
    text = state.get("text") or ""
    sections = split_sections(text)
    named = relied_on(sections, text)
    used = [x for x in sources if x["id"].split("/")[-1] in named]        # none named (no answer found): no source
    return {"mode": "fast", "kind": "fast", "text": text, "sections": sections, "sources": used,
            "others": [x for x in sources if x not in used], "unverified": False,
            "language": "ar" if is_rtl(text) else state.get("language") or "fr", "notes": [], "rewritten": None, "dropped": 0, "caveat": None, "claims": []}


def view_error(kind: str) -> dict:
    return {"mode": None, "kind": "error", "error": kind, "text": T.ERRORS.get(kind, T.ERRORS["other"])}


def classify_error(exc: BaseException) -> str:
    """quota_daily | rate_limit | auth | other, with the harness's rules (harness_core)."""
    msg = hc.full_message(exc)
    if hc.is_daily_quota(msg):
        return "quota_daily"
    if hc.is_rate_limit(msg):
        return "rate_limit"
    if "401" in msg or "AuthenticationError" in msg or "invalid api key" in msg.lower() or "invalid_api_key" in msg:
        return "auth"
    if "APIConnectionError" in msg or "APITimeoutError" in msg or hc.is_transient(msg):
        return "network"
    return "other"


def technical_detail(exc: BaseException) -> str:
    """For the console only: exception types and code locations, never the exception messages (a provider error can
    quote the question or the model output, which must not be logged)."""
    parts, seen, e = [], set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        frames = traceback.extract_tb(e.__traceback__)[-4:]
        where = " < ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames))
        parts.append(f"{type(e).__name__} at {where or '?'}")
        e = e.__cause__ or e.__context__
    return " <- ".join(parts)


def node_label(name: str) -> str:
    return T.NODE_LABELS.get(name, name)
