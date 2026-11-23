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


def code_name(source) -> str:
    stem = Path(str(source or "")).stem
    return T.CODE_NAMES.get(stem, stem or "?")


def citation_label(cid: str, evidence: dict) -> str:
    """'Code pénal, art. 220-bis': the article label comes from the chunk id, as in agent/answer.py."""
    source = (evidence.get(cid) or {}).get("source") or cid.split("/")[0]
    return f"{code_name(source)}, {T.ARTICLE} {cid.split('/')[-1]}"


def citation(cid: str, evidence: dict) -> dict:
    text = (evidence.get(cid) or {}).get("text")
    return {"id": cid, "label": citation_label(cid, evidence), "text": text or T.CITATION_UNAVAILABLE,
            "available": bool(text), "rtl": is_rtl(text or "")}


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
            "notes": _degraded_notes(degraded), "rewritten": None, "dropped": 0, "caveat": None, "claims": [],
            "text": state.get("text") or ""}
    if state.get("condensed_from") and state.get("question"):
        view["rewritten"] = state["question"]
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
