"""Query normalization used by agent/retrieve_g.py (config.NORMALIZE_QUERY)."""
import re
import json
import time
import hashlib

MODEL = "openai/gpt-oss-120b"       # config.LLM_MODEL
TEMPERATURE = 0.0
REASONING_EFFORT = "low"            # as config.TRANSLATE_REASONING_EFFORT
MAX_ATTEMPTS = 2                    # first call + at most one more
MAX_WORDS = 40
MAX_WAIT_S = 120                    # a per-minute 429 that asks for a longer wait is not waited for

NORMALIZE_PROMPT = """
You turn a user's question into ONE search query for a database of TUNISIAN legal codes, which are written in French.

The question is already in French. It may be misspelled, informal, use wrong or everyday words, or describe a
situation without naming the legal concept.

Write one precise French search query that uses the vocabulary of the Tunisian legal codes:
- When the user only describes a situation, name the offence, institution or legal concept it falls under.
- Correct the spelling.
- Keep every number, date, duration and amount exactly as the user wrote it.
- NEVER write an article number that the user did not write.
- NEVER add facts that are not in the question.
- At most 40 words.
- Do not answer the question.

Output only a JSON object, with no other text: {"legal_query": "..."}

Examples (invented, illustration only):
Question: "l'archi qui a dessiné nos plans et l'entrepreneur ont fini les travaux, et 3 ans aprés le plafond s'est fissuré, on peut se retourner contre eux ?"
Output: {"legal_query": "garantie de l'architecte et de l'entrepreneur pour les vices de construction de l'édifice apparus 3 ans après la fin des travaux"}
Question: "un temoin a raconté des mensonges a l'audiance pour m'enfoncer, il risque quoi ?"
Output: {"legal_query": "sanction du faux témoignage à l'audience"}
Question: "le gars d'à coté a planté des arbres colés au mur entre nos jardins, il a le droit ?"
Output: {"legal_query": "distance légale des plantations d'arbres par rapport à la limite séparative des fonds"}
"""

PROMPT_SHA256 = hashlib.sha256(NORMALIZE_PROMPT.encode("utf-8")).hexdigest()

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set:
    """Digit sequences of a text ("1.000" and "1000" are the same number; "2,5" keeps its decimal part)."""
    out = set()
    for m in _NUM_RE.findall(text or ""):
        m = re.sub(r"[.,](?=\d{3}(?:\D|$))", "", m)          # thousands separators
        out.add(m.replace(",", "."))
    return out


def new_numbers(input_query: str, legal_query: str) -> list:
    """Numbers in legal_query that are not in input_query (flagged, not a failure)."""
    return sorted(numbers(legal_query) - numbers(input_query))


def parse_output(text: str) -> tuple:
    """-> (legal_query or None, error or None). Valid = a JSON object whose "legal_query" is a non-empty string."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    if not t:
        return None, "empty output"
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e.msg})"
    if not isinstance(obj, dict) or "legal_query" not in obj:
        return None, "no legal_query key"
    q = obj["legal_query"]
    if not isinstance(q, str) or not q.strip():
        return None, "empty legal_query"
    return " ".join(q.split()), None


def _stated_wait(msg: str):
    m = re.search(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?", msg)
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def _is_rate_limit(msg: str) -> bool:
    m = msg.lower()
    return "429" in m or "rate limit" in m or "rate_limit" in m


_LLM = {"llm": None}


def get_llm():
    """ChatGroq, temperature 0, JSON mode, no client-side retries (the attempts are counted here)."""
    if _LLM["llm"] is None:
        from langchain_groq import ChatGroq
        from config import GROQ_API_KEY
        _LLM["llm"] = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=TEMPERATURE, max_retries=0,
                               reasoning_effort=REASONING_EFFORT).bind(response_format={"type": "json_object"})
    return _LLM["llm"]


def _usage(msg) -> dict:
    u = getattr(msg, "usage_metadata", None) or {}
    return {k: int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "total_tokens")}


def normalize(question: str, llm=None, fatal=lambda e: False, sleep=time.sleep) -> dict:
    """At most MAX_ATTEMPTS calls. -> {"legal_query" (None on failure), "status" ok | invalid | api_error, "error",
    "attempts", "tokens", "raw"}. fatal(e) True (e.g. a daily-quota 429) re-raises at once."""
    llm = llm or get_llm()
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    status, error, raw = "api_error", None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            msg = llm.invoke([("system", NORMALIZE_PROMPT), ("human", question)])
        except Exception as e:
            if fatal(e):
                raise
            status, error = "api_error", f"{type(e).__name__}: {str(e)[:300]}"
            wait = _stated_wait(str(e)) if _is_rate_limit(str(e)) else None
            if attempt < MAX_ATTEMPTS and wait is not None:
                if wait + 1 > MAX_WAIT_S:
                    break
                sleep(wait + 1)
            continue
        for k, v in _usage(msg).items():
            tokens[k] += v
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)
        q, err = parse_output(raw)
        if q is not None:
            return {"legal_query": q, "status": "ok", "error": None, "attempts": attempt, "tokens": tokens, "raw": raw}
        status, error = "invalid", err
    return {"legal_query": None, "status": status, "error": error, "attempts": attempt, "tokens": tokens, "raw": raw}


def normalize_for_retrieval(query: str, llm=None) -> tuple:
    """For the agent (config.NORMALIZE_QUERY = "v3"). Never raises. -> (query to use, record): the legal_query when
    the normalization succeeded, else the input query (record["fell_back"] True: failure, empty or invalid output,
    API error, or any other exception)."""
    try:
        r = normalize(query, llm=llm)
    except Exception as e:                       # e.g. the client cannot be built: fall back, never block the turn
        r = {"legal_query": None, "status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}", "attempts": 0,
             "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "raw": None}
    ok = r["status"] == "ok" and bool((r["legal_query"] or "").strip())
    rec = {"input_query": query, "legal_query": r["legal_query"], "status": r["status"], "error": r["error"],
           "attempts": r["attempts"], "tokens": r["tokens"], "prompt_sha256": PROMPT_SHA256, "fell_back": not ok}
    return (r["legal_query"] if ok else query), rec
