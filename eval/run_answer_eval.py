"""Answer-level evaluation harness (v2 = agent graph, v1.1 = baseline pipeline, v1 = original pipeline).

    python eval/run_answer_eval.py --stage 1 --system all --v1-worktree ../RAG_PDF_v1 --dry-run   # plan, no call
    python eval/run_answer_eval.py --stage 1 --system all --v1-worktree ../RAG_PDF_v1 [--min-gap S] [--skip-judge]
           [--daily-budget-tokens N] [--wait-on-quota [MINUTES]] [--max-wait-hours H] [--allow-mixed] [--allow-changed-exam]
    python eval/run_answer_eval.py --stage 1 --system all --judge-only --judge-provider gemini   # judge saved answers later
    python eval/run_answer_eval.py --summarize [results/answer_eval_answer_set_stage1_v2.jsonl ...]   # no path: all stage files
    python eval/run_answer_eval.py --stage 1 --system v2 --export-audit 20   # CSV for a human check of the judge
    python eval/run_answer_eval.py --set eval/plumbing_set.json --system all --results-dir results/plumbing --dry-run

Run mechanics (eval/harness_core.py): question-major order (for each question: v1, v1.1, v2); one result file per stage
and system (results/answer_eval_answer_set_stage<k>_<system>.jsonl), rewritten atomically after each finished
(item, system); the same command resumes (finished pairs skipped, degraded ones redone). A 429 naming a per-day limit
(TPD/RPD) is QuotaExhausted: never retried, the pair in progress is discarded (not saved, not degraded), results/run_state.json
and results/run_log.jsonl are written, exit code 75. Any other exception of one (item, system) is tried again once in
the same run (v2: a new thread id on every call); a second failure saves a row with status "error" (exception type and
short message, attempt k) and the run goes on; resumes run "error" rows again, and the 3rd failed attempt (run) saves
status "failed": final, no answer, shown by --summarize as "system failure" and counted as incorrect in a
sensitivity line. --daily-budget-tokens stops (exit 75) before a pair that would pass
N agent-model tokens today (estimate: mean of the measured rows of that system, else DEFAULT_TOKENS). --wait-on-quota
sleeps and probes instead of stopping, for at most --max-wait-hours. Every row records date, provenance (git commit, tags,
agent model, prompt hashes, exam hashes, judge) and tokens; a file holding rows of another commit is refused unless
--allow-mixed; a changed exam (vs eval/exam_hashes.txt) is refused unless --allow-changed-exam. A judge quota stop
during a run leaves judge_pending rows for --judge-only.

Items (eval/answer_set.json, see eval/check_answer_set.py): id, question, language, expected (answer|clarify|refuse),
gold [{source, article}], reference_points [str], reason; a clarify item may add "reply" (a scripted user reply that
picks one reading): v2 gets the question, then the reply, on the SAME thread_id (first-turn status in first_status,
used for gate accuracy); v1.1 and v1 (single-turn) get one merged question "<question> (précision : <reply>)" in the
item's language (field "asked"). Results go to the
stage file above. Another --set (e.g. eval/plumbing_set.json, a 3-item pipe check): no exam fingerprint, --stage all
only, <results-dir>/answer_eval_<setname>_<system>.jsonl in its own --results-dir (results/ itself, or a folder holding
stage files, is refused; the exam is refused in a folder holding another set's files).
Timing per item: seconds (system only), elapsed_total (system + metrics + judge; the --min-gap sleep is outside),
n_429 (rate-limited Groq responses the client retried) and waited_seconds (sum of the retry waits the Groq client
announced); n_429 and waited_seconds are None for v1, whose LLM calls run in a subprocess.
v2 rows also keep the verifier loop: verdicts (last pass), first_verdicts (first pass), repairs, dropped,
draft_claims / draft_text (the first, unverified answer). A v2 item whose `degraded` list is not empty (a node fell back) is stored with status "degraded"
(graph status in graph_status), gets no metrics and no judge, and is run again on resume.

Judge: a model different from the generator (env JUDGE_MODEL or --judge-model), --judge-provider groq (Groq model id)
or gemini (langchain-google-genai, key from GOOGLE_API_KEY), temperature 0, own prompt (hash recorded), structured
output. Never guessed: without a judge model the run stops unless --skip-judge. Clarify and refuse items (every system)
get a second judge prompt (GATE_PROMPT, hash recorded): gives_substantive_answer and enumerates_alternatives, on the
first-turn response; --summarize reports for refuse items "no substantive answer" (explicit refusal OR no substantive
answer, the primary metric) and "explicit refusal status", for clarify items "asked for clarification (status)",
"answered directly" and "enumerated alternatives", and "gate accuracy (explicit status)" for both.
v1 needs a git worktree of tag v1-baseline-runnable with the old index (the script prints the setup commands).
"""
import os
import re
import sys
import csv
import json
import time
import random
import sqlite3
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "eval"))
from pdf_truth import from_chunk_id, pdf_article_text, source_stem, normalize
from pdf_factcheck import claim_numbers_ok, implicit_reference_numbers
import harness_core as hc

SEED = 20261022
N_BOOT = 5000
GENERATOR_MODEL = "openai/gpt-oss-120b"        # config.LLM_MODEL and agent/analyze.py
V1_TAG = "v1-baseline-runnable"
SYSTEM_ORDER = ["v1", "v1.1", "v2"]            # question-major order: for each question, every system in this order
TPD = 200_000                                  # Groq tokens per day on the agent model (projection only)
DEFAULT_TOKENS = {"v2": 15_000, "v1.1": 4_000, "v1": 5_000}   # guess per question until measured rows exist
EXAM_DIR = ROOT / "eval"
# successful Groq requests (n), rate-limited responses (n429) seen by httpx in this process, and the retry waits
# the Groq client announced ("Retrying request to ... in X seconds", logger groq._base_client; 429s and
# connection errors alike). Only the Groq client in this process is seen: not the v1 subprocess.
CALLS = {"n": 0, "n429": 0, "waited": 0.0}


class _CountGroq(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "api.groq.com" in msg and " 200 " in msg:       # retried 429s are not LLM calls
            CALLS["n"] += 1
        elif "api.groq.com" in msg and " 429 " in msg:
            CALLS["n429"] += 1


class _GroqWaits(logging.Handler):
    def emit(self, record):
        if record.msg.startswith("Retrying request to") and len(record.args or ()) == 2:
            CALLS["waited"] += float(record.args[1])


logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpx").addHandler(_CountGroq())
logging.getLogger("httpx").propagate = False
logging.getLogger("groq._base_client").setLevel(logging.INFO)
logging.getLogger("groq._base_client").addHandler(_GroqWaits())
logging.getLogger("groq._base_client").propagate = False


# ---------------------------------------------------------------- helpers

def key(cid: str) -> tuple:
    """chunk id 'PenalFr/s02/220-bis' -> ('PenalFr', '220-bis')."""
    return (cid.split("/")[0], cid.split("/")[-1].split("~")[0])


def gold_set(item) -> set:
    return {(source_stem(g["source"]), str(g["article"])) for g in item.get("gold") or []}


def is_exam(set_path: Path) -> bool:
    return Path(set_path).name == "answer_set.json"


def out_path(set_path: Path, system: str, stage: str = "all", results_dir: Path = ROOT / "results") -> Path:
    if is_exam(set_path):
        tag = "all" if stage == "all" else f"stage{stage}"
        return Path(results_dir) / f"answer_eval_{Path(set_path).stem}_{tag}_{system}.jsonl"
    return Path(results_dir) / f"answer_eval_{Path(set_path).stem}_{system}.jsonl"


MERGED = {"fr": "{q} (précision : {r})", "en": "{q} (clarification: {r})", "ar": "{q} (توضيح: {r})"}


def asked(item) -> str:
    """What a single-turn system (v1.1, v1) is asked: the question, merged with the scripted reply if any."""
    if not item.get("reply"):
        return item["question"]
    return MERGED.get(item.get("language"), MERGED["fr"]).format(q=item["question"], r=item["reply"])


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


class _NoUsage:
    usage_metadata = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def usage_callback():
    """LangChain usage-metadata callback (every chat-model call in the context, LangGraph nodes included)."""
    try:
        from langchain_core.callbacks import get_usage_metadata_callback
        return get_usage_metadata_callback()
    except ImportError:
        return _NoUsage()


def token_record(usage, sub_usage, est_in: str = "", est_out: str = "", calls: int = 0,
                 model: str = GENERATOR_MODEL) -> dict:
    """Tokens of one (item, system): exact from usage metadata ({model: {input_tokens, output_tokens}}, in process or
    from the v1 process); else, when LLM calls happened, an estimate of characters/4, labelled "estimated"."""
    usage = usage or sub_usage or {}
    by_model = {m: {"input": int(u.get("input_tokens", 0)), "output": int(u.get("output_tokens", 0))}
                for m, u in usage.items()}
    source = "usage_metadata" if by_model else "none"
    if not by_model and calls:
        by_model = {model: {"input": len(est_in) // 4, "output": len(est_out) // 4}}
        source = "estimated (characters/4 of question + evidence and of the answer; a lower bound)"
    agent = by_model.get(model) or {"input": sum(v["input"] for v in by_model.values()),
                                    "output": sum(v["output"] for v in by_model.values())}
    return {"source": source, "by_model": by_model, "input": agent["input"], "output": agent["output"],
            "total": agent["input"] + agent["output"]}


# ---------------------------------------------------------------- systems

def run_v2(item) -> dict:
    from agent_graph import app
    cfg = {"configurable": {"thread_id": f"eval-{item['id']}-{time.time_ns()}"}}      # a new thread per item
    s = app.invoke({"question": item["question"]}, cfg)
    first = None
    if item.get("reply"):                       # clarify-then-answer: the scripted reply on the same thread
        first = {"first_status": s.get("status"), "first_text": s.get("text"), "reply": item["reply"]}
        s = app.invoke({"question": item["reply"]}, cfg)
    ev = (s.get("evidence") or {}) if s.get("status") == "done" else {}
    rec = {"intent": s.get("intent"), "status": s.get("status"), "retries": s.get("retries"), "tried": s.get("tried"),
           "evidence_ids": list(ev), "cited": s.get("cited") or [], "claims": s.get("claims") or [],
           "caveat": s.get("caveat"), "text": s.get("text"), "degraded": s.get("degraded"),
           "verdicts": s.get("verdicts") or [], "repairs": s.get("repairs"), "dropped": s.get("dropped"),
           "draft_claims": s.get("draft_claims") or [], "draft_text": s.get("draft_text") or "",
           "first_verdicts": s.get("first_verdicts") or []}
    if s.get("status") == "clarify":
        rec["clarification_question"] = s.get("text")
    rec["_evidence_text"] = {cid: e["text"] for cid, e in ev.items()}
    return rec | (first or {})


_V11 = {}


def run_v11(item) -> dict:
    from pipeline import load_vectorstore, ask_question
    if "vs" not in _V11:
        _V11["vs"] = load_vectorstore()
    answer, docs = ask_question(_V11["vs"], asked(item), None)     # no law_type hint (fair, like A2)
    ids = [d.metadata.get("chunk_id") for d in docs]
    return {"status": "done" if docs else "no_docs", "evidence_ids": ids, "cited": None, "claims": None, "text": answer,
            "_evidence_text": {d.metadata.get("chunk_id"): d.page_content for d in docs}} | (
            {"asked": asked(item)} if item.get("reply") else {})


class V1Process:
    def __init__(self, worktree: Path):
        if not (worktree / "pipeline.py").exists() or not (worktree / "db").exists():
            sys.exit(f"--system v1 needs a git worktree of {V1_TAG} that contains the old index as db/. Create it with:\n"
                     f"  git worktree add <folder> {V1_TAG}\n"
                     f"  mklink /J <folder>\\db <repo>\\db          (Windows; or copy db/)\n"
                     f"then pass --v1-worktree <folder>. Nothing was run.")
        self.p = subprocess.Popen([sys.executable, str(ROOT / "eval" / "v1_runner.py"), str(worktree)], cwd=worktree,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    def run(self, item) -> dict:
        self.p.stdin.write(json.dumps({"id": item["id"], "question": asked(item)}, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if r.get("quota"):                       # a daily-quota 429 inside the v1 process
            raise hc.QuotaExhausted(r["quota"], hc.parse_retry_after(r["quota"]))
        if "error" in r:
            raise RuntimeError(r["error"])
        # v1 chunks have no chunk ids: use "<stem>/v1/<article>" (article = base number in the v1 index)
        ids = [f"{source_stem(d['source'])}/v1/{d['article']}" for d in r["docs"]]
        return {"status": "done" if r["docs"] else "no_docs", "evidence_ids": ids, "cited": None, "claims": None,
                "text": r["answer"], "_evidence_text": {i: d["text"] for i, d in zip(ids, r["docs"])},
                "_seconds": r["seconds"], "_llm_calls": r["llm_calls"], "_tokens": r.get("usage")} | (
                {"asked": asked(item)} if item.get("reply") else {})


def pdf_text(cid: str):
    if "/v1/" in cid:
        stem, _, art = cid.split("/")
        return pdf_article_text(stem, art)          # None if the label is ambiguous in a bundled PDF
    return from_chunk_id(cid)


# ---------------------------------------------------------------- deterministic metrics

def numeric_check(text: str, cids: list) -> dict:
    """eval/pdf_factcheck.py rule: every number of the text in the cited PDF texts. A missing "article N" that a
    cited article designates as "article précédent/suivant" (ci-dessus/ci-après) is accepted and listed in
    `implicit`: ok is then True (not a hallucination), and the answer is also counted under "implicit reference"."""
    texts = [pdf_text(c) for c in cids]
    ok, nums, missing = claim_numbers_ok(text, texts)
    implicit = implicit_reference_numbers(text, missing, cids, texts) if ok is False else set()
    if ok is False and implicit == missing:
        ok = True
    return {"ok": ok, "numbers": sorted(nums), "missing": sorted(missing - implicit), "implicit": sorted(implicit)}


def deterministic(item, rec) -> dict:
    gold = gold_set(item)
    ev = rec.get("evidence_ids") or []
    m = {"gold_hit": any(key(c) in gold for c in ev) if gold else None,
         "gold_cited": (any(key(c) in gold for c in rec["cited"]) if gold else None) if rec.get("cited") is not None else None}
    if rec.get("claims") is not None:                       # v2: structured claims
        m["cited_not_in_evidence"] = sorted({c for cl in rec["claims"] for c in cl["cited_ids"] if c not in ev})
        m["pdf_numeric_claims"] = [numeric_check(cl["text"], cl["cited_ids"]) for cl in rec["claims"]]
    else:                                                   # v1 / v1.1: the answer text as one block vs the 6 docs
        m["cited_not_in_evidence"] = None
        m["pdf_numeric_claims"] = [numeric_check(rec.get("text") or "", ev) | {"block": True}]
    oks = [c["ok"] for c in m["pdf_numeric_claims"]]
    m["pdf_numeric_ok"] = False if False in oks else (None if all(o is None for o in oks) else True)
    m["pdf_numeric_implicit"] = any(c.get("implicit") for c in m["pdf_numeric_claims"])
    if rec.get("draft_claims"):                             # v2 with the verifier loop: same check on the first draft
        m["draft_cited_not_in_evidence"] = sorted({c for cl in rec["draft_claims"] for c in cl["cited_ids"] if c not in ev})
        m["pdf_numeric_draft_claims"] = [numeric_check(cl["text"], cl["cited_ids"]) for cl in rec["draft_claims"]]
        oks = [c["ok"] for c in m["pdf_numeric_draft_claims"]]
        m["pdf_numeric_ok_draft"] = False if False in oks else (None if all(o is None for o in oks) else True)
    exp, st = item["expected"], rec.get("status")
    gate_st = rec.get("first_status", st)                  # v2 with a scripted reply: the gate is the first turn
    m["gate_correct"] = (gate_st == exp) if exp in ("clarify", "refuse") else None
    m["false_clarify"] = (st == "clarify") if exp == "answer" else None
    m["false_refuse"] = (st == "refuse") if exp == "answer" else None
    return m


# ---------------------------------------------------------------- judge

JUDGE_PROMPT = """You grade an answer to a question about Tunisian law against reference points written by a lawyer and
against the legal articles the answer relies on.

You receive: the question, the numbered reference points, the answer, and the text of the cited articles
(when an article is marked FALLBACK, its text comes from the search index, not from the official PDF).

Return:
- covered: one boolean per reference point, in order. True only if the answer states that point (same rule,
  same numbers); a vague mention is not enough.
- contradiction: true if the answer states something that contradicts a reference point.
- unsupported_statements: short quotes or paraphrases of statements in the answer that the cited articles do not
  support (facts, numbers, conditions, penalties). Empty list if every statement is supported.
Judge only from the texts given. Do not use outside legal knowledge."""


JUDGE_PROMPT_SHA = hashlib.sha1(JUDGE_PROMPT.encode("utf-8")).hexdigest()[:12]

GATE_PROMPT = """You read the response of a legal assistant on Tunisian law to a user's message. Answer two yes/no
questions about the response only (not about whether it is right):
- gives_substantive_answer: true if the response states legal rules, penalties, durations, amounts, deadlines or
  conditions as an answer to the message. False if it only refuses, says the information was not found or is not
  covered, or asks the user a question back.
- enumerates_alternatives: true if the response lists several cases, situations or readings of the message (with a
  rule for each, or as options for the user to pick from) instead of committing to one.
Judge only from the response text."""


GATE_PROMPT_SHA = hashlib.sha1(GATE_PROMPT.encode("utf-8")).hexdigest()[:12]


JUDGE_THINKING = "minimal"         # --judge-thinking (Gemini only)


def gemini_thinking(model: str, level: str = "minimal") -> dict:
    """Thinking arguments of ChatGoogleGenerativeAI (langchain-google-genai 4.4.2): Gemini 3+ take thinking_level (alias
    of reasoning_effort: minimal | low | medium | high; 'minimal' is the lowest, support varies by model); Gemini 2.5
    take thinking_budget (0 = off; Flash and Flash-Lite only). level 'default': nothing is passed (model default)."""
    m = re.search(r"gemini-(\d+)", model)
    if level == "default" or not m:
        return {}
    if int(m.group(1)) >= 3:
        return {"thinking_level": "minimal"}
    return {"thinking_budget": 0} if "2.5-flash" in model else {}


def judge_llm(provider: str, model: str, kind: str = "answer"):
    """Structured-output judge (kind "answer": JUDGE_PROMPT schema; "gate": GATE_PROMPT schema). groq: ChatGroq; gemini:
    langchain-google-genai, key from GOOGLE_API_KEY, thinking set by JUDGE_THINKING (gemini_thinking)."""
    from typing import List
    from pydantic import BaseModel

    class Answer(BaseModel):
        covered: List[bool]
        contradiction: bool
        unsupported_statements: List[str]

    class Gate(BaseModel):
        gives_substantive_answer: bool
        enumerates_alternatives: bool

    Judgement = Gate if kind == "gate" else Answer

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0, timeout=120, max_retries=0,   # retries: invoke_judge
                                      google_api_key=os.environ.get("GOOGLE_API_KEY"),
                                      **gemini_thinking(model, JUDGE_THINKING)).with_structured_output(Judgement)
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=0).with_structured_output(Judgement, method="json_schema")


JUDGE_FACTORY = judge_llm          # tests replace it with a stub


def invoke_judge(llm, messages, provider: str, sleep=time.sleep):
    """A daily-quota 429 -> QuotaExhausted(who='judge'), never retried. Gemini per-minute 429 or a temporary server
    error (503 'high demand', ...): up to 5 retries with backoff (the stated delay + 1 s, else 5, 10, 20, 40, 80 s; at
    most 120 s). Groq: the client's own retries."""
    for attempt in range(6):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = hc.full_message(e)
            if hc.is_daily_quota(msg):
                raise hc.QuotaExhausted(msg[:600], hc.parse_retry_after(msg), "judge") from e
            if provider != "gemini" or not (hc.is_rate_limit(msg) or hc.is_transient(msg)) or attempt == 5:
                raise
            sleep(min((hc.parse_retry_after(msg) or 5 * 2 ** attempt) + 1, 120))


_INDEX = {}


def index_text(cid: str) -> str | None:
    """Chunk text from the index (config.DB_DIR, read-only), for cited ids without PDF text (judge FALLBACK)."""
    if "db" not in _INDEX:
        from config import DB_DIR
        p = Path(DB_DIR) / "chroma.sqlite3"
        _INDEX["db"] = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True) if p.exists() else None
    if _INDEX["db"] is None:
        return None
    row = _INDEX["db"].execute(
        """select d.string_value from embedding_metadata c join embedding_metadata d on d.id = c.id
           and d.key = 'chroma:document' where c.key = 'chunk_id' and c.string_value = ?""", (cid,)).fetchone()
    return row[0] if row else None


def judge(item, rec, model: str, provider: str = "groq") -> dict:
    cited = rec["cited"] if rec.get("cited") is not None else rec.get("evidence_ids") or []
    arts = []
    for cid in cited:
        t = pdf_text(cid)
        fb = (rec.get("_evidence_text") or {}).get(cid) or (None if t else index_text(cid))
        if t:
            arts.append(f"[{cid}] (PDF)\n{normalize(t)[:2000]}")
        elif fb:
            arts.append(f"[{cid}] (FALLBACK: index chunk)\n{normalize(fb)[:2000]}")
    rp = item.get("reference_points") or []
    human = (f"Question: {item['question']}\n\nReference points:\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(rp, 1))
             + f"\n\nAnswer:\n{rec.get('text') or ''}\n\nCited articles:\n" + ("\n\n".join(arts) or "(none)"))
    with usage_callback() as cb:
        j = invoke_judge(JUDGE_FACTORY(provider, model), [("system", JUDGE_PROMPT), ("human", human)], provider)
    out = {"covered": j.covered, "contradiction": j.contradiction, "unsupported_statements": j.unsupported_statements,
           "fallback_articles": sum("FALLBACK" in a for a in arts), "provider": provider, "model": model,
           "prompt_sha": JUDGE_PROMPT_SHA, "tokens": token_record(cb.usage_metadata, None, model=model)}
    if len(j.covered) != len(rp):
        out["error"] = f"covered has {len(j.covered)} values for {len(rp)} reference points"
        return out
    n_cov = sum(j.covered)
    out["verdict"] = ("correct" if n_cov == len(rp) and not j.contradiction else
                      "partial" if n_cov >= 1 else "incorrect")
    return out


def gate_judge(item, rec, model: str, provider: str = "groq") -> dict:
    """GATE_PROMPT on a clarify/refuse item, any system: the first-turn response (v2 with a scripted reply: first_text,
    to the question; else text, to what the system was asked)."""
    first = "first_text" in rec
    q = item["question"] if first else rec.get("asked") or item["question"]
    human = f"Message: {q}\n\nResponse:\n{(rec.get('first_text') if first else rec.get('text')) or ''}"
    with usage_callback() as cb:
        j = invoke_judge(JUDGE_FACTORY(provider, model, "gate"), [("system", GATE_PROMPT), ("human", human)], provider)
    return {"gives_substantive_answer": bool(j.gives_substantive_answer),
            "enumerates_alternatives": bool(j.enumerates_alternatives), "turn": "first" if first else "only",
            "provider": provider, "model": model, "prompt_sha": GATE_PROMPT_SHA,
            "tokens": token_record(cb.usage_metadata, None, model=model)}


def check_judge(args, judge_state):
    """One small judge call before any agent call, so a wrong judge model or key stops the run before agent tokens are
    spent. A daily quota here turns judging off (rows get judge_pending), as during the run."""
    try:
        invoke_judge(JUDGE_FACTORY(args.judge_provider, args.judge_model),
                     [("system", JUDGE_PROMPT), ("human", "Question: ping ?\n\nReference points:\n1. ping\n\nAnswer:\n"
                                                          "ping\n\nCited articles:\n(none)")], args.judge_provider)
        print(f"judge check: {args.judge_provider} {args.judge_model} answered")
    except Exception as e:
        q = hc.classify(e, "judge")
        if not q:
            sys.exit(f"STOPPED before any agent call: judge check failed ({args.judge_provider} {args.judge_model}): "
                     f"{type(e).__name__}: {str(e)[:600]}")
        judge_state.update(on=False, quota=q)
        print(f"JUDGE QUOTA EXHAUSTED at the judge check: judging is off; rows are saved with judge_pending.",
              file=sys.stderr)


def hallucinated(rec, m, j) -> bool | None:
    if j is None or "verdict" not in j:
        return None
    return bool(j["unsupported_statements"]) or bool(m.get("cited_not_in_evidence")) or m["pdf_numeric_ok"] is False


# ---------------------------------------------------------------- summary

def metric_values(rows) -> dict:
    """metric -> list of per-item values (0/1 or float), None = not applicable; expected=answer items unless noted.
    Degraded, error and failed rows are left out."""
    vals = defaultdict(list)
    for r in rows:
        if r.get("status") in NO_ANSWER:
            continue
        m, j = r["metrics"], r.get("judge")
        if r["expected"] in ("clarify", "refuse"):
            gate_st, gj = r.get("first_status", r.get("status")), r.get("gate_judge")
            subst = None if not gj else int(gj["gives_substantive_answer"])
            vals["gate accuracy (explicit status)"].append(int(m["gate_correct"]))
            if r["expected"] == "refuse":                   # primary: explicit refusal OR no substantive answer
                vals["refuse: no substantive answer"].append(1 if gate_st == "refuse" else
                                                             None if subst is None else 1 - subst)
                vals["refuse: explicit refusal status"].append(int(gate_st == "refuse"))
                continue
            vals["clarify: asked for clarification (status)"].append(int(gate_st == "clarify"))
            vals["clarify: answered directly"].append(subst)
            vals["clarify: enumerated alternatives"].append(None if not gj else int(gj["enumerates_alternatives"]))
            if r.get("reply") or r.get("asked"):           # clarify-then-answer, once gold is filled
                vals["Hit@6 after the reply (clarify items)"].append(None if m["gold_hit"] is None else int(m["gold_hit"]))
            continue
        vals["Hit@6 (gold in evidence)"].append(None if m["gold_hit"] is None else int(m["gold_hit"]))
        vals["gold cited"].append(None if m["gold_cited"] is None else int(m["gold_cited"]))
        vals["false clarify"].append(int(m["false_clarify"]))
        vals["false refuse"].append(int(m["false_refuse"]))
        vals["numeric check ok (answer)"].append(None if m["pdf_numeric_ok"] is None else int(m["pdf_numeric_ok"]))
        vals["numeric check unverifiable (answer)"].append(int(m["pdf_numeric_ok"] is None))
        vals["numeric implicit reference (answer)"].append(          # absent in rows written before this rule
            None if "pdf_numeric_implicit" not in m else int(m["pdf_numeric_implicit"]))
        claims = [c["ok"] for c in m["pdf_numeric_claims"] if c["ok"] is not None]
        vals["numeric check ok (claim share)"].append(sum(claims) / len(claims) if claims else None)
        vals["seconds"].append(r["seconds"])
        vals["LLM calls"].append(r["llm_calls"])
        vals["elapsed total (s)"].append(r.get("elapsed_total"))          # absent in rows written before these fields
        vals["429 responses"].append(r.get("n_429"))
        vals["client retry waits (s)"].append(r.get("waited_seconds"))
        v = j.get("verdict") if j else None
        for name in ("correct", "partial", "incorrect"):
            vals[name].append(None if v is None else int(v == name))
        h = r.get("hallucinated")
        vals["hallucination (answer)"].append(None if h is None else int(h))
        n_claims = len(r.get("claims") or [])
        vals["hallucination (claim-level, judge)"].append(
            None if v is None or not n_claims else min(len(j["unsupported_statements"]), n_claims) / n_claims)
        # verifier loop (v2 rows with a first draft, i.e. the verifier ran). Claim-level share of draft claims not
        # judged "supported" on the first pass (a claim without a verdict counts as flagged, as verify.bad_claims).
        drafted = bool(r.get("draft_claims"))
        first = {v["claim_index"]: v["verdict"] for v in r.get("first_verdicts") or []}
        n_draft = len(r.get("draft_claims") or [])
        vals["claims flagged on 1st pass"].append(
            sum(first.get(i) != "supported" for i in range(1, n_draft + 1)) / n_draft
            if drafted and r.get("first_verdicts") else None)
        vals["repaired (repairs == 1)"].append(int(r["repairs"] == 1) if drafted else None)
        vals["dropped claims (mean)"].append(r["dropped"] if drafted else None)
        vals["no supported claim left"].append(int(not r.get("claims")) if drafted else None)
        # numeric check, answer level: 1 if a claim has a number missing from its cited PDF articles; same items for
        # draft and final (an answer whose claims were all dropped states no number: 0)
        vals["numeric miss, draft claims (answer)"].append(
            int(any(c["ok"] is False for c in m["pdf_numeric_draft_claims"])) if drafted else None)
        vals["numeric miss, final claims (answer)"].append(
            int(any(c["ok"] is False for c in m["pdf_numeric_claims"])) if drafted else None)
        jd = r.get("judge_draft")
        vd = jd.get("verdict") if jd else None
        vals["correct (draft)"].append(None if vd is None else int(vd == "correct"))
        hd = r.get("hallucinated_draft")
        vals["hallucination (answer, draft)"].append(None if hd is None else int(hd))
    return vals


def boot_ci(xs, rng) -> tuple:
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]


def print_block(title: str, rows: list[dict]):
    n_deg = sum(r.get("status") == "degraded" for r in rows)
    n_err = sum(r.get("status") == "error" for r in rows)
    failed = [r for r in rows if r.get("status") == "failed"]
    rows = [r for r in rows if r.get("status") not in NO_ANSWER]
    vals = metric_values(rows)
    print(f"\n=== {title}: {len(rows)} items (answer: {sum(r['expected'] == 'answer' for r in rows)}, "
          f"clarify/refuse: {sum(r['expected'] != 'answer' for r in rows)}); degraded (excluded, re-run on resume): "
          f"{n_deg}; error (excluded, re-run on resume): {n_err}; system failure (no answer, excluded): {len(failed)}; "
          f"95% bootstrap CI over questions, {N_BOOT} resamples, seed {SEED}")
    versions = sorted({(r.get("provenance") or {}).get("commit", "?")[:10] + " " +
                       ",".join((r.get("provenance") or {}).get("tags") or []) for r in rows})
    print(f"  commits/tags in these rows: {versions}")
    for name, xs in vals.items():
        xs = [x for x in xs if x is not None]
        if not xs:
            print(f"  {name:<42} n=0   -")
            continue
        lo, hi = boot_ci(xs, random.Random(SEED))
        print(f"  {name:<42} n={len(xs):<4} {sum(xs) / len(xs):8.3f}   [{lo:.3f}, {hi:.3f}]")
    n_fail = {"answer": sum(r["expected"] == "answer" for r in failed)}
    n_fail["gate"] = len(failed) - n_fail["answer"]
    print(f"  {'system failure (no answer)':<42} answer items {n_fail['answer']}, clarify/refuse items {n_fail['gate']}")
    for name, kind in (("correct", "answer"), ("gate accuracy (explicit status)", "gate")):
        xs = [x for x in vals.get(name, []) if x is not None] + [0] * n_fail[kind]
        if n_fail[kind] and xs:
            lo, hi = boot_ci(xs, random.Random(SEED))
            print(f"  {'sensitivity: ' + name + ', failures = 0':<42} n={len(xs):<4} {sum(xs) / len(xs):8.3f}   "
                  f"[{lo:.3f}, {hi:.3f}]")


def summarize(paths: list[Path]):
    runs = {}
    for p in paths:
        runs[p.stem] = {r["id"]: r for r in read_jsonl(p)}         # latest row per item (a resume re-runs degraded)
        print_block(p.name, list(runs[p.stem].values()))
    by_system = defaultdict(list)                                   # stage files of one system -> pooled line
    for p in paths:
        rows = list(runs[p.stem].values())
        if rows and "_stage" in p.stem:
            by_system[rows[0].get("system")].append(p)
    for system, ps in by_system.items():
        if len(ps) > 1:
            pooled = {i: r for p in ps for i, r in runs[p.stem].items()}
            print_block(f"pooled (valid only if the same tag ran both stages) {system}: "
                        f"{' + '.join(p.name for p in ps)}", list(pooled.values()))
    names = list(runs)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = runs[names[a]], runs[names[b]]
            ids = sorted(i for i in set(A) & set(B) if not {A[i].get("status"), B[i].get("status")} & set(NO_ANSWER))
            if not ids:
                continue
            print(f"\n=== paired difference {names[b]} - {names[a]} on {len(ids)} common items")
            va = metric_values([A[i] for i in ids])
            vb = metric_values([B[i] for i in ids])
            for name in va:
                d = [y - x for x, y in zip(va[name], vb.get(name, [])) if x is not None and y is not None]
                if not d:
                    print(f"  {name:<42} n=0   -")
                    continue
                lo, hi = boot_ci(d, random.Random(SEED))
                print(f"  {name:<42} n={len(d):<4} {sum(d) / len(d):+8.3f}   [{lo:+.3f}, {hi:+.3f}]")


# ---------------------------------------------------------------- audit export

def export_audit(path: Path, system: str, n: int):
    rows = [r for r in {r["id"]: r for r in read_jsonl(path)}.values()
            if r["expected"] == "answer" and r.get("status") not in NO_ANSWER]
    pick = random.Random(SEED).sample(rows, min(n, len(rows)))
    out = ROOT / "results" / f"audit_sample_{system}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        per_claim = [h for k in range(1, 7) for h in (f"claim_{k}", f"claim_{k}_relevant_to_question (y/n)")]
        w.writerow(["id", "question", "answer", "cited_article_pdf_text", "judge_verdict", "judge_hallucination",
                    "human_correct", "human_hallucination"] + per_claim)
        for r in pick:
            cited = r.get("cited") if r.get("cited") is not None else r.get("evidence_ids") or []
            texts = " || ".join(f"{c}: {normalize(pdf_text(c) or '(no PDF text)')[:600]}" for c in cited)
            j = r.get("judge") or {}
            claims = [c["text"] for c in r.get("claims") or []][:6]          # Answer allows at most 6 claims
            claims += [""] * (6 - len(claims))
            w.writerow([r["id"], r["question"], r.get("text"), texts, j.get("verdict", ""),
                        "" if r.get("hallucinated") is None else int(r["hallucinated"]), "", ""]
                       + [x for c in claims for x in (c, "")])
    print(f"wrote {out.relative_to(ROOT)} ({len(pick)} rows); fill human_correct / human_hallucination with 1 or 0, "
          f"then run: python eval/audit_agreement.py {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- main

CALL_RANGES = {   # (min, max) Groq calls per question, without the judge
    "v2": ((1, 10), "Analyze 1; clarify/refuse stops there. Else: translation 0-1 (AR/EN), Grade 1, up to 2 retries "
                    "(Reformulate, Grade, Reformulate = +3), Answer 1, Verify 1, at most one repair (Answer, Verify = +2)"),
    "v1.1": ((1, 2), "translation 0-1 (AR/EN), generation 1"),
    "v1": ((2, 3), "rewrite 1, sub-queries 0-1 (complex questions), generation 1"),
}
UNFINISHED = ("degraded", "error")             # statuses run again on resume ("failed" is final)
NO_ANSWER = UNFINISHED + ("failed",)            # rows left out of the metrics
TRIES_PER_RUN, MAX_ATTEMPTS = 2, 3
SYSTEMS = {"v2": run_v2, "v1.1": run_v11}       # v1: a V1Process, created in main; tests replace entries with stubs
SLEEP = time.sleep                              # --wait-on-quota sleeps (tests replace it)


def prompt_hashes() -> dict:
    """sha1[:12] of the agent prompts, as eval/preflight.py computes them (no model is loaded, no call is made)."""
    import analyze, plan, grade, reformulate, verify, answer, condense
    sha = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
    return {"analyze": sha(analyze.SYSTEM_PROMPT), "plan": sha(plan.PLAN_PROMPT),
            "codes": sha((ROOT / "agent" / "codes.py").read_text(encoding="utf-8")),
            "verify": sha(verify.SYSTEM_PROMPT), "grade": sha(grade.GRADE_PROMPT),
            "reformulate": sha(reformulate.REFORMULATE_PROMPT), "answer": sha(answer.SYSTEM_PROMPT),
            "condense": sha(condense.CONDENSE_PROMPT)}


def provenance(args, exam_hashes: dict) -> dict:
    """Stored in every row. Tests replace this function."""
    return {**hc.git_provenance(ROOT), "agent_model": GENERATOR_MODEL, "prompt_hashes": prompt_hashes(),
            "exam_hashes": exam_hashes,
            "judge": None if args.skip_judge else {"provider": args.judge_provider, "model": args.judge_model,
                                                   "prompt_sha": JUDGE_PROMPT_SHA, "gate_prompt_sha": GATE_PROMPT_SHA,
                                                   "thinking": judge_thinking(args)}}


def judge_thinking(args) -> dict | None:
    return gemini_thinking(args.judge_model or "", args.judge_thinking) if args.judge_provider == "gemini" else None


def probe_agent() -> bool:
    """One tiny request on the agent model: True if it answers, False on a daily-quota 429. Tests replace it."""
    from langchain_groq import ChatGroq
    try:
        ChatGroq(model=GENERATOR_MODEL, temperature=0, max_tokens=1, max_retries=0).invoke("ping")
        return True
    except Exception as e:
        if hc.classify(e):
            return False
        raise


def run_one(item, system, run, args, judge_state, stage, prov) -> dict:
    """Run one (item, system) and build its row. Raises QuotaExhausted (the pair is then discarded by the caller)."""
    hc.QUOTA_SEEN["msg"] = None
    c0, t0 = CALLS["n"], time.time()
    r0, w0 = CALLS["n429"], CALLS["waited"]
    try:
        with usage_callback() as cb:
            rec = run(item)
    except Exception as e:
        q = hc.classify(e)
        if q:
            raise q from e
        raise
    if hc.QUOTA_SEEN["msg"]:                        # a node swallowed a daily-quota error (v2 marks it degraded)
        raise hc.QuotaExhausted(hc.QUOTA_SEEN["msg"], hc.parse_retry_after(hc.QUOTA_SEEN["msg"]))
    seconds = rec.pop("_seconds", round(time.time() - t0, 1))
    calls = rec.pop("_llm_calls", CALLS["n"] - c0)
    ev_text = rec.get("_evidence_text") or {}
    tokens = token_record(cb.usage_metadata, rec.pop("_tokens", None),
                          est_in=asked(item) + "".join(ev_text.values()), est_out=rec.get("text") or "", calls=calls)
    degraded = bool(rec.get("degraded"))
    if degraded:                                    # a node fell back: no metrics, re-run on resume
        rec["graph_status"], rec["status"] = rec.get("status"), "degraded"
    m = None if degraded else deterministic(item, rec)
    jd = jdd = gj = None
    judgeable = item["expected"] == "answer" and rec.get("status") == "done"
    gate_judgeable = item["expected"] in ("clarify", "refuse") and not degraded
    if gate_judgeable and judge_state["on"]:
        hc.QUOTA_SEEN["msg"] = None
        try:
            gj = gate_judge(item, rec, args.judge_model, args.judge_provider)
        except Exception as e:
            judge_failed(e, item, args, judge_state)
    if judgeable and judge_state["on"]:
        c1 = CALLS["n"]
        hc.QUOTA_SEEN["msg"] = None
        try:
            jd = judge(item, rec, args.judge_model, args.judge_provider)
            if args.judge_draft and rec.get("draft_text"):
                if rec["draft_text"] == rec.get("text"):                 # nothing repaired or dropped
                    jdd = dict(jd, same_as_final=True)
                else:
                    d_cited = sorted({c for cl in rec["draft_claims"] for c in cl["cited_ids"]})
                    jdd = judge(item, rec | {"text": rec["draft_text"], "cited": d_cited}, args.judge_model,
                                args.judge_provider)
            jd["llm_calls"] = CALLS["n"] - c1
        except Exception as e:
            jd = jdd = None
            judge_failed(e, item, args, judge_state)
    rec.pop("_evidence_text", None)
    v1 = system == "v1"                             # its LLM calls run in the subprocess: 429s and waits not seen
    row = {"id": item["id"], "question": item["question"], "language": item.get("language"),
           "expected": item["expected"], "system": system, "stage": stage, **rec, "seconds": seconds,
           "llm_calls": calls, "elapsed_total": round(time.time() - t0, 1),         # system + metrics + judge
           "n_429": None if v1 else CALLS["n429"] - r0, "waited_seconds": None if v1 else round(CALLS["waited"] - w0, 1),
           "metrics": m, "judge": jd, "gate_judge": gj,
           "judge_pending": bool(not degraded and ((judgeable and jd is None) or (gate_judgeable and gj is None))),
           "tokens": tokens, "date": datetime.now().isoformat(timespec="seconds"), "provenance": prov}
    row["hallucinated"] = None if degraded else hallucinated(rec, m, jd)
    if args.judge_draft and m and "pdf_numeric_ok_draft" in m:
        row["judge_draft"] = jdd
        row["hallucinated_draft"] = hallucinated(rec, {"cited_not_in_evidence": m["draft_cited_not_in_evidence"],
                                                       "pdf_numeric_ok": m["pdf_numeric_ok_draft"]}, jdd)
    return row


def judge_failed(e, item, args, judge_state):
    """Quota or any other judge error: the answer is kept (its agent tokens are spent); judge later with --judge-only."""
    q = hc.classify(e, "judge")
    judge_state.update(on=False, quota=q, error=None if q else f"{type(e).__name__}: {str(e)[:300]}")
    print(f"JUDGE {'QUOTA EXHAUSTED' if q else 'FAILED: ' + judge_state['error']} on {item['id']} "
          f"({args.judge_provider} {args.judge_model}): judging is off for the rest of this run; rows are saved "
          f"with judge_pending; run --judge-only later.", file=sys.stderr)


def run_isolated(item, system, run, args, judge_state, stage, prov, prev) -> dict:
    """run_one, tried TRIES_PER_RUN times in this run (v2: run_v2 opens a new thread id on every call). A non-quota
    exception on every try -> a row with status "error" (attempt k, re-run on resume) or, at attempt MAX_ATTEMPTS,
    "failed" (final). QuotaExhausted is raised to the caller (exit 75)."""
    errors = []
    for _ in range(TRIES_PER_RUN):
        try:
            return run_one(item, system, run, args, judge_state, stage, prov)
        except hc.QuotaExhausted:
            raise
        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            print(f"  {item['id']} {system} failed: {errors[-1]}", file=sys.stderr)
    attempt = ((prev or {}).get("attempt", 0) if (prev or {}).get("status") == "error" else 0) + 1
    status = "failed" if attempt >= MAX_ATTEMPTS else "error"
    print(f"  {item['id']} {system}: saved as {status} (attempt {attempt} of {MAX_ATTEMPTS}); the run goes on",
          file=sys.stderr)
    return {"id": item["id"], "question": item["question"], "language": item.get("language"),
            "expected": item["expected"], "system": system, "stage": stage, "status": status, "error": errors[-1],
            "errors": errors, "attempt": attempt, "metrics": None, "judge": None, "gate_judge": None,
            "judge_pending": False, "hallucinated": None, "tokens": None,
            "date": datetime.now().isoformat(timespec="seconds"), "provenance": prov}


def measured_tokens(f: hc.ResultFile) -> list[int]:
    return [r["tokens"]["total"] for r in f.latest().values() if (r.get("tokens") or {}).get("source") not in (None, "none")]


def projection(files: dict, todo_by_system: dict):
    """Tokens used (agent model), tokens per question, days needed at TPD for what remains."""
    per_q_all, need = 0.0, 0.0
    print(f"\nToken projection (agent model {GENERATOR_MODEL}, {TPD} tokens/day):")
    for s, f in files.items():
        xs = measured_tokens(f)
        est = sum("estimated" in (r.get("tokens") or {}).get("source", "") for r in f.latest().values())
        per_q = sum(xs) / len(xs) if xs else DEFAULT_TOKENS[s]
        per_q_all += per_q
        need += per_q * todo_by_system.get(s, 0)
        print(f"  {s:<5} used {sum(xs):>9} tokens on {len(xs)} rows ({est} estimated); per question "
              f"{per_q:>8.0f} ({'measured' if xs else 'DEFAULT_TOKENS guess'}); {todo_by_system.get(s, 0)} to run")
    print(f"  per question, all selected systems: {per_q_all:.0f}; remaining: {need:.0f} tokens = "
          f"{need / TPD:.1f} day(s) at {TPD}/day")


def mixed_folder(set_path: Path, results_dir: Path) -> str | None:
    """Exam stage results and results of any other set never share a folder (or run_state.json / run_log.jsonl)."""
    files = [p.name for p in Path(results_dir).glob("answer_eval_*.jsonl")]
    if is_exam(set_path):
        other = [n for n in files if not n.startswith("answer_eval_answer_set_")]
        if other:
            return (f"{results_dir} holds results of another set ({', '.join(sorted(other)[:3])}); the exam stage "
                    f"results must not be mixed with them. Use another --results-dir.")
        return None
    stage_files = [n for n in files if n.startswith("answer_eval_answer_set_stage")]
    if Path(results_dir).resolve() == (ROOT / "results").resolve() or stage_files:
        return (f"{set_path.name} is not the exam; its results must not go to the folder of the exam stage results "
                f"({results_dir}). Use its own folder, e.g. --results-dir results\\plumbing.")
    return None


def stage_ids(stages_path: Path, stage: str) -> set | None:
    if stage == "all":
        return None
    st = json.loads(Path(stages_path).read_text(encoding="utf-8"))
    return set(st[f"stage{stage}"])


def stop(results_dir, reason, msg, todo, k, files, args, retry_after=None, code=hc.EXIT_QUOTA):
    remaining = defaultdict(int)
    for _, s in todo[k:]:
        remaining[s] += 1
    hc.write_state(results_dir, {"reason": reason, "message": msg[:600], "resume_estimate": hc.resume_estimate(retry_after),
                                 "command": " ".join(sys.argv), "stage": args.stage,
                                 "done": {s: len(f.finished()) for s, f in files.items()}, "remaining": dict(remaining)})
    print(msg, file=sys.stderr)
    sys.exit(code)


def judge_only(args, items, files, results_dir):
    by_id = {i["id"]: i for i in items}
    todo = [(r, s) for s, f in files.items() for r in f.latest().values() if r["id"] in by_id and (
            (r["expected"] == "answer" and r.get("status") == "done" and not (r.get("judge") and "verdict" in r["judge"]))
            or (r["expected"] in ("clarify", "refuse") and r.get("status") not in NO_ANSWER
                and not r.get("gate_judge")))]
    print(f"--judge-only: {len(todo)} saved rows without a judgement ({args.judge_provider} {args.judge_model}; "
          f"answer items: JUDGE_PROMPT, clarify/refuse items: GATE_PROMPT)")
    for k, (r, s) in enumerate(todo):
        if k and args.min_gap:
            time.sleep(args.min_gap)
        gate = r["expected"] != "answer"
        try:
            jd = (gate_judge if gate else judge)(by_id[r["id"]], r, args.judge_model, args.judge_provider)
        except Exception as e:
            q = hc.classify(e, "judge")
            if not q:
                stop(results_dir, "judge error", f"STOPPED: judge failed on {r['id']} ({s}): {type(e).__name__}: "
                     f"{str(e)[:400]}\nJudged so far are saved; re-run the same command to continue.",
                     [(x["id"], y) for x, y in todo], k, files, args, code=1)
            stop(results_dir, "judge daily quota", f"STOPPED: daily quota of the judge ({args.judge_provider} "
                 f"{args.judge_model}) reached on {r['id']} ({s}). Judged so far are saved; re-run the same command to "
                 f"continue.\n{q}", [(x["id"], y) for x, y in todo], k, files, args, q.retry_after)
        if gate:
            files[s].replace_latest(dict(r, gate_judge=jd, judge_pending=False))
            print(f"  judged {r['id']} ({s}): substantive {jd['gives_substantive_answer']}, alternatives "
                  f"{jd['enumerates_alternatives']}")
            continue
        files[s].replace_latest(dict(r, judge=jd, judge_pending=False, hallucinated=hallucinated(r, r["metrics"], jd)))
        print(f"  judged {r['id']} ({s}): {jd.get('verdict', jd.get('error'))}")
    hc.write_state(results_dir, {"reason": "judge-only completed", "stage": args.stage, "judged": len(todo)})


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default=str(EXAM_DIR / "answer_set.json"))
    ap.add_argument("--stages-file", default=str(EXAM_DIR / "stages.json"))
    ap.add_argument("--hashes-file", default=str(EXAM_DIR / "exam_hashes.txt"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    ap.add_argument("--system", default="v2", help="v2, v1.1, v1, a comma list, or all (= v1,v1.1,v2)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-ids")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--judge-only", action="store_true", help="judge saved answers that have no judgement yet")
    ap.add_argument("--judge-provider", choices=["groq", "gemini"], default="groq")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    ap.add_argument("--judge-thinking", choices=["minimal", "default"], default="minimal",
                    help="gemini judge: minimal thinking (see gemini_thinking) or the model default")
    ap.add_argument("--judge-draft", action="store_true",
                    help="v2: also judge draft_text (the unverified first answer); one more judge call when it differs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-gap", type=float, default=0, metavar="SECONDS", help="sleep between (item, system) runs")
    ap.add_argument("--daily-budget-tokens", type=int, metavar="N")
    ap.add_argument("--wait-on-quota", type=float, nargs="?", const=20, metavar="MINUTES")
    ap.add_argument("--max-wait-hours", type=float, default=30)
    ap.add_argument("--allow-mixed", action="store_true")
    ap.add_argument("--allow-changed-exam", action="store_true")
    ap.add_argument("--summarize", nargs="*", metavar="JSONL")
    ap.add_argument("--export-audit", type=int, metavar="N")
    ap.add_argument("--v1-worktree", help=f"git worktree of {V1_TAG} with the old index in db/ (for --system v1)")
    return ap.parse_args(argv)


def main(argv=None):
    global JUDGE_THINKING
    args = parse_args(argv)
    JUDGE_THINKING = args.judge_thinking
    set_path, results_dir = Path(args.set), Path(args.results_dir)
    systems = SYSTEM_ORDER if args.system == "all" else args.system.split(",")
    bad = [s for s in systems if s not in SYSTEM_ORDER]
    if bad:
        sys.exit(f"unknown system(s) {bad}; choose from {SYSTEM_ORDER} or all")
    systems = [s for s in SYSTEM_ORDER if s in systems]                 # question-major order v1, v1.1, v2
    if args.summarize is not None:
        pattern = f"answer_eval_{set_path.stem}_stage*_*.jsonl" if is_exam(set_path) else f"answer_eval_{set_path.stem}_*.jsonl"
        summarize([Path(p) for p in args.summarize] or sorted(results_dir.glob(pattern)))
        return
    if not is_exam(set_path) and args.stage != "all":
        sys.exit("--stage 1/2 applies to the exam (eval/answer_set.json) only")
    mixing = mixed_folder(set_path, results_dir)
    if mixing:
        sys.exit("REFUSED: " + mixing)
    files = {s: hc.ResultFile(out_path(set_path, s, args.stage, results_dir)) for s in systems}
    if args.export_audit:
        export_audit(files[systems[0]].path, systems[0], args.export_audit)
        return

    if is_exam(set_path):
        cur_hashes, diffs = hc.exam_check(set_path, args.stages_file, args.hashes_file)
        print(f"exam fingerprint ({Path(args.hashes_file).name}): {'matches' if not diffs else 'DIFFERS'}")
        if diffs and not args.allow_changed_exam:
            sys.exit("REFUSED: the exam changed since the fingerprint:\n  " + "\n  ".join(diffs) +
                     "\nResults on a changed exam are not comparable. Re-run eval/build_exam.py and eval/make_stages.py "
                     "if the change is intended (and commit), or pass --allow-changed-exam.")
    else:                                          # not the exam (e.g. eval/plumbing_set.json): no fingerprint required
        cur_hashes = {set_path.name: hc.sha256(set_path)}
        print(f"set {set_path.name} is not the exam: no fingerprint check; results in {results_dir}")
    items = json.loads(set_path.read_text(encoding="utf-8"))
    ids = stage_ids(args.stages_file, args.stage) if is_exam(set_path) else None
    if ids is not None:
        items = [i for i in items if i["id"] in ids]
    if args.only_ids:
        wanted = set(args.only_ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    if args.limit:
        items = items[:args.limit]

    prov = provenance(args, cur_hashes)
    for s, f in files.items():                     # no pooling across versions
        other = hc.mixed_versions(f.rows(), prov["commit"])
        if other:
            msg = (f"{f.path.name} holds rows of another commit/tag ({sorted(map(str, other))}; now {prov['commit']} "
                   f"{prov['tags']}). Results of different versions must not be pooled: the systems, prompts or "
                   f"index may differ, so the numbers would mix two experiments.")
            if not args.allow_mixed:
                sys.exit("REFUSED: " + msg + " Use a new --results-dir, or pass --allow-mixed.")
            print("WARNING (--allow-mixed): " + msg)
    if args.judge_only:
        if not args.judge_model:
            sys.exit("--judge-only needs a judge model (JUDGE_MODEL or --judge-model).")
        judge_only(args, items, files, results_dir)
        return

    todo = [(i, s) for i in items for s in systems if i["id"] not in files[s].finished()]
    todo_by_system = defaultdict(int)
    for _, s in todo:
        todo_by_system[s] += 1
    need_judge = not args.skip_judge
    j = 0 if args.skip_judge else 1
    judge_desc = "skipped" if args.skip_judge else (f"{args.judge_provider} {args.judge_model or 'NOT SET'}"
                                                    + (f", thinking {judge_thinking(args)}" if judge_thinking(args) is not None else ""))
    print(f"set {set_path.name}, stage {args.stage}: {len(items)} items; systems {systems} (question-major order); "
          f"{len(todo)} (item, system) runs to do; judge: {judge_desc}")
    for s in systems:
        (lo, hi), how = CALL_RANGES[s]
        print(f"  {s}: {files[s].path.name}: {len(files[s].finished() & {i['id'] for i in items})} done, "
              f"{todo_by_system[s]} to run; LLM calls per question {lo + j}-{hi + j} ({how})")
    replies = [i for i in items if i.get("reply")]
    if replies:
        print(f"{len(replies)} clarify items with a scripted reply (v2: question then reply on the same thread_id; "
              f"v1.1/v1: one merged question):")
        for i in replies:
            print(f"  {i['id']}: turn 1 {i['question']!r} | turn 2 {i['reply']!r} | merged {asked(i)!r}")
    if args.judge_provider == "gemini":
        print(f"GOOGLE_API_KEY: {'set' if os.environ.get('GOOGLE_API_KEY') else 'unset'}")
    projection(files, todo_by_system)
    if args.dry_run:
        return
    if need_judge and not args.judge_model:
        sys.exit("No judge model: set JUDGE_MODEL or pass --judge-model (a model different from the generator), "
                 "or use --skip-judge. No model name is guessed.")
    if need_judge and args.judge_model == GENERATOR_MODEL:
        sys.exit(f"The judge must be a different model from the generator ({GENERATOR_MODEL}).")

    hc.install_groq_patches()
    runners = dict(SYSTEMS)
    v1proc = None
    if "v1" in systems and "v1" not in runners:
        v1proc = V1Process(Path(args.v1_worktree or ""))
        runners["v1"] = v1proc.run
    judge_state = {"on": need_judge, "quota": None, "error": None}
    if need_judge and todo:
        check_judge(args, judge_state)
    today = datetime.now().date().isoformat()
    today_tokens = lambda rs: sum(r["tokens"]["total"] for r in rs if (r.get("date") or "").startswith(today) and r.get("tokens"))
    mine = {f.path.resolve() for f in files.values()}       # agent tokens already spent today by other stage files
    used_other = sum(today_tokens(read_jsonl(p)) for p in results_dir.glob(f"answer_eval_{set_path.stem}_*.jsonl")
                     if p.resolve() not in mine)
    waited, k = 0.0, 0
    while k < len(todo):
        item, s = todo[k]
        if args.daily_budget_tokens:
            used = used_other + sum(today_tokens(f.rows()) for f in files.values())
            xs = measured_tokens(files[s])
            est = sum(xs) / len(xs) if xs else DEFAULT_TOKENS[s]
            if used + est > args.daily_budget_tokens:
                stop(results_dir, "daily token budget", f"STOPPED: daily token budget: {used} agent tokens used today "
                     f"+ about {est:.0f} for {item['id']} ({s}) > {args.daily_budget_tokens}. Finished pairs are saved; "
                     f"re-run the same command to continue.", [(i["id"], x) for i, x in todo], k, files, args)
        if k and args.min_gap:
            time.sleep(args.min_gap)
        print(f"[{k + 1}/{len(todo)}] {item['id']} {s} ...", file=sys.stderr, flush=True)
        try:
            row = run_isolated(item, s, runners[s], args, judge_state, args.stage, prov, files[s].latest().get(item["id"]))
        except hc.QuotaExhausted as q:
            if args.wait_on_quota is not None:
                ok, waited = hc.wait_for_quota(probe_agent, args.wait_on_quota, args.max_wait_hours, q.retry_after,
                                               waited, sleep=SLEEP)
                if ok:
                    continue                        # the discarded pair is run again
            stop(results_dir, "agent daily quota", f"STOPPED: daily quota reached on {item['id']} ({s}); this pair is "
                 f"discarded (not saved, not degraded). Finished pairs are saved. Re-run the same command to continue"
                 + (f" (about {q.retry_after / 60:.0f} min from now)" if q.retry_after else "") + f".\n{str(q)[:400]}",
                 [(i["id"], x) for i, x in todo], k, files, args, q.retry_after)
        files[s].append(row)
        k += 1
    if v1proc:
        v1proc.p.stdin.close()
    pending = sum(r.get("judge_pending", False) for f in files.values() for r in f.latest().values())
    hc.write_state(results_dir, {"reason": "completed", "stage": args.stage, "command": " ".join(sys.argv),
                                 "done": {s: len(f.finished()) for s, f in files.items()}, "remaining": {},
                                 "judge_pending": pending, "judge_error": judge_state["error"]})
    if pending:
        print(f"{pending} rows wait for the judge: run the same command with --judge-only.")
    projection(files, {})
    summarize([f.path for f in files.values()])


if __name__ == "__main__":
    main()
