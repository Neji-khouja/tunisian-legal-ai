"""Runs the v1 pipeline (tag v1-baseline-runnable) from a git worktree, in its own process. Used by
eval/run_answer_eval.py --system v1; reads {"id", "question"} JSON lines on stdin, writes one JSON line per item.

    python eval/v1_runner.py <worktree>      (cwd = <worktree>, which must contain the old index in db/)

Only the worktree's modules are imported (its config, pipeline, retriever, ...), never the current ones; the one
exception is eval/harness_core.py (standard library only), loaded by file path: a daily-quota 429 is not retried
and is returned as "quota". Token usage (LangChain usage metadata) is returned as "usage". Warnings of a step that fell
back (harness_core.capture_fallbacks: query rewrite, sub-queries, translation) are returned as "fallbacks".
"""
import os
import sys
import json
import time
import logging

worktree = os.path.abspath(sys.argv[1])
sys.path = [worktree] + [p for p in sys.path[1:] if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]
os.chdir(worktree)

calls = {"n": 0}


class _Count(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "api.groq.com" in msg and " 200 " in msg:        # retried 429s are not LLM calls
            calls["n"] += 1


logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpx").addHandler(_Count())

from pipeline import load_vectorstore, ask_question     # the worktree's v1 pipeline
import importlib.util
_spec = importlib.util.spec_from_file_location("harness_core", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "harness_core.py"))
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)
hc.install_groq_patches()
try:
    from langchain_core.callbacks import get_usage_metadata_callback
except ImportError:
    get_usage_metadata_callback = None

vs = load_vectorstore()
for line in sys.stdin:
    item = json.loads(line)
    before, t0 = calls["n"], time.time()
    hc.QUOTA_SEEN["msg"], usage = None, None
    try:
        with hc.capture_fallbacks() as fallbacks:
            if get_usage_metadata_callback:
                with get_usage_metadata_callback() as cb:
                    answer, docs = ask_question(vs, item["question"], None)
                usage = cb.usage_metadata
            else:
                answer, docs = ask_question(vs, item["question"], None)
        out = {"id": item["id"], "answer": answer, "fallbacks": fallbacks,
               "docs": [{"source": d.metadata.get("source"), "article": d.metadata.get("article"),
                         "text": d.page_content} for d in docs]}
    except Exception as e:
        out = {"id": item["id"], "error": f"{type(e).__name__}: {e}"}
        q = hc.classify(e)
        if q:
            out["quota"] = str(q)
    if hc.QUOTA_SEEN["msg"] and "quota" not in out:
        out["quota"] = hc.QUOTA_SEEN["msg"]
    out["usage"] = usage
    out["seconds"] = round(time.time() - t0, 1)
    out["llm_calls"] = calls["n"] - before
    print(json.dumps(out, ensure_ascii=False), flush=True)
