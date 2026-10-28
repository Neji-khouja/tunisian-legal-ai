"""Runs the v1 pipeline (tag v1-baseline-runnable) from a git worktree, in its own process. Used by
eval/run_answer_eval.py --system v1; reads {"id", "question"} JSON lines on stdin, writes one JSON line per item.

    python eval/v1_runner.py <worktree>      (cwd = <worktree>, which must contain the old index in db/)

Only the worktree's modules are imported (its config, pipeline, retriever, ...), never the current ones.
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

vs = load_vectorstore()
for line in sys.stdin:
    item = json.loads(line)
    before, t0 = calls["n"], time.time()
    try:
        answer, docs = ask_question(vs, item["question"], None)
        out = {"id": item["id"], "answer": answer,
               "docs": [{"source": d.metadata.get("source"), "article": d.metadata.get("article"),
                         "text": d.page_content} for d in docs]}
    except Exception as e:
        out = {"id": item["id"], "error": f"{type(e).__name__}: {e}"}
    out["seconds"] = round(time.time() - t0, 1)
    out["llm_calls"] = calls["n"] - before
    print(json.dumps(out, ensure_ascii=False), flush=True)
