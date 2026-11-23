"""HTTP API over the v2 agent. From the repo root:

    set PYTHONIOENCODING=utf-8
    uvicorn api.api:app --port 8000

Set API_KEY to require the header X-API-Key on /ask (without it the endpoint is open: do not expose it that way).
"""
import os
import sys
import uuid
import hmac
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # models come from the local cache
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "agent")]

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agent_graph import app as graph

app = FastAPI()


def answer_question(question: str) -> dict:
    cfg = {"configurable": {"thread_id": f"api-{uuid.uuid4()}"}}       # a new conversation per request
    s = graph.invoke({"question": question}, cfg)
    ev = (s.get("evidence") or {}) if s.get("status") == "done" else {}
    sources = [{"id": cid, "source": ev[cid]["source"], "article": ev[cid]["article"]}
               for cid in s.get("cited") or [] if cid in ev]
    return {"answer": s.get("text"), "sources": sources}


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.post("/ask")
def ask(body: Ask, x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    result = answer_question(body.question)
    return {"answer": result["answer"],
            "sources": result["sources"]}
