"""Tests of the HTTP API with a stubbed agent (no model, no network):  python api/test_api.py"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "no-owner-key-configured")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi.testclient import TestClient
import api as A

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or not detail else f": {detail}"))


A.answer_question = lambda q: {"answer": "réponse à " + q, "sources": [{"id": "PenalFr/s02/264", "source": "PenalFr.pdf", "article": "264"}]}
c = TestClient(A.app)
os.environ.pop("API_KEY", None)
r = c.post("/ask", json={"question": "vol ?"})
check("no API_KEY configured: open, returns answer and sources", r.status_code == 200 and r.json()["answer"] == "réponse à vol ?" and r.json()["sources"][0]["article"] == "264")
os.environ["API_KEY"] = "secret-1"
check("API_KEY set, header missing: 401", c.post("/ask", json={"question": "vol ?"}).status_code == 401)
check("API_KEY set, wrong header: 401", c.post("/ask", json={"question": "vol ?"}, headers={"X-API-Key": "nope"}).status_code == 401)
check("API_KEY set, right header: 200", c.post("/ask", json={"question": "vol ?"}, headers={"X-API-Key": "secret-1"}).status_code == 200)
check("empty question: 422", c.post("/ask", json={"question": ""}, headers={"X-API-Key": "secret-1"}).status_code == 422)
check("question over 1000 characters: 422", c.post("/ask", json={"question": "x" * 1001}, headers={"X-API-Key": "secret-1"}).status_code == 422)
print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
