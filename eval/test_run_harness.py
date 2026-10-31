"""Offline tests of the run mechanics of eval/run_answer_eval.py (stub systems, stub judge, temp folders; no network).

    python eval/test_run_harness.py

(a) a stub raising QuotaExhausted on its Nth call: exit 75, the pair in progress not saved, run_state.json + run_log.jsonl;
    a daily-quota error swallowed by a node (degraded + QUOTA_SEEN) is discarded too, not saved as degraded
(b) re-running continues from the next pair; earlier lines byte-identical; no duplicates
(c) a per-minute 429 is retried (agent/llm_retry.py) and does not stop the run; a daily one is not retried
(d) a result file of another commit is refused (unless --allow-mixed)
(e) a changed exam file is refused (unless --allow-changed-exam)
(f) --judge-only judges what has no judgement and skips already judged rows
(g) --wait-on-quota: sleeps, probes, then continues    (h) --daily-budget-tokens stops before the pair, exit 75
(i) error classes: daily vs per-minute messages (Groq, Gemini), retry-after parsing
(j) token usage: the LangChain usage callback sees a chat model called inside a LangGraph node
"""
import io
import sys
import json
import tempfile
import contextlib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "agent"))
import harness_core as hc
import run_answer_eval as rae

RESULTS = []
DAILY = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization "
         "`org_x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199800, Requested 900. "
         "Please try again in 7m12.5s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}")
MINUTE = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
          "minute (TPM): Limit 8000, Used 7900, Requested 600. Please try again in 1.5s.'}}")
GEMINI_DAY = ("429 You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, "
              "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 39.7s.")
GEMINI_MIN = "429 Resource has been exhausted. quotaId: GenerateRequestsPerMinutePerProjectPerModel. retry_delay { seconds: 12 }"


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


ITEMS = [
    {"id": "a1", "question": "Q1 ?", "language": "fr", "expected": "answer", "gold": [{"source": "PenalFr.pdf", "article": "1"}],
     "reference_points": ["p1", "p2"], "reason": "r"},
    {"id": "a2", "question": "Q2 ?", "language": "en", "expected": "answer", "gold": [{"source": "PenalFr.pdf", "article": "2"}],
     "reference_points": ["p1", "p2"], "reason": "r"},
    {"id": "a3", "question": "Q3 ?", "language": "ar", "expected": "answer", "gold": [{"source": "PenalFr.pdf", "article": "3"}],
     "reference_points": ["p1", "p2"], "reason": "r"},
    {"id": "c1", "question": "C1 ?", "language": "fr", "expected": "clarify", "gold": [], "reason": "r"},
    {"id": "r1", "question": "R1 ?", "language": "fr", "expected": "refuse", "gold": [], "reason": "r"},
]


class Stub:
    """System stub: logs (item, system), returns a fixed record; raise_on = {call number: exception}."""

    def __init__(self, raise_on=None, swallow_on=None):
        self.calls, self.raise_on, self.swallow_on = [], raise_on or {}, swallow_on or set()

    def system(self, name):
        def run(item):
            self.calls.append((item["id"], name))
            n = len(self.calls)
            if n in self.raise_on:
                raise self.raise_on[n]
            st = {"answer": "done"}.get(item["expected"], item["expected"])
            rec = {"status": st, "evidence_ids": [], "cited": None, "claims": None, "text": "texte sans nombre",
                   "_evidence_text": {}, "_tokens": {rae.GENERATOR_MODEL: {"input_tokens": 1000, "output_tokens": 200}}}
            if n in self.swallow_on:                       # a v2 node caught the daily-quota error
                hc.QUOTA_SEEN["msg"] = DAILY
                rec["degraded"] = ["verify_failed: RateLimitError"]
            return rec
        return run

    def install(self):
        rae.SYSTEMS = {s: self.system(s) for s in rae.SYSTEM_ORDER}


class JudgeStub:
    def __init__(self):
        self.n = 0

    def __call__(self, provider, model):
        stub = self

        class LLM:
            def invoke(self, messages):
                stub.n += 1
                return SimpleNamespace(covered=[True, True], contradiction=False, unsupported_statements=[])
        return LLM()


def setup(tmp: Path, commit="A"):
    (tmp / "answer_set.json").write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    (tmp / "stages.json").write_text(json.dumps({"stage1": ["a1", "a2", "c1"], "stage2": ["a3", "r1"]}), encoding="utf-8")
    write_hashes(tmp)
    set_commit(commit)


def write_hashes(tmp: Path):
    (tmp / "exam_hashes.txt").write_text(f"eval/answer_set.json {hc.sha256(tmp / 'answer_set.json')}\n"
                                         f"eval/stages.json {hc.sha256(tmp / 'stages.json')}\n", encoding="utf-8")


def set_commit(commit):
    rae.provenance = lambda args, h: {"commit": commit, "tags": [], "dirty_py": [], "agent_model": rae.GENERATOR_MODEL,
                                      "prompt_hashes": {}, "exam_hashes": h, "judge": None}


def run(tmp: Path, *extra) -> tuple[object, str]:
    """main() with the temp exam; -> (exit code or None, captured output)."""
    argv = ["--set", str(tmp / "answer_set.json"), "--stages-file", str(tmp / "stages.json"),
            "--hashes-file", str(tmp / "exam_hashes.txt"), "--results-dir", str(tmp / "results"),
            "--system", "all", "--skip-judge", *extra]
    buf = io.StringIO()
    code = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rae.main(argv)
        except SystemExit as e:
            code = e.code
    return code, buf.getvalue()


def rows(tmp: Path, system: str, stage="all") -> list[dict]:
    p = rae.out_path(tmp / "answer_set.json", system, stage, tmp / "results")
    return rae.read_jsonl(p)


def raw(tmp: Path, system: str, stage="all") -> bytes:
    p = rae.out_path(tmp / "answer_set.json", system, stage, tmp / "results")
    return p.read_bytes() if p.exists() else b""


def test_quota_and_resume():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        stub = Stub(raise_on={5: hc.QuotaExhausted(DAILY, hc.parse_retry_after(DAILY))})
        stub.install()
        code, out = run(tmp)
        check("(a) quota on the 5th call: exit code 75", code == 75, f"code {code!r}\n{out[-600:]}")
        check("(a) question-major order v1, v1.1, v2", stub.calls[:5] == [("a1", "v1"), ("a1", "v1.1"), ("a1", "v2"),
                                                                         ("a2", "v1"), ("a2", "v1.1")], str(stub.calls))
        n = {s: len(rows(tmp, s)) for s in rae.SYSTEM_ORDER}
        check("(a) finished pairs saved, the pair in progress (a2 v1.1) not saved", n == {"v1": 2, "v1.1": 1, "v2": 1}, str(n))
        st = json.loads((tmp / "results" / "run_state.json").read_text(encoding="utf-8"))
        log = (tmp / "results" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
        check("(a) run_state.json: reason, resume estimate, done/remaining",
              st["reason"] == "agent daily quota" and st["resume_estimate"] and st["remaining"] == {"v1.1": 4, "v2": 4, "v1": 3}
              and st["done"] == {"v1": 2, "v1.1": 1, "v2": 1}, json.dumps(st)[:400])
        check("(a) run_log.jsonl appended", len(log) == 1 and json.loads(log[0])["reason"] == "agent daily quota")
        check("(a) STOPPED message", "STOPPED: daily quota" in out and "re-run the same command" in out.lower())
        check("(a) rows record date, provenance, tokens, stage",
              all(k in rows(tmp, "v2")[0] for k in ("date", "provenance", "tokens", "stage"))
              and rows(tmp, "v2")[0]["tokens"]["total"] == 1200)
        before = {s: raw(tmp, s) for s in rae.SYSTEM_ORDER}

        stub2 = Stub(swallow_on={2})                       # 2nd call of the resumed run: a2 v2, swallowed quota
        stub2.install()
        code, out = run(tmp)
        check("(a) quota swallowed by a node: exit 75, not saved as degraded",
              code == 75 and not any(r.get("status") == "degraded" for s in rae.SYSTEM_ORDER for r in rows(tmp, s)),
              f"code {code!r}")
        check("(b) resume starts at the discarded pair (a2 v1.1)", stub2.calls[0] == ("a2", "v1.1"), str(stub2.calls))

        stub3 = Stub()
        stub3.install()
        code, out = run(tmp)
        check("(b) resumed run completes", code is None, f"code {code!r}\n{out[-500:]}")
        check("(b) resume starts at the next unfinished pair (a2 v2)", stub3.calls[0] == ("a2", "v2"), str(stub3.calls))
        check("(b) earlier lines byte-identical", all(raw(tmp, s).startswith(before[s]) for s in rae.SYSTEM_ORDER))
        ids = {s: [r["id"] for r in rows(tmp, s)] for s in rae.SYSTEM_ORDER}
        check("(b) no duplicates, every item once per system",
              all(sorted(v) == sorted(i["id"] for i in ITEMS) for v in ids.values()), str(ids))
        check("(b) completion written to run_state.json",
              json.loads((tmp / "results" / "run_state.json").read_text(encoding="utf-8"))["reason"] == "completed")
        code, out = run(tmp)
        check("(b) a third run has nothing to do", code is None and "0 (item, system) runs to do" in out)

        set_commit("B")                                    # (d)
        code, out = run(tmp)
        check("(d) another commit is refused", isinstance(code, str) and code.startswith("REFUSED") and "pooled" in code,
              str(code)[:300])
        code, out = run(tmp, "--allow-mixed")
        check("(d) --allow-mixed goes on with a warning", code is None and "WARNING (--allow-mixed)" in out, str(code))


def test_per_minute_retry():
    import llm_retry
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        slept, attempts = [], []

        def system(item):
            attempts.append(item["id"])

            class R:
                n = 0

                def invoke(self, messages):
                    R.n += 1
                    if R.n == 1:
                        raise RuntimeError(MINUTE)
                    return "ok"
            llm_retry.invoke_with_retry(R(), [], "stub", sleep=slept.append)
            return {"status": {"answer": "done"}.get(item["expected"], item["expected"]), "evidence_ids": [],
                    "cited": None, "claims": None, "text": "t", "_evidence_text": {}}
        rae.SYSTEMS = {s: system for s in rae.SYSTEM_ORDER}
        code, out = run(tmp, "--stage", "1")
        check("(c) per-minute 429 retried, run completes", code is None and len(rows(tmp, "v2", "1")) == 3
              and len(slept) == 9 and abs(slept[0] - 2.5) < 1e-9, f"code {code!r}, slept {slept}")

        calls = []

        def daily_system(item):
            class R:
                def invoke(self, messages):
                    calls.append(1)
                    raise RuntimeError(DAILY)
            return llm_retry.invoke_with_retry(R(), [], "stub", sleep=slept.append)
        rae.SYSTEMS = {s: daily_system for s in rae.SYSTEM_ORDER}
        code, out = run(tmp, "--stage", "2")
        check("(c) a daily 429 through llm_retry is not retried and stops with 75", code == 75 and len(calls) == 1,
              f"code {code!r}, attempts {len(calls)}")


def test_changed_exam():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        Stub().install()
        changed = ITEMS[:4] + [dict(ITEMS[4], question="R1 changed ?")]
        (tmp / "answer_set.json").write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
        code, out = run(tmp, "--dry-run")
        check("(e) a changed exam is refused", isinstance(code, str) and code.startswith("REFUSED: the exam changed"),
              str(code)[:200])
        code, out = run(tmp, "--allow-changed-exam")
        check("(e) --allow-changed-exam runs, rows record the new hash",
              code is None and rows(tmp, "v2")[0]["provenance"]["exam_hashes"]["eval/answer_set.json"]
              == hc.sha256(tmp / "answer_set.json"), str(code))


def test_judge_only():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        Stub().install()
        code, out = run(tmp)
        js = JudgeStub()
        rae.JUDGE_FACTORY = js
        before = raw(tmp, "v1")
        argv = ["--set", str(tmp / "answer_set.json"), "--stages-file", str(tmp / "stages.json"), "--hashes-file",
                str(tmp / "exam_hashes.txt"), "--results-dir", str(tmp / "results"), "--system", "all",
                "--judge-only", "--judge-model", "stub-judge"]
        with contextlib.redirect_stdout(io.StringIO()):
            rae.main(argv)
        judged = [r for s in rae.SYSTEM_ORDER for r in rows(tmp, s) if r.get("judge")]
        check("(f) --judge-only judges every answered item (3 items x 3 systems)", js.n == 9 and len(judged) == 9
              and all(r["judge"]["verdict"] == "correct" and not r["judge_pending"] for r in judged), f"calls {js.n}")
        check("(f) clarify/refuse rows untouched", raw(tmp, "v1").count(b"\n") == before.count(b"\n")
              and [l for l in raw(tmp, "v1").splitlines() if b'"c1"' in l] == [l for l in before.splitlines() if b'"c1"' in l])
        with contextlib.redirect_stdout(io.StringIO()):
            rae.main(argv)
        check("(f) a second --judge-only skips already judged rows", js.n == 9, f"calls {js.n}")


def test_wait_and_budget():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        stub = Stub(raise_on={2: hc.QuotaExhausted(DAILY, 30.0)})
        stub.install()
        slept, probes = [], []
        rae.SLEEP = slept.append
        rae.probe_agent = lambda: probes.append(1) or len(probes) >= 2
        code, out = run(tmp, "--wait-on-quota", "1", "--stage", "1")
        check("(g) --wait-on-quota: sleeps, probes until the quota is back, then completes",
              code is None and len(probes) == 2 and slept == [35.0, 60.0] and len(rows(tmp, "v1.1", "1")) == 3
              and stub.calls[1] == stub.calls[2] == ("a1", "v1.1"), f"code {code!r}, slept {slept}, calls {stub.calls[:4]}")
        Stub().install()
        code, out = run(tmp, "--stage", "2", "--daily-budget-tokens", "4000")
        n = sum(len(rows(tmp, s, "2")) for s in rae.SYSTEM_ORDER)
        check("(h) --daily-budget-tokens stops before the pair that would pass it, exit 75",
              code == 75 and "daily token budget" in out and n == 0, f"code {code!r}, rows {n}\n{out[-300:]}")


def test_classes():
    q = hc.classify(RuntimeError(DAILY))
    check("(i) Groq TPD 429 -> QuotaExhausted, retry after 432.5 s", q is not None and abs(q.retry_after - 432.5) < 1e-6)
    check("(i) Groq TPM 429 -> not QuotaExhausted", hc.classify(RuntimeError(MINUTE)) is None)
    check("(i) requests per day (RPD) -> QuotaExhausted",
          hc.classify(RuntimeError("Error code: 429 - Rate limit reached on requests per day (RPD): Limit 1000")) is not None)
    g = hc.classify(RuntimeError(GEMINI_DAY), "judge")
    check("(i) Gemini PerDay 429 -> QuotaExhausted (judge), retry 39.7 s", g is not None and g.who == "judge"
          and abs(g.retry_after - 39.7) < 1e-6)
    check("(i) Gemini PerMinute 429 -> not QuotaExhausted; retry_delay parsed",
          hc.classify(RuntimeError(GEMINI_MIN)) is None and hc.parse_retry_after(GEMINI_MIN) == 12.0)
    wrapped = RuntimeError("verify: Groq rate limit (429) after 0 retries")
    wrapped.__cause__ = RuntimeError(DAILY)
    check("(i) a wrapped error is classified through its cause", hc.classify(wrapped) is not None)


def test_usage_in_graph():
    from typing import TypedDict
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.graph import StateGraph, START, END
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="x", response_metadata={"model_name": rae.GENERATOR_MODEL},
                                                        usage_metadata={"input_tokens": 70, "output_tokens": 5,
                                                                        "total_tokens": 75})]))

    class S(TypedDict):
        q: str

    g = StateGraph(S)
    g.add_node("n", lambda s: (llm.invoke(s["q"]), {"q": s["q"]})[1])
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile()
    with rae.usage_callback() as cb:
        app.invoke({"q": "hi"})
    t = rae.token_record(cb.usage_metadata, None)
    check("(j) usage callback sees the model call inside a LangGraph node", t["total"] == 75 and t["source"] == "usage_metadata",
          json.dumps(t))
    e = rae.token_record({}, None, est_in="x" * 400, est_out="y" * 40, calls=2)
    check("(j) no usage metadata -> characters/4 estimate, labelled", e["total"] == 110 and "estimated" in e["source"])


if __name__ == "__main__":
    for t in (test_quota_and_resume, test_per_minute_retry, test_changed_exam, test_judge_only, test_wait_and_budget,
              test_classes, test_usage_in_graph):
        t()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed" + ("" if n_fail else ": all checks passed"))
    sys.exit(1 if n_fail else 0)
