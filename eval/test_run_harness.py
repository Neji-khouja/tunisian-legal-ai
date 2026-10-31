"""Offline tests of the run mechanics of eval/run_answer_eval.py (stub systems, stub judge, temp folders; no network).

    python eval/test_run_harness.py

(a) a stub raising QuotaExhausted on its Nth call: exit 75, the pair in progress not saved, run_state.json + run_log.jsonl;
    a daily-quota error swallowed by a node (degraded + QUOTA_SEEN) is discarded too, not saved as degraded
(b) re-running continues from the next pair; earlier lines byte-identical; no duplicates
(c) a per-minute 429 is retried (agent/llm_retry.py) and does not stop the run; a daily one is not retried
(d) a result file of another system fingerprint is refused (unless --allow-mixed); another commit alone is not
(e) a changed exam file is refused (unless --allow-changed-exam)
(f) --judge-only judges what has no judgement and skips already judged rows
(g) --wait-on-quota: sleeps, probes, then continues    (h) --daily-budget-tokens stops before the pair, exit 75
(i) error classes: daily vs per-minute messages (Groq, Gemini), retry-after parsing
(j) token usage: the LangChain usage callback sees a chat model called inside a LangGraph node
(k) another --set (plumbing): no fingerprint, own --results-dir; never in the folder of the stage results (and back)
(l) a failing judge check stops before any agent call; a judge error mid-run keeps the answers as judge_pending
(m) eval/judge_check.py with a stub judge: verdicts, successes, the selection rule; the Gemini thinking argument
(n) the clarify/refuse judge (GATE_PROMPT) runs on every clarify/refuse row of every system; --summarize metrics
(o) the 5 exam clarify items with a scripted reply have gold: "Hit@6 after the reply" is computed on them (n = 5)
(p) eval/run_batch.py with a stub runner: next K ids, one process per system, exit 75 forwarded, other failures go on
(q) crash isolation: a parse error once (recovers in the same run), on both tries (error row, the run goes on), in 3
    runs (failed: final, "system failure" in --summarize); a quota error still exits 75
(s) clarify behaviour metrics on the clarify items without a scripted reply; "clarify then answer" on the reply items
(t) judge thinking auto (minimal -> low on a 400, kept), retry on json_validate_failed / parser errors, judge_check
    --candidates order, stop at the first passing, log, --min-gap, error texts
(r) system fingerprint: a harness (eval/) or agent test file change keeps it, an agent file or index change does not;
    --judge-only ignores provenance
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


JUDGED = dict(covered=[True, True], contradiction=False, unsupported_statements=[], gives_substantive_answer=True,
              enumerates_alternatives=False)


class JudgeStub:
    def __init__(self):
        self.n, self.kinds = 0, []

    def __call__(self, provider, model, kind="answer"):
        stub = self

        class LLM:
            def invoke(self, messages):
                stub.n += 1
                stub.kinds.append(kind)
                return SimpleNamespace(**JUDGED)
        return LLM()


def setup(tmp: Path, commit="A"):
    (tmp / "answer_set.json").write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    (tmp / "stages.json").write_text(json.dumps({"stage1": ["a1", "a2", "c1"], "stage2": ["a3", "r1"]}), encoding="utf-8")
    write_hashes(tmp)
    set_commit(commit)


def write_hashes(tmp: Path):
    (tmp / "exam_hashes.txt").write_text(f"eval/answer_set.json {hc.sha256(tmp / 'answer_set.json')}\n"
                                         f"eval/stages.json {hc.sha256(tmp / 'stages.json')}\n", encoding="utf-8")


def set_commit(commit, fp="F"):
    rae.provenance = lambda args, h: {"fingerprint": {"sha256": fp}, "commit": commit, "tags": [], "dirty_py": [],
                                      "agent_model": rae.GENERATOR_MODEL, "prompt_hashes": {}, "exam_hashes": h,
                                      "judge": None}


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

        set_commit("B")                                    # (d) a new commit, same system fingerprint
        code, out = run(tmp)
        check("(d) another commit with the same fingerprint goes on", code is None and "WARNING" not in out, str(code)[:300])
        set_commit("B", fp="G")
        code, out = run(tmp)
        check("(d) another fingerprint is refused", isinstance(code, str) and code.startswith("REFUSED")
              and "fingerprint" in code and "pooled" in code, str(code)[:300])
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


def test_other_set():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        Stub().install()
        plumb = tmp / "plumbing_set.json"
        plumb.write_text(json.dumps([ITEMS[0], ITEMS[3], ITEMS[4]], ensure_ascii=False), encoding="utf-8")
        (tmp / "exam_hashes.txt").write_text("eval/answer_set.json 0\neval/stages.json 0\n", encoding="utf-8")  # stale
        argv = ["--set", str(plumb), "--stages-file", str(tmp / "stages.json"), "--hashes-file", str(tmp / "exam_hashes.txt"),
                "--system", "all", "--skip-judge"]

        def go(*extra):
            buf, code = io.StringIO(), None
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    rae.main(argv + list(extra))
                except SystemExit as e:
                    code = e.code
            return code, buf.getvalue()
        code, out = go("--results-dir", str(tmp / "plumbing"))
        made = sorted(p.name for p in (tmp / "plumbing").glob("*.jsonl"))
        r = rae.read_jsonl(tmp / "plumbing" / "answer_eval_plumbing_set_v2.jsonl")
        check("(k) another --set runs without the exam fingerprint, into its own --results-dir",
              code is None and "answer_eval_plumbing_set_v1.jsonl" in made and len(r) == 3
              and r[0]["provenance"]["exam_hashes"] == {"plumbing_set.json": hc.sha256(plumb)}, f"{code!r} {made}")
        code, out = run(tmp, "--dry-run")
        check("(k) the fingerprint stays enforced for the exam", isinstance(code, str) and code.startswith("REFUSED: the exam"),
              str(code)[:200])
        write_hashes(tmp)
        run(tmp, "--stage", "1")
        code, out = go("--results-dir", str(tmp / "results"))
        check("(k) another set is refused in the folder of the stage results", isinstance(code, str)
              and "not the exam" in code and not list((tmp / "results").glob("answer_eval_plumbing_set_*")), str(code)[:200])
        code, out = go("--results-dir", str(rae.ROOT / "results"), "--dry-run")
        check("(k) another set is refused in results/", isinstance(code, str) and "not the exam" in code, str(code)[:200])
        argv[1] = str(tmp / "answer_set.json")
        code, out = go("--results-dir", str(tmp / "plumbing"), "--stage", "1")
        check("(k) the exam is refused in a folder holding another set", isinstance(code, str)
              and "another set" in code, str(code)[:200])
        argv[1] = str(plumb)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rae.main(argv + ["--results-dir", str(tmp / "plumbing"), "--summarize"])
        heads = [l for l in buf.getvalue().splitlines() if l.startswith("=== ") and ".jsonl:" in l]
        check("(k) --summarize on that folder reads the set's 3 files", len(heads) == 3
              and all(l.startswith("=== answer_eval_plumbing_set_") for l in heads), "\n".join(heads))


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
        gated = [r for s in rae.SYSTEM_ORDER for r in rows(tmp, s) if r.get("gate_judge")]
        check("(f) --judge-only judges every answered item (3 items x 3 systems)", js.kinds.count("answer") == 9
              and len(judged) == 9 and all(r["judge"]["verdict"] == "correct" and not r["judge_pending"] for r in judged),
              f"calls {js.kinds}")
        check("(f) --judge-only applies GATE_PROMPT to every clarify/refuse row (2 items x 3 systems), no other line",
              js.kinds.count("gate") == 6 and len(gated) == 6 and raw(tmp, "v1").count(b"\n") == before.count(b"\n")
              and all(r["gate_judge"]["prompt_sha"] == rae.GATE_PROMPT_SHA for r in gated), f"calls {js.kinds}")
        with contextlib.redirect_stdout(io.StringIO()):
            rae.main(argv)
        check("(f) a second --judge-only skips already judged rows", js.n == 15, f"calls {js.n}")


def test_judge_errors():
    class Failing:
        def __init__(self, fail_from):
            self.n, self.fail_from = 0, fail_from

        def __call__(self, provider, model, kind="answer"):
            stub = self

            class LLM:
                def invoke(self, messages):
                    stub.n += 1
                    if stub.n >= stub.fail_from:
                        raise RuntimeError("404 NOT_FOUND. models/x is no longer available to new users.")
                    return SimpleNamespace(**JUDGED)
            return LLM()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        stub = Stub()
        stub.install()
        rae.JUDGE_FACTORY = Failing(1)
        argv = ["--set", str(tmp / "answer_set.json"), "--stages-file", str(tmp / "stages.json"), "--hashes-file",
                str(tmp / "exam_hashes.txt"), "--results-dir", str(tmp / "results"), "--system", "all",
                "--judge-model", "stub-judge", "--stage", "1"]
        buf, code = io.StringIO(), None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rae.main(argv)
            except SystemExit as e:
                code = e.code
        check("(l) a failing judge check stops before any agent call", isinstance(code, str)
              and "judge check failed" in code and stub.calls == [], f"{str(code)[:150]} calls {stub.calls[:2]}")
        rae.JUDGE_FACTORY = Failing(3)            # check ok, a1 v1 judged, fails on a1 v1.1
        buf, code = io.StringIO(), None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rae.main(argv)
            except SystemExit as e:
                code = e.code
        rs = {s: rows(tmp, s, "1") for s in rae.SYSTEM_ORDER}
        state = json.loads((tmp / "results" / "run_state.json").read_text(encoding="utf-8"))
        check("(l) a judge error mid-run keeps the answers (judge_pending), the run completes",
              code is None and all(len(r) == 3 for r in rs.values()) and rs["v1"][0]["judge"]["verdict"] == "correct"
              and rs["v1.1"][0]["judge_pending"] and "NOT_FOUND" in (state.get("judge_error") or ""),
              f"code {code!r} {buf.getvalue()[-300:]}")


def test_gate_judge():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        Stub().install()
        js = JudgeStub()
        rae.JUDGE_FACTORY = js
        argv = ["--set", str(tmp / "answer_set.json"), "--stages-file", str(tmp / "stages.json"), "--hashes-file",
                str(tmp / "exam_hashes.txt"), "--results-dir", str(tmp / "results"), "--system", "all",
                "--judge-model", "stub-judge", "--stage", "1"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rae.main(argv)
        c1 = [r for s in rae.SYSTEM_ORDER for r in rows(tmp, s, "1") if r["id"] == "c1"]
        check("(n) a normal run applies GATE_PROMPT to the clarify row of every system", len(c1) == 3
              and all(r["gate_judge"]["prompt_sha"] == rae.GATE_PROMPT_SHA and not r["judge_pending"] for r in c1),
              str([r.get("gate_judge") for r in c1])[:300])
    ref, cla = ITEMS[4], dict(ITEMS[3], reply="R ?", gold=[{"source": "PenalFr.pdf", "article": "9"}])

    def row(item, status, gj, **kw):
        rec = {"status": status, "evidence_ids": kw.pop("ev", []), "cited": None, "claims": None, "text": "t", **kw}
        g = None if gj is None else {"gives_substantive_answer": gj[0], "enumerates_alternatives": gj[1]}
        return {"id": item["id"], "expected": item["expected"], **rec, "metrics": rae.deterministic(item, rec),
                "gate_judge": g}
    vals = rae.metric_values([
        row(ref, "done", (False, False)),        # "not found": no substantive answer, no refusal status
        row(ref, "done", (True, False)),         # answered instead of refusing
        row(ref, "refuse", None),                # explicit refusal (no judgement needed)
        row(cla, "done", (True, True), asked="C1 ? (précision : R ?)", ev=["PenalFr/s01/9"]),
        row(cla, "done", None, first_status="clarify", first_text="?", reply="R ?", ev=["PenalFr/s01/8"]),
    ])
    want = {"refuse: no substantive answer": [1, 0, 1], "refuse: explicit refusal status": [0, 0, 1],
            "gate accuracy (explicit status)": [0, 0, 1, 0, 1]}
    got = {k: vals[k] for k in want}
    check("(n) summary: refuse = explicit refusal OR no substantive answer; gate accuracy renamed",
          got == want and "gate accuracy (clarify/refuse items)" not in vals, str(got))


def test_clarify_subsets():
    plain, cla = ITEMS[3], dict(ITEMS[3], reply="R ?", gold=[{"source": "PenalFr.pdf", "article": "9"}])

    def row(item, status, gj, **kw):
        rec = {"status": status, "evidence_ids": kw.pop("ev", []), "cited": None, "claims": None, "text": "t", **kw}
        g = None if gj is None else {"gives_substantive_answer": gj[0], "enumerates_alternatives": gj[1]}
        return {"id": item["id"], "expected": item["expected"], **rec, "metrics": rae.deterministic(item, rec),
                "gate_judge": g}
    rs = [row(plain, "done", (True, True)),                      # no reply: answered directly, enumerated
          row(plain, "clarify", (False, False)),                 # no reply: asked
          row(cla, "done", (True, False), asked="C1 ? (précision : R ?)", ev=["PenalFr/s01/9"],
              judge={"verdict": "correct"}),                     # reply: gold hit, judged correct
          row(cla, "done", None, first_status="clarify", first_text="?", reply="R ?", ev=["PenalFr/s01/8"])]
    vals = rae.metric_values(rs)
    want = {"clarify w/o reply: asked for clarification (status)": [0, 1],
            "clarify w/o reply: answered directly": [1, 0], "clarify w/o reply: enumerated alternatives": [1, 0],
            "clarify then answer: correct after the reply": [1, None],
            "clarify then answer: Hit@6 after the reply": [1, 0], "gate accuracy (explicit status)": [0, 1, 0, 1]}
    got = {k: vals[k] for k in want}
    check("(s) clarify behaviour on the items without a reply only; reply items under 'clarify then answer'",
          got == want and not [k for k in vals if k.startswith("clarify:")], str(dict(vals))[:500])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rae.print_block("stub", [dict(r, provenance={}) for r in rs])
    lines = {l.split(" n=")[0].strip(): l.split(" n=")[1].split()[0] for l in buf.getvalue().splitlines() if " n=" in l}
    check("(s) --summarize prints n and the subset name in each label",
          lines.get("clarify w/o reply: asked for clarification (status)") == "2"
          and lines.get("clarify then answer: correct after the reply") == "1"
          and lines.get("clarify then answer: Hit@6 after the reply") == "2", str(lines))


def test_reply_gold():
    exam = json.loads((ROOT / "eval" / "answer_set.json").read_text(encoding="utf-8"))
    replies = [i for i in exam if i.get("reply")]
    rs = []
    for k, item in enumerate(replies):
        cid = item["gold"][0]["chunk_ids"][0] if k % 2 == 0 else "PenalFr/s02/264"    # hit, miss, hit, miss, hit
        rec = {"status": "done", "evidence_ids": [cid], "cited": None, "claims": None, "text": "t", "asked": rae.asked(item)}
        rs.append({"id": item["id"], "expected": item["expected"], **rec, "metrics": rae.deterministic(item, rec)})
    xs = rae.metric_values(rs)["clarify then answer: Hit@6 after the reply"]
    check("(o) exam: 5 reply items with gold, Hit@6 after the reply non-empty (n=5)",
          sorted(i["id"] for i in replies if i["gold"]) == ["c01", "c03", "c06", "c11", "c15"] and xs == [1, 0, 1, 0, 1],
          f"{[i['id'] for i in replies]} {xs}")


def test_run_batch():
    import run_batch as rb
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        res = tmp / "results"
        res.mkdir()
        for s, ids in {"v1": ["a1", "a2"], "v1.1": ["a1"], "v2": ["a1"]}.items():   # a1 finished everywhere
            rae.out_path(tmp / "answer_set.json", s, "1", res).write_text(
                "".join(json.dumps({"id": i, "status": "done"}) + "\n" for i in ids), encoding="utf-8")
        calls = []

        def fake(codes):
            def run(cmd, cwd=None):
                sysname = cmd[cmd.index("--system") + 1]
                calls.append((sysname, cmd[cmd.index("--only-ids") + 1], "--min-gap" in cmd))
                return SimpleNamespace(returncode=codes.get(sysname, 0))
            return run

        def go(codes, *extra):
            calls.clear()
            rb.RUN = fake(codes)
            buf, code = io.StringIO(), None
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    rb.main(["--stage", "1", "--set", str(tmp / "answer_set.json"), "--stages-file", str(tmp / "stages.json"),
                             "--results-dir", str(res), "--questions-per-run", "2", "--min-gap", "3", *extra])
                except SystemExit as e:
                    code = e.code
            return code, buf.getvalue()
        code, out = go({})
        check("(p) run_batch: next 2 unfinished ids (a2, c1), v1 v1.1 v2 in separate runs, extra flags passed on",
              code is None and calls == [("v1", "a2,c1", True), ("v1.1", "a2,c1", True), ("v2", "a2,c1", True)]
              and "remaining" in out and "day(s)" in out, f"{code!r} {calls}")
        code, out = go({"v1.1": 75})
        check("(p) run_batch: exit 75 is forwarded at once (v2 not run)", code == 75 and [c[0] for c in calls] == ["v1", "v1.1"],
              f"{code!r} {calls}")
        code, out = go({"v1": 1})
        check("(p) run_batch: another failure is reported and the next systems run (exit 1 at the end)",
              code == 1 and [c[0] for c in calls] == ["v1", "v1.1", "v2"] and "v1 FAILED" in out, f"{code!r} {calls}")
        rae.out_path(tmp / "answer_set.json", "v1", "1", res).write_text(
            "".join(json.dumps({"id": i, "status": "done"}) + "\n" for i in ["a1", "a2", "c1"]), encoding="utf-8")
        code, out = go({})
        check("(p) run_batch: a system with the whole batch finished is skipped", [c[0] for c in calls] == ["v1.1", "v2"],
              str(calls))


def test_crash_isolation():
    from langchain_core.exceptions import OutputParserException
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        stub = Stub(raise_on={1: OutputParserException("Invalid json output: {'description': ...}")})
        stub.install()
        code, out = run(tmp, "--stage", "1")
        a1 = {r["id"]: r for r in rows(tmp, "v1", "1")}["a1"]
        check("(q) a parse error once: tried again in the same run, row done", code is None and a1["status"] == "done"
              and stub.calls[:2] == [("a1", "v1"), ("a1", "v1")], f"{code!r} {stub.calls[:3]}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        good = Stub().system("v1")
        tries = []

        def flaky(item):                                   # a1 on v1 always raises a parse error
            if item["id"] == "a1":
                tries.append(1)
                raise OutputParserException("Invalid json output")
            return good(item)
        Stub().install()
        rae.SYSTEMS["v1"] = flaky
        statuses = []
        for k in range(4):
            code, out = run(tmp, "--stage", "1")
            r = {r["id"]: r for r in rows(tmp, "v1", "1")}["a1"]
            statuses.append((code, r["status"], r.get("attempt"), len(tries)))
        n = {s: len({r["id"] for r in rows(tmp, s, "1")}) for s in rae.SYSTEM_ORDER}
        check("(q) a parse error on both tries: error row (type + message), the run goes on",
              statuses[0] == (None, "error", 1, 2) and n == {"v1": 3, "v1.1": 3, "v2": 3}
              and r["error"].startswith("OutputParserException: Invalid json"), f"{statuses} {n}")
        check("(q) resumes retry the error row; the 3rd failed attempt is final (failed), not retried after",
              [x[1:] for x in statuses] == [("error", 1, 2), ("error", 2, 4), ("failed", 3, 6), ("failed", 3, 6)],
              str(statuses))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rae.summarize([rae.out_path(tmp / "answer_set.json", "v1", "1", tmp / "results")])
        o = buf.getvalue()
        check("(q) --summarize: system failure shown separately, sensitivity line counts it as incorrect",
              "system failure (no answer, excluded): 1" in o and "answer items 1, clarify/refuse items 0" in o
              and "sensitivity: correct, failures = 0" in o, o[-700:])
        rae.SYSTEMS["v1"] = lambda item: (_ for _ in ()).throw(hc.QuotaExhausted(DAILY, 30.0))
        code, out = run(tmp, "--stage", "2")
        check("(q) a quota error is not isolated: exit 75", code == 75, f"{code!r}")


def test_fingerprint():
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for f in ["agent/answer.py", "agent/test_graph_logic.py", "agent/run_demo.py", "eval/run_answer_eval.py",
                  "config.py", "splitter.py", "retriever.py", "reranker.py", "pipeline.py"]:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_text(f"# {f}\n", encoding="utf-8")
        (root / "db").mkdir()
        con = sqlite3.connect(root / "db" / "chroma.sqlite3")
        con.execute("create table embeddings (id integer)")
        con.executemany("insert into embeddings values (?)", [(1,), (2,), (3,)])
        con.commit()
        fp = lambda: hc.system_fingerprint(root, root / "db")
        f0 = fp()
        (root / "eval" / "run_answer_eval.py").write_text("# harness changed\n", encoding="utf-8")
        (root / "agent" / "test_graph_logic.py").write_text("# test changed\n", encoding="utf-8")
        (root / "agent" / "run_demo.py").write_text("# runner changed\n", encoding="utf-8")
        check("(r) fingerprint: a harness file, an agent test_*/run_* file change does not trigger it",
              fp() == f0 and f0["files"] == 6 and f0["index"] == {"path": "db", "chunks": 3}, json.dumps(f0))
        (root / "agent" / "answer.py").write_text("# agent changed\n", encoding="utf-8")
        f1 = fp()
        check("(r) fingerprint: an agent file change triggers it", f1["sha256"] != f0["sha256"])
        con.execute("insert into embeddings values (4)")
        con.commit()
        con.close()
        check("(r) fingerprint: an index with another chunk count triggers it", fp()["sha256"] != f1["sha256"])
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        Stub().install()
        run(tmp, "--stage", "1")
        set_commit("A", fp="G")
        rae.JUDGE_FACTORY = JudgeStub()
        argv = ["--set", str(tmp / "answer_set.json"), "--stages-file", str(tmp / "stages.json"), "--hashes-file",
                str(tmp / "exam_hashes.txt"), "--results-dir", str(tmp / "results"), "--system", "all", "--stage", "1",
                "--judge-only", "--judge-model", "stub-judge"]
        buf, code = io.StringIO(), None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rae.main(argv)
            except SystemExit as e:
                code = e.code
        check("(r) --judge-only ignores provenance (rows of another fingerprint judged, no refusal)",
              code is None and all(r.get("judge") for r in rows(tmp, "v2", "1") if r["expected"] == "answer"),
              f"{str(code)[:200]}")


def test_judge_thinking_retry():
    import judge_check as jc
    levels = []
    unsupported = RuntimeError("400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Thinking level MINIMAL is not "
                               "supported for this model.', 'status': 'INVALID_ARGUMENT'}}")

    def factory(provider, model, kind="answer"):
        levels.append(rae.thinking_level())

        class LLM:
            def invoke(self, messages):
                if rae.thinking_level() == "minimal":
                    raise unsupported
                return SimpleNamespace(**JUDGED)
        return LLM()
    rae.JUDGE_FACTORY = factory
    rec = {"text": "t", "cited": [], "_evidence_text": {}}
    rae.set_thinking("auto")
    j1 = rae.judge(ITEMS[0], rec, "gemini-3.8-flash", "gemini")
    j2 = rae.judge(ITEMS[0], rec, "gemini-3.8-flash", "gemini")
    check("(t) thinking auto: minimal -> 400 not supported -> once with low, kept; effective level in the judgement",
          levels == ["minimal", "low", "low"] and j1["thinking"] == j2["thinking"] == {"thinking_level": "low"}
          and j1["verdict"] == "correct", f"{levels} {j1.get('thinking')}")
    rae.set_thinking("minimal")
    try:
        rae.judge(ITEMS[0], rec, "gemini-3.8-flash", "gemini")
        raised = False
    except RuntimeError:
        raised = True
    check("(t) thinking minimal (not auto): the 400 is not retried with low", raised and rae.thinking_level() == "minimal")
    rae.set_thinking("auto")
    check("(t) groq: thinking n/a (None)", rae.effective_thinking("groq", "openai/gpt-oss-20b") is None)

    bad_json = RuntimeError("Error code: 400 - {'error': {'message': 'Failed to validate JSON. Please adjust your "
                            "prompt.', 'type': 'invalid_request_error', 'code': 'json_validate_failed'}}")

    class Flaky:
        def __init__(self, n_bad, err=bad_json):
            self.n, self.n_bad, self.err = 0, n_bad, err

        def invoke(self, messages):
            self.n += 1
            if self.n <= self.n_bad:
                raise self.err
            return "ok"
    f2, f3 = Flaky(2), Flaky(3)
    out = rae.invoke_judge(f2, [], "groq")
    try:
        rae.invoke_judge(f3, [], "groq")
        raised = False
    except RuntimeError:
        raised = True
    from langchain_core.exceptions import OutputParserException
    fp = Flaky(1, OutputParserException("Invalid json output: {"))
    fq = Flaky(1, RuntimeError(DAILY))
    try:
        rae.invoke_judge(fq, [], "groq")
        quota = False
    except hc.QuotaExhausted:
        quota = True
    check("(t) json_validate_failed: retried up to 2 more times (3 calls), then an error; parser error retried; "
          "a daily quota still stops at once",
          out == "ok" and f2.n == 3 and raised and f3.n == 3 and rae.invoke_judge(fp, [], "gemini") == "ok" and fp.n == 2
          and quota and fq.n == 1, f"{f2.n} {f3.n} {fp.n} {fq.n}")

    class ByModel:
        def __init__(self):
            self.models = []

        def __call__(self, provider, model, kind="answer"):
            stub = self
            stub.models.append(model)

            class LLM:
                def invoke(self, messages):
                    if model == "bad":
                        raise RuntimeError(MINUTE)
                    good = "cinq ans d'emprisonnement" in messages[1][1].split("Answer:")[1].split("Cited articles:")[0]
                    return SimpleNamespace(covered=[good, good], contradiction=not good, unsupported_statements=[])
            return LLM()
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "judge_check_log.json"
        bm, slept = ByModel(), []
        rae.JUDGE_FACTORY = bm
        ticks = iter(range(0, 10000, 3))
        buf, code = io.StringIO(), None
        with contextlib.redirect_stdout(buf):
            try:
                jc.main(["--candidates", "groq:bad,groq:good,groq:never", "--log", str(log)], sleep=slept.append,
                        clock=lambda: next(ticks))
            except SystemExit as e:
                code = e.code
        o, L = buf.getvalue(), json.loads(log.read_text(encoding="utf-8"))
        check("(t) --candidates: in order, stops at the first meeting the rule (never not called)",
              code is None and "SELECTED: groq:good" in o and set(bm.models) == {"bad", "good"}
              and bm.models.index("good") > bm.models.index("bad"), f"{code!r} {sorted(set(bm.models))}")
        check("(t) log: every attempt (candidate, thinking, time, results, error texts) and the earlier attempts as note",
              [a["candidate"] for a in L["attempts"]] == ["groq:bad", "groq:good"] and L["note"] == jc.EARLIER
              and all(k in L["attempts"][0] for k in ("thinking_effective", "time", "results", "errors"))
              and "TPM" in L["attempts"][0]["errors"]["RuntimeError 429"], json.dumps(L)[:400])
        check("(t) --min-gap 4 s (default) between calls: 5 sleeps per candidate", slept == [4.0] * 10, str(slept))
        check("(t) the first full error text per type, 429 period / limit / retry delay",
              "first error of type RuntimeError 429" in o
              and "429 looks per-minute; quota metric not stated; limit 8000; retry delay 1.5 s" in o, o[-600:])


def test_judge_check():
    import judge_check as jc
    from pdf_truth import from_chunk_id

    class ByContent:
        def __init__(self, fail=0):
            self.fail = fail

        def __call__(self, provider, model, kind="answer"):
            stub = self

            class LLM:
                def invoke(self, messages):
                    if stub.fail:
                        stub.fail -= 1
                        raise RuntimeError("ReadTimeout: The read operation timed out")
                    good = "cinq ans d'emprisonnement" in messages[1][1].split("Answer:")[1].split("Cited articles:")[0]
                    return SimpleNamespace(covered=[good, good], contradiction=not good, unsupported_statements=[])
            return LLM()
    ticks = iter(range(0, 1000, 2))
    clock = lambda: next(ticks)
    rae.JUDGE_FACTORY = ByContent()
    with contextlib.redirect_stdout(io.StringIO()):
        s = jc.summary(jc.run_cases("groq", "stub", clock=clock))
    check("(m) judge_check: 6/6, both canned verdicts as expected, meets the rule",
          s["successes"] == 6 and s["verdicts_as_expected"] and s["passes"] and s["median_latency"] == 2, str(s))
    rae.JUDGE_FACTORY = ByContent(fail=2)
    with contextlib.redirect_stdout(io.StringIO()):
        s = jc.summary(jc.run_cases("groq", "stub", clock=clock))
    check("(m) judge_check: 4/6 successes does not meet the rule", s["successes"] == 4 and not s["passes"], str(s))
    check("(m) the canned article is the verbatim PDF text of PenalFr/s02/264, not in the exam",
          from_chunk_id(jc.CID) == jc.ARTICLE and "264" not in (ROOT / "eval" / "answer_set.json").read_text(encoding="utf-8"))
    from langchain_google_genai import ChatGoogleGenerativeAI
    kw = rae.gemini_thinking("gemini-3.8-flash")
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key="stub", **kw)
    check("(m) Gemini 3+: thinking_level minimal (reasoning_effort); 2.5 Flash: thinking_budget 0; default: nothing",
          kw == {"thinking_level": "minimal"} and llm.reasoning_effort == "minimal"
          and rae.gemini_thinking("gemini-2.5-flash") == {"thinking_budget": 0}
          and rae.gemini_thinking("gemini-3.8-flash", "default") == {}, str(kw))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jc.main(["--judge-provider", "gemini", "--judge-model", "gemini-3.8-flash", "--dry-run"])
    check("(m) judge_check --dry-run prints the cases and the rule, no call",
          "Selection rule" in buf.getvalue() and "no call made" in buf.getvalue() and "minimal" in buf.getvalue())


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
    busy = RuntimeError("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high "
                        "demand. Spikes in demand are usually temporary. Please try again later.'}}")
    tries, waits = [], []

    class Busy:
        def invoke(self, messages):
            tries.append(1)
            if len(tries) < 3:
                raise busy
            return "ok"
    out = rae.invoke_judge(Busy(), [], "gemini", sleep=waits.append)
    check("(i) Gemini 503 'high demand': not quota, retried with backoff, then answers",
          hc.classify(busy) is None and out == "ok" and waits == [6, 11], f"waits {waits}")
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
    for t in (test_quota_and_resume, test_per_minute_retry, test_changed_exam, test_other_set, test_judge_only,
              test_judge_errors, test_gate_judge, test_clarify_subsets, test_reply_gold, test_run_batch, test_crash_isolation, test_fingerprint,
              test_judge_thinking_retry, test_judge_check,
              test_wait_and_budget,
              test_classes, test_usage_in_graph):
        t()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed" + ("" if n_fail else ": all checks passed"))
    sys.exit(1 if n_fail else 0)
