"""Offline tests of eval/latency_benchmark.py with a fake runner (temp folders; no model, no network).

    python eval/test_latency_benchmark.py
"""
import io
import os
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
import latency_benchmark as lb

RESULTS = []
FP = {"fingerprint": {"sha256": "a" * 64}}
TEXT = "SENTINEL-ANSWER-TEXT"
QUESTION = "SENTINEL-QUESTION-TEXT"


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if not cond and detail else ""))


def items(n=4):
    return [{"id": f"q{k}", "question": f"{QUESTION} {k}", "language": "fr", "expected": "answer"} for k in range(n)]


class Fake:
    """A runner: records its calls, applies `script[id]` (an exception to raise, or fields to add to the record)."""

    def __init__(self, script=None):
        self.calls, self.script = [], script or {}

    def __call__(self, item):
        self.calls.append(item["id"])
        todo = self.script.get(item["id"])
        if callable(todo):
            todo = todo()
        if isinstance(todo, BaseException):
            raise todo
        return {"status": "done", "text": TEXT, "_evidence_text": {"c": "evidence"}} | (todo or {})


def context(runner, questions=None, warmup=None, prov=FP, counters=None):
    questions = questions or items()
    warmup = warmup or {"id": "warm", "question": QUESTION, "language": "fr", "expected": "answer"}
    pauses = []
    ctx = SimpleNamespace(questions=questions, warmup=warmup,
                          selection={"question_ids": [i["id"] for i in questions], "warmup_id": warmup["id"]},
                          provenance=prov, machine={"os": "stub"}, runner=runner, sleep=pauses.append,
                          clock=lb.time.perf_counter, close=lambda: None,
                          read_counters=counters or (lambda: {"n429": 0, "client_wait": 0.0, "llm_retry_wait": 0.0}))
    return ctx, pauses


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        code = fn(*a)
    return code, out.getvalue() + err.getvalue()


def test_selection():
    check("allocate: exam pool 23 fr, 9 ar, 8 en -> 6, 2, 2", lb.allocate({"fr": 23, "ar": 9, "en": 8}) == {"fr": 6, "ar": 2, "en": 2})
    check("allocate: proportional 25, 15, 10 -> 5, 3, 2", lb.allocate({"fr": 25, "en": 15, "ar": 10}) == {"fr": 5, "en": 3, "ar": 2})
    check("allocate: at least 2, the excess taken from the largest", lb.allocate({"fr": 90, "en": 5, "ar": 5}) == {"fr": 6, "en": 2, "ar": 2})
    check("allocate: a language with 1 item gives 1", lb.allocate({"fr": 40, "en": 9, "ar": 1}) == {"fr": 7, "en": 2, "ar": 1})
    pool = [{"id": f"{g}{k}", "language": g, "expected": "answer"} for g, n in (("fr", 23), ("ar", 9), ("en", 8)) for k in range(n)]
    pool += [{"id": "c0", "language": "fr", "expected": "clarify"}, {"id": "other", "language": "fr", "expected": "answer"}]
    stage = {i["id"] for i in pool} - {"other"}
    a, wa, ia = lb.select_items(pool, stage)
    b, wb, _ = lb.select_items(list(reversed(pool)), stage)
    check("selection: the same 10 in the same order, whatever the input order", [i["id"] for i in a] == [i["id"] for i in b])
    check("selection: same warm-up, not among the 10", wa["id"] == wb["id"] and wa["id"] not in ia["question_ids"])
    check("selection: 10 answerable items of the stage, 6 fr, 2 ar, 2 en",
          len(a) == 10 and ia["questions_by_language"] == {"ar": 2, "en": 2, "fr": 6}
          and not {"c0", "other"} & (set(ia["question_ids"]) | {wa["id"]}))
    real, _, info = lb.exam_selection()
    check("selection on the exam: all three languages, at least 2 each", len(real) == 10 and
          all(info["questions_by_language"][g] >= 2 for g in ("fr", "en", "ar")))


def test_run_and_pause():
    with tempfile.TemporaryDirectory() as d:
        path, fake = Path(d) / "r.json", Fake()
        ctx, pauses = context(fake)
        code, out = quiet(lb.run_system, "v1.1", path, ctx)
        doc = read(path)
        check("run: exit 0, warm-up first, questions in the fixed order", code == 0 and fake.calls == ["warm", "q0", "q1", "q2", "q3"], fake.calls)
        check("run: warm-up not counted in the questions or the summary",
              "warm" not in doc["systems"]["v1.1"]["questions"] and doc["summary"]["v1.1"]["finished"] == 4
              and doc["systems"]["v1.1"]["runs"][0]["warmup"]["id"] == "warm")
        check("run: a 5 s pause before each question", pauses == [5, 5, 5, 5], pauses)
        raw = path.read_text(encoding="utf-8") + out
        check("run: no answer, question or evidence text saved or printed", TEXT not in raw and QUESTION not in raw and "evidence\"" not in raw)
        q = doc["systems"]["v1.1"]["questions"]["q0"]
        check("run: run and question start timestamps", doc["systems"]["v1.1"]["runs"][0]["started_at"] and q["started_at"])


def test_resume():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.json"
        first = Fake({"q2": hc.QuotaExhausted("TPD", 60.0)})
        ctx, _ = context(first)
        code, _ = quiet(lb.run_system, "v2", path, ctx)
        doc = read(path)
        saved = doc["systems"]["v2"]["questions"]
        check("quota: exit 75", code == 75, code)
        check("quota: the question in progress is not saved, earlier ones are", set(saved) == {"q0", "q1"}, list(saved))
        check("quota: the stop is recorded without the provider message",
              doc["systems"]["v2"]["runs"][0]["stopped"] == {"reason": "daily quota", "question": "q2", "retry_after_seconds": 60.0}
              and doc["systems"]["v2"]["runs"][0]["exit_code"] == 75)
        again = Fake()
        ctx, pauses = context(again)
        code, _ = quiet(lb.run_system, "v2", path, ctx)
        doc2 = read(path)
        check("resume: warm-up again, only the questions left", code == 0 and again.calls == ["warm", "q2", "q3"], again.calls)
        check("resume: saved rows unchanged", all(doc2["systems"]["v2"]["questions"][k] == saved[k] for k in saved))
        check("resume: two runs recorded, 4 finished", len(doc2["systems"]["v2"]["runs"]) == 2 and doc2["summary"]["v2"]["finished"] == 4)
        third = Fake()
        ctx, _ = context(third)
        code, _ = quiet(lb.run_system, "v2", path, ctx)
        check("resume: nothing left means no call, no warm-up, exit 0", code == 0 and third.calls == [] and len(read(path)["systems"]["v2"]["runs"]) == 2)
        ctx, _ = context(Fake(), prov={"fingerprint": {"sha256": "b" * 64}})
        try:
            quiet(lb.run_system, "v2", path, ctx)
            refused = ""
        except SystemExit as e:
            refused = str(e.code)
        check("resume: another system fingerprint is refused", refused.startswith("REFUSED"), refused)
        ctx, _ = context(Fake(), questions=items(3))
        try:
            quiet(lb.run_system, "v2", path, ctx)
            refused = ""
        except SystemExit as e:
            refused = str(e.code)
        check("resume: another selection is refused", refused.startswith("REFUSED"), refused)
        ctx, _ = context(Fake())
        code, _ = quiet(lb.run_system, "v1", path, ctx)
        check("a second system in the same file keeps the first", code == 0 and set(read(path)["summary"]) == {"v1", "v2"})


def test_quota_variants():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "swallowed.json"

        def swallowed():
            hc.QUOTA_SEEN["msg"] = "429 tokens per day (TPD). Please try again in 1m0s."
            return {"degraded": ["answer"]}
        code, _ = quiet(lb.run_system, "v2", path, context(Fake({"q1": swallowed}))[0])
        check("quota swallowed by a node: exit 75, not saved as degraded", code == 75 and set(read(path)["systems"]["v2"]["questions"]) == {"q0"})
        path = Path(d) / "wrapped.json"
        code, _ = quiet(lb.run_system, "v2", path, context(Fake({"q0": RuntimeError("Error code: 429 tokens per day (TPD)")}))[0])
        check("quota in an ordinary exception: exit 75", code == 75 and read(path)["systems"]["v2"]["questions"] == {})
        path, fake = Path(d) / "warm.json", Fake({"warm": hc.QuotaExhausted("TPD")})
        code, _ = quiet(lb.run_system, "v2", path, context(fake)[0])
        check("quota on the warm-up: exit 75, no retry, nothing counted", code == 75 and fake.calls == ["warm"] and read(path)["systems"]["v2"]["questions"] == {})
        path = Path(d) / "main.json"
        ctx, _ = context(Fake({"q1": hc.QuotaExhausted("TPD")}))
        closed = []
        ctx.close = lambda: closed.append(1)
        old = lb.prepare, lb.exam_selection
        lb.prepare = lambda *a: ctx
        lb.exam_selection = lambda: (ctx.questions, ctx.warmup, ctx.selection)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                lb.main(["--system", "v1.1", "--results", str(path)])
            code = None
        except SystemExit as e:
            code = e.code
        finally:
            lb.prepare, lb.exam_selection = old
        check("main: the process exits with 75 and closes the runner", code == 75 and closed == [1], code)
        check("eval/harness exit code is 75", hc.EXIT_QUOTA == 75)


def test_error_and_degraded():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.json"
        fake = Fake({"q1": ValueError(f"model said {TEXT}"), "q2": {"degraded": ["answer"]}})
        code, out = quiet(lb.run_system, "v2", path, context(fake)[0])
        doc = read(path)
        q = doc["systems"]["v2"]["questions"]
        check("error: the run goes on, exit 1, no retry", code == 1 and fake.calls == ["warm", "q0", "q1", "q2", "q3"], (code, fake.calls))
        check("error: only the exception type is kept", q["q1"]["status"] == "error" and q["q1"]["error_type"] == "ValueError" and TEXT not in path.read_text(encoding="utf-8") + out)
        check("degraded: status saved", q["q2"]["status"] == "degraded")
        s = doc["summary"]["v2"]
        check("summary: finished, degraded, error counted apart", (s["finished"], s["degraded"], s["error"]) == (2, 1, 1), s)
        again = Fake()
        code, _ = quiet(lb.run_system, "v2", path, context(again)[0])
        doc = read(path)
        check("resume: error and degraded questions run again; old rows kept as superseded", code == 0 and again.calls == ["warm", "q1", "q2"]
              and doc["summary"]["v2"]["finished"] == 4 and {r["status"] for r in doc["systems"]["v2"]["superseded"]} == {"error", "degraded"})


def test_process_died():
    fake = Fake({"q0": lb.ProcessDied("gone")})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.json"
        try:
            quiet(lb.run_system, "v1", path, context(fake)[0])
            raised = False
        except lb.ProcessDied:
            raised = True
        check("a dead v1 process stops the run; the warm-up is saved, the question is not",
              raised and read(path)["systems"]["v1"]["questions"] == {} and read(path)["systems"]["v1"]["runs"][0]["exit_code"] is None)


def test_atomic_save():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.json"
        lb.save_atomic(path, {"a": 1})
        real = os.replace

        def refuse(*a):
            raise PermissionError("locked")
        lb.os.replace = refuse
        old_sleep = lb.time.sleep
        lb.time.sleep = lambda s: None
        try:
            lb.save_atomic(path, {"a": 2})
            raised = False
        except PermissionError:
            raised = True
        finally:
            lb.os.replace, lb.time.sleep = real, old_sleep
        check("atomic save: a failed replace leaves the old file whole, no temp file", raised and read(path) == {"a": 1} and [p.name for p in Path(d).iterdir()] == ["r.json"])
        try:
            lb.save_atomic(path, {"a": object()})
            raised = False
        except TypeError:
            raised = True
        check("atomic save: data that cannot be written leaves the old file whole", raised and read(path) == {"a": 1})
        tries = []

        def once(*a):
            tries.append(1)
            if len(tries) == 1:
                raise PermissionError("locked for a moment")
            return real(*a)
        lb.os.replace, lb.time.sleep = once, lambda s: None
        try:
            lb.save_atomic(path, {"a": 3})
        finally:
            lb.os.replace, lb.time.sleep = real, old_sleep
        check("atomic save: a short lock is waited out", read(path) == {"a": 3} and len(tries) == 2)
        doc = {"selection": {"x": 1}}
        path.write_text(json.dumps(doc), encoding="utf-8")
        check("load: an existing file is read back", lb.load_doc(path, {"x": 1}, {}) == doc)


def test_summary():
    rows = [{"wall_seconds": 10.0, "wall_excluding_retry_waits": 10.0, "llm_calls": 2, "tokens_total": 1000, "n_429": 0, "status": "done", "tokens_source": "usage_metadata"},
            {"wall_seconds": 20.0, "wall_excluding_retry_waits": 15.0, "llm_calls": 4, "tokens_total": 2000, "n_429": 1, "status": "done", "tokens_source": "usage_metadata"},
            {"wall_seconds": 60.0, "wall_excluding_retry_waits": 50.0, "llm_calls": 6, "tokens_total": 6000, "n_429": 2, "status": "refuse", "tokens_source": "estimated (x)"},
            {"wall_seconds": 90.0, "wall_excluding_retry_waits": 1.0, "llm_calls": 9, "tokens_total": 9, "n_429": 9, "status": "degraded", "tokens_source": "usage_metadata"},
            {"wall_seconds": 95.0, "wall_excluding_retry_waits": 2.0, "llm_calls": 9, "tokens_total": 9, "n_429": 9, "status": "error", "tokens_source": "none"}]
    s = lb.summarize(rows)
    check("summary: wall median, mean, min, max of finished rows", s["wall_seconds"] == {"median": 20.0, "mean": 30.0, "min": 10.0, "max": 60.0}, s["wall_seconds"])
    check("summary: the same without waits", s["wall_excluding_retry_waits"] == {"median": 15.0, "mean": 25.0, "min": 10.0, "max": 50.0}, s["wall_excluding_retry_waits"])
    check("summary: means of calls, tokens, 429", (s["mean_llm_calls"], s["mean_tokens"], s["mean_429"]) == (4.0, 3000.0, 1.0), s)
    check("summary: counts and estimated tokens", (s["planned"], s["finished"], s["degraded"], s["error"], s["questions_with_estimated_tokens"]) == (10, 3, 1, 1, 1), s)
    check("summary: even count: median between the two middle values", lb.stats([1, 2, 3, 10])["median"] == 2.5)
    e = lb.summarize([])
    check("summary: no row gives empty values", e["finished"] == 0 and e["wall_seconds"]["median"] is None and e["mean_tokens"] is None)


def test_measure():
    ticks = iter([100.0, 142.5])
    before = dict(rae.CALLS)
    retry0 = lb.LLM_RETRY["waited"]

    def runner(item):
        rae.CALLS["n"] += 3
        rae.CALLS["n429"] += 2
        rae.CALLS["waited"] += 7.5
        lb.LLM_RETRY["waited"] += 3.0
        return {"status": "done", "text": TEXT}
    r = lb.measure(items(1)[0], runner, lb.in_process_counters, clock=lambda: next(ticks))
    check("measure: wall, waits, wall without waits",
          (r["wall_seconds"], r["groq_client_wait_seconds"], r["llm_retry_wait_seconds"], r["retry_wait_seconds"], r["wall_excluding_retry_waits"])
          == (42.5, 7.5, 3.0, 10.5, 32.0), r)
    check("measure: calls and 429s are the deltas of this question", r["llm_calls"] == 3 and r["n_429"] == 2, r)
    rae.CALLS.update(before)
    lb.LLM_RETRY["waited"] = retry0
    r = lb.measure(items(1)[0], lambda i: {"status": "done", "text": TEXT, "_llm_calls": 5, "_fallbacks": [],
                                           "_tokens": {"m": {"input_tokens": 100, "output_tokens": 50}}},
                   lambda: {"n429": 0, "client_wait": 0.0, "llm_retry_wait": 0.0})
    check("measure: v1's own calls and tokens are used", r["llm_calls"] == 5 and r["tokens_total"] == 150 and r["tokens_source"] == "usage_metadata", r)
    r = lb.measure(items(1)[0], lambda i: {"status": "done", "text": TEXT, "_llm_calls": 1, "_fallbacks": ["x"]},
                   lambda: {"n429": 0, "client_wait": 0.0, "llm_retry_wait": 0.0})
    check("measure: a step that fell back in the v1 process makes the row degraded", r["status"] == "degraded" and r["n_fallbacks"] == 1, r)


def test_llm_retry_wait():
    import llm_retry
    keep = llm_retry.invoke_with_retry.__defaults__
    slept = []
    llm_retry.invoke_with_retry.__defaults__ = (slept.append,)
    w0 = lb.LLM_RETRY["waited"]
    try:
        lb.install_llm_retry_counter()
        lb.install_llm_retry_counter()

        class Limited:
            n = 0

            def invoke(self, messages):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("429 tokens per minute. Please try again in 2s.")
                return "ok"
        got = llm_retry.invoke_with_retry(Limited(), [], "test")
        check("llm_retry: its sleep is counted once and still happens", got == "ok" and lb.LLM_RETRY["waited"] - w0 == 3.0 and slept == [3.0], (slept, lb.LLM_RETRY))
    finally:
        llm_retry.invoke_with_retry.__defaults__ = keep
        lb.LLM_RETRY["waited"] = w0


STUB_RUNNER = r'''
import sys, json, logging
assert sys.argv[1:] == [sys.argv[1]] and "latency_v1_child" not in sys.argv[0]
for line in sys.stdin:
    item = json.loads(line)
    if item["id"] == "die":
        sys.exit(3)
    logging.getLogger("httpx").info("HTTP Request: POST https://api.groq.com/x \"HTTP/1.1 429 Too Many Requests\"")
    logging.getLogger("httpx").info("HTTP Request: POST https://api.groq.com/x \"HTTP/1.1 200 OK\"")
    logging.getLogger("groq._base_client").info("Retrying request to %s in %f seconds", "/x", 1.5)
    print(json.dumps({"id": item["id"], "answer": "a", "docs": [], "fallbacks": [], "usage": None, "seconds": 0.1,
                      "llm_calls": 1}), flush=True)
'''


def test_v1_side_channel():
    with tempfile.TemporaryDirectory() as d:
        stub, events = Path(d) / "stub_runner.py", str(Path(d) / "events.txt")
        stub.write_text(STUB_RUNNER, encoding="utf-8")
        os.environ["GROQ_API_KEY_2"] = "backup"
        proc = lb.BenchV1(Path(d), events, runner_args=(str(stub),))
        del os.environ["GROQ_API_KEY_2"]
        counters = lambda: lb.side_file_counters(events)
        check("side file: nothing yet", counters() == {"n429": 0, "client_wait": 0.0, "llm_retry_wait": 0.0})
        rows = [lb.measure({"id": f"v{k}", "question": "q", "language": "fr", "expected": "answer"}, proc.run, counters) for k in range(2)]
        check("v1 child: 429s and waits of each question, through a real process",
              all(r["n_429"] == 1 and r["groq_client_wait_seconds"] == 1.5 and r["llm_calls"] == 1 for r in rows), rows)
        try:
            lb.measure({"id": "die", "question": "q", "language": "fr", "expected": "answer"}, proc.run, counters)
            died = False
        except lb.ProcessDied:
            died = True
        check("v1 child: a process that stops is a crash, not a question error", died)
        proc.p.stdin.close()
        proc.p.stdout.close()
        proc.p.wait(timeout=30)


def test_keys():
    os.environ["GROQ_API_KEY_2"], os.environ["GROQ_API_KEY_BACKUP"] = "x", "y"
    keep = os.environ.get("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = "main"
    lb.drop_backup_keys()
    check("keys: backups are removed, GROQ_API_KEY stays",
          "GROQ_API_KEY_2" not in os.environ and "GROQ_API_KEY_BACKUP" not in os.environ and os.environ["GROQ_API_KEY"] == "main")
    if keep is None:
        del os.environ["GROQ_API_KEY"]
    else:
        os.environ["GROQ_API_KEY"] = keep
    check("keys: run_batch.py is not loaded", "run_batch" not in sys.modules)
    os.environ["GROQ_API_KEY_3"] = "z"
    lb.measure(items(1)[0], Fake(), lambda: {"n429": 0, "client_wait": 0.0, "llm_retry_wait": 0.0})
    check("keys: a backup that load_dotenv brought back is removed before the next question", "GROQ_API_KEY_3" not in os.environ)


def test_machine():
    m = lb.machine()
    check("machine: OS, RAM, python version", m["os"] and m["ram_total_gb"] and m["python"], m)


def main():
    for t in (test_selection, test_run_and_pause, test_resume, test_quota_variants, test_error_and_degraded, test_process_died,
              test_atomic_save, test_summary, test_measure, test_llm_retry_wait, test_v1_side_channel, test_keys, test_machine):
        t()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
