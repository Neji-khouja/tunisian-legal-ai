"""Warm latency benchmark of v1, v1.1 and v2 (pre-registered: 10 questions, not taken from the batch runs). No judge, no
scoring: answer text is never printed or saved.

    python eval/latency_benchmark.py --system v1|v1.1|v2 [--v1-worktree ../RAG_PDF_v1] [--dry-run]

Questions: the answerable items of Stage 2, 10 drawn with random.Random(20261022), stratified by language (proportional,
at least 2 of each language when available), in a fixed shuffled order, plus 1 separate warm-up item. Same for every system.
--dry-run prints the ids and stops (no key, no model, no call).

One process per system: the harness's runners (run_answer_eval.run_v2, run_v11, V1Process.run) on the harness's system
fingerprint, which must equal the pre-registration addendum's. Models load in the warm-up, which is not counted and runs
again in every process that has questions left. Then the questions, one at a time, each preceded by a fixed 5 s pause.

Per question: wall_seconds (around the runner call only), llm_calls, tokens_total, n_429, retry_wait_seconds, status,
started_at. retry_wait_seconds = groq_client_wait_seconds (waits the Groq client announced) + llm_retry_wait_seconds
(agent/llm_retry.py's own sleeps, which the client does not announce). wall_excluding_retry_waits = wall_seconds -
retry_wait_seconds. v1 runs in its own process: eval/latency_v1_child.py reports its 429s and waits through a side file.

Status: the system's own status (done, no_docs, clarify, refuse), or "degraded" (a node or step fell back, as in the
harness), or "error" (an exception; its type only is kept, because a provider error can quote model output). Finished =
neither degraded nor error. The summary uses finished questions only; the same command runs degraded and error questions
again and keeps the old row under "superseded".

results/latency_benchmark.json is rewritten atomically after each question; the same command resumes. A Groq daily-quota
error discards the question in progress and exits 75. Exit 1: a question ended in error, or a refusal (changed exam, other
fingerprint, other selection). Only GROQ_API_KEY is used: this script has no key rotation, run_batch.py is never imported,
and any GROQ_API_KEY_<n> is removed from the environment of this process and of the v1 process.
"""
import os
import re
import sys
import json
import time
import ctypes
import random
import argparse
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import harness_core as hc
import run_answer_eval as rae
from latency_v1_child import EVENTS_ENV

SEED = 20261022
N_QUESTIONS = 10
MIN_PER_LANGUAGE = 2
PAUSE_SECONDS = 5
RESULTS = ROOT / "results" / "latency_benchmark.json"
ADDENDUM_FINGERPRINT = "dc8fd664c66230b19fa88aad2195be1ab8b67c5be5540a010ba091c4537ec5ca"   # agent/LEARNING.md
NOT_FINISHED = ("degraded", "error")
BACKUP_KEY = re.compile(r"^GROQ_API_KEY_.+$")
LLM_RETRY = {"waited": 0.0}             # agent/llm_retry.py sleeps in this process


class ProcessDied(RuntimeError):
    """The v1 process stopped or wrote something unreadable: a crash of the run, not an error of one question."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- selection

def allocate(counts: dict, n: int = N_QUESTIONS) -> dict:
    """Questions per language: proportional, at least MIN_PER_LANGUAGE when available, largest remainder for the rest."""
    total = sum(counts.values())
    if total <= n:
        sys.exit(f"REFUSED: {total} items in the pool, {n} questions and 1 warm-up are needed")
    quota = {g: n * c / total for g, c in counts.items()}
    alloc = {g: min(c, max(MIN_PER_LANGUAGE, int(quota[g]))) for g, c in counts.items()}
    while sum(alloc.values()) < n:
        g = max((g for g in counts if alloc[g] < counts[g]), key=lambda g: (quota[g] - alloc[g], g))
        alloc[g] += 1
    while sum(alloc.values()) > n:
        g = max((g for g in counts if alloc[g] > min(counts[g], MIN_PER_LANGUAGE)), key=lambda g: (alloc[g] - quota[g], g))
        alloc[g] -= 1
    return alloc


def select_items(items: list, stage_ids: set, seed: int = SEED) -> tuple[list, dict, dict]:
    """-> (the questions in run order, the warm-up item, what is saved as "selection")."""
    pool = sorted((i for i in items if i["id"] in stage_ids and i["expected"] == "answer" and not i.get("reply")),
                  key=lambda i: i["id"])
    by_language = {}
    for i in pool:
        by_language.setdefault(i["language"], []).append(i)
    counts = {g: len(v) for g, v in sorted(by_language.items())}
    alloc, rng = allocate(counts), random.Random(seed)
    chosen = [i for g in sorted(by_language) for i in rng.sample(by_language[g], alloc[g])]
    rng.shuffle(chosen)
    ids = {i["id"] for i in chosen}
    warmup = rng.choice([i for i in pool if i["id"] not in ids])
    info = {"stage": 2, "seed": seed, "pool_by_language": counts, "questions_by_language": alloc,
            "question_ids": [i["id"] for i in chosen], "warmup_id": warmup["id"]}
    return chosen, warmup, info


def exam_selection() -> tuple[list, dict, dict]:
    items = json.loads((ROOT / "eval" / "answer_set.json").read_text(encoding="utf-8"))
    return select_items(items, rae.stage_ids(ROOT / "eval" / "stages.json", "2"))


# ---------------------------------------------------------------- key, machine, wait counters

def drop_backup_keys():
    """Backup keys are off: config.py and agent/analyze.py call load_dotenv, so this runs again before every question."""
    for name in [n for n in os.environ if BACKUP_KEY.match(n)]:
        del os.environ[name]


def ram_gb() -> tuple[float | None, float | None]:
    """-> (total, available) in GiB."""
    try:
        if sys.platform == "win32":
            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = Status()
            s.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return round(s.ullTotalPhys / 2 ** 30, 1), round(s.ullAvailPhys / 2 ** 30, 1)
        page = os.sysconf("SC_PAGE_SIZE")
        return (round(page * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1),
                round(page * os.sysconf("SC_AVPHYS_PAGES") / 2 ** 30, 1))
    except (AttributeError, OSError, ValueError):
        return None, None


def machine() -> dict:
    return {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "ram_total_gb": ram_gb()[0], "python": platform.python_version()}


def install_llm_retry_counter():
    """agent/llm_retry.py sleeps through a default argument: wrap that default (in memory, the file is not changed) so
    its waits are counted. The sleep itself still happens."""
    import llm_retry
    real = llm_retry.invoke_with_retry.__defaults__[0]
    if getattr(real, "_counted", False):
        return

    def counted(seconds):
        LLM_RETRY["waited"] += seconds
        real(seconds)
    counted._counted = True
    llm_retry.invoke_with_retry.__defaults__ = (counted,)


def in_process_counters() -> dict:
    return {"n429": rae.CALLS["n429"], "client_wait": rae.CALLS["waited"], "llm_retry_wait": LLM_RETRY["waited"]}


def side_file_counters(path) -> dict:
    """The v1 process' events so far (written by eval/latency_v1_child.py)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines() if Path(path).exists() else []
    return {"n429": sum(l == "429" for l in lines), "llm_retry_wait": 0.0,
            "client_wait": sum(float(l.split()[1]) for l in lines if l.startswith("wait "))}


# ---------------------------------------------------------------- v1 process

class BenchV1(rae.V1Process):
    """The harness's V1Process.run on the same runner, started through eval/latency_v1_child.py."""

    def __init__(self, worktree: Path, events: str, runner_args=()):
        env = {k: v for k, v in os.environ.items() if not BACKUP_KEY.match(k)}
        self.p = subprocess.Popen([sys.executable, str(ROOT / "eval" / "latency_v1_child.py"), str(worktree), *runner_args],
                                  cwd=worktree, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                  env={**env, "PYTHONIOENCODING": "utf-8", EVENTS_ENV: events})

    def run(self, item) -> dict:
        try:
            return super().run(item)
        except (ValueError, OSError) as e:
            raise ProcessDied(f"the v1 process stopped or wrote an unreadable line ({type(e).__name__})") from e

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=60)


# ---------------------------------------------------------------- one question

def measure(item, runner, read_counters, clock=time.perf_counter) -> dict:
    """One call of the runner -> the saved row. QuotaExhausted goes to the caller (nothing is saved for this question).
    The answer stays in `rec`: only numbers, ids and the status leave this function."""
    drop_backup_keys()
    hc.QUOTA_SEEN["msg"] = None
    calls_before, before = rae.CALLS["n"], read_counters()
    rec, error, started_at = {}, None, now()
    with rae.usage_callback() as cb, hc.capture_fallbacks() as fallbacks:
        t0 = clock()
        try:
            rec = runner(item)
        except ProcessDied:
            raise
        except Exception as e:
            q = hc.classify(e)
            if q:
                raise q from e
            error = type(e).__name__
        wall = clock() - t0
    if hc.QUOTA_SEEN["msg"]:                        # a node swallowed a daily-quota error
        raise hc.QuotaExhausted(hc.QUOTA_SEEN["msg"], hc.parse_retry_after(hc.QUOTA_SEEN["msg"]))
    after = read_counters()
    calls = rec.pop("_llm_calls", rae.CALLS["n"] - calls_before)
    evidence = rec.get("_evidence_text") or {}
    tokens = rae.token_record(cb.usage_metadata, rec.pop("_tokens", None), est_in=rae.asked(item) + "".join(evidence.values()),
                              est_out=rec.get("text") or "", calls=calls)
    n_fallbacks = len(fallbacks) + len(rec.pop("_fallbacks", []))
    client_wait = after["client_wait"] - before["client_wait"]
    retry_wait = after["llm_retry_wait"] - before["llm_retry_wait"]
    waits = client_wait + retry_wait
    row = {"id": item["id"], "language": item.get("language"), "started_at": started_at, "wall_seconds": round(wall, 3),
           "retry_wait_seconds": round(waits, 3), "groq_client_wait_seconds": round(client_wait, 3),
           "llm_retry_wait_seconds": round(retry_wait, 3), "wall_excluding_retry_waits": round(wall - waits, 3),
           "llm_calls": calls, "tokens_total": tokens["total"], "tokens_source": tokens["source"],
           "n_429": after["n429"] - before["n429"], "n_fallbacks": n_fallbacks,
           "status": "error" if error else "degraded" if rec.get("degraded") or n_fallbacks else rec.get("status")}
    if error:
        row["error_type"] = error
    return row


# ---------------------------------------------------------------- results file

def save_atomic(path: Path, doc: dict, tries: int = 5):
    """Temp file, then os.replace: a failure leaves the previous file whole. OneDrive can hold the file for a moment."""
    text = json.dumps(doc, ensure_ascii=False, indent=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == tries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.5)


def load_doc(path: Path, selection: dict, mach: dict) -> dict:
    if not path.exists():
        return {"pause_seconds": PAUSE_SECONDS, "selection": selection, "machine": mach, "systems": {}, "summary": {}}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc["selection"] != selection:
        sys.exit(f"REFUSED: {path.name} holds another selection of questions")
    return doc


# ---------------------------------------------------------------- summary

def stats(xs: list) -> dict:
    if not xs:
        return {"median": None, "mean": None, "min": None, "max": None}
    return {"median": round(statistics.median(xs), 3), "mean": round(statistics.fmean(xs), 3),
            "min": round(min(xs), 3), "max": round(max(xs), 3)}


def mean(xs: list) -> float | None:
    return round(statistics.fmean(xs), 3) if xs else None


def summarize(rows: list, planned: int = N_QUESTIONS) -> dict:
    done = [r for r in rows if r["status"] not in NOT_FINISHED]
    return {"planned": planned, "finished": len(done), "degraded": sum(r["status"] == "degraded" for r in rows),
            "error": sum(r["status"] == "error" for r in rows),
            "wall_seconds": stats([r["wall_seconds"] for r in done]),
            "wall_excluding_retry_waits": stats([r["wall_excluding_retry_waits"] for r in done]),
            "mean_llm_calls": mean([r["llm_calls"] for r in done]), "mean_tokens": mean([r["tokens_total"] for r in done]),
            "mean_429": mean([r["n_429"] for r in done]),
            "questions_with_estimated_tokens": sum("estimated" in r["tokens_source"] for r in done)}


def print_summary(doc: dict):
    print(f"\n{'system':<7}{'finished':>9}  {'wall median/mean/min/max (s)':<32}{'excl. waits median/mean/min/max (s)':<37}"
          f"{'LLM calls':>10}{'tokens':>10}{'429s':>7}")
    for s in rae.SYSTEM_ORDER:
        x = doc["summary"].get(s)
        if x:
            w, e = x["wall_seconds"], x["wall_excluding_retry_waits"]
            line = lambda d: f"{d['median']}/{d['mean']}/{d['min']}/{d['max']}"
            print(f"{s:<7}{x['finished']:>6}/{x['planned']:<2}  {line(w):<32}{line(e):<37}{str(x['mean_llm_calls']):>10}"
                  f"{str(x['mean_tokens']):>10}{str(x['mean_429']):>7}")
    print(f"machine: {doc['machine']}")


# ---------------------------------------------------------------- one system

def run_system(system: str, path: Path, ctx) -> int:
    """-> exit code: 0, 1 (a question ended in error), or 75 (daily quota)."""
    doc = load_doc(path, ctx.selection, ctx.machine)
    sd = doc["systems"].setdefault(system, {"provenance": ctx.provenance, "runs": [], "questions": {}, "superseded": []})
    fp, stored = ctx.provenance["fingerprint"]["sha256"], sd["provenance"]["fingerprint"]["sha256"]
    if fp != stored:
        sys.exit(f"REFUSED: {path.name} holds {system} rows of another system fingerprint ({stored[:12]}; now {fp[:12]})")
    todo = [i for i in ctx.questions if (sd["questions"].get(i["id"]) or {"status": "error"})["status"] in NOT_FINISHED]
    code = 0
    if todo:
        run = {"started_at": now(), "ram_available_gb": ram_gb()[1], "warmup": None, "finished_at": None, "exit_code": None}
        sd["runs"].append(run)
        save_atomic(path, doc)
        current = ctx.warmup
        try:
            print(f"[bench] {system}: warm-up {current['id']}", flush=True)
            run["warmup"] = measure(current, ctx.runner, ctx.read_counters, ctx.clock)
            save_atomic(path, doc)
            for k, current in enumerate(todo, 1):
                ctx.sleep(PAUSE_SECONDS)
                row = measure(current, ctx.runner, ctx.read_counters, ctx.clock)
                old = sd["questions"].get(current["id"])
                if old:
                    sd["superseded"].append(old)
                sd["questions"][current["id"]] = row
                doc["summary"][system] = summarize(list(sd["questions"].values()))
                save_atomic(path, doc)
                print(f"[bench] {system}: {k}/{len(todo)} {current['id']} wall {row['wall_seconds']} s, status {row['status']}",
                      flush=True)
        except hc.QuotaExhausted as q:
            code = hc.EXIT_QUOTA
            run["stopped"] = {"reason": "daily quota", "question": current["id"], "retry_after_seconds": q.retry_after}
            print(f"[bench] {system}: STOPPED, daily quota on {current['id']}; that question is discarded; saved questions "
                  f"are kept; the same command resumes", file=sys.stderr, flush=True)
        run["finished_at"] = now()
    if code == 0 and any(r["status"] == "error" for r in sd["questions"].values()):
        code = 1
    if todo:
        run["exit_code"] = code
    doc["summary"][system] = summarize(list(sd["questions"].values()))
    save_atomic(path, doc)
    print_summary(doc)
    return code


def prepare(system: str, worktree: Path, questions: list, warmup: dict, selection: dict) -> SimpleNamespace:
    """Key, exam and fingerprint checks, the harness's Groq patches, the runner. Loads no model: that is the warm-up."""
    import config  # noqa: F401  (load_dotenv, as in a harness run; the v1 process gets GROQ_API_KEY from this environment)
    drop_backup_keys()
    if "run_batch" in sys.modules:
        sys.exit("REFUSED: run_batch.py (backup keys) is loaded")
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("REFUSED: GROQ_API_KEY is unset")
    hashes, diffs = hc.exam_check(ROOT / "eval" / "answer_set.json", ROOT / "eval" / "stages.json", ROOT / "eval" / "exam_hashes.txt")
    if diffs:
        sys.exit("REFUSED: the exam changed since the fingerprint:\n  " + "\n  ".join(diffs))
    prov = rae.provenance(SimpleNamespace(skip_judge=True), hashes)
    if prov["fingerprint"]["sha256"] != ADDENDUM_FINGERPRINT:
        sys.exit(f"REFUSED: system fingerprint {prov['fingerprint']['sha256']} is not the addendum's {ADDENDUM_FINGERPRINT}")
    hc.install_groq_patches()
    ctx = SimpleNamespace(questions=questions, warmup=warmup, selection=selection, provenance=prov, machine=machine(),
                          sleep=time.sleep, clock=time.perf_counter, close=lambda: None)
    if system != "v1":
        install_llm_retry_counter()
        ctx.runner, ctx.read_counters = rae.SYSTEMS[system], in_process_counters
        return ctx
    tags = hc.git(worktree, "tag", "--points-at", "HEAD").split()
    if rae.V1_TAG not in tags or not (worktree / "pipeline.py").exists() or not (worktree / "db").exists():
        sys.exit(f"REFUSED: {worktree} is not a worktree of {rae.V1_TAG} holding db/ (tags at its HEAD: {tags})")
    prov["v1_worktree"] = {"commit": hc.git(worktree, "rev-parse", "HEAD"), "tags": tags}
    folder = tempfile.TemporaryDirectory(prefix="latency_v1_")
    events = str(Path(folder.name) / "events.txt")
    proc = BenchV1(worktree, events)
    ctx.runner, ctx.read_counters = proc.run, lambda: side_file_counters(events)
    ctx.close = lambda: (proc.close(), folder.cleanup())
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True, choices=rae.SYSTEM_ORDER)
    ap.add_argument("--v1-worktree", default=str(ROOT.parent / "RAG_PDF_v1"))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--dry-run", action="store_true", help="print the ids and stop: no key, no model, no call")
    args = ap.parse_args(argv)
    questions, warmup, selection = exam_selection()
    print(f"[bench] question ids: {','.join(selection['question_ids'])}\n[bench] warm-up id: {selection['warmup_id']}", flush=True)
    if args.dry_run:
        return
    ctx = prepare(args.system, Path(args.v1_worktree), questions, warmup, selection)
    try:
        code = run_system(args.system, Path(args.results), ctx)
    finally:
        ctx.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
