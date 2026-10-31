"""Run mechanics for eval/run_answer_eval.py: quota errors, resumable result files, run state, provenance, exam hashes.
Standard library only at import time (eval/v1_runner.py loads this file by path inside the v1 worktree).

Error classes:
- a per-minute 429 (Groq RPM/TPM) keeps the existing retries (groq client, agent/llm_retry.py);
- a 429 whose message names a per-day limit (tokens/requests per day, TPD/RPD, Gemini "...PerDay...") is QuotaExhausted:
  never retried; the (item, system) in progress is discarded and the run stops with exit code 75.
"""
import os
import re
import sys
import json
import time
import sqlite3
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

EXIT_QUOTA = 75
DAILY = re.compile(r"per[ _-]?day|\bTPD\b|\bRPD\b", re.I)
QUOTA_SEEN = {"msg": None}          # set by the Groq patches when a daily-quota 429 comes back (even if a node swallows it)


class QuotaExhausted(Exception):
    def __init__(self, msg: str, retry_after: float | None = None, who: str = "agent"):
        super().__init__(msg)
        self.retry_after, self.who = retry_after, who


def full_message(e: BaseException) -> str:
    """The message of an exception and of its causes (agent/llm_retry.py wraps the Groq error)."""
    parts, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        parts.append(f"{type(e).__name__}: {e}")
        e = e.__cause__ or e.__context__
    return " <- ".join(parts)


def is_rate_limit(msg: str) -> bool:
    m = msg.lower()
    return "429" in m or "rate limit" in m or "rate_limit" in m or "ratelimit" in m or "resource_exhausted" in m \
        or "resourceexhausted" in m or "quota" in m


TRANSIENT = re.compile(r"\b(500|502|503|504)\b|UNAVAILABLE|DEADLINE_EXCEEDED|overloaded|high demand|try again later", re.I)


def is_transient(msg: str) -> bool:
    """A temporary server error (Gemini 503 'high demand' and the like): retried with backoff like a per-minute 429."""
    return bool(TRANSIENT.search(msg))


def is_daily_quota(msg: str) -> bool:
    return is_rate_limit(msg) and bool(DAILY.search(msg))


def parse_retry_after(msg: str) -> float | None:
    """Groq 'try again in 1h2m3.5s' / '850ms'; Gemini 'retry in 39.7s' or retryDelay '39s'."""
    m = re.search(r"(?:try again|retry) in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?", msg)
    if m and any(m.groups()):
        h, mi, s, ms = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + s + ms / 1000
    m = re.search(r"retry_?delay\W*(?:seconds\W*)?([\d.]+)", msg, re.I)      # "retryDelay": "39s" / retry_delay { seconds: 12 }
    return float(m.group(1)) if m else None


def classify(e: BaseException, who: str = "agent") -> QuotaExhausted | None:
    if isinstance(e, QuotaExhausted):
        return e
    msg = full_message(e)
    return QuotaExhausted(msg[:600], parse_retry_after(msg), who) if is_daily_quota(msg) else None


def install_groq_patches():
    """Groq client: never retry a daily-quota 429 and remember it in QUOTA_SEEN. agent/llm_retry.py (when importable):
    a daily-quota 429 is not a rate limit to wait for. Per-minute 429s keep the existing retries."""
    try:
        import groq._base_client as gb
        import groq._client as gc
    except ImportError:
        return
    if not getattr(gb.SyncAPIClient, "_quota_patched", False):
        orig_retry = gb.SyncAPIClient._should_retry

        def _should_retry(self, response):
            if response.status_code == 429:
                try:
                    body = response.read().decode("utf-8", "replace")
                except Exception:
                    body = ""
                if is_daily_quota("429 " + body):
                    QUOTA_SEEN["msg"] = body[:600]
                    return False
            return orig_retry(self, response)

        orig_err = gc.Groq._make_status_error

        def _make_status_error(self, err_msg, *, body, response):
            err = orig_err(self, err_msg, body=body, response=response)
            if response.status_code == 429 and is_daily_quota(f"429 {err_msg} {body}"):
                QUOTA_SEEN["msg"] = f"{err_msg} {body}"[:600]
            return err

        gb.SyncAPIClient._should_retry = _should_retry
        gc.Groq._make_status_error = _make_status_error
        gb.SyncAPIClient._quota_patched = True
    try:
        import llm_retry
    except ImportError:
        return
    if not getattr(llm_retry, "_quota_patched", False):
        orig_rl = llm_retry._is_rate_limit
        llm_retry._is_rate_limit = lambda e: orig_rl(e) and not is_daily_quota(full_message(e))
        llm_retry._quota_patched = True


# ---------------------------------------------------------------- result files (atomic, append-only content)

class ResultFile:
    """JSONL rows; the latest row per id wins (a resume re-runs "degraded" and "error" rows by appending; "failed" is
    final). Earlier lines are kept byte for byte; every write goes to a temp file, then os.replace."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True) if self.path.exists() else []
        if self.lines and not self.lines[-1].endswith("\n"):
            self.lines[-1] += "\n"

    def rows(self) -> list[dict]:
        return [json.loads(l) for l in self.lines if l.strip()]

    def latest(self) -> dict:
        return {r["id"]: r for r in self.rows()}

    def finished(self) -> set:
        return {i for i, r in self.latest().items() if r.get("status") not in ("degraded", "error")}

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(self.lines), encoding="utf-8")
        os.replace(tmp, self.path)

    def append(self, row: dict):
        self.lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        self._write()

    def replace_latest(self, row: dict):
        """--judge-only: rewrite the latest line of this id in place (other lines unchanged)."""
        idx = max(k for k, l in enumerate(self.lines) if l.strip() and json.loads(l)["id"] == row["id"])
        self.lines[idx] = json.dumps(row, ensure_ascii=False) + "\n"
        self._write()


# ---------------------------------------------------------------- run state / log

def write_state(results_dir: Path, state: dict):
    results_dir.mkdir(parents=True, exist_ok=True)
    state = {"time": datetime.now().isoformat(timespec="seconds"), **state}
    tmp = results_dir / "run_state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, results_dir / "run_state.json")
    with (results_dir / "run_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(state, ensure_ascii=False) + "\n")


def resume_estimate(retry_after: float | None) -> str | None:
    return (datetime.now() + timedelta(seconds=retry_after)).isoformat(timespec="seconds") if retry_after else None


# ---------------------------------------------------------------- exam fingerprint

def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "missing"


def read_hashes(p: Path) -> dict:
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines() if Path(p).exists() else []:
        if line.strip():
            name, h = line.split()
            out[name] = h
    return out


def exam_check(set_path: Path, stages_path: Path, hashes_path: Path) -> tuple[dict, list[str]]:
    """-> (current hashes, differences vs the hashes file)."""
    cur = {"eval/answer_set.json": sha256(set_path), "eval/stages.json": sha256(stages_path)}
    want = read_hashes(hashes_path)
    if not want:
        return cur, [f"{hashes_path} missing or empty"]
    return cur, [f"{k}: {cur.get(k)} (expected {v})" for k, v in want.items() if cur.get(k) != v]


# ---------------------------------------------------------------- provenance

def git(root: Path, *args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    except OSError:
        return ""


def git_provenance(root: Path) -> dict:
    dirty = [l[3:] for l in git(root, "status", "--porcelain", "--", "agent", "eval", "*.py").splitlines()
             if l[3:].endswith(".py")]
    return {"commit": git(root, "rev-parse", "HEAD"), "tags": git(root, "tag", "--points-at", "HEAD").split(),
            "dirty_py": dirty}


FINGERPRINT_FILES = ["config.py", "splitter.py", "retriever.py", "reranker.py", "pipeline.py"]


def fingerprint_files(root: Path) -> list[Path]:
    """agent/*.py except test_*.py and run_*.py, then FINGERPRINT_FILES (repo root). eval/ (the harness) is not in it."""
    root = Path(root)
    agent = sorted(p for p in (root / "agent").glob("*.py") if not p.name.startswith(("test_", "run_")))
    return agent + [root / f for f in FINGERPRINT_FILES]


def index_identity(root: Path, db_dir: Path) -> dict:
    """Index path (relative to the repo) and chunk count (rows of the chroma embeddings table, read-only)."""
    p, n = Path(db_dir) / "chroma.sqlite3", None
    if p.exists():
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
        n = con.execute("select count(*) from embeddings").fetchone()[0]
        con.close()
    return {"path": Path(os.path.relpath(db_dir, root)).as_posix(), "chunks": n}


def system_fingerprint(root: Path, db_dir: Path) -> dict:
    """sha256 over the path and contents of every fingerprint file, then the index identity."""
    h, files = hashlib.sha256(), fingerprint_files(root)
    for f in files:
        h.update(f.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update((f.read_bytes() if f.exists() else b"missing") + b"\0")
    idx = index_identity(root, db_dir)
    h.update(json.dumps(idx, sort_keys=True).encode("utf-8"))
    return {"sha256": h.hexdigest(), "files": len(files), "index": idx}


def mixed_versions(rows: list[dict], fingerprint: str) -> set:
    """System fingerprints (sha256) of rows made by another system version; a row without one counts as None."""
    return {((r.get("provenance") or {}).get("fingerprint") or {}).get("sha256") for r in rows} - {fingerprint}


# ---------------------------------------------------------------- waiting for the quota

def wait_for_quota(probe, minutes: float, max_hours: float, retry_after: float | None, waited: float,
                   sleep=time.sleep, log=print) -> tuple[bool, float]:
    """Sleep and probe until the quota returns. -> (quota back, total waited seconds)."""
    first = True
    while waited < max_hours * 3600:
        pause = min(retry_after + 5, minutes * 60) if (first and retry_after) else minutes * 60
        pause = min(pause, max_hours * 3600 - waited)
        log(f"WAITING {pause / 60:.1f} min for the daily quota (waited so far {waited / 3600:.2f} h of {max_hours} h)")
        sleep(pause)
        waited += pause
        first = False
        try:
            if probe():
                log("quota is back (probe answered); continuing")
                return True, waited
        except Exception as e:                                   # a probe error is "not back yet"
            log(f"probe failed: {type(e).__name__}: {str(e)[:200]}")
    return False, waited
