"""SQLite log of every question and answer, for later evaluation. Never stores an API key.

Local file: logs/interactions.db (RAG_LOG_DB overrides). On a host with a disk that is wiped at restart (Hugging Face
free Spaces), set one of:
- RAG_LOG_BACKUP_DIR: a mounted persistent folder (a Storage Bucket, e.g. /data/logs). After every question the whole
  database is copied there as one file. The live database stays on the local disk: the bucket mount only takes whole
  sequential writes, which SQLite's in-place writes are not.
- HF_LOG_REPO (a private dataset repo, "user/name") and HF_TOKEN: the copy is uploaded, at most every few minutes.
The copy's name is unique to this start, so a restart never overwrites older logs.
"""
import os
import json
import uuid
import shutil
import logging
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOT = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
LOG = logging.getLogger("rag_ui")
SYNC_EVERY = 300          # seconds between uploads

SCHEMA = """CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, turn INTEGER, question TEXT, status TEXT, language TEXT,
    answer TEXT, claims TEXT, cited TEXT, dropped INTEGER, unverified INTEGER, degraded TEXT, error TEXT, seconds REAL,
    key_source TEXT, mode TEXT, tokens_in INTEGER, tokens_out INTEGER)"""
COLUMNS = ("ts", "session", "turn", "question", "status", "language", "answer", "claims", "cited", "dropped", "unverified",
           "degraded", "error", "seconds", "key_source", "mode",
           "tokens_in", "tokens_out")


class Store:
    def __init__(self, path=None, upload=None, clock=time.time):
        self.path = Path(path or os.environ.get("RAG_LOG_DB") or ROOT / "logs" / "interactions.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock, self._clock, self._last_sync = threading.Lock(), clock, 0.0
        self._sync_lock = threading.Lock()
        folder = os.environ.get("RAG_LOG_BACKUP_DIR") if upload is None else None
        self.every = 0 if folder else SYNC_EVERY
        self.upload = upload if upload is not None else (folder_copier(folder) if folder else hf_uploader())
        with self._connect() as con:
            con.execute(SCHEMA)
            have = {r[1] for r in con.execute("PRAGMA table_info(interactions)")}
            for col, kind in (("mode", "TEXT"), ("tokens_in", "INTEGER"), ("tokens_out", "INTEGER")):
                if col not in have:                      # database created before these columns existed
                    con.execute(f"ALTER TABLE interactions ADD COLUMN {col} {kind}")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def log(self, session: str, turn: int, question: str, state: dict | None, error: str | None, seconds: float,
            key_source: str, mode: str = "agent", tokens: tuple = (0, 0)):
        """One row per question. `state` is the agent's final state (None when the run failed with `error`)."""
        state = state or {}
        claims = [{"text": c.get("text", ""), "cited_ids": c.get("cited_ids") or []} for c in state.get("claims") or []]
        cited = sorted({cid for c in claims for cid in c["cited_ids"]} | {x["id"] for x in state.get("sources") or []})
        row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), session, turn, question,
               "error" if error else state.get("status"), state.get("language"), state.get("text") or "",
               json.dumps(claims, ensure_ascii=False), json.dumps(cited), int(state.get("dropped") or 0),
               int(any(str(d).startswith("verify_failed") for d in state.get("degraded") or [])),
               json.dumps([str(d) for d in state.get("degraded") or []], ensure_ascii=False), error,
               round(seconds, 2), key_source, mode, int(tokens[0]), int(tokens[1]))
        with self._lock, self._connect() as con:
            con.execute(f"INSERT INTO interactions ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})", row)
        self._maybe_sync()

    def rows(self) -> list[dict]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute("SELECT * FROM interactions ORDER BY id")]

    def _maybe_sync(self):
        if not self.upload or self._clock() - self._last_sync < self.every:
            return
        self._last_sync = self._clock()
        threading.Thread(target=self.sync, daemon=True).start()

    def sync(self):
        """Upload a consistent copy of the database. Never raises: a failed backup must not break a question."""
        try:
            with self._sync_lock, tempfile.TemporaryDirectory() as tmp:
                copy = Path(tmp) / f"interactions-{BOOT}.db"
                src, dst = self._connect(), sqlite3.connect(copy)
                src.backup(dst)
                src.close()
                dst.close()
                self.upload(copy)
        except Exception as e:
            LOG.error("log backup failed: %s", type(e).__name__)


def folder_copier(folder):
    """A function that copies one file into a mounted folder (created if needed)."""
    def copy(path: Path):
        Path(folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, Path(folder) / path.name)
    return copy


def hf_uploader():
    """A function that uploads one file to the private dataset repo named by HF_LOG_REPO, or None when not configured."""
    repo, token = os.environ.get("HF_LOG_REPO"), os.environ.get("HF_TOKEN")
    if not (repo and token):
        return None

    def upload(path: Path):
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"logs/{path.name}", repo_id=repo, repo_type="dataset")
    return upload


def load_all(folder) -> list[dict]:
    """Every row of every interactions-*.db / interactions.db in a folder (downloaded backups), oldest first."""
    out = []
    for p in sorted(Path(folder).glob("interactions*.db")):
        out += Store(p, upload=False).rows()
    return sorted(out, key=lambda r: r["ts"])
