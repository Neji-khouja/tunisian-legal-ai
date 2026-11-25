"""Tokens left on a Groq key, estimated: the provider does not publish what is left of its daily limit, so the tokens used
through this app in the last 24 hours are subtracted from the limit (UI_TOKENS_PER_DAY). Tokens spent elsewhere on the same
key are not seen until the provider refuses a call and states the real figure, which is then taken over.
A personal key is only known here by a hash, in memory."""
import hashlib
import re
import threading
import time
from collections import deque
from datetime import datetime

DAY = 24 * 3600
OWNER = "owner"
_DAILY = re.compile(r"tokens per day.*?Limit\s+([\d,]+).*?Used\s+([\d,]+)", re.I | re.S)


def key_id(personal_key: str | None) -> str:
    key = (personal_key or "").strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] if key else OWNER


def counted(tokens_in: int, tokens_out: int, cached: int = 0) -> int:
    """Tokens that count toward the limit (cached ones do not)."""
    return max(0, int(tokens_in or 0) - int(cached or 0)) + int(tokens_out or 0)


class TokenBudget:
    def __init__(self, limit: int = 200_000, clock=time.time):
        self.limit, self._clock, self._lock, self._used = max(1, int(limit)), clock, threading.Lock(), {}

    def _entries(self, key: str) -> deque:
        q = self._used.setdefault(key, deque())
        while q and q[0][0] < self._clock() - DAY:
            q.popleft()
        return q

    def add(self, personal_key: str | None, tokens: int, when: float | None = None):
        if tokens:
            with self._lock:
                q = self._entries(key_id(personal_key))
                q.append((self._clock() if when is None else when, int(tokens)))

    def used(self, personal_key: str | None = None) -> int:
        with self._lock:
            return sum(n for _, n in self._entries(key_id(personal_key)))

    def left(self, personal_key: str | None = None) -> int:
        return max(0, self.limit - self.used(personal_key))

    def learn(self, personal_key: str | None, message: str) -> bool:
        """The provider's daily refusal states 'Limit 200000, Used 199990': take its figures over."""
        m = _DAILY.search(message or "")
        if not m:
            return False
        limit, used = (int(x.replace(",", "")) for x in m.groups())
        if key_id(personal_key) == OWNER:
            self.limit = limit
        missing = used - self.used(personal_key)
        if missing > 0:
            self.add(personal_key, missing)
        return True

    def seed(self, rows: list[dict]):
        """The owner's usage of the last 24 hours from the log (after a restart). Rows present twice count once."""
        seen = set()
        for r in sorted(rows, key=lambda r: r.get("ts") or ""):
            ident = (r.get("ts"), r.get("session"), r.get("turn"))
            if r.get("key_source") != "owner" or ident in seen:
                continue
            seen.add(ident)
            try:
                when = datetime.fromisoformat(r["ts"]).timestamp()
            except (TypeError, ValueError, KeyError):
                continue
            if when >= self._clock() - DAY:
                self.add(None, counted(r.get("tokens_in"), r.get("tokens_out"), r.get("tokens_cached")), when)
