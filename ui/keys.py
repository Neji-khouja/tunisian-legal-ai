"""Groq API key per session, without editing agent/.

agent/analyze.py builds one ChatGroq at import (the key is read from the environment then), and every node reuses it;
pipeline.py and translate.py build ChatGroq clients from config.GROQ_API_KEY. All of them end in the groq SDK, which
signs every request with the `auth_headers` property of its client. install() replaces that property, at class level,
with one that returns the key of the current session when one is set (a ContextVar), else the client's own key (the
owner's key from the environment). A ContextVar is per thread and per context: Streamlit runs each session's script in
its own thread, and LangGraph runs its nodes with a copy of the caller's context, so a key set around a graph call
reaches every LLM call of that call and no other session's (ui/tests/test_ui.py checks it with concurrent threads).
This relies on groq SDK internals (`auth_headers`, groq 0.37.1); the test fails if they change.

Owner key policy (environment variables, read at start):
    UI_OWNER_KEY        on (default) | off: answer with the owner's GROQ_API_KEY when the user gives no key
    UI_OWNER_DAILY_CAP  questions per day on the owner's key, all sessions together (0 = no cap; default 0)
The counter lives in the process memory and restarts at midnight (local time) or when the app restarts.
"""
import os
import datetime
import threading
import contextvars
from contextlib import contextmanager

_SESSION_KEY = contextvars.ContextVar("rag_ui_groq_key", default=None)
PLACEHOLDER = "no-owner-key-configured"


def install():
    """Idempotent: Groq and AsyncGroq read the session key first."""
    import groq
    for cls in (groq.Groq, groq.AsyncGroq):
        if getattr(cls, "_ui_session_key", False):
            continue
        original = cls.auth_headers

        def auth_headers(self, _original=original):
            key = _SESSION_KEY.get()
            return {"Authorization": f"Bearer {key}"} if key else _original.fget(self)
        cls.auth_headers = property(auth_headers)
        cls._ui_session_key = True


@contextmanager
def session_key(key: str | None):
    """Every Groq request made inside the block (and in the threads LangGraph starts from it) uses `key`."""
    token = _SESSION_KEY.set(key or None)
    try:
        yield
    finally:
        _SESSION_KEY.reset(token)


def prepare_environment():
    """Before importing agent/: load .env through config.py (as the harness does); without an owner key, set a
    placeholder so the module-level ChatGroq can be built. The placeholder is never sent: no question runs without a
    real key (KeyPolicy.resolve)."""
    import config  # noqa: F401  (load_dotenv)
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = PLACEHOLDER


def owner_key_present() -> bool:
    key = os.environ.get("GROQ_API_KEY", "")
    return bool(key) and key != PLACEHOLDER


class KeyPolicy:
    def __init__(self, owner_enabled: bool, daily_cap: int, owner_present, today=datetime.date.today):
        self.owner_enabled, self.daily_cap = owner_enabled, max(0, int(daily_cap))
        self._owner_present, self._today = owner_present, today
        self._lock, self._day, self._used = threading.Lock(), None, 0

    @classmethod
    def from_env(cls, owner_present=owner_key_present):
        return cls(os.environ.get("UI_OWNER_KEY", "on").strip().lower() not in ("off", "0", "false", "no"),
                   int(os.environ.get("UI_OWNER_DAILY_CAP", "0") or 0), owner_present)

    def owner_usable(self) -> bool:
        return self.owner_enabled and self._owner_present()

    def remaining(self) -> int | None:
        """Questions left today on the owner's key; None = no cap."""
        if not self.daily_cap:
            return None
        with self._lock:
            self._roll()
            return max(0, self.daily_cap - self._used)

    def _roll(self):
        if self._day != self._today():
            self._day, self._used = self._today(), 0

    def resolve(self, personal_key: str | None) -> tuple[str | None, str | None]:
        """-> (key for this question: the personal key, or None = the owner's key, error kind). A question on the
        owner's key is counted here, before it runs."""
        if personal_key and personal_key.strip():
            return personal_key.strip(), None
        if not self.owner_usable():
            return None, "no_key"
        with self._lock:
            self._roll()
            if self.daily_cap and self._used >= self.daily_cap:
                return None, "owner_cap"
            self._used += 1
        return None, None
