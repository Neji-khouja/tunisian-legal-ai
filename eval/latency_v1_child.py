"""Runs eval/v1_runner.py unchanged, with two extra log handlers: every 429 and every announced Groq retry wait of the v1
process goes to a side file that eval/latency_benchmark.py reads (v1_runner.py itself reports neither).

    python eval/latency_v1_child.py <worktree> [<runner path>]      (env LATENCY_V1_EVENTS = the side file)

Standard library only, like eval/v1_runner.py's own start: only the worktree's modules are imported by the runner. The
handlers apply the same tests as run_answer_eval.py's _CountGroq and _GroqWaits. Side file: one line per event, "429" or
"wait <seconds>".
"""
import os
import sys
import runpy
import logging
from pathlib import Path

EVENTS_ENV = "LATENCY_V1_EVENTS"


class EventLog(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def _write(self, line: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def emit(self, record):
        msg = record.getMessage()
        if "api.groq.com" in msg and " 429 " in msg:
            self._write("429")
        elif record.msg.startswith("Retrying request to") and len(record.args or ()) == 2:
            self._write(f"wait {float(record.args[1])}")


def install(path):
    handler = EventLog(path)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpx").addHandler(handler)
    groq_log = logging.getLogger("groq._base_client")
    groq_log.setLevel(logging.INFO)
    groq_log.addHandler(handler)
    groq_log.propagate = False


def main():
    worktree = sys.argv[1]
    runner = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent / "v1_runner.py")
    install(os.environ[EVENTS_ENV])
    sys.argv = [runner, worktree]
    runpy.run_path(runner, run_name="__main__")


if __name__ == "__main__":
    main()
