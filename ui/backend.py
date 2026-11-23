"""The agent (agent_graph.app) through app.stream, so the steps can be shown as they happen. Heavy imports happen on
first use. The app logs questions and answers through ui/store.py; nothing here does."""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# INFO and DEBUG lines of these modules can contain the query (pipeline logs "Search query: ..."): never shown
QUIET = ("pipeline", "translate", "retriever", "reranker", "httpx", "httpcore", "groq", "sentence_transformers",
         "chromadb", "langchain", "langgraph")


def quiet_logs():
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)


class Backend:
    """stream_v2(question, thread_id) yields ("step", node name) then ("final", state); forget(thread_id) drops a
    conversation from memory."""

    def __init__(self):
        sys.path[:0] = [str(ROOT), str(ROOT / "agent"), str(ROOT / "eval")]
        import keys
        keys.prepare_environment()
        keys.install()
        quiet_logs()
        import harness_core as hc
        hc.install_groq_patches()        # a daily-quota 429 is not retried (the harness's rule)
        import agent_graph
        self.graph = agent_graph.app

    def stream_v2(self, question: str, thread_id: str):
        cfg = {"configurable": {"thread_id": thread_id}}
        for update in self.graph.stream({"question": question}, cfg, stream_mode="updates"):
            for node in update:
                yield "step", node
        yield "final", dict(self.graph.get_state(cfg).values)

    def forget(self, thread_id: str):
        try:
            self.graph.checkpointer.delete_thread(thread_id)
        except Exception:                # memory release only; never blocks a new conversation
            pass
