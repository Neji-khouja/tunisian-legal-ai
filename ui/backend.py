"""The two systems, through the entry points of the evaluation harness (eval/run_answer_eval.py):
- v2: agent_graph.app with a thread_id (run_v2 calls app.invoke; here app.stream, the same compiled graph, so that the
  steps can be shown as they happen; the final state is read from the graph's checkpointer);
- v1.1: pipeline.load_vectorstore() then pipeline.ask_question(vs, question, None) (as run_v11).
Heavy imports happen on first use. Nothing here stores or logs questions or answers."""
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
    """stream_v2(question, thread_id) yields ("step", node name) then ("final", state); run_v11(question) returns
    (answer, docs as dicts: id, source, article, text); forget(thread_id) drops a conversation from memory."""

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
        self._vs = None

    def stream_v2(self, question: str, thread_id: str):
        cfg = {"configurable": {"thread_id": thread_id}}
        for update in self.graph.stream({"question": question}, cfg, stream_mode="updates"):
            for node in update:
                yield "step", node
        yield "final", dict(self.graph.get_state(cfg).values)

    def run_v11(self, question: str):
        from pipeline import load_vectorstore, ask_question
        if self._vs is None:
            self._vs = load_vectorstore()
        answer, docs = ask_question(self._vs, question, None)
        return answer, [{"id": d.metadata.get("chunk_id"), "source": d.metadata.get("source"),
                         "article": d.metadata.get("article"), "text": d.page_content} for d in docs]

    def forget(self, thread_id: str):
        try:
            self.graph.checkpointer.delete_thread(thread_id)
        except Exception:                # memory release only; never blocks a new conversation
            pass
