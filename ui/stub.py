"""Demo backend: canned results with the same shapes as the real ones, no model, no network. Every text is a fictitious
example (labelled as such); the article texts are placeholders, not the law.
"""
import os
import time

PLACEHOLDER_ARTICLE = "[Exemple de démonstration] Texte fictif d'article : il remplace le texte réel du code dans ce mode."
PLACEHOLDER_ARTICLE_2 = "[Exemple de démonstration] Second texte fictif, pour montrer plusieurs citations."
EVIDENCE = {"PenalFr/s02/264": {"source": "PenalFr.pdf", "article": "264", "law_type": "penal", "text": PLACEHOLDER_ARTICLE},
            "PenalFr/s02/264-bis": {"source": "PenalFr.pdf", "article": "264", "law_type": "penal", "text": PLACEHOLDER_ARTICLE_2},
            "travailFr/s01/14": {"source": "travailFr.pdf", "article": "14", "law_type": "travail", "text": PLACEHOLDER_ARTICLE}}

FAST_TEXT = ("[Exemple] Réponse courte en un seul passage, sans vérification (article 264). "
             "Elle vaut sauf dans les cas particuliers prévus par le texte.")

ANSWER_STEPS = ["condense", "analyze", "retrieve", "answer", "verify", "repair", "verify", "finalize"]


class DemoRateLimit(Exception):
    """Same message shape as a Groq daily-quota 429 (organization id removed)."""


def _answer_state(question):
    return {"status": "done", "language": "fr", "question": question, "evidence": EVIDENCE, "degraded": [],
            "claims": [{"text": "[Exemple] Première affirmation, citée par un article.", "cited_ids": ["PenalFr/s02/264"]},
                       {"text": "[Exemple] Seconde affirmation, avec deux citations.",
                        "cited_ids": ["PenalFr/s02/264", "PenalFr/s02/264-bis"]}],
            "caveat": "[Exemple] Les articles ne traitent qu'une partie de la question.", "dropped": 1}


def case_for(question: str, pending_clarify: bool) -> str:
    q = question.lower()
    if any("؀" <= c <= "ۿ" for c in question):
        return "arabic"
    if pending_clarify:
        return "answer"
    for word, case in (("contrat", "clarify"), ("météo", "refuse"), ("recette", "refuse"), ("football", "refuse"),
                       ("panne", "degraded"), ("quota", "quota"), ("erreur", "error")):
        if word in q:
            return case
    return "answer"


def final_state(case: str, question: str) -> dict:
    if case == "clarify":
        return {"status": "clarify", "language": "fr", "question": question, "degraded": [],
                "text": "[Exemple] S'agit-il d'un contrat de travail, de bail ou de vente ?"}
    if case == "refuse":
        return {"status": "refuse", "language": "fr", "question": question, "degraded": [],
                "text": "Cette question sort du cadre des codes tunisiens que je couvre."}
    state = _answer_state(question)
    if case == "degraded":
        return state | {"degraded": ["verify_failed: RateLimitError"], "dropped": 0}
    if case == "arabic":
        return state | {"language": "ar", "caveat": "", "dropped": 0,
                        "claims": [{"text": "[مثال] هذه إجابة تجريبية مكتوبة بالعربية للعرض فقط.", "cited_ids": ["PenalFr/s02/264"]}]}
    return state


def steps_for(case: str) -> list[str]:
    return {"clarify": ["condense", "analyze", "clarify"], "refuse": ["condense", "analyze", "refuse"],
            "degraded": ["condense", "analyze", "retrieve", "answer", "verify", "finalize"]}.get(case, ANSWER_STEPS)


class DemoBackend:
    def __init__(self, delay: float | None = None):
        self.delay = float(os.environ.get("RAG_UI_DEMO_DELAY", "0.35")) if delay is None else delay
        self.pending = set()

    def stream_v2(self, question: str, thread_id: str):
        case = case_for(question, thread_id in self.pending)
        if case == "quota":
            time.sleep(self.delay)
            raise DemoRateLimit("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day "
                                "(TPD): Limit 200000, Used 199990, Requested 900. Please try again in 7m12.5s.")
        if case == "error":
            time.sleep(self.delay)
            raise RuntimeError("demo failure")
        for node in steps_for(case):
            time.sleep(self.delay)
            yield "step", node
        (self.pending.add if case == "clarify" else self.pending.discard)(thread_id)
        yield "usage", {"in": 4200, "out": 650, "cached": 1500, "reasoning": 300}
        yield "final", final_state(case, question)

    def stream_fast(self, question: str, thread_id: str):
        time.sleep(self.delay)
        yield "step", "retrieve"
        time.sleep(self.delay)
        yield "step", "answer"
        yield "usage", {"in": 900, "out": 180, "cached": 0, "reasoning": 60}
        yield "final", {"status": "done", "mode": "fast", "degraded": [],
                        "text": FAST_TEXT,
                        "sources": [{"id": i, "source": e["source"], "article": e["article"], "text": e["text"]}
                                    for i, e in list(EVIDENCE.items())[:2]]}

    def remember(self, thread_id: str, state: dict):
        pass

    def forget(self, thread_id: str):
        self.pending.discard(thread_id)
