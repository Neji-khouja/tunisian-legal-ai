"""Answer node: write a short cited answer from the retrieved articles ONLY.

make_answer  -> LLM writes claims, each citing chunk ids from the evidence
check_answer -> plain Python: do the cited ids exist, is anything empty
render       -> turns the Answer into readable text with [Source art. N] citations
"""
import sys
from pathlib import Path

from pydantic import BaseModel,Field

sys.path.insert(0, str(Path(__file__).resolve().parent))   # agent/, for analyze
from analyze import llm # same shared Groq model object as plan.py

ANSWER_CHARS = 3000                          # max characters of each article shown to the model
LANG_NAMES = {"fr": "French", "en": "English", "ar": "Arabic"}
NOTE_WORD = {"fr": "Remarque", "en": "Note", "ar": "ملاحظة"}
MOJIBAKE = ("â€", "Ã©", "Ã¨", "Ã ", "Â", "�")     # UTF-8 read as cp1252, or the replacement character

class Claim(BaseModel):
    text :str# one sentence, in the user's language
    cited_ids: list[str]# chunk ids taken from the evidence only

class Answer(BaseModel):
    claims: list[Claim]=Field(min_length=1,max_length=6)
    caveat: str=""# filled when the evidence only partly answers


SYSTEM_PROMPT = """You answer questions about Tunisian law for a legal research assistant.
You receive a question and a list of articles retrieved from Tunisian legal codes.
Each article starts with its id in square brackets, for example [PenalFr/s02/220].

Rules:
1. Use ONLY the articles shown. Never use outside legal knowledge. Never invent an article, a penalty, a duration, an amount or a condition.
2. Write 1 to 6 claims. Each claim is ONE sentence stating ONE rule, and cites the id(s) of the article(s) that state it. Copy ids exactly as shown. Never cite an id that is not in the list. In cited_ids, write the id only, without square brackets.
3. When the articles give different rules for different cases (for example a basic offence and an aggravated one, or a general rule and an exception), write a separate claim for each case and say which case it covers. Never merge cases and never pick only one.
4. Keep numbers, durations, amounts and penalties exactly as the article states them.
5. If the articles answer only part of the question, answer that part and put one sentence in `caveat` saying what the articles shown do not cover. Do not fill the gap yourself.
6. If none of the articles answers the question, write one claim describing what the most related article actually says (cite it), and put in `caveat` that the articles shown do not answer the question.
7. Write the claims and the caveat in the answer language given in the request. Keep article numbers as they are."""


def _user_block(question, language, evidence):
    parts = [f"Question: {question}",
             f"Answer language: {LANG_NAMES.get(language, 'French')}",
             "",
             "Articles:"]
    for cid, e in evidence.items():
        parts.append(f"[{cid}] {Path(str(e['source'])).stem}, article {cid.split('/')[-1]}")
        parts.append(str(e["text"])[:ANSWER_CHARS])
        parts.append("")
    return "\n".join(parts)

def make_answer(question, language, evidence, feedback="") -> Answer:
    structured = llm.with_structured_output(Answer, method="json_schema")
    block = _user_block(question, language, evidence)
    if feedback:
        block += ("\n\nA previous draft of your answer was checked against the articles and these "
                  "problems were found:\n" + feedback +
                  "\nWrite a new answer that fixes them. Use only what the articles state; "
                  "remove or correct any claim they do not support.")
    ans = structured.invoke([("system", SYSTEM_PROMPT), ("human", block)])
    for c in ans.claims:
        c.cited_ids = [i.strip().strip("[]").strip() for i in c.cited_ids]
    return ans


def check_answer(answer: Answer, evidence: dict) -> list[str]:
    """No LLM. Returns a list of problems; an empty list means the answer is structurally fine."""
    problems = []
    for i, c in enumerate(answer.claims, 1):
        if not c.text.strip():
            problems.append(f"claim {i}: empty text")
        if not c.cited_ids:
            problems.append(f"claim {i}: no citation")
        for cid in c.cited_ids:
            if cid not in evidence:
                problems.append(f"claim {i}: cited id {cid!r} is not in the evidence")
        if any(m in c.text for m in MOJIBAKE):
            problems.append(f"claim {i}: mojibake")
    if any(m in answer.caveat for m in MOJIBAKE):
        problems.append("caveat: mojibake")
    return problems

def _label(cid, evidence):
    e = evidence[cid]
    return f"{Path(str(e['source'])).stem} art. {cid.split('/')[-1]}"     # label from the chunk id: 13-bis, not 13


def render(answer: Answer, evidence: dict, language: str = "fr") -> str:
    lines = []
    for c in answer.claims:
        cites = ", ".join(_label(i, evidence) for i in c.cited_ids if i in evidence)
        lines.append(f"- {c.text} [{cites}]")
    if answer.caveat:
        lines.append(f"\n{NOTE_WORD.get(language, 'Note')}: {answer.caveat}")
    return "\n".join(lines)