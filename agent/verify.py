"""Verifier, support half: does each cited article really say what the claim says?"""
import sys
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import llm
from llm_retry import invoke_with_retry
from answer import Answer

VERIFY_CHARS = 3000


class ClaimVerdict(BaseModel):
    claim_index: int                                  # 1-based, same numbering as in the request
    verdict: Literal["supported", "partly", "unsupported"]
    problem: str = ""                                 # one short sentence; empty if supported


class Verification(BaseModel):
    verdicts: list[ClaimVerdict]


SYSTEM_PROMPT = """You check claims against the legal articles they cite.
For each numbered claim you receive the claim and the text of the article(s) it cites.
Judge ONLY whether the cited text states what the claim says. Do not use outside knowledge and do not judge whether the law is good or complete.

Verdicts:
- supported: every fact in the claim (rule, number, duration, amount, condition, case) appears in the cited text.
- partly: the main rule is in the cited text, but the claim adds or changes a detail (a condition, a number, a case, an exception) that the text does not state.
- unsupported: the cited text does not say this, says something different, or is about another subject.

Numbers, durations, amounts and penalties must match exactly. A claim is not supported just because it sounds plausible or the article is on the same topic.
Return exactly one verdict per claim, using the claim number given. For partly and unsupported, write in `problem` one short sentence in English naming the detail that is not supported. For supported, leave `problem` empty."""


# config.VERIFY_QUESTION: the claims may use the question's own words for the situation the article covers
QUESTION_PROMPT = """You check claims against the legal articles they cite.
You receive the user's question, then for each numbered claim the claim and the text of the article(s) it cites.
Judge ONLY whether the cited text states what the claim says. Do not use outside knowledge and do not judge whether the law is good or complete.

How to read a claim:
1. The name of the situation. The claims answer the question, so they call the situation and the people in it by the question's words or by ordinary words, in the language of the question; the articles use legal words. Decide whether the article covers that situation, also when it covers it by exclusion ("all other cases", "outside the cases of articles ..."). If it does, the name is NOT a difference and must not be reported. Example: question about "a parking ticket", article on "contravention de stationnement": same situation.
2. What the claim says about that situation: the rule, the numbers, durations, amounts and penalties, the conditions, the cases, the exceptions, the order of events, and the persons or things these depend on. Each of them must be in the cited text with the same meaning, also in translation: not added, not changed, not replaced by a narrower or broader one.
3. What the claim leaves out. A claim is one sentence of an answer: leaving out another case, an exception, a formality or who applies the rule is NOT a problem, unless the claim denies it (for example with "only" or "always").

Verdicts:
- supported: the article covers the situation and states everything of point 2.
- partly: the main rule is in the cited text, but the claim adds or changes a detail of point 2.
- unsupported: the cited text does not say this, says something different, or is about another situation.

Numbers, durations, amounts and penalties must match exactly. A claim is not supported just because it sounds plausible or the article is on the same topic.
Return exactly one verdict per claim, using the claim number given. For partly and unsupported, write in `problem` one short sentence in English naming the detail of point 2 that is not supported. For supported, leave `problem` empty."""


def _claims_block(answer: Answer, evidence: dict) -> str:
    import config as C
    once, shown = getattr(C, "VERIFY_TEXT_ONCE", False), {}
    parts = []
    for i, c in enumerate(answer.claims, 1):
        parts.append(f"Claim {i}: {c.text}")
        for cid in c.cited_ids:
            e = evidence.get(cid)
            if e is None:
                parts.append(f"  Cited [{cid}]: (not in evidence)")
            elif once and cid in shown:
                parts.append(f"  Cited [{cid}] article {cid.split('/')[-1]}: (text given above, under claim {shown[cid]})")
            else:
                shown[cid] = i
                parts.append(f"  Cited [{cid}] article {cid.split('/')[-1]}: {str(e['text'])[:VERIFY_CHARS]}")
        parts.append("")
    return "\n".join(parts)


def verify(answer: Answer, evidence: dict, question: str = "") -> Verification:
    structured = llm.with_structured_output(Verification, method="json_schema")
    if question:
        return invoke_with_retry(structured, [("system", QUESTION_PROMPT),
                                  ("human", f"Question: {question}\n\n{_claims_block(answer, evidence)}")], "Verify")
    return invoke_with_retry(structured, [("system", SYSTEM_PROMPT),
                              ("human", _claims_block(answer, evidence))], "Verify")


def bad_claims(answer: Answer, verification: Verification) -> list[int]:
    """1-based indexes of claims that are not fully supported. A claim with no verdict counts as bad."""
    by_index = {v.claim_index: v.verdict for v in verification.verdicts}
    return [i for i in range(1, len(answer.claims) + 1) if by_index.get(i) != "supported"]