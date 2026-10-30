"""Verifier, support half: does each cited article really say what the claim says?"""
import sys
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import llm
from answer import Answer

VERIFY_CHARS = 1500


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


def _claims_block(answer: Answer, evidence: dict) -> str:
    parts = []
    for i, c in enumerate(answer.claims, 1):
        parts.append(f"Claim {i}: {c.text}")
        for cid in c.cited_ids:
            e = evidence.get(cid)
            if e is None:
                parts.append(f"  Cited [{cid}]: (not in evidence)")
            else:
                parts.append(f"  Cited [{cid}] article {cid.split('/')[-1]}: {str(e['text'])[:VERIFY_CHARS]}")
        parts.append("")
    return "\n".join(parts)


def verify(answer: Answer, evidence: dict) -> Verification:
    structured = llm.with_structured_output(Verification, method="json_schema")
    return structured.invoke([("system", SYSTEM_PROMPT),
                              ("human", _claims_block(answer, evidence))])


def bad_claims(answer: Answer, verification: Verification) -> list[int]:
    """1-based indexes of claims that are not fully supported. A claim with no verdict counts as bad."""
    by_index = {v.claim_index: v.verdict for v in verification.verdicts}
    return [i for i in range(1, len(answer.claims) + 1) if by_index.get(i) != "supported"]