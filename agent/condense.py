"""Follow-up handling: rewrite a message that depends on earlier turns as one standalone question (one LLM call)."""
import hashlib
from pydantic import BaseModel
from analyze import llm
from llm_retry import invoke_with_retry


class Standalone(BaseModel):
    question: str
    is_followup: bool


structured_llm = llm.with_structured_output(Standalone, method="json_schema")

CONDENSE_PROMPT = """You prepare messages for a legal assistant that answers questions about Tunisian law.
You receive the last turns of the conversation (the user's question and the assistant's answer) and the user's
new message.

Decide whether the new message depends on the earlier turns to be understood.
- If it does (it continues, narrows or refers back to an earlier question, for example "and for minors?" or
  "what is its penalty?"), rewrite it as ONE standalone question that can be understood without the
  conversation, and set is_followup to true.
- If it does not (it is already a complete question, or it starts a new topic), return it exactly as written
  and set is_followup to false.

Rules:
- Write the question in the SAME language as the new message, even if the earlier turns are in another
  language. Never translate it.
- Never answer the question.
- Take from the earlier turns only what the new message refers to. Never add legal facts, conditions, numbers
  or article numbers that are not in the conversation.
- Fill in the structured result only. No other text.

Example 1
Previous turns:
User: How do I open a bank account?
Assistant: You need an identity document and proof of address.
New message: And for minors?
Result: {"question": "How can a minor open a bank account?", "is_followup": true}

Example 2
Previous turns:
User: How do I open a bank account?
Assistant: You need an identity document and proof of address.
New message: Quel est le délai pour contester une amende de stationnement ?
Result: {"question": "Quel est le délai pour contester une amende de stationnement ?", "is_followup": false}
"""
PROMPT_SHA1 = hashlib.sha1(CONDENSE_PROMPT.encode("utf-8")).hexdigest()[:12]     # as eval_plan_vs_raw._sha


def condense(history: list[dict], message: str) -> Standalone:
    """history: [{"q": standalone question, "a": answer text}, ...], oldest first (the caller keeps the last 2)."""
    turns = "\n\n".join(f"User: {t['q']}\nAssistant: {t['a']}" for t in history)
    human = f"Previous turns:\n{turns}\n\nNew message: {message}"
    return invoke_with_retry(structured_llm, [("system", CONDENSE_PROMPT), ("human", human)], "Condense")
