"""Live check of the verify loop in the agent graph (calls Groq and loads the models). Run it yourself:

    python agent/run_verify_graph_tests.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_graph as g
from verify import ClaimVerdict

HIT = "Que risque-t-on si on frappe quelqu'un ?"
LEAVE = "Quelle est la durée du congé annuel payé ?"
WAITS = (10, 30, 60)
RUN = int(time.time())


def transient(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}"
    return any(k in s for k in ("RateLimit", "429", "APIConnection", "Connect", "Timeout", "RemoteProtocol"))


def with_retry(fn, *args, **kwargs):
    for attempt in range(len(WAITS) + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not transient(e) or attempt == len(WAITS):
                raise
            print(f"   transient error ({type(e).__name__}); retry in {WAITS[attempt]} s", flush=True)
            time.sleep(WAITS[attempt])


real_analyze, real_verify, real_feedback = g.analyze, g.verify, g.feedback_from
g.analyze = lambda *a, **k: with_retry(real_analyze, *a, **k)
retrying_verify = lambda *a, **k: with_retry(real_verify, *a, **k)
g.verify = retrying_verify


def logged_feedback(state):
    text = real_feedback(state)
    print("   feedback_from text used by repair_node:\n" + "\n".join("      " + l for l in text.splitlines()))
    return text
g.feedback_from = logged_feedback


def forcing_verify(n_forced):
    """The first n_forced calls return the real verification with claim 1 set to unsupported."""
    calls = {"n": 0}
    def fake(answer, evidence):
        v = retrying_verify(answer, evidence)
        calls["n"] += 1
        if calls["n"] <= n_forced:
            hit = [x for x in v.verdicts if x.claim_index == 1]
            if hit:
                hit[0].verdict, hit[0].problem = "unsupported", "forced test"
            else:
                v.verdicts.append(ClaimVerdict(claim_index=1, verdict="unsupported", problem="forced test"))
            print(f"   [verify call {calls['n']}: claim 1 forced to unsupported]")
        else:
            print(f"   [verify call {calls['n']}: real verdicts]")
        return v
    return fake


def show_verdicts(verdicts):
    print("   verdicts:")
    for v in verdicts or []:
        print(f"      {v['claim_index']}  {v['verdict']:<11}  {v.get('problem', '')}")


def show(question, state):
    print("   question:", question)
    for k in ("status", "intent", "language", "retries", "tried", "repairs", "dropped", "degraded"):
        print(f"   {k}: {state.get(k)}")
    show_verdicts(state.get("verdicts"))
    if state.get("repairs"):
        print("   draft_text (first, unverified):\n" + state.get("draft_text", ""))
    print("   text:\n" + state.get("text", ""))


def run_graph(name, question, n_forced=0):
    for attempt in range(len(WAITS) + 1):
        g.verify = forcing_verify(n_forced) if n_forced else retrying_verify
        try:
            return g.app.invoke({"question": question},
                                {"configurable": {"thread_id": f"vg-{name}-{RUN}-a{attempt}"}})
        except Exception as e:
            if not transient(e) or attempt == len(WAITS):
                raise
            print(f"   transient error ({type(e).__name__}); retry in {WAITS[attempt]} s", flush=True)
            time.sleep(WAITS[attempt])
        finally:
            g.verify = retrying_verify


def wrong_evidence(evidence):
    s = {"question": LEAVE, "full_question": LEAVE, "language": "fr", "evidence": evidence, "degraded": []}
    s.update(with_retry(g.answer_node, s))
    print("   question:", LEAVE)
    print("   evidence (from A):", list(evidence))
    print("   claims:")
    for i, c in enumerate(s["claims"], 1):
        print(f"      {i}. {c['text']}  {c['cited_ids']}")
    s.update(g.verify_node(s))
    show_verdicts(s["verdicts"])
    print("   bad:", s["bad"], " degraded:", s["degraded"])
    s.update(g.finalize_node(s))
    print("   dropped:", s["dropped"])
    print("   text:\n" + s["text"])


hit_evidence = None
for name, label, fn in [
    ("hit_fr", "A. hit_fr (normal run)", lambda: run_graph("hit_fr", HIT)),
    ("leave_fr", "B. leave_fr (normal run)", lambda: run_graph("leave_fr", LEAVE)),
    ("wrong_evidence", "C. leave question answered from A's evidence (nodes called directly)", None),
    ("forced_repair", "D. forced repair: claim 1 unsupported on the first verify call", lambda: run_graph("forced_repair", HIT, 1)),
    ("forced_drop", "E. forced drop: claim 1 unsupported on the first two verify calls", lambda: run_graph("forced_drop", HIT, 2)),
]:
    print(f"\n=== {label}")
    t = time.time()
    try:
        if name == "wrong_evidence":
            if not hit_evidence:
                print("   skipped: test A produced no evidence")
                continue
            wrong_evidence(hit_evidence)
        else:
            state = fn()
            show(HIT if name != "leave_fr" else LEAVE, state)
            if name == "hit_fr":
                hit_evidence = state.get("evidence")
    except Exception as e:
        print(f"   ERROR: {e.__class__.__name__}: {e}")
    print(f"   ({time.time() - t:.1f} s)")
