import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from analyze import analyze
from retrieve_g import retrieve_g
from agent_graph import to_evidence
from answer import Answer, Claim, make_answer
from verify import verify, bad_claims


def get_evidence(q):
    a = analyze(q)
    return to_evidence(retrieve_g(q, list(a.target_codes)))


def show(answer, v):
    by_index = {x.claim_index: x for x in v.verdicts}
    for i, c in enumerate(answer.claims, 1):
        x = by_index.get(i)
        print(f"  {i}. [{x.verdict if x else 'MISSING'}] {c.text[:110]}")
        if x and x.problem:
            print(f"       problem: {x.problem}")


# ---- Part 1: known-bad claims must NOT come back "supported" ----
q = "Que risque-t-on si on frappe quelqu'un ?"
evidence = get_evidence(q)
id220 = next((cid for cid in evidence if cid.startswith("PenalFr/") and cid.endswith("/220")), None)  # not 220-bis
if id220 is None:
    sys.exit("article 220 is not in the evidence for this question; pick another mutation base")
other = next(cid for cid in evidence if cid != id220)

good = Claim(text="Participer à une rixe au cours de laquelle sont portés des coups et blessures entraîne "
                  "un emprisonnement de six mois pour ce seul fait.", cited_ids=[id220])
wrong_number = Claim(text="Participer à une rixe au cours de laquelle sont portés des coups et blessures "
                          "entraîne un emprisonnement de deux ans pour ce seul fait.", cited_ids=[id220])
wrong_article = Claim(text=good.text, cited_ids=[other])
invented = Claim(text="Celui qui frappe quelqu'un est puni d'une amende de cent dinars.", cited_ids=[id220])

test = Answer(claims=[good, wrong_number, wrong_article, invented])
v = verify(test, evidence)
print("PART 1: mutation test (expect: 1 supported; 2, 3, 4 NOT supported)")
show(test, v)
bad = bad_claims(test, v)
print("  ->", "PASS" if bad == [2, 3, 4] else f"FAIL (not-supported indexes: {bad})")

# ---- Part 2: real answers, read the verdicts yourself ----
for q, lang in [("Que risque-t-on si on frappe quelqu'un ?", "fr"),
                ("Quelle est la durée du congé annuel payé ?", "fr"),
                ("How many days of paid annual leave is a worker entitled to?", "en")]:
    print("\n" + "=" * 70 + f"\nPART 2: {q}")
    try:
        ev = get_evidence(q)
        ans = make_answer(q, lang, ev)
        show(ans, verify(ans, ev))
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")