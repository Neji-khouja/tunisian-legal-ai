import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from agent_graph import app
from answer import make_answer, check_answer, render

tests = json.loads((HERE / "answer_tests.json").read_text(encoding="utf-8"))
evidence_by_name = {}
passed = 0


def stem_art(e):
    return (Path(str(e["source"])).stem, str(e["article"]))


for t in tests:
    print("\n" + "=" * 70)
    print(f"[{t['name']}] {t['question']}")
    try:
        if t.get("evidence_from"):
            # wrong evidence on purpose: make_answer must say so in the caveat
            evidence = evidence_by_name[t["evidence_from"]]
            ans = make_answer(t["question"], t["language"], evidence)
            problems = check_answer(ans, evidence)
            print(render(ans, evidence, t["language"]))
            ok = not problems and bool(ans.caveat.strip())
            print(f"  caveat present: {bool(ans.caveat.strip())}")
            print(f"  problems: {problems}")
        else:
            state = app.invoke({"question": t["question"]}, {"configurable": {"thread_id": t["name"]}})
            if state.get("status") != "done":
                print(f"  ERROR: graph ended in {state.get('status')!r}: {state.get('text')}")
                continue
            evidence_by_name[t["name"]] = state["evidence"]
            gold = {(t["gold_source"], a) for a in t["gold_articles"]}
            in_evidence = any(stem_art(e) in gold for e in state["evidence"].values())
            cited = {stem_art(state["evidence"][i]) for i in state["cited"] if i in state["evidence"]}
            ok = state["problems"] == [] and bool(cited & gold)
            print(f"  retries: {state.get('retries')}")
            print(f"  tried: {state.get('tried')}")
            print(f"  gold in evidence: {in_evidence}")
            print(f"  gold cited: {bool(cited & gold)}   cited: {sorted(cited)}")
            print(f"  problems: {state['problems']}")
            print(state["text"])
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        passed += ok
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

print(f"\n{passed}/{len(tests)} passed")
