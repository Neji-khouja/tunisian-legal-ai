"""Split the exam (eval/answer_set.json) into two stages -> eval/stages.json, then refresh eval/exam_hashes.txt.

    python eval/make_stages.py
"""
import json
import random
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
EXAM = ROOT / "eval" / "answer_set.json"
OUT = ROOT / "eval" / "stages.json"
SEED, N_DRAWS = 20261022, 20000
TARGET = {"answer": (30, 70), "clarify": (10, 15), "refuse": (10, 15)}
REPLY_S1 = (3, 5)                     # clarify items with a scripted reply: 3 of 5 in stage 1


def main_code(e):
    return next(g for g in e["gold"] if g.get("role", "main") == "main")["source"][:-4]


def strata(e):
    if e["expected"] == "answer":
        return [("code", main_code(e)), ("type", e.get("type", "")), ("lang", e["language"])]
    return [("lang", e["language"])]


def cost(sel, members, k):
    n = len(members)
    full = Counter(s for e in members for s in strata(e))
    got = Counter(s for e in sel for s in strata(e))
    return sum(abs(got[s] - full[s] * k / n) for s in full)


def draw(members, k, rng, k_reply):
    if k_reply is None:
        return rng.sample(members, k)
    rep = [e for e in members if e.get("reply")]
    rest = [e for e in members if not e.get("reply")]
    return rng.sample(rep, k_reply) + rng.sample(rest, k - k_reply)


def main():
    items = json.loads(EXAM.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    s1, notes = [], []
    for g, (k, n0) in TARGET.items():
        members = [e for e in items if e["expected"] == g]
        if len(members) != n0:                                # exam size differs: about half of the group
            k = round(len(members) * k / n0) if n0 else len(members) // 2
            notes.append(f"{g}: {len(members)} items instead of {n0}, stage 1 takes {k}")
        k_reply = None
        if g == "clarify":
            n_rep = sum(bool(e.get("reply")) for e in members)
            k_reply = REPLY_S1[0] if n_rep == REPLY_S1[1] else round(n_rep * REPLY_S1[0] / REPLY_S1[1])
            if n_rep != REPLY_S1[1]:
                notes.append(f"clarify with a reply: {n_rep} instead of {REPLY_S1[1]}, stage 1 takes {k_reply}")
        best = min((draw(members, k, rng, k_reply) for _ in range(N_DRAWS)), key=lambda sel: cost(sel, members, k))
        print(f"{g}: best cost {cost(best, members, k):.2f} over {N_DRAWS} draws")
        s1 += best
    ids1 = {e["id"] for e in s1}
    stage1 = [e["id"] for e in items if e["id"] in ids1]
    stage2 = [e["id"] for e in items if e["id"] not in ids1]
    OUT.write_text(json.dumps({"seed": SEED, "exam": "eval/answer_set.json", "method": __doc__.split("Method")[1].strip(),
                               "notes": notes, "stage1": stage1, "stage2": stage2}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: stage 1 {len(stage1)} items, stage 2 {len(stage2)} items")
    for n in notes:
        print("NOTE (scaled):", n)
    by_id = {e["id"]: e for e in items}
    for name, ids in (("stage 1", stage1), ("stage 2", stage2)):
        es = [by_id[i] for i in ids]
        print(f"\n{name}: {len(es)} items; by expected {dict(Counter(e['expected'] for e in es))}; "
              f"scripted replies {sum(bool(e.get('reply')) for e in es)}")
        types = sorted({e.get("type", e["expected"]) for e in items})
        print(f"  {'type':<12}" + "".join(f"{l:>5}" for l in ("fr", "en", "ar")) + "  total")
        for t in types:
            row = [sum(e.get("type", e["expected"]) == t and e["language"] == l for e in es) for l in ("fr", "en", "ar")]
            print(f"  {t:<12}" + "".join(f"{x:>5}" for x in row) + f"{sum(row):>7}")
        print("  answer by source:", dict(sorted(Counter(main_code(e) for e in es if e["expected"] == "answer").items())))
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "missing"
    hashes = ROOT / "eval" / "exam_hashes.txt"
    hashes.write_text("eval/answer_set.json " + sha(ROOT / "eval" / "answer_set.json") + "\n"
                      "eval/stages.json " + sha(OUT) + "\n", encoding="utf-8")
    print(f"wrote {hashes.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
