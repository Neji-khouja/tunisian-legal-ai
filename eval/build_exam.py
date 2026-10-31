"""Build eval/answer_set.json (the exam) from the two candidate files, then run the checks. No LLM call.

    python eval/build_exam.py            # build + checks + eval/exam_hashes.txt
    python eval/build_exam.py --check    # checks only (the exam is not rewritten)

Rule: every candidate except REJECTED_IDS, status "final"; order answerable, clarify, refuse. Nothing else is decided here.
Checks: eval/check_answer_set.py, every gold / candidate_gold chunk id in the index (chroma metadata of config.DB_DIR, the
index the agent retrieves from) and in the PDF text (pdf_truth.from_chunk_id, as the harness reads it), duplicate
questions, gold chunks shared by two items (reported), eval/prompt_overlap.py --extra (no pair >= 0.4).
"""
import sys
import json
import sqlite3
import hashlib
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from config import DB_DIR
import pdf_truth

REJECTED_IDS = []      # filled in by the user: ids left out of the exam
SOURCES = [ROOT / "eval" / "answer_set_answerable_candidates.json", ROOT / "eval" / "answer_set_candidates.json"]
EXAM = ROOT / "eval" / "answer_set.json"
STAGES = ROOT / "eval" / "stages.json"
HASHES = ROOT / "eval" / "exam_hashes.txt"
ORDER = {"answer": 0, "clarify": 1, "refuse": 2}
WANT = {"answer": 70, "clarify": 15, "refuse": 15}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def write_hashes():
    HASHES.write_text(f"eval/answer_set.json {sha256(EXAM)}\neval/stages.json {sha256(STAGES)}\n", encoding="utf-8")
    print(f"wrote {HASHES.relative_to(ROOT).as_posix()}:\n" + HASHES.read_text(encoding="utf-8"))


def build() -> list:
    items = [e for p in SOURCES for e in json.loads(p.read_text(encoding="utf-8"))]
    kept = [dict(e, status="final") for e in items if e["id"] not in REJECTED_IDS]
    kept.sort(key=lambda e: ORDER[e["expected"]])                     # stable: file order inside each type
    EXAM.write_text("[\n" + ",\n".join(json.dumps(e, ensure_ascii=False) for e in kept) + "\n]\n", encoding="utf-8")
    print(f"wrote {EXAM.relative_to(ROOT).as_posix()}: {len(kept)} items ({len(items) - len(kept)} rejected: {REJECTED_IDS})")
    return kept


def index_chunk_ids() -> set:
    con = sqlite3.connect(f"file:{(Path(DB_DIR) / 'chroma.sqlite3').as_posix()}?mode=ro", uri=True)
    ids = {r[0] for r in con.execute("select string_value from embedding_metadata where key='chunk_id'")}
    con.close()
    return ids


def checks(items) -> int:
    bad = 0
    r = subprocess.run([sys.executable, str(ROOT / "eval" / "check_answer_set.py"), "--set", str(EXAM)],
                       capture_output=True, text=True, encoding="utf-8")
    print("check_answer_set.py:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
    bad += r.returncode != 0
    ids = Counter(e["id"] for e in items)
    dup_ids = [i for i, n in ids.items() if n > 1]
    print(f"duplicate ids: {dup_ids or 'none'}")
    bad += bool(dup_ids)
    qs = Counter(" ".join(e["question"].split()).lower() for e in items)
    dup_q = [q for q, n in qs.items() if n > 1]
    print(f"duplicate questions: {dup_q or 'none'}")
    bad += bool(dup_q)
    index, owners, missing, no_pdf = index_chunk_ids(), defaultdict(set), [], []
    for e in items:
        cids = [c for g in e.get("gold") or [] for c in g.get("chunk_ids") or []] + list(e.get("candidate_gold") or [])
        for g in e.get("gold") or []:
            if not g.get("chunk_ids"):
                missing.append(f"{e['id']}: gold {g['source']} {g['article']} has no chunk_ids")
        for c in cids:
            owners[c].add(e["id"])
            if c not in index:
                missing.append(f"{e['id']}: {c} not in the index ({Path(DB_DIR).name})")
            elif not pdf_truth.from_chunk_id(c):
                no_pdf.append(f"{e['id']}: {c}")
    print(f"gold chunk ids checked: {len(owners)}; not in the index: {missing or 'none'}")
    print(f"in the index but no PDF text by chunk id (judge then reads the index chunk, marked FALLBACK; "
          f"reported, not an error): {no_pdf or 'none'}")
    bad += bool(missing)
    shared = {c: sorted(o) for c, o in owners.items() if len(o) > 1}
    print(f"gold chunks shared by two or more items (reported, not an error): {shared or 'none'}")
    extra = [f"{e['question']}|{e['language']}" for e in items if e["expected"] == "answer"]
    r = subprocess.run([sys.executable, str(ROOT / "eval" / "prompt_overlap.py"), "--extra", *extra],
                       capture_output=True, text=True, encoding="utf-8")
    out = r.stdout
    pairs = out[out.find("Pairs with"):out.find("pairs)") + 6] if "Pairs with" in out else out[-500:]
    print("prompt_overlap.py --extra (answerable items as new1..; gate + clarify/refuse candidates read from their files):")
    print("  " + pairs.replace("\n", "\n  "))
    n_pairs = int(pairs.rsplit("(", 1)[1].split()[0]) if "pairs)" in pairs else -1
    bad += n_pairs != 0

    print("\nCounts")
    print("  by expected:", dict(Counter(e["expected"] for e in items)))
    short = {k: v - sum(e["expected"] == k for e in items) for k, v in WANT.items()}
    print("  shortfall vs 70/15/15:", {k: v for k, v in short.items() if v > 0} or "none")
    print("  by type:", dict(Counter(e.get("type", e["expected"]) for e in items)))
    print("  by language:", dict(Counter(e["language"] for e in items)))
    print("  by language x expected:", dict(Counter(f"{e['expected']}/{e['language']}" for e in items)))
    code = Counter()
    for e in items:
        for s in {g["source"].replace(".pdf", "") for g in e.get("gold") or [] if g.get("role", "main") == "main"}:
            code[s] += 1
    print("  answerable by source (main gold):", dict(sorted(code.items())))
    print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
    return bad


if __name__ == "__main__":
    items = json.loads(EXAM.read_text(encoding="utf-8")) if "--check" in sys.argv else build()
    n_bad = checks(items)
    if "--check" not in sys.argv:
        write_hashes()
    sys.exit(1 if n_bad else 0)
