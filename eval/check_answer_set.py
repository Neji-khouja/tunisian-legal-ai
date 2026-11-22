"""Validate eval/answer_set.json and print counts. Loads no model: the corpus (source, article) pairs are read
directly from the Chroma SQLite metadata of the index. Gold articles are also checked against the PDF text
(eval/pdf_truth.py, pypdf) when the source is verified; otherwise a "pdf unverified" warning is printed.

    python eval/check_answer_set.py
    python eval/check_answer_set.py --show a001        # PDF text of the gold articles of one question
"""
import sys
import json
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from config import DB_DIR
import pdf_truth

SET_PATH = ROOT / "eval" / "answer_set.json"
LANGS = {"ar", "fr", "en"}
EXPECTED = {"answer", "clarify", "refuse"}


def corpus_index():
    """-> ({(source, article or article_label)}, {source: law_type}), or (None, None) if the index is missing."""
    db = Path(DB_DIR) / "chroma.sqlite3"
    if not db.exists():
        return None, None
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    meta = {}
    for cid, key, val in con.execute(
            "select id, key, string_value from embedding_metadata where key in ('source','article','article_label','law_type')"):
        meta.setdefault(cid, {})[key] = val
    con.close()
    pairs, law = set(), {}
    for m in meta.values():
        pairs.add((m.get("source"), m.get("article")))
        pairs.add((m.get("source"), m.get("article_label")))
        law[m.get("source")] = m.get("law_type")
    return pairs, law


def check(entries, pairs) -> tuple[list[str], list[str]]:
    """-> (errors, warnings)."""
    errors, warnings, seen = [], [], set()
    if not isinstance(entries, list):
        return ["the file must contain a JSON list"], []
    for k, e in enumerate(entries):
        where = f"entry {k} (id={e.get('id') if isinstance(e, dict) else '?'})"
        if not isinstance(e, dict):
            errors.append(f"{where}: not an object"); continue
        for key in ("id", "question", "language", "expected"):
            if not e.get(key):
                errors.append(f"{where}: missing '{key}'")
        if e.get("id") in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(e.get("id"))
        if e.get("language") not in LANGS:
            errors.append(f"{where}: language must be one of {sorted(LANGS)}")
        exp, gold = e.get("expected"), e.get("gold", [])
        if exp not in EXPECTED:
            errors.append(f"{where}: expected must be one of {sorted(EXPECTED)}")
        elif exp == "answer":
            if not isinstance(gold, list) or not gold:
                errors.append(f"{where}: answer needs a non-empty gold list")
            else:
                check_gold(where, gold, pairs, errors, warnings)
            rp = e.get("reference_points")
            if not (isinstance(rp, list) and rp and all(isinstance(x, str) and x.strip() for x in rp)):
                errors.append(f"{where}: answer needs a non-empty reference_points list of strings")
        else:
            reply = e.get("reply")
            if reply is not None and (exp != "clarify" or not (isinstance(reply, str) and reply.strip())):
                errors.append(f"{where}: 'reply' must be a non-empty string, on clarify entries only")
            if gold and reply is None:
                errors.append(f"{where}: {exp} must have an empty gold list (unless it has a 'reply')")
            elif gold:
                check_gold(where, gold, pairs, errors, warnings)
            cand = e.get("candidate_gold")
            if cand is not None:
                if reply is None or not isinstance(cand, list) or not all(isinstance(c, str) and "/" in c for c in cand):
                    errors.append(f"{where}: 'candidate_gold' must be a list of chunk ids, on entries with a 'reply' only")
                else:
                    for c in cand:
                        stem, label = c.split("/")[0], c.split("/")[-1].split("~")[0]
                        if stem not in pdf_truth.load() or label not in pdf_truth.load()[stem]["articles"]:
                            errors.append(f"{where}: candidate_gold {c} not found in the PDF text")
            if not (isinstance(e.get("reason"), str) and e["reason"].strip()):
                errors.append(f"{where}: {exp} needs a non-empty 'reason'")
    return errors, warnings


def check_gold(where, gold, pairs, errors, warnings):
    for g in gold:
        if not (isinstance(g, dict) and g.get("source") and g.get("article")):
            errors.append(f"{where}: gold item {g!r} needs 'source' and 'article'")
            continue
        if pairs is not None and (g["source"], str(g["article"])) not in pairs:
            errors.append(f"{where}: gold ({g['source']}, {g['article']}) not found in the corpus index")
        if not pdf_truth.is_verified(g["source"]):
            warnings.append(f"{where}: gold ({g['source']}, {g['article']}): pdf unverified")
        elif str(g["article"]) not in pdf_truth.load()[pdf_truth.source_stem(g["source"])]["articles"]:
            errors.append(f"{where}: gold ({g['source']}, {g['article']}) not found in the PDF text")


def show(entries, qid):
    """Print the PDF text of every gold article of one question, to check the gold by eye."""
    e = next((x for x in entries if isinstance(x, dict) and x.get("id") == qid), None)
    if e is None:
        sys.exit(f"no entry with id {qid!r}")
    print(f"[{qid}] ({e.get('language')}, expected {e.get('expected')}) {e.get('question')}")
    for p in e.get("reference_points") or []:
        print(f"  reference point: {p}")
    for g in e.get("gold") or []:
        stem, label = pdf_truth.source_stem(g["source"]), str(g["article"])
        print(f"\n--- gold {stem} article {label}")
        if not pdf_truth.is_verified(stem):
            print("    pdf unverified: no ground-truth text")
            continue
        occ = pdf_truth.load()[stem]["articles"].get(label, [])
        if not occ:
            print("    not found in the PDF text")
        for sec, text in occ:
            print(f"    (PDF section {sec})\n{text}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Validate eval/answer_set.json (or --set FILE).")
    ap.add_argument("--set", default=str(SET_PATH))
    ap.add_argument("--show", metavar="ID", help="print the PDF text of the gold articles of one question")
    args = ap.parse_args()
    entries = json.loads(Path(args.set).read_text(encoding="utf-8"))
    if args.show:
        show(entries, args.show)
        return
    pairs, law = corpus_index()
    db_rel = Path(DB_DIR).relative_to(ROOT)
    if pairs is None:
        print(f"WARNING: no index at {db_rel}; the gold-exists-in-corpus check is SKIPPED.")
    else:
        print(f"Corpus index: {len(law)} sources, {len(pairs)} (source, article) keys, read from {db_rel}")
    pdf_states = {s: d["status"] for s, d in pdf_truth.load().items()}
    print(f"PDF text (pypdf): {sum(v == 'verified' for v in pdf_states.values())} verified sources, "
          f"unverified: {[s for s, v in pdf_states.items() if v != 'verified'] or 'none'}")

    errors, warnings = check(entries, pairs)
    ok = entries if isinstance(entries, list) else []
    print(f"\nTotal entries: {len(ok)}")
    print("By expected:", dict(Counter(e.get("expected") for e in ok if isinstance(e, dict))) or "{}")
    print("By language:", dict(Counter(e.get("language") for e in ok if isinstance(e, dict))) or "{}")

    if law is not None:
        per_code = Counter()
        for e in ok:
            if isinstance(e, dict) and e.get("expected") == "answer":
                for code in {law.get(g.get("source")) for g in e.get("gold", []) if isinstance(g, dict)} - {None}:
                    per_code[code] += 1
        print("By source (answer questions per code):")
        for code in sorted(set(law.values())):
            n = per_code[code]
            print(f"  {code:<28}{n:>4}{'   WARNING: no question' if n == 0 else ''}")

    for msg in warnings:
        print("WARNING:", msg)
    if errors:
        print(f"\nINVALID: {len(errors)} problem(s)")
        for msg in errors:
            print("  -", msg)
        sys.exit(1)
    print("\nAll entries valid.")


if __name__ == "__main__":
    main()
