"""Validate a fresh-question file (fresh/questions_template.json format). No LLM; the indexes are read through
sqlite in read-only immutable mode (never written, no Chroma open).

    python fresh/validate_fresh.py fresh/questions.json [--json REPORT.json] [--threshold 0.68]
    python fresh/validate_fresh.py --lookup PenalFr 243        # chunk ids and start of text of an article (labellers)
    python fresh/validate_fresh.py --calibrate                 # near-duplicate threshold calibration -> fresh/near_dup_calibration.json
"""
import re
import sys
import json
import sqlite3
import argparse
import unicodedata
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "agent")]

LANGUAGES = ("fr", "en", "ar")
TYPES = ("clear", "messy", "vague", "out_of_scope")
REQUIRED = ("id", "language", "type", "question", "gold_articles", "gold_notes", "labeller", "labelled_on", "anonymised")
GOLD_ID_RE = re.compile(r"^[A-Za-z_]+/s\d{2}/[0-9A-Za-z-]+$")
TARGET_TOTAL = 150
MIN_PER_LANGUAGE = 30
NEAR_DUP_THRESHOLD = 0.68
                                # 43/47 translations, 1/1825 distinct same-language exam pairs
INDEXES = {"db_v1_2": ROOT / "db_v1_2", "db_v1_1": ROOT / "db_v1_1"}


# ---------------------------------------------------------------- inputs

def load_entries(path) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["items"] if isinstance(data, dict) else data


def index_ids(db_dir) -> set:
    """chunk ids of an index, read-only (sqlite immutable URI: no write, no lock)."""
    f = Path(db_dir) / "chroma.sqlite3"
    con = sqlite3.connect(f"file:{f.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        return {r[0] for r in con.execute("select string_value from embedding_metadata where key = 'chunk_id'")}
    finally:
        con.close()


def lookup(stem, article, db_dir=INDEXES["db_v1_2"]) -> list:
    """[(chunk id, start of the article text)] for a code file stem and an article label (e.g. PenalFr 243, COC 94)."""
    f = Path(db_dir) / "chroma.sqlite3"
    con = sqlite3.connect(f"file:{f.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        rows = con.execute("""select c.string_value, coalesce(t.string_value, d.string_value) from embedding_metadata c
                              join embedding_metadata d on d.id = c.id and d.key = 'chroma:document'
                              left join embedding_metadata t on t.id = c.id and t.key = 'article_text'
                              where c.key = 'chunk_id' and c.string_value like ?""", (f"{stem}/%",)).fetchall()
    finally:
        con.close()
    out = {}
    for cid, text in rows:
        if cid.split("/")[-1] == str(article):
            out.setdefault(cid, " ".join(text.split())[:300])
    return sorted(out.items())


def reference_questions() -> list:
    """(source, id, text): every eval/*.json question, the dev items and translations, the prompt examples."""
    import prompt_overlap as po
    out = [(f, i, q) for f, i, q, _ in po.corpus_questions()]
    out += [(f"prompt {e['file']}", "-", e["question"]) for e in po.extract()]
    import normalize as N
    out += [("prompt normalize.py", kind, q) for kind, q, _ in po.prompt_examples(N.NORMALIZE_PROMPT)]
    return out


def real_embed():
    from pipeline import get_embeddings
    emb = get_embeddings()
    return lambda texts: emb.embed_documents(list(texts))


# ---------------------------------------------------------------- checks

def norm_text(q) -> str:
    q = unicodedata.normalize("NFKC", q).casefold().replace("’", "'")
    return " ".join(re.sub(r"[^\w\s]", " ", q).split())


def check_entry(e, ids_by_index) -> tuple:
    """-> (errors, warnings) for one entry."""
    err, warn = [], []
    for k in REQUIRED:
        if k not in e:
            err.append(f"missing field {k}")
    if err:
        return err, warn
    if not isinstance(e["id"], str) or not e["id"].strip():
        err.append("id must be a non-empty string")
    if e["language"] not in LANGUAGES:
        err.append(f"language {e['language']!r} not in {LANGUAGES}")
    if e["type"] not in TYPES:
        err.append(f"type {e['type']!r} not in {TYPES}")
    if not isinstance(e["question"], str) or not e["question"].strip():
        err.append("question must be a non-empty string")
    gold = e["gold_articles"]
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
        err.append("gold_articles must be a list of strings")
        gold = []
    if e["type"] == "out_of_scope" and gold:
        err.append("out_of_scope: gold_articles must be empty")
    if e["type"] in ("clear", "messy", "vague") and not gold:
        err.append(f"{e['type']}: at least one gold article")
    if e["type"] == "vague" and not str(e["gold_notes"] or "").strip():
        err.append("vague: gold_notes must say which clarification resolves the question")
    if len(set(gold)) != len(gold):
        err.append("duplicate gold article")
    for g in gold:
        if not GOLD_ID_RE.match(g):
            err.append(f"gold id {g!r}: format <code>/sNN/<article>, e.g. PenalFr/s02/243")
            continue
        for name, ids in ids_by_index.items():
            if g not in ids:
                err.append(f"gold id {g} not in {name}")
    if not isinstance(e["labeller"], str) or not e["labeller"].strip():
        err.append("labeller must be a non-empty string")
    try:
        date.fromisoformat(str(e["labelled_on"]))
    except ValueError:
        err.append(f"labelled_on {e['labelled_on']!r}: YYYY-MM-DD")
    if e["anonymised"] is not True:
        err.append("anonymised must be true (no personal data in the question)")
    return err, warn


def cosine_matrix(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def near_duplicates(entries, refs, embed, threshold) -> dict:
    """{"reference": [(fresh id, similarity, source, ref id, ref text)], "within": [(id1, id2, similarity)]}."""
    if not entries:
        return {"reference": [], "within": []}
    qe = embed([e["question"] for e in entries])
    out = {"reference": [], "within": []}
    if refs:
        S = cosine_matrix(qe, embed([r[2] for r in refs]))
        for i, e in enumerate(entries):
            for j in sorted(range(len(refs)), key=lambda j: -S[i, j]):
                if S[i, j] < threshold:
                    break
                out["reference"].append((e["id"], round(float(S[i, j]), 4), refs[j][0], refs[j][1], refs[j][2]))
    W = cosine_matrix(qe, qe)
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if W[i, j] >= threshold:
                out["within"].append((entries[i]["id"], entries[j]["id"], round(float(W[i, j]), 4)))
    return out


def validate(entries, ids_by_index, refs=(), embed=None, threshold=NEAR_DUP_THRESHOLD) -> dict:
    examples = [e.get("id") for e in entries if e.get("example")]
    real = [e for e in entries if not e.get("example")]
    per = {}
    for e in real:
        err, warn = check_entry(e, ids_by_index)
        if err or warn:
            per[str(e.get("id"))] = {"errors": err, "warnings": warn}
    seen, dup_ids = {}, []
    for e in real:
        if e.get("id") in seen:
            dup_ids.append(e.get("id"))
        seen[e.get("id")] = 1
    by_text = {}
    for e in real:
        if isinstance(e.get("question"), str):
            by_text.setdefault(norm_text(e["question"]), []).append(e.get("id"))
    dup_q = [v for v in by_text.values() if len(v) > 1]
    ok_q = [e for e in real if isinstance(e.get("question"), str) and e["question"].strip()]
    nd = near_duplicates(ok_q, list(refs), embed, threshold) if embed else {"reference": [], "within": []}
    lang = {l: sum(e.get("language") == l for e in real) for l in LANGUAGES}
    typ = {t: sum(e.get("type") == t for e in real) for t in TYPES}
    minimums = {"total": {"target": TARGET_TOTAL, "n": len(real), "met": len(real) >= TARGET_TOTAL},
                **{f"language {l}": {"target": MIN_PER_LANGUAGE, "n": lang[l], "met": lang[l] >= MIN_PER_LANGUAGE}
                   for l in LANGUAGES}}
    n_err = sum(len(v["errors"]) for v in per.values()) + len(dup_ids) + len(dup_q)
    return {"entries": len(real), "examples_ignored": examples, "per_entry": per, "duplicate_ids": dup_ids,
            "duplicate_questions": dup_q, "near_duplicates": nd, "threshold": threshold, "by_language": lang,
            "by_type": typ, "minimums": minimums, "errors": n_err,
            "warnings": len(nd["reference"]) + len(nd["within"]) + sum(not m["met"] for m in minimums.values())
                        + (1 if examples else 0)}


def report_text(r) -> str:
    L = [f"{r['entries']} entries (examples ignored: {', '.join(r['examples_ignored']) or 'none'})"]
    for i, v in r["per_entry"].items():
        for m in v["errors"]:
            L.append(f"ERROR   {i}: {m}")
        for m in v["warnings"]:
            L.append(f"WARNING {i}: {m}")
    L += [f"ERROR   duplicate id {i}" for i in r["duplicate_ids"]]
    L += [f"ERROR   same question in {', '.join(map(str, v))}" for v in r["duplicate_questions"]]
    L.append(f"near-duplicates (cosine >= {r['threshold']}):")
    L += [f"WARNING {i} ~ {src} {rid} ({s:.3f}): {txt[:100]!r}" for i, s, src, rid, txt in r["near_duplicates"]["reference"]]
    L += [f"WARNING {a} ~ {b} in this file ({s:.3f})" for a, b, s in r["near_duplicates"]["within"]]
    if not r["near_duplicates"]["reference"] and not r["near_duplicates"]["within"]:
        L.append("  none")
    L.append("per language: " + ", ".join(f"{k} {v}" for k, v in r["by_language"].items()))
    L.append("per type: " + ", ".join(f"{k} {v}" for k, v in r["by_type"].items()))
    for k, m in r["minimums"].items():
        L.append(f"minimum {k}: {m['n']} / {m['target']} -> {'met' if m['met'] else 'NOT met'}")
    L.append(f"{r['errors']} error(s), {r['warnings']} warning(s)")
    return "\n".join(L)


# ---------------------------------------------------------------- threshold calibration

def calibration_pairs() -> dict:
    """Known near-duplicates and known distinct questions of the exam files."""
    exam = json.loads((ROOT / "eval" / "answer_set.json").read_text(encoding="utf-8"))
    noisy = json.loads((ROOT / "eval" / "noisy_set.json").read_text(encoding="utf-8"))
    tr = json.loads((ROOT / "diagnostics" / "translations.json").read_text(encoding="utf-8"))["items"]
    by = {i["id"]: i for i in exam}
    pos_noisy = [(n["question"], by[n["original_id"]]["question"], n["language"], n["noise_type"])
                 for n in noisy if n.get("original_id") in by]
    pos_trans = [(t["french_query"], t["query"], t["language"], "translation") for t in tr]
    neg = [(a["question"], b["question"], a["language"], b["language"])
           for k, a in enumerate(exam) for b in exam[k + 1:]]
    return {"pos_noisy": pos_noisy, "pos_translation": pos_trans, "neg": neg}


def near_dup_rule(pos_same, neg_same) -> float:
    """The threshold: the largest value (step 0.01) that still flags at least 90% of the same-language known
    near-duplicates (noisy rewrites of an exam question), reported with the share of distinct exam pairs it flags."""
    t = 1.0
    while t > 0 and sum(s >= t for s in pos_same) < 0.9 * len(pos_same):
        t = round(t - 0.01, 2)
    return t


def calibrate(embed, out=HERE / "near_dup_calibration.json") -> dict:
    P = calibration_pairs()
    texts = sorted({x for k in P for p in P[k] for x in p[:2]})
    E = dict(zip(texts, embed(texts)))
    sim = lambda a, b: float(cosine_matrix([E[a]], [E[b]])[0, 0])
    pos_noisy = [(sim(a, b), l, t) for a, b, l, t in P["pos_noisy"]]
    pos_tr = [(sim(a, b), l) for a, b, l, _ in P["pos_translation"]]
    neg = [(sim(a, b), la, lb) for a, b, la, lb in P["neg"]]
    neg_same = [s for s, la, lb in neg if la == lb]
    rule_t = near_dup_rule([s for s, *_ in pos_noisy], neg_same)
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    rep = {"model": __import__("config").EMBEDDING_MODEL, "rule": near_dup_rule.__doc__.strip(), "threshold": rule_t,
           "positives_noisy": {"n": len(pos_noisy), "min": min(s for s, *_ in pos_noisy), "p10": q([s for s, *_ in pos_noisy], 0.1),
                               "median": q([s for s, *_ in pos_noisy], 0.5),
                               "flagged": sum(s >= rule_t for s, *_ in pos_noisy),
                               "by_noise_type": {t: {"n": sum(x[2] == t for x in pos_noisy),
                                                     "flagged": sum(x[2] == t and x[0] >= rule_t for x in pos_noisy)}
                                                 for t in sorted({x[2] for x in pos_noisy})},
                               "by_language": {l: {"n": sum(x[1] == l for x in pos_noisy),
                                                   "flagged": sum(x[1] == l and x[0] >= rule_t for x in pos_noisy)}
                                               for l in LANGUAGES}},
           "positives_translation": {"n": len(pos_tr), "median": q([s for s, _ in pos_tr], 0.5),
                                     "flagged": sum(s >= rule_t for s, _ in pos_tr),
                                     "by_language": {l: {"n": sum(x[1] == l for x in pos_tr),
                                                         "flagged": sum(x[1] == l and x[0] >= rule_t for x in pos_tr)}
                                                     for l in ("ar", "en")}},
           "negatives_same_language": {"n": len(neg_same), "max": max(neg_same), "p99": q(neg_same, 0.99),
                                       "flagged": sum(s >= rule_t for s in neg_same)},
           "negatives_cross_language": {"n": len(neg) - len(neg_same), "max": max(s for s, la, lb in neg if la != lb),
                                        "flagged": sum(s >= rule_t for s, la, lb in neg if la != lb)},
           "grid": [{"threshold": t, "noisy_flagged": sum(s >= t for s, *_ in pos_noisy),
                     "translations_flagged": sum(s >= t for s, _ in pos_tr),
                     "distinct_same_language_flagged": sum(s >= t for s in neg_same)}
                    for t in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95)]}
    Path(out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--threshold", type=float, default=NEAR_DUP_THRESHOLD)
    ap.add_argument("--no-embeddings", action="store_true", help="skip the near-duplicate check")
    ap.add_argument("--lookup", nargs=2, metavar=("CODE", "ARTICLE"))
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args(argv)
    if a.lookup:
        rows = lookup(*a.lookup)
        print("\n".join(f"{cid}  |  {t}" for cid, t in rows) or "no article with this label in db_v1_2")
        return 0
    if a.calibrate:
        print(json.dumps(calibrate(real_embed()), ensure_ascii=False, indent=1))
        return 0
    if not a.file:
        ap.error("a questions file is required")
    entries = load_entries(a.file)
    ids = {name: index_ids(p) for name, p in INDEXES.items()}
    embed = None if a.no_embeddings else real_embed()
    r = validate(entries, ids, reference_questions() if embed else (), embed, a.threshold)
    print(report_text(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if r["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
