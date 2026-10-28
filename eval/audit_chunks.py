"""Audit the index chunks against the PDF text (pypdf, eval/pdf_truth.py). No model, no LLM.

    python eval/audit_chunks.py        -> prints the report, saves results/audit_chunks.json

(1) coverage: article labels found in the PDF vs labels present in the index, per source
(2) fidelity: 150 chunks sampled with seed 20261022; normalize(chunk) in normalize(PDF article) or vice versa
(3) base-number collisions: chunks of one source+section sharing the same base article number (220 vs 220-bis)
"""
import sys
import json
import random
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_truth import load, normalize, from_chunk_id, HEADING, ROOT

SEED = 20261022
N_SAMPLE = 150
DB = ROOT / "db_v1_1" / "chroma.sqlite3"
OUT = ROOT / "results" / "audit_chunks.json"


def index_rows() -> list[dict]:
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    meta = defaultdict(dict)
    for i, key, val in con.execute("select id, key, string_value from embedding_metadata where key in "
                                   "('chunk_id', 'article', 'article_label', 'source', 'chroma:document')"):
        meta[i][key] = val
    con.close()
    rows = [{"cid": m["chunk_id"], "stem": m["chunk_id"].split("/")[0], "section": int(m["chunk_id"].split("/")[1][1:]),
             "label": m["article_label"], "article": m["article"], "text": m["chroma:document"]} for m in meta.values()]
    return sorted(rows, key=lambda r: r["cid"])


def strip_heading(t: str) -> str:
    m = HEADING.match(t)
    return t[m.end():] if m else t


def _flat(t: str) -> str:
    """normalize() without any whitespace: pypdf and PyMuPDF space words differently ("j uge", "justice (*)")."""
    return normalize(t).replace(" ", "")


def contains_either(a: str, b: str) -> bool:
    """chunk text inside the PDF article text or vice versa, whitespace-insensitive; also tried without the heading."""
    for x, y in ((a, b), (strip_heading(a), strip_heading(b))):
        x, y = _flat(x), _flat(y)
        if x and y and (x in y or y in x):
            return True
    return False


def main():
    pdf = load()
    rows = index_rows()
    out = {"seed": SEED, "n_sample": N_SAMPLE, "extractor": "pypdf", "coverage": {}, "fidelity": {}, "base_collisions": {}}

    # (1) coverage
    print("(1) COVERAGE: article labels in the PDF vs in the index (sections ignored)")
    print(f"  {'source':<20}{'status':<12}{'pdf':>6}{'index':>7}{'only pdf':>10}{'only idx':>10}{'pdf sect':>10}{'idx sect':>10}")
    idx_by = defaultdict(list)
    for r in rows:
        idx_by[r["stem"]].append(r)
    for stem in sorted(set(pdf) | set(idx_by)):
        d = pdf.get(stem, {"status": "missing", "articles": {}})
        pdf_labels = set(d["articles"])
        idx_labels = {r["label"] for r in idx_by.get(stem, [])}
        only_pdf, only_idx = sorted(pdf_labels - idx_labels), sorted(idx_labels - pdf_labels)
        pdf_sections = len({s for occ in d["articles"].values() for s, _ in occ})
        idx_sections = len({r["section"] for r in idx_by.get(stem, [])})
        out["coverage"][stem] = {"status": d["status"], "pdf_labels": len(pdf_labels), "index_labels": len(idx_labels),
                                 "only_pdf": only_pdf, "only_index": only_idx,
                                 "pdf_sections": pdf_sections, "index_sections": idx_sections}
        print(f"  {stem:<20}{d['status']:<12}{len(pdf_labels):>6}{len(idx_labels):>7}{len(only_pdf):>10}{len(only_idx):>10}"
              f"{pdf_sections:>10}{idx_sections:>10}")
    for stem, c in out["coverage"].items():
        if c["only_pdf"] or c["only_index"]:
            print(f"    {stem}: only in PDF (first 10): {c['only_pdf'][:10]}   only in index (first 10): {c['only_index'][:10]}")

    # (2) fidelity
    sample = random.Random(SEED).sample(rows, N_SAMPLE)
    res = []
    for r in sample:
        text = from_chunk_id(r["cid"])
        status = "no_pdf_article" if text is None else ("match" if contains_either(r["text"], text) else "mismatch")
        res.append({"cid": r["cid"], "stem": r["stem"], "status": status,
                    "chunk": r["text"], "pdf": text})
    by_src = defaultdict(Counter)
    for x in res:
        by_src[x["stem"]][x["status"]] += 1
    tot = Counter(x["status"] for x in res)
    checked = tot["match"] + tot["mismatch"]
    print(f"\n(2) FIDELITY: {N_SAMPLE} chunks, seed {SEED}")
    print(f"  overall: match {tot['match']}, mismatch {tot['mismatch']}, no PDF article (absent/ambiguous/unverified) "
          f"{tot['no_pdf_article']}; mismatch rate among checked: {tot['mismatch']}/{checked}"
          f" = {tot['mismatch'] / checked:.1%}" if checked else "  nothing checked")
    print(f"  {'source':<20}{'n':>4}{'match':>7}{'mismatch':>10}{'no pdf':>8}")
    for stem in sorted(by_src):
        c = by_src[stem]
        print(f"  {stem:<20}{sum(c.values()):>4}{c['match']:>7}{c['mismatch']:>10}{c['no_pdf_article']:>8}")
    mism = [x for x in res if x["status"] == "mismatch"]
    for x in mism[:10]:
        c, p = _flat(x["chunk"]), _flat(x["pdf"])
        i = next((k for k in range(min(len(c), len(p))) if c[k] != p[k]), min(len(c), len(p)))
        print(f"\n  MISMATCH {x['cid']}   (whitespace-free texts first differ at char {i}: "
              f"chunk ...{c[max(0, i - 25):i + 45]!r}  pdf ...{p[max(0, i - 25):i + 45]!r})"
              f"\n    chunk: {normalize(x['chunk'])[:300]}\n    pdf:   {normalize(x['pdf'])[:300]}")
    for x in [x for x in res if x["status"] == "no_pdf_article"][:10]:
        print(f"  NO PDF ARTICLE {x['cid']}")
    out["fidelity"] = {"counts": dict(tot), "by_source": {k: dict(v) for k, v in by_src.items()},
                       "rows": [{k: x[k] for k in ("cid", "status")} for x in res],
                       "mismatches": [{"cid": x["cid"], "chunk": x["chunk"][:300], "pdf": (x["pdf"] or "")[:300]} for x in mism]}

    # (3) base-number collisions
    groups = defaultdict(set)
    for r in rows:
        groups[(r["stem"], r["section"], r["article"])].add(r["label"])
    coll = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    src_with = Counter(k[0] for k in coll)
    print(f"\n(3) BASE NUMBER SHARED BY SEVERAL CHUNK LABELS (field 'article' holds the base number): "
          f"{len(coll)} cases in {len(src_with)} sources")
    for stem, n in sorted(src_with.items()):
        ex = [f"s{k[1]:02d}:{'/'.join(v)}" for k, v in sorted(coll.items()) if k[0] == stem][:5]
        print(f"  {stem:<20}{n:>5} cases   e.g. {', '.join(ex)}")
    out["base_collisions"] = {"cases": len(coll), "sources": dict(src_with),
                              "list": [{"stem": k[0], "section": k[1], "article": k[2], "labels": v} for k, v in sorted(coll.items())]}

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
