"""Mojibake scan of the index chunks and of two PDF extractions (pypdf, PyMuPDF). No model, no LLM. Fixes nothing.

    python eval/mojibake_scan.py

Patterns: "â€", "Ã©", "Ã¨", "Ã " (also "Ã" + no-break space, the usual mojibake of "à"), "Â", U+FFFD.
A "text" is one index chunk, or one PDF page for the extractions (raw page text, before any cleaning).
Up to 5 examples per (where, source), 200 characters around the first match, printed with repr().
Then: every place where "quelqu" is followed by "â€™" (raw repr of 200 characters around it).
"""
import sys
import sqlite3
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_truth import ROOT, DATA_DIR

DB = ROOT / "db_v1_1" / "chroma.sqlite3"
PATTERNS = ["â€", "Ã©", "Ã¨", "Ã ", "Ã ", "Â", "�"]
QUELQU = "quelqu" + "â€™"


def index_texts() -> list[tuple[str, str, str]]:
    """[(source, chunk_id, text)]"""
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        """select c.string_value, d.string_value from embedding_metadata c
           join embedding_metadata d on d.id = c.id and d.key = 'chroma:document'
           where c.key = 'chunk_id'""").fetchall()
    con.close()
    return sorted((cid.split("/")[0], cid, t) for cid, t in rows)


def pypdf_texts() -> list[tuple[str, str, str]]:
    from pypdf import PdfReader
    return [(pdf.stem, f"{pdf.stem} p.{i + 1}", page.extract_text() or "")
            for pdf in sorted(DATA_DIR.glob("*.pdf")) for i, page in enumerate(PdfReader(pdf).pages)]


def pymupdf_texts() -> list[tuple[str, str, str]]:
    import fitz
    out = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            out += [(pdf.stem, f"{pdf.stem} p.{i + 1}", page.get_text()) for i, page in enumerate(doc)]
    return out


def around(text: str, i: int, width: int = 200) -> str:
    a = max(0, i - width // 2)
    return text[a:a + width]


def scan(name: str, texts: list[tuple[str, str, str]]) -> dict:
    hits = defaultdict(list)                     # source -> [(where, pattern list, first index)]
    per_pattern = defaultdict(lambda: defaultdict(int))
    for src, where, t in texts:
        found = [p for p in PATTERNS if p in t]
        for p in found:
            per_pattern[src][p] += 1
        if found:
            hits[src].append((where, found, min(t.index(p) for p in found), t))
    sources = sorted({s for s, _, _ in texts})
    total = sum(len(v) for v in hits.values())
    print(f"\n=== {name}: {len(texts)} texts, {total} contain at least one pattern")
    print(f"  {'source':<20}{'texts':>7}{'with any':>10}  per pattern (texts containing it)")
    n_by = defaultdict(int)
    for s, _, _ in texts:
        n_by[s] += 1
    for s in sources:
        pp = ", ".join(f"{p!r}:{n}" for p, n in per_pattern[s].items()) or "-"
        print(f"  {s:<20}{n_by[s]:>7}{len(hits[s]):>10}  {pp}")
    for s in sources:
        for where, found, i, t in hits[s][:5]:
            print(f"  [{where}] {found}\n      {around(t, i)!r}")
    return {s: {"texts": n_by[s], "with_any": len(hits[s]), "per_pattern": dict(per_pattern[s]),
                "examples": [{"where": w, "patterns": f, "context": around(t, i)} for w, f, i, t in hits[s][:5]]}
            for s in sources}


def find_quelqu(name: str, texts) -> int:
    n = 0
    for src, where, t in texts:
        start = 0
        while (i := t.find(QUELQU, start)) >= 0:
            n += 1
            print(f"  {name} [{where}] at char {i}:\n      {around(t, i)!r}")
            start = i + 1
    return n


def main():
    sets = [("index chunks (db_v1_1)", index_texts()), ("PDF extraction, pypdf", pypdf_texts()),
            ("PDF extraction, PyMuPDF", pymupdf_texts())]
    res = {name: scan(name, texts) for name, texts in sets}
    print(f"\n=== 'quelqu' followed by {'â€™'!r}")
    found = {name: find_quelqu(name, texts) for name, texts in sets}
    print("  occurrences: " + ", ".join(f"{k}: {v}" for k, v in found.items()))
    # "â€™" is what U+2019 (’) looks like when its UTF-8 bytes are read as cp1252
    print(f"\n=== control: 'quelqu\\u2019un' UTF-8 bytes decoded as cp1252 -> "
          f"{'quelqu’un'.encode('utf-8').decode('cp1252')!r}")
    print("=== characters right after 'quelqu' when it is not 'e' (index chunks, raw):")
    for src, where, t in sets[0][1]:
        start = 0
        while (i := t.find("quelqu", start)) >= 0:
            if i + 6 < len(t) and t[i + 6] != "e":
                print(f"  [{where}] U+{ord(t[i + 6]):04X} at char {i}:\n      {around(t, i)!r}")
            start = i + 1
    where = {name: sum(d["with_any"] for d in r.values()) for name, r in res.items()}
    print("\n=== texts with any pattern: " + ", ".join(f"{k}: {v}" for k, v in where.items()))


if __name__ == "__main__":
    main()
