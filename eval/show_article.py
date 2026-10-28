"""Print an article as it is in the PDF (pypdf, eval/pdf_truth.py) and as it is in the index (sqlite, no model).

    python eval/show_article.py PenalFr 218
    python eval/show_article.py PenalFr.pdf 220-bis
"""
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_truth import load, source_stem, ROOT

DB = ROOT / "db_v1_1" / "chroma.sqlite3"


def index_chunks(stem: str, label: str) -> list[tuple[str, str]]:
    """[(chunk_id, text)] for every chunk of this source with this article label (any section, any ~k)."""
    if not DB.exists():
        return []
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        """select c.string_value, d.string_value from embedding_metadata c
           join embedding_metadata d on d.id = c.id and d.key = 'chroma:document'
           where c.key = 'chunk_id' and c.string_value like ?""", (f"{stem}/%",)).fetchall()
    con.close()
    return sorted((cid, t) for cid, t in rows if cid.split("/")[-1].split("~")[0] == label)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python eval/show_article.py <source> <article label>   e.g. PenalFr 218")
    stem, label = source_stem(sys.argv[1]), sys.argv[2]
    d = load().get(stem)
    if d is None:
        sys.exit(f"unknown source {stem!r}; known: {', '.join(load())}")
    print(f"=== {stem} article {label}   (PDF status: {d['status']})")
    occ = d["articles"].get(label, [])
    if d["status"] != "verified":
        print(f"--- PDF: source unverified ({'; '.join(d['reasons'])}); text not shown as ground truth")
    elif not occ:
        print("--- PDF: no article with this label found")
    for sec, text in occ if d["status"] == "verified" else []:
        print(f"\n--- PDF (pypdf), section {sec}, {len(text)} chars:\n{text}")
    chunks = index_chunks(stem, label)
    if not chunks:
        print("\n--- index: no chunk with this label")
    for cid, text in chunks:
        print(f"\n--- index chunk {cid}, {len(text)} chars:\n{text}")


if __name__ == "__main__":
    main()
