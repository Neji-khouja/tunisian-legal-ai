"""What does opening the Chroma index (as pipeline.load_vectorstore does) change in its sqlite file? Runs on a temporary
COPY of db_v1_1: the real index is only read (sqlite read-only mode) to count its rows.

    python diagnostics/chroma_open_check.py    -> diagnostics/chroma_open_check.json
"""
import sys
import json
import shutil
import sqlite3
import hashlib
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "diagnostics" / "chroma_open_check.json"


def tables(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    out = {}
    for (t,) in con.execute("select name from sqlite_master where type='table' order by name"):
        h, n = hashlib.sha256(), 0
        for row in con.execute(f'select * from "{t}" order by 1'):
            h.update(repr(row).encode())
            n += 1
        out[t] = {"rows": n, "sha256": h.hexdigest()}
    con.close()
    return out


def main():
    sys.path.insert(0, str(ROOT))
    from config import DB_DIR
    real = Path(DB_DIR) / "chroma.sqlite3"
    real_tables = tables(real)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:     # Windows: Chroma may keep the copy's files open
        copy = Path(d) / "db"
        shutil.copytree(DB_DIR, copy)
        before = tables(copy / "chroma.sqlite3")
        from langchain_chroma import Chroma
        from pipeline import get_embeddings
        vs = Chroma(persist_directory=str(copy), embedding_function=get_embeddings())
        vs.similarity_search("vol simple", k=3)
        vs._collection.get(include=["documents", "metadatas"])
        del vs
        import gc
        gc.collect()
        after = tables(copy / "chroma.sqlite3")
        changed = {t: {"rows_before": before[t]["rows"], "rows_after": after.get(t, {}).get("rows")}
                   for t in before if before[t] != after.get(t)}
        res = {"copy_tables": len(before), "copy_tables_changed_by_one_open": changed,
               "copy_tables_unchanged": sum(before[t] == after.get(t) for t in before),
               "real_index_acquire_write_rows_now": real_tables.get("acquire_write", {}).get("rows"),
               "real_index_embeddings_rows": real_tables.get("embeddings", {}).get("rows")}
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
        print("temporary copy:", d)


if __name__ == "__main__":
    main()
