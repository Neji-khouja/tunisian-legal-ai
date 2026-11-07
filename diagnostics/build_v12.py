"""Build the v2.1 index db_v1_2 (pipeline.build_index_v12) and record the build facts. Local only (no LLM call).

    python diagnostics/build_v12.py    -> db_v1_2/, diagnostics/index_v12_build.json

Resumable: entries already stored are skipped; the facts of every run are appended (the first run carries the full
embedding time). db_v1_1 is never opened for writing (pipeline._refuse_frozen).
"""
import sys
import json
import time
import sqlite3
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "diagnostics" / "index_v12_build.json"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    import config as C
    from pipeline import build_index_v12
    t0 = time.perf_counter()
    facts = build_index_v12()
    facts["seconds_total"] = round(time.perf_counter() - t0, 1)
    con = sqlite3.connect(f"file:{(C.DB_V12 / 'chroma.sqlite3').as_posix()}?mode=ro", uri=True)
    facts["rows_in_index"] = con.execute("select count(*) from embeddings").fetchone()[0]
    con.close()
    facts["index_bytes"] = dir_size(C.DB_V12)
    facts["db_v1_1_bytes"] = dir_size(C.DB_V11)
    facts["machine"] = f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}"
    import torch
    facts["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    facts["date"] = time.strftime("%Y-%m-%d %H:%M")
    runs = json.loads(OUT.read_text(encoding="utf-8"))["runs"] if OUT.exists() else []
    runs.append(facts)
    OUT.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
