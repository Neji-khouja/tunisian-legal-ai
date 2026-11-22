"""Blind human audit of the judge: a mixed-system sheet to fill in and a separate key. No LLM call, no model.

    python eval/audit_export.py [--force]            # then fill results/audit_sheet.csv
    python eval/audit_agreement.py                   # when the sheet is finished
"""
import sys
import csv
import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_answer_eval as rae

SYSTEMS = ["v1", "v1.1", "v2"]
FALLBACK_PAIRS = {("ac02", "v2"), ("ac29", "v1"), ("c02", "v1")}
PER_SYSTEM = 10
MAX_PER_QUESTION = 2
MIN_LANGUAGE = {"ar": 5, "en": 5}
MAX_DRAWS = 200000
MISSING = "(article text missing)"
SHEET_COLUMNS = ["row_id", "language", "question", "answer", "cited_articles", "reference_points",
                 "human_correct", "human_hallucination", "human_cannot_verify", "notes"]
KEY_COLUMNS = ["row_id", "item_id", "system", "judge_correct", "judge_hallucination", "auto_numeric_ok",
               "auto_cited_not_in_evidence", "hallucination_metric"]


def candidates(results_dir: Path, exam_path: Path) -> list[dict]:
    """Every usable (question, system) pair of Stage 1, as {"id", "system", "language", "row", "item"}."""
    items = {i["id"]: i for i in json.loads(Path(exam_path).read_text(encoding="utf-8"))}
    out = []
    for s in SYSTEMS:
        rows = {r["id"]: r for r in rae.read_jsonl(rae.out_path(rae.EXAM_DIR / "answer_set.json", s, "1", results_dir))}
        for i, r in rows.items():
            if (r.get("expected") == "answer" and r.get("status") not in rae.NO_ANSWER and (i, s) not in FALLBACK_PAIRS
                    and (r.get("judge") or {}).get("verdict") and r.get("hallucinated") is not None):
                out.append({"id": i, "system": s, "language": items[i]["language"], "row": r, "item": items[i]})
    return out


def language_minimum(pool: list[dict], language: str, wanted: int) -> int:
    """The minimum that can be asked of the pool: a question counts at most MAX_PER_QUESTION times."""
    per_q = {}
    for p in pool:
        if p["language"] == language:
            per_q[p["id"]] = per_q.get(p["id"], 0) + 1
    return min(wanted, sum(min(n, MAX_PER_QUESTION) for n in per_q.values()))


def meets_constraints(pick: list[dict], minimum: dict) -> bool:
    per_q, per_lang = {}, {}
    for p in pick:
        per_q[p["id"]] = per_q.get(p["id"], 0) + 1
        per_lang[p["language"]] = per_lang.get(p["language"], 0) + 1
    return max(per_q.values()) <= MAX_PER_QUESTION and all(per_lang.get(k, 0) >= v for k, v in minimum.items())


def draw(pool: list[dict], rng: random.Random) -> list[dict]:
    by_system = {s: [p for p in pool if p["system"] == s] for s in SYSTEMS}
    short = {s: len(v) for s, v in by_system.items() if len(v) < PER_SYSTEM}
    if short:
        sys.exit(f"not enough usable pairs per system (need {PER_SYSTEM}): {short}")
    minimum = {k: language_minimum(pool, k, v) for k, v in MIN_LANGUAGE.items()}
    for _ in range(MAX_DRAWS):
        pick = [p for s in SYSTEMS for p in rng.sample(by_system[s], PER_SYSTEM)]
        if meets_constraints(pick, minimum):
            rng.shuffle(pick)
            return pick
    sys.exit(f"no draw met the constraints in {MAX_DRAWS} tries (language minimums {minimum})")


def article_text(cid: str) -> str | None:
    """The PDF text, else the index chunk (as the judge falls back), else None. Never truncated."""
    t = rae.pdf_text(cid) or rae.index_text(cid)
    return rae.normalize(t) if t else None


def cited_block(row: dict) -> tuple[str, int]:
    """-> (text of the cited articles under neutral labels, number of articles without text)."""
    cited = row["cited"] if row.get("cited") is not None else row.get("evidence_ids") or []
    seen, parts, missing = set(), [], 0
    for cid in cited:
        label = "{}, article {}".format(*rae.key(cid))
        if label in seen:
            continue
        seen.add(label)
        text = article_text(cid)
        missing += text is None
        parts.append(f"[{label}]\n{text or MISSING}")
    return "\n\n".join(parts), missing


def as_text(t: str) -> str:
    return " " + t if t[:1] in ("=", "+", "-", "@") else t


def sheet_row(row_id: str, p: dict) -> dict:
    r, refs = p["row"], p["item"].get("reference_points") or []
    return {"row_id": row_id, "language": p["language"], "question": as_text(r["question"]), "answer": as_text(r.get("text") or ""),
            "cited_articles": cited_block(r)[0], "reference_points": "\n".join(f"{i}. {x}" for i, x in enumerate(refs, 1)),
            "human_correct": "", "human_hallucination": "", "human_cannot_verify": "", "notes": ""}


def key_row(row_id: str, p: dict) -> dict:
    r, m = p["row"], p["row"]["metrics"]
    numeric, not_in_ev = m.get("pdf_numeric_ok"), m.get("cited_not_in_evidence")
    return {"row_id": row_id, "item_id": p["id"], "system": p["system"], "judge_correct": int(r["judge"]["verdict"] == "correct"),
            "judge_hallucination": int(bool(r["judge"]["unsupported_statements"])),
            "auto_numeric_ok": "" if numeric is None else int(numeric),
            "auto_cited_not_in_evidence": "" if not_in_ev is None else int(bool(not_in_ev)),
            "hallucination_metric": int(r["hallucinated"])}


def write_csv(path: Path, columns: list[str], rows: list[dict]):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:     # the BOM keeps Arabic readable in Excel
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def export(results_dir: Path, exam_path: Path, sheet: Path, key: Path) -> list[dict]:
    pick = draw(candidates(results_dir, exam_path), random.Random(rae.SEED))
    row_ids = [f"r{k:02d}" for k in range(1, len(pick) + 1)]
    write_csv(sheet, SHEET_COLUMNS, [sheet_row(i, p) for i, p in zip(row_ids, pick)])
    write_csv(key, KEY_COLUMNS, [key_row(i, p) for i, p in zip(row_ids, pick)])
    return pick


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--exam", default=str(rae.EXAM_DIR / "answer_set.json"))
    ap.add_argument("--sheet", default=str(ROOT / "results" / "audit_sheet.csv"))
    ap.add_argument("--key", default=str(ROOT / "results" / "audit_key.csv"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet (it may hold human labels)")
    args = ap.parse_args(argv)
    if Path(args.sheet).exists() and not args.force:
        sys.exit(f"REFUSED: {args.sheet} exists and may hold human labels. Pass --force to overwrite it.")
    pick = export(Path(args.results_dir), Path(args.exam), Path(args.sheet), Path(args.key))
    print(f"wrote {args.sheet} and {args.key} ({len(pick)} rows). Do not open the key before the sheet is finished.")


if __name__ == "__main__":
    main()
