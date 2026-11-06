"""Offline tests of eval/audit_export.py and eval/audit_agreement.py with stub rows (temp folders; no model, no network).

    python eval/test_audit.py

Hand-computed agreement, 10 rows used: judge 1 1 1 1 1 1 0 0 0 0, human 1 1 1 1 1 0 0 0 0 1 -> confusion 5, 1, 1, 3;
agreement 8/10 = 80% (threshold met exactly); judge rate .6, human rate .6, chance .36 + .16 = .52,
kappa (.8 - .52) / .48 = 0.583.
"""
import io
import csv
import sys
import json
import random
import tempfile
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import audit_export as ae
import audit_agreement as ag
import run_answer_eval as rae

RESULTS = []
VERDICT_MARK, UNSUPPORTED_MARK, MODEL_MARK = "VERDICTMARK", "UNSUPPORTEDMARK", "MODELMARK"


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   {detail}"))


def language(k: int) -> str:
    return "ar" if k < 6 else "en" if k < 13 else "fr"


def row(i: str, s: str, k: int, status="done", verdict="correct"):
    cited = [f"Code/v1/{k}"] if s == "v1" else [f"Code/s01/{k}", f"Code/s01/{k}~2"]
    return {"id": i, "system": s, "status": status, "expected": "answer", "question": f"question {k}", "text": f"answer {k}",
            "cited": None if s != "v2" else cited, "evidence_ids": cited, "hallucinated": k % 2 == 0,
            "provenance": {"commit": "COMMITMARK"}, "metrics": {"pdf_numeric_ok": True, "cited_not_in_evidence": None},
            "judge": {"verdict": verdict, "unsupported_statements": [UNSUPPORTED_MARK] if k % 2 == 0 else [],
                      "model": MODEL_MARK, "contradiction": False}}


def setup(tmp: Path, n=30) -> Path:
    items = [{"id": f"ac{k + 50}", "language": language(k), "reference_points": [f"ref {k} a", f"ref {k} b"]} for k in range(n)]
    items += [{"id": i, "language": "fr", "reference_points": ["ref 90 a", "ref 90 b"]} for i in ("ac02", "ac29", "c02")]
    exam = tmp / "exam.json"
    exam.write_text(json.dumps(items), encoding="utf-8")
    res = tmp / "res"
    res.mkdir()
    for s in ae.SYSTEMS:
        rows = [row(f"ac{k + 50}", s, k) for k in range(n)] + [row("ac02", s, 90), row("ac29", s, 91), row("c02", s, 92)]
        rows += [dict(row("ac50", s, 0), status="error")] if s == "v1" else []     # the latest row of ac50, v1 is an error
        rows += [dict(row("ac51", s, 1), status="degraded")] if s == "v1.1" else []
        rows += [dict(row("ac52", s, 2), status="failed")] if s == "v2" else []
        rae.out_path(rae.EXAM_DIR / "answer_set.json", s, "1", res).write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return exam


def sampling():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        res_exam = setup(tmp)
        pool = ae.candidates(tmp / "res", res_exam)
        pairs = {(p["id"], p["system"]) for p in pool}
        check("the pool skips the known fallback pairs", not pairs & ae.FALLBACK_PAIRS and ("ac02", "v1") in pairs
              and ("ac29", "v2") in pairs)
        check("the pool skips pairs whose latest row is error, degraded or failed",
              not {("ac50", "v1"), ("ac51", "v1.1"), ("ac52", "v2")} & pairs and ("ac50", "v2") in pairs)
        rows = {"no verdict": dict(row("ac60", "v1", 10), verdict=None)}
        rows["no verdict"]["judge"] = {}
        rae.out_path(rae.EXAM_DIR / "answer_set.json", "v1", "1", tmp / "res").open("a", encoding="utf-8").write(
            json.dumps(rows["no verdict"]) + "\n")
        check("the pool skips a row without a judge verdict",
              ("ac60", "v1") not in {(p["id"], p["system"]) for p in ae.candidates(tmp / "res", res_exam)})
        ok = True
        for seed in range(200):
            pick = ae.draw(pool, random.Random(seed))
            per_s = {s: sum(p["system"] == s for p in pick) for s in ae.SYSTEMS}
            per_q = {}
            for p in pick:
                per_q[p["id"]] = per_q.get(p["id"], 0) + 1
            pairs_ok = len({(p["id"], p["system"]) for p in pick}) == 30
            ok = ok and len(pick) == 30 and set(per_s.values()) == {10} and pairs_ok and max(per_q.values()) <= 2 \
                and sum(p["language"] == "ar" for p in pick) >= 5 and sum(p["language"] == "en" for p in pick) >= 5
        check("200 seeds: 30 pairs, 10 per system, none repeated, a question at most twice, >= 5 ar and >= 5 en", ok)
        a, b = ae.draw(pool, random.Random(rae.SEED)), ae.draw(pool, random.Random(rae.SEED))
        ids = [(p["id"], p["system"]) for p in a]
        check("the same seed gives the same sheet order", ids == [(p["id"], p["system"]) for p in b])
        check("the order is shuffled: the systems are not in blocks", [p["system"] for p in a] != sorted(p["system"] for p in a))
        few = [p for p in pool if p["language"] != "ar" or p["id"] in ("ac53", "ac54")]
        pick = ae.draw(few, random.Random(rae.SEED))
        check("only 2 Arabic questions in the pool: the minimum falls to what is available (4 pairs), the draw still works",
              ae.language_minimum(few, "ar", 5) == 4 and sum(p["language"] == "ar" for p in pick) >= 4)
        code = None
        try:
            ae.draw([p for p in pool if p["system"] != "v2"], random.Random(1))
        except SystemExit as e:
            code = e.code
        check("too few pairs for a system stops with a message", isinstance(code, str) and "not enough" in code, str(code))


def sheet_and_key():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        exam = setup(tmp)
        texts = {"Code/v1/": "PDF TEXT ", "Code/s01/": "PDF TEXT "}
        long_text = "x" * 9000                                          # longer than any old or judge cut (600, 2000)

        def fake_pdf(cid):
            art = cid.split("/")[-1].split("~")[0]
            if art == "3":
                return None
            return texts["Code/v1/" if "/v1/" in cid else "Code/s01/"] + art + " " + long_text
        rae.pdf_text, rae.index_text = fake_pdf, lambda cid: None
        sheet, key = tmp / "audit_sheet.csv", tmp / "audit_key.csv"
        args = ["--results-dir", str(tmp / "res"), "--exam", str(exam), "--sheet", str(sheet), "--key", str(key)]
        with contextlib.redirect_stdout(io.StringIO()):
            ae.main(args)
        raw = sheet.read_text(encoding="utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(raw, newline="")))
        check("sheet: exactly the agreed columns", list(rows[0]) == ["row_id", "language", "question", "answer", "cited_articles",
              "reference_points", "human_correct", "human_hallucination", "human_cannot_verify", "notes"], str(list(rows[0])))
        check("sheet: no column about the judge, the system or the item",
              not [c for c in rows[0] if any(w in c.lower() for w in ("judge", "system", "verdict", "item", "model", "auto"))])
        check("sheet: row ids r01..r30 in order, human columns empty",
              [r["row_id"] for r in rows] == [f"r{k:02d}" for k in range(1, 31)]
              and not any(r[c] for r in rows for c in ("human_correct", "human_hallucination", "human_cannot_verify", "notes")))
        leaks = {w: w in raw for w in (VERDICT_MARK, UNSUPPORTED_MARK, MODEL_MARK, "COMMITMARK", "/v1/", "/s01/", "v1.1", "ac5", "ac6", "ac7", "ac0", "ac2", "c02")}
        check("sheet contents: no judge text, no model, no commit, no chunk id (a v1 id names the system), no item id",
              not any(leaks.values()), str(leaks))
        check("sheet contents: no system name as a word", not any(f" {w} " in f" {raw} " for w in ("v1", "v2")))
        full = [r for r in rows if "PDF TEXT" in r["cited_articles"]]
        check("sheet: the article text is not truncated, and sits under a neutral label",
              full and all(long_text in r["cited_articles"] and r["cited_articles"].startswith("[Code, article ") for r in full))
        check("sheet: an article cut in chunks (~2) is shown once",
              all(r["cited_articles"].count("[Code, article ") == 1 for r in rows))
        no_text = [r for r in rows if ae.MISSING in r["cited_articles"]]
        with contextlib.redirect_stdout(io.StringIO()):
            pool = ae.candidates(tmp / "res", exam)
        p3 = next(p for p in pool if p["id"] == "ac53")
        check("sheet: an article without text is marked as missing and counted",
              ae.cited_block(p3["row"]) == ("[Code, article 3]\n" + ae.MISSING, 1) and all("article 3]" in r["cited_articles"] for r in no_text),
              str(ae.cited_block(p3["row"])))
        check("sheet: the reference points the judge gets are numbered",
              all(r["reference_points"].startswith("1. ref ") and "\n2. ref " in r["reference_points"] for r in rows))
        k = list(csv.DictReader(key.open(encoding="utf-8-sig", newline="")))
        check("key: the agreed columns, the same 30 row ids", list(k[0]) == ["row_id", "item_id", "system", "judge_correct",
              "judge_hallucination", "auto_numeric_ok", "auto_cited_not_in_evidence", "hallucination_metric"]
              and [r["row_id"] for r in k] == [r["row_id"] for r in rows])
        check("key: 10 rows per system", all(sum(r["system"] == s for r in k) == 10 for s in ae.SYSTEMS))
        by = {(p["id"], p["system"]): p for p in pool}
        same = all(r["question"] == by[(x["item_id"], x["system"])]["row"]["question"] for r, x in zip(rows, k))
        check("key: each row id points at the pair shown on that row of the sheet", same)
        one = {"id": "z", "system": "v2", "row": dict(row("z", "v2", 2), metrics={"pdf_numeric_ok": False, "cited_not_in_evidence": ["a"]},
                                                      hallucinated=True, judge={"verdict": "partial", "unsupported_statements": []})}
        check("key: LLM field and automatic checks are separate (judge 0, numeric failed, not in evidence, metric 1)",
              ae.key_row("r01", one) == {"row_id": "r01", "item_id": "z", "system": "v2", "judge_correct": 0, "judge_hallucination": 0,
                                         "auto_numeric_ok": 0, "auto_cited_not_in_evidence": 1, "hallucination_metric": 1})
        one["row"]["metrics"] = {"pdf_numeric_ok": None, "cited_not_in_evidence": None}
        kr = ae.key_row("r01", one)
        check("key: a check that does not apply stays empty", kr["auto_numeric_ok"] == "" and kr["auto_cited_not_in_evidence"] == "")
        check("sheet: text starting like an Excel formula gets one leading space, other text is unchanged",
              [ae.as_text(v) for v in ("- a", "=b", "+c", "@d", "e - f", "")] == [" - a", " =b", " +c", " @d", "e - f", ""])
        before = sheet.read_bytes()
        code = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ae.main(args)
        except SystemExit as e:
            code = e.code
        check("an existing sheet is not overwritten without --force", isinstance(code, str) and "REFUSED" in code
              and sheet.read_bytes() == before, str(code))


def write(path: Path, cols: list, rows: list, delimiter=","):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(cols)
        w.writerows(rows)


def agreement():
    judge = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    human = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ids = [f"r{k:02d}" for k in range(1, 14)]
        key_rows, sheet_rows = [], []
        for k, i in enumerate(ids):
            j = judge[k] if k < 10 else 1
            h = human[k] if k < 10 else 0
            llm = 0 if k == 0 else j                       # r01: the metric says 1 by an automatic check, the LLM says 0
            numeric, ev = (0, 1) if k == 0 else (1, 0) if k < 8 else ("", "")
            key_rows.append([i, "ac", "v2", j, llm, numeric, ev, j])
            cv = "1" if k == 10 else ""                    # r11: cannot verify, a disagreement that must not count
            hh = "" if k == 11 else "2" if k == 12 else h  # r12: left empty; r13: not a readable value
            sheet_rows.append([i, "q", "a", hh, hh, cv])
        kc = ["row_id", "item_id", "system", "judge_correct", "judge_hallucination", "auto_numeric_ok",
              "auto_cited_not_in_evidence", "hallucination_metric"]
        sc = ["row_id", "question", "answer", "human_correct", "human_hallucination", "human_cannot_verify"]
        write(tmp / "key.csv", kc, key_rows)
        write(tmp / "sheet.csv", sc, sheet_rows)
        lines = ag.report(tmp / "sheet.csv", tmp / "key.csv")
        text = "\n".join(lines)
        check("agreement: rows used leave out human_cannot_verify", "rows in the sheet: 13; left out as human_cannot_verify: 1; rows used: 12" in text, text)
        check("agreement, hand-computed: n=10, 80.0%, kappa 0.583", "correct: n=10  agreement 80.0%" in text and "Cohen's kappa 0.583" in text, text)
        check("agreement, hand-computed: confusion 5, 1, 1, 3",
              "judge 1 / human 1: 5; judge 1 / human 0: 1; judge 0 / human 1: 1; judge 0 / human 0: 3" in text, text)
        check("agreement: exactly 80% meets the threshold, and the CI is printed", "threshold >= 80%: MET (point estimate 80.0%; 95% CI [" in text, text)
        check("agreement: an empty value and an unreadable value are not counted, and are reported",
              "not counted: 1 rows with human_correct empty; unreadable value in ['r13']" in text, text)
        block = text.split("hallucination (LLM judge only): ")[1]
        check("LLM-only hallucination is its own line: r01 turns from agreement into a disagreement, n=10, 70.0%, NOT MET",
              block.startswith("n=10  agreement 70.0%") and "NOT MET (point estimate 70.0%" in block, block[:150])
        check("automatic checks are reported apart, counted against the human label",
              "numeric check failed: 1 of 8 rows where the check applies; of those 1, the human marked hallucination on 1" in text
              and "cited article not in the evidence: 1 of 8 rows where the check applies; of those 1, the human marked hallucination on 1" in text, text)
        lo, hi = rae.boot_ci([1] * 8 + [0] * 2, random.Random(rae.SEED))
        check("the CI is the harness bootstrap on the row agreements (seed 20261022)", f"95% CI [{lo:.1%}, {hi:.1%}]" in text, text)
        write(tmp / "semi.csv", sc, sheet_rows, delimiter=";")
        check("a sheet saved by Excel with ';' gives the same report", ag.report(tmp / "semi.csv", tmp / "key.csv") == lines)
        (tmp / "ansi.csv").write_bytes(";".join(sc).encode() + b"\r\n" + b"\r\n".join(
            ";".join(str(x) for x in r).encode() + b" \xe9" for r in sheet_rows))
        check("a sheet saved without UTF-8 still reads the human columns", "rows in the sheet: 13" in "\n".join(ag.report(tmp / "ansi.csv", tmp / "key.csv")))
        write(tmp / "short.csv", sc, sheet_rows[:-1])
        code = None
        try:
            ag.report(tmp / "short.csv", tmp / "key.csv")
        except SystemExit as e:
            code = e.code
        check("a sheet missing a row stops with a message", isinstance(code, str) and "r13" in code, str(code))
        write(tmp / "empty.csv", sc, [[r[0], "q", "a", "", "", ""] for r in sheet_rows])
        check("an unfilled sheet: no measure, no crash", "correct: no row with both" in "\n".join(ag.report(tmp / "empty.csv", tmp / "key.csv")))


def main():
    sampling()
    sheet_and_key()
    agreement()
    code = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rae.main(["--stage", "1", "--system", "v2", "--export-audit", "20"])
    except SystemExit as e:
        code = e.code
    check("the old --export-audit flag writes nothing and points to eval/audit_export.py",
          isinstance(code, str) and "audit_export.py" in code, str(code))
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed: {'all checks passed' if not n_fail else 'FAILURES'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
