"""Agreement between the LLM judge and a human, on the filled sheet and the key of eval/audit_export.py.

    python eval/audit_agreement.py [--sheet results/audit_sheet.csv] [--key results/audit_key.csv]

Rows with human_cannot_verify = 1 are left out of everything; a row with an empty human column is left out of that
measure. Human values: 1/0, yes/no, true/false, oui/non. Excel may save the sheet with ";" between columns: both are read.
For each measure: n, raw agreement, 95% bootstrap CI over rows (5000 resamples, seed 20261022), Cohen's kappa, the
confusion counts, and the pre-registered threshold (agreement >= 80%, on the point estimate). "correct" uses the judge's
verdict; "hallucination (answer)" the metric of --summarize (the judge field or an automatic check); "hallucination
(LLM judge only)" the judge's own field. The automatic checks are reported separately, against the human label.
"""
import sys
import csv
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_answer_eval as rae

YES, NO = {"1", "yes", "y", "true", "oui", "o"}, {"0", "no", "n", "false", "non"}
THRESHOLD = 0.80
MEASURES = [("correct", "judge_correct", "human_correct"),
            ("hallucination (answer)", "hallucination_metric", "human_hallucination"),
            ("hallucination (LLM judge only)", "judge_hallucination", "human_hallucination")]


def label(v: str):
    v = (v or "").strip().lower()
    return 1 if v in YES else 0 if v in NO else None


def kappa(pairs) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def read_rows(path) -> dict:
    """row_id -> row. Delimiter from the header; bytes that are not UTF-8 (a sheet saved as plain CSV) do not stop it:
    only row_id and the human columns are used."""
    with Path(path).open(encoding="utf-8-sig", errors="replace", newline="") as f:
        head = f.readline()
        f.seek(0)
        return {r["row_id"].strip(): r for r in csv.DictReader(f, delimiter=";" if head.count(";") > head.count(",") else ",")}


def measure_lines(name: str, pairs: list) -> list[str]:
    if not pairs:
        return [f"{name}: no row with both a judge value and a human label"]
    match = [int(j == h) for j, h in pairs]
    agree = sum(match) / len(match)
    lo, hi = rae.boot_ci(match, random.Random(rae.SEED))
    k = kappa(pairs)
    n11, n10 = sum(p == (1, 1) for p in pairs), sum(p == (1, 0) for p in pairs)
    n01, n00 = sum(p == (0, 1) for p in pairs), sum(p == (0, 0) for p in pairs)
    return [f"{name}: n={len(pairs)}  agreement {agree:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  Cohen's kappa "
            + (f"{k:.3f}" if k is not None else "undefined (no variation)"),
            f"  confusion: judge 1 / human 1: {n11}; judge 1 / human 0: {n10}; judge 0 / human 1: {n01}; "
            f"judge 0 / human 0: {n00}",
            f"  pre-registered threshold >= {THRESHOLD:.0%}: {'MET' if agree >= THRESHOLD else 'NOT MET'} "
            f"(point estimate {agree:.1%}; 95% CI [{lo:.1%}, {hi:.1%}])"]


def report(sheet_path, key_path) -> list[str]:
    sheet, key = read_rows(sheet_path), read_rows(key_path)
    if set(sheet) != set(key):
        sys.exit(f"the sheet and the key do not hold the same rows: only in the sheet {sorted(set(sheet) - set(key))}, "
                 f"only in the key {sorted(set(key) - set(sheet))}")
    out_cv = [i for i, r in sheet.items() if label(r.get("human_cannot_verify")) == 1]
    used = {i: r for i, r in sheet.items() if i not in out_cv}
    L = [f"rows in the sheet: {len(sheet)}; left out as human_cannot_verify: {len(out_cv)}; rows used: {len(used)}"]
    for name, jcol, hcol in MEASURES:
        pairs, blank, bad = [], 0, []
        for i, r in used.items():
            raw, h = (r.get(hcol) or "").strip(), label(r.get(hcol))
            if h is None:
                blank += not raw
                if raw:
                    bad.append(i)
                continue
            pairs.append((int(key[i][jcol]), h))
        L += ["", *measure_lines(name, pairs)]
        if blank or bad:
            L.append(f"  not counted: {blank} rows with {hcol} empty" + (f"; unreadable value in {sorted(bad)}" if bad else ""))
    L += ["", "automatic checks (not part of the LLM judge), against human_hallucination:"]
    for name, col, flagged in (("numeric check failed", "auto_numeric_ok", "0"),
                               ("cited article not in the evidence", "auto_cited_not_in_evidence", "1")):
        rows = [i for i in used if key[i][col].strip() != ""]
        hit = [i for i in rows if key[i][col].strip() == flagged]
        said = sum(label(used[i].get("human_hallucination")) == 1 for i in hit)
        L.append(f"  {name}: {len(hit)} of {len(rows)} rows where the check applies; "
                 f"of those {len(hit)}, the human marked hallucination on {said}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", default=str(ROOT / "results" / "audit_sheet.csv"))
    ap.add_argument("--key", default=str(ROOT / "results" / "audit_key.csv"))
    args = ap.parse_args(argv)
    print("\n".join(report(args.sheet, args.key)))


if __name__ == "__main__":
    main()
