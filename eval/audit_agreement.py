"""Agreement between the LLM judge and a human, on the filled sheet and the key of eval/audit_export.py.

    python eval/audit_agreement.py [--sheet results/audit_sheet.csv] [--key results/audit_key.csv]
    python eval/audit_agreement.py --rater ai --sheet results/audit_claude.csv     # an AI second rater: ai_* columns
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
MEASURES = [("correct", "judge_correct", "correct"),
            ("hallucination (answer)", "hallucination_metric", "hallucination"),
            ("hallucination (LLM judge only)", "judge_hallucination", "hallucination")]


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
    only row_id and the human columns are used. Excel's From Text/CSV can add a row Column1, Column2, ... above the real
    header: the header is the first record holding row_id."""
    with Path(path).open(encoding="utf-8-sig", errors="replace", newline="") as f:
        head = f.readline()
        f.seek(0)
        records = csv.reader(f, delimiter=";" if head.count(";") > head.count(",") else ",")
        for names in records:
            if "row_id" in [c.strip() for c in names]:
                names = [c.strip() for c in names]
                return {r["row_id"].strip(): r for r in (dict(zip(names, x)) for x in records) if r.get("row_id", "").strip()}
    sys.exit(f"{path}: no column named row_id")


def measure_lines(name: str, pairs: list, rater: str = "human") -> list[str]:
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
            f"  confusion: judge 1 / {rater} 1: {n11}; judge 1 / {rater} 0: {n10}; judge 0 / {rater} 1: {n01}; "
            f"judge 0 / {rater} 0: {n00}",
            f"  pre-registered threshold >= {THRESHOLD:.0%}: {'MET' if agree >= THRESHOLD else 'NOT MET'} "
            f"(point estimate {agree:.1%}; 95% CI [{lo:.1%}, {hi:.1%}])" if rater == "human" else
            "  pre-registered threshold: not assessed, it applies to a human audit"]


def report(sheet_path, key_path, rater: str = "human") -> list[str]:
    sheet, key = read_rows(sheet_path), read_rows(key_path)
    if set(sheet) != set(key):
        sys.exit(f"the sheet and the key do not hold the same rows: only in the sheet {sorted(set(sheet) - set(key))}, "
                 f"only in the key {sorted(set(key) - set(sheet))}")
    out_cv = [i for i, r in sheet.items() if label(r.get(f"{rater}_cannot_verify")) == 1]
    used = {i: r for i, r in sheet.items() if i not in out_cv}
    L = [f"rows in the sheet: {len(sheet)}; left out as {rater}_cannot_verify: {len(out_cv)}; rows used: {len(used)}"]
    if rater != "human":
        L.insert(0, f"RATER: {rater.upper()}, NOT A HUMAN AUDIT")
    for name, jcol, what in MEASURES:
        hcol, pairs, blank, bad = f"{rater}_{what}", [], 0, []
        for i, r in used.items():
            raw, h = (r.get(hcol) or "").strip(), label(r.get(hcol))
            if h is None:
                blank += not raw
                if raw:
                    bad.append(i)
                continue
            pairs.append((int(key[i][jcol]), h))
        L += ["", *measure_lines(name, pairs, rater)]
        if blank or bad:
            L.append(f"  not counted: {blank} rows with {hcol} empty" + (f"; unreadable value in {sorted(bad)}" if bad else ""))
    L += ["", f"automatic checks (not part of the LLM judge), against {rater}_hallucination:"]
    for name, col, flagged in (("numeric check failed", "auto_numeric_ok", "0"),
                               ("cited article not in the evidence", "auto_cited_not_in_evidence", "1")):
        rows = [i for i in used if key[i][col].strip() != ""]
        hit = [i for i in rows if key[i][col].strip() == flagged]
        said = sum(label(used[i].get(f"{rater}_hallucination")) == 1 for i in hit)
        L.append(f"  {name}: {len(hit)} of {len(rows)} rows where the check applies; "
                 f"of those {len(hit)}, the {'human' if rater == 'human' else rater + ' rater'} marked hallucination on {said}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", default=str(ROOT / "results" / "audit_sheet.csv"))
    ap.add_argument("--key", default=str(ROOT / "results" / "audit_key.csv"))
    ap.add_argument("--rater", choices=["human", "ai"], default="human")
    args = ap.parse_args(argv)
    print("\n".join(report(args.sheet, args.key, args.rater)))


if __name__ == "__main__":
    main()
