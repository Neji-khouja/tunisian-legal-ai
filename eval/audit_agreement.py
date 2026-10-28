"""Agreement between the LLM judge and a human on the audit CSV written by run_answer_eval.py --export-audit.

    python eval/audit_agreement.py results/audit_sample_v2.csv

Rows where the human column is empty are skipped. Human values: 1/0, yes/no, true/false, oui/non.
"""
import sys
import csv

YES, NO = {"1", "yes", "y", "true", "oui", "o"}, {"0", "no", "n", "false", "non"}


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


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python eval/audit_agreement.py results/audit_sample_<system>.csv")
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
    tasks = {"correct": ("judge_verdict", "human_correct", lambda v: None if not v else int(v == "correct")),
             "hallucination": ("judge_hallucination", "human_hallucination", label)}
    for name, (jcol, hcol, jconv) in tasks.items():
        pairs = [(jconv(r[jcol]), label(r[hcol])) for r in rows]
        pairs = [(j, h) for j, h in pairs if j is not None and h is not None]
        if not pairs:
            print(f"{name}: no row with both a judge value and a human label")
            continue
        agree = sum(j == h for j, h in pairs) / len(pairs)
        k = kappa(pairs)
        print(f"{name}: n={len(pairs)}  agreement {agree:.1%}  Cohen's kappa {k:.3f}" if k is not None else
              f"{name}: n={len(pairs)}  agreement {agree:.1%}  Cohen's kappa undefined (no variation)")


if __name__ == "__main__":
    main()
