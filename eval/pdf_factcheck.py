"""Deterministic fact-check of answer claims against the PDF text (no LLM): every number in a claim must appear in
the PDF text of at least one article it cites.

    python eval/pdf_factcheck.py results/noise_check.json
"""
import re
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_truth import from_chunk_id, normalize

UNITS = {
    # French
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16, "vingt": 20, "vingts": 20,
    "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60,
    # English
    "two": 2, "three": 3, "four": 4, "five": 5, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
SCALES = {"cent": 100, "cents": 100, "hundred": 100, "mille": 1000, "thousand": 1000}
JOINERS = {"and"}
ART_REF = re.compile(r"\b(?:articles?|art\.)\s+(\d+)\b", re.I)               # "article 260" in a claim
IMPLICIT = re.compile(r"\barticles?\s+(?:(pr[ée]c[ée]dente?s?|ci-dessus)|(suivante?s?|ci-apr[èe]s))(?!\w)", re.I)
WORD = "|".join(sorted(list(UNITS) + list(SCALES), key=len, reverse=True))
WORD_RUN = re.compile(rf"\b(?:{WORD})(?:(?:[\s-]+(?:and[\s-]+)?)(?:{WORD}))*\b", re.I)
DIGITS = re.compile(r"\d{1,3}(?:[.  ]\d{3})+(?!\d)|\d+(?:,\d+)?")


def _words_value(run: str) -> int:
    total = current = 0
    for w in re.split(r"[\s-]+", run.lower()):
        if w in JOINERS or not w:
            continue
        if w in SCALES:
            if SCALES[w] == 1000:
                total += max(current, 1) * 1000
                current = 0
            else:
                current = max(current, 1) * 100
        else:
            current += UNITS[w]
    return total + current


def numbers_in(text: str) -> set:
    """Comparable numbers in a text: ints (decimals kept as strings like '1,5')."""
    t = normalize(text)
    out = set()
    for m in DIGITS.finditer(t):
        s = re.sub(r"[.  ]", "", m.group(0))
        out.add(s if "," in s else str(int(s)))
    for m in WORD_RUN.finditer(t):
        out.add(str(_words_value(m.group(0))))
    return out


def claim_numbers_ok(claim_text: str, cited_pdf_texts: list):
    """True if every number of the claim appears in at least one cited PDF text; None if no cited PDF text is
    available (unverifiable). Returns (ok, numbers, missing)."""
    nums = numbers_in(claim_text)
    texts = [t for t in cited_pdf_texts if t]
    if not texts:
        return None, nums, set()
    pdf_nums = set().union(*(numbers_in(t) for t in texts))
    missing = nums - pdf_nums
    return not missing, nums, missing


def implicit_reference_numbers(claim_text: str, missing: set, cited_ids: list, cited_texts: list) -> set:
    """The numbers of `missing` that the claim writes as "article N" and that a cited article designates implicitly
    ("article précédent" = its number - 1, "article suivant" = its number + 1)."""
    refs = {str(int(m.group(1))) for m in ART_REF.finditer(normalize(claim_text))}
    accepted = set()
    for cid, text in zip(cited_ids, cited_texts):
        if not text:
            continue
        label = cid.split("/")[-1].split("~")[0]
        base, plain = int(label.split("-")[0]), "-" not in label
        neighbours = {(base - 1 if plain else base) if m.group(1) else base + 1
                      for m in IMPLICIT.finditer(normalize(text))}
        accepted |= {n for n in missing & refs if int(n) in neighbours}
    return accepted


def check_claim(claim: dict):
    texts = [from_chunk_id(cid) for cid in claim.get("cited_ids", [])]
    ok, nums, missing = claim_numbers_ok(claim["text"], texts)
    return ok, nums, missing, texts


def claim_status(claim: dict):
    """(status, numbers, missing, implicit, texts); status: ok | implicit_reference_ok | missing | unverifiable |
    no_numbers. `implicit` = missing numbers accepted as implicit references (status implicit_reference_ok only if
    that covers every missing number)."""
    ok, nums, missing, texts = check_claim(claim)
    implicit = implicit_reference_numbers(claim["text"], missing, claim.get("cited_ids", []), texts) if ok is False else set()
    if ok is None:
        status = "unverifiable"
    elif not nums:
        status = "no_numbers"
    elif ok:
        status = "ok"
    else:
        status = "implicit_reference_ok" if implicit == missing else "missing"
    return status, nums, missing, implicit, texts


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python eval/pdf_factcheck.py results/noise_check.json")
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    totals = {"claims": 0, "with_numbers": 0, "numbers_ok": 0, "implicit_reference_ok": 0, "numbers_missing": 0,
              "unverifiable": 0, "no_numbers": 0}
    key = {"ok": "numbers_ok", "implicit_reference_ok": "implicit_reference_ok", "missing": "numbers_missing"}
    shown = {"ok": "ok", "implicit_reference_ok": "IMPLICIT REFERENCE OK", "missing": "MISSING",
             "unverifiable": "UNVERIFIABLE", "no_numbers": "ok"}
    for q in data.get("questions", []):
        for k, run in enumerate(q.get("runs", []), 1):
            for claim in run.get("claims") or []:
                status, nums, missing, implicit, texts = claim_status(claim)
                totals["claims"] += 1
                if status in key:
                    totals["with_numbers"] += 1
                    totals[key[status]] += 1
                else:
                    totals[status] += 1
                print(f"\n[{q['id']} run {k}] {shown[status]}  claim: {claim['text']}")
                print(f"    cites: {claim.get('cited_ids')}")
                print(f"    numbers in claim: {sorted(nums, key=lambda x: (len(x), x))}   missing from cited PDF text: "
                      f"{sorted(missing, key=lambda x: (len(x), x)) or '-'}"
                      + (f"   accepted as implicit reference: {sorted(implicit)}" if implicit else ""))
                for cid, t in zip(claim.get("cited_ids", []), texts):
                    print(f"    PDF {cid}: {normalize(t)[:600] if t else '(no PDF text: unverified source or ambiguous label)'}")
    print("\nTOTALS: " + ", ".join(f"{k} {v}" for k, v in totals.items()))
    print("  (numbers_ok / implicit_reference_ok / numbers_missing count claims that contain at least one number and whose cited PDF text is available)")


if __name__ == "__main__":
    main()
