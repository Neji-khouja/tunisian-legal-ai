"""Coverage discrepancies between the PDF article split (pypdf, eval/pdf_truth.py) and the index. No model, no LLM.

    python eval/coverage_discrepancies.py      -> prints, saves results/coverage_discrepancies.json

For each label found only in the PDF or only in the index (eval/audit_chunks.py, part 1):
  - PDF text (first 300 chars, with section) and index chunk text (first 300 chars) when present
  - where the text is on the other side: a 100-character window of the body (text after the first line,
    whitespace removed) is looked up in the other side's texts
  - for labels only in the index: is the chunk's first line (its heading) in the raw pypdf text?
Classes (rule applied by the script, printed with its evidence):
  (a) really missing from the index   only-PDF label whose body is in no index chunk
  (b) PDF-side regex artifact         only-index label whose heading line is in the pypdf text but was not
                                      split out by pdf_truth.HEADING
  (c) label mismatch/typo             the text is on both sides under different labels
Anything else is printed as "unclassified". Fixes nothing.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_truth import load, extract, normalize, DATA_DIR, ROOT
from audit_chunks import index_rows

OUT = ROOT / "results" / "coverage_discrepancies.json"
ONLY_PDF = {"COC": ["1273", "882"], "DIPFR": ["10"], "dtreelFr": ["335"], "societeFr": ["193", "194", "212", "221"],
            "travailFr": ["48"]}
ONLY_INDEX = {"CommerceFr": ["410-ter"], "Procedurecivilecom": ["192"],
              "travailFr": ["376-ter", "382-bis", "385-bis", "386-bis", "386-ter"], "tun_RecAcFreeLi": ["135", "27", "64"]}


def flat(t: str) -> str:
    return normalize(t).replace(" ", "")


def body_window(t: str) -> str:
    """100 whitespace-free characters of the body (after the first line, so the heading is not part of it)."""
    rest = t.split("\n", 1)[1] if "\n" in t else t
    return flat(rest)[:100]


def first_line(t: str) -> str:
    return t.strip().split("\n", 1)[0].strip()


def main():
    pdf = load()
    rows = index_rows()
    raw = {stem: extract(DATA_DIR / f"{stem}.pdf") for stem in set(ONLY_PDF) | set(ONLY_INDEX)}
    raw_flat = {stem: flat(t) for stem, t in raw.items()}
    out = []

    print("=== ONLY IN PDF")
    for stem, labels in ONLY_PDF.items():
        chunks = [r for r in rows if r["stem"] == stem]
        for label in labels:
            occ = pdf[stem]["articles"].get(label, [])
            print(f"\n--- {stem} {label}")
            rec = {"source": stem, "label": label, "side": "only_pdf", "pdf": [], "index": [], "found_in": []}
            for sec, text in occ:
                print(f"  PDF section {sec} ({len(text)} chars): {text[:300]!r}")
                rec["pdf"].append({"section": sec, "text300": text[:300]})
                win = body_window(text)
                hosts = [r["cid"] for r in chunks if win and win in flat(r["text"])]
                rec["found_in"] += hosts
                print(f"  body window found in index chunk(s): {hosts or 'none'}")
                for h in hosts:
                    t = next(r["text"] for r in chunks if r["cid"] == h)
                    i = flat(t).find(win)
                    print(f"    {h} ({len(t)} chars) first 300: {t[:300]!r}\n"
                          f"    window starts at whitespace-free char {i} of that chunk")
            idx = [r for r in chunks if r["label"] == label]
            print(f"  index chunks with this label: {[r['cid'] for r in idx] or 'none'}")
            if not rec["found_in"]:
                cls = "a"
            elif all(h.split("/")[-1].split("~")[0] != label for h in rec["found_in"]):
                cls = "c"
            else:
                cls = "unclassified"
            rec["class"] = cls
            before = {str(int(label) - k) for k in (1, 2)}
            if cls == "c" and all(h.split("/")[-1].split("~")[0] in before for h in rec["found_in"]):
                rec["note"] = "text is inside the index chunk of a preceding article (no chunk of its own)"
            print(f"  CLASS ({cls})" + (f": {rec['note']}" if rec.get("note") else ""))
            out.append(rec)

    print("\n=== ONLY IN INDEX")
    for stem, labels in ONLY_INDEX.items():
        for label in labels:
            idx = [r for r in rows if r["stem"] == stem and r["label"] == label]
            print(f"\n--- {stem} {label}")
            rec = {"source": stem, "label": label, "side": "only_index", "pdf": [], "index": [], "found_in": []}
            heading_in_pdf = False
            for r in idx:
                t = r["text"]
                print(f"  index chunk {r['cid']} ({len(t)} chars): {t[:300]!r}")
                rec["index"].append({"cid": r["cid"], "text300": t[:300]})
                head = first_line(t)
                hf = flat(head)
                pos = raw_flat[stem].find(hf) if hf else -1
                heading_in_pdf |= pos >= 0
                print(f"  first line {head!r} in raw pypdf text: {pos >= 0}")
                if pos >= 0:
                    # the raw line of the pypdf text that holds this heading
                    for line in raw[stem].splitlines():
                        if flat(line).startswith(hf[:12]) and hf[:12]:
                            print(f"    raw pypdf line: {line!r}")
                            break
                win = body_window(t)
                hosts = [f"{lab} (section {sec})" for lab, occ in pdf[stem]["articles"].items()
                         for sec, ptxt in occ if win and win in flat(ptxt)]
                rec["found_in"] += hosts
                print(f"  body window found in PDF article(s): {hosts or 'none'}")
            rec["heading_line_in_pdf"] = heading_in_pdf
            if heading_in_pdf and rec["found_in"]:
                cls = "b"
            elif rec["found_in"]:
                cls = "c"
            else:
                cls = "unclassified"
            rec["class"] = cls
            print(f"  CLASS ({cls})")
            out.append(rec)

    print("\n=== SUMMARY")
    for rec in out:
        print(f"  {rec['side']:<11}{rec['source']:<20}{rec['label']:<10}({rec['class']})  found in: {rec['found_in'] or '-'}"
              + (f"  [{rec['note']}]" if rec.get("note") else ""))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
