"""PDF ground truth, independent of the indexing path.

The index is built with PyMuPDF (loaders.py) + preprocessing.clean_text + splitter.py. Here the text is
extracted with pypdf (a different extractor) and split with a simple regex of our own, so the PDFs can be
compared with our chunks instead of trusting them.

    python eval/pdf_truth.py            # (re)build eval/pdf_articles.json and print the quality report

Cache: eval/pdf_articles.json (derived, not committed)
    {source_stem: {"status": "verified"|"unverified", "reasons": [...], "quality": {...},
                   "articles": {label: [[section, text], ...]}}}
A label can occur several times in a PDF that bundles several texts (numbering restarts at 1); `section`
counts those restarts (0 = before the first restart), like the s<NN> part of a chunk id.
"""
import re
import sys
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE = ROOT / "eval" / "pdf_articles.json"

# the corpus also spells some suffixes "quarter", "quinter", "sexties", "sexis", "septiès"; two can follow ("ter bis")
SUFFIXES = r"bis|ter|quater|quarter|quinquies|quinquiès|quinter|sexies|sexties|sexis|septies|septiès|octies|nonies|decies"
HEADING = re.compile(
    rf"^[ \t]*(?:Article|ARTICLE|Art\.|ART\.)[ \t]*(premier|Premier|PREMIER|1er|\d+)"
    rf"(?:[ \t]*[-–][ \t]*(\d+)(?!\d))?"
    rf"(?:[ \t]*[\"«(]?[ \t]*((?:{SUFFIXES})(?:[ \t]+(?:{SUFFIXES}))?)\b[ \t]*[\"»)]?)?"
    rf"(?=[ \t]*(?:[.(\-–—:]|$))",
    re.M | re.I,
)
FURNITURE = [
    re.compile(r"^\s*Imp\w*rie\s+Officielle\s+de\s+la\s+R[ée]publique\s+Tu\w*\s*$", re.I),   # also "Impeimerie"
    re.compile(r"^\s*\d{1,4}\s*$"),                                   # page number alone on a line
    re.compile(r"^\s*Page\s+\d+\s+sur\s+\d+\s*$", re.I),
]
ARABIC = re.compile(r"[؀-ۿ]")
GLYPH_NAMES = re.compile(r"(?:/tatweel)+")      # pypdf output for Arabic glyphs it cannot map (garbled text)
GARBLED = {}                                    # stem -> lines where pypdf produced such glyph names


def normalize(text: str) -> str:
    """Collapse whitespace, unify apostrophes / quotes / hyphens, keep accents."""
    t = unicodedata.normalize("NFC", text or "")
    t = t.replace("­", "")                                        # soft hyphen
    t = re.sub(r"[‘’‛´`ʼ]", "'", t)
    t = re.sub(r"[“”«»„]", '"', t)
    t = re.sub(r"[‐‑‒–—―−]", "-", t)
    return " ".join(t.split())


def _label(num: str, sub: str | None, suf: str | None) -> str:
    base = "1" if not num.isdigit() else str(int(num))
    suf = "-".join(suf.lower().replace("è", "e").split()) if suf else None
    return base + (f"-{sub}" if sub else "") + (f"-{suf}" if suf else "")


def extract(pdf: Path) -> str:
    from pypdf import PdfReader
    lines = []
    GARBLED[pdf.stem] = 0
    for page in PdfReader(pdf).pages:
        for line in (page.extract_text() or "").splitlines():
            if not any(p.match(line) for p in FURNITURE):
                GARBLED[pdf.stem] += bool(GLYPH_NAMES.search(line))
                lines.append(GLYPH_NAMES.sub(" ", line))
    return "\n".join(lines)


def split_articles(text: str) -> dict:
    """{label: [[section, text], ...]}; a new section starts when the numbering restarts at 1 (a plain label 1
    after a higher base number). A label repeated inside a section stays as two occurrences (ambiguous)."""
    heads = list(HEADING.finditer(text))
    out, section, prev_base = {}, 0, None
    for i, m in enumerate(heads):
        label = _label(m.group(1), m.group(2), m.group(3))
        base = int(label.split("-")[0])
        plain = "-" not in label
        if prev_base is not None and plain and base == 1 and prev_base > 1:
            section += 1
        prev_base = base
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.setdefault(label, []).append([section, text[m.start():end].strip()])
    return out


def quality(text: str, articles: dict) -> dict:
    letters = sum(c.isalpha() for c in text) or 1
    lines = [l for l in text.splitlines() if l.strip()]

    def broken(l):
        s = l.strip()
        toks = s.split()
        if "�" in s:
            return True
        if len(s) >= 20 and sum(c.isalpha() for c in s) / len(s) < 0.5:
            return True
        return len(toks) >= 6 and sum(len(t) == 1 for t in toks) / len(toks) > 0.5    # "l e t t r e s"
    return {
        "chars": len(text),
        "articles": sum(len(v) for v in articles.values()),
        "distinct_labels": len(articles),
        "arabic_letter_ratio": round(len(ARABIC.findall(text)) / letters, 4),
        "has_article_headings": bool(re.search(r"^\s*(Article|Art\.)\s", text, re.M | re.I)),
        "has_arabic_article_headings": "المادة" in text or "الفصل" in text,
        "broken_line_share": round(sum(map(broken, lines)) / (len(lines) or 1), 4),
    }


def build() -> dict:
    data = {}
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        text = extract(pdf)
        arts = split_articles(text)
        q = quality(text, arts)
        q["garbled_arabic_lines"] = GARBLED[pdf.stem]
        reasons = []
        if q["articles"] == 0:
            reasons.append("no article heading found")
        if q["broken_line_share"] > 0.10:
            reasons.append(f"broken lines {q['broken_line_share']:.1%} > 10%")
        if q["arabic_letter_ratio"] > 0.20:
            reasons.append(f"Arabic letters {q['arabic_letter_ratio']:.1%} of letters; Arabic extraction not trusted")
        data[pdf.stem] = {"status": "unverified" if reasons else "verified", "reasons": reasons,
                          "quality": q, "articles": arts}
    CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


_DATA = None


def load() -> dict:
    global _DATA
    if _DATA is None:
        _DATA = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else build()
    return _DATA


def source_stem(source: str) -> str:
    return Path(str(source)).stem


def is_verified(source: str) -> bool:
    d = load().get(source_stem(source))
    return bool(d) and d["status"] == "verified"


def pdf_article_text(source: str, label: str, section: int | None = None) -> str | None:
    """PDF text of one article, or None if the source is unverified, the label is absent, or it is ambiguous
    (several occurrences and no `section` that picks exactly one)."""
    d = load().get(source_stem(source))
    if not d or d["status"] != "verified":
        return None
    occ = d["articles"].get(str(label), [])
    if section is not None:
        occ = [o for o in occ if o[0] == section]
    return occ[0][1] if len(occ) == 1 else None


def from_chunk_id(cid: str) -> str | None:
    """'PenalFr/s02/220-bis' -> PDF text of PenalFr article 220-bis in section 2 (a '~k' duplicate suffix is ignored)."""
    parts = cid.split("/")
    if len(parts) != 3:
        return None
    stem, sec, label = parts
    sec_num = int(sec[1:]) if sec[1:].isdigit() else None
    return pdf_article_text(stem, label.split("~")[0], sec_num)


if __name__ == "__main__":
    data = build()
    print("Extractor: pypdf (index uses PyMuPDF); cache written to eval/pdf_articles.json\n")
    print(f"{'source':<22}{'status':<12}{'chars':>9}{'articles':>9}{'labels':>8}{'arabic':>8}{'broken':>8}  headings(Art/المادة)  garbled-AR lines")
    for stem, d in data.items():
        q = d["quality"]
        print(f"{stem:<22}{d['status']:<12}{q['chars']:>9}{q['articles']:>9}{q['distinct_labels']:>8}"
              f"{q['arabic_letter_ratio']:>8.1%}{q['broken_line_share']:>8.1%}  {q['has_article_headings']}/{q['has_arabic_article_headings']}{q['garbled_arabic_lines']:>10}"
              + (f"   reasons: {'; '.join(d['reasons'])}" if d["reasons"] else ""))
    sys.exit(0)
