"""Article-level chunking with stable chunk IDs."""
import re
import logging
import unicodedata
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from preprocessing import clean_text

logger = logging.getLogger(__name__)

# Latin ordinal suffixes as they occur in the corpus, including misspellings ("quinquiès", "sexties",
# "sexis", "Septies", "« quinter »") and doubled ones ("ter bis")
_ONE_SUFFIX = (r"bis|ter|quater|quarter|quinter|quinqui[eèé]s|sext?[iy]?[eè]?s|sexies|sexis|sexties|sept[iy]?[eè]?s|septies|"
               r"oct[iy]?[eè]?s|octies|non[iy]?[eè]?s|nonies|dec[iy]?[eè]?s|decies|undecies|duodecies")
_SUFFIXES = rf"(?:{_ONE_SUFFIX})(?:[ 	]+(?:{_ONE_SUFFIX}))?"

_HEADING_RE = re.compile(
    rf"""(?P<kw>Article|ARTICLE|Art\.?|ART\.?)[ 	]*\.?[ 	]*
         (?P<num>\d+|[Pp]remier|PREMIER|1er)
         (?:[ 	]*[-–][ 	]*(?P<sub>\d+))?
         (?:[ 	]*[-–.]?[ 	]*[«"(]?[ 	]*(?P<suf>(?i:{_SUFFIXES}))[ 	]*[»")]?)?
         (?![\w])""",
    re.X,
)
# at line start, a heading that is really a reference ("Article 5 du code ...") is not a heading
_REFERENCE_AFTER = re.compile(r"[ \t]+(?:du|de|des|dudit|ci|et|à|au|aux|précit|susvis|ainsi|sus)\b")
# mid-line headings are only accepted when followed by a dash separator
_SEPARATOR_AFTER = re.compile(r"[ \t]*\.?[ \t]*[-–—]")

# first line of the preamble of a new bundled text (searched backwards from a numbering reset)
_PREAMBLE_RE = re.compile(
    r"^[ \t(]*\d{0,2}[ \t)]*(?:Loi|LOI|Décret|DECRET|Décret-loi|Arrêté|ARRETE|Arrêté|Code|CODE|Convention|CONVENTION|"
    r"Accord|ACCORD|Au nom du peuple|Nous|Le Président de la République|La Chambre|La chambre|"
    r"Vu|Texte|TEXTE|Projet|Annexe|ANNEXE)\b",
    re.M,
)
_PREAMBLE_WINDOW = 2500


def _label(m: re.Match) -> tuple[str, str]:
    """-> (base number, label) e.g. ("226", "226-ter"), ("6", "6-2"), ("1", "1")."""
    num = m.group("num")
    base = "1" if not num.isdigit() else str(int(num))
    label = base
    if m.group("sub"):
        label += "-" + m.group("sub")
    if m.group("suf"):
        suf = unicodedata.normalize("NFKD", m.group("suf").lower())
        suf = "".join(ch for ch in suf if not unicodedata.combining(ch))
        label += "-" + re.sub(r"\s+", "-", suf.strip())
    return base, label


def _find_headings(text: str) -> list[dict]:
    found = []
    for m in _HEADING_RE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        at_line_start = not text[line_start:m.start()].strip()
        after = text[m.end():m.end() + 40]
        if at_line_start:
            if _REFERENCE_AFTER.match(after):
                continue
        else:
            prev_word = re.search(r"(\S+)\s*$", text[line_start:m.start()])
            ok_prev = bool(prev_word) and (prev_word.group(1)[0].isupper() or prev_word.group(1)[-1] in ".:;)")
            if not (ok_prev and _SEPARATOR_AFTER.match(after)):
                continue
        base, label = _label(m)
        found.append({"start": m.start(), "end": m.end(), "base": int(base), "base_s": base, "label": label})
    return found


def _split_sections(headings: list[dict]) -> None:
    """Annotate headings with a section number. A new text starts when the numbering restarts:
    a plain "Article 1/2" after a higher number, or any plain label already used in this section."""
    section, prev, seen = 0, None, set()
    for h in headings:
        plain = "-" not in h["label"]
        if prev is not None and plain:
            restart = (h["label"] in seen) or (h["base"] <= 2 and (
                prev["base"] >= 3 or (h["base"] == 1 and prev["base"] >= 2)))
            if restart:
                section += 1
                seen = set()
        seen.add(h["label"])
        h["section"] = section
        prev = h


_TITLE_RE = re.compile(
    r"(?:Loi|LOI|Décret-loi|Décret|DECRET|Arrêté|ARRETE|Convention|CONVENTION|Accord|CODE|Code)\b[^\n]{0,25}?"
    r"(?:n°|du\b|des\b|de\b|DE\b|ENTRE\b)[^\n]{0,150}"
)


def _clip_preamble(body: str) -> tuple[str, str]:
    """Remove the preamble of the next bundled text from the end of an article body.
    -> (clean body, title of the next text)."""
    window_start = max(0, len(body) - _PREAMBLE_WINDOW)
    m = _PREAMBLE_RE.search(body, window_start)
    if not m:
        return body, ""
    cut = m.start()
    preamble = body[cut:cut + 900]
    t = _TITLE_RE.search(preamble)
    title = re.sub(r"\s+", " ", t.group(0) if t else preamble[:300]).strip()
    return body[:cut].rstrip(), title[:200]


# ── structural headings (v2.1: SPLIT_HEADINGS_FORWARD) ──────────────────────────────────────────────────────
# "LIVRE PREMIER", "TITRE III", "Chapitre II", "Section Première - Absence de criminalité", "Sous-section 2",
# "Sous titre premier", "Titre Deuxième". A keyword line is only a heading when the ordinal ends the line or is
# followed by a separator and a title: "Section 2 du présent chapitre ..." (a sentence) is not.
_HEADING_LEVELS = {"livre": 1, "partie": 1, "titre": 2, "sous-titre": 3, "chapitre": 4, "section": 5,
                   "sous-section": 6, "paragraphe": 7,
                   "الكتاب": 1, "الجزء": 1, "العنوان": 2, "الباب": 4, "القسم": 5}
_ORDINAL = (r"(?:[IVXLC]+|\d+|premi[eè]re?|1[eè]re?|1er|unique|pr[eé]liminaire|second(?:e)?|deux|trois|quatre|cinq|"
            r"six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingt|"
            r"[a-zé]+i[eè]me)(?:\s*\(\d+\))?(?:\s+(?:bis|ter|quater))?")
_HEADING_LINE_RE = re.compile(
    rf"^[ \t]*(?P<kw>(?i:livre|partie|titre|sous[- ]titre|chapitre|section|sous[- ]section|paragraphe)|"
    rf"الكتاب|الجزء|العنوان|الباب|القسم)[ \t]+(?P<ord>(?i:{_ORDINAL})|[٠-٩]+|[ء-ي]+)"
    rf"[ \t]*(?:(?P<sep>[-–—:.])[ \t]*(?P<title>.*?)|\((?P<paren>[^)]*)\)?)?[ \t]*$"
)
_DOT_LEADER = re.compile(r"\.{4,}|…{2,}|(?:\.\s){4,}")
_TITLE_LINE_MAX = 160


def _heading_line(line: str):
    m = _HEADING_LINE_RE.match(line)
    if not m or _DOT_LEADER.search(line):
        return None
    kw = re.sub(r"[ \t]+", "-", m.group("kw").lower())
    return {"level": _HEADING_LEVELS[kw], "line": " ".join(line.split()), "inline": bool((m.group("title") or "").strip())}


def _title_line(line: str) -> bool:
    """A line that can be the title under a heading keyword line ("DES PEINES ET DE LEUR EXECUTION")."""
    s = line.strip()
    return bool(s) and len(s) <= _TITLE_LINE_MAX and not _DOT_LEADER.search(s) and s[-1] not in ".;," \
        and not re.match(r"[\d«\"\-–•*]", s) and not _HEADING_RE.match(s)


def _parse_heading_block(block: str) -> list[dict]:
    """Heading lines of a block, each with its title lines joined: [{"level", "title"}] in order."""
    out, cur, extra = [], None, 0
    for line in block.split("\n"):
        h = _heading_line(line)
        if h:
            cur, extra = {"level": h["level"], "title": h["line"]}, 0
            out.append(cur)
        elif cur is not None and _title_line(line) and extra < 3:
            cur["title"] += (" - " if extra == 0 else " ") + " ".join(line.split())
            extra += 1
        elif line.strip():
            cur = None
    for h in out:
        h["title"] = h["title"][:200]
    return out


def trailing_heading_start(body: str, max_lines: int = 16) -> int | None:
    """Offset in body where a trailing block of headings starts (the headings of the NEXT article), else None.
    The block starts at a heading keyword line and every non-empty line after it is a heading line or a title line
    (at most 3 under each heading)."""
    lines = body.split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    for k in nonempty[-max_lines:]:
        if not _heading_line(lines[k]):
            continue
        ok, extra = True, 0
        for ln in lines[k + 1:]:
            if not ln.strip():
                continue
            if _heading_line(ln):
                extra = 0
            elif _title_line(ln) and extra < 3:
                extra += 1
            else:
                ok = False
                break
        if ok:
            return offs[k]
    return None


def toc_start(body: str) -> int | None:
    """Offset of a table of contents at the end of a document (heading lines with dot leaders), else None."""
    m = re.search(r"^[ \t]*(?i:livre|titre|chapitre|section|table des mati[eè]res|sommaire)\b[^\n]*$", body, re.M)
    while m:
        window = body[m.start():m.start() + 1500]
        if len(_DOT_LEADER.findall(window)) >= 3:
            return m.start()
        m = re.compile(r"^[ \t]*(?i:livre|titre|chapitre|section|table des mati[eè]res|sommaire)\b[^\n]*$", re.M).search(body, m.end())
    return None


def apply_headings(state: dict, headings: list[dict]) -> dict:
    """Heading hierarchy after these headings: a heading replaces its level and clears the deeper levels."""
    state = dict(state)
    for h in headings:
        state = {lv: t for lv, t in state.items() if lv < h["level"]}
        state[h["level"]] = h["title"]
    return state


def hierarchy_text(state: dict) -> str:
    return " > ".join(state[lv] for lv in sorted(state))


def split_by_article(docs: list[Document], headings_forward: bool | None = None) -> list[Document]:
    """headings_forward (default config.SPLIT_HEADINGS_FORWARD): a heading block at the end of an article body is
    removed from it and becomes the "hierarchy" of the following articles; a table of contents after the last
    article is removed too. Chunk ids do not change (the stub rule is applied to the v1.1 text)."""
    if headings_forward is None:
        from config import SPLIT_HEADINGS_FORWARD as headings_forward
    chunks: list[Document] = []

    for doc in docs:
        text = clean_text(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        stem = Path(source).stem
        headings = _find_headings(text)
        _split_sections(headings)

        section_titles: dict[int, str] = {0: ""}
        seen: Counter = Counter()
        state: dict = {}
        gap = text[:headings[0]["start"]] if headings else ""      # text before the first article
        for i, h in enumerate(headings):
            if headings_forward:
                if i and headings[i - 1]["section"] != h["section"]:
                    state = {}
                state = apply_headings(state, _parse_heading_block(gap))
            gap = ""
            body_end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
            body = text[h["end"]:body_end]
            if i + 1 < len(headings) and headings[i + 1]["section"] != h["section"]:
                clipped, title = _clip_preamble(body)
                section_titles[headings[i + 1]["section"]] = title
                gap = body[len(clipped):]
                body = clipped
            content = (text[h["start"]:h["end"]] + body).strip()
            moved = ""
            if headings_forward:
                fwd = body
                cut = toc_start(fwd) if i + 1 == len(headings) else None
                if cut is not None:
                    fwd = fwd[:cut]
                cut = trailing_heading_start(fwd)
                if cut is not None:
                    moved = fwd[cut:]
                    gap = moved + gap
                    fwd = fwd[:cut]
            if len(content) < 30:  # skip stubs (decided on the v1.1 text, so chunk ids stay the same)
                continue
            if headings_forward:
                content = (text[h["start"]:h["end"]] + fwd.rstrip()).strip()

            base_id = f"{stem}/s{h['section']:02d}/{h['label']}"
            seen[base_id] += 1
            chunk_id = base_id if seen[base_id] == 1 else f"{base_id}~{seen[base_id]}"
            meta = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "section": h["section"],
                "article": h["base_s"],
                "article_label": h["label"],
            }
            if headings_forward:
                meta["hierarchy"] = hierarchy_text(state)
                meta["headings_moved"] = len(_parse_heading_block(moved))
            chunks.append(Document(page_content=content, metadata=meta))
        # section titles are known only after the pass (they come from the preceding text)
        for c in chunks:
            if c.metadata.get("source") == source and "section_title" not in c.metadata:
                c.metadata["section_title"] = section_titles.get(c.metadata["section"], "")
        if headings_forward:
            mine = [c for c in chunks if c.metadata.get("source") == source]
            if mine:
                main = Counter(c.metadata["section"] for c in mine).most_common(1)[0][0]
                for c in mine:
                    c.metadata["main_section"] = c.metadata["section"] == main

    logger.info("Article splitter: %d chunks from %d documents", len(chunks), len(docs))
    return chunks


def _trim_title(title: str) -> str:
    """Title of a bundled text without the headings that follow it ("CODE DE COMMERCE(1) LIVRE PREMIER ...")."""
    m = re.search(r"\s(?:LIVRE|Livre|TITRE|Titre|CHAPITRE|Chapitre|SECTION|Section)\s", title[10:])
    return (title[:10 + m.start()] if m else title).strip()


def code_name(meta: dict) -> str:
    """Display name of the code for the main text of a PDF (its largest section); the title of a bundled text."""
    from config import LAW_DISPLAY_NAMES
    where = LAW_DISPLAY_NAMES.get(meta.get("law_type"), meta.get("law_type", "") or "")
    if not meta.get("main_section", True) and meta.get("section_title"):
        where = _trim_title(meta["section_title"])
    return where


def context_line(meta: dict) -> str:
    """"<code name> > <heading hierarchy>", e.g. "Code Pénal > LIVRE II - ... > Section III - La concussion"."""
    h = meta.get("hierarchy") or ""
    return f"{code_name(meta)} > {h}" if h else code_name(meta)


def context_prefix(meta: dict) -> str:
    """Short context line for embedding: which text the chunk belongs to and which article it is.
    e.g. "Code Pénal - Article 226 ter" or "Loi n° 93-65 ... fonds de garantie ... - Article 2"."""
    from config import LAW_DISPLAY_NAMES
    where = LAW_DISPLAY_NAMES.get(meta.get("law_type"), meta.get("law_type", ""))
    if meta.get("section") and meta.get("section_title"):
        where = meta["section_title"]
    label = str(meta.get("article_label", meta.get("article", ""))).replace("-", " ")
    return f"{where} - Article {label}"
