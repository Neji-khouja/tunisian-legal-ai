"""Unit tests of the v2.1 wave-1 retrieval changes, on stub data (no index, no model, no LLM).

    python test_v21_retrieval.py

heading logic (splitter), direct-lookup rule (retriever.article_numbers), union of direct lookup and hybrid pool
(pipeline.step_retrieve), cross-reference parser and expansion caps (retriever), sub-chunking (pipeline.index_entries,
retriever._to_article, reranker), evidence assembly, BM25 tokenizers, flags off = v1.1 behaviour, frozen index.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from langchain_core.documents import Document
import config as C
import splitter as sp
import retriever as R
import pipeline as P
import reranker as RK

FAILS = []
N = {"n": 0}


def check(name, ok, detail=""):
    N["n"] += 1
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"   {detail}"))
    if not ok:
        FAILS.append(name)


class flags:
    """with flags(XREF_EXPAND=False, ...): temporarily set config switches."""
    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(C, k)
            setattr(C, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(C, k, v)


# ---------------------------------------------------------------- headings

PDF = """CODE X
LIVRE PREMIER
DISPOSITIONS GENERALES
CHAPITRE PREMIER
Du champ d'application
Article premier.- Le présent code s'applique à toutes les personnes.
Article 2.- La section 2 du présent chapitre est applicable aux mineurs.
Section 2 du présent chapitre ne vise pas les majeurs incapables.
Section II - La concussion
Article 3.- Est puni de cinq ans tout fonctionnaire qui exige une somme.
CHAPITRE II
DES PEINES ET DE LEUR
EXECUTION
Article 4.- Les peines principales sont prévues par l'article 3.
LIVRE II - Des infractions..................... 10 à 20
TITRE PREMIER - Atteintes.......................... 10 à 15
CHAPITRE I - Divers ............................... 10 à 12
"""


def split(forward):
    return sp.split_by_article([Document(page_content=PDF, metadata={"source": "X.pdf", "law_type": "penal"})],
                               headings_forward=forward)


def test_headings():
    old, new = split(False), split(True)
    ids = lambda cs: [c.metadata["chunk_id"] for c in cs]
    by = {c.metadata["chunk_id"]: c for c in new}
    check("headings: chunk ids unchanged by the fix", ids(old) == ids(new) == ["X/s00/1", "X/s00/2", "X/s00/3", "X/s00/4"], ids(new))
    check("headings: v1.1 split keeps the heading at the end of the previous article",
          old[1].page_content.rstrip().endswith("Section II - La concussion"))
    check("headings: heading removed from the previous article", "Section II" not in by["X/s00/2"].page_content)
    check("headings: a sentence starting with 'Section 2 du présent chapitre' stays in the article",
          "Section 2 du présent chapitre ne vise pas" in by["X/s00/2"].page_content)
    check("headings: the heading attaches to the FOLLOWING article",
          by["X/s00/3"].metadata["hierarchy"].endswith("Section II - La concussion"), by["X/s00/3"].metadata["hierarchy"])
    check("headings: hierarchy of article 1 from the text before it",
          by["X/s00/1"].metadata["hierarchy"] == "LIVRE PREMIER - DISPOSITIONS GENERALES > CHAPITRE PREMIER - Du champ d'application",
          by["X/s00/1"].metadata["hierarchy"])
    check("headings: article 2 keeps its chapter, no section", by["X/s00/2"].metadata["hierarchy"].endswith("Du champ d'application"))
    h4 = by["X/s00/4"].metadata["hierarchy"]
    check("headings: a new chapter clears the section; multi-line title joined",
          h4 == "LIVRE PREMIER - DISPOSITIONS GENERALES > CHAPITRE II - DES PEINES ET DE LEUR EXECUTION", h4)
    check("headings: chapter heading removed from article 3", "CHAPITRE II" not in by["X/s00/3"].page_content)
    check("headings: table of contents after the last article removed",
          "...." not in by["X/s00/4"].page_content and by["X/s00/4"].page_content.rstrip().endswith("l'article 3."),
          by["X/s00/4"].page_content[-80:])
    check("headings: moved-heading count", [c.metadata["headings_moved"] for c in new] == [0, 1, 1, 0],
          [c.metadata["headings_moved"] for c in new])
    check("headings: heading-line rule", bool(sp._heading_line("Section Première - Absence de criminalité"))
          and bool(sp._heading_line("TITRE III")) and bool(sp._heading_line("Sous-section 2"))
          and bool(sp._heading_line("Titre Deuxième")) and not sp._heading_line("Section 2 du présent chapitre est applicable")
          and not sp._heading_line("LIVRE PREMIER - Des obligations........... 1 à 563"))
    check("headings: apply_headings clears deeper levels",
          sp.apply_headings({1: "L", 4: "C", 5: "S"}, [{"level": 4, "title": "C2"}]) == {1: "L", 4: "C2"})
    check("headings: context line", sp.context_line({"law_type": "penal", "section": 0, "hierarchy": "A > B"}) == "Code Pénal > A > B")


# ---------------------------------------------------------------- direct lookup rule

def test_direct_rule():
    cases = {
        "Quelle est la peine prévue par l'article 227 bis ?": ["227"],
        "Que disent les art. 5 et 6 du code ?": ["5", "6"],
        "Que dit l'article premier ?": ["1"],
        "What does Article 12 of the labour code say?": ["12"],
        "ماذا ينص الفصل 12 ؟": ["12"],
        "الفصل عدد 7 من المجلة": ["7"],
        "article 5 de la loi du 2 janvier 2000": ["5"],
        "articles 260 à 263": ["260", "263"],
    }
    for q, want in cases.items():
        got = R.article_numbers(q)
        check(f"direct rule: {q!r} -> {want}", got == want, got)
    for q in ["Un contrat signé le 2 janvier 2000 est-il valable ?", "La loi n° 2000-10 du 24 janvier 2000",
              "J'ai 17 ans, puis-je travailler ?", "Une amende de 500 dinars", "Le 12/03/2021 j'ai été licencié",
              "Combien de jours de congé après 3 ans ?", "عمري 17 سنة", "In 2019 I signed a lease"]:
        check(f"direct rule: no trigger for {q!r}", R.article_numbers(q) == [], R.article_numbers(q))


# ---------------------------------------------------------------- union

def doc(cid, text="t", **meta):
    return Document(page_content=text, metadata={"chunk_id": cid, **meta})


def test_union():
    calls = []
    real = (R._direct_lookup, P.hybrid_retrieve)
    R._direct_lookup = lambda vs, n, lt, k, relax=True, query=None: (calls.append((n, relax, query)), [doc(f"D/s00/{n}"), doc("H/s00/1")])[1]
    P.hybrid_retrieve = lambda vs, qs, q, lt: [doc("H/s00/1"), doc("H/s00/2")]
    try:
        with flags(DIRECT_LOOKUP_STRICT=True, DIRECT_LOOKUP_UNION=True):
            out = P.step_retrieve(None, "Que dit l'article 5 ?", "fq", [], None)
            check("union: direct articles in front of the hybrid pool, deduplicated",
                  [d.metadata["chunk_id"] for d in out] == ["D/s00/5", "H/s00/1", "H/s00/2"], [d.metadata["chunk_id"] for d in out])
            check("union: strict lookup (no relaxed search), ordered by the French query", calls == [("5", False, "fq")], calls)
            calls.clear()
            out = P.step_retrieve(None, "Contrat signé le 2 janvier 2000", "fq", [], None)
            check("union: a date does not trigger the lookup", calls == [] and len(out) == 2, calls)
        calls.clear()
        with flags(DIRECT_LOOKUP_STRICT=False, DIRECT_LOOKUP_UNION=False):
            out = P.step_retrieve(None, "Contrat signé le 2 janvier 2000", "fq", [], None)
            check("flags off (v1.1): the date's number triggers the lookup and replaces the pool",
                  calls == [("2", True, None)] and [d.metadata["chunk_id"] for d in out] == ["D/s00/2", "H/s00/1"], (calls, out))
        calls.clear()
        with flags(DIRECT_LOOKUP_STRICT=True, DIRECT_LOOKUP_UNION=False):
            P.step_retrieve(None, "Que dit l'article 5 ?", "fq", [], None)
            check("strict without union: v1.1 replacement on the marked number", calls == [("5", True, None)], calls)
    finally:
        R._direct_lookup, P.hybrid_retrieve = real


# ---------------------------------------------------------------- cross-references

def labels(text, own=None, range_max=4):
    return [(r["label"], r["kind"], r["neighbour"]) for r in R.parse_references(text, own, range_max)]


def test_xref_parser():
    check("xref: own header skipped", labels("Article 494.- Rien.", "494") == [])
    check("xref: numbered", labels("Article 539.- dans les formes prescrites par l'article 494, mais") == [("494", "numbered", 0)])
    check("xref: list", [x[0] for x in labels("Voir les articles 12, 13 et 15.")] == ["12", "13", "15"])
    check("xref: range up to 4 articles expanded", [x[0] for x in labels("prévues aux articles 260 à 263.")] == ["260", "261", "262", "263"])
    check("xref: range wider than 4 dropped", labels("prévues aux articles 1 à 50.") == [])
    check("xref: suffix", [x[0] for x in labels("l'article 226 bis est applicable")] == ["226-bis"])
    check("xref: another code dropped", labels("prévue à l'article 62 du code de procédure pénale") == [])
    check("xref: another law dropped", labels("conformément à l'article 3 de la loi n° 2000-10") == [])
    check("xref: 'du présent code' kept", [x[0] for x in labels("l'article 62 du présent code")] == ["62"])
    check("xref: relative", labels("dans le cas de l'article précédent et de l'article suivant") ==
          [(None, "relative", -1), (None, "relative", 1)])
    check("xref: vague references skipped", labels("les dispositions ci-dessus et l'article susvisé") == [])
    check("xref: arabic numbered", [x[0] for x in labels("حسب الفصل 5 من هذه المجلة")] == ["5"])
    check("xref: arabic range", [x[0] for x in labels("الفصول من 5 إلى 8")] == ["5", "6", "7", "8"])
    check("xref: arabic list", [x[0] for x in labels("الفصلين 12 و13")] == ["12", "13"])
    check("xref: arabic relative", labels("المنصوص عليها بالفصل السابق") == [(None, "relative", -1)])
    check("xref: arabic other code dropped", labels("الفصل 3 من مجلة الشغل") == [])
    check("xref: arabic suffix", [x[0] for x in labels("الفصل 12 مكرر")] == ["12-bis"])


class StubVS:
    def __init__(self, docs):
        self._collection = SimpleNamespace(get=lambda include=None: {"documents": [d.page_content for d in docs],
                                                                      "metadatas": [d.metadata for d in docs]})


def test_xref_expand():
    corpus = [doc(f"X/s01/{n}", f"Article {n}.- texte {n}") for n in range(1, 31)] + \
             [doc("X/s01/226", "Article 226.- a"), doc("X/s01/226-bis", "Article 226 bis.- voir l'article précédent"),
              doc("X/s02/5", "Article 5.- autre texte")]
    vs = StubVS(corpus)
    ev = [doc("X/s01/1", "Article 1.- voir les articles 2, 3, 4 et 5"),
          doc("X/s01/10", "Article 10.- voir l'article 11 et l'article 12 et l'article 13"),
          doc("X/s01/20", "Article 20.- voir l'article 2, l'article 21, l'article 22"),
          doc("X/s01/25", "Article 25.- voir l'article 26 et l'article 27")]
    added = R.expand_cross_references(vs, ev, per_article=2, total=6)
    got = [(d.metadata["chunk_id"], d.metadata["cited_by"]) for d in added]
    check("xref caps: 2 per article, 6 in total, deduplicated, evidence order",
          got == [("X/s01/2", "1"), ("X/s01/3", "1"), ("X/s01/11", "10"), ("X/s01/12", "10"), ("X/s01/21", "20"),
                  ("X/s01/22", "20")], got)
    added = R.expand_cross_references(vs, [doc("X/s01/5", "Article 5.- voir l'article 4"), doc("X/s01/4", "Article 4.- x")])
    check("xref: an article already in the evidence is not added", added == [])
    added = R.expand_cross_references(vs, [doc("X/s01/226-bis", "Article 226 bis.- voir l'article précédent")])
    check("xref: 'article précédent' of 226-bis is 226 (section order)", [d.metadata["chunk_id"] for d in added] == ["X/s01/226"])
    added = R.expand_cross_references(vs, [doc("X/s02/9", "Article 9.- voir l'article 5")])
    check("xref: resolved in the same section", [d.metadata["chunk_id"] for d in added] == ["X/s02/5"])
    added = R.expand_cross_references(vs, [doc("X/s01/1", "Article 1.- voir l'article 99")])
    check("xref: unknown article not added", added == [])
    check("xref: labelled header in the evidence text",
          P.evidence_text(Document(page_content="T", metadata={"cited_by": "539"})).startswith("Article cité par l'article 539\nT"))


# ---------------------------------------------------------------- sub-chunking

def test_subchunks():
    length = lambda t: len(t.split())                 # stub tokenizer: one token per word
    long_text = "Article 9.- " + "\n".join(f"ligne {i} " + "mot " * 8 for i in range(12))
    chunks = [Document(page_content="Article 1.- court", metadata={"chunk_id": "X/s00/1", "law_type": "penal", "section": 0}),
              Document(page_content=long_text, metadata={"chunk_id": "X/s00/9", "law_type": "penal", "section": 0, "hierarchy": "TITRE I"})]
    ents = P.index_entries(chunks, length, max_tokens=40, contextual=True, subchunk=True)
    short = [e for e in ents if e["metadata"]["parent_id"] == "X/s00/1"]
    parts = [e for e in ents if e["metadata"]["parent_id"] == "X/s00/9"]
    check("subchunk: a short article is one entry with its own id", [e["id"] for e in short] == ["X/s00/1"])
    check("subchunk: indexed text = context line + article", short[0]["document"] == "Code Pénal\nArticle 1.- court", short[0]["document"])
    check("subchunk: plain text and context in metadata", short[0]["metadata"]["article_text"] == "Article 1.- court"
          and short[0]["metadata"]["context"] == "Code Pénal")
    check("subchunk: a long article is split, every piece within the limit", len(parts) > 1 and all(length(e["document"]) <= 40 for e in parts),
          [length(e["document"]) for e in parts])
    check("subchunk: piece ids <id>#k, chunk_id metadata = parent id",
          [e["id"] for e in parts] == [f"X/s00/9#{k}" for k in range(1, len(parts) + 1)]
          and all(e["metadata"]["chunk_id"] == "X/s00/9" for e in parts))
    check("subchunk: every piece carries the context line", all(e["document"].startswith("Code Pénal > TITRE I\n") for e in parts))
    check("subchunk: pieces cover the article", "\n".join(e["metadata"]["piece_text"] for e in parts) == long_text)
    d2 = Document(page_content=parts[1]["document"], metadata=parts[1]["metadata"])
    art = R._to_article(d2)
    check("subchunk: a matching piece returns the full article", art.page_content == long_text
          and art.metadata["chunk_id"] == "X/s00/9" and art.metadata["match_text"] == parts[1]["metadata"]["piece_text"])
    two = R._articles([d2, Document(page_content=parts[0]["document"], metadata=parts[0]["metadata"])])
    check("subchunk: two pieces of one article -> one candidate", len(two) == 1)
    one_line = P.split_long("a " * 100, length, 30)
    check("subchunk: a line longer than the limit is cut at spaces", len(one_line) == 4 and all(length(p) <= 30 for p in one_line))
    v11 = Document(page_content="Article 3.- x", metadata={"chunk_id": "X/s00/3"})
    check("subchunk: a v1.1 document is unchanged", R._to_article(v11) is v11)
    # evidence text of a long article: the window holds the matched piece
    big = Document(page_content="Article 9.- début\n" + "a" * 5000 + " CIBLE " + "b" * 5000,
                   metadata={"chunk_id": "X/s00/9", "parts": 3, "match_text": "CIBLE " + "b" * 50, "context": "Code Pénal"})
    with flags(EVIDENCE_TEXT_CAP=3000, EVIDENCE_CONTEXT_HEADER=True):
        t = P.evidence_text(big)
    check("subchunk: evidence = context header + capped window containing the match",
          t.startswith("Contexte : Code Pénal\nArticle 9.- début\n[…]") and "CIBLE" in t and len(t) <= 3000 + 40, len(t))
    # the reranker scores the matched piece
    seen = []
    real = RK._reranker
    RK._reranker = SimpleNamespace(predict=lambda pairs, **kw: (seen.extend(p[1] for p in pairs), [1.0] * len(pairs))[1])
    try:
        with flags(SUBCHUNK_LONG=True):
            RK.rerank("q", [art, v11])
        check("subchunk: reranker scores the matched piece, returns the article", seen == [art.metadata["match_text"], "Article 3.- x"], seen)
    finally:
        RK._reranker = real


# ---------------------------------------------------------------- evidence assembly, BM25 tokens, flags off, frozen index

def test_misc():
    d = Document(page_content="Article 5.- x", metadata={"chunk_id": "X/s00/5", "context": "Code Pénal > TITRE I"})
    with flags(EVIDENCE_CONTEXT_HEADER=True):
        check("evidence: labelled context header", P.evidence_text(d) == "Contexte : Code Pénal > TITRE I\nArticle 5.- x")
    with flags(EVIDENCE_CONTEXT_HEADER=False, XREF_EXPAND=False):
        check("flags off: evidence text = article text", P.evidence_text(d) == "Article 5.- x")
        check("flags off: no cross-reference expansion", P.evidence_docs([doc("X/s00/1", "voir l'article 2")]) == [doc("X/s00/1", "voir l'article 2")])
    with flags(XREF_EXPAND=True):
        check("evidence: no reference -> index never opened", P.evidence_docs([doc("X/s00/1", "rien")]) == [doc("X/s00/1", "rien")])
    s = "L'article Pénal, l’emprisonnement des condamnés"
    check("bm25: whitespace (v1.1)", R.bm25_tokens(s, "whitespace") == s.split())
    check("bm25: lower", R.bm25_tokens(s, "lower") == ["l", "article", "pénal", "l", "emprisonnement", "des", "condamnés"])
    check("bm25: fold", R.bm25_tokens(s, "fold") == ["l", "article", "penal", "l", "emprisonnement", "des", "condamnes"])
    check("bm25: elision", R.bm25_tokens(s, "elision") == ["article", "penal", "emprisonnement", "des", "condamnes"])
    check("bm25: stem", R.bm25_tokens(s, "stem") == ["articl", "penal", "emprisonn", "des", "condamn"], R.bm25_tokens(s, "stem"))
    for target in (C.DB_V11, str(C.DB_V11)):
        try:
            P.build_vectorstore(db_dir=target)
            check("frozen: build into db_v1_1 refused", False)
        except RuntimeError:
            check("frozen: build into db_v1_1 refused", True)
    try:
        P.build_index_v12(db_dir=C.DB_V11)
        check("frozen: v1.2 build into db_v1_1 refused", False)
    except RuntimeError:
        check("frozen: v1.2 build into db_v1_1 refused", True)


if __name__ == "__main__":
    for t in (test_headings, test_direct_rule, test_union, test_xref_parser, test_xref_expand, test_subchunks, test_misc):
        t()
    print(f"\n{N['n'] - len(FAILS)}/{N['n']} checks passed")
    sys.exit(1 if FAILS else 0)
