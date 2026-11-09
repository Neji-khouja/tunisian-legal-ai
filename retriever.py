import re
import logging
import unicodedata
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import MMR_K, BM25_K, RERANK_POOL, FUSION, RRF_K

logger = logging.getLogger(__name__)

_ARABIC_RE       = re.compile(r"[؀-ۿ]")
_FRENCH_ACCENTS  = re.compile(r"[éèêëàâùûôîïçœæ]", re.IGNORECASE)
_EN_MARKERS      = {
    "the", "what", "are", "is", "does", "how", "can", "which", "who",
    "when", "where", "define", "explain", "what's", "i", "my", "do",
}


def detect_arabic(text: str) -> bool:
    arabic = sum(1 for _ in _ARABIC_RE.finditer(text))
    return arabic > len(text) * 0.2


def detect_english(text: str) -> bool:
    if detect_arabic(text):
        return False
    if _FRENCH_ACCENTS.search(text):
        return False
    return bool(set(text.lower().split()) & _EN_MARKERS)


# BM25 (fallback)
try:
    from langchain_community.retrievers import BM25Retriever as _BM25Cls
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning(
        "rank_bm25 not installed — BM25 retrieval disabled. "
        "Run: pip install rank_bm25"
    )

_bm25_instance = None   # built once per process from the Chroma corpus (v1.1 cache, BM25_TOKENIZER "whitespace")
_bm25_cache: dict = {}  # (id(vectorstore), tokenizer) -> BM25Retriever
_corpus_cache: dict = {}


# ── BM25 tokenizers (config.BM25_TOKENIZER) ─────────────────────────────────────────────────────────────────
_ELISION_RE = re.compile(r"\b(?:[cdjlmnst]|qu|lorsqu|puisqu|jusqu|quoiqu)['’]", re.I)
_LIGHT_SUFFIXES = ("ement", "ation", "atrice", "ateur", "euse", "eur", "ee", "er", "e")


def _fold(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _light_stem(w: str) -> str:
    """French light stemmer (plural and a few inflectional/derivational endings; stems keep >= 4 letters):
    condamnés, condamnation, condamné -> condamn; peines, peine -> pein; travaux -> traval."""
    if len(w) < 4 or not w.isalpha():
        return w
    if w.endswith("aux") and len(w) > 4:
        w = w[:-3] + "al"
    elif w[-1] in "sx" and not w.endswith("ss"):
        w = w[:-1]
    for suf in _LIGHT_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def bm25_tokens(text: str, mode: str | None = None) -> list[str]:
    """whitespace: str.split (v1.1, the langchain default); lower: lowercase, word characters only (punctuation and
    apostrophes split); fold: + accents removed; elision: + elided articles (l', d', qu' ...) dropped; stem: + light
    French stemming."""
    import config as C
    mode = mode or C.BM25_TOKENIZER
    if mode == "whitespace":
        return text.split()
    t = text.lower()
    if mode in ("elision", "stem"):
        t = _ELISION_RE.sub(" ", t)
    if mode in ("fold", "elision", "stem"):
        t = _fold(t)
    toks = re.findall(r"\w+", t)
    if mode == "stem":
        toks = [_light_stem(x) for x in toks]
    return toks


def _corpus(vectorstore: Chroma) -> dict:
    key = id(vectorstore)
    if key not in _corpus_cache:
        _corpus_cache[key] = vectorstore._collection.get(include=["documents", "metadatas"])
    return _corpus_cache[key]


def _get_bm25(vectorstore: Chroma):
    global _bm25_instance
    import config as C
    if not _BM25_AVAILABLE:
        return None
    mode = C.BM25_TOKENIZER
    key = (id(vectorstore), mode)
    if mode == "whitespace" and _bm25_instance is not None and not _bm25_cache:
        return _bm25_instance
    if key in _bm25_cache:
        return _bm25_cache[key]
    try:
        result = _corpus(vectorstore)
        if not result["documents"]:
            logger.warning("Chroma collection is empty — BM25 index not built")
            return None
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        # Use k * 4 internally so post-hoc domain filtering still leaves enough
        kw = {} if mode == "whitespace" else {"preprocess_func": lambda t: bm25_tokens(t, mode)}
        inst = _BM25Cls.from_documents(docs, k=BM25_K * 4, **kw)
        _bm25_cache[key] = inst
        if mode == "whitespace":
            _bm25_instance = inst
        logger.info("BM25 index built from %d documents (tokenizer %s)", len(docs), mode)
        return inst
    except Exception as exc:
        logger.warning("BM25 index build failed (%s) — semantic-only mode", exc)
        return None


# ── v1.2 index entries -> articles ──────────────────────────────────────────────────────────────────────────
def _to_article(doc: Document) -> Document:
    """A v1.2 entry (context line + article or sub-chunk) as the article: page_content = the plain article text,
    metadata chunk_id = the parent article id; the matched sub-chunk text is kept in metadata "match_text".
    A v1.1 document is returned unchanged."""
    m = doc.metadata
    if "article_text" not in m:
        return doc
    meta = {k: v for k, v in m.items() if k not in ("article_text", "piece_text")}
    meta["chunk_id"] = m.get("parent_id") or m.get("chunk_id")
    if m.get("parts", 1) > 1:
        meta["match_text"] = m.get("piece_text", "")
    return Document(page_content=m["article_text"], metadata=meta)


def _articles(docs: list[Document]) -> list[Document]:
    """Entries -> articles, first (best-ranked) entry of each article kept."""
    return _deduplicate([_to_article(d) for d in docs])


def _build_filter(law_type: str | None, article_num: str | None = None) -> dict | None:
    parts: list[dict] = []
    if law_type:
        parts.append({"law_type": {"$eq": law_type}})
    if article_num:
        parts.append({"article": {"$eq": article_num}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks by chunk_id (fallback: source + article). Preserves first-seen order."""
    seen: set = set()
    out:  list[Document] = []
    for doc in docs:
        key = doc.metadata.get("chunk_id") or (doc.metadata.get("source"), doc.metadata.get("article"))
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def _semantic_search(
    vectorstore: Chroma,
    query: str,
    law_type: str | None,
    k: int,
) -> list[Document]:
    docs = vectorstore.similarity_search(query, k=k, filter=_build_filter(law_type))
    if not docs and law_type:
        logger.debug("Domain filter '%s' returned nothing — retrying without filter", law_type)
        docs = vectorstore.similarity_search(query, k=k)
    return _articles(docs)


def _bm25_search(bm25, query: str, law_type: str | None, k: int) -> list[Document]:
    candidates = _articles(bm25.invoke(query))
    if law_type:
        candidates = [d for d in candidates if d.metadata.get("law_type") == law_type]
    return candidates[:k]


def _direct_lookup(
    vectorstore: Chroma,
    article_num: str,
    law_type: str | None,
    k: int,
    relax: bool = True,
    query: str | None = None,
) -> list[Document]:
    """Articles numbered article_num (metadata "article"), ordered by similarity to query (default "Article N").
    relax (v1.1): when none exists, a domain-only similarity search for "Article N"."""
    query = query or f"Article {article_num}"
    # Try with full filter first (domain + article number)
    docs = vectorstore.similarity_search(
        query, k=k, filter=_build_filter(law_type, article_num)
    )
    if docs or not relax:
        return _articles(docs)
    # Relax to domain-only
    docs = vectorstore.similarity_search(query, k=k, filter=_build_filter(law_type))
    return _articles(docs)


# ── direct article lookup trigger (config.DIRECT_LOOKUP_STRICT) ─────────────────────────────────────────────
# a number counts only right after an article marker: "art. 5", "article 227 bis", "articles 12 et 13",
# "l'article premier", "الفصل 5", "الفصل عدد 12". Dates, law numbers ("loi n° 2000-10"), amounts and bare numbers do not.
_ART_MARKER_RE = re.compile(
    r"(?:(?<![\w])(?:articles?|art)(?![\w])\.?|الفصول|الفصلين|الفصل|فصل)[ \t]*"
    r"(?:(?:n[°ºo]\.?|num[ée]ro|number|no\.|عدد|رقم)[ \t]*)?"
    r"(?P<first>\d+|premier|1er)(?![\d/.,-]\d)"
    r"(?P<more>(?:[ \t]*(?:,|et|à|au|ou|and|to|or|و|إلى|الى|أو)[ \t]*\d+(?![\d/.-]\d))*)",
    re.I,
)


def article_numbers(question: str, limit: int = 3) -> list[str]:
    """Article numbers the question names after an article marker, in order, at most limit."""
    out: list[str] = []
    for m in _ART_MARKER_RE.finditer(question or ""):
        first = m.group("first").lower()
        nums = ["1" if first in ("premier", "1er") else str(int(first))]
        nums += [str(int(x)) for x in re.findall(r"\d+", m.group("more") or "")]
        for n in nums:
            if n not in out:
                out.append(n)
    return out[:limit]


# ── cross-references between articles (config.XREF_EXPAND) ──────────────────────────────────────────────────
_XR_SUFFIX = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)"
_XR_NUM = rf"\d+(?:[ \t]*-?[ \t]*{_XR_SUFFIX}(?![\w]))?"
_XR_OWN_HEADER = re.compile(rf"^\s*(?:article|art\.?)\s*(?:{_XR_NUM}|premier|1er)\s*(?:\([^)]*\))?\s*(?:\.?\s*[-–]|\.|:)?", re.I)
_XR_FR = re.compile(rf"(?<![\w])(?:articles?|art\.)\s+(?P<list>{_XR_NUM}(?:\s*(?:,|et|à|au|ou)\s*{_XR_NUM})*)", re.I)
_XR_REL_FR = re.compile(r"(?<![\w])article\s+(?P<rel>pr[ée]c[ée]dent|suivant)(?![\w])", re.I)
_XR_EXTERNAL_FR = re.compile(
    r"^\s*(?:,\s*)?(?:(?:alin[ée]a|paragraphe|al\.|§)\s*\S+\s+)?(?:du|de\s+la|de\s+l['’]|des|dudit|de\s+ladite)\s+"
    r"(?:m[êe]me\s+)?(?!pr[ée]sente?\b)(?:code|loi|d[ée]cret|convention|constitution|r[èe]glement|arr[êe]t[ée]|"
    r"statut|trait[ée]|ordonnance)", re.I)
_XR_NUM_AR = r"[\d٠-٩]+(?:\s*(?:مكرر|ثالثا|رابعا))?"
_XR_AR = re.compile(rf"(?:الفصول|الفصلين|الفصلان|للفصل|بالفصل|الفصل)\s+(?:عدد\s+)?(?:من\s+)?"
                    rf"(?P<list>{_XR_NUM_AR}(?:\s*(?:،|,|و|إلى|الى|أو)\s*(?:الفصل\s+)?{_XR_NUM_AR})*)")
_XR_REL_AR = re.compile(r"الفصل\s+(?P<rel>السابق|المتقدم|الموالي|اللاحق|التالي)")
_XR_EXTERNAL_AR = re.compile(r"^\s*من\s+(?!هذه|هذا)(?:ال)?(?:مجلة|قانون|أمر|دستور|مرسوم)")
_SUFFIX_RANK = {s: i for i, s in enumerate(["", "bis", "ter", "quater", "quinquies", "sexies", "septies", "octies",
                                            "nonies", "decies"])}


def _xr_label(raw: str) -> str:
    """'494 bis' -> '494-bis', '٥' -> '5', '12 مكرر' -> '12-bis' (the chunk-id spelling)."""
    raw = raw.strip().replace("مكرر", "bis").replace("ثالثا", "ter").replace("رابعا", "quater")
    m = re.match(rf"(\d+)\s*-?\s*({_XR_SUFFIX})?", raw, re.I)
    return str(int(m.group(1))) + (f"-{m.group(2).lower()}" if m.group(2) else "")


def _label_key(label: str) -> tuple:
    parts = str(label).split("~")[0].split("-")
    base = int(parts[0]) if parts[0].isdigit() else 0
    rest = parts[1] if len(parts) > 1 else ""
    return (base, int(rest) if rest.isdigit() else _SUFFIX_RANK.get(rest, 50))


def parse_references(text: str, own_label: str | None = None, range_max: int | None = None) -> list[dict]:
    """References of an article to other articles OF THE SAME TEXT, in order: [{"label", "kind", "neighbour"}].
    numbered: "article 494", "articles 260 à 263" (a range is expanded only when it names at most range_max
    articles, else dropped), "articles 12, 13 et 15", "الفصل 5", "الفصول من 5 إلى 8"; a reference followed by
    "du code ...", "de la loi ..." (not "du présent code") or "من مجلة ..." is to another text and dropped.
    relative: "article précédent" / "suivant", "الفصل السابق" / "الموالي" -> {"neighbour": -1 | +1} (resolved
    against the article order of the section by the caller). Vague references (ci-dessus, susvisé, أعلاه) are not
    returned. The article's own header and its own label are skipped."""
    import config as C
    range_max = C.XREF_RANGE_MAX if range_max is None else range_max
    body = _XR_OWN_HEADER.sub("", text or "", count=1)
    found: list[tuple[int, dict]] = []

    def numbered(m, external_re, range_seps):
        if external_re.match(body[m.end():m.end() + 80]):
            return
        parts = re.split(r"\s*(,|،|\bet\b|\bà\b|\bau\b|\bou\b|و|إلى|الى|أو)\s*(?:الفصل\s+)?", m.group("list"))
        nums, seps = parts[0::2], parts[1::2]
        labels: list[str] = []
        for i, n in enumerate(nums):
            if not n.strip():
                continue
            lab = _xr_label(n)
            if i and seps[i - 1] in range_seps:
                lo, hi = int(re.match(r"\d+", labels[-1]).group(0)) if labels else None, int(re.match(r"\d+", lab).group(0))
                if lo is not None and 0 < hi - lo < range_max:
                    labels += [str(x) for x in range(lo + 1, hi + 1)]
                elif labels:
                    labels.pop()          # a range wider than range_max: neither end is kept
                continue
            labels.append(lab)
        for lab in labels:
            found.append((m.start(), {"label": lab, "kind": "numbered", "neighbour": 0}))

    for m in _XR_FR.finditer(body):
        numbered(m, _XR_EXTERNAL_FR, ("à", "au"))
    for m in _XR_AR.finditer(body):
        numbered(m, _XR_EXTERNAL_AR, ("إلى", "الى"))
    for m in _XR_REL_FR.finditer(body):
        found.append((m.start(), {"label": None, "kind": "relative", "neighbour": -1 if m.group("rel").lower().startswith("pr") else 1}))
    for m in _XR_REL_AR.finditer(body):
        found.append((m.start(), {"label": None, "kind": "relative",
                                  "neighbour": -1 if m.group("rel") in ("السابق", "المتقدم") else 1}))
    out, seen = [], set()
    for _, r in sorted(found, key=lambda x: x[0]):
        k = (r["label"], r["neighbour"])
        if (r["label"] and r["label"] == own_label) or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _article_index(vectorstore: Chroma) -> dict:
    """{"by_label": {(stem, section, label): Document}, "order": {(stem, section): [labels in article order]}}."""
    key = ("articles", id(vectorstore))
    if key in _corpus_cache:
        return _corpus_cache[key]
    got = _corpus(vectorstore)
    by_label, order = {}, {}
    for text, meta in zip(got["documents"], got["metadatas"]):
        d = _to_article(Document(page_content=text or "", metadata=meta))
        cid = d.metadata.get("chunk_id") or ""
        parts = cid.split("/")
        if len(parts) < 3:
            continue
        label = parts[2].split("~")[0]
        k = (parts[0], parts[1], label)
        if k not in by_label:
            by_label[k] = d
            order.setdefault((parts[0], parts[1]), []).append(label)
    for labs in order.values():
        labs.sort(key=_label_key)
    _corpus_cache[key] = {"by_label": by_label, "order": order}
    return _corpus_cache[key]


def expand_cross_references(vectorstore: Chroma, docs: list[Document], per_article: int | None = None,
                            total: int | None = None, range_max: int | None = None) -> list[Document]:
    """Articles referenced by the evidence articles (same PDF and section), not already in the evidence:
    at most per_article per evidence article and total in all, in evidence order then reference order. Each added
    Document has metadata cited_by (label of the citing article), cited_by_id and xref_kind. The index is read only
    when some evidence article has a reference."""
    import config as C
    per_article = C.XREF_PER_ARTICLE if per_article is None else per_article
    total = C.XREF_TOTAL if total is None else total
    refs = [(d, parse_references(d.page_content, (d.metadata.get("chunk_id") or "").split("/")[-1].split("~")[0], range_max))
            for d in docs]
    if not any(r for _, r in refs):
        return []
    idx = _article_index(vectorstore)
    have = {d.metadata.get("chunk_id") for d in docs}
    added: list[Document] = []
    for d, rs in refs:
        if len(added) >= total:
            break
        parts = (d.metadata.get("chunk_id") or "").split("/")
        if len(parts) < 3:
            continue
        stem, sec, own = parts[0], parts[1], parts[2].split("~")[0]
        n = 0
        for r in rs:
            if n >= per_article or len(added) >= total:
                break
            label = r["label"]
            if label is None:
                labs = idx["order"].get((stem, sec), [])
                if own not in labs:
                    continue
                j = labs.index(own) + r["neighbour"]
                if not 0 <= j < len(labs):
                    continue
                label = labs[j]
            hit = idx["by_label"].get((stem, sec, label))
            if hit is None or hit.metadata.get("chunk_id") in have:
                continue
            added.append(Document(page_content=hit.page_content,
                                  metadata={**hit.metadata, "cited_by": own, "cited_by_id": d.metadata.get("chunk_id"),
                                            "xref_kind": r["kind"]}))
            have.add(hit.metadata.get("chunk_id"))
            n += 1
    return added


def _doc_key(doc: Document):
    return doc.metadata.get("chunk_id") or (doc.metadata.get("source"), doc.metadata.get("article"))


def _fuse(ranked_lists: list[list[Document]]) -> list[Document]:
    """Merge ranked lists into one candidate order.
    rrf: reciprocal rank fusion, score(d) = sum over lists of 1 / (RRF_K + rank), so a chunk that several
    sources rank highly beats one that a single source ranks first. roundrobin: rank 1 of every list, then rank 2, ..."""
    if FUSION == "roundrobin":
        merged: list[Document] = []
        for rank in range(max((len(l) for l in ranked_lists), default=0)):
            for l in ranked_lists:
                if rank < len(l):
                    merged.append(l[rank])
        return merged
    scores: dict = {}
    first: dict = {}
    for l in ranked_lists:
        for rank, doc in enumerate(l, 1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            first.setdefault(key, doc)
    return [first[k] for k in sorted(scores, key=lambda k: -scores[k])]


def fuse_pools(pool_a: list[Document], pool_b: list[Document], k: int = RRF_K) -> list[Document]:
    """Wave 2 V3 (config.NORMALIZE_QUERY = "v3"): reciprocal rank fusion of two candidate pools, score = sum of
    1 / (k + rank), ties in order of first appearance (pool_a first), cut to the larger pool size (the reranker
    scores no more candidates than for one query). Same as diagnostics/wave2_core.fused_pool (tested)."""
    scores: dict = {}
    first: dict = {}
    order: list = []
    for pool in (pool_a, pool_b):
        for rank, doc in enumerate(pool, 1):
            key = _doc_key(doc)
            if key not in scores:
                scores[key], first[key] = 0.0, doc
                order.append(key)
            scores[key] += 1.0 / (k + rank)
    pos = {key: i for i, key in enumerate(order)}
    merged = [first[key] for key in sorted(order, key=lambda x: (-scores[x], pos[x]))]
    return merged[:max(len(pool_a), len(pool_b))]


def hybrid_retrieve(
    vectorstore: Chroma,
    queries: list[str],
    original_query: str,
    law_type: str | None = None,
) -> list[Document]:
    # One ranked list per evidence source, then a round-robin merge (rank 1 of every list, rank 2 ...)
    # so that truncating to RERANK_POOL keeps the best of each source instead of only the first source.
    ranked_lists: list[list[Document]] = []

    for query in queries:
        docs = _semantic_search(vectorstore, query, law_type, k=MMR_K)
        ranked_lists.append(docs)
        logger.debug("semantic(%r): %d docs", query[:60], len(docs))

    bm25 = _get_bm25(vectorstore)
    if bm25:
        bm25_docs = _bm25_search(bm25, queries[0], law_type, k=BM25_K)
        ranked_lists.append(bm25_docs)
        logger.debug("BM25(%r): %d docs", queries[0][:60], len(bm25_docs))

    if detect_arabic(original_query) or detect_english(original_query):
        orig_docs = _semantic_search(vectorstore, original_query, law_type, k=10)
        ranked_lists.append(orig_docs)
        logger.debug("multilingual fallback: %d docs", len(orig_docs))

    merged = _fuse(ranked_lists)

    deduped = _deduplicate(merged)
    if RERANK_POOL:
        deduped = deduped[:RERANK_POOL]
    logger.debug(
        "hybrid_retrieve: %d raw -> %d unique | %d queries",
        len(merged), len(deduped), len(queries),
    )
    return deduped


def retrieve(
    vectorstore: Chroma,
    question: str,
    law_type: str | None = None,
    k: int = MMR_K,
    search_query: str | None = None,
) -> list[Document]:
    article_match = re.search(r"\b(\d+)\b", question)
    if article_match:
        docs = _direct_lookup(vectorstore, article_match.group(1), law_type, k)
        if docs:
            return docs

    primary = search_query or question
    return hybrid_retrieve(vectorstore, [primary], question, law_type)
