
import re
import logging
import textwrap
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import CONTEXTUAL_CHUNKS, EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX, EMBED_MAX_SEQ_LENGTH
from config import DB_DIR, DATA_DIR, EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY, SUBQUERY_N, RAW_FIRST, USE_SUBQUERIES, RERANK_ENABLED, RERANK_TOP_K
from loaders import load_all_documents
from splitter import split_by_article, context_prefix
from retriever import hybrid_retrieve, detect_arabic, detect_english
from translate import rewrite_query, generate_subqueries, translate_to_french
from reranker import rerank

logger = logging.getLogger(__name__)

class _PrefixedEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings plus the document/query prefixes some models require (e5: "passage: " / "query: ")."""
    doc_prefix: str = ""
    query_prefix: str = ""

    def embed_documents(self, texts):
        return super().embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text):
        return super().embed_query(self.query_prefix + text)


_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = _PrefixedEmbeddings(
            model_name=EMBEDDING_MODEL, doc_prefix=EMBED_DOC_PREFIX, query_prefix=EMBED_QUERY_PREFIX,
            model_kwargs={"device": "cpu"},
        )
        if EMBED_MAX_SEQ_LENGTH:
            client = getattr(_embeddings, "_client", None) or getattr(_embeddings, "client")
            client.max_seq_length = EMBED_MAX_SEQ_LENGTH
    return _embeddings


_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Tunisian legal assistant. Legal source documents are written in French or Arabic.
    Users may submit questions in French, Arabic, or English.

    STRICT GROUNDING RULES — violating any of these is an error:
    1. Base your answer EXCLUSIVELY on the provided context passages.
       Never add information, penalties, dates, or article numbers not present in the context.
    2. If the context does not contain enough information to answer, say so explicitly.
       Do NOT guess, infer, or use general legal knowledge.
    3. If context passages are in a different language from the question, translate only
       the parts you cite — do not paraphrase beyond what the text says.
    4. Respond in the SAME language as the user's question.

    REQUIRED RESPONSE STRUCTURE
    Use the section labels that match your response language:

    **Direct Answer / Réponse directe / الإجابة المباشرة:**
    One or two sentences directly answering the question.

    **Applicable Articles / Articles applicables / الفصول المنطبقة:**
    For each relevant article: state the article number, source code, and summarise its content.
    Be exhaustive — list every relevant article found in the context.

    **Source(s):**
    Name(s) of the source document(s).

    **Explanation / Explication / شرح:**
    Brief legal explanation of how the cited articles answer the question.
    Highlight any important conditions, exceptions, or cross-references found in the context.

    ── Context ──────────────────────────────────────────────────────────────────
    {context}
""").strip()

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{input}"),
])

_ARTICLE_RE = re.compile(r"\b\d+\b")


def _is_complex(question: str) -> bool:
    if _ARTICLE_RE.search(question):
        return False
    return len(question.split()) >= 4


# ── Vectorstore management ───────────────────────────────────────────────────
def _refuse_frozen(db_dir) -> None:
    from config import DB_V11
    if Path(db_dir).resolve() == Path(DB_V11).resolve():
        raise RuntimeError(f"{DB_V11.name} is the frozen v1.1/v2 index: it is never rebuilt or written")


def split_long(text: str, length, max_tokens: int) -> list[str]:
    """Pieces of text, cut at line ends (else at spaces), each with length(piece) <= max_tokens when possible."""
    pieces, cur = [], ""
    units = []
    for line in text.split("\n"):
        if length(line) <= max_tokens:
            units.append(line)
        else:                                   # a line longer than the limit: cut at spaces
            words, part = line.split(" "), ""
            for w in words:
                if part and length(part + " " + w) > max_tokens:
                    units.append(part)
                    part = w
                else:
                    part = (part + " " + w) if part else w
            if part:
                units.append(part)
    for u in units:
        cand = (cur + "\n" + u) if cur else u
        if cur and length(cand) > max_tokens:
            pieces.append(cur)
            cur = u
        else:
            cur = cand
    if cur.strip():
        pieces.append(cur)
    return [p for p in pieces if p.strip()]


def index_entries(chunks: list[Document], length, max_tokens: int | None = None,
                  contextual: bool | None = None, subchunk: bool | None = None) -> list[dict]:
    """v1.2 index entries: [{"id", "document", "metadata"}]. document (embedded and indexed for BM25) =
    "<context line>\\n<article text>"; metadata keeps the plain article text ("article_text"), the context line
    ("context") and the parent article id ("parent_id"; "chunk_id" is the parent id too, so every consumer keyed on
    chunk_id sees the article). An article whose document is longer than max_tokens (length = embedding tokens) is
    split into sub-chunks "<chunk_id>#k" (metadata "part" k of "parts", "piece_text")."""
    import config as C
    from splitter import context_line
    max_tokens = C.SUBCHUNK_MAX_TOKENS if max_tokens is None else max_tokens
    contextual = C.INDEX_CONTEXT if contextual is None else contextual
    subchunk = C.SUBCHUNK_LONG if subchunk is None else subchunk
    out = []
    for c in chunks:
        cid = c.metadata["chunk_id"]
        ctx = context_line(c.metadata) if contextual else ""
        head = ctx + "\n" if ctx else ""
        meta = {**c.metadata, "article_text": c.page_content, "context": ctx, "parent_id": cid, "part": 0, "parts": 1}
        if subchunk and length(head + c.page_content) > max_tokens:
            pieces = split_long(c.page_content, lambda t: length(head + t), max_tokens)
        else:
            pieces = [c.page_content]
        if len(pieces) == 1:
            out.append({"id": cid, "document": head + c.page_content, "metadata": meta})
            continue
        for k, p in enumerate(pieces, 1):
            out.append({"id": f"{cid}#{k}", "document": head + p,
                        "metadata": {**meta, "part": k, "parts": len(pieces), "piece_text": p}})
    return out


def build_index_v12(folder: Path = DATA_DIR, db_dir: Path | str | None = None, batch: int = 64) -> dict:
    """Build db_v1_2 (config.DB_V12): split with headings moved forward, index_entries, embed. Resumable (entries
    whose id is stored are skipped). Never writes db_v1_1. -> build facts."""
    import time
    import config as C
    db_dir = Path(db_dir or C.DB_V12)
    _refuse_frozen(db_dir)
    t0 = time.perf_counter()
    chunks = split_by_article(load_all_documents(folder), headings_forward=C.SPLIT_HEADINGS_FORWARD)
    emb = get_embeddings()
    tok = emb._client.tokenizer
    length = lambda t: len(tok(t, add_special_tokens=True)["input_ids"])
    entries = index_entries(chunks, length)
    lens = [length(e["document"]) for e in entries]
    t_split = time.perf_counter() - t0
    store = Chroma(persist_directory=str(db_dir), embedding_function=emb)
    done = set(store._collection.get(include=[])["ids"])
    # embedded in order of length, so a batch is padded to texts of about the same length (CPU time)
    todo = [e for _, e in sorted(zip(lens, entries), key=lambda x: x[0]) if e["id"] not in done]
    logger.info("v1.2 index: %d entries (%d articles), %d to embed -> %s", len(entries), len(chunks), len(todo), db_dir)
    t1 = time.perf_counter()
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        store._collection.upsert(ids=[e["id"] for e in part], documents=[e["document"] for e in part],
                                 embeddings=emb.embed_documents([e["document"] for e in part]),
                                 metadatas=[e["metadata"] for e in part])
        logger.info("  %d / %d", min(i + batch, len(todo)), len(todo))
    t_embed = time.perf_counter() - t1
    return {"articles": len(chunks), "entries": len(entries), "embedded_this_run": len(todo),
            "split_articles": len({e["metadata"]["parent_id"] for e in entries if e["metadata"]["parts"] > 1}),
            "sub_chunks": sum(e["metadata"]["parts"] > 1 for e in entries),
            "entries_over_max_tokens": sum(n > C.SUBCHUNK_MAX_TOKENS for n in lens), "max_tokens": max(lens),
            "chunks_with_headings_moved": sum(bool(c.metadata.get("headings_moved")) for c in chunks),
            "headings_moved": sum(c.metadata.get("headings_moved", 0) for c in chunks),
            "articles_without_hierarchy": sum(not c.metadata.get("hierarchy") for c in chunks),
            "seconds_split_and_count": round(t_split, 1), "seconds_embed": round(t_embed, 1),
            "embedding_model": EMBEDDING_MODEL}


def build_vectorstore(folder: Path = DATA_DIR, db_dir: Path | str | None = None,
                      contextual: bool | None = None, batch: int = 250) -> Chroma:
    """Chunk the PDFs and embed them into Chroma at db_dir (default config.DB_DIR).
    Resumable: chunks whose id is already stored are skipped, so an interrupted build can simply be re-run.
    contextual: prepend context_prefix() to the text that is embedded (the stored text is unchanged).
    With config.INDEX_V12 the db_v1_2 build (build_index_v12) is run instead. db_v1_1 is never written."""
    import config as C
    if C.INDEX_V12 and db_dir is None:
        build_index_v12(folder)
        return load_vectorstore()
    contextual = CONTEXTUAL_CHUNKS if contextual is None else contextual
    db_dir = str(db_dir or DB_DIR)
    _refuse_frozen(db_dir)
    chunks = split_by_article(load_all_documents(folder), headings_forward=False)
    store = Chroma(persist_directory=db_dir, embedding_function=get_embeddings())
    done = set(store._collection.get(include=[])["ids"])
    todo = [c for c in chunks if c.metadata["chunk_id"] not in done]
    logger.info("Embedding %d new chunks (%d already stored) -> '%s' (contextual=%s)", len(todo), len(done), db_dir, contextual)
    emb = get_embeddings()
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        texts = [c.page_content for c in part]
        to_embed = [context_prefix(c.metadata) + "\n" + t for c, t in zip(part, texts)] if contextual else texts
        store._collection.upsert(
            ids=[c.metadata["chunk_id"] for c in part],
            documents=texts,
            embeddings=emb.embed_documents(to_embed),
            metadatas=[c.metadata for c in part],
        )
        logger.info("  %d / %d", min(i + batch, len(todo)), len(todo))
    return store


def load_vectorstore() -> Chroma:
    vs = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=get_embeddings(),
    )
    _VS["vs"] = vs
    return vs


def step_rewrite(question: str) -> str:
    return rewrite_query(question)


def step_query(question: str) -> str:
    """The French search query. Raw-first: French is used as written (no LLM call); Arabic and English are
    translated faithfully. With RAW_FIRST off this is the v1 LLM rewrite."""
    if not RAW_FIRST:
        return step_rewrite(question)
    if detect_arabic(question) or detect_english(question):
        return translate_to_french(question)
    return question


def step_subqueries_if_enabled(question: str, french_query: str) -> list[str]:
    return step_subqueries(french_query) if (USE_SUBQUERIES and _is_complex(question)) else []


def step_subqueries(french_query: str, n: int = SUBQUERY_N) -> list[str]:
    return generate_subqueries(french_query, n=n)


def step_retrieve(
    vectorstore: Chroma,
    question: str,
    french_query: str,
    sub_queries: list[str],
    law_type: str | None,
) -> list[Document]:
    """Candidate pool. Direct article lookup: v1.1 = any number in the question, its articles REPLACE the hybrid
    pool; config.DIRECT_LOOKUP_STRICT = only numbers after an article marker (retriever.article_numbers);
    config.DIRECT_LOOKUP_UNION = the articles with those numbers (ordered by similarity to the French query) are
    added in front of the hybrid pool."""
    import config as C
    from retriever import _direct_lookup, MMR_K, article_numbers, _deduplicate
    if C.DIRECT_LOOKUP_STRICT:
        numbers = article_numbers(question)
    else:
        article_match = _ARTICLE_RE.search(question)
        numbers = [article_match.group(0)] if article_match else []
    direct: list[Document] = []
    if numbers and not C.DIRECT_LOOKUP_UNION:
        docs = _direct_lookup(vectorstore, numbers[0], law_type, MMR_K)
        if docs:
            logger.debug("Direct article lookup hit: %s", numbers[0])
            return docs
    elif numbers:
        for n in numbers:
            direct += _direct_lookup(vectorstore, n, law_type, MMR_K, relax=False, query=french_query)

    all_queries = [french_query] + sub_queries
    pool = hybrid_retrieve(vectorstore, all_queries, question, law_type)
    return _deduplicate(direct + pool) if direct else pool


# ── evidence assembly (v2.1) ─────────────────────────────────────────────────
_VS: dict = {"vs": None}


def evidence_docs(docs: list[Document], vectorstore: Chroma | None = None) -> list[Document]:
    """The documents passed on as evidence: docs, then (config.XREF_EXPAND) the articles they reference
    (retriever.expand_cross_references; caps XREF_PER_ARTICLE, XREF_TOTAL, XREF_RANGE_MAX)."""
    import config as C
    if not C.XREF_EXPAND or not docs:
        return list(docs)
    from retriever import expand_cross_references, parse_references
    if not any(parse_references(d.page_content) for d in docs):     # no reference: the index is not opened
        return list(docs)
    vs = vectorstore or _VS["vs"] or load_vectorstore()
    added = expand_cross_references(vs, docs)
    if added and C.XREF_TOKEN_CAP is not None:
        added = cap_added(added, C.XREF_TOKEN_CAP, _token_len())
    return list(docs) + added


def _token_len():
    tok = get_embeddings()._client.tokenizer
    return lambda t: len(tok(t, add_special_tokens=True)["input_ids"])


def cap_added(added: list[Document], cap: int, length) -> list[Document]:
    """At most cap tokens (length = embedding tokens of the evidence text, header lines included) for the added
    cross-referenced articles of one question, in order: an article that does not fit is cut (text only, never its
    header lines) to the tokens left; once no text fits, the following articles are not added. The citing evidence
    articles are never cut."""
    out, left = [], cap
    for d in added:
        full = length(evidence_text(d))
        if full <= left:
            out.append(d)
            left -= full
            continue
        head = length(evidence_text(Document(page_content="", metadata=d.metadata)))
        if head >= left:
            break
        lo, hi, text = 0, len(d.page_content), d.page_content
        while lo < hi:                                    # longest prefix whose evidence text fits
            mid = (lo + hi + 1) // 2
            cand = Document(page_content=text[:mid].rstrip() + " […]", metadata={**d.metadata, "parts": 1})
            if length(evidence_text(cand)) <= left:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            break
        out.append(Document(page_content=text[:lo].rstrip() + " […]",
                            metadata={**d.metadata, "parts": 1, "xref_truncated": True}))
        break
    return out


def _window(text: str, match: str | None, cap: int) -> str:
    """At most cap characters of a long article: its first line, then the part around the matched sub-chunk."""
    if len(text) <= cap:
        return text
    first = text.split("\n", 1)[0][:200]
    pos = text.find(match[:200]) if match else -1
    if pos <= len(first):
        return text[:cap - 2] + " […]"
    room = cap - len(first) - 8
    start = max(len(first), min(pos - room // 4, len(text) - room))
    return f"{first}\n[…] {text[start:start + room]} […]"


def context_header(meta: dict, mode=None) -> str:
    """What the "Contexte :" line shows (config.EVIDENCE_CONTEXT_HEADER): "full" = the indexed context line
    (code > headings), "deepest" = the deepest heading only (the code name when the article has no heading),
    "none" = nothing. True / False are read as "full" / "none". Empty for a v1.1 document (no context)."""
    import config as C
    mode = C.EVIDENCE_CONTEXT_HEADER if mode is None else mode
    mode = {True: "full", False: "none", None: "none"}.get(mode, mode)
    ctx = meta.get("context") or ""
    if not ctx or mode == "none":
        return ""
    if mode == "deepest":
        h = meta.get("hierarchy") or ""
        return h.split(" > ")[-1] if h else ctx
    return ctx


def evidence_text(doc: Document) -> str:
    """Evidence text: "Article cité par l'article N" (an added cross-referenced article), "Contexte : <code > headings>"
    (config.EVIDENCE_CONTEXT_HEADER, v1.2 index), then the article text (config.EVIDENCE_TEXT_CAP characters of a long
    article, around the matched sub-chunk)."""
    import config as C
    m = doc.metadata
    lines = []
    if m.get("cited_by"):
        lines.append(f"Article cité par l'article {m['cited_by']}")
    ctx = context_header(m)
    if ctx:
        lines.append(f"Contexte : {ctx}")
    text = doc.page_content
    if C.EVIDENCE_TEXT_CAP and m.get("parts", 1) > 1:
        text = _window(text, m.get("match_text"), C.EVIDENCE_TEXT_CAP)
    return "\n".join(lines + [text])


def step_rerank(french_query: str, candidates: list[Document], top_k: int = RERANK_TOP_K) -> list[Document]:
    if not RERANK_ENABLED:
        return candidates[:top_k]
    return rerank(french_query, candidates, top_k)


def step_generate(question: str, docs: list[Document]) -> str:
    context_blocks = [
        f"[{doc.metadata.get('source', 'unknown')} — Article {doc.metadata.get('article', '?')}]\n"
        f"{doc.page_content}"
        for doc in docs
    ]
    context = "\n\n---\n\n".join(context_blocks)
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)
    return (_PROMPT | llm | StrOutputParser()).invoke({
        "context": context,
        "input":   question,
    })


def ask_question(
    vectorstore: Chroma,
    question: str,
    law_type: str | None = None,
) -> tuple[str, list[Document]]:
    is_arabic  = detect_arabic(question)
    is_english = detect_english(question)

    french_query = step_query(question)
    logger.info("Search query: %r", french_query)

    sub_queries = step_subqueries_if_enabled(question, french_query)
    if sub_queries:
        logger.info("Sub-queries: %s", sub_queries)

    candidates = step_retrieve(vectorstore, question, french_query, sub_queries, law_type)

    if not candidates:
        if is_arabic:
            msg = "لم يتم العثور على معلومات ذات صلة في الوثائق."
        elif is_english:
            msg = "No relevant information found in the legal documents."
        else:
            msg = "Aucune information pertinente trouvée dans les documents juridiques."
        return msg, []
    docs = step_rerank(french_query, candidates)
    answer = step_generate(question, docs)

    return answer, docs
