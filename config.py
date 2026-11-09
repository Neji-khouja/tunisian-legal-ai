"""Central configuration for the Tunisian Legal RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent   # absolute, so scripts started from agent/ open the same index
DATA_DIR = ROOT_DIR / "data"
DB_V11   = ROOT_DIR / "db_v1_1"   # v1.1 / v2 index: never rebuilt or written (v1 index stays untouched in ./db)
DB_V12   = ROOT_DIR / "db_v1_2"   # v2.1 index: headings moved forward, context in the indexed text, sub-chunks

# ── v2.1 wave 1 (branch agent-v2.1). Every switch can be turned off; all False = the v2 retrieval on db_v1_1.
INDEX_V12 = True                 # use db_v1_2 (built with SPLIT_HEADINGS_FORWARD, INDEX_CONTEXT, SUBCHUNK_LONG)
# build time (pipeline.build_vectorstore, splitter.split_by_article)
SPLIT_HEADINGS_FORWARD = True    # a book/title/chapter/section heading belongs to the articles after it
INDEX_CONTEXT          = True    # embedded and BM25 text = "<code> > <headings>\n<article>"; plain text kept in metadata
SUBCHUNK_LONG          = True    # articles over the embedding limit are split into sub-chunks (parent id kept)
SUBCHUNK_MAX_TOKENS    = 500     # embedding tokens per indexed text, special tokens included (Solon limit 512)
# query time
DIRECT_LOOKUP_STRICT   = True    # direct article lookup only for a number after an article marker (art., article, الفصل)
DIRECT_LOOKUP_UNION    = True    # direct-lookup articles are added to the hybrid pool instead of replacing it
XREF_EXPAND            = True    # articles referenced by an evidence article are added to the evidence
XREF_PER_ARTICLE       = 2       # added articles per evidence article
XREF_TOTAL             = 6       # added articles per question
XREF_RANGE_MAX         = 4       # "articles N à M" is expanded only when it names at most this many articles
EVIDENCE_CONTEXT_HEADER = "deepest"   # "Contexte : ..." line of each evidence text: "full" (code > all headings),
                                      # "deepest" (deepest heading only; the code name without headings), "none"
XREF_TOKEN_CAP         = 300     # embedding tokens of the added cross-referenced articles per question (None = no cap);
                                 # 300 = smallest cap losing <= 1 dev item vs no cap (diagnostics/EVIDENCE_BUDGET.md)
EVIDENCE_TEXT_CAP      = 3000    # characters of a long article shown in the evidence (window around the match)
BM25_TOKENIZER         = "whitespace"   # "whitespace" (v1.1) | "lower" | "fold" | "elision" | "stem" (retriever.py)
NORMALIZE_QUERY        = "off"   # "off" | "v3": the initial query of a turn is also normalized (agent/normalize.py);
                                 # pools of the original and normalized queries fused by RRF, reranked on the
                                 # normalized query (diagnostics/WAVE2_PREREG.md V3). Reformulated queries: never.

DB_DIR   = DB_V12 if INDEX_V12 else DB_V11

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

EMBEDDING_MODEL = "OrdalieTech/Solon-embeddings-large-0.1"
LLM_MODEL       = "openai/gpt-oss-120b"
RERANKER_MODEL  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Embedding input. CONTEXTUAL_CHUNKS: prepend the code/text name and article heading to the text that is
# EMBEDDED (the stored text, BM25 and the reranker still see the original chunk). Build-time only.
CONTEXTUAL_CHUNKS  = False
EMBED_DOC_PREFIX   = ""    # e.g. "passage: " for the e5 family
EMBED_QUERY_PREFIX = ""    # e.g. "query: "
EMBED_MAX_SEQ_LENGTH = None  # cap tokens per chunk (None = model default; Solon uses 512)

# Retrieval knobs
MMR_K        = 20   # semantic candidates per query
BM25_K       = 15   # BM25 candidates (primary query only)
RERANK_TOP_K = 6    # docs returned to the user after reranking
RERANK_ENABLED    = True   # False: keep the retrieval order (ablation)
RERANK_POOL       = 20     # max candidates sent to the reranker (None = all retrieved)
RERANK_MAX_LENGTH = 256    # tokens per (query, passage) pair
RERANK_QUANTIZE   = True   # int8 dynamic quantization of the cross-encoder (CPU speed-up)
RERANK_BATCH      = 16
FUSION            = "rrf"  # how semantic / BM25 / original-language lists are merged before the pool cut: "rrf" | "roundrobin"
RRF_K             = 60     # reciprocal rank fusion constant: score = sum 1 / (RRF_K + rank)
SUBQUERY_N   = 2    # sub-queries generated for complex (non-article) questions

# Query preparation. RAW_FIRST: French questions are searched as written (no LLM call); Arabic/English are
# translated to French faithfully (no enrichment). USE_SUBQUERIES: LLM sub-queries (v1 behaviour, off).
RAW_FIRST                   = True
USE_SUBQUERIES              = False
TRANSLATE_REASONING_EFFORT  = "low"   # translation needs no deliberation; keeps Groq token use small

# Maps law_type substrings to match against the lowercased PDF filename
LAW_TYPE_MAP: dict[str, list[str]] = {
    "procedure_penale":       ["procedurepenal"],
    "procedure_civile":       ["procedurecivile"],
    "penal":                  ["penal", "crim"],
    "constitution":           ["constitution", "recac", "freeli"],
    "travail":                ["travail"],
    "commerce":               ["commerce"],
    "societes":               ["societe"],
    "obligations_contrats":   ["coc"],
    "statut_personnel":       ["statut", "personnel"],
    "protection_enfant":      ["enfant"],
    "nationalite":            ["nationalite"],
    "arbitrage":              ["arbitrage"],
    "droit_international_prive": ["dip"],
    "droits_reels":           ["dtree", "dreel"],
}

LAW_DISPLAY_NAMES: dict[str | None, str] = {
    None:                        "All Documents",
    "constitution":              "Constitution",
    "penal":                     "Code Pénal",
    "procedure_penale":          "Code de Procédure Pénale",
    "procedure_civile":          "Code de Procédure Civile et Commerciale",
    "obligations_contrats":      "Code des Obligations et Contrats (COC)",
    "commerce":                  "Code de Commerce",
    "societes":                  "Code des Sociétés Commerciales",
    "travail":                   "Code du Travail",
    "statut_personnel":          "Code du Statut Personnel",
    "protection_enfant":         "Code de Protection de l'Enfant",
    "droit_international_prive": "Droit International Privé",
    "droits_reels":              "Droits Réels",
    "arbitrage":                 "Code de l'Arbitrage",
    "nationalite":               "Code de la Nationalité",
}

# PDF cleaning patterns
HEADER_PATTERNS: list[str] = [
    r"Page\s+\d+\s+sur\s+\d+",
    r"Imp[re]+imerie\s+Officielle\s+de\s+la\s+R[eé]publique\s+Tunisienne",
    r"Accès aux lois[^\n]*",
    r"WWW\.LEGISLATION-SECURITE\.TN",
    r"www\.legislation-securite\.tn",
    r"Base de donn[eé]es\s*\n?\s*LA LEGISLATION[^\n]*",
    r"LA LEGISLATION DU SECTEUR[^\n]*",
    r"J\.?O\.?R\.?T\.?\s+n°[^\n]{0,80}",
]

# Used for documentation / potential future augmentation. Not injected into
# queries use rewrite_query() from translate.py for proper translation.
LEGAL_TRANSLATIONS: dict[str, str] = {
    "سرقة":   "vol soustraction frauduleuse",
    "قتل":    "meurtre homicide",
    "اغتصاب": "viol agression sexuelle",
    "احتيال": "escroquerie fraude",
    "رشوة":   "corruption soudoiement",
    "اعتداء": "agression voies de fait",
    "حبس":    "emprisonnement prison",
    "غرامة":  "amende pénalité",
    "إعدام":  "peine de mort",
    "طلاق":   "divorce répudiation",
    "زواج":   "mariage union conditions consentement capacité",
    "ميراث":  "succession héritage",
    "عقد":    "contrat obligation",
    "فسخ":    "résiliation annulation",
    "تعويض":  "indemnisation réparation",
    "موقوف":  "détenu arrestation garde à vue droits",
    "حقوق":   "droits garanties libertés fondamentaux",
    "شروط":   "conditions exigences capacité",
    "اعتقال": "arrestation garde à vue",
    "توقيف":  "détention provisoire garde à vue",
    "محكمة":  "tribunal juridiction",
    "دفاع":   "défense avocat",
    "حرية":   "liberté droit fondamental",
    "عمل":    "travail emploi",
    "عامل":   "travailleur salarié",
    "طفل":    "enfant mineur",
    "جنسية":  "nationalité citoyenneté",
    "شركة":   "société entreprise",
    "عقوبة":  "peine sanction",
    "جريمة":  "crime délit infraction",
    "تحقيق":  "enquête instruction",
    "شاهد":   "témoin témoignage",
    "ضحية":   "victime plaignant",
}
