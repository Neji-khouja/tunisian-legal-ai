"""Pure helpers of the retrieval diagnostic: ranks, funnel classes, summaries, Hit@k, cross-reference patterns.
No model, no index: tested on stub lists by diagnostics/test_funnel.py."""
import re
from collections import Counter

CLASSES = ["top6", "rerank_7_10", "rerank_11_20", "fused_rank_gt20", "absent_from_pool"]
CLASS_LABELS = {"top6": "found in top 6 (reranked)", "rerank_7_10": "in pool, reranked 7-10",
                "rerank_11_20": "in pool, reranked 11-20", "fused_rank_gt20": "fused rank > 20 (cut before the reranker)",
                "absent_from_pool": "absent from the pool (not cut: never a candidate)"}
FLAGS = ["absent_dense_only", "absent_bm25_only", "absent_both", "present_both"]


def rank_of(keys: list, gold: set) -> int | None:
    """1-based rank of the first list entry whose key is a gold article, else None."""
    for i, k in enumerate(keys, 1):
        if k in gold:
            return i
    return None


def classify(rank_reranked: int | None, rank_fused_full: int | None, path: str) -> str:
    """rank_reranked: rank of the gold in the cross-encoder order of the pool actually used (None = not in it);
    rank_fused_full: rank in the fused, deduplicated list before the pool cut (hybrid path only)."""
    if rank_reranked is not None:
        if rank_reranked <= 6:
            return "top6"
        return "rerank_7_10" if rank_reranked <= 10 else "rerank_11_20"
    if path == "hybrid" and rank_fused_full is not None:
        return "fused_rank_gt20"
    return "absent_from_pool"


def list_flag(rank_dense: int | None, rank_bm25: int | None) -> str:
    if rank_dense is None and rank_bm25 is None:
        return "absent_both"
    if rank_dense is None:
        return "absent_dense_only"
    if rank_bm25 is None:
        return "absent_bm25_only"
    return "present_both"


def hit_at(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def summarize(records: list) -> dict:
    """Counts and percentages of the funnel classes and list flags, Hit@6/8/10, path counts, over analysed records."""
    n = len(records)
    pct = lambda c: round(100 * c / n, 1) if n else None
    classes = Counter(r["class"] for r in records)
    flags = Counter(r["list_flag"] for r in records)
    paths = Counter(r["path"] for r in records)
    return {"n": n,
            "classes": {c: {"n": classes[c], "pct": pct(classes[c])} for c in CLASSES},
            "flags": {f: {"n": flags[f], "pct": pct(flags[f])} for f in FLAGS},
            "paths": {p: {"n": paths[p], "pct": pct(paths[p])} for p in ("hybrid", "direct")},
            "hit": {k: {"n": sum(hit_at(r["rank_reranked"], k) for r in records), "pct": pct(sum(hit_at(r["rank_reranked"], k) for r in records))}
                    for k in (6, 8, 10)}}


# ---------------------------------------------------------------- cross-references

SUFFIX = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)"
NUM = rf"\d+(?:\s*-?\s*{SUFFIX})?"
HEADER = re.compile(rf"^\s*(?:article|art\.?)\s*{NUM}\s*(?:\.?\s*-|\.|:)?", re.I)
NUMBERED_FR = re.compile(rf"\b(?:articles?|art\.)\s+({NUM}(?:\s*(?:,|et|à|au|ou)\s*{NUM})*)", re.I)
RELATIVE_FR = re.compile(r"\barticles?\s+(pr[ée]c[ée]dente?s?|suivante?s?)(?!\w)", re.I)
VAGUE_FR = re.compile(r"\b(ci-dessus|ci-apr[èe]s|sus-?vis[ée]e?s?|pr[ée]cit[ée]e?s?)(?!\w)", re.I)
NUMBERED_AR = re.compile(r"(?:الفصل|الفصول|الفصلين)\s+(\d+)")
RELATIVE_AR = re.compile(r"الفصل\s+(السابق|الموالي|المتقدم)")
VAGUE_AR = re.compile(r"(أعلاه|المذكور|المشار إليه|السالف الذكر)")
KINDS = ["numbered", "relative", "vague"]


def label(raw: str) -> str:
    """'220 bis' -> '220-bis' (the chunk-id spelling)."""
    m = re.match(rf"(\d+)\s*-?\s*({SUFFIX})?", raw.strip(), re.I)
    return m.group(1) + (f"-{m.group(2).lower()}" if m.group(2) else "")


def body(text: str) -> str:
    """The chunk text without its own 'Article N.-' header."""
    return HEADER.sub("", text or "", count=1)


def references(text: str) -> dict:
    """{kind: [matched strings]} for the reference patterns found in a chunk body (French and Arabic)."""
    b = body(text)
    return {"numbered": [m.group(0) for m in NUMBERED_FR.finditer(b)] + [m.group(0) for m in NUMBERED_AR.finditer(b)],
            "relative": [m.group(0) for m in RELATIVE_FR.finditer(b)] + [m.group(0) for m in RELATIVE_AR.finditer(b)],
            "vague": [m.group(0) for m in VAGUE_FR.finditer(b)] + [m.group(0) for m in VAGUE_AR.finditer(b)]}


def resolve(text: str, own_label: str, max_span: int = 50) -> set:
    """Article labels a chunk refers to, in its own code: explicit numbers and ranges 'N à M' (expanded when the span
    is at most max_span), 'article précédent' = own base number - 1, 'article suivant' = + 1 (the harness's
    implicit-reference rule, eval/pdf_factcheck.py). Vague references (ci-dessus, susvisé...) are not resolvable."""
    b = body(text)
    out = set()
    for m in NUMBERED_FR.finditer(b):
        parts = re.split(r"\s*(,|et|à|au|ou)\s*", m.group(1))
        nums, seps = parts[0::2], parts[1::2]
        for i, n in enumerate(nums):
            out.add(label(n))
            if i and seps[i - 1] in ("à", "au"):
                lo, hi = int(re.match(r"\d+", nums[i - 1]).group(0)), int(re.match(r"\d+", n).group(0))
                if 0 < hi - lo <= max_span:
                    out |= {str(x) for x in range(lo, hi + 1)}
    for m in NUMBERED_AR.finditer(b):
        out.add(m.group(1))
    base = re.match(r"\d+", own_label or "")
    if base:
        for m in RELATIVE_FR.finditer(b):
            out.add(str(int(base.group(0)) + (-1 if m.group(1).lower().startswith("pr") else 1)))
        for m in RELATIVE_AR.finditer(b):
            out.add(str(int(base.group(0)) + (1 if m.group(1) == "الموالي" else -1)))
    out.discard(own_label)
    return out
