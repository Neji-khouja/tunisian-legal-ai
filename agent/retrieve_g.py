import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import pipeline, retriever, config

from pipeline import step_query, step_subqueries_if_enabled, step_retrieve, step_rerank
from retriever import _fuse, fuse_pools
from retrieve import get_vs

def retrieve_g(question, target_codes, info=None):
    """Variant G as measured in eval_plan_vs_raw.py / eval_loop.py --start g: baseline retrieval on the raw
    question (A2), plus, when Analyze gave exactly one code, the same baseline filtered on that code, fused
    with A2 (soft hint). The French query and sub-queries are computed once and shared by both searches.
    config.NORMALIZE_QUERY == "v3": see _retrieve_v3 (info, a dict, receives the normalization record)."""
    import config as C
    k = C.AGENT_TOP_K
    vs = get_vs()
    french_q = step_query(question)
    subs = step_subqueries_if_enabled(question, french_q)
    if getattr(C, "NORMALIZE_QUERY", "off") == "v3":
        return _retrieve_v3(vs, question, french_q, subs, target_codes, info)

    def baseline(law_type):
        cands = step_retrieve(vs, question, french_q, subs, law_type)
        return (step_rerank(french_q, cands, k) if cands else [])[:k]

    a2 = baseline(None)
    if len(target_codes) == 1:
        return _fuse([baseline(target_codes[0]), a2])[:k]
    return a2


def _retrieve_v3(vs, question, french_q, subs, target_codes, info):
    """V3: the French query is normalized once (agent/normalize.py); each
    baseline pool = RRF of the pool of the French query and the pool of the normalized query
    (retriever.fuse_pools), reranked on the normalized query. A failed, empty or invalid normalization, or any
    error, falls back to the French query alone: exactly the "off" retrieval. Only this initial query of the turn
    is normalized (the reformulate step's queries go through retrieve.retrieve_for_plan, unchanged)."""
    import config as C
    from normalize import normalize_for_retrieval
    k = C.AGENT_TOP_K
    norm_q, rec = normalize_for_retrieval(french_q)
    if info is not None:
        info.update(rec)

    def baseline(law_type):
        pool_o = step_retrieve(vs, question, french_q, subs, law_type)
        if rec["fell_back"]:
            cands, rq = pool_o, french_q
        else:
            cands, rq = fuse_pools(pool_o, step_retrieve(vs, question, norm_q, [], law_type)), norm_q
        return (step_rerank(rq, cands, k) if cands else [])[:k]

    a2 = baseline(None)
    if len(target_codes) == 1:
        return _fuse([baseline(target_codes[0]), a2])[:k]
    return a2
