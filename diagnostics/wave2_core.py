"""Wave 2 (query normalization) pure helpers, fixed in the pre-registration commit (diagnostics/WAVE2_PREREG.md):
reciprocal rank fusion of two candidate pools, the fallback of a failed normalization, and the decision rule.
No model, no index, no LLM.
"""
VARIANTS = ("V0", "V1", "V2", "V3")
CANDIDATES = ("V1", "V2", "V3")
RRF_K = 60                      # config.RRF_K
MIN_NOISY_GAIN = 3              # (i)  net Hit@6 gain over V0 on the noisy set
MAX_LOST_STAGES = 1             # (ii) items lost versus V0 on Stage 1 + Stage 2
MAX_LOST_LANGUAGE = 1           # (iii) items lost versus V0 in each language


def rrf_merge(lists, key, k=RRF_K) -> list:
    """score(d) = sum over lists of 1 / (k + rank); ties keep the order of first appearance (list 1 first)."""
    scores, first, order = {}, {}, []
    for lst in lists:
        for rank, d in enumerate(lst, 1):
            kk = key(d)
            if kk not in scores:
                scores[kk], first[kk] = 0.0, d
                order.append(kk)
            scores[kk] += 1.0 / (k + rank)
    pos = {kk: i for i, kk in enumerate(order)}
    return [first[kk] for kk in sorted(order, key=lambda x: (-scores[x], pos[x]))]


def fused_pool(pool_original, pool_normalized, key) -> list:
    """V2/V3 pool: RRF of the two pools, cut to the size of the larger one (the reranker scores no more candidates
    than for one query)."""
    return rrf_merge([pool_original, pool_normalized], key)[:max(len(pool_original), len(pool_normalized))]


def query_for(entry, input_query) -> tuple:
    """-> (normalized query to use, fell back?). A missing, failed (status != ok) or empty normalization falls back to
    the input query, so V1 = V0 and V2 = V3 = V0 for that item."""
    if entry and entry.get("status") == "ok" and (entry.get("legal_query") or "").strip():
        return entry["legal_query"], False
    return input_query, True


def _hits(recs) -> dict:
    return {r["id"]: r for r in recs}


def conditions(v0, v, extra_tokens=0) -> dict:
    """The three conditions of the rule for one candidate (records with id, set, language, hit6)."""
    a, b = _hits(v0), _hits(v)
    ids = [i for i in a if i in b]
    lost = [i for i in ids if a[i]["hit6"] and not b[i]["hit6"]]
    gained = [i for i in ids if b[i]["hit6"] and not a[i]["hit6"]]
    noisy = [i for i in ids if a[i].get("set") == "N"]
    noisy_net = sum(b[i]["hit6"] - a[i]["hit6"] for i in noisy)
    lost_stages = [i for i in lost if a[i].get("set") in ("S1", "S2")]
    langs = sorted({a[i].get("language") for i in ids}, key=str)
    lost_lang = {l: [i for i in lost if a[i].get("language") == l] for l in langs}
    c = {"total_hit6": sum(b[i]["hit6"] for i in ids), "v0_total_hit6": sum(a[i]["hit6"] for i in ids),
         "gained": gained, "lost": lost, "extra_tokens": extra_tokens,
         "noisy_net": noisy_net, "i_noisy_gain_ge_3": noisy_net >= MIN_NOISY_GAIN,
         "lost_stage1_2": lost_stages, "ii_lost_stages_le_1": len(lost_stages) <= MAX_LOST_STAGES,
         "lost_by_language": lost_lang, "iii_lost_each_language_le_1": all(len(x) <= MAX_LOST_LANGUAGE for x in lost_lang.values())}
    c["passes"] = c["i_noisy_gain_ge_3"] and c["ii_lost_stages_le_1"] and c["iii_lost_each_language_le_1"]
    return c


def decide(recs: dict, extra_tokens: dict) -> dict:
    """recs: variant -> records; extra_tokens: variant -> extra LLM tokens (V0 = 0).
    The candidate is the variant among V1..V3 with the highest total Hit@6; ties -> fewer extra tokens, then the
    lower-numbered variant. It is adopted only if it passes (i), (ii) and (iii); otherwise none is adopted (the
    runner-up is not considered)."""
    cond = {v: conditions(recs["V0"], recs[v], extra_tokens.get(v, 0)) for v in CANDIDATES if v in recs}
    if not cond:
        return {"conditions": {}, "candidate": None, "selected": "none"}
    cand = sorted(cond, key=lambda v: (-cond[v]["total_hit6"], cond[v]["extra_tokens"], CANDIDATES.index(v)))[0]
    return {"conditions": cond, "candidate": cand, "selected": cand if cond[cand]["passes"] else "none"}
