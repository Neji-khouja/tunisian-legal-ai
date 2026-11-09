"""Retrieval-only evaluation: gold rank and funnel class per question, Hit@6/8/10 with bootstrap CIs, and a paired
comparison of two (index, configuration) pairs. No LLM call, ever.

    python eval/retrieval_only.py --items ITEMS.json --translations diagnostics/translations.json \
        --index-a db_v1_1 --config-a baseline [--index-b db_v1_2 --config-b v1.2] [--set-b KEY=VALUE ...] \
        [--out OUT.json] [--md OUT.md]
    python eval/retrieval_only.py --export-dev diagnostics/dev_items.json     # the 110 development items
    python eval/retrieval_only.py --items diagnostics/dev_items.json --index-a db_v1_2 --config-a v1.2 \
        --variants V0 V1 V2 V3 --normalized diagnostics/normalized_queries.json [--out OUT.json] [--md OUT.md]
        # wave 2 query normalization variants (diagnostics/WAVE2_PREREG.md); still no LLM call (cached outputs)

ITEMS.json: a list (or {"items": [...]}) of {"id", "language", "question", "gold", optional "noise_type", "set"};
"gold" as in the exam files ([{"source": "PenalFr.pdf", "article": "94", ...}]) or chunk ids ["PenalFr/s02/94"].
Gold matching: source stem + article label (run_answer_eval.key / gold_set).
Translations: Arabic/English questions (retriever.detect_arabic / detect_english) need a cached French query
(diagnostics/translations.json format, or {"<id>": "<French query>"}); a missing one, or one cached for another
question text, is reported and the item skipped.
Configurations: presets "baseline" (every v2.1 switch off = the v2 retrieval core) and "v1.2" (switches on), each
switch can be overridden with --set-a / --set-b KEY=VALUE (config.py names; VALUE is read as JSON, else a string).
Per item, with the real code: pool = pipeline.step_retrieve, reranker.rerank of the whole pool; dense, BM25 (and the
original-language dense list) give the funnel classes; evidence = first 6 + pipeline.evidence_docs.
Indexes are always opened from a temporary byte copy (a Chroma open writes a row in the sqlite file), so db_v1_1 and
db_v1_2 are never written. Summary: Hit@k with 95% bootstrap CIs (5000 resamples, seed 20261022, percentile, as
run_answer_eval.boot_ci) per set, language and noise_type; paired: B - A per item, bootstrap CI of the mean
difference, gained / lost items, exact two-sided sign test on the discordant items.
"""
import gc
import sys
import json
import math
import random
import shutil
import argparse
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "diagnostics")]

SEED = 20261022
N_BOOT = 5000
KS = (6, 8, 10)
CLASSES = ["top6", "rerank_7_10", "rerank_11_20", "fused_rank_gt20", "absent_from_pool"]
PRESETS = {
    "baseline": dict(DIRECT_LOOKUP_STRICT=False, DIRECT_LOOKUP_UNION=False, XREF_EXPAND=False, EVIDENCE_CONTEXT_HEADER="none",
                     SUBCHUNK_LONG=False, BM25_TOKENIZER="whitespace", XREF_TOKEN_CAP=None),
    "v1.2": dict(DIRECT_LOOKUP_STRICT=True, DIRECT_LOOKUP_UNION=True, XREF_EXPAND=True, EVIDENCE_CONTEXT_HEADER="deepest",
                 SUBCHUNK_LONG=True, BM25_TOKENIZER="whitespace", XREF_TOKEN_CAP=300),
}


# ---------------------------------------------------------------- inputs

def load_items(path) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    for it in items:
        missing = [k for k in ("id", "question", "gold") if k not in it]
        if missing:
            raise ValueError(f"item {it.get('id')!r}: missing {missing}")
    return items


def gold_keys(item) -> set:
    out = set()
    for g in item.get("gold") or []:
        if isinstance(g, str):
            out.add((g.split("/")[0], g.split("/")[-1].split("~")[0]))
        else:
            out.add((Path(str(g["source"])).stem, str(g["article"])))
    return out


def key(cid: str) -> tuple:
    return (cid.split("/")[0], cid.split("/")[-1].split("~")[0])


def load_translations(path) -> dict:
    """id -> (question the translation was made for or None, French query)."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        return {x["id"]: (x.get("query"), x["french_query"]) for x in data["items"]}
    return {k: (None, v) for k, v in data.items()}


def french_query(item, translations) -> tuple:
    """-> (French query or None, other language?, skip reason)."""
    from retriever import detect_arabic, detect_english
    q = item["question"]
    other = detect_arabic(q) or detect_english(q)
    if not other:
        return q, False, None
    if item["id"] not in translations:
        return None, True, "no cached translation (Arabic/English question)"
    for_q, fq = translations[item["id"]]
    if for_q is not None and for_q != q:
        return None, True, "cached translation was made for another question text"
    return fq, True, None


def export_dev(path):
    """The development items: Stage 1 and Stage 2 answerable exam items, noisy-set answerable items (exam files read only)."""
    exam = json.loads((ROOT / "eval" / "answer_set.json").read_text(encoding="utf-8"))
    stages = json.loads((ROOT / "eval" / "stages.json").read_text(encoding="utf-8"))
    noisy = json.loads((ROOT / "eval" / "noisy_set.json").read_text(encoding="utf-8"))
    out = []
    for s in ("1", "2"):
        ids = set(stages[f"stage{s}"])
        out += [{"set": f"S{s}", **{k: i.get(k) for k in ("id", "language", "question", "gold", "noise_type")}}
                for i in exam if i["id"] in ids and i["expected"] == "answer"]
    out += [{"set": "N", **{k: i.get(k) for k in ("id", "language", "question", "gold", "noise_type")}}
            for i in noisy if i.get("expected", "answer") == "answer"]
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}: {len(out)} items")


# ---------------------------------------------------------------- retrieval

def apply_config(preset: str, overrides: dict) -> dict:
    import config as C
    cfg = {**PRESETS[preset], **overrides}
    for k, v in cfg.items():
        if not hasattr(C, k):
            raise ValueError(f"unknown config switch {k}")
        setattr(C, k, v)
    return cfg


def retrieval_docs(vs, q, fq, other) -> dict:
    """The documents of one retrieval of the French query fq (question q decides the article numbers and, for an
    Arabic/English question, the original-language dense list), with the real code and config."""
    import config as C
    from pipeline import step_retrieve, _ARTICLE_RE
    from retriever import _semantic_search, _bm25_search, _get_bm25, _fuse, _deduplicate, _direct_lookup, article_numbers
    bm25 = _get_bm25(vs)
    dense = _semantic_search(vs, fq, None, k=C.MMR_K)
    bm = _bm25_search(bm25, fq, None, k=C.BM25_K) if bm25 else []
    orig = _semantic_search(vs, q, None, k=10) if other else []
    fused = _deduplicate(_fuse([dense, bm] + ([orig] if other else [])))
    nums = article_numbers(q) if C.DIRECT_LOOKUP_STRICT else ([m.group(0)] if (m := _ARTICLE_RE.search(q)) else [])
    if nums and not C.DIRECT_LOOKUP_UNION:
        path = "direct" if _direct_lookup(vs, nums[0], None, C.MMR_K) else "hybrid"
    else:
        path = "direct+hybrid" if nums else "hybrid"
    pool = step_retrieve(vs, q, fq, [], None)
    return {"dense": dense, "bm25": bm, "orig": orig, "fused": fused, "pool": pool, "path": path}


def _finish(vs, D, rerank_query) -> dict:
    from pipeline import evidence_docs
    from reranker import rerank
    reranked = rerank(rerank_query, D["pool"], top_k=len(D["pool"]))
    ev = evidence_docs(reranked[:6], vs)
    ids = lambda ds: [d.metadata["chunk_id"] for d in ds]
    return {**{n: ids(D[n]) for n in ("dense", "bm25", "orig", "fused", "pool")},
            "reranked": ids(reranked), "evidence": ids(ev), "path": D["path"]}


def real_backend(vs):
    """(question, French query, other language?) -> the lists of one retrieval, with the real code and config."""
    def run(q, fq, other):
        return _finish(vs, retrieval_docs(vs, q, fq, other), fq)
    return run


# ---------------------------------------------------------------- wave 2: query normalization variants

def load_normalized(path) -> dict:
    """diagnostics/normalized_queries.json -> id -> entry."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["id"]: e for e in (data["items"] if isinstance(data, dict) else data)}


def variant_backend(vs, variant, normalized, docs=retrieval_docs, finish=_finish):
    """V0 = the current retrieval of fq; V1 = of the normalized query nq; V2 / V3 = RRF of the two pools
    (diagnostics/wave2_core.fused_pool), reranked on fq / nq (diagnostics/WAVE2_PREREG.md). A failed normalization
    falls back to fq (wave2_core.query_for). Called with the item: run(q, fq, other, item)."""
    import wave2_core as W
    if variant not in W.VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    cid = lambda d: d.metadata["chunk_id"]

    def run(q, fq, other, item):
        nq, fell_back = W.query_for(normalized.get(item["id"]), fq)
        if variant == "V0":
            L = finish(vs, docs(vs, q, fq, other), fq)
        elif variant == "V1":
            L = finish(vs, docs(vs, q, nq, other), nq)
        else:
            Do, Dn = docs(vs, q, fq, other), docs(vs, q, nq, other)
            D = {**Do, "fused": W.rrf_merge([Do["fused"], Dn["fused"]], cid), "pool": W.fused_pool(Do["pool"], Dn["pool"], cid)}
            L = finish(vs, D, fq if variant == "V2" else nq)
        return {**L, "normalized_fallback": fell_back}
    run.wants_item = True
    return run


def rank_of(ids, gold) -> int | None:
    for i, c in enumerate(ids, 1):
        if key(c) in gold:
            return i
    return None


def classify(rank_reranked, rank_fused, path) -> str:
    if rank_reranked is not None:
        return "top6" if rank_reranked <= 6 else "rerank_7_10" if rank_reranked <= 10 else "rerank_11_20"
    return "fused_rank_gt20" if path != "direct" and rank_fused is not None else "absent_from_pool"


def evaluate(items, translations, backend) -> tuple:
    recs, skipped = [], []
    for it in items:
        base = {"id": it["id"], "set": it.get("set"), "language": it.get("language"), "noise_type": it.get("noise_type")}
        fq, other, why = french_query(it, translations)
        if fq is None:
            skipped.append(base | {"reason": why})
            continue
        gold = gold_keys(it)
        if not gold:
            skipped.append(base | {"reason": "no gold article"})
            continue
        L = backend(it["question"], fq, other, it) if getattr(backend, "wants_item", False) else backend(it["question"], fq, other)
        if "normalized_fallback" in L:
            base = base | {"normalized_fallback": L["normalized_fallback"]}
        r = base | {"query": "translation" if other else "as written", "path": L["path"],
                    **{f"rank_{n}": rank_of(L[n], gold) for n in ("dense", "bm25", "orig", "fused", "pool", "reranked")},
                    "pool_size": len(L["pool"]), "gold_in_evidence": rank_of(L["evidence"], gold) is not None,
                    "top6": L["reranked"][:6]}
        r["class"] = classify(r["rank_reranked"], r["rank_fused"], L["path"])
        r.update({f"hit{k}": int(r["rank_reranked"] is not None and r["rank_reranked"] <= k) for k in KS})
        recs.append(r)
    return recs, skipped


# ---------------------------------------------------------------- statistics

def boot_ci(xs, seed=SEED, n_boot=N_BOOT) -> tuple:
    """Percentile 95% CI of the mean, as run_answer_eval.boot_ci (random.Random(seed), n_boot resamples)."""
    n = len(xs)
    if not n:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def sign_test(gained: int, lost: int) -> float:
    """Exact two-sided sign test on the discordant items (p = 1 when there are none)."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def group_keys(r) -> list:
    g = ["all", f"language {r['language']}"]
    if r.get("set"):
        g.append(f"set {r['set']}")
    if r.get("noise_type"):
        g.append(f"noise {r['noise_type']}")
    return g


def summarize(recs) -> dict:
    groups = {}
    for r in recs:
        for g in group_keys(r):
            groups.setdefault(g, []).append(r)
    out = {}
    for g in sorted(groups, key=lambda x: (x != "all", x)):
        rs = groups[g]
        out[g] = {"n": len(rs), "classes": {c: sum(r["class"] == c for r in rs) for c in CLASSES},
                  "gold_in_evidence": sum(r["gold_in_evidence"] for r in rs),
                  "hit": {k: {"n": sum(r[f"hit{k}"] for r in rs), "rate": sum(r[f"hit{k}"] for r in rs) / len(rs),
                              "ci": boot_ci([r[f"hit{k}"] for r in rs])} for k in KS}}
    return out


def paired(recs_a, recs_b) -> dict:
    a = {r["id"]: r for r in recs_a}
    b = {r["id"]: r for r in recs_b}
    ids = [i for i in a if i in b]
    out = {"n": len(ids), "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a))}
    for k in KS:
        d = [b[i][f"hit{k}"] - a[i][f"hit{k}"] for i in ids]
        gained = [i for i in ids if b[i][f"hit{k}"] > a[i][f"hit{k}"]]
        lost = [i for i in ids if b[i][f"hit{k}"] < a[i][f"hit{k}"]]
        out[k] = {"a": sum(a[i][f"hit{k}"] for i in ids), "b": sum(b[i][f"hit{k}"] for i in ids),
                  "diff": sum(d) / len(d) if d else None, "ci": boot_ci(d), "gained": gained, "lost": lost,
                  "sign_p": sign_test(len(gained), len(lost))}
    return out


# ---------------------------------------------------------------- output

def fmt_rate(h) -> str:
    lo, hi = h["ci"]
    return f"{h['n']} ({100 * h['rate']:.1f} %) [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else "-"


def markdown(runs: dict, comparison: dict | None) -> str:
    L = ["# Retrieval-only evaluation", ""]
    for name, run in runs.items():
        L += [f"### {name}: Hit@k with 95% bootstrap CI ({N_BOOT} resamples, seed {SEED})",
              "| Group | n | Hit@6 | Hit@8 | Hit@10 | gold in evidence (top 6 + cross-refs) |", "|---|---|---|---|---|---|"]
        for g, s in run["summary"].items():
            L.append(f"| {g} | {s['n']} | " + " | ".join(fmt_rate(s["hit"][k]) for k in KS) + f" | {s['gold_in_evidence']} |")
        L += ["", f"### {name}: funnel classes", "| Group | n | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * len(CLASSES)]
        for g, s in run["summary"].items():
            L.append(f"| {g} | {s['n']} | " + " | ".join(str(s["classes"][c]) for c in CLASSES) + " |")
        L += ["", f"### {name}: skipped items", "| Id | Language | Reason |", "|---|---|---|"]
        L += [f"| {x['id']} | {x['language']} | {x['reason']} |" for x in run["skipped"]] or ["| - | - | - |"]
        L.append("")
    if comparison:
        L += [f"### Paired comparison B - A ({comparison['n']} items in both; 95% bootstrap CI of the mean difference)",
              "| k | A hits | B hits | difference (B - A) | 95% CI | gained | lost | sign test p |", "|---|---|---|---|---|---|---|---|"]
        for k in KS:
            c = comparison[k]
            lo, hi = c["ci"]
            L.append(f"| Hit@{k} | {c['a']} | {c['b']} | {100 * c['diff']:+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                     f"{len(c['gained'])} | {len(c['lost'])} | {c['sign_p']:.3f} |")
        L += ["", "### Gained and lost items (Hit@6)", "| Change | Items |", "|---|---|",
              f"| gained | {', '.join(comparison[6]['gained']) or '-'} |", f"| lost | {', '.join(comparison[6]['lost']) or '-'} |", ""]
    return "\n".join(L)


def sample_rows(entries, n=12, seed=SEED) -> list:
    """n entries drawn with random.Random(seed).sample, in item order."""
    entries = list(entries)
    idx = sorted(random.Random(seed).sample(range(len(entries)), min(n, len(entries))))
    return [entries[i] for i in idx]


def _cell(t) -> str:
    return (t or "-").replace("|", "\\|").replace("\n", " ")


def variants_markdown(runs: dict, comps: dict, decision: dict, entries: list) -> str:
    """Tables of the wave 2 variant run (development data, not evidence of generalisation)."""
    names = list(runs)
    L = ["# Wave 2: query normalization variants (retrieval only)", "",
         "Development data (110 items), not evidence of generalisation. Pre-registration: diagnostics/WAVE2_PREREG.md.", ""]
    es = list(entries)
    tok = {k: sum(e["tokens"][k] for e in es) for k in ("input_tokens", "output_tokens", "total_tokens")}
    fails = [e for e in es if e["status"] != "ok"]
    flagged = [e for e in es if e["flags"].get("new_numbers") or e["flags"].get("over_40_words")]
    L += ["### Normalization calls", "| Items | ok | failures (fallback to the input query) | 2 attempts | input tokens | output tokens | total tokens | mean total / item | prompt sha256 |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| {len(es)} | {len(es) - len(fails)} | {len(fails)} | {sum(e['attempts'] > 1 for e in es)} | {tok['input_tokens']} | "
          f"{tok['output_tokens']} | {tok['total_tokens']} | {tok['total_tokens'] / max(1, len(es)):.0f} | "
          f"{', '.join(sorted({e['prompt_sha256'][:12] for e in es}))} |", ""]
    L += ["### Failed normalizations", "| Id | Status | Error |", "|---|---|---|"]
    L += [f"| {e['id']} | {e['status']} | {_cell(e['error'])} |" for e in fails] or ["| - | - | - |"]
    L += ["", "### Flagged outputs (numbers not in the input query; more than 40 words)", "| Id | New numbers | Words | Input query | legal_query |", "|---|---|---|---|---|"]
    L += [f"| {e['id']} | {', '.join(e['flags'].get('new_numbers') or []) or '-'} | {len(e['legal_query'].split())} | "
          f"{_cell(e['input_query'])} | {_cell(e['legal_query'])} |" for e in flagged] or ["| - | - | - | - | - |"]
    L += ["", f"### 12 random rows (random.Random({SEED}).sample)", "| Id | Set | Language | Input query | legal_query | Flags |", "|---|---|---|---|---|---|"]
    for e in sample_rows(es):
        fl = [k for k in ("new_numbers", "over_40_words") if e["flags"].get(k)] + (["failure"] if e["status"] != "ok" else [])
        L.append(f"| {e['id']} | {e['set']} | {e['language']} | {_cell(e['input_query'])} | {_cell(e['legal_query'])} | {', '.join(fl) or '-'} |")
    groups = list(runs[names[0]]["summary"])
    for k in KS:
        L += ["", f"### Hit@{k} per group, 95% bootstrap CI ({N_BOOT} resamples, seed {SEED})",
              "| Group | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for g in groups:
            L.append(f"| {g} | {runs[names[0]]['summary'][g]['n']} | " + " | ".join(
                fmt_rate(runs[v]["summary"][g]["hit"][k]) if g in runs[v]["summary"] else "-" for v in names) + " |")
    L += ["", "### Funnel classes (all items)", "| Variant | n | " + " | ".join(CLASSES) + " | gold in evidence | pool size mean |",
          "|---|---|" + "---|" * (len(CLASSES) + 2)]
    for v in names:
        s, recs = runs[v]["summary"]["all"], runs[v]["records"]
        L.append(f"| {v} | {s['n']} | " + " | ".join(str(s["classes"][c]) for c in CLASSES) +
                 f" | {s['gold_in_evidence']} | {sum(r['pool_size'] for r in recs) / max(1, len(recs)):.1f} |")
    for v in names:
        L += ["", f"### {v}: funnel classes per group", "| Group | n | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * len(CLASSES)]
        L += [f"| {g} | {s['n']} | " + " | ".join(str(s["classes"][c]) for c in CLASSES) + " |" for g, s in runs[v]["summary"].items()]
    L += ["", "### Paired differences versus V0 (95% bootstrap CI of the mean difference; exact two-sided sign test)",
          "| Variant | k | V0 hits | variant hits | difference | 95% CI | gained | lost | sign test p |", "|---|---|---|---|---|---|---|---|---|"]
    for v, c in comps.items():
        for k in KS:
            lo, hi = c[k]["ci"]
            L.append(f"| {v} | Hit@{k} | {c[k]['a']} | {c[k]['b']} | {100 * c[k]['diff']:+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                     f"{len(c[k]['gained'])} | {len(c[k]['lost'])} | {c[k]['sign_p']:.3f} |")
    L += ["", "### Gained and lost items versus V0 (Hit@6)", "| Variant | gained | lost |", "|---|---|---|"]
    L += [f"| {v} | {', '.join(c[6]['gained']) or '-'} | {', '.join(c[6]['lost']) or '-'} |" for v, c in comps.items()]
    L += ["", "### Decision rule (pre-registered)", "| Variant | total Hit@6 (V0) | extra tokens | (i) noisy net vs V0 >= +3 | (ii) lost S1+S2 <= 1 | "
          "(iii) lost per language <= 1 | passes |", "|---|---|---|---|---|---|---|"]
    for v, c in decision["conditions"].items():
        ll = ", ".join(f"{l}: {len(x)}" + (f" ({', '.join(x)})" if x else "") for l, x in c["lost_by_language"].items())
        L.append(f"| {v} | {c['total_hit6']} ({c['v0_total_hit6']}) | {c['extra_tokens']} | {c['noisy_net']:+d} "
                 f"({'yes' if c['i_noisy_gain_ge_3'] else 'no'}) | {len(c['lost_stage1_2'])}"
                 f"{' (' + ', '.join(c['lost_stage1_2']) + ')' if c['lost_stage1_2'] else ''} ({'yes' if c['ii_lost_stages_le_1'] else 'no'}) | "
                 f"{ll} ({'yes' if c['iii_lost_each_language_le_1'] else 'no'}) | {'yes' if c['passes'] else 'no'} |")
    L += ["", "| Candidate (highest total Hit@6) | Selected |", "|---|---|", f"| {decision['candidate']} | {decision['selected']} |", ""]
    return "\n".join(L)


def run_variants(items, tr, vs, normalized, variants, preset, overrides) -> tuple:
    """-> (runs, comparisons versus V0, decision)."""
    import wave2_core as W
    cfg = apply_config(preset, overrides)
    runs = {}
    for v in variants:
        recs, skipped = evaluate(items, tr, variant_backend(vs, v, normalized))
        runs[v] = {"config": cfg, "records": recs, "skipped": skipped, "summary": summarize(recs)}
        print(f"{v}: {len(recs)} items, {len(skipped)} skipped; Hit@6/8/10 = "
              + " / ".join(str(runs[v]["summary"]["all"]["hit"][k]["n"]) if recs else "-" for k in KS), flush=True)
    comps = {v: paired(runs["V0"]["records"], runs[v]["records"]) for v in variants if v != "V0" and "V0" in runs}
    total = sum(e["tokens"]["total_tokens"] for e in normalized.values())
    decision = W.decide({v: r["records"] for v, r in runs.items()}, {v: (0 if v == "V0" else total) for v in runs})
    return runs, comps, decision


def open_copy(path, opener=None) -> tuple:
    """Open a temporary byte copy of the index at path. -> (vectorstore, temporary folder)."""
    tmp = Path(tempfile.mkdtemp(prefix="retrieval_only_"))
    shutil.copytree(path, tmp / "db")
    if opener is None:
        from langchain_chroma import Chroma
        from pipeline import get_embeddings
        opener = lambda p: Chroma(persist_directory=str(p), embedding_function=get_embeddings())
    return opener(tmp / "db"), tmp


def parse_sets(pairs) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items")
    ap.add_argument("--translations", default=str(ROOT / "diagnostics" / "translations.json"))
    ap.add_argument("--index-a")
    ap.add_argument("--config-a", default="baseline", choices=list(PRESETS))
    ap.add_argument("--set-a", nargs="*")
    ap.add_argument("--index-b")
    ap.add_argument("--config-b", default="v1.2", choices=list(PRESETS))
    ap.add_argument("--set-b", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--export-dev")
    ap.add_argument("--variants", nargs="+", help="wave 2: V0 V1 V2 V3 on --index-a / --config-a (needs --normalized)")
    ap.add_argument("--normalized", default=str(ROOT / "diagnostics" / "normalized_queries.json"))
    a = ap.parse_args(argv)
    if a.export_dev:
        export_dev(a.export_dev)
        return 0
    if not (a.items and a.index_a):
        ap.error("--items and --index-a are required")
    items, tr = load_items(a.items), load_translations(a.translations)
    if a.variants:
        normalized = load_normalized(a.normalized)
        vs, tmp = open_copy(a.index_a)
        runs, comps, decision = run_variants(items, tr, vs, normalized, a.variants, a.config_a, parse_sets(a.set_a))
        del vs
        if a.out:
            Path(a.out).write_text(json.dumps({"index": str(a.index_a), "preset": a.config_a, "runs": runs, "comparisons": comps,
                                               "decision": decision}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        md = variants_markdown(runs, comps, decision, normalized.values())
        if a.md:
            Path(a.md).write_text(md + "\n", encoding="utf-8")
        else:
            print(md)
        print(f"decision: candidate {decision['candidate']}, selected {decision['selected']}")
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)
        if tmp.exists():
            print(f"temporary index copy left (files still open): {tmp}", file=sys.stderr)
        return 0
    runs, temps = {}, []
    specs = [("A", a.index_a, a.config_a, a.set_a)] + ([("B", a.index_b, a.config_b, a.set_b)] if a.index_b else [])
    for tag, idx, preset, sets in specs:
        cfg = apply_config(preset, parse_sets(sets))
        vs, tmp = open_copy(idx)
        temps.append(tmp)
        recs, skipped = evaluate(items, tr, real_backend(vs))
        name = f"{tag} = {Path(idx).name} / {preset}" + (f" + {sets}" if sets else "")
        runs[name] = {"index": str(idx), "config": cfg, "records": recs, "skipped": skipped, "summary": summarize(recs)}
        print(f"{name}: {len(recs)} items, {len(skipped)} skipped; Hit@6/8/10 = "
              + " / ".join(str(runs[name]["summary"]["all"]["hit"][k]["n"]) if recs else "-" for k in KS), flush=True)
        del vs
    names = list(runs)
    comparison = paired(runs[names[0]]["records"], runs[names[1]]["records"]) if len(names) == 2 else None
    if a.out:
        Path(a.out).write_text(json.dumps({"runs": runs, "comparison": comparison}, ensure_ascii=False, indent=1, default=str),
                               encoding="utf-8")
    md = markdown(runs, comparison)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")
    else:
        print(md)
    gc.collect()
    for t in temps:
        shutil.rmtree(t, ignore_errors=True)
        if t.exists():
            print(f"temporary index copy left (files still open): {t}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
