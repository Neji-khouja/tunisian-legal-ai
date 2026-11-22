"""Offline comparison of the retrieval core on the development items: db_v1_1 with every v2.1 switch off (baseline)
against db_v1_2 with the switches on; BM25 tokenizer experiment; evidence window experiment. No LLM call: Arabic and
English items use the French queries of diagnostics/translations.json.

    python diagnostics/compare_v12.py --v11 COPY_OF_DB_V1_1
"""
import re
import sys
import json
import time
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(Path(__file__).resolve().parent)]
import config as C
import run_answer_eval as rae
import funnel_core as fc
from retrieval_diagnostic import items_by_set, load_translations, SETS, table, pct

OUT_JSON = ROOT / "diagnostics" / "compare_v12.json"
OUT_MD = ROOT / "diagnostics" / "COMPARISON_V12.md"
DEV = "development data, not evidence of generalisation"
VARIANTS = ["whitespace", "lower", "fold", "elision", "stem"]
OFF = dict(DIRECT_LOOKUP_STRICT=False, DIRECT_LOOKUP_UNION=False, XREF_EXPAND=False, EVIDENCE_CONTEXT_HEADER=False,
           SUBCHUNK_LONG=False, BM25_TOKENIZER="whitespace")
ON = dict(DIRECT_LOOKUP_STRICT=True, DIRECT_LOOKUP_UNION=True, XREF_EXPAND=True, EVIDENCE_CONTEXT_HEADER=True,
          SUBCHUNK_LONG=True)


def set_flags(**kw):
    for k, v in kw.items():
        setattr(C, k, v)


def open_index(path):
    from langchain_chroma import Chroma
    from pipeline import get_embeddings
    return Chroma(persist_directory=str(path), embedding_function=get_embeddings())


def missing_refs(vs, evidence, articles):
    """For each article in `articles` (Documents in the evidence): its references that resolve to an article of the
    index (same PDF and section; no caps) and are not in the evidence. -> [(chunk_id, n_resolved, n_missing)]."""
    from retriever import expand_cross_references
    have = {d.metadata["chunk_id"] for d in evidence}
    out = []
    for d in articles:
        res = expand_cross_references(vs, [d], per_article=99, total=99)      # resolved, not already d itself
        resolved = {x.metadata["chunk_id"] for x in res}
        out.append((d.metadata["chunk_id"], len(resolved), len(resolved - have)))
    return out


def run_config(name, vs, items, translations, tok_len, dense_cache):
    from pipeline import step_retrieve, evidence_docs, evidence_text, _ARTICLE_RE
    from retriever import (_semantic_search, _bm25_search, _get_bm25, _fuse, _deduplicate, _direct_lookup,
                           detect_arabic, detect_english, article_numbers)
    from reranker import rerank
    bm25 = _get_bm25(vs)
    keys = lambda docs: [rae.key(d.metadata["chunk_id"]) for d in docs]
    recs = []
    for s, its in items.items():
        for it in its:
            q = it["question"]
            other = detect_arabic(q) or detect_english(q)
            fq = translations[(s, it["id"])] if other else q
            gold = rae.gold_set(it)
            ck = (id(vs), s, it["id"])
            if ck not in dense_cache:
                dense_cache[ck] = (_semantic_search(vs, fq, None, k=C.MMR_K), _semantic_search(vs, q, None, k=10) if other else [])
            dense, orig = dense_cache[ck]
            bm = _bm25_search(bm25, fq, None, k=C.BM25_K) if bm25 else []
            fused_full = _deduplicate(_fuse([dense, bm] + ([orig] if other else [])))
            nums = article_numbers(q) if C.DIRECT_LOOKUP_STRICT else ([m.group(0)] if (m := _ARTICLE_RE.search(q)) else [])
            if nums and not C.DIRECT_LOOKUP_UNION:
                path = "direct" if _direct_lookup(vs, nums[0], None, C.MMR_K) else "hybrid"
            else:
                path = "direct+hybrid" if nums else "hybrid"
            pool = step_retrieve(vs, q, fq, [], None)
            reranked = rerank(fq, pool, top_k=len(pool))
            top6 = reranked[:6]
            ev = evidence_docs(top6, vs)
            added = ev[len(top6):]
            r = {"set": s, "id": it["id"], "language": it.get("language"), "noise_type": it.get("noise_type"), "path": path,
                 "direct_numbers": nums,
                 "rank_dense": fc.rank_of(keys(dense), gold), "rank_bm25": fc.rank_of(keys(bm), gold),
                 "rank_orig_lang": fc.rank_of(keys(orig), gold) if other else None,
                 "rank_fused_full": fc.rank_of(keys(fused_full), gold), "pool_size": len(pool),
                 "rank_pool": fc.rank_of(keys(pool), gold), "rank_reranked": fc.rank_of(keys(reranked), gold),
                 "gold_in_evidence_with_xref": any(k in gold for k in keys(ev)),
                 "gold_only_via_xref": not any(k in gold for k in keys(top6)) and any(k in gold for k in keys(added)),
                 "xref_added": [d.metadata["chunk_id"] for d in added],
                 "xref_added_chars": sum(len(evidence_text(d)) for d in added),
                 "xref_added_tokens": sum(tok_len(evidence_text(d)) for d in added),
                 "evidence6_tokens": sum(tok_len(evidence_text(d)) for d in top6),
                 "context_header_tokens": sum(tok_len(evidence_text(d)) - tok_len(evidence_text(d).split("\n", 1)[-1])
                                              for d in top6 if d.metadata.get("context") and C.EVIDENCE_CONTEXT_HEADER),
                 "window_7_8_tokens": sum(tok_len(evidence_text(d)) for d in reranked[6:8]),
                 "window_7_10_tokens": sum(tok_len(evidence_text(d)) for d in reranked[6:10]),
                 "top6_ids": [d.metadata["chunk_id"] for d in top6]}
            cls_path = "direct" if path == "direct" else "hybrid"
            r["class"] = fc.classify(r["rank_reranked"], r["rank_fused_full"], cls_path)
            r["list_flag"] = fc.list_flag(r["rank_dense"], r["rank_bm25"])
            gold_docs = [d for d in top6 if rae.key(d.metadata["chunk_id"]) in gold]
            r["refs_gold"] = missing_refs(vs, ev, gold_docs)
            r["refs_top6"] = missing_refs(vs, ev, top6)
            recs.append(r)
        print(f"  {name}: {s} done", flush=True)
    return recs


# ---------------------------------------------------------------- summaries

def groups(recs):
    g = {}
    for s in SETS:
        g[SETS[s]] = [r for r in recs if r["set"] == s]
    for lang in sorted({r["language"] for r in recs}):
        g[f"language {lang}"] = [r for r in recs if r["language"] == lang]
    for nt in sorted({r["noise_type"] for r in recs if r["noise_type"]}):
        g[f"noise {nt}"] = [r for r in recs if r["noise_type"] == nt]
    g["all"] = recs
    return g


def hits(rs, k):
    return sum(fc.hit_at(r["rank_reranked"], k) for r in rs)


def stat(xs):
    if not xs:
        return ["-"] * 5
    xs = sorted(xs)
    return [round(statistics.fmean(xs), 1), statistics.median(xs), xs[max(0, int(0.9 * len(xs)) - 1)], xs[-1], sum(xs)]


def ref_share(rs, field):
    pairs = [p for r in rs for p in r[field] if p[1] > 0]
    allp = [p for r in rs for p in r[field]]
    return len(allp), len(pairs), sum(p[2] > 0 for p in pairs)


def markdown(doc):
    base, final = doc["records"]["baseline"], doc["records"]["v1.2"]
    L = [f"# Offline comparison: db_v1_1 (switches off) vs db_v1_2 (v2.1 wave 1) — {DEV}", ""]
    L += table(f"Configurations ({DEV})", ["Configuration", "Index", "Switches"],
               [["baseline", "db_v1_1 (byte copy)", "all v2.1 switches off (= v2 retrieval core)"],
                ["v1.2", "db_v1_2", f"headings forward, context in indexed text, sub-chunks, strict direct lookup + union, "
                                    f"cross-reference expansion, context header; BM25 tokenizer {doc['adopted_bm25']}"]])
    L += table(f"Index build facts ({DEV})", ["Fact", "db_v1_1", "db_v1_2"], doc["build_rows"])
    L += table(f"Baseline reproduces the diagnostic of RETRIEVAL_DIAGNOSTIC.md ({DEV})", ["Items", "same funnel class", "different"],
               [[len(base), doc["baseline_check"]["same"], doc["baseline_check"]["different"]]])
    hdr = ["Group", "n"] + [fc.CLASS_LABELS[c] for c in fc.CLASSES] + ["Hit@6", "Hit@8", "Hit@10", "gold in evidence with cross-refs"]
    for name, recs in (("baseline (db_v1_1)", base), ("v1.2 (db_v1_2)", final)):
        rows = []
        for g, rs in groups(recs).items():
            sm = fc.summarize(rs)
            rows.append([g, sm["n"]] + [pct(sm["classes"][c]["n"], sm["n"]) for c in fc.CLASSES] +
                        [pct(hits(rs, k), len(rs)) for k in (6, 8, 10)] + [pct(sum(r["gold_in_evidence_with_xref"] for r in rs), len(rs))])
        L += table(f"Funnel classes and Hit@k, {name} ({DEV})", hdr, rows)
    rows = []
    gb, gf = groups(base), groups(final)
    for g in gb:
        rows.append([g, len(gb[g])] + [f"{hits(gb[g], k)} -> {hits(gf[g], k)}" for k in (6, 8, 10)] +
                    [f"{sum(r['gold_in_evidence_with_xref'] for r in gb[g])} -> {sum(r['gold_in_evidence_with_xref'] for r in gf[g])}"])
    L += table(f"Hit@k, baseline -> v1.2 (items) ({DEV})", ["Group", "n", "Hit@6", "Hit@8", "Hit@10", "gold in evidence with cross-refs"], rows)
    bi = {(r["set"], r["id"]): r for r in base}
    ch = Counter()
    for r in final:
        a, b = fc.hit_at(bi[(r["set"], r["id"])]["rank_reranked"], 6), fc.hit_at(r["rank_reranked"], 6)
        ch["gained" if b > a else "lost" if a > b else "same hit" if a else "same miss"] += 1
    L += table(f"Hit@6 per item, baseline -> v1.2 ({DEV})", ["Change", "Items"], [[k, ch[k]] for k in ("gained", "lost", "same hit", "same miss")])
    lost = [r for r in final if fc.hit_at(bi[(r["set"], r["id"])]["rank_reranked"], 6) > fc.hit_at(r["rank_reranked"], 6)]
    L += table(f"Items that lost Hit@6 ({DEV})", ["Set", "Id", "Language", "baseline class", "v1.2 class"],
               [[SETS[r["set"]], r["id"], r["language"], fc.CLASS_LABELS[bi[(r["set"], r["id"])]["class"]], fc.CLASS_LABELS[r["class"]]]
                for r in lost] or [["-"] * 5])
    rows = []
    for name, recs in (("baseline", base), ("v1.2", final)):
        for s in SETS:
            rs = [r for r in recs if r["set"] == s]
            rows.append([name, SETS[s], len(rs)] + [sum(r["path"] == p for r in rs) for p in ("hybrid", "direct", "direct+hybrid")] +
                        [sum(r["path"] != "hybrid" and fc.hit_at(r["rank_reranked"], 6) for r in rs)])
    L += table(f"Retrieval path ({DEV})", ["Configuration", "Set", "n", "hybrid", "direct (replaces the pool)",
                                           "direct + hybrid (union)", "direct paths with Hit@6"], rows)
    rows = []
    for name, recs in (("baseline", base), ("v1.2", final)):
        for field, what in (("refs_gold", "gold articles in the top 6"), ("refs_top6", "all top-6 articles")):
            for g in [SETS[s] for s in SETS] + ["all"]:
                rs = groups(recs)[g]
                n, with_ref, miss = ref_share(rs, field)
                rows.append([name, what, g, n, pct(with_ref, n), pct(miss, n)])
    L += table(f"Cited-style articles whose referenced article is missing from the evidence ({DEV})",
               ["Configuration", "Articles", "Group", "articles", "with a resolvable reference", "with a referenced article not in the evidence"], rows)
    rows = []
    for g, rs in groups(final).items():
        n_added = [len(r["xref_added"]) for r in rs]
        rows.append([g, len(rs), sum(x > 0 for x in n_added)] + stat(n_added)[:2] + stat([r["xref_added_tokens"] for r in rs]) +
                    [sum(r["gold_only_via_xref"] for r in rs)])
    L += table(f"Cross-reference expansion cost on the dev items (v1.2; tokens = embedding-model tokenizer, not the LLM's) ({DEV})",
               ["Group", "items", "items with added articles", "added articles mean", "median", "added tokens mean", "median", "p90", "max",
                "total", "gold only via an added article"], rows)
    rows = [[g, len(rs)] + stat([r["evidence6_tokens"] for r in rs])[:4] + stat([r["context_header_tokens"] for r in rs])[:2]
            for g, rs in groups(final).items() if g in [SETS[s] for s in SETS] + ["all"]]
    L += table(f"Evidence size per question, v1.2 top 6 (embedding-model tokens) ({DEV})",
               ["Group", "items", "top-6 evidence tokens mean", "median", "p90", "max", "context-header tokens mean", "median"], rows)
    rows = []
    for name, recs in (("baseline", base), ("v1.2", final)):
        for g in [SETS[s] for s in SETS] + ["all"]:
            rs = groups(recs)[g]
            rows.append([name, g, len(rs)] + [pct(hits(rs, k), len(rs)) for k in (6, 8, 10)] +
                        [stat([r["window_7_8_tokens"] for r in rs])[0], stat([r["window_7_10_tokens"] for r in rs])[0]])
    L += table(f"Evidence window: top 6 vs top 8 vs top 10 (default unchanged: 6) ({DEV})",
               ["Configuration", "Group", "n", "Hit@6", "Hit@8", "Hit@10", "added tokens for top 8 (mean)", "added tokens for top 10 (mean)"], rows)
    # BM25
    rows = []
    for name, recs in (("baseline", base), ("v1.2", final)):
        for g in [SETS[s] for s in SETS] + ["all"]:
            rs = groups(recs)[g]
            fl = Counter(r["list_flag"] for r in rs)
            rows.append([name, g, len(rs), fl["present_both"], fl["absent_dense_only"], fl["absent_bm25_only"], fl["absent_both"],
                         sum(r["list_flag"] == "absent_dense_only" and fc.hit_at(r["rank_reranked"], 6) for r in rs)])
    L += table(f"BM25 contribution: gold in the dense (k=20) and BM25 (k=15) lists ({DEV})",
               ["Configuration", "Group", "n", "in both", "BM25 only", "dense only", "neither", "BM25 only and Hit@6"], rows)
    rows = []
    for v in VARIANTS:
        rs = doc["records"]["bm25"][v]
        for g in [SETS[s] for s in SETS] + ["all"]:
            x = groups(rs)[g]
            ranks = [r["rank_bm25"] for r in x if r["rank_bm25"]]
            rows.append([v, g, len(x), sum(r["rank_bm25"] is not None for r in x), statistics.median(ranks) if ranks else "-",
                         pct(hits(x, 6), len(x)), pct(hits(x, 10), len(x))])
    L += table(f"BM25 tokenizer variants on db_v1_2 (other v2.1 switches on) ({DEV})",
               ["Tokenizer", "Group", "n", "gold in the BM25 list (k=15)", "median BM25 rank of the gold", "Hit@6", "Hit@10"], rows)
    L += table(f"BM25 tokenizer adoption rule ({DEV})", ["Tokenizer", "BM25-list gold, all", "no set/language with fewer BM25-list gold",
                                                       "no set/language with lower Hit@6", "adopted"],
               [[v, x["bm25_all"], x["bm25_not_worse"], x["hit6_not_worse"], v == doc["adopted_bm25"]] for v, x in doc["bm25_rule"].items()])
    L += table(f"Run ({DEV})", ["Item", "Value"], [[k, v] for k, v in doc["run"].items()])
    return "\n".join(L)


def bm25_rule(variants: dict) -> tuple[dict, str]:
    """A variant is adopted when the gold is in the BM25 list more often (all items) and no set or language gets fewer
    BM25-list golds or a lower Hit@6 than whitespace; the qualifying variant with the most BM25-list golds wins."""
    base = groups(variants["whitespace"])
    keys = [g for g in base if not g.startswith("noise") and g != "all"]
    out, best = {}, ("whitespace", sum(r["rank_bm25"] is not None for r in variants["whitespace"]))
    for v, recs in variants.items():
        g = groups(recs)
        n_all = sum(r["rank_bm25"] is not None for r in recs)
        not_worse = all(sum(r["rank_bm25"] is not None for r in g[k]) >= sum(r["rank_bm25"] is not None for r in base[k]) for k in keys)
        hit_ok = all(hits(g[k], 6) >= hits(base[k], 6) for k in keys)
        out[v] = {"bm25_all": n_all, "bm25_not_worse": not_worse, "hit6_not_worse": hit_ok}
        if v != "whitespace" and not_worse and hit_ok and n_all > best[1]:
            best = (v, n_all)
    return out, best[0]


def build_rows(v11_docs, v12_vs, tok_len):
    facts = json.loads((ROOT / "diagnostics" / "index_v12_build.json").read_text(encoding="utf-8"))["runs"]
    first = facts[0]
    got = v12_vs._collection.get(include=["documents", "metadatas"])
    arts = {m["parent_id"]: m["article_text"] for m in got["metadatas"]}
    title_any = re.compile(r"(^|\n)\s*(chapitre|section|sous-section|titre|livre)\s+[\wIVXLC]+", re.I)
    v11_ids = {d["id"] for d in v11_docs}
    over11 = sum(tok_len(d["text"]) > 512 for d in v11_docs)
    over12 = sum(tok_len(t) > 512 for t in got["documents"])
    rows = [["articles (chunk ids)", len(v11_docs), len(arts)],
            ["article ids identical to db_v1_1", "-", set(arts) == v11_ids],
            ["index entries (embedded texts)", len(v11_docs), len(got["documents"])],
            ["articles split into sub-chunks", 0, first["split_articles"]],
            ["sub-chunk entries", 0, first["sub_chunks"]],
            ["embedded texts over 512 tokens (Solon max_seq_length)", over11, over12],
            ["articles containing a section/chapter/title line", sum(bool(title_any.search(d["text"])) for d in v11_docs),
             sum(bool(title_any.search(t)) for t in arts.values())],
            ["headings moved to the following article", 0, first["headings_moved"]],
            ["articles whose text lost a trailing heading", 0, first["chunks_with_headings_moved"]],
            ["articles without a heading hierarchy", "-", first["articles_without_hierarchy"]],
            ["index size on disk (MB)", round(first["db_v1_1_bytes"] / 1e6, 1), round(first["index_bytes"] / 1e6, 1)],
            ["embedding time (s)", "not recorded", first["seconds_embed"]],
            ["split + token count time (s)", "-", first["seconds_split_and_count"]],
            ["build total time (s)", "-", first["seconds_total"]],
            ["device", "-", f"{first['device']} ({first['machine']})"]]
    if len(facts) > 1:
        rows.append(["build runs (resumed)", "-", len(facts)])
    return rows


def main():
    from pipeline import get_embeddings
    import harness_core as hc
    from retrieval_diagnostic import index_docs
    v11_path = Path(sys.argv[sys.argv.index("--v11") + 1])
    t0 = time.perf_counter()
    items = items_by_set()
    translations = load_translations()
    tok = get_embeddings()._client.tokenizer
    tok_len = lambda t: len(tok(t, add_special_tokens=True)["input_ids"])
    v11, v12 = open_index(v11_path), open_index(C.DB_V12)
    cache = {}
    set_flags(**OFF)
    base = run_config("baseline", v11, items, translations, tok_len, cache)
    variants = {}
    for v in VARIANTS:
        set_flags(**ON, BM25_TOKENIZER=v)
        variants[v] = run_config(f"v1.2 bm25={v}", v12, items, translations, tok_len, cache)
    rule, adopted = bm25_rule(variants)
    diag = {(r["set"], r["id"]): r["class"] for r in json.loads((ROOT / "diagnostics" / "retrieval_funnel.json").read_text(encoding="utf-8"))["records"]}
    same = sum(diag.get((r["set"], r["id"])) == r["class"] for r in base)
    v11_docs = index_docs(v11)
    doc = {"records": {"baseline": base, "v1.2": variants[adopted], "bm25": variants}, "adopted_bm25": adopted, "bm25_rule": rule,
           "baseline_check": {"same": same, "different": len(base) - same},
           "build_rows": build_rows(v11_docs, v12, tok_len)}
    set_flags(**ON, BM25_TOKENIZER=adopted)
    doc["run"] = {"items": f"{len(base)} answerable items (Stage 1, Stage 2, noisy set; all languages)",
                  "French queries of Arabic/English items": "diagnostics/translations.json",
                  "LLM calls": 0, "seconds": round(time.perf_counter() - t0, 1),
                  "fingerprint of the configuration files at run time (db_v1_2)": hc.system_fingerprint(ROOT, C.DB_V12)["sha256"],
                  "token counts": "embedding-model tokenizer (Solon / XLM-R), special tokens included; the LLM tokenizer is not available offline"}
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(doc) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}; adopted BM25 tokenizer: {adopted}")


def markdown_only():
    doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    OUT_MD.write_text(markdown(doc) + "\n", encoding="utf-8")
    print(f"rewrote {OUT_MD.name}")


if __name__ == "__main__":
    markdown_only() if "--md-only" in sys.argv else main()
