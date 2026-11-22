"""Retrieval funnel of v1.1, cross-references and chunk structure. Local compute only: no LLM call, the index is read.

    python diagnostics/retrieval_diagnostic.py [--db COPY_OF_DB_V1_1]
"""
import re
import sys
import json
import random
import hashlib
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(Path(__file__).resolve().parent)]
import run_answer_eval as rae
import funnel_core as fc

OUT_JSON = ROOT / "diagnostics" / "retrieval_funnel.json"
OUT_MD = ROOT / "diagnostics" / "RETRIEVAL_DIAGNOSTIC.md"
TRANSLATIONS = ROOT / "diagnostics" / "translations.json"
SEED = 20261022
SETS = {"S1": "Stage 1", "S2": "Stage 2", "N": "Noisy set"}
TITLE = re.compile(r"^\s*(chapitre|section|sous-section|titre|livre|paragraphe|partie|chap\.)\b", re.I)
TITLE_ANY = re.compile(r"(^|\n)\s*(chapitre|section|sous-section|titre|livre)\s+[\wIVXLC]+", re.I)


def items_by_set() -> dict:
    exam = json.loads((ROOT / "eval" / "answer_set.json").read_text(encoding="utf-8"))
    stages = json.loads((ROOT / "eval" / "stages.json").read_text(encoding="utf-8"))
    noisy = json.loads((ROOT / "eval" / "noisy_set.json").read_text(encoding="utf-8"))
    out = {k: [i for i in exam if i["id"] in set(stages[f"stage{k[1]}"]) and i["expected"] == "answer"] for k in ("S1", "S2")}
    out["N"] = [i for i in noisy if i.get("expected", "answer") == "answer"]
    return out


def saved_rows(key: str, system: str) -> dict:
    p = (ROOT / "results" / "noisy" / f"answer_eval_noisy_set_{system}.jsonl" if key == "N"
         else ROOT / "results" / f"answer_eval_answer_set_stage{key[1]}_{system}.jsonl")
    return {r["id"]: r for r in rae.read_jsonl(p)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_translations() -> dict:
    """(set, id) -> French query, from diagnostics/translations.json (diagnostics/translate_skipped.py)."""
    if not TRANSLATIONS.exists():
        return {}
    return {(x["set"], x["id"]): x["french_query"] for x in json.loads(TRANSLATIONS.read_text(encoding="utf-8"))["items"]}


def funnel(vs, items: dict, rows11: dict, rows2: dict) -> tuple[list, list]:
    from pipeline import _ARTICLE_RE
    from retriever import (_semantic_search, _bm25_search, _get_bm25, _fuse, _deduplicate, _direct_lookup,
                           detect_arabic, detect_english)
    from reranker import rerank
    from config import MMR_K, BM25_K, RERANK_POOL
    bm25 = _get_bm25(vs)
    keys = lambda docs: [rae.key(d.metadata["chunk_id"]) for d in docs]
    translations = load_translations()
    records, skipped = [], []
    for s, its in items.items():
        for it in its:
            q, base = it["question"], {"set": s, "id": it["id"], "language": it.get("language"), "noise_type": it.get("noise_type")}
            other_lang = detect_arabic(q) or detect_english(q)
            fq = q
            if other_lang:
                if (s, it["id"]) not in translations:
                    why = "Arabic: needs the LLM translation" if detect_arabic(q) else "detected as English: needs the LLM translation"
                    skipped.append(base | {"reason": why + " (no French query in diagnostics/translations.json)"})
                    continue
                fq = translations[(s, it["id"])]
            gold = rae.gold_set(it)
            dense = _semantic_search(vs, fq, None, k=MMR_K)
            bm = _bm25_search(bm25, fq, None, k=BM25_K) if bm25 else []
            lists = [dense, bm]
            orig = []
            if other_lang:                               # retriever.hybrid_retrieve: original-language dense list, k=10
                orig = _semantic_search(vs, q, None, k=10)
                lists.append(orig)
            fused_full = _deduplicate(_fuse(lists))
            path, number, pool = "hybrid", None, fused_full[:RERANK_POOL]
            m = _ARTICLE_RE.search(q)
            if m:
                direct = _direct_lookup(vs, m.group(0), None, MMR_K)
                if direct:
                    path, number, pool = "direct", m.group(0), direct
            reranked = rerank(fq, pool, top_k=len(pool))
            r = base | {"path": path, "direct_number": number, "query_source": "translation" if other_lang else "as written",
                        "rank_dense": fc.rank_of(keys(dense), gold), "n_dense": len(dense),
                        "rank_bm25": fc.rank_of(keys(bm), gold), "n_bm25": len(bm),
                        "rank_orig_lang": fc.rank_of(keys(orig), gold) if other_lang else None,
                        "rank_fused_full": fc.rank_of(keys(fused_full), gold), "n_fused_full": len(fused_full),
                        "pool_size": len(pool), "rank_pool": fc.rank_of(keys(pool), gold),
                        "rank_reranked": fc.rank_of(keys(reranked), gold)}
            r["class"] = fc.classify(r["rank_reranked"], r["rank_fused_full"], path)
            r["list_flag"] = fc.list_flag(r["rank_dense"], r["rank_bm25"])
            saved = (rows11[s].get(it["id"]) or {}).get("metrics") or {}
            r["saved_v11_hit6"] = saved.get("gold_hit")
            v2 = rows2[s].get(it["id"]) or {}
            r["v2_status"] = v2.get("status")
            r["v2_evidence_has_gold"] = any(rae.key(c) in gold for c in v2.get("evidence_ids") or []) if v2 else None
            records.append(r)
            print(f"  {s} {it['id']}: {r['class']} ({path})", flush=True)
    return records, skipped


def v2_on_saved_misses(items: dict, rows11: dict, rows2: dict) -> dict:
    """All answerable items (every language): the saved v1.1 row missed the gold (gold_hit false); did the saved v2
    row have it in its evidence?"""
    out = {}
    for s, its in items.items():
        n_miss = n_v2 = n_v2_missing_row = 0
        for it in its:
            m = ((rows11[s].get(it["id"]) or {}).get("metrics") or {})
            if m.get("gold_hit") is False:
                n_miss += 1
                v2 = rows2[s].get(it["id"])
                if not v2:
                    n_v2_missing_row += 1
                elif any(rae.key(c) in rae.gold_set(it) for c in v2.get("evidence_ids") or []):
                    n_v2 += 1
        out[s] = {"v11_missed": n_miss, "v2_had_gold": n_v2, "v2_row_missing": n_v2_missing_row}
    return out


def index_docs(vs) -> list[dict]:
    got = vs._collection.get(include=["documents", "metadatas"])
    return [{"id": m.get("chunk_id"), "source": m.get("source"), "article": m.get("article"), "text": t or ""}
            for t, m in zip(got["documents"], got["metadatas"])]


def cross_references(docs: list, rows2: dict) -> dict:
    by_id = {d["id"]: d for d in docs}
    kinds = {d["id"]: fc.references(d["text"]) for d in docs}
    n = len(docs)
    index = {k: sum(bool(v[k]) for v in kinds.values()) for k in fc.KINDS}
    index["any"] = sum(any(v.values()) for v in kinds.values())
    pairs, distinct = [], set()
    for s, rows in rows2.items():
        for r in rows.values():
            ev = {rae.key(c) for c in r.get("evidence_ids") or []}
            for cid in r.get("cited") or []:
                d = by_id.get(cid)
                if d is None:
                    pairs.append({"set": s, "cited": cid, "in_index": False})
                    continue
                stem, lab = rae.key(cid)
                refs = kinds[cid]
                resolved = {(stem, x) for x in fc.resolve(d["text"], lab)}
                pairs.append({"set": s, "cited": cid, "in_index": True, "any": any(refs.values()),
                              "numbered": bool(refs["numbered"]), "relative": bool(refs["relative"]), "vague": bool(refs["vague"]),
                              "resolved": len(resolved), "not_in_evidence": len(resolved - ev - {(stem, lab)})})
                distinct.add(cid)
    ok = [p for p in pairs if p["in_index"]]
    return {"index": {"n_chunks": n, **{k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in index.items()}},
            "cited_pairs": {"n": len(pairs), "not_in_index": len(pairs) - len(ok), "distinct_cited": len(distinct),
                            **{k: sum(p[k] for p in ok) for k in ("any", "numbered", "relative", "vague")},
                            "with_resolvable_reference": sum(p["resolved"] > 0 for p in ok),
                            "with_referenced_not_in_evidence": sum(p["not_in_evidence"] > 0 for p in ok)},
            "per_set": {s: {"n": sum(p["set"] == s and p["in_index"] for p in pairs),
                            "any": sum(p["set"] == s and p.get("any", False) for p in pairs),
                            "with_referenced_not_in_evidence": sum(p["set"] == s and p.get("not_in_evidence", 0) > 0 for p in pairs)}
                        for s in SETS}}


def chunk_structure(docs: list, tokenizer, max_len: int) -> dict:
    starts = Counter("article header" if fc.HEADER.match(d["text"]) else "section/chapter title" if TITLE.match(d["text"])
                     else "other" for d in docs)
    chars = [len(d["text"]) for d in docs]
    tokens = [len(tokenizer(d["text"], add_special_tokens=True)["input_ids"]) for d in docs]
    stat = lambda xs: {"min": min(xs), "median": statistics.median(xs), "max": max(xs), "mean": round(statistics.fmean(xs), 1),
                       "p90": sorted(xs)[int(0.9 * len(xs)) - 1]}
    ids = sorted(d["id"] for d in docs)
    by_id = {d["id"]: d for d in docs}
    sample = [{"id": i, "first_300": by_id[i]["text"][:300]} for i in random.Random(SEED).sample(ids, 10)]
    empty = sorted(d["id"] for d in docs if not d["text"].strip())
    no_pdf = sorted(d["id"] for d in docs if "(no PDF text)" in d["text"])
    return {"n_chunks": len(docs), "starts": dict(starts), "title_anywhere": sum(bool(TITLE_ANY.search(d["text"])) for d in docs),
            "chars": stat(chars), "tokens": stat(tokens), "max_seq_length": max_len,
            "over_max_seq_length": sum(t > max_len for t in tokens), "sample": sample, "empty": empty, "no_pdf_text": no_pdf}


# ---------------------------------------------------------------- markdown

def table(title: str, header: list, rows: list) -> list[str]:
    cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ⏎ ")
    return [f"### {title}", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows] + [""]


def pct(n, d):
    return f"{n} ({100 * n / d:.1f} %)" if d else f"{n}"


def class_rows(groups: dict) -> list:
    rows = []
    for g, recs in groups.items():
        s = fc.summarize(recs)
        rows.append([g, s["n"]] + [pct(s["classes"][c]["n"], s["n"]) for c in fc.CLASSES] +
                    [pct(s["hit"][k]["n"], s["n"]) for k in (6, 8, 10)])
    return rows


def markdown(doc: dict) -> str:
    recs, skipped = doc["records"], doc["skipped"]
    L = ["# Retrieval diagnostic (v1.1 path, db_v1_1, Stage 1, Stage 2, noisy set: answerable items, all languages)", ""]
    L += table("Definitions", ["Term", "Definition"],
               [[fc.CLASS_LABELS[c], {"top6": "gold rank 1-6 in the cross-encoder order of the pool used",
                                      "rerank_7_10": "gold in the pool, cross-encoder rank 7-10",
                                      "rerank_11_20": "gold in the pool, cross-encoder rank 11-20",
                                      "fused_rank_gt20": "hybrid path: gold in the dense or BM25 list, fused rank > RERANK_POOL (20)",
                                      "absent_from_pool": "gold in no candidate list (hybrid), or not in the direct-lookup pool (direct)"}[c]]
                for c in fc.CLASSES] +
               [["direct path", "a number in the question (pipeline._ARTICLE_RE = \\b\\d+\\b) and retriever._direct_lookup returned documents: they are the pool"],
                ["hybrid path", "dense (k=20) + BM25 (k=15) lists on the French query (+ dense k=10 on the original question for Arabic/English), reciprocal rank fusion, deduplication, first 20"],
                ["French query", "French items: the question as written; Arabic/English items: translate.translate_to_french, run once and stored in diagnostics/translations.json"],
                ["Hit@k", "gold within the first k of the cross-encoder order of the pool used (the system keeps 6)"],
                ["gold match", "source stem + article label (run_answer_eval.key / gold_set)"]])
    L += table("Items analysed and skipped", ["Set", "Answerable items", "Analysed", "Skipped", "Skip reasons"],
               [[SETS[s], doc["n_items"][s], sum(r["set"] == s for r in recs), sum(k["set"] == s for k in skipped),
                 "; ".join(f"{why}: {n}" for why, n in Counter(k["reason"] for k in skipped if k["set"] == s).items()) or "-"]
                for s in SETS])
    L += table("Skipped items by language", ["Set", "Language", "Skipped"],
               [[SETS[s], lang, n] for (s, lang), n in sorted(Counter((k["set"], k["language"]) for k in skipped).items())]
               or [["-", "-", 0]])
    hdr = ["Group", "n"] + [fc.CLASS_LABELS[c] for c in fc.CLASSES] + ["Hit@6", "Hit@8", "Hit@10"]
    L += table("Funnel classes per set", hdr, class_rows({SETS[s]: [r for r in recs if r["set"] == s] for s in SETS}))
    by_lang = defaultdict(list)
    for r in recs:
        by_lang[f"{SETS[r['set']]}, {r['language']}"].append(r)
    L += table("Funnel classes per set and language", hdr, class_rows(dict(sorted(by_lang.items()))))
    L += table("Funnel classes per language (all three sets)", hdr,
               class_rows({lang: [r for r in recs if r["language"] == lang] for lang in sorted({r["language"] for r in recs})}))
    xl = [r for r in recs if r.get("query_source") == "translation"]
    if xl:
        L += table("Original-language dense list (Arabic/English items, k=10)",
                   ["Set", "items", "gold in the original-language list", "gold only there (not in dense or BM25 on the French query)"],
                   [[SETS[s], len(rs := [r for r in xl if r["set"] == s]), sum(r["rank_orig_lang"] is not None for r in rs),
                     sum(r["rank_orig_lang"] is not None and r["list_flag"] == "absent_both" for r in rs)] for s in SETS])
    by_noise = defaultdict(list)
    for r in recs:
        if r["set"] == "N":
            by_noise[r["noise_type"]].append(r)
    L += table("Funnel classes per noise type (noisy set)", hdr, class_rows(dict(sorted(by_noise.items()))))
    L += table("Gold in the candidate lists (hybrid lists computed for every analysed item)",
               ["Set", "n", "in dense and BM25", "absent from dense only", "absent from BM25 only", "absent from both"],
               [[SETS[s], (x := fc.summarize([r for r in recs if r["set"] == s]))["n"]] +
                [pct(x["flags"][f]["n"], x["n"]) for f in ("present_both", "absent_dense_only", "absent_bm25_only", "absent_both")]
                for s in SETS])
    L += table("Retrieval path per set", ["Set", "n", "hybrid", "direct (number in the question)", "direct: gold in the pool",
                                          "direct: gold would be in the hybrid pool"],
               [[SETS[s], len(rs := [r for r in recs if r["set"] == s]), pct(sum(r["path"] == "hybrid" for r in rs), len(rs)),
                 pct(sum(r["path"] == "direct" for r in rs), len(rs)),
                 sum(r["path"] == "direct" and r["rank_pool"] is not None for r in rs),
                 sum(r["path"] == "direct" and r["rank_fused_full"] is not None and r["rank_fused_full"] <= 20 for r in rs)] for s in SETS])
    L += table("Reproduced Hit@6 against the saved v1.1 rows", ["Set", "n", "same", "different", "saved row missing"],
               [[SETS[s], len(rs := [r for r in recs if r["set"] == s]),
                 sum(r["saved_v11_hit6"] is not None and bool(fc.hit_at(r["rank_reranked"], 6)) == r["saved_v11_hit6"] for r in rs),
                 sum(r["saved_v11_hit6"] is not None and bool(fc.hit_at(r["rank_reranked"], 6)) != r["saved_v11_hit6"] for r in rs),
                 sum(r["saved_v11_hit6"] is None for r in rs)] for s in SETS])
    diff = [r for r in recs if r["saved_v11_hit6"] is not None and bool(fc.hit_at(r["rank_reranked"], 6)) != r["saved_v11_hit6"]]
    L += table("Items whose reproduced Hit@6 differs from the saved v1.1 row",
               ["Set", "Id", "Language", "French query", "Saved v1.1 Hit@6", "Reproduced Hit@6", "Reproduced class"],
               [[SETS[r["set"]], r["id"], r["language"], r.get("query_source", "as written"), r["saved_v11_hit6"],
                 bool(fc.hit_at(r["rank_reranked"], 6)), fc.CLASS_LABELS[r["class"]]] for r in diff] or [["-"] * 7])
    L += table("v2 evidence where the reproduced v1.1 funnel missed (analysed items, not in top 6)",
               ["Set", "v1.1 missed", "v2 evidence has the gold", "v2 status clarify/refuse", "v2 row missing"],
               [[SETS[s], len(ms := [r for r in recs if r["set"] == s and r["class"] != "top6"]),
                 sum(bool(r["v2_evidence_has_gold"]) for r in ms), sum(r["v2_status"] in ("clarify", "refuse") for r in ms),
                 sum(r["v2_status"] is None for r in ms)] for s in SETS])
    L += table("v2 evidence where the saved v1.1 row missed (all answerable items, every language)",
               ["Set", "saved v1.1 missed", "v2 evidence has the gold", "v2 row missing"],
               [[SETS[s], x["v11_missed"], x["v2_had_gold"], x["v2_row_missing"]] for s, x in doc["v2_saved"].items()])
    xr = doc["cross_references"]
    n = xr["index"]["n_chunks"]
    L += table("Cross-references in the index", ["Reference kind", "Chunks", "Share of the index"],
               [[k, xr["index"][k]["n"], f"{xr['index'][k]['pct']} %"] for k in ("numbered", "relative", "vague", "any")] +
               [["chunks in the index", n, "100 %"]])
    c = xr["cited_pairs"]
    m = c["n"] - c["not_in_index"]
    L += table("Cross-references in the articles cited by saved v2 answers ((answer, cited id) pairs)", ["Measure", "Value"],
               [["pairs", c["n"]], ["cited id not in the index", c["not_in_index"]], ["distinct cited ids", c["distinct_cited"]],
                ["pairs whose cited article contains a reference", pct(c["any"], m)],
                ["... numbered (article N, articles N à M)", pct(c["numbered"], m)],
                ["... relative (article précédent / suivant)", pct(c["relative"], m)],
                ["... vague (ci-dessus, susvisé, précité, ci-après)", pct(c["vague"], m)],
                ["pairs with a resolvable reference", pct(c["with_resolvable_reference"], m)],
                ["pairs with a referenced article not in that answer's evidence", pct(c["with_referenced_not_in_evidence"], m)]])
    L += table("Cross-references in cited articles per set", ["Set", "pairs", "with a reference", "referenced article not in the evidence"],
               [[SETS[s], x["n"], pct(x["any"], x["n"]), pct(x["with_referenced_not_in_evidence"], x["n"])] for s, x in xr["per_set"].items()])
    ch = doc["chunks"]
    L += table("Chunk start", ["Start", "Chunks", "Share"],
               [[k, ch["starts"].get(k, 0), f"{100 * ch['starts'].get(k, 0) / ch['n_chunks']:.1f} %"]
                for k in ("article header", "section/chapter title", "other")] +
               [["a section/chapter/title line anywhere in the chunk", ch["title_anywhere"], f"{100 * ch['title_anywhere'] / ch['n_chunks']:.1f} %"]])
    L += table("Chunk length", ["Unit", "min", "median", "mean", "p90", "max"],
               [["characters"] + [ch["chars"][k] for k in ("min", "median", "mean", "p90", "max")],
                ["tokens (embedding model tokenizer, special tokens included)"] + [ch["tokens"][k] for k in ("min", "median", "mean", "p90", "max")]])
    L += table("Chunks longer than the embedding model's max_seq_length", ["max_seq_length", "chunks over it", "share"],
               [[ch["max_seq_length"], ch["over_max_seq_length"], f"{100 * ch['over_max_seq_length'] / ch['n_chunks']:.1f} %"]])
    L += table(f"Ten random chunks (seed {SEED}), first 300 characters", ["Chunk id", "First 300 characters"],
               [[x["id"], x["first_300"]] for x in ch["sample"]])
    L += table("Empty chunks and '(no PDF text)' chunks", ["Kind", "Chunks", "Ids"],
               [["empty text", len(ch["empty"]), ", ".join(ch["empty"]) or "-"], ["'(no PDF text)'", len(ch["no_pdf_text"]), ", ".join(ch["no_pdf_text"]) or "-"]])
    L += table("Run", ["Item", "Value"], [[k, v] for k, v in doc["run"].items()])
    chk = ROOT / "diagnostics" / "chroma_open_check.json"
    if chk.exists():
        c = json.loads(chk.read_text(encoding="utf-8"))
        L += table("Index file after the run (diagnostics/chroma_open_check.py, one Chroma open of a temporary copy of db_v1_1)",
                   ["Measure", "Value"],
                   [["tables in the index", c["copy_tables"]], ["tables unchanged by one open", c["copy_tables_unchanged"]]] +
                   [[f"table changed by one open: {t}", f"rows {v['rows_before']} -> {v['rows_after']}"]
                    for t, v in c["copy_tables_changed_by_one_open"].items()] +
                   [["acquire_write rows in the real index now (read-only count)", c["real_index_acquire_write_rows_now"]],
                    ["embeddings rows in the real index (read-only count)", c["real_index_embeddings_rows"]]])
    return "\n".join(L)


def open_index(path: Path):
    from langchain_chroma import Chroma
    from pipeline import get_embeddings
    return Chroma(persist_directory=str(path), embedding_function=get_embeddings())


def main():
    from pipeline import get_embeddings
    from config import DB_DIR, MMR_K, BM25_K, RERANK_POOL, RERANK_TOP_K, EMBEDDING_MODEL, RERANKER_MODEL
    import harness_core as hc
    db = Path(DB_DIR) / "chroma.sqlite3"
    before = sha(db)
    items = items_by_set()
    rows11 = {s: saved_rows(s, "v1.1") for s in SETS}
    rows2 = {s: saved_rows(s, "v2") for s in SETS}
    opened = Path(sys.argv[sys.argv.index("--db") + 1]) if "--db" in sys.argv else Path(DB_DIR)
    vs = open_index(opened)
    records, skipped = funnel(vs, items, rows11, rows2)
    docs = index_docs(vs)
    client = get_embeddings()._client
    doc = {"records": records, "skipped": skipped, "n_items": {s: len(v) for s, v in items.items()},
           "summary": {s: fc.summarize([r for r in records if r["set"] == s]) for s in SETS},
           "v2_saved": v2_on_saved_misses(items, rows11, rows2),
           "cross_references": cross_references(docs, rows2),
           "chunks": chunk_structure(docs, client.tokenizer, client.max_seq_length)}
    after = sha(db)
    fp = hc.system_fingerprint(ROOT, Path(DB_DIR))
    doc["run"] = {"index": f"{Path(DB_DIR).name} ({len(docs)} chunks)",
                  "index opened": "byte copy of " + Path(DB_DIR).name if opened != Path(DB_DIR) else Path(DB_DIR).name,
                  f"{Path(DB_DIR).name}/chroma.sqlite3 bytes identical before and after the run": before == after,
                  "French queries of Arabic/English items": f"diagnostics/translations.json ({len(load_translations())} items)",
                  "system fingerprint": fp["sha256"], "embedding model": EMBEDDING_MODEL, "reranker": RERANKER_MODEL,
                  "MMR_K / BM25_K / RERANK_POOL / RERANK_TOP_K": f"{MMR_K} / {BM25_K} / {RERANK_POOL} / {RERANK_TOP_K}",
                  "LLM calls in this run": "0 (translations made once by diagnostics/translate_skipped.py)"}
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(doc) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}; analysed {len(records)}, skipped {len(skipped)}; index unchanged: {before == after}")


def markdown_only():
    """Rewrite the .md from the saved .json (no index opened)."""
    doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    run = doc["run"]
    if "index sha256 before = after" in run:          # key name of the first run
        run = {("index file bytes identical before and after the run" if k == "index sha256 before = after" else k): v
               for k, v in run.items()}
        doc["run"] = run
        OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(doc) + "\n", encoding="utf-8")
    print(f"rewrote {OUT_MD.name} from {OUT_JSON.name}")


if __name__ == "__main__":
    markdown_only() if "--md-only" in sys.argv else main()
