"""Retrieval-only evaluation: gold rank and funnel class per question, Hit@6/8/10 with bootstrap CIs, and a paired
comparison of two (index, configuration) pairs. No LLM call, ever.

    python eval/retrieval_only.py --items ITEMS.json --translations diagnostics/translations.json \
        --index-a db_v1_1 --config-a baseline [--index-b db_v1_2 --config-b v1.2] [--set-b KEY=VALUE ...] \
        [--out OUT.json] [--md OUT.md]
    python eval/retrieval_only.py --export-dev diagnostics/dev_items.json     # the 110 development items

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
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

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


def real_backend(vs):
    """(question, French query, other language?) -> the lists of one retrieval, with the real code and config."""
    import config as C
    from pipeline import step_retrieve, evidence_docs, _ARTICLE_RE
    from retriever import _semantic_search, _bm25_search, _get_bm25, _fuse, _deduplicate, _direct_lookup, article_numbers
    from reranker import rerank

    def run(q, fq, other):
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
        reranked = rerank(fq, pool, top_k=len(pool))
        ev = evidence_docs(reranked[:6], vs)
        ids = lambda ds: [d.metadata["chunk_id"] for d in ds]
        return {"dense": ids(dense), "bm25": ids(bm), "orig": ids(orig), "fused": ids(fused), "pool": ids(pool),
                "reranked": ids(reranked), "evidence": ids(ev), "path": path}
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
        L = backend(it["question"], fq, other)
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
    a = ap.parse_args(argv)
    if a.export_dev:
        export_dev(a.export_dev)
        return 0
    if not (a.items and a.index_a):
        ap.error("--items and --index-a are required")
    items, tr = load_items(a.items), load_translations(a.translations)
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
