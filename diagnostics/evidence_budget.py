"""Evidence budget on the 110 development items (db_v1_2, v1.2 top 6 of diagnostics/compare_v12.json). Local only:
no LLM call, no embedding (index read from a byte copy, tokenizer of the embedding model).

    python diagnostics/evidence_budget.py --v12 COPY_OF_DB_V1_2   -> diagnostics/evidence_budget.json, EVIDENCE_BUDGET.md
"""
import sys
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(Path(__file__).resolve().parent)]
import config as C
import run_answer_eval as rae
from retrieval_diagnostic import items_by_set, SETS, table, pct

OUT_JSON = ROOT / "diagnostics" / "evidence_budget.json"
OUT_MD = ROOT / "diagnostics" / "EVIDENCE_BUDGET.md"
DEV = "development data, not evidence of generalisation"
CAPS = [300, 600, None]
HEADERS = ["full", "deepest", "none"]


def choose_cap(lost: dict) -> int | None:
    """lost: cap -> items lost versus unlimited. The smallest cap losing at most 1 item, else None (unlimited)."""
    for cap in sorted(c for c in lost if c is not None):
        if lost[cap] <= 1:
            return cap
    return None


def stat(xs):
    xs = sorted(xs)
    return {"mean": round(statistics.fmean(xs), 1), "median": statistics.median(xs),
            "p90": xs[max(0, int(0.9 * len(xs)) - 1)], "max": xs[-1], "total": sum(xs)}


def load_index(path):
    from langchain_chroma import Chroma
    from retriever import _corpus, _to_article
    from langchain_core.documents import Document
    vs = Chroma(persist_directory=str(path))
    got = _corpus(vs)
    by_id = {}
    for t, m in zip(got["documents"], got["metadatas"]):
        d = _to_article(Document(page_content=t or "", metadata=m))
        by_id.setdefault(d.metadata["chunk_id"], d)
    return vs, by_id


def main():
    from transformers import AutoTokenizer
    from langchain_core.documents import Document
    from pipeline import evidence_text, cap_added
    from retriever import expand_cross_references
    v12 = Path(sys.argv[sys.argv.index("--v12") + 1])
    tok = AutoTokenizer.from_pretrained(C.EMBEDDING_MODEL)
    length = lambda t: len(tok(t, add_special_tokens=True)["input_ids"])
    vs, by_id = load_index(v12)
    items = {(s, it["id"]): it for s, its in items_by_set().items() for it in its}
    recs = json.loads((ROOT / "diagnostics" / "compare_v12.json").read_text(encoding="utf-8"))["records"]["v1.2"]
    C.XREF_TOKEN_CAP = None
    rows = []
    for r in recs:
        it = items[(r["set"], r["id"])]
        gold = rae.gold_set(it)
        top6 = [by_id[i] for i in r["top6_ids"]]
        added = expand_cross_references(vs, top6)
        row = {"set": r["set"], "id": r["id"], "language": r["language"], "noise_type": r["noise_type"], "header": {}, "cap": {}}
        for h in HEADERS:
            C.EVIDENCE_CONTEXT_HEADER = h
            hdr = lambda d: length(evidence_text(d)) - length(evidence_text(Document(page_content=d.page_content,
                                                                                        metadata={**d.metadata, "context": ""})))
            row["header"][h] = {"top6": sum(hdr(d) for d in top6), "top6_plus_added": sum(hdr(d) for d in top6 + added),
                                "chars_top6": sum(len(evidence_text(d)) - len(d.page_content) for d in top6 if d.metadata.get("parts", 1) <= 1)}
        C.EVIDENCE_CONTEXT_HEADER = "deepest"
        for cap in CAPS:
            kept = cap_added(added, cap, length) if cap is not None else added
            ev = top6 + kept
            have = {d.metadata["chunk_id"] for d in ev}
            n_missing = 0
            for d in top6:
                res = {x.metadata["chunk_id"] for x in expand_cross_references(vs, [d], per_article=99, total=99)}
                n_missing += bool(res - have)
            row["cap"][str(cap)] = {"added": len(kept), "truncated": sum(bool(d.metadata.get("xref_truncated")) for d in kept),
                                    "tokens": sum(length(evidence_text(d)) for d in kept),
                                    "gold_in_evidence": any(rae.key(d.metadata["chunk_id"]) in gold for d in ev),
                                    "top6_with_missing_ref": n_missing}
        rows.append(row)
    lost = {cap: sum(r["cap"]["None"]["gold_in_evidence"] and not r["cap"][str(cap)]["gold_in_evidence"] for r in rows) for cap in CAPS}
    chosen = choose_cap(lost)
    doc = {"rows": rows, "lost_vs_unlimited": {str(k): v for k, v in lost.items()}, "chosen_cap": chosen,
           "header_default": "deepest", "decision_rule": "smallest cap losing at most 1 item versus unlimited"}
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(doc) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}; lost vs unlimited {lost}; chosen cap {chosen}")


def groups(rows):
    g = {SETS[s]: [r for r in rows if r["set"] == s] for s in SETS}
    g.update({f"language {l}": [r for r in rows if r["language"] == l] for l in sorted({r["language"] for r in rows})})
    g["all"] = rows
    return g


def markdown(doc):
    rows = doc["rows"]
    L = [f"# Evidence budget: context header and cross-reference cap — {DEV}", ""]
    out = []
    for h in HEADERS:
        for g, rs in groups(rows).items():
            a, b = stat([r["header"][h]["top6"] for r in rs]), stat([r["header"][h]["top6_plus_added"] for r in rs])
            out.append([h + (" (default)" if h == doc["header_default"] else ""), g, len(rs), a["mean"], a["median"], a["p90"], a["max"],
                        b["mean"], b["p90"]])
    L += table(f"'Contexte :' header tokens per question (embedding tokens) ({DEV})",
               ["Header", "Group", "n", "top 6: mean", "median", "p90", "max", "top 6 + added: mean", "p90"], out)
    out = []
    for cap in CAPS:
        k = str(cap)
        for g, rs in groups(rows).items():
            t = stat([r["cap"][k]["tokens"] for r in rs])
            n6 = 6 * len(rs)
            out.append(["unlimited" if cap is None else cap, g, len(rs),
                        t["mean"], t["median"], t["p90"], t["max"], t["total"],
                        sum(r["cap"][k]["added"] for r in rs), sum(r["cap"][k]["truncated"] for r in rs),
                        pct(sum(r["cap"][k]["gold_in_evidence"] for r in rs), len(rs)),
                        pct(sum(r["cap"][k]["top6_with_missing_ref"] for r in rs), n6)])
    L += table(f"Cross-reference expansion with a token cap per question, header 'deepest' ({DEV})",
               ["Cap (tokens)", "Group", "n", "added tokens mean", "median", "p90", "max", "total", "added articles",
                "articles cut", "gold in evidence with cross-refs", "top-6 articles with a referenced article missing"], out)
    L += table(f"Decision ({DEV})", ["Rule (fixed before the run)", "Items lost vs unlimited", "Chosen cap"],
               [[doc["decision_rule"], ", ".join(f"{'unlimited' if k == 'None' else k}: {v}" for k, v in doc["lost_vs_unlimited"].items()),
                 "unlimited" if doc["chosen_cap"] is None else doc["chosen_cap"]]])
    return "\n".join(L)


if __name__ == "__main__":
    main()
