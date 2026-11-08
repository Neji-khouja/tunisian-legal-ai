"""Tests of eval/retrieval_only.py on a stub index (no model, no LLM, no real index).

    python eval/test_retrieval_only.py
"""
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from langchain_core.documents import Document
import retrieval_only as ro
import reranker as RK

FAILS, N = [], {"n": 0}


def check(name, ok, detail=""):
    N["n"] += 1
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"   {detail}"))
    if not ok:
        FAILS.append(name)


def words(t):
    return set(w.strip(".,;:?!'").lower() for w in t.split())


class StubVS:
    """similarity_search: documents ranked by word overlap with the query (filter on metadata "article")."""
    def __init__(self, docs):
        self.docs = docs
        self._collection = SimpleNamespace(get=lambda include=None: {"documents": [d.page_content for d in docs],
                                                                      "metadatas": [d.metadata for d in docs]})

    def similarity_search(self, query, k=4, filter=None):
        ds = self.docs
        if filter:
            conds = filter.get("$and", [filter])
            for c in conds:
                (f, v), = c.items()
                ds = [d for d in ds if str(d.metadata.get(f)) == str(v["$eq"])]
        q = words(query)
        return sorted(ds, key=lambda d: -len(q & words(d.page_content)))[:k]


TOPICS = ["vol simple peine emprisonnement", "divorce consentement mutuel juge", "contrat de travail licenciement préavis",
          "société anonyme capital minimum", "nationalité acquisition mariage", "enfant protection délégué signalement"]


def corpus():
    docs = []
    for n in range(1, 41):
        t = TOPICS[n % len(TOPICS)]
        docs.append(Document(page_content=f"Article {n}.- texte {n} sur {t} numéro{n}",
                             metadata={"chunk_id": f"Code/s01/{n}", "article": str(n), "source": "Code.pdf", "law_type": "penal"}))
    return docs


ITEMS = [
    {"id": "f1", "language": "fr", "question": "peine du vol simple numéro6", "gold": [{"source": "Code.pdf", "article": "6"}], "set": "S1"},
    {"id": "f2", "language": "fr", "question": "divorce par consentement mutuel numéro8", "gold": ["Code/s01/8"], "set": "S1"},
    {"id": "f3", "language": "fr", "question": "rien à voir xyz", "gold": [{"source": "Code.pdf", "article": "99"}], "set": "N", "noise_type": "describe"},
    {"id": "f4", "language": "fr", "question": "Que dit l'article 12 ?", "gold": [{"source": "Code.pdf", "article": "12"}], "set": "S2"},
    {"id": "f5", "language": "fr", "question": "licenciement signé le 2 janvier 2000 numéro9", "gold": [{"source": "Code.pdf", "article": "9"}], "set": "S2"},
    {"id": "a1", "language": "ar", "question": "ما هي عقوبة السرقة", "gold": [{"source": "Code.pdf", "article": "18"}], "set": "N", "noise_type": "typos"},
    {"id": "e1", "language": "en", "question": "What is the penalty for theft?", "gold": [{"source": "Code.pdf", "article": "1"}], "set": "N"},
    {"id": "e2", "language": "en", "question": "What does the divorce rule say?", "gold": [{"source": "Code.pdf", "article": "2"}], "set": "S1"},
]
TRANSLATIONS = {"items": [{"id": "a1", "query": "ما هي عقوبة السرقة", "french_query": "peine du vol simple numéro18"},
                          {"id": "e2", "query": "an older wording of the question", "french_query": "divorce"}]}


def run(preset, vs):
    ro.apply_config(preset, {})
    tr = ro.load_translations(write_tmp(TRANSLATIONS))
    return ro.evaluate(ITEMS, tr, ro.real_backend(vs))


def write_tmp(obj):
    f = Path(tempfile.mkdtemp()) / "x.json"
    f.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return f


def main():
    real = RK._reranker
    # stub cross-encoder: word overlap with the query (the gold "numéroN" word decides the order)
    RK._reranker = SimpleNamespace(predict=lambda pairs, **kw: [len(words(q) & words(t)) for q, t in pairs])
    try:
        vs = StubVS(corpus())
        recs, skipped = run("baseline", vs)
        by = {r["id"]: r for r in recs}
        check("items: exam-style and chunk-id gold both read", ro.gold_keys(ITEMS[0]) == {("Code", "6")} and ro.gold_keys(ITEMS[1]) == {("Code", "8")})
        check("skip: English item without a cached translation", any(s["id"] == "e1" and "no cached translation" in s["reason"] for s in skipped))
        check("skip: translation cached for another question text", any(s["id"] == "e2" and "another question" in s["reason"] for s in skipped))
        check("analysed: 6 items", sorted(by) == ["a1", "f1", "f2", "f3", "f4", "f5"], sorted(by))
        check("gold rank 1 when the gold word is in the query", by["f1"]["rank_reranked"] == 1 and by["f1"]["class"] == "top6")
        check("Arabic item uses the cached French query", by["a1"]["query"] == "translation" and by["a1"]["rank_reranked"] == 1)
        check("absent gold -> absent_from_pool, no hit", by["f3"]["class"] == "absent_from_pool" and by["f3"]["hit10"] == 0)
        check("baseline: a date number triggers the v1.1 direct lookup", by["f5"]["path"] == "direct", by["f5"]["path"])
        check("baseline: the article number triggers the direct lookup", by["f4"]["path"] == "direct" and by["f4"]["rank_reranked"] == 1)
        recs2, _ = run("v1.2", vs)
        by2 = {r["id"]: r for r in recs2}
        check("v1.2: a date does not trigger, 'article 12' -> union", by2["f5"]["path"] == "hybrid" and by2["f4"]["path"] == "direct+hybrid")
        s = ro.summarize(recs)
        check("summary: Hit@6 over all analysed items", s["all"]["hit"][6]["n"] == sum(r["hit6"] for r in recs) and s["all"]["n"] == 6)
        check("summary: groups per language, set and noise type", {"language fr", "language ar", "set S1", "noise describe"} <= set(s))
        check("summary: classes add up", sum(s["all"]["classes"].values()) == 6)
        check("bootstrap: all hits -> CI [1, 1]", ro.boot_ci([1] * 20) == (1.0, 1.0))
        check("bootstrap: deterministic with the seed", ro.boot_ci([0, 1] * 10) == ro.boot_ci([0, 1] * 10))
        lo, hi = ro.boot_ci([0, 1] * 50)
        check("bootstrap: CI brackets the mean", lo < 0.5 < hi and 0.35 < lo and hi < 0.65, (lo, hi))
        check("bootstrap: 5000 resamples, seed 20261022", ro.N_BOOT == 5000 and ro.SEED == 20261022)
        check("sign test: 5 gained 0 lost -> p = 0.0625", abs(ro.sign_test(5, 0) - 0.0625) < 1e-12)
        check("sign test: 3 / 1 -> p = 0.625", abs(ro.sign_test(3, 1) - 0.625) < 1e-12)
        check("sign test: no discordant item -> p = 1", ro.sign_test(0, 0) == 1.0)
        A = [{"id": i, "hit6": h, "hit8": h, "hit10": h} for i, h in (("x", 1), ("y", 0), ("z", 0), ("w", 1))]
        B = [{"id": i, "hit6": h, "hit8": h, "hit10": h} for i, h in (("x", 1), ("y", 1), ("z", 1), ("w", 0), ("v", 1))]
        p = ro.paired(A, B)
        check("paired: items in both only", p["n"] == 4 and p["only_in_b"] == ["v"])
        check("paired: gained / lost lists", p[6]["gained"] == ["y", "z"] and p[6]["lost"] == ["w"])
        check("paired: mean difference and sign test", abs(p[6]["diff"] - 0.25) < 1e-12 and abs(p[6]["sign_p"] - 1.0) < 1e-12)
        lo, hi = p[6]["ci"]
        check("paired: CI of the mean difference", lo <= 0.25 <= hi)
        md = ro.markdown({"A": {"summary": s, "skipped": skipped}}, p)
        check("markdown: tables and skipped reasons", "| all | 6 |" in md and "e1" in md and "sign test p" in md)
        check("config: unknown switch refused", raises(lambda: ro.apply_config("v1.2", {"NOPE": 1}), ValueError))
        check("config: overrides parsed as JSON", ro.parse_sets(["XREF_TOKEN_CAP=600", "BM25_TOKENIZER=stem", "XREF_EXPAND=false"]) ==
              {"XREF_TOKEN_CAP": 600, "BM25_TOKENIZER": "stem", "XREF_EXPAND": False})
        src = Path(tempfile.mkdtemp()) / "idx"
        src.mkdir()
        (src / "chroma.sqlite3").write_bytes(b"x")
        opened, tmp = ro.open_copy(src, opener=lambda p: p)
        check("index opened from a temporary byte copy", opened != src and (opened / "chroma.sqlite3").read_bytes() == b"x"
              and str(opened).startswith(str(tmp)))
    finally:
        RK._reranker = real
        ro.apply_config("v1.2", {})
    print(f"\n{N['n'] - len(FAILS)}/{N['n']} checks passed")
    return 1 if FAILS else 0


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
