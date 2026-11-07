"""Funnel arithmetic on a stub index (no model, no index file, no network).

    python diagnostics/test_funnel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import funnel_core as fc

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or not detail else f": {detail}"))


def stub_record(gold, dense, bm25, pool_size=20, reranked=None, path="hybrid", direct_pool=None):
    """A stub index: lists of article keys. Fusion stub = dense then bm25 order without duplicates (enough for ranks)."""
    fused = list(dict.fromkeys(dense + bm25))
    pool = direct_pool if path == "direct" else fused[:pool_size]
    order = reranked if reranked is not None else pool
    r = {"rank_dense": fc.rank_of(dense, gold), "rank_bm25": fc.rank_of(bm25, gold), "rank_fused_full": fc.rank_of(fused, gold),
         "rank_pool": fc.rank_of(pool, gold), "rank_reranked": fc.rank_of(order, gold), "path": path}
    r["class"] = fc.classify(r["rank_reranked"], r["rank_fused_full"], path)
    r["list_flag"] = fc.list_flag(r["rank_dense"], r["rank_bm25"])
    return r


def test_classes():
    g = {("Penal", "218")}
    a = [("Penal", str(i)) for i in range(300, 340)]                 # 40 non-gold keys
    gold = ("Penal", "218")
    recs = [stub_record(g, [gold] + a[:19], a[19:34]),                                       # top 1
            stub_record(g, a[:20], a[20:35]),                                                 # gold not in lists -> absent
            stub_record(g, a[:7] + [gold] + a[7:19], a[20:35], reranked=a[:7] + [gold] + a[7:19]),   # reranked 8
            stub_record(g, a[:14] + [gold] + a[14:19], a[20:35], reranked=a[:14] + [gold] + a[14:19]),   # reranked 15
            stub_record(g, a[:20], a[20:34] + [gold]),                                       # bm25 rank 15, fused 35 -> cut
            stub_record(g, a[:20], a[20:35], path="direct", direct_pool=a[:20]),             # direct, gold absent
            stub_record(g, a[:5] + [gold], [gold] + a[20:30], path="direct", direct_pool=[gold] + a[:5])]   # direct, top 1
    got = [r["class"] for r in recs]
    check("classes", got == ["top6", "absent_from_pool", "rerank_7_10", "rerank_11_20", "fused_rank_gt20", "absent_from_pool", "top6"], got)
    flags = [r["list_flag"] for r in recs]
    check("list flags", flags == ["absent_bm25_only", "absent_both", "absent_bm25_only", "absent_bm25_only", "absent_dense_only",
                                  "absent_both", "present_both"], flags)
    s = fc.summarize(recs)
    check("class counts and percentages", (s["n"], s["classes"]["top6"], s["classes"]["absent_from_pool"], s["classes"]["fused_rank_gt20"])
          == (7, {"n": 2, "pct": 28.6}, {"n": 2, "pct": 28.6}, {"n": 1, "pct": 14.3}), s["classes"])
    check("class counts add up to n", sum(v["n"] for v in s["classes"].values()) == s["n"])
    check("Hit@6, Hit@8, Hit@10", (s["hit"][6]["n"], s["hit"][8]["n"], s["hit"][10]["n"]) == (2, 3, 3), s["hit"])
    check("Hit@k is monotone and equals the class counts", s["hit"][6]["n"] == s["classes"]["top6"]["n"]
          and s["hit"][10]["n"] == s["classes"]["top6"]["n"] + s["classes"]["rerank_7_10"]["n"])
    check("paths", s["paths"]["direct"]["n"] == 2 and s["paths"]["hybrid"]["n"] == 5)
    check("rank_of: first gold, 1-based; None when absent", fc.rank_of(["a", "g", "g"], {"g"}) == 2 and fc.rank_of([], {"g"}) is None)
    check("empty summary", fc.summarize([])["n"] == 0 and fc.summarize([])["hit"][6]["pct"] is None)


def test_references():
    t = "Article 261.- Les peines prévues à l'article précédent et aux articles 263 à 265 sont doublées, sauf le cas visé ci-dessus."
    r = fc.references(t)
    check("own header not counted; numbered, relative and vague found",
          r["numbered"] and r["relative"] and r["vague"] and not any("261" in x for x in r["numbered"]), r)
    check("resolution: range expanded, précédent = n - 1, own label dropped",
          fc.resolve(t, "261") == {"260", "263", "264", "265"}, fc.resolve(t, "261"))
    check("suffix label", fc.resolve("Article 5.- Voir l'article 220 bis.", "5") == {"220-bis"})
    check("suivant on a suffixed label uses the base number", fc.resolve("Article 13-bis.- comme à l'article suivant.", "13-bis") == {"14"})
    check("no reference", not any(fc.references("Article 7.- Le contrat est nul.").values()))
    check("Arabic patterns", fc.references("الفصل 12.- تطبق أحكام الفصل 10 والفصل السابق والمذكور أعلاه.")["numbered"])


if __name__ == "__main__":
    test_classes()
    test_references()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
