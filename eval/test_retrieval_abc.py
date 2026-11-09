"""Stub tests of eval/retrieval_only.py --abc (configs A, B, C on fresh questions). No model, no LLM, no real index.

    python eval/test_retrieval_abc.py
"""
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "agent"), str(ROOT / "diagnostics")]
import retrieval_only as ro
import reranker as RK
from test_retrieval_only import StubVS, corpus, words

FAILS, CNT = [], {"n": 0}


def check(name, ok, detail=""):
    CNT["n"] += 1
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"   {detail}"))
    if not ok:
        FAILS.append(name)


ITEMS = {"items": [
    {"example": True, "id": "EXAMPLE-1", "language": "fr", "type": "clear", "question": "exemple", "gold_articles": ["Code/s01/1"]},
    {"id": "F1", "language": "fr", "type": "messy", "question": "le voleur a pris un truc numéro9", "gold_articles": ["Code/s01/6"],
     "gold_notes": "", "labeller": "t", "labelled_on": "2026-11-20", "anonymised": True},
    {"id": "F2", "language": "fr", "type": "clear", "question": "divorce consentement mutuel numéro8", "gold_articles": ["Code/s01/8"]},
    {"id": "F3", "language": "ar", "type": "clear", "question": "ما هي عقوبة السرقة", "gold_articles": ["Code/s01/18"]},
    {"id": "F4", "language": "fr", "type": "out_of_scope", "question": "capitale de la France", "gold_articles": []},
    {"id": "F5", "language": "en", "type": "vague", "question": "What is the penalty for theft?", "gold_articles": ["Code/s01/12"]},
]}
NORM = {"le voleur a pris un truc numéro9": "vol simple peine numéro6", "divorce consentement mutuel numéro8": "divorce numéro8",
        "peine du vol simple numéro18": "peine vol numéro18", "theft penalty numéro12": ""}


def tmp_file(obj):
    f = Path(tempfile.mkdtemp()) / "q.json"
    f.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return f


class LLM:
    def __init__(self, fail_translation=(), quota_at=None):
        self.calls, self.fail, self.quota_at = [], set(fail_translation), quota_at

    def tr(self, q):
        self.calls.append(("tr", q))
        if q in self.fail:
            return q                                   # translate_to_french falls back to the question
        return {"ما هي عقوبة السرقة": "peine du vol simple numéro18", "What is the penalty for theft?": "theft penalty numéro12"}[q]

    def norm(self, fq):
        self.calls.append(("norm", fq))
        if self.quota_at == fq:
            raise RuntimeError("429 tokens per day")
        lq = NORM.get(fq, fq)
        return {"legal_query": lq or None, "status": "ok" if lq else "invalid", "error": None if lq else "empty legal_query",
                "attempts": 1, "tokens": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}

    def fns(self):
        return self.tr, self.norm, (lambda e: "per day" in str(e))


def never(*a, **k):
    raise AssertionError("LLM called without --allow-llm")


def main():
    real = RK._reranker
    RK._reranker = SimpleNamespace(predict=lambda pairs, **kw: [len(words(q) & words(t)) for q, t in pairs])
    try:
        qf = tmp_file(ITEMS)
        items = ro.load_items(qf)
        check("fresh format: example skipped, gold_articles read as gold",
              [i["id"] for i in items] == ["F1", "F2", "F3", "F4", "F5"] and items[0]["gold"] == ["Code/s01/6"])
        vss = {"db_v1_1": StubVS(corpus()), "db_v1_2": StubVS(corpus())}

        cache = Path(tempfile.mkdtemp())
        prep = ro.prepare_caches(items, cache, False, never, never)
        check("no --allow-llm: missing translations reported (F3 Arabic, F5 English), no call",
              prep["missing_translation"] == ["F3", "F5"] and prep["llm_calls"] == 0)
        check("no --allow-llm: missing normalizations reported (gold items only; out_of_scope F4 skipped)",
              prep["missing_normalization"] == ["F1", "F2"], prep["missing_normalization"])
        runs, comps, rules = ro.run_abc(items, prep, vss)
        check("no --allow-llm: A and B analyse the French items, C none",
              len(runs["A"]["records"]) == 2 and len(runs["B"]["records"]) == 2 and runs["C"]["records"] == [])
        check("no --allow-llm: skipped items with reasons",
              {s["id"] for s in runs["C"]["skipped"]} >= {"F1", "F2", "F3", "F5"}
              and any("no cached normalization" in s["reason"] for s in runs["C"]["skipped"])
              and any(s["id"] == "F4" and "no gold" in s["reason"] for s in runs["A"]["skipped"]))

        llm = LLM()
        prep = ro.prepare_caches(items, cache, True, *llm.fns())
        check("--allow-llm: 2 translations + 4 normalizations made", prep["llm_calls"] == 6 and [c[0] for c in llm.calls].count("tr") == 2)
        t = json.loads((cache / "translations.json").read_text(encoding="utf-8"))
        n = json.loads((cache / "normalized.json").read_text(encoding="utf-8"))
        check("caches written in the cache folder", {x["id"] for x in t["items"]} == {"F3", "F5"} and {x["id"] for x in n["items"]} == {"F1", "F2", "F3", "F5"})
        check("normalization entries: input query, legal_query, status, prompt hash, tokens",
              all({"input_query", "legal_query", "status", "prompt_sha256", "tokens"} <= set(x) for x in n["items"]))
        check("an empty normalization is saved as a failure", next(x for x in n["items"] if x["id"] == "F5")["flags"]["failure"])
        prep2 = ro.prepare_caches(items, cache, False, never, never)
        check("second run without --allow-llm: everything from the cache, no call",
              prep2["llm_calls"] == 0 and not prep2["missing_translation"] and not prep2["missing_normalization"])
        runs, comps, rules = ro.run_abc(items, prep2, vss)
        check("configs A, B, C analyse the 4 gold items", all(len(runs[c]["records"]) == 4 for c in "ABC"))
        check("paired comparisons B-A, C-B, C-A", set(comps) == {"B-A", "C-B", "C-A"} and comps["C-B"]["n"] == 4)
        byc = {r["id"]: r for r in runs["C"]["records"]}
        byb = {r["id"]: r for r in runs["B"]["records"]}
        check("C uses the normalized query (F1 gains)", byc["F1"]["hit6"] >= byb["F1"]["hit6"] and byc["F1"]["rank_reranked"] == 1)
        check("C: a failed normalization falls back (F5 = B)", byc["F5"]["normalized_fallback"] and byc["F5"]["top6"] == byb["F5"]["top6"])
        check("summary groups by type", {"type clear", "type messy", "type vague"} <= set(runs["B"]["summary"]))
        check("rules: C failure share counted (1 of 4)", rules["C"]["normalize_failures"] == ["F5"] and abs(rules["C"]["failure_share"] - 0.25) < 1e-12
              and not rules["C"]["failures_le_2pct"] and not rules["C"]["adopted"])
        check("rules: B computed", "adopted" in rules["B"] and set(rules["B"]["net_by_language"]) <= {"fr", "ar", "en"})

        items2 = ro.load_items(tmp_file({"items": [dict(i, question="ما هي عقوبة السرقة؟") if i.get("id") == "F3" else i for i in ITEMS["items"]]}))
        prep3 = ro.prepare_caches(items2, cache, False, never, never)
        check("a changed question: stale translation reported and the item skipped", "F3" in prep3["stale"] and "F3" in prep3["missing_translation"])
        cache_f = Path(tempfile.mkdtemp())
        llm = LLM(fail_translation={"What is the penalty for theft?"})
        prep4 = ro.prepare_caches(items, cache_f, True, *llm.fns())
        check("failed translation: reported, not saved, item skipped",
              prep4["failed_translation"] == ["F5"] and "F5" not in {x["id"] for x in json.loads((cache_f / "translations.json").read_text(encoding="utf-8"))["items"]}
              and "F5" not in prep4["normalized"])
        cache_q = Path(tempfile.mkdtemp())
        llm = LLM(quota_at="divorce consentement mutuel numéro8")
        prep5 = ro.prepare_caches(items, cache_q, True, *llm.fns())
        check("daily quota: stops, reports, keeps what was saved",
              prep5["quota"] and {x["id"] for x in json.loads((cache_q / "normalized.json").read_text(encoding="utf-8"))["items"]} == {"F1"})

        recs = lambda spec: [{"id": i, "language": l, "hit6": h} for i, l, h in spec]
        A = recs([("a", "fr", 1), ("b", "fr", 1), ("c", "fr", 1), ("d", "ar", 0), ("e", "ar", 0), ("f", "ar", 0)])
        B = recs([("a", "fr", 0), ("b", "fr", 0), ("c", "fr", 0), ("d", "ar", 1), ("e", "ar", 1), ("f", "ar", 1)])
        net = ro.net_by(A, B, "language")
        check("net by language", net == {"fr": -3, "ar": 3})
        comp = {"B-A": {6: {"diff": 0.0, "ci": (-0.1, 0.1)}}}
        r = ro.abc_rules({"A": {"records": A}, "B": {"records": B}}, comp, {})
        check("rule B: difference >= 0 but a language net loss of 3 -> not adopted", r["B"]["diff_ge_0"] and not r["B"]["adopted"])
        comp = {"C-B": {6: {"diff": 0.1, "ci": (0.01, 0.2)}}}
        C = recs([("a", "fr", 1), ("b", "fr", 1), ("c", "fr", 1), ("d", "ar", 1), ("e", "ar", 1), ("f", "ar", 1)])
        normd = {i: {"status": "ok", "legal_query": "x"} for i in "abcdef"}
        r = ro.abc_rules({"B": {"records": B}, "C": {"records": C}}, comp, normd)
        check("rule C: CI above 0, no language loss, no failure -> adopted", r["C"]["adopted"])
        r = ro.abc_rules({"B": {"records": B}, "C": {"records": C}}, {"C-B": {6: {"diff": 0.1, "ci": (0.0, 0.2)}}}, normd)
        check("rule C: CI lower bound exactly 0 -> not adopted", not r["C"]["adopted"])
        check("rule constants", ro.NET_LOSS_LIMIT == 3 and ro.MAX_NORMALIZE_FAILURES == 0.02)

        dirs = {}
        for name in ("v11", "v12"):
            d = Path(tempfile.mkdtemp()) / name
            d.mkdir()
            (d / "chroma.sqlite3").write_bytes(b"x")
            dirs[name] = d
        out, md = Path(tempfile.mkdtemp()) / "o.json", Path(tempfile.mkdtemp()) / "o.md"
        ap = ro.argparse.ArgumentParser()
        a = SimpleNamespace(items=str(qf), allow_llm=False, cache_dir=str(cache), index_v11=str(dirs["v11"]), index_v12=str(dirs["v12"]),
                            configs=["A", "B", "C"], out=str(out), md=str(md))
        opened = []
        code = ro.main_abc(a, ap, opener=lambda p: (opened.append(p), StubVS(corpus()))[1])
        text = md.read_text(encoding="utf-8")
        check("main --abc: exit 0, json and markdown written", code == 0 and json.loads(out.read_text(encoding="utf-8"))["comparisons"].keys() >= {"B-A", "C-B", "C-A"})
        check("main --abc: tables (paired, net by language/type, draft rules)",
              "Paired C-B" in text and "net Hit@6 by language" in text and "net Hit@6 by type" in text and "Draft decision rules" in text)
        check("main --abc: indexes opened from temporary copies only", len(opened) == 2 and all(dirs["v11"] != p and dirs["v12"] != p for p in opened))
        a.allow_llm, a.cache_dir = True, str(Path(tempfile.mkdtemp()))
        llm = LLM(quota_at="le voleur a pris un truc numéro9")
        check("main --abc --allow-llm: daily quota -> exit 75", ro.main_abc(a, ap, opener=lambda p: StubVS(corpus()), llm=llm.fns()) == 75)
    finally:
        RK._reranker = real
        ro.apply_config("v1.2", {"NORMALIZE_QUERY": "off"})
    print(f"\n{CNT['n'] - len(FAILS)}/{CNT['n']} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
