"""Wave 2 (query normalization) stub tests: fusion, failure fallback, attempts, cache/resume, overlap check, decision
rule, retrieval_only variants. No model, no LLM, no real index.

    python eval/test_wave2.py
"""
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "agent"), str(ROOT / "diagnostics")]
import wave2_core as W
import normalize as N
import normalize_queries as NQ
import retrieval_only as ro
import reranker as RK
from test_retrieval_only import StubVS, corpus, words

FAILS, CNT = [], {"n": 0}


def check(name, ok, detail=""):
    CNT["n"] += 1
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"   {detail}"))
    if not ok:
        FAILS.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- fusion

def test_fusion():
    ident = lambda x: x
    check("rrf: an item in both lists beats one ranked first in a single list",
          W.rrf_merge([["a", "b", "c"], ["d", "b", "e"]], ident)[0] == "b")
    check("rrf: ties keep first appearance, list 1 first", W.rrf_merge([["a", "c"], ["b", "d"]], ident) == ["a", "b", "c", "d"])
    check("rrf: identical lists keep their order", W.rrf_merge([list("abcd"), list("abcd")], ident) == list("abcd"))
    check("rrf: scores 1/(60 + rank)", W.RRF_K == 60 and W.rrf_merge([["x", "y"], ["y"]], ident) == ["y", "x"])
    po, pn = list("abcdef"), list("gbhaij")
    fp = W.fused_pool(po, pn, ident)
    check("fused pool: cut to the larger pool size", len(fp) == 6 and len(set(fp)) == 6, fp)
    check("fused pool: shared candidates first", fp[:2] == ["b", "a"], fp)
    check("fused pool: unequal sizes -> the larger", len(W.fused_pool(list("ab"), list("cdef"), ident)) == 4)
    check("fused pool: same pool twice = the pool", W.fused_pool(po, po, ident) == po)


# ---------------------------------------------------------------- normalizer and fallback

class StubLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def invoke(self, messages):
        self.calls += 1
        o = self.outputs.pop(0)
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(content=o, usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})


def test_normalizer():
    check("parse: valid object", N.parse_output('{"legal_query": "vol  simple"}') == ("vol simple", None))
    check("parse: code fence stripped", N.parse_output('```json\n{"legal_query": "x"}\n```') == ("x", None))
    check("parse: empty -> failure", N.parse_output("   ")[0] is None)
    check("parse: invalid JSON -> failure", N.parse_output("vol simple")[0] is None)
    check("parse: wrong key -> failure", N.parse_output('{"query": "x"}')[0] is None)
    check("parse: empty legal_query -> failure", N.parse_output('{"legal_query": "  "}')[0] is None)
    check("parse: non-string -> failure", N.parse_output('{"legal_query": 5}')[0] is None)
    check("numbers: new article number flagged", N.new_numbers("vol de 3 moutons", "vol de 3 moutons article 264") == ["264"])
    check("numbers: same numbers not flagged", N.new_numbers("délai de 15 jours, 1.000 dinars", "délai 15 jours 1000 dinars") == [])
    check("numbers: date kept", N.new_numbers("signé le 2 janvier 2000", "acte du 2 janvier 2000") == [])
    llm = StubLLM(['{"legal_query": "a"}'])
    r = N.normalize("q", llm=llm)
    check("normalize: one call when valid", r["status"] == "ok" and r["attempts"] == 1 and llm.calls == 1 and r["tokens"]["total_tokens"] == 120)
    llm = StubLLM(["oops", '{"legal_query": "b"}'])
    r = N.normalize("q", llm=llm)
    check("normalize: invalid then valid -> 2 attempts, tokens of both", r["legal_query"] == "b" and r["attempts"] == 2 and r["tokens"]["total_tokens"] == 240)
    llm = StubLLM(["oops", "", '{"legal_query": "never"}'])
    r = N.normalize("q", llm=llm)
    check("normalize: at most 2 attempts, then invalid", r["status"] == "invalid" and llm.calls == 2 and r["legal_query"] is None)
    waits = []
    llm = StubLLM([RuntimeError("Error code: 429 rate_limit_exceeded, Please try again in 2.5s"), '{"legal_query": "c"}'])
    r = N.normalize("q", llm=llm, sleep=waits.append)
    check("normalize: per-minute 429 waits the stated time once", r["legal_query"] == "c" and waits == [3.5], waits)
    llm = StubLLM([RuntimeError("500 server"), RuntimeError("500 server"), '{"legal_query": "x"}'])
    r = N.normalize("q", llm=llm)
    check("normalize: two API errors -> api_error, 2 calls", r["status"] == "api_error" and llm.calls == 2)
    llm = StubLLM([RuntimeError("tokens per day exceeded"), '{"legal_query": "x"}'])
    check("normalize: fatal error (daily quota) re-raised at once",
          raises(lambda: N.normalize("q", llm=llm, fatal=lambda e: "per day" in str(e)), RuntimeError) and llm.calls == 1)
    check("normalize: temperature 0, 2 attempts, hash of the prompt",
          N.TEMPERATURE == 0 and N.MAX_ATTEMPTS == 2 and len(N.PROMPT_SHA256) == 64)
    check("fallback: ok entry used", W.query_for({"status": "ok", "legal_query": "nq"}, "fq") == ("nq", False))
    check("fallback: failure -> input query", W.query_for({"status": "invalid", "legal_query": None}, "fq") == ("fq", True))
    check("fallback: api error -> input query", W.query_for({"status": "api_error", "legal_query": None}, "fq") == ("fq", True))
    check("fallback: missing entry -> input query", W.query_for(None, "fq") == ("fq", True))
    check("fallback: empty legal_query -> input query", W.query_for({"status": "ok", "legal_query": " "}, "fq") == ("fq", True))


# ---------------------------------------------------------------- cache / resume

def test_resume():
    tmp = Path(tempfile.mkdtemp()) / "norm.json"
    pairs = [({"id": f"i{k}", "set": "S1", "language": "fr"}, f"question {k} de 12 jours") for k in range(4)]
    seen = []

    def call(q):
        seen.append(q)
        return {"legal_query": q.replace("question", "requête") + (" article 99" if q.startswith("question 2") else ""),
                "status": "ok", "error": None, "attempts": 1, "tokens": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
    done = {"items": []}
    code = NQ.run(pairs[:2], done, call, lambda e: False, path=tmp, sleep=lambda s: None)
    saved = json.loads(tmp.read_text(encoding="utf-8"))
    check("resume: first run saves each item", code == 0 and [e["id"] for e in saved["items"]] == ["i0", "i1"])
    check("resume: entry fields", {"id", "input_query", "legal_query", "prompt_sha256", "tokens", "timestamp", "flags"} <= set(saved["items"][0])
          and saved["items"][0]["prompt_sha256"] == N.PROMPT_SHA256)
    seen.clear()
    done = NQ.load(tmp)
    code = NQ.run(pairs, done, call, lambda e: False, path=tmp, sleep=lambda s: None)
    check("resume: saved items not called again", code == 0 and seen == ["question 2 de 12 jours", "question 3 de 12 jours"], seen)
    t = NQ.totals(NQ.load(tmp))
    check("resume: totals and the new-number flag", t["items"] == 4 and t["tokens"]["total_tokens"] == 48 and t["new_numbers"] == {"i2": ["99"]}, t)

    def quota(q):
        raise RuntimeError("429 tokens per day")
    tmp2 = Path(tempfile.mkdtemp()) / "norm.json"
    done = {"items": []}
    code = NQ.run(pairs, done, lambda q: call(q) if q.startswith("question 0") else quota(q), lambda e: "per day" in str(e),
                  path=tmp2, sleep=lambda s: None)
    check("resume: daily quota -> exit 75, done items saved", code == 75 and [e["id"] for e in NQ.load(tmp2)["items"]] == ["i0"])
    failing = lambda q: {"legal_query": None, "status": "invalid", "error": "empty output", "attempts": 2, "raw": "",
                         "tokens": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1}}
    tmp3 = Path(tempfile.mkdtemp()) / "norm.json"
    NQ.run(pairs[:1], {"items": []}, failing, lambda e: False, path=tmp3, sleep=lambda s: None)
    t = NQ.totals(NQ.load(tmp3))
    check("resume: a failure is saved, counted, and not retried", t["failures"] == ["i0"] and t["ok"] == 0)
    sl, clock = [], iter([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    NQ.run(pairs[:2], {"items": []}, call, lambda e: False, path=Path(tempfile.mkdtemp()) / "n.json", sleep=sl.append,
           clock=lambda: next(clock), interval=7.0)
    check("pacing: waits the rest of the minimum interval", sl == [6.0], sl)


# ---------------------------------------------------------------- overlap check

def test_overlap():
    import prompt_overlap as po
    lines, worst, pairs = po.check_prompt("agent/normalize.py:NORMALIZE_PROMPT")
    check("overlap: the normalization prompt has no pair >= 0.4", pairs == 0 and worst < 0.4, (worst, pairs))
    check("overlap: 3 example questions and 3 outputs extracted",
          len(po.prompt_examples(N.NORMALIZE_PROMPT)) == 6, po.prompt_examples(N.NORMALIZE_PROMPT))
    q = po.corpus_questions()
    fr = next(x for x in q if x[3] == "fr")
    f = Path(tempfile.mkdtemp()) / "leaky.py"
    f.write_text("P = '''\nQuestion: \"" + fr[2].replace('"', "'") + "\"\nOutput: {\"legal_query\": \"x\"}\n'''\n", encoding="utf-8")
    lines, worst, pairs = po.check_prompt(f"{f}:P")
    check("overlap: an example copied from a corpus question is caught", worst >= 0.99 and pairs >= 1, (worst, pairs))
    check("overlap: corpus = every eval/*.json question + dev items + translations",
          any(x[0] == "dev_items.json" for x in q) and any(x[0] == "translations.json" for x in q) and any(x[0] == "answer_set.json" for x in q))


# ---------------------------------------------------------------- decision rule

def recs(spec):
    """spec: list of (id, set, language, hit6)."""
    return [{"id": i, "set": s, "language": l, "hit6": h} for i, s, l, h in spec]


def test_rule():
    base = [(f"s{k}", "S1", "fr", 1) for k in range(5)] + [(f"n{k}", "N", "fr", 0) for k in range(5)] + \
           [("a1", "S2", "ar", 1), ("a2", "S2", "ar", 1), ("e1", "N", "en", 1)]
    v0 = recs(base)
    flip = lambda ids, val: recs([(i, s, l, (val if i in ids else h)) for i, s, l, h in base])
    good = flip({"n0", "n1", "n2"}, 1)
    d = W.decide({"V0": v0, "V1": good, "V2": v0, "V3": v0}, {"V0": 0, "V1": 9, "V2": 9, "V3": 9})
    check("rule: +3 noisy, nothing lost -> adopted", d["candidate"] == "V1" and d["selected"] == "V1", d["selected"])
    d = W.decide({"V0": v0, "V1": flip({"n0", "n1"}, 1), "V2": v0, "V3": v0}, {"V1": 9, "V2": 9, "V3": 9})
    check("rule: (i) +2 noisy -> none", d["candidate"] == "V1" and d["selected"] == "none" and not d["conditions"]["V1"]["i_noisy_gain_ge_3"])
    worse = recs([(i, s, l, (1 if i in {"n0", "n1", "n2", "n3", "n4"} else 0 if i in {"s0", "s1"} else h)) for i, s, l, h in base])
    d = W.decide({"V0": v0, "V1": worse}, {"V1": 9})
    check("rule: (ii) 2 lost on Stage 1+2 -> none", d["selected"] == "none" and d["conditions"]["V1"]["lost_stage1_2"] == ["s0", "s1"])
    ok1 = recs([(i, s, l, (1 if i in {"n0", "n1", "n2", "n3"} else 0 if i == "s0" else h)) for i, s, l, h in base])
    check("rule: (ii) 1 lost on Stage 1+2 is allowed", W.decide({"V0": v0, "V1": ok1}, {"V1": 9})["selected"] == "V1")
    lang = recs([(i, s, l, (1 if i in {"n0", "n1", "n2", "n3", "n4"} else 0 if i in {"a1", "a2"} else h)) for i, s, l, h in base])
    d = W.decide({"V0": v0, "V1": lang}, {"V1": 9})
    check("rule: (iii) 2 lost in one language -> none", d["selected"] == "none" and d["conditions"]["V1"]["lost_by_language"]["ar"] == ["a1", "a2"]
          and not d["conditions"]["V1"]["iii_lost_each_language_le_1"])
    lang1 = recs([(i, s, l, (1 if i in {"n0", "n1", "n2", "n3", "n4"} else 0 if i == "e1" else h)) for i, s, l, h in base])
    check("rule: (iii) noisy-set loss counts in its language", W.decide({"V0": v0, "V1": lang1}, {"V1": 9})["conditions"]["V1"]["lost_by_language"]["en"] == ["e1"])
    d = W.decide({"V0": v0, "V1": good, "V2": good, "V3": good}, {"V1": 9, "V2": 9, "V3": 9})
    check("rule: tie, same tokens -> lower-numbered variant", d["candidate"] == "V1")
    d = W.decide({"V0": v0, "V1": good, "V2": good}, {"V1": 20, "V2": 9})
    check("rule: tie -> fewer extra tokens", d["candidate"] == "V2")
    best_bad = recs([(i, s, l, (1 if i.startswith("n") else 0 if i in {"s0", "s1"} else h)) for i, s, l, h in base])
    d = W.decide({"V0": v0, "V1": best_bad, "V2": good}, {"V1": 9, "V2": 9})
    check("rule: the highest fails -> none (runner-up not considered)", d["candidate"] == "V1" and d["selected"] == "none"
          and d["conditions"]["V2"]["passes"])
    check("rule: constants as pre-registered", (W.MIN_NOISY_GAIN, W.MAX_LOST_STAGES, W.MAX_LOST_LANGUAGE) == (3, 1, 1))


# ---------------------------------------------------------------- retrieval_only variants on a stub index

ITEMS = [
    {"id": "f1", "set": "N", "language": "fr", "question": "le voleur a pris un truc numéro9", "gold": [{"source": "Code.pdf", "article": "6"}]},
    {"id": "f2", "set": "S1", "language": "fr", "question": "divorce consentement mutuel numéro8", "gold": [{"source": "Code.pdf", "article": "8"}]},
    {"id": "f3", "set": "S2", "language": "fr", "question": "Que dit l'article 12 ?", "gold": [{"source": "Code.pdf", "article": "12"}]},
]
_T = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
NORMALIZED = {"f1": {"status": "ok", "legal_query": "vol simple peine numéro6", "tokens": _T},
              "f2": {"status": "invalid", "legal_query": None, "tokens": _T},
              "f3": {"status": "ok", "legal_query": "contenu article 12 numéro12", "tokens": _T}}


def test_variants():
    real = RK._reranker
    RK._reranker = SimpleNamespace(predict=lambda pairs, **kw: [len(words(q) & words(t)) for q, t in pairs])
    try:
        vs = StubVS(corpus())
        ro.apply_config("v1.2", {})
        out = {}
        for v in W.VARIANTS:
            recs_, skipped = ro.evaluate(ITEMS, {}, ro.variant_backend(vs, v, NORMALIZED))
            out[v] = {r["id"]: r for r in recs_}
        base = {r["id"]: r for r in ro.evaluate(ITEMS, {}, ro.real_backend(vs))[0]}
        same = lambda a, b: all(a[i]["top6"] == b[i]["top6"] and a[i]["rank_reranked"] == b[i]["rank_reranked"] for i in a)
        check("variants: V0 = the existing v1.2 retrieval", same(out["V0"], base))
        check("variants: V1 retrieves with the normalized query", out["V1"]["f1"]["rank_reranked"] == 1 and out["V0"]["f1"]["rank_reranked"] != 1,
              (out["V0"]["f1"]["rank_reranked"], out["V1"]["f1"]["rank_reranked"]))
        check("variants: failed normalization -> V1 = V2 = V3 = V0 and flagged",
              all(out[v]["f2"]["top6"] == out["V0"]["f2"]["top6"] and out[v]["f2"]["normalized_fallback"] for v in W.VARIANTS))
        check("variants: V2/V3 pool no larger than the larger single pool",
              all(out[v][i]["pool_size"] <= max(out["V0"][i]["pool_size"], out["V1"][i]["pool_size"]) for v in ("V2", "V3") for i in out[v]))
        check("variants: V3 reranks the fused pool on the normalized query", out["V3"]["f1"]["rank_reranked"] == 1)
        check("variants: the question (not the normalized query) decides the direct lookup",
              all(out[v]["f3"]["path"] == "direct+hybrid" for v in W.VARIANTS) and out["V1"]["f1"]["path"] == "hybrid")
        check("variants: unknown variant refused", raises(lambda: ro.variant_backend(vs, "V9", {}), ValueError))
        runs, comps, decision = ro.run_variants(ITEMS, {}, vs, NORMALIZED, list(W.VARIANTS), "v1.2", {})
        check("run_variants: comparisons versus V0 and a decision", set(comps) == {"V1", "V2", "V3"} and decision["selected"] in ("none", "V1", "V2", "V3"))
        entries = [{"id": i, "set": "S1", "language": "fr", "input_query": "q", "legal_query": e["legal_query"] or "", "status": e["status"],
                    "error": None, "attempts": 1, "prompt_sha256": "ab" * 32, "tokens": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    "flags": {"new_numbers": ["12"] if i == "f3" else []}} for i, e in NORMALIZED.items()]
        md = ro.variants_markdown(runs, comps, decision, entries)
        check("markdown: decision, flagged, random rows and paired tables",
              "Decision rule" in md and "| f3 | 12 |" in md and "12 random rows" in md and "Paired differences" in md, md[:400])
        check("sample rows: deterministic, seed 20261022", ro.sample_rows(range(110)) == ro.sample_rows(range(110)) and len(ro.sample_rows(range(110))) == 12)
    finally:
        RK._reranker = real
        ro.apply_config("v1.2", {})


def main():
    test_fusion()
    test_normalizer()
    test_resume()
    test_overlap()
    test_rule()
    test_variants()
    print(f"\n{CNT['n'] - len(FAILS)}/{CNT['n']} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
