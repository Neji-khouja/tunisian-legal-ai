"""config.NORMALIZE_QUERY (wave 2 V3 in the agent): stub tests, no Groq, no model, no index.

    python agent/test_normalize_switch.py

1. OFF equivalence: the graph and retrieve_g of commit REF (before the switch existed) and the current ones run
   side by side on the same stubs (every LLM node and every retrieval step stubbed); the final states of every
   conversation must be identical, and the normalizer must never be called.
2. V3 on: one normalization per turn (reformulated queries are not normalized), RRF pool reranked on the
   normalized query, every kind of failure falls back to exactly the OFF retrieval (never raises), the record is
   in the state and reset at the next turn, the normalizer's tokens reach the usage callback.
"""
import sys
import random
import subprocess
import tempfile
import importlib
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "diagnostics")]
REF = "cc46906"          # last commit before config.NORMALIZE_QUERY

import config as C
import pipeline
import normalize as N
import agent_graph as new_g
import retrieve_g as new_rg
from answer import Answer, Claim
from verify import Verification, ClaimVerdict
from langchain_core.documents import Document

FAILS, CNT = [], {"n": 0}


def check(name, ok, detail=""):
    CNT["n"] += 1
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"   {detail}"))
    if not ok:
        FAILS.append(name)


def load_reference():
    """agent_graph.py and retrieve_g.py of REF as modules old_agent_graph / old_retrieve_g."""
    tmp = Path(tempfile.mkdtemp(prefix="ref_graph_"))
    for name in ("retrieve_g", "agent_graph"):
        src = subprocess.run(["git", "show", f"{REF}:agent/{name}.py"], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout
        src = src.replace("from retrieve_g import retrieve_g", "from old_retrieve_g import retrieve_g")
        (tmp / f"old_{name}.py").write_text(src, encoding="utf-8")
    sys.path.insert(0, str(tmp))
    return importlib.import_module("old_agent_graph"), importlib.import_module("old_retrieve_g")


# ---------------------------------------------------------------- stubs

TOPICS = {"penal": "vol simple peine emprisonnement voleur", "famille": "divorce garde enfant pension",
          "travail": "licenciement préavis salaire contrat"}
DOCS = [Document(page_content=f"Article {n}. {TOPICS[law]} texte{n} numéro{n}",
                 metadata={"chunk_id": f"{law.capitalize()}/s01/{n}", "article": str(n), "source": f"{law}.pdf",
                           "law_type": law})
        for n, law in enumerate(["penal", "famille", "travail"] * 12, 1)]


def words(t):
    return set(w.strip(".,;:?!'()").lower() for w in t.split())


def stub_retrieve(vs, question, fq, subs, law_type):
    if "vide" in fq:
        return []
    ds = [d for d in DOCS if law_type in (None, d.metadata["law_type"])]
    q = words(fq + " " + " ".join(subs))
    return sorted(ds, key=lambda d: -len(q & words(d.page_content)))[:20]


def stub_rerank(q, cands):
    w = words(q)
    return sorted(cands, key=lambda d: -len(w & words(d.page_content)))[:6]


def stub_query(q):
    return ("traduction: " + q) if any("؀" <= c <= "ۿ" for c in q) else q


CALLS = {"retrieve_for_plan": [], "normalize": 0}


def stub_retrieve_for_plan(sub_queries, top_k=6, use_code_filter=True, soft=False):
    CALLS["retrieve_for_plan"].append(list(sub_queries))
    text = " ".join(t for t, _ in sub_queries)
    return stub_rerank(text, stub_retrieve(None, text, text, [], None))[:top_k]


def stub_analyze(q, hint=None):
    lq = q.lower()
    if "capitale" in lq:
        return SimpleNamespace(language="fr", intent="out_of_scope", target_codes=[], clarification_question=None)
    if "durée" in lq and "garde" not in lq:
        return SimpleNamespace(language="fr", intent="ambiguous", target_codes=[], clarification_question="Quelle durée ?")
    lang = "ar" if any("؀" <= c <= "ۿ" for c in q) else "fr"
    return SimpleNamespace(language=lang, intent="clear", target_codes=["penal"] if "vol" in lq else [],
                           clarification_question=None)


def stub_grade(q, evidence):
    return SimpleNamespace(enough="facile" in q, missing="la peine exacte")


def stub_reformulate(q, lang, tried, evidence, missing):
    return SimpleNamespace(sub_queries=[SimpleNamespace(text=f"peine vol essai{len(tried)}", code="penal")])


def stub_make_answer(q, lang, evidence, feedback=""):
    ids = list(evidence)[:2]
    tag = "r" if feedback else "c"
    return Answer(claims=[Claim(text=f"{tag}{i}", cited_ids=[cid]) for i, cid in enumerate(ids, 1)], caveat="")


def stub_verify(ans, evidence):
    return Verification(verdicts=[ClaimVerdict(claim_index=i, verdict="unsupported" if c.text == "c2" else "supported",
                                               problem="x" if c.text == "c2" else "")
                                  for i, c in enumerate(ans.claims, 1)])


def stub_condense(history, q):
    return SimpleNamespace(question=q + " (vol simple)", is_followup=True)


def install(g, rg):
    for k, v in dict(analyze=stub_analyze, grade=stub_grade, reformulate=stub_reformulate, make_answer=stub_make_answer,
                     verify=stub_verify, condense=stub_condense, retrieve_for_plan=stub_retrieve_for_plan).items():
        setattr(g, k, v)
    for k, v in dict(step_query=stub_query, step_subqueries_if_enabled=lambda q, fq: [], step_retrieve=stub_retrieve,
                     step_rerank=stub_rerank, get_vs=lambda: object()).items():
        setattr(rg, k, v)


pipeline.evidence_docs = lambda docs, vs=None: list(docs)
pipeline.evidence_text = lambda d: d.page_content

CONVERSATIONS = [
    ["Quelle est la peine du vol simple ? facile"],                         # clear, one code, enough at once
    ["Quelle est la peine du vol aggravé ?"],                               # 2 reformulations, repair, finalize
    ["Quelle est la capitale de la France ?"],                              # out of scope
    ["Quelle est la durée ?", "la garde à vue du suspect"],                 # clarify, then the reply
    ["ما هي عقوبة السرقة"],                                                  # Arabic (translated query)
    ["Quelle est la peine du vol simple ? facile", "et en cas de récidive ? facile"],   # follow-up (condense)
    ["question vide sans résultat facile"],                                  # nothing retrieved
]


def run(app, turns, tag):
    cfg = {"configurable": {"thread_id": f"{tag}-{random.random()}"}}
    return [app.invoke({"question": q}, cfg) for q in turns]


# ---------------------------------------------------------------- 1. OFF equivalence

def test_off(old_g, old_rg):
    check("default: config.NORMALIZE_QUERY == 'v3' (owner decision)", C.NORMALIZE_QUERY == "v3")
    C.NORMALIZE_QUERY = "off"
    real = N.normalize

    def never(*a, **k):
        CALLS["normalize"] += 1
        raise AssertionError("normalizer called with NORMALIZE_QUERY off")
    N.normalize = never
    try:
        for conv in CONVERSATIONS:
            CALLS["retrieve_for_plan"].clear()
            a = run(old_g.app, conv, "old")
            calls_a = list(CALLS["retrieve_for_plan"])
            CALLS["retrieve_for_plan"].clear()
            b = run(new_g.app, conv, "new")
            same = all(x == y for x, y in zip(a, b)) and len(a) == len(b)
            diff = [k for x, y in zip(a, b) for k in set(x) | set(y) if x.get(k) != y.get(k)]
            check(f"OFF identical final states: {conv[0][:40]!r} ({len(conv)} turn(s), statuses "
                  f"{[s.get('status') for s in b]})", same and calls_a == CALLS["retrieve_for_plan"], diff)
            check(f"OFF no normalize key in the state: {conv[0][:30]!r}", all("normalize" not in s for s in b))
        check("OFF: the normalizer was never called", CALLS["normalize"] == 0)
        for codes in ([], ["penal"], ["penal", "famille"]):
            check(f"OFF retrieve_g == reference retrieve_g (target codes {codes})",
                  new_rg.retrieve_g("vol simple peine", codes) == old_rg.retrieve_g("vol simple peine", codes))
    finally:
        N.normalize = real


# ---------------------------------------------------------------- 2. V3 on

TOK = {"input_tokens": 900, "output_tokens": 60, "total_tokens": 960}


def norm_stub(result):
    def f(query, llm=None, **kw):
        CALLS["normalize"] += 1
        if isinstance(result, Exception):
            raise result
        return dict(result, tokens=TOK, attempts=1, raw=None)
    return f


def test_v3():
    real = N.normalize
    C.NORMALIZE_QUERY = "v3"
    try:
        N.normalize = norm_stub({"legal_query": "divorce garde enfant numéro5", "status": "ok", "error": None})
        CALLS["normalize"], CALLS["retrieve_for_plan"] = 0, []
        s = run(new_g.app, ["Quelle est la peine du vol aggravé ?"], "v3")[0]
        check("V3: one normalization for a turn with 2 reformulations", CALLS["normalize"] == 1, CALLS["normalize"])
        check("V3: reformulated queries go to retrieve_for_plan unchanged (not normalized)",
              CALLS["retrieve_for_plan"] == [[("peine vol essai0", "penal")], [("peine vol essai1", "penal")]], CALLS["retrieve_for_plan"])
        check("V3: record in the state (query, status, tokens, prompt hash)",
              s["normalize"]["legal_query"] == "divorce garde enfant numéro5" and s["normalize"]["tokens"] == TOK
              and s["normalize"]["prompt_sha256"] == N.PROMPT_SHA256 and not s["normalize"]["fell_back"])
        check("V3: no degraded event on success", not any("normalize" in x for x in s["degraded"]))
        fq, nq = "vol simple peine", "divorce garde enfant numéro5"
        from retriever import fuse_pools, _fuse
        expect = stub_rerank(nq, fuse_pools(stub_retrieve(None, fq, fq, [], None), stub_retrieve(None, fq, nq, [], None)))[:6]
        check("V3 retrieve_g: RRF of the two pools, reranked on the normalized query", new_rg.retrieve_g(fq, []) == expect)
        exp1 = _fuse([stub_rerank(nq, fuse_pools(stub_retrieve(None, fq, fq, [], "penal"), stub_retrieve(None, fq, nq, [], "penal")))[:6],
                      expect])[:6]
        CALLS["normalize"] = 0
        check("V3 retrieve_g with one code: both baselines fused, one normalization",
              new_rg.retrieve_g(fq, ["penal"]) == exp1 and CALLS["normalize"] == 1)
        check("V3 changes the retrieval when the normalization succeeds (stub)",
              expect != stub_rerank(fq, stub_retrieve(None, fq, fq, [], None))[:6])
        # turn reset
        s2 = run(new_g.app, ["Quelle est la peine du vol simple ? facile", "Quelle est la capitale de la France ?"], "v3r")
        check("V3: record reset at the next turn (refused turn has an empty record)", s2[0]["normalize"]["legal_query"] and s2[1]["normalize"] == {})
        # every failure falls back to exactly the OFF result
        C.NORMALIZE_QUERY = "off"
        N.normalize = real
        offs = {tuple(conv): run(new_g.app, conv, "off") for conv in CONVERSATIONS}
        C.NORMALIZE_QUERY = "v3"
        failures = {"invalid": {"legal_query": None, "status": "invalid", "error": "invalid JSON"},
                    "empty": {"legal_query": "  ", "status": "ok", "error": None},
                    "api_error": {"legal_query": None, "status": "api_error", "error": "500"},
                    "exception": RuntimeError("client cannot be built")}
        keys = ("docs", "evidence", "text", "status", "claims", "cited", "retries", "tried")
        for name, res in failures.items():
            N.normalize = norm_stub(res)
            for conv in CONVERSATIONS:
                try:
                    got = run(new_g.app, conv, f"fail-{name}")
                except Exception as e:
                    check(f"V3 {name}: never raises ({conv[0][:30]!r})", False, repr(e))
                    continue
                ref = offs[tuple(conv)]
                same = all(all(x.get(k) == y.get(k) for k in keys) for x, y in zip(got, ref))
                fb = [x for x in got if x.get("normalize")]
                check(f"V3 {name}: same retrieval and answer as OFF, fell_back recorded ({conv[0][:30]!r})",
                      same and all(x["normalize"]["fell_back"] and any(d.startswith("normalize_failed") for d in x["degraded"]) for x in fb))
        N.normalize = real
        # normalize_for_retrieval never raises, falls back on every failure
        for name, res in failures.items():
            N.normalize = norm_stub(res)
            q, rec = N.normalize_for_retrieval("ma question")
            check(f"normalize_for_retrieval {name}: input query kept, fell_back", q == "ma question" and rec["fell_back"])
        N.normalize = real
    finally:
        N.normalize = real
        C.NORMALIZE_QUERY = "off"


def test_tokens_and_fusion():
    from langchain_core.callbacks import get_usage_metadata_callback
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    msg = AIMessage(content='{"legal_query": "faux témoignage"}', response_metadata={"model_name": "stub-model"},
                    usage_metadata={"input_tokens": 700, "output_tokens": 40, "total_tokens": 740})
    fake = GenericFakeChatModel(messages=iter([msg]))
    with get_usage_metadata_callback() as cb:
        q, rec = N.normalize_for_retrieval("un témoin a menti", llm=fake)
    check("tokens: the normalizer's call is counted by the agent's usage callback",
          cb.usage_metadata.get("stub-model", {}).get("total_tokens") == 740, cb.usage_metadata)
    check("tokens: and recorded in the normalization record", q == "faux témoignage" and rec["tokens"]["total_tokens"] == 740)
    import wave2_core as W
    from retriever import fuse_pools
    rng = random.Random(20261022)
    ok = True
    for _ in range(300):
        a = [DOCS[i] for i in rng.sample(range(len(DOCS)), rng.randint(0, 20))]
        b = [DOCS[i] for i in rng.sample(range(len(DOCS)), rng.randint(0, 20))]
        ok &= fuse_pools(a, b) == W.fused_pool(a, b, lambda d: d.metadata["chunk_id"])
    check("fusion: retriever.fuse_pools == wave2_core.fused_pool (pre-registered) on 300 random pool pairs", ok)


def main():
    old_g, old_rg = load_reference()
    for g, rg in ((old_g, old_rg), (new_g, new_rg)):
        install(g, rg)
    test_off(old_g, old_rg)
    test_v3()
    test_tokens_and_fusion()
    print(f"\n{CNT['n'] - len(FAILS)}/{CNT['n']} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
