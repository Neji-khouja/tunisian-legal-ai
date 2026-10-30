"""Answer-level evaluation harness (v2 = agent graph, v1.1 = baseline pipeline, v1 = original pipeline).

    python eval/run_answer_eval.py --system v2 --dry-run                     # plan + LLM-call estimate, no call
    python eval/run_answer_eval.py --system v2 [--limit N] [--only-ids a001,a002] [--skip-judge] [--judge-draft] [--min-gap SECONDS]
    python eval/run_answer_eval.py --summarize results/answer_eval_answer_set_v2.jsonl [more.jsonl ...]
    python eval/run_answer_eval.py --system v2 --export-audit 20             # CSV for a human check of the judge

Items (eval/answer_set.json, see eval/check_answer_set.py): id, question, language, expected (answer|clarify|refuse),
gold [{source, article}], reference_points [str], reason. Results are appended to
results/answer_eval_<setname>_<system>.jsonl; items already there are skipped (resume).
Timing per item: seconds (system only), elapsed_total (system + metrics + judge; the --min-gap sleep is outside),
n_429 (rate-limited Groq responses the client retried) and waited_seconds (sum of the retry waits the Groq client
announced); n_429 and waited_seconds are None for v1, whose LLM calls run in a subprocess.
v2 rows also keep the verifier loop: verdicts (last pass), first_verdicts (first pass), repairs, dropped,
draft_claims / draft_text (the first, unverified answer). A v2 item whose `degraded` list is not empty (a node fell back) is stored with status "degraded"
(graph status in graph_status), gets no metrics and no judge, and is run again on resume.

Judge: a model different from the generator (env JUDGE_MODEL or --judge-model; a Groq model id), temperature 0,
own prompt, structured output. Never guessed: without a judge model the run stops unless --skip-judge.
v1 needs a git worktree of tag v1-baseline-runnable with the old index (the script prints the setup commands).
"""
import os
import sys
import csv
import json
import time
import random
import logging
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "eval"))
from pdf_truth import from_chunk_id, pdf_article_text, source_stem, normalize
from pdf_factcheck import claim_numbers_ok, implicit_reference_numbers

SEED = 20261022
N_BOOT = 5000
GENERATOR_MODEL = "openai/gpt-oss-120b"        # config.LLM_MODEL and agent/analyze.py
V1_TAG = "v1-baseline-runnable"
# successful Groq requests (n), rate-limited responses (n429) seen by httpx in this process, and the retry waits
# the Groq client announced ("Retrying request to ... in X seconds", logger groq._base_client; 429s and
# connection errors alike). Only the Groq client in this process is seen: not the v1 subprocess.
CALLS = {"n": 0, "n429": 0, "waited": 0.0}


class _CountGroq(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "api.groq.com" in msg and " 200 " in msg:       # retried 429s are not LLM calls
            CALLS["n"] += 1
        elif "api.groq.com" in msg and " 429 " in msg:
            CALLS["n429"] += 1


class _GroqWaits(logging.Handler):
    def emit(self, record):
        if record.msg.startswith("Retrying request to") and len(record.args or ()) == 2:
            CALLS["waited"] += float(record.args[1])


logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpx").addHandler(_CountGroq())
logging.getLogger("httpx").propagate = False
logging.getLogger("groq._base_client").setLevel(logging.INFO)
logging.getLogger("groq._base_client").addHandler(_GroqWaits())
logging.getLogger("groq._base_client").propagate = False


# ---------------------------------------------------------------- helpers

def key(cid: str) -> tuple:
    """chunk id 'PenalFr/s02/220-bis' -> ('PenalFr', '220-bis')."""
    return (cid.split("/")[0], cid.split("/")[-1].split("~")[0])


def gold_set(item) -> set:
    return {(source_stem(g["source"]), str(g["article"])) for g in item.get("gold") or []}


def out_path(set_path: Path, system: str) -> Path:
    return ROOT / "results" / f"answer_eval_{set_path.stem}_{system}.jsonl"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


# ---------------------------------------------------------------- systems

def run_v2(item) -> dict:
    from agent_graph import app
    s = app.invoke({"question": item["question"]},
                   {"configurable": {"thread_id": f"eval-{item['id']}-{time.time_ns()}"}})
    ev = (s.get("evidence") or {}) if s.get("status") == "done" else {}
    rec = {"intent": s.get("intent"), "status": s.get("status"), "retries": s.get("retries"), "tried": s.get("tried"),
           "evidence_ids": list(ev), "cited": s.get("cited") or [], "claims": s.get("claims") or [],
           "caveat": s.get("caveat"), "text": s.get("text"), "degraded": s.get("degraded"),
           "verdicts": s.get("verdicts") or [], "repairs": s.get("repairs"), "dropped": s.get("dropped"),
           "draft_claims": s.get("draft_claims") or [], "draft_text": s.get("draft_text") or "",
           "first_verdicts": s.get("first_verdicts") or []}
    if s.get("status") == "clarify":
        rec["clarification_question"] = s.get("text")
    rec["_evidence_text"] = {cid: e["text"] for cid, e in ev.items()}
    return rec


_V11 = {}


def run_v11(item) -> dict:
    from pipeline import load_vectorstore, ask_question
    if "vs" not in _V11:
        _V11["vs"] = load_vectorstore()
    answer, docs = ask_question(_V11["vs"], item["question"], None)     # no law_type hint (fair, like A2)
    ids = [d.metadata.get("chunk_id") for d in docs]
    return {"status": "done" if docs else "no_docs", "evidence_ids": ids, "cited": None, "claims": None, "text": answer,
            "_evidence_text": {d.metadata.get("chunk_id"): d.page_content for d in docs}}


class V1Process:
    def __init__(self, worktree: Path):
        if not (worktree / "pipeline.py").exists() or not (worktree / "db").exists():
            sys.exit(f"--system v1 needs a git worktree of {V1_TAG} that contains the old index as db/. Create it with:\n"
                     f"  git worktree add <folder> {V1_TAG}\n"
                     f"  mklink /J <folder>\\db <repo>\\db          (Windows; or copy db/)\n"
                     f"then pass --v1-worktree <folder>. Nothing was run.")
        self.p = subprocess.Popen([sys.executable, str(ROOT / "eval" / "v1_runner.py"), str(worktree)], cwd=worktree,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    def run(self, item) -> dict:
        self.p.stdin.write(json.dumps({"id": item["id"], "question": item["question"]}, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "error" in r:
            raise RuntimeError(r["error"])
        # v1 chunks have no chunk ids: use "<stem>/v1/<article>" (article = base number in the v1 index)
        ids = [f"{source_stem(d['source'])}/v1/{d['article']}" for d in r["docs"]]
        return {"status": "done" if r["docs"] else "no_docs", "evidence_ids": ids, "cited": None, "claims": None,
                "text": r["answer"], "_evidence_text": {i: d["text"] for i, d in zip(ids, r["docs"])},
                "_seconds": r["seconds"], "_llm_calls": r["llm_calls"]}


def pdf_text(cid: str):
    if "/v1/" in cid:
        stem, _, art = cid.split("/")
        return pdf_article_text(stem, art)          # None if the label is ambiguous in a bundled PDF
    return from_chunk_id(cid)


# ---------------------------------------------------------------- deterministic metrics

def numeric_check(text: str, cids: list) -> dict:
    """eval/pdf_factcheck.py rule: every number of the text in the cited PDF texts. A missing "article N" that a
    cited article designates as "article précédent/suivant" (ci-dessus/ci-après) is accepted and listed in
    `implicit`: ok is then True (not a hallucination), and the answer is also counted under "implicit reference"."""
    texts = [pdf_text(c) for c in cids]
    ok, nums, missing = claim_numbers_ok(text, texts)
    implicit = implicit_reference_numbers(text, missing, cids, texts) if ok is False else set()
    if ok is False and implicit == missing:
        ok = True
    return {"ok": ok, "numbers": sorted(nums), "missing": sorted(missing - implicit), "implicit": sorted(implicit)}


def deterministic(item, rec) -> dict:
    gold = gold_set(item)
    ev = rec.get("evidence_ids") or []
    m = {"gold_hit": any(key(c) in gold for c in ev) if gold else None,
         "gold_cited": (any(key(c) in gold for c in rec["cited"]) if gold else None) if rec.get("cited") is not None else None}
    if rec.get("claims") is not None:                       # v2: structured claims
        m["cited_not_in_evidence"] = sorted({c for cl in rec["claims"] for c in cl["cited_ids"] if c not in ev})
        m["pdf_numeric_claims"] = [numeric_check(cl["text"], cl["cited_ids"]) for cl in rec["claims"]]
    else:                                                   # v1 / v1.1: the answer text as one block vs the 6 docs
        m["cited_not_in_evidence"] = None
        m["pdf_numeric_claims"] = [numeric_check(rec.get("text") or "", ev) | {"block": True}]
    oks = [c["ok"] for c in m["pdf_numeric_claims"]]
    m["pdf_numeric_ok"] = False if False in oks else (None if all(o is None for o in oks) else True)
    m["pdf_numeric_implicit"] = any(c.get("implicit") for c in m["pdf_numeric_claims"])
    if rec.get("draft_claims"):                             # v2 with the verifier loop: same check on the first draft
        m["draft_cited_not_in_evidence"] = sorted({c for cl in rec["draft_claims"] for c in cl["cited_ids"] if c not in ev})
        m["pdf_numeric_draft_claims"] = [numeric_check(cl["text"], cl["cited_ids"]) for cl in rec["draft_claims"]]
        oks = [c["ok"] for c in m["pdf_numeric_draft_claims"]]
        m["pdf_numeric_ok_draft"] = False if False in oks else (None if all(o is None for o in oks) else True)
    exp, st = item["expected"], rec.get("status")
    m["gate_correct"] = (st == exp) if exp in ("clarify", "refuse") else None
    m["false_clarify"] = (st == "clarify") if exp == "answer" else None
    m["false_refuse"] = (st == "refuse") if exp == "answer" else None
    return m


# ---------------------------------------------------------------- judge

JUDGE_PROMPT = """You grade an answer to a question about Tunisian law against reference points written by a lawyer and
against the legal articles the answer relies on.

You receive: the question, the numbered reference points, the answer, and the text of the cited articles
(when an article is marked FALLBACK, its text comes from the search index, not from the official PDF).

Return:
- covered: one boolean per reference point, in order. True only if the answer states that point (same rule,
  same numbers); a vague mention is not enough.
- contradiction: true if the answer states something that contradicts a reference point.
- unsupported_statements: short quotes or paraphrases of statements in the answer that the cited articles do not
  support (facts, numbers, conditions, penalties). Empty list if every statement is supported.
Judge only from the texts given. Do not use outside legal knowledge."""


def judge(item, rec, model: str) -> dict:
    from typing import List
    from pydantic import BaseModel
    from langchain_groq import ChatGroq

    class Judgement(BaseModel):
        covered: List[bool]
        contradiction: bool
        unsupported_statements: List[str]

    cited = rec["cited"] if rec.get("cited") is not None else rec.get("evidence_ids") or []
    arts = []
    for cid in cited:
        t = pdf_text(cid)
        if t:
            arts.append(f"[{cid}] (PDF)\n{normalize(t)[:2000]}")
        elif cid in rec.get("_evidence_text", {}):
            arts.append(f"[{cid}] (FALLBACK: index chunk)\n{normalize(rec['_evidence_text'][cid])[:2000]}")
    rp = item.get("reference_points") or []
    human = (f"Question: {item['question']}\n\nReference points:\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(rp, 1))
             + f"\n\nAnswer:\n{rec.get('text') or ''}\n\nCited articles:\n" + ("\n\n".join(arts) or "(none)"))
    llm = ChatGroq(model=model, temperature=0).with_structured_output(Judgement, method="json_schema")
    j = llm.invoke([("system", JUDGE_PROMPT), ("human", human)])
    out = {"covered": j.covered, "contradiction": j.contradiction, "unsupported_statements": j.unsupported_statements,
           "fallback_articles": sum("FALLBACK" in a for a in arts)}
    if len(j.covered) != len(rp):
        out["error"] = f"covered has {len(j.covered)} values for {len(rp)} reference points"
        return out
    n_cov = sum(j.covered)
    out["verdict"] = ("correct" if n_cov == len(rp) and not j.contradiction else
                      "partial" if n_cov >= 1 else "incorrect")
    return out


def hallucinated(rec, m, j) -> bool | None:
    if j is None or "verdict" not in j:
        return None
    return bool(j["unsupported_statements"]) or bool(m.get("cited_not_in_evidence")) or m["pdf_numeric_ok"] is False


# ---------------------------------------------------------------- summary

def metric_values(rows) -> dict:
    """metric -> list of per-item values (0/1 or float), None = not applicable; expected=answer items unless noted.
    Degraded rows are left out."""
    vals = defaultdict(list)
    for r in rows:
        if r.get("status") == "degraded":
            continue
        m, j = r["metrics"], r.get("judge")
        if r["expected"] in ("clarify", "refuse"):
            vals["gate accuracy (clarify/refuse items)"].append(int(m["gate_correct"]))
            continue
        vals["Hit@6 (gold in evidence)"].append(None if m["gold_hit"] is None else int(m["gold_hit"]))
        vals["gold cited"].append(None if m["gold_cited"] is None else int(m["gold_cited"]))
        vals["false clarify"].append(int(m["false_clarify"]))
        vals["false refuse"].append(int(m["false_refuse"]))
        vals["numeric check ok (answer)"].append(None if m["pdf_numeric_ok"] is None else int(m["pdf_numeric_ok"]))
        vals["numeric check unverifiable (answer)"].append(int(m["pdf_numeric_ok"] is None))
        vals["numeric implicit reference (answer)"].append(          # absent in rows written before this rule
            None if "pdf_numeric_implicit" not in m else int(m["pdf_numeric_implicit"]))
        claims = [c["ok"] for c in m["pdf_numeric_claims"] if c["ok"] is not None]
        vals["numeric check ok (claim share)"].append(sum(claims) / len(claims) if claims else None)
        vals["seconds"].append(r["seconds"])
        vals["LLM calls"].append(r["llm_calls"])
        vals["elapsed total (s)"].append(r.get("elapsed_total"))          # absent in rows written before these fields
        vals["429 responses"].append(r.get("n_429"))
        vals["client retry waits (s)"].append(r.get("waited_seconds"))
        v = j.get("verdict") if j else None
        for name in ("correct", "partial", "incorrect"):
            vals[name].append(None if v is None else int(v == name))
        h = r.get("hallucinated")
        vals["hallucination (answer)"].append(None if h is None else int(h))
        n_claims = len(r.get("claims") or [])
        vals["hallucination (claim-level, judge)"].append(
            None if v is None or not n_claims else min(len(j["unsupported_statements"]), n_claims) / n_claims)
        # verifier loop (v2 rows with a first draft, i.e. the verifier ran). Claim-level share of draft claims not
        # judged "supported" on the first pass (a claim without a verdict counts as flagged, as verify.bad_claims).
        drafted = bool(r.get("draft_claims"))
        first = {v["claim_index"]: v["verdict"] for v in r.get("first_verdicts") or []}
        n_draft = len(r.get("draft_claims") or [])
        vals["claims flagged on 1st pass"].append(
            sum(first.get(i) != "supported" for i in range(1, n_draft + 1)) / n_draft
            if drafted and r.get("first_verdicts") else None)
        vals["repaired (repairs == 1)"].append(int(r["repairs"] == 1) if drafted else None)
        vals["dropped claims (mean)"].append(r["dropped"] if drafted else None)
        vals["no supported claim left"].append(int(not r.get("claims")) if drafted else None)
        # numeric check, answer level: 1 if a claim has a number missing from its cited PDF articles; same items for
        # draft and final (an answer whose claims were all dropped states no number: 0)
        vals["numeric miss, draft claims (answer)"].append(
            int(any(c["ok"] is False for c in m["pdf_numeric_draft_claims"])) if drafted else None)
        vals["numeric miss, final claims (answer)"].append(
            int(any(c["ok"] is False for c in m["pdf_numeric_claims"])) if drafted else None)
        jd = r.get("judge_draft")
        vd = jd.get("verdict") if jd else None
        vals["correct (draft)"].append(None if vd is None else int(vd == "correct"))
        hd = r.get("hallucinated_draft")
        vals["hallucination (answer, draft)"].append(None if hd is None else int(hd))
    return vals


def boot_ci(xs, rng) -> tuple:
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]


def summarize(paths: list[Path]):
    runs = {}
    for p in paths:
        runs[p.stem] = {r["id"]: r for r in read_jsonl(p)}         # latest row per item (a resume re-runs degraded)
        rows = list(runs[p.stem].values())
        n_deg = sum(r.get("status") == "degraded" for r in rows)
        rows = [r for r in rows if r.get("status") != "degraded"]
        vals = metric_values(rows)
        print(f"\n=== {p.name}: {len(rows)} items (answer: {sum(r['expected'] == 'answer' for r in rows)}, "
              f"clarify/refuse: {sum(r['expected'] != 'answer' for r in rows)}); degraded (excluded, re-run on resume): "
              f"{n_deg}; 95% bootstrap CI over questions, {N_BOOT} resamples, seed {SEED}")
        for name, xs in vals.items():
            xs = [x for x in xs if x is not None]
            if not xs:
                print(f"  {name:<40} n=0   -")
                continue
            lo, hi = boot_ci(xs, random.Random(SEED))
            print(f"  {name:<40} n={len(xs):<4} {sum(xs) / len(xs):8.3f}   [{lo:.3f}, {hi:.3f}]")
    names = list(runs)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = runs[names[a]], runs[names[b]]
            ids = sorted(i for i in set(A) & set(B) if "degraded" not in (A[i].get("status"), B[i].get("status")))
            print(f"\n=== paired difference {names[b]} - {names[a]} on {len(ids)} common items")
            va = metric_values([A[i] for i in ids])
            vb = metric_values([B[i] for i in ids])
            for name in va:
                d = [y - x for x, y in zip(va[name], vb.get(name, [])) if x is not None and y is not None]
                if not d:
                    print(f"  {name:<40} n=0   -")
                    continue
                lo, hi = boot_ci(d, random.Random(SEED))
                print(f"  {name:<40} n={len(d):<4} {sum(d) / len(d):+8.3f}   [{lo:+.3f}, {hi:+.3f}]")


# ---------------------------------------------------------------- audit export

def export_audit(path: Path, system: str, n: int):
    rows = [r for r in {r["id"]: r for r in read_jsonl(path)}.values()
            if r["expected"] == "answer" and r.get("status") != "degraded"]
    pick = random.Random(SEED).sample(rows, min(n, len(rows)))
    out = ROOT / "results" / f"audit_sample_{system}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        per_claim = [h for k in range(1, 7) for h in (f"claim_{k}", f"claim_{k}_relevant_to_question (y/n)")]
        w.writerow(["id", "question", "answer", "cited_article_pdf_text", "judge_verdict", "judge_hallucination",
                    "human_correct", "human_hallucination"] + per_claim)
        for r in pick:
            cited = r.get("cited") if r.get("cited") is not None else r.get("evidence_ids") or []
            texts = " || ".join(f"{c}: {normalize(pdf_text(c) or '(no PDF text)')[:600]}" for c in cited)
            j = r.get("judge") or {}
            claims = [c["text"] for c in r.get("claims") or []][:6]          # Answer allows at most 6 claims
            claims += [""] * (6 - len(claims))
            w.writerow([r["id"], r["question"], r.get("text"), texts, j.get("verdict", ""),
                        "" if r.get("hallucinated") is None else int(r["hallucinated"]), "", ""]
                       + [x for c in claims for x in (c, "")])
    print(f"wrote {out.relative_to(ROOT)} ({len(pick)} rows); fill human_correct / human_hallucination with 1 or 0, "
          f"then run: python eval/audit_agreement.py {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- main

CALL_RANGES = {   # (min, max) Groq calls per question, without the judge
    "v2": ((1, 10), "Analyze 1; clarify/refuse stops there. Else: translation 0-1 (AR/EN), Grade 1, up to 2 retries "
                    "(Reformulate, Grade, Reformulate = +3), Answer 1, Verify 1, at most one repair (Answer, Verify = +2)"),
    "v1.1": ((1, 2), "translation 0-1 (AR/EN), generation 1"),
    "v1": ((2, 3), "rewrite 1, sub-queries 0-1 (complex questions), generation 1"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default=str(ROOT / "eval" / "answer_set.json"))
    ap.add_argument("--system", choices=["v2", "v1.1", "v1"], default="v2")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-ids")
    ap.add_argument("--out")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    ap.add_argument("--judge-draft", action="store_true",
                    help="v2: also judge draft_text (the unverified first answer); one more judge call when it differs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-gap", type=float, default=0, metavar="SECONDS", help="sleep between questions")
    ap.add_argument("--summarize", nargs="+", metavar="JSONL")
    ap.add_argument("--export-audit", type=int, metavar="N")
    ap.add_argument("--v1-worktree", help=f"git worktree of {V1_TAG} with the old index in db/ (for --system v1)")
    args = ap.parse_args()

    if args.summarize:
        summarize([Path(p) for p in args.summarize])
        return
    set_path = Path(args.set)
    out = Path(args.out) if args.out else out_path(set_path, args.system)
    if args.export_audit:
        export_audit(out, args.system, args.export_audit)
        return

    items = json.loads(set_path.read_text(encoding="utf-8"))
    if args.only_ids:
        wanted = set(args.only_ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    if args.limit:
        items = items[:args.limit]
    done = {r["id"]: r.get("status") != "degraded" for r in read_jsonl(out)}      # latest row per item
    done = {i for i, ok in done.items() if ok}
    todo = [i for i in items if i["id"] not in done]
    need_judge = not args.skip_judge
    (lo, hi), how = CALL_RANGES[args.system]
    j = 0 if args.skip_judge else 1
    print(f"set {set_path.name}: {len(items)} items selected, {len(done & {i['id'] for i in items})} already in "
          f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}, {len(todo)} to run; system {args.system}; "
          f"judge: {'skipped' if args.skip_judge else (args.judge_model or 'NOT SET')}")
    print(f"LLM calls per question: {lo + j}-{hi + j} ({how}{'' if args.skip_judge else '; judge 1 on answered items only'}); "
          f"for these {len(todo)} items: {len(todo) * (lo + j)}-{len(todo) * (hi + j)}; for 100 items: {100 * (lo + j)}-{100 * (hi + j)}")
    if args.dry_run:
        return
    if need_judge and not args.judge_model:
        sys.exit("No judge model: set JUDGE_MODEL or pass --judge-model (a model different from the generator), "
                 "or use --skip-judge. No model name is guessed.")
    if need_judge and args.judge_model == GENERATOR_MODEL:
        sys.exit(f"The judge must be a different model from the generator ({GENERATOR_MODEL}).")

    runner = V1Process(Path(args.v1_worktree or "")) if args.system == "v1" else None
    run = {"v2": run_v2, "v1.1": run_v11}.get(args.system) or runner.run
    out.parent.mkdir(parents=True, exist_ok=True)
    for k, item in enumerate(todo, 1):
        if k > 1 and args.min_gap:
            time.sleep(args.min_gap)
        print(f"[{k}/{len(todo)}] {item['id']} ...", file=sys.stderr, flush=True)
        c0, t0 = CALLS["n"], time.time()
        r0, w0 = CALLS["n429"], CALLS["waited"]
        try:
            rec = run(item)
        except Exception as e:
            sys.exit(f"STOPPED on {item['id']}: {type(e).__name__}: {e}\nCompleted items are in {out}; re-run to resume.")
        seconds = rec.pop("_seconds", round(time.time() - t0, 1))
        calls = rec.pop("_llm_calls", CALLS["n"] - c0)
        degraded = bool(rec.get("degraded"))
        if degraded:                                # a node fell back: no metrics, re-run on resume
            rec["graph_status"], rec["status"] = rec.get("status"), "degraded"
        m = None if degraded else deterministic(item, rec)
        jd = jdd = None
        if need_judge and item["expected"] == "answer" and rec.get("status") == "done":
            c1 = CALLS["n"]
            try:
                jd = judge(item, rec, args.judge_model)
                if args.judge_draft and rec.get("draft_text"):
                    if rec["draft_text"] == rec.get("text"):                 # nothing repaired or dropped
                        jdd = dict(jd, same_as_final=True)
                    else:
                        d_cited = sorted({c for cl in rec["draft_claims"] for c in cl["cited_ids"]})
                        jdd = judge(item, rec | {"text": rec["draft_text"], "cited": d_cited}, args.judge_model)
            except Exception as e:
                sys.exit(f"STOPPED: judge failed on {item['id']}: {type(e).__name__}: {e}\nRe-run to resume.")
            jd["llm_calls"] = CALLS["n"] - c1
        rec.pop("_evidence_text", None)
        v1 = args.system == "v1"                    # its LLM calls run in the subprocess: 429s and waits not seen
        row = {"id": item["id"], "question": item["question"], "language": item.get("language"),
               "expected": item["expected"], "system": args.system, **rec, "seconds": seconds, "llm_calls": calls,
               "elapsed_total": round(time.time() - t0, 1),               # system + deterministic metrics + judge
               "n_429": None if v1 else CALLS["n429"] - r0,              # system + judge
               "waited_seconds": None if v1 else round(CALLS["waited"] - w0, 1),
               "metrics": m, "judge": jd}
        row["hallucinated"] = None if degraded else hallucinated(rec, m, jd)
        if args.judge_draft and m and "pdf_numeric_ok_draft" in m:
            row["judge_draft"] = jdd
            row["hallucinated_draft"] = hallucinated(rec, {"cited_not_in_evidence": m["draft_cited_not_in_evidence"],
                                                           "pdf_numeric_ok": m["pdf_numeric_ok_draft"]}, jdd)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if runner:
        runner.p.stdin.close()
    summarize([out])


if __name__ == "__main__":
    main()
