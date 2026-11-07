"""ESTIMATE of the extra LLM input tokens per question that the v2.1 evidence assembly (header "deepest", cross-reference
expansion capped at 300 embedding tokens) would add to v2, on the 110 development items. No LLM call: prompts are not
sent, only their evidence blocks are rebuilt with the agent's own formatting functions.

    python diagnostics/token_estimate.py --v11 COPY_OF_DB_V1_1 --v12 COPY_OF_DB_V1_2
        -> diagnostics/token_estimate.json, diagnostics/TOKEN_ESTIMATE.md

Graph nodes that receive the evidence (agent/agent_graph.py): grade (grade.format_evidence, 500 characters per article),
reformulate (format_evidence, on a retry), answer (answer._user_block, ANSWER_CHARS 3000 per article), verify
(verify._claims_block, only the cited articles, VERIFY_CHARS 3000), repair (make_answer again: _user_block).
condense and analyze do not receive it.
1. Characters -> LLM tokens: least squares on the saved v2 rows with exactly 4 LLM calls (analyze, grade, answer,
   verify) and measured usage: input tokens = a + b x (characters of the grade, answer and verify user blocks rebuilt
   from the row's evidence ids on db_v1_1 and its claims).
2. Extra characters per node call: v1.2 top 6 of each item (compare_v12.json) with the header and the capped expansion
   minus the same top 6 without them. verify: the first k top-6 articles, k = number of ids the saved v2 row cited
   (added articles assumed not cited).
3. Calls per question from the saved v2 row (llm_calls; the rows do not say which nodes ran): base path grade + answer
   + verify; extra calls either "retries first" (reformulate, grade, reformulate, then repair, verify) or "repair
   first" (repair, verify, then reformulate, grade, reformulate). Rows that did not answer (no evidence): 0.
"""
import sys
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "agent"), str(ROOT / "eval"), str(Path(__file__).resolve().parent)]
import config as C
from retrieval_diagnostic import items_by_set, saved_rows, SETS, table
from evidence_budget import load_index

OUT_JSON = ROOT / "diagnostics" / "token_estimate.json"
OUT_MD = ROOT / "diagnostics" / "TOKEN_ESTIMATE.md"
DEV = "development data, not evidence of generalisation"
EXTRA = {"retries first": ["reformulate", "grade", "reformulate", "repair", "verify"],
         "repair first": ["repair", "verify", "reformulate", "grade", "reformulate"]}


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot


def stat(xs):
    xs = sorted(xs)
    return {"mean": round(statistics.fmean(xs), 1), "median": statistics.median(xs), "p90": xs[max(0, int(0.9 * len(xs)) - 1)],
            "max": xs[-1]}


def blocks(question, language, evidence, cited):
    from grade import format_evidence
    from answer import _user_block, Answer, Claim
    from verify import _claims_block
    verify = 0
    if cited:
        verify = len(_claims_block(Answer(claims=[Claim(text="x", cited_ids=[c]) for c in cited], caveat=""), evidence))
    return {"grade": len(format_evidence(evidence)), "reformulate": len(format_evidence(evidence)),
            "answer": len(_user_block(question, language, evidence)), "repair": len(_user_block(question, language, evidence)),
            "verify": verify}


def main():
    from agent_graph import to_evidence
    v11 = Path(sys.argv[sys.argv.index("--v11") + 1])
    v12 = Path(sys.argv[sys.argv.index("--v12") + 1])
    _, by11 = load_index(v11)
    vs12, by12 = load_index(v12)
    import pipeline as P
    P._VS["vs"] = vs12
    rows2 = {s: saved_rows(s, "v2") for s in SETS}
    # 1. calibration on the saved v2 rows (v2 = db_v1_1, evidence text = article text)
    C.EVIDENCE_CONTEXT_HEADER, C.XREF_EXPAND = "none", False
    xs, ys = [], []
    for s, rows in rows2.items():
        for r in rows.values():
            tok = (r.get("tokens") or {})
            if r.get("llm_calls") != 4 or r.get("status") != "done" or tok.get("source") != "usage_metadata" or not r.get("evidence_ids"):
                continue
            if not all(i in by11 for i in r["evidence_ids"]):
                continue
            ev = to_evidence([by11[i] for i in r["evidence_ids"]])
            b = blocks(r["question"], r.get("language") or "fr", ev, [c for c in r.get("cited") or [] if c in ev])
            xs.append(b["grade"] + b["answer"] + b["verify"])
            ys.append(tok["input"])
    a, slope, r2 = ols(xs, ys)
    # 2. extra characters per node call, 3. calls per question
    recs = {(r["set"], r["id"]): r for r in json.loads((ROOT / "diagnostics" / "compare_v12.json").read_text(encoding="utf-8"))["records"]["v1.2"]}
    out = []
    for s, its in items_by_set().items():
        for it in its:
            rec, row = recs[(s, it["id"])], rows2[s].get(it["id"]) or {}
            top6 = [by12[i] for i in rec["top6_ids"]]
            k = min(len(row.get("cited") or []), 6)
            cited = [d.metadata["chunk_id"] for d in top6[:k]]
            C.EVIDENCE_CONTEXT_HEADER, C.XREF_EXPAND, C.XREF_TOKEN_CAP = "none", False, None
            b0 = blocks(it["question"], it.get("language") or "fr", to_evidence(top6), cited)
            C.EVIDENCE_CONTEXT_HEADER, C.XREF_EXPAND, C.XREF_TOKEN_CAP = "deepest", True, 300
            ev1 = to_evidence(top6)
            b1 = blocks(it["question"], it.get("language") or "fr", ev1, cited)
            extra_chars = {n: b1[n] - b0[n] for n in b0}
            calls = row.get("llm_calls") or 0
            answered = row.get("status") == "done" and bool(row.get("evidence_ids"))
            est = {}
            for name, order in EXTRA.items():
                nodes = (["grade", "answer"] + (["verify"] if cited else []) + order[:max(0, calls - 4)]) if answered else []
                est[name] = round(slope * sum(extra_chars[n] for n in nodes))
            out.append({"set": s, "id": it["id"], "language": it.get("language"), "llm_calls": calls, "answered": answered,
                        "added_articles": len(ev1) - len(top6), "extra_chars": extra_chars, "estimate": est,
                        "saved_v2_input_tokens": (row.get("tokens") or {}).get("input")})
    doc = {"calibration": {"rows": len(xs), "intercept": round(a, 1), "tokens_per_char": round(slope, 4), "r2": round(r2, 3)}, "items": out}
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(doc) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}; calibration {doc['calibration']}")


def markdown(doc):
    c, items = doc["calibration"], doc["items"]
    L = [f"# ESTIMATE: extra LLM input tokens per question of the v2.1 evidence (header 'deepest', expansion cap 300) — {DEV}", "",
         "These are estimates, not measurements: no prompt was sent; characters are converted with a fitted rate and the nodes "
         "each saved v2 row ran are inferred from its call count.", ""]
    L += table(f"Calibration: LLM input tokens = a + b x evidence-block characters (saved v2 rows, 4 calls) ({DEV})",
               ["rows", "a (tokens)", "b (tokens per character)", "R²"], [[c["rows"], c["intercept"], c["tokens_per_char"], c["r2"]]])
    L += table(f"Graph nodes that receive the evidence (agent/agent_graph.py) ({DEV})", ["Node", "Evidence shown", "Extra characters per call: mean", "p90"],
               [[n, w, stat([i["extra_chars"][n] for i in items])["mean"], stat([i["extra_chars"][n] for i in items])["p90"]]
                for n, w in (("grade", "every article, 500 characters"), ("reformulate", "every article, 500 characters (retries)"),
                             ("answer", "every article, 3000 characters"), ("verify", "cited articles, 3000 characters"),
                             ("repair", "every article, 3000 characters (after a failed verify)"))] +
               [["condense, analyze", "none", 0, 0]])
    rows = []
    groups = {SETS[s]: [i for i in items if i["set"] == s] for s in SETS}
    groups.update({f"language {l}": [i for i in items if i["language"] == l] for l in sorted({i["language"] for i in items})})
    groups["all"] = items
    for g, its in groups.items():
        saved = [i["saved_v2_input_tokens"] for i in its if i["saved_v2_input_tokens"]]
        row = [g, len(its), sum(i["answered"] for i in its), round(statistics.fmean(i["llm_calls"] for i in its), 2)]
        for name in EXTRA:
            st = stat([i["estimate"][name] for i in its])
            row += [st["mean"], st["p90"]]
        row += [round(statistics.fmean(saved)) if saved else "-",
                f"{100 * statistics.fmean(i['estimate']['retries first'] for i in its) / statistics.fmean(saved):.1f} %" if saved else "-"]
        rows.append(row)
    L += table(f"ESTIMATED extra input tokens per question added to v2 ({DEV})",
               ["Group", "items", "answered (saved v2)", "LLM calls mean (saved v2)", "retries first: mean", "p90",
                "repair first: mean", "p90", "saved v2 input tokens mean", "extra / saved (retries first)"], rows)
    return "\n".join(L)


if __name__ == "__main__":
    main()
