"""Computes every number of the report from files (results/, eval/, git) and writes:
    report/numbers.tex   \\newcommand macros (\\N...), "\\textit{à compléter}" when the source is missing
    report/tables/*.tex  booktabs tables
    report/data.json     data for make_figures.py
Prints a check of the facts stated in the task against the computed values. No number is typed in by hand.
Standard library only (plus eval/pdf_truth.py and eval/pdf_factcheck.py, which read the PDF cache).
"""
import re
import sys
import json
import glob
import random
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "eval"))
SEED, N_BOOT = 20261022, 5000
MISSING = r"\textit{à compléter}"

MACROS: dict[str, str] = {}
FACTS: list[tuple] = []
DATA: dict = {}


# ---------------------------------------------------------------- helpers

def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def mname(*parts) -> str:
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
              "8": "Eight", "9": "Nine"}
    s = "".join(digits.get(c, c) for c in "_".join(map(str, parts)))
    return "N" + "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", s) if w)


def fr(x, nd=1) -> str:
    return f"{x:.{nd}f}".replace(".", "{,}")


def put(value, *parts, nd=1, pct=False):
    """Define a macro; value None -> 'à compléter'. pct: value in [0,1] shown as a percentage number."""
    name = mname(*parts)
    if value is None:
        MACROS[name] = MISSING
    elif isinstance(value, str):
        MACROS[name] = value
    elif isinstance(value, int) and not pct:
        MACROS[name] = str(value)
    else:
        MACROS[name] = fr(value * 100 if pct else value, nd)
    return name


def agg(scores) -> dict | None:
    n = len(scores)
    if not n:
        return None
    return {"n": n, "hit": sum(s["hit"] for s in scores) / n, "hit1": sum(s["hit_at_1"] for s in scores) / n,
            "recall": sum(s["recall"] for s in scores) / n, "mrr": sum(s["rr"] for s in scores) / n}


def put_metrics(m, *parts):
    put(m and m["hit"], *parts, "hitsix", pct=True)
    put(m and m["hit1"], *parts, "hitone", pct=True)
    put(m and m["recall"], *parts, "recall", pct=True)
    put(m and m["mrr"], *parts, "mrr", nd=3)
    put(m and m["n"], *parts, "n")


def fact(label, claimed, computed):
    FACTS.append((label, claimed, computed))


def tex_escape(s: str) -> str:
    from pdf_truth import normalize
    s = normalize(s)
    s = "".join(c if ord(c) < 0x180 else " " for c in s)          # pdflatex-safe: Latin only (no Arabic script)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def write_table(name, header, rows, caption, label, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[H]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    (REP / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def placeholder_table(name, caption):
    (REP / "tables" / f"{name}.tex").write_text(
        r"\begin{center}\fbox{\parbox{0.8\linewidth}{\centering Tableau à générer : " + caption +
        r" (\textit{à compléter})}}\end{center}" + "\n", encoding="utf-8")


def cell(v, nd=1, pct=True):
    return MISSING if v is None else fr(v * 100 if pct else v, nd)


def concept_boot(rows_by_variant: dict, concept_of: dict) -> dict:
    """Hit@6 per variant and paired difference vs the first variant, 95% CI resampling CONCEPTS."""
    names = list(rows_by_variant)
    concepts = sorted(set(concept_of.values()))
    ids_by_c = defaultdict(list)
    for qid, c in concept_of.items():
        ids_by_c[c].append(qid)
    rng = random.Random(SEED)
    samples = {n: [] for n in names}
    diffs = {n: [] for n in names[1:]}
    for _ in range(N_BOOT):
        pick = [ids_by_c[rng.choice(concepts)] for _ in concepts]
        ids = [i for grp in pick for i in grp]
        means = {n: sum(rows_by_variant[n][i] for i in ids) / len(ids) for n in names}
        for n in names:
            samples[n].append(means[n])
        for n in names[1:]:
            diffs[n].append(means[n] - means[names[0]])

    def ci(xs):
        xs = sorted(xs)
        return xs[int(0.025 * N_BOOT)], xs[int(0.975 * N_BOOT) - 1]
    out = {}
    for n in names:
        vals = list(rows_by_variant[n].values())
        out[n] = {"hit": sum(vals) / len(vals), "ci": ci(samples[n])}
        if n in diffs:
            out[n]["diff"] = out[n]["hit"] - out[names[0]]["hit"]
            out[n]["diff_ci"] = ci(diffs[n])
    return out


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------- sections of numbers

def splits():
    sp = json.loads((ROOT / "eval" / "split.json").read_text(encoding="utf-8"))
    qs = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    dev = set(sp["dev"])
    nd, nt = sum(q["concept_id"] in dev for q in qs), sum(q["concept_id"] not in dev for q in qs)
    put(len(qs), "q", "total"); put(nd, "q", "dev"); put(nt, "q", "test")
    put(len(dev), "concepts", "dev"); put(len(sp["test"]), "concepts", "test"); put(len(dev) + len(sp["test"]), "concepts", "total")
    put(str(sp["seed"]), "seed")
    put(N_BOOT, "boot")
    langs = Counter(q["language"] for q in qs)
    for l in ("fr", "ar", "en"):
        put(langs[l], "q", "lang", l)
    fact("dev questions/concepts", "35/15", f"{nd}/{len(dev)}")
    fact("test questions/concepts", "52/22", f"{nt}/{len(sp['test'])}")
    fact("seed", "20261022", str(sp["seed"]))


def baseline():
    v1 = load("official_v1_final_labels.json")
    if v1:
        for split in ("dev", "test"):
            m = agg([c["score"] for c in v1["cases"] if c["split"] == split])
            put_metrics(m, "vone", split)
            if split == "test":
                fact("v1 test Hit6/Hit1/Recall/MRR", "57.7/32.7/34.4/0.405",
                     f"{m['hit']*100:.1f}/{m['hit1']*100:.1f}/{m['recall']*100:.1f}/{m['mrr']:.3f}")
        s = v1["meta"]["avg_seconds"]
        put(s["t_retrieve"] + s["t_rerank"], "vone", "sec", "retrieval")
        put(s["t_llm"], "vone", "sec", "llm")
    else:
        put_metrics(None, "vone", "test"); put_metrics(None, "vone", "dev")
    v11t, v11d = load("final_v1_1_test.json"), load("item5_rrf_dev.json")
    mt = agg([c["score"] for c in v11t["cases"]]) if v11t else None
    md = agg([c["score"] for c in v11d["cases"]]) if v11d else None
    put_metrics(mt, "voneone", "test"); put_metrics(md, "voneone", "dev")
    if mt:
        fact("v1.1 (A1) test", "80.8/59.6/50.4/0.687", f"{mt['hit']*100:.1f}/{mt['hit1']*100:.1f}/{mt['recall']*100:.1f}/{mt['mrr']:.3f}")
        s = v11t["meta"]["avg_seconds"]
        put(s["t_llm"] + s["t_retrieve"] + s["t_rerank"], "voneone", "sec", "total")
        put(v11t["meta"]["llm_calls"], "voneone", "test", "llmcalls")
        fact("v1.1 seconds per question (test)", "about 3", f"{s['t_llm'] + s['t_retrieve'] + s['t_rerank']:.2f}")
    if md:
        fact("v1.1 (A1) dev", "88.6/57.1/54.1/0.659", f"{md['hit']*100:.1f}/{md['hit1']*100:.1f}/{md['recall']*100:.1f}/{md['mrr']:.3f}")
    if v1:
        fact("v1 seconds per question", "about 18 (plus LLM)", f"retrieve+rerank {s_v1(v1):.2f}; LLM {v1['meta']['avg_seconds']['t_llm']} (cache waits)")
    DATA["f2"] = {"v1": agg([c["score"] for c in v1["cases"] if c["split"] == "test"]) if v1 else None, "v11": mt}
    rows = []
    for label, m in (("v1 (test)", DATA["f2"]["v1"]), ("v1.1, A1 (test)", mt), ("v1.1, A1 (dev)", md)):
        rows.append([label, cell(m and m["hit"]), cell(m and m["hit1"]), cell(m and m["recall"]),
                     cell(m and m["mrr"], 3, False), str(m["n"]) if m else MISSING])
    write_table("t_v1_v11", ["Système", "Hit@6", "Hit@1", "Rappel", "MRR", "n"], rows,
                "Récupération, v1 et v1.1 (A1 : avec l'indice de code du jeu de données). Source : \\texttt{results/}.",
                "tab:v1v11")


STEPS = [("v1 (étiquettes finales)", "official_v1_final_labels.json", "dev", "--"),
         ("1. Découpage aux vrais intitulés, identifiants stables", "chunkfix_dev_final_labels.json", None, "oui"),
         ("2. Question brute d'abord (traduction AR/EN seulement)", "item2_rawfirst_dev.json", None, "oui"),
         ("3. Suppression de la table de synonymes (fuite)", "item3_nosyn_dev.json", None, "oui"),
         ("4. Reranker : correction, réserve de 20, int8", "item4_combo_dev.json", None, "oui"),
         ("5. Fusion par rang réciproque (RRF)", "item5_rrf_dev.json", None, "oui"),
         ("6. Texte de contexte avant l'\\emph{embedding}", "item6_ctx_dev.json", None, "non"),
         ("7. bge-m3 au lieu de Solon", "item7_bge_dev.json", None, "non")]


def steps():
    rows, out = [], []
    for label, f, split, kept in STEPS:
        d = load(f)
        m = agg([c["score"] for c in d["cases"] if split is None or c.get("split") == split]) if d else None
        rows.append([label, cell(m and m["hit"]), cell(m and m["hit1"]), cell(m and m["recall"]),
                     cell(m and m["mrr"], 3, False), kept])
        out.append((label, m))
    n_dev = next((m["n"] for _, m in out[1:] if m), None)
    write_table("t_steps", [f"Étape (dev, {n_dev} questions)", "Hit@6", "Hit@1", "Rappel", "MRR", "Gardée"], rows,
                "Les sept étapes de v1 à v1.1, chacune cumulative, mesurées sur le dev. Scores lus dans "
                "\\texttt{results/} (fichiers \\texttt{chunkfix\\_dev\\_final\\_labels} et \\texttt{item2} à \\texttt{item7}).",
                "tab:steps", align="p{7.2cm}rrrrc")
    DATA["steps"] = [(l, m and m["hit"]) for l, m in out]


def s_v1(v1):
    s = v1["meta"]["avg_seconds"]
    return s["t_retrieve"] + s["t_rerank"]


VARIANTS = {"A1 baseline + law hint": "A1", "A2 baseline, no hint": "A2", "B1 plan, code filter": "B1",
            "B2 plan, no filter": "B2", "C  A2 + B1 merged": "C", "D  plan, soft filter": "D", "E  A2 + D merged": "E",
            "G  raw + Analyze hint (soft)": "G"}


def plan_vs_raw():
    out = {}
    for split in ("dev", "test"):
        d = load(f"plan_vs_raw_{split}.json")
        out[split] = {}
        if not d:
            continue
        for long, short in VARIANTS.items():
            if long in d["variants"]:
                m = agg([r["score"] for r in d["variants"][long]])
                out[split][short] = m
                put_metrics(m, "pvr", split, short)
        out[split + "_prov"] = d.get("provenance")
    t = out.get("test", {})
    if "A2" in t:
        fact("A2 test", "75.0/57.7/49.2/0.646", f"{t['A2']['hit']*100:.1f}/{t['A2']['hit1']*100:.1f}/{t['A2']['recall']*100:.1f}/{t['A2']['mrr']:.3f}")
    if "D" in t:
        fact("Plan D test", "73.1/51.9/47.0/0.609", f"{t['D']['hit']*100:.1f}/{t['D']['hit1']*100:.1f}/{t['D']['recall']*100:.1f}/{t['D']['mrr']:.3f}")
    DATA["f3"] = {s: {v: (out[s].get(v) or {}).get("hit") for v in ("A2", "D")} for s in ("dev", "test")}
    DATA["f3"]["n"] = {s: (out[s].get("A2") or {}).get("n") for s in ("dev", "test")}
    rows = []
    for v in ("A1", "A2", "B1", "B2", "C", "D", "E", "G"):
        dv, tv = out["dev"].get(v), out["test"].get(v)
        rows.append([v, cell(dv and dv["hit"]), cell(dv and dv["mrr"], 3, False), cell(tv and tv["hit"]) if tv else "--",
                     cell(tv and tv["mrr"], 3, False) if tv else "--"])
    nd, nt = (out["dev"].get("A2") or {}).get("n"), (out["test"].get("A2") or {}).get("n")
    write_table("t_plan", ["Variante", "Hit@6 dev", "MRR dev", "Hit@6 test", "MRR test"], rows,
                f"Planification contre question brute (dev : {nd} questions, test : {nt}). A1 utilise l'indice de code du "
                "jeu de données ; A2 est la référence équitable. G n'a été évaluée que sur le dev.", "tab:plan")
    return out


def loop(pvr):
    a2, g = load("eval_loop_dev_a2.json"), load("eval_loop_dev_g.json")
    dev_pvr = load("plan_vs_raw_dev.json")
    if not (a2 and g and dev_pvr):
        for k in ("A2", "G", "LoopAtwo", "LoopG", "Aone"):
            put(None, "loop", k, "hitsix")
        placeholder_table("t_ablation", "ablation A2 → G → boucle → oracle")
        return
    hits = {"A2": {r["id"]: int(r["score"]["hit"]) for r in a2["rows"]["A2 baseline, no hint"]},
            "G": {r["id"]: int(r["score"]["hit"]) for r in g["rows"]["G  round 0 (A2 + hint)"]},
            "Boucle depuis A2": {r["id"]: int(r["score"]["hit"]) for r in a2["rows"]["LOOP final"]},
            "Boucle depuis G": {r["id"]: int(r["score"]["hit"]) for r in g["rows"]["LOOP final"]},
            "A1 (oracle)": {r["id"]: int(r["score"]["hit"]) for r in dev_pvr["variants"]["A1 baseline + law hint"]}}
    concept_of = {r["id"]: r["concept_id"] for r in a2["rows"]["A2 baseline, no hint"]}
    bs = concept_boot(hits, concept_of)
    keys = {"A2": "Atwo", "G": "G", "Boucle depuis A2": "LoopAtwo", "Boucle depuis G": "LoopG", "A1 (oracle)": "Aone"}
    for n, k in keys.items():
        put(bs[n]["hit"], "loop", k, "hitsix", pct=True)
        put(bs[n]["ci"][0], "loop", k, "lo", pct=True); put(bs[n]["ci"][1], "loop", k, "hi", pct=True)
        if "diff" in bs[n]:
            put(bs[n]["diff"], "loop", k, "diff", pct=True)
            put(bs[n]["diff_ci"][0], "loop", k, "difflo", pct=True); put(bs[n]["diff_ci"][1], "loop", k, "diffhi", pct=True)
    fact("dev Hit@6 A2 / G / loop-from-A2 / loop-from-G / A1", "71.4 / 80.0 / 80.0 / 85.7 / 88.6",
         " / ".join(f"{bs[n]['hit']*100:.1f}" for n in keys))
    rows = [[n, cell(bs[n]["hit"]), f"[{cell(bs[n]['ci'][0])} ; {cell(bs[n]['ci'][1])}]",
             (r"$+$" if bs[n]["diff"] >= 0 else r"$-$") + cell(abs(bs[n]["diff"])) if "diff" in bs[n] else "--",
             f"[{cell(bs[n]['diff_ci'][0])} ; {cell(bs[n]['diff_ci'][1])}]" if "diff" in bs[n] else "--"] for n in keys]
    write_table("t_ablation", ["Variante (dev)", "Hit@6", "IC 95 \\%", "Écart vs A2", "IC 95 \\% de l'écart"], rows,
                f"Ablation sur le dev ({len(concept_of)} questions, {len(set(concept_of.values()))} concepts). "
                f"Intervalles par bootstrap apparié rééchantillonnant les "
                f"concepts ({N_BOOT} tirages, graine {SEED}).", "tab:ablation")
    DATA["f4"] = {n: bs[n] for n in keys}
    # by language
    lang_of = {r["id"]: r["language"] for r in a2["rows"]["A2 baseline, no hint"]}
    bylang = {n: {l: (lambda ids: sum(hits[n][i] for i in ids) / len(ids))([i for i in hits[n] if lang_of[i] == l])
                  for l in ("fr", "ar", "en")} for n in keys}
    DATA["f5"] = {"by_lang": bylang, "n": dict(Counter(lang_of.values()))}
    rows = [[n] + [cell(bylang[n][l]) for l in ("fr", "ar", "en")] for n in keys]
    write_table("t_bylang", ["Variante (dev)", f"FR (n={DATA['f5']['n']['fr']})", f"AR (n={DATA['f5']['n']['ar']})",
                             f"EN (n={DATA['f5']['n']['en']})"], rows, "Hit@6 par langue sur le dev.", "tab:bylang")
    # grade grid, retries, LLM requests
    DATA["f6"], DATA["f7"] = {}, {}
    for start, d in (("A2", a2), ("G", g)):
        grid = d["stats"]["grid"]
        DATA["f6"][start] = {k: grid.get(k, 0) for k in ("enough|True", "enough|False", "weak|True", "weak|False")}
        DATA["f7"][start] = {"retries": {int(k): v for k, v in d["stats"]["retries"].items()},
                             "llm_requests": [x["llm_requests"] for x in d["loop"]]}
        k = "Atwo" if start == "A2" else "G"
        reqs = DATA["f7"][start]["llm_requests"]
        put(sum(reqs) / len(reqs), "loop", k, "llmper", nd=2)
        put(sum(v for r, v in DATA["f7"][start]["retries"].items() if r > 0), "loop", k, "retried")
        oc = d["stats"]["retried_outcomes"]
        put(oc.get("better", 0), "loop", k, "better"); put(oc.get("worse", 0), "loop", k, "worse"); put(oc.get("same", 0), "loop", k, "same")
        for key in DATA["f6"][start]:
            v, h = key.split("|")
            put(DATA["f6"][start][key], "grade", k, v, "hit" if h == "True" else "miss")
    for k, v in (("analyze", "analyze_prompt_sha1"), ("plan", "plan_prompt_sha1"), ("codes", "codes_py_sha1"),
                 ("grade", "grade_prompt_sha1"), ("reformulate", "reformulate_prompt_sha1"), ("combined", "prompt_hash")):
        put(r"\texttt{" + g["provenance"][v] + "}", "hash", k)


def calibration():
    c = load("verifier_calibration.json")
    names = {"good": "Good", "number_changed": "Number", "wrong_article": "Wrong", "added_condition": "Cond"}
    if not c:
        for k in names.values():
            put(None, "calib", k, "rate")
        put(None, "calib", "n"); put(None, "hash", "verify")
        placeholder_table("t_calib", "étalonnage du vérificateur")
        return
    put(c["n"], "calib", "n"); put(r"\texttt{" + c["verify_prompt_sha1"] + "}", "hash", "verify")
    put(len(c["chunk_mismatch"]), "calib", "chunkmismatch")
    rows, DATA["f8"] = [], {}
    for t, k in names.items():
        s = c["summary"][t]
        rate = s["flagged"] / s["n"] if s["n"] else None
        put(rate, "calib", k, "rate", pct=True); put(s["wilson95"][0], "calib", k, "lo", pct=True)
        put(s["wilson95"][1], "calib", k, "hi", pct=True); put(s["flagged"], "calib", k, "flagged")
        v = s["verdicts"]
        rows.append([t.replace("_", r"\_"), str(s["n"]), str(s["flagged"]), cell(rate),
                     f"[{cell(s['wilson95'][0])} ; {cell(s['wilson95'][1])}]",
                     str(v.get("supported", 0)), str(v.get("partly", 0)), str(v.get("unsupported", 0))])
        DATA["f8"][t] = {"rate": rate, "ci": s["wilson95"], "n": s["n"]}
    misses = [r for r in c["rows"] if (r["type"] == "good") == (r["verdict"] != "supported")]
    put(len(misses), "calib", "misses")
    write_table("t_calib", ["Type d'affirmation", "n", "Signalées", "Taux", "IC 95 \\% (Wilson)", "supported", "partly",
                            "unsupported"], rows,
                "Étalonnage du vérificateur par mutations (une affirmation de chaque type par article). « Signalée » : "
                "verdict différent de \\emph{supported} ; pour \\emph{good}, c'est une fausse alarme.", "tab:calib")
    DATA["calib_misses"] = misses


def audit():
    a = load("audit_chunks.json")
    if not a:
        for k in ("match", "mismatch", "nopdf", "rate", "collisions", "collsources"):
            put(None, "audit", k)
        placeholder_table("t_audit", "audit des chunks")
        return
    f = a["fidelity"]["counts"]
    checked = f.get("match", 0) + f.get("mismatch", 0)
    put(a["n_sample"], "audit", "n"); put(f.get("match", 0), "audit", "match"); put(f.get("mismatch", 0), "audit", "mismatch")
    put(f.get("no_pdf_article", 0), "audit", "nopdf"); put(checked, "audit", "checked")
    put(f.get("mismatch", 0) / checked if checked else None, "audit", "rate", pct=True)
    put(a["base_collisions"]["cases"], "audit", "collisions"); put(len(a["base_collisions"]["sources"]), "audit", "collsources")
    cov = a["coverage"]
    put(sum(len(c["only_pdf"]) for c in cov.values()), "audit", "onlypdf")
    put(sum(len(c["only_index"]) for c in cov.values()), "audit", "onlyindex")
    rows = []
    for stem, c in cov.items():
        fs = a["fidelity"]["by_source"].get(stem, {})
        rows.append([stem.replace("_", r"\_"), str(c["pdf_labels"]), str(c["index_labels"]), str(len(c["only_pdf"])),
                     str(len(c["only_index"])), f"{c['pdf_sections']}/{c['index_sections']}",
                     str(sum(fs.values())), str(fs.get("mismatch", 0))])
    write_table("t_audit", ["Source", "Art. PDF", "Art. index", "PDF seul", "Index seul", "Sections", "Échant.", "Écarts"],
                rows, "Audit des \\emph{chunks} contre le texte des PDF (extraction pypdf indépendante). Étiquettes d'articles "
                "et sections (PDF/index) ; fidélité sur l'échantillon de 150 \\emph{chunks}.", "tab:audit")


def noise():
    n = load("noise_check.json")
    if not n or not n.get("questions"):
        for k in ("jaccard", "sameev", "questions", "runs"):
            put(None, "noise", k)
        placeholder_table("t_noise", "contrôle de stabilité")
        DATA["f9"] = None
        return
    qs = n["questions"]
    put(n.get("overall_mean_jaccard_cited"), "noise", "jaccard", nd=3)
    put(sum(bool(q.get("summary", {}).get("same_evidence_ids")) for q in qs), "noise", "sameev")
    put(len(qs), "noise", "questions"); put(n["runs_per_question"], "noise", "runs")
    rows, DATA["f9"] = [], []
    for q in qs:
        s = q.get("summary", {})
        rows.append([q["id"], q["language"], str(len(q["runs"])), "oui" if s.get("same_evidence_ids") else "non",
                     cell(s.get("mean_pairwise_jaccard_cited"), 3, False),
                     f"{s.get('retries_min')}--{s.get('retries_max')}"])
        DATA["f9"].append({"id": q["id"], "jaccard": s.get("mean_pairwise_jaccard_cited"),
                           "same": s.get("same_evidence_ids")})
    write_table("t_noise", ["Question (dev)", "Langue", "Exécutions", "6 preuves identiques", "Jaccard moyen (citées)",
                            "Reprises"], rows, "Stabilité : graphe complet exécuté plusieurs fois par question.", "tab:noise")
    # PDF numeric fact-check of the noise-check claims
    from pdf_factcheck import claim_status
    tot = Counter()
    short = {"no_numbers": "nonum", "implicit_reference_ok": "implicit"}
    for q in qs:
        for r in q["runs"]:
            for cl in r.get("claims") or []:
                status = claim_status(cl)[0]
                tot["claims"] += 1
                tot[short.get(status, status)] += 1
    for k in ("claims", "ok", "implicit", "missing", "unverifiable", "nonum"):
        put(tot[k], "factcheck", k)


def answer_eval():
    files = [p for p in sorted(glob.glob(str(RES / "answer_eval_answer_set_*.jsonl"))) if "_demo_" not in Path(p).name]
    DATA["final_eval"] = None
    if not files:
        for k in ("correct", "halluc", "hit", "gate"):
            for s in ("vone", "voneone", "vtwo"):
                put(None, "final", s, k)
        placeholder_table("t_final", "évaluation finale sur le jeu de 100 questions")
        return
    from run_answer_eval import metric_values      # same definitions as the harness
    rows, DATA["final_eval"] = [], {}
    for p in files:
        system = Path(p).stem.split("_")[-1]
        recs = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        vals = {k: [x for x in v if x is not None] for k, v in metric_values(recs).items()}
        mean = lambda k: (sum(vals[k]) / len(vals[k])) if vals.get(k) else None
        key = {"v1": "vone", "v1.1": "voneone", "v2": "vtwo"}.get(system, system)
        put(mean("correct"), "final", key, "correct", pct=True); put(mean("hallucination (answer)"), "final", key, "halluc", pct=True)
        put(mean("Hit@6 (gold in evidence)"), "final", key, "hit", pct=True)
        put(mean("gate accuracy (clarify/refuse items)"), "final", key, "gate", pct=True)
        DATA["final_eval"][system] = {k: mean(k) for k in vals}
        rows.append([system, cell(mean("correct")), cell(mean("hallucination (answer)")), cell(mean("Hit@6 (gold in evidence)")),
                     cell(mean("gate accuracy (clarify/refuse items)")), cell(mean("seconds"), 1, False), cell(mean("LLM calls"), 2, False)])
    write_table("t_final", ["Système", "Correct", "Hallucination", "Hit@6", "Portes", "Secondes", "Appels LLM"], rows,
                "Évaluation finale au niveau de la réponse (\\texttt{results/answer\\_eval\\_*}).", "tab:final")


def quotes_and_example():
    from pdf_truth import pdf_article_text
    from show_article import index_chunks
    # quote the index chunk (PyMuPDF) when it equals the pypdf text once all whitespace is removed: pypdf splits
    # some words ("ju illet"); otherwise quote pypdf and say so in a footnote
    for art in ("218", "219", "220"):
        t = pdf_article_text("PenalFr", art, 2)
        chunk = dict(index_chunks("PenalFr", art)).get(f"PenalFr/s02/{art}")
        same = bool(t and chunk) and re.sub(r"\s", "", chunk) == re.sub(r"\s", "", t)
        if not t:
            body, how = r"\textit{Texte du PDF non disponible.}", "no PDF text"
        elif same:
            body, how = r"\begin{quote}\small " + tex_escape(chunk) + r"\end{quote}", "index chunk (equal to pypdf without whitespace)"
        else:
            body = (r"\begin{quote}\small " + tex_escape(t) + r"\footnote{Citation tirée de l'extraction indépendante "
                    r"(pypdf) ; elle peut contenir des artefacts d'extraction.}\end{quote}")
            how = "pypdf text + footnote (index chunk " + ("absent" if not chunk else "differs beyond whitespace") + ")"
        (REP / "tables" / f"quote_penal_{art}.tex").write_text(body, encoding="utf-8")
        print(f"quote PenalFr {art}: {how}")
    demo = RES / "answer_eval_answer_set_demo_v2.jsonl"
    rec = None
    if demo.exists():
        rec = next((json.loads(l) for l in demo.read_text(encoding="utf-8").splitlines()
                    if l.strip() and json.loads(l)["id"] == "demo1"), None)
    if rec and rec.get("claims"):
        lines = [r"\begin{enumerate}\small"]
        for cl in rec["claims"]:
            lines.append(r"\item " + tex_escape(cl["text"]) + r" \hfill{\footnotesize[" +
                         ", ".join(tex_escape(c) for c in cl["cited_ids"]) + "]}")
        lines.append(r"\end{enumerate}")
        if rec.get("caveat"):
            lines.append(r"\noindent\textit{Réserve :} " + tex_escape(rec["caveat"]))
        (REP / "tables" / "example_claims.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        put(rec.get("retries"), "example", "retries"); put(len(rec["claims"]), "example", "claims")
        put(", ".join(r"\texttt{" + tex_escape(c) + "}" for c in rec.get("cited") or []), "example", "cited")
        put(rec.get("seconds"), "example", "seconds"); put(rec.get("llm_calls"), "example", "llmcalls")
    else:
        placeholder_table("example_claims", "exemple de réponse de bout en bout")
        for k in ("retries", "claims", "cited", "seconds", "llmcalls"):
            put(None, "example", k)


def corpus():
    import sqlite3
    from pdf_truth import load as pdf_load
    db = ROOT / "db_v1_1" / "chroma.sqlite3"
    chunks = Counter()
    if db.exists():
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        for (cid,) in con.execute("select string_value from embedding_metadata where key = 'chunk_id'"):
            chunks[cid.split("/")[0]] += 1
        con.close()
    try:
        from pypdf import PdfReader
        pages = {p.stem: len(PdfReader(p).pages) for p in sorted((ROOT / "data").glob("*.pdf"))}
    except Exception:
        pages = {}
    pdf = pdf_load()
    rows = []
    for stem in sorted(set(pages) | set(chunks), key=str.lower):
        rows.append([stem.replace("_", r"\_"), str(pages.get(stem, MISSING)), str(chunks.get(stem, 0)),
                     pdf.get(stem, {}).get("status", MISSING).replace("verified", "vérifiée").replace("unvérifiée", "non vérifiée")])
    put(len(pages) or None, "corpus", "pdfs"); put(sum(pages.values()) or None, "corpus", "pages")
    put(sum(chunks.values()) or None, "corpus", "chunks")
    put(sum(d["status"] == "verified" for d in pdf.values()), "corpus", "verified")
    write_table("t_corpus", ["Fichier source", "Pages", "\\emph{Chunks} (index)", "Extraction PDF"], rows,
                "Les quatorze PDF sources, leurs pages, le nombre de \\emph{chunks} de l'index et le statut de "
                "l'extraction indépendante (pypdf).", "tab:corpus")


def misc():
    n = load("noise_check.json")
    lines = (n or {}).get("temperature_lines") or []
    (REP / "tables" / "temperature.tex").write_text(
        ("\\begin{lstlisting}\n" + "\n".join(lines) + "\n\\end{lstlisting}\n") if lines else MISSING + "\n", encoding="utf-8")
    branches = git("branch", "--list", "rescue/*") or ""
    put(r"\texttt{" + branches.strip().replace("_", r"\_") + "}" if branches.strip() else None, "git", "rescue")
    top = (git("ls-tree", "--name-only", "HEAD") or "").split()
    (REP / "tables" / "repo_tree.tex").write_text(
        "\\begin{lstlisting}\n" + "\n".join(top) + "\n\\end{lstlisting}\n" if top else MISSING + "\n", encoding="utf-8")


def git_meta():
    put(r"\texttt{" + (git("rev-parse", "--short", "HEAD") or "?") + "}", "git", "head")
    put(r"\texttt{" + (git("rev-parse", "--abbrev-ref", "HEAD") or "?").replace("_", r"\_") + "}", "git", "branch")
    tags = (git("tag") or "").split()
    put(", ".join(r"\texttt{" + t.replace("_", r"\_") + "}" for t in tags) or None, "git", "tags")


def main():
    (REP / "tables").mkdir(parents=True, exist_ok=True)
    splits(); baseline(); steps(); pvr = plan_vs_raw(); loop(pvr); calibration(); audit(); noise(); answer_eval()
    quotes_and_example(); corpus(); misc(); git_meta()
    lines = ["% generated by report/make_numbers.py from results/, eval/ and git; do not edit by hand"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (REP / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REP / "data.json").write_text(json.dumps(DATA, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = sorted(k for k, v in MACROS.items() if v == MISSING)
    print(f"numbers.tex: {len(MACROS)} macros, {len(missing)} 'à compléter': {missing}")
    print("\nFacts stated in the task vs computed from files:")
    for label, claimed, computed in FACTS:
        print(f"  {label:<55} stated {claimed:<32} computed {computed}")


if __name__ == "__main__":
    main()
