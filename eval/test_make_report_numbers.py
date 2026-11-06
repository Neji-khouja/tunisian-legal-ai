"""Tests of eval/make_report_numbers.py against the saved outputs (no model, no network, result files only read).

    python eval/test_make_report_numbers.py

(a) every line of results/summary_exam_100.txt and results/noisy/summary_noisy.txt (per system and paired) is reproduced
(b) every line of results/noisy_vs_clean.txt with an interval (sections A, B, C) is reproduced
(c) results/audit_agreement.txt and the summary of results/latency_benchmark.json are reproduced
(d) formats; the missing latency file gives \\acompleter
(e) the claims the report's text makes about intervals still hold on the data
(f) the LaTeX: balanced braces and environments, every macro, \\input, figure and \\ref defined; no number typed by hand
    in the evaluation text except the listed exceptions
"""
import re
import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import make_report_numbers as mnr

RESULTS = []
REPORT = ROOT / "report"
EVAL_TEX = ["sections/09_evaluation_finale.tex", "sections/11_conclusion.tex"]
# digits allowed in the evaluation text, with the reason: article numbers of the addendum's known weaknesses (recel has no
# number); the names v1, v1.1, v2, Hit@6, étape 1, étape 2, the ids and the commands are removed before the scan
ALLOWED_DIGITS = {"370": "article cité dans l'addendum (période d'essai)", "429": "code de statut HTTP, un nom et non un résultat"}


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    if not cond:
        print(f"FAIL {name}: {detail}")
    else:
        print(f"ok   {name}")


def close(a, b, eps=0.00051) -> bool:
    return abs(a - b) <= eps


def same(st, saved, flip=False) -> bool:
    """A computed stat against a saved (n, mean, lo, hi). flip: the saved line is the same difference with the other sign."""
    n, mean, lo, hi = saved
    if flip:
        mean, lo, hi = -mean, -hi, -lo
    return st["n"] == n and close(st["mean"], mean) and close(st["lo"], lo) and close(st["hi"], hi)


def test_summaries(nb):
    for path, prefix, sets in ((mnr.RESULTS / "summary_exam_100.txt", "answer_eval_answer_set_stage{}_", {"1": "S1", "2": "S2"}),
                               (mnr.RESULTS / "noisy" / "summary_noisy.txt", "answer_eval_noisy_set_", {"": "N"})):
        blocks = mnr.parse_blocks(path.read_text(encoding="utf-8"))
        for tag, key in sets.items():
            stem = prefix.format(tag)
            bad, k = [], 0
            for s in mnr.SYSTEMS:
                saved = next(v for t, v in blocks.items() if t.startswith(f"{stem}{s}.jsonl"))
                for metric, line in saved.items():
                    if metric in mnr.BY_NAME:
                        k += 1
                        if not same(nb.stats[key, s, metric], line):
                            bad.append((s, metric))
            check(f"{key}: {k} per-system lines of {path.name} reproduced", k >= 39 and not bad, bad)
            bad, k = [], 0
            for b, a, flip, title in (("v2", "v1.1", False, f"{stem}v2 - {stem}v1.1"), ("v1.1", "v1", True, f"{stem}v1 - {stem}v1.1"), ("v2", "v1", False, f"{stem}v2 - {stem}v1")):
                saved = next(v for t, v in blocks.items() if t.startswith(f"paired difference {title} on"))
                for metric, line in saved.items():
                    if metric in mnr.BY_NAME:
                        k += 1
                        if not same(nb.diff[key, b, a, metric], line, flip):
                            bad.append((b, a, metric))
            check(f"{key}: {k} paired lines of {path.name} reproduced (v1.1 - v1 is the saved v1 - v1.1, sign reversed)", k > 30 and not bad, bad)


def test_noisy(nb):
    blocks = mnr.parse_blocks((mnr.RESULTS / "noisy_vs_clean.txt").read_text(encoding="utf-8"))
    bad, k = [], 0
    for s in mnr.SYSTEMS:
        for metric, line in next(v for t, v in blocks.items() if t.startswith(f"{s}: ")).items():
            k += 1
            if not same(nb.deg[s, metric], line):
                bad.append((s, metric))
    check(f"noisy_vs_clean A: {k} lines reproduced", k == 21 and not bad, bad)
    bad, k = [], 0
    for a, b in (("v2", "v1.1"), ("v2", "v1")):
        for metric, line in next(v for t, v in blocks.items() if t.startswith(f"{a} vs {b} ")).items():
            k += 1
            if not same(nb.did[a, b, metric], line):
                bad.append((a, b, metric))
    check(f"noisy_vs_clean B (difference in differences): {k} lines reproduced", k == 8 and not bad, bad)
    rows = {(r["field"], r["group"], r["system"], r["metric"]): r for r in nb.csv["noisy_exploratory"]}
    bad, k = [], 0
    for title, lines in blocks.items():
        m = re.match(r"(noise_type|language) (\w+), (v[\d.]+): ", title)
        for metric, line in (lines.items() if m else ()):
            k += 1
            if not same(rows[m[1], m[2], m[3], metric], line):
                bad.append((title, metric))
    check(f"noisy_vs_clean C (exploratory): {k} lines reproduced", k >= 140 and not bad, bad)


def test_audit_latency(nb):
    saved = mnr.parse_agreement((mnr.RESULTS / "audit_agreement.txt").read_text(encoding="utf-8"))
    rows = {r["measure"]: r for r in nb.csv["audit"]}
    bad = [m for m, x in saved.items() if not (x["n"] == rows[m]["n"] and close(x["agreement"], rows[m]["agreement"]) and close(x["lo"], rows[m]["lo"])
                                               and close(x["hi"], rows[m]["hi"]) and close(x["kappa"], rows[m]["kappa"]))]
    check(f"audit_agreement.txt: {len(saved)} measures reproduced", len(saved) == 3 and not bad, bad)
    text = (mnr.RESULTS / "audit_agreement.txt").read_text(encoding="utf-8")
    m = re.search(r"rows in the sheet: (\d+); left out as human_cannot_verify: (\d+); rows used: (\d+)", text)
    check("audit: rows, cannot verify, used", (m[1], m[2], m[3]) == (nb.macros["EvAuditRows"], nb.macros["EvAuditCannot"], nb.macros["EvAuditUsed"]))
    cells = re.findall(r"judge 1 / human 1: (\d+); judge 1 / human 0: (\d+); judge 0 / human 1: (\d+); judge 0 / human 0: (\d+)", text)
    check("audit: confusion counts", [tuple(c) for c in cells] == [tuple(str(rows[m][f"judge{a}_human{b}"]) for a, b in ((1, 1), (1, 0), (0, 1), (0, 0))) for m in saved])
    check("audit: every interval holds its kappa", all(r["kappa_lo"] <= r["kappa"] <= r["kappa_hi"] for r in rows.values()))
    doc = json.loads((mnr.RESULTS / "latency_benchmark.json").read_text(encoding="utf-8"))
    lat = {r["system"]: r for r in nb.csv["latency"]}
    bad = []
    for s, x in doc["summary"].items():
        for stem, d in (("wall", x["wall_seconds"]), ("excl", x["wall_excluding_retry_waits"])):
            if any(lat[s][f"{stem}_{f}"] != d[f] for f in ("median", "mean", "min", "max")):
                bad.append((s, stem))
        if not lat[s]["wall_lo"] <= x["wall_seconds"]["mean"] <= lat[s]["wall_hi"]:
            bad.append((s, "interval"))
    check("latency_benchmark.json: summary reproduced", not bad, bad)


def test_formats():
    check("decimal comma", mnr.number(66.666, 1) == "66{,}7")
    check("signs", (mnr.signed(6.7, 1), mnr.signed(-6.7, 1), mnr.signed(0.0, 1), mnr.signed(-0.0001, 1)) == ("+6{,}7", "$-$6{,}7", "0{,}0", "0{,}0"))
    check("rate shown in %", mnr.show(0.667, "rate") == "66{,}7" and mnr.show_signed(-0.125, "rate") == "$-$12{,}5")
    check("interval", mnr.ci_text(0.0, 0.325, "rate", True) == "[0{,}0~; +32{,}5]")
    check("tokens", mnr.tokens_text(10793.6) == "10\\,794")
    check("underscore in an id", mnr.tex_id("n_ac70") == r"n\_ac70")
    nb = mnr.Numbers()
    try:
        nb.m("Bad1", 1)
        ok = False
    except ValueError:
        ok = True
    check("a macro name with a digit is refused", ok)
    keep = mnr.RESULTS
    with tempfile.TemporaryDirectory() as d:
        mnr.RESULTS = Path(d)
        try:
            nb = mnr.Numbers()
            out = mnr.latency(nb)
        finally:
            mnr.RESULTS = keep
    check("no latency file: every latency macro is \\acompleter", out is None and nb.macros["EvLatVtwoWallMedian"] == mnr.MISSING and nb.warnings)


def test_claims(nb):
    """What the text says about intervals: if the data ever differ, this fails and the text must change."""
    d, ex = nb.diff, mnr.excludes_zero
    H, C, I, A = "Hit@6 (gold in evidence)", "correct", "incorrect", "hallucination (answer)"
    check("v2 - v1.1, correct: ahead in all three sets, no interval excludes zero",
          all(d[k, "v2", "v1.1", C]["mean"] > 0 and not ex(d[k, "v2", "v1.1", C]) for k in mnr.SETS))
    check("v2 - v1.1, stages 1 and 2: Hit@6, correct, incorrect, hallucination: no interval excludes zero",
          not any(ex(d[k, "v2", "v1.1", m]) for k in ("S1", "S2") for m in (H, C, I, A)))
    check("v2 - v1.1, noisy set: Hit@6, incorrect, hallucination exclude zero; correct does not",
          all(ex(d["N", "v2", "v1.1", m]) for m in (H, I, A)) and not ex(d["N", "v2", "v1.1", C]))
    check("v2 - v1.1, stage 1, incorrect: v2 is not ahead (positive mean)", d["S1", "v2", "v1.1", I]["mean"] > 0)
    check("v1.1 - v1, correct: excludes zero on stage 2 only; same direction on stage 1 and the noisy set",
          ex(d["S2", "v1.1", "v1", C]) and not ex(d["S1", "v1.1", "v1", C]) and not ex(d["N", "v1.1", "v1", C])
          and all(d[k, "v1.1", "v1", C]["mean"] > 0 for k in mnr.SETS))
    did = nb.did
    check("difference in differences: Hit@6 interval touches zero, others include it",
          did["v2", "v1.1", H]["lo"] == 0 and did["v2", "v1.1", H]["hi"] > 0 and not any(ex(did["v2", "v1.1", m]) for m in (C, I, A)))
    rows = {(r["group"], r["system"], r["metric"]): r for r in nb.csv["noisy_exploratory"]}
    check("describing an offence: correct falls for every version", all(rows["describe", s, C]["mean"] < 0 for s in mnr.SYSTEMS))
    check("no substantive answer: 100% for all three systems in both stages",
          all(nb.stats[k, s, "refuse: no substantive answer"]["mean"] == 1 for k in ("S1", "S2") for s in mnr.SYSTEMS))
    check("explicit refusal and clarification status: 0% for v1 and v1.1",
          all(nb.stats[k, s, m]["mean"] == 0 for k in ("S1", "S2") for s in ("v1", "v1.1") for m in ("refuse: explicit refusal status", "clarify w/o reply: asked for clarification (status)")))
    check("LLM calls, v2 - v1.1: about 4 more in every set", all(3.5 <= d[k, "v2", "v1.1", "LLM calls"]["mean"] <= 5 for k in mnr.SETS))
    s = nb.sensitivity
    check("sensitivity: no interval changes side of zero, no sign flip, one touching interval crosses",
          not s["crossed"] and not s["flipped"] and [(r["pair"], r["metric"]) for r in s["touch"]] == [("v2 - v1.1", A)])
    rows = nb.pre_rows
    check("every threshold of the addendum is met", all(met for src, _, _, met in rows if src == "addendum"))
    draft = {c.split(" :")[0] + c[-8:]: met for src, c, _, met in rows if src != "addendum"}
    check("draft thresholds on correct and hallucination are not met", not any(met for src, c, _, met in rows if src != "addendum" and c.startswith(("H1", "H2"))), draft)
    check("draft refusal threshold is not met, clarification ones are", not next(met for _, c, _, met in rows if c.startswith("H4"))
          and all(met for _, c, _, met in rows if c.startswith("H3")))
    check("all rows carry the addendum's fingerprint", nb.macros["EvRowFingerprintMatches"] == "oui" and nb.macros["EvRowFingerprintCount"] == "1")
    check("backup-key code is still in HEAD (the text says so)", nb.extra["keys_at_head"])
    reruns = {(r["set"], r["system"], r["id"]) for r in nb.csv["reruns"]}
    check("rows rerun by the harness", reruns == {("S1", "v1", "ac24"), ("S1", "v1", "ac41"), ("S1", "v1", "c01"), ("S1", "v2", "ac11"), ("S1", "v2", "ac33"), ("S2", "v2", "ac20"), ("N", "v2", "n_ac70")}, reruns)
    check("no row left unscored", all(nb.macros[f"Ev{mnr.SETS[k][0]}{mnr.SYS_MACRO[x]}NotScored"] == "0" for k in mnr.SETS for x in mnr.SYSTEMS))
    check("every set used more tokens in a day than the daily limit", all(float(nb.macros[f"EvTokensDayRatio{mnr.SETS[k][0]}"].replace("{,}", ".")) > 1 for k in mnr.SETS))


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def all_tex() -> list[Path]:
    return [REPORT / "main.tex", *sorted((REPORT / "sections").glob("*.tex")), *sorted((REPORT / "generated").glob("*.tex")), *sorted((REPORT / "tables").glob("*.tex")), REPORT / "numbers.tex"]


def test_latex():
    defined = set()
    for f in (REPORT / "numbers.tex", REPORT / "generated" / "numbers.tex", REPORT / "main.tex"):
        defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read_text(encoding="utf-8")))
    bad_braces, bad_env, undefined, bad_inputs, bad_figs, used_labels, labels = [], [], set(), [], [], set(), set()
    for f in all_tex():
        raw = f.read_text(encoding="utf-8")
        text = strip_comments(raw)
        body = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text, flags=re.S)
        clean = body.replace("\\\\", "").replace(r"\{", "").replace(r"\}", "")
        if clean.count("{") != clean.count("}"):
            bad_braces.append(f"{f.name}: {clean.count('{')} / {clean.count('}')}")
        stack = []
        for kind, env in re.findall(r"\\(begin|end)\{([a-zA-Z*]+)\}", text):
            if kind == "begin":
                stack.append(env)
            elif not stack or stack.pop() != env:
                bad_env.append(f"{f.name}: {env}")
        if stack:
            bad_env.append(f"{f.name}: open {stack}")
        undefined |= {m for m in re.findall(r"\\((?:Ev|N)[A-Z][A-Za-z]*)", body) if m not in defined and not m.startswith("Newcommand")}
        for target in re.findall(r"\\input\{([^}]+)\}", body):
            p = REPORT / target
            if not (p.exists() or p.with_suffix(".tex").exists()):
                bad_inputs.append(target)
        for name in re.findall(r"\\figfile\{([^}#]+)\}", body):
            if not (REPORT / "figures" / f"{name}.pdf").exists():
                bad_figs.append(name)
        used_labels |= set(re.findall(r"\\ref\{([^}#]+)\}", body))
        labels |= set(re.findall(r"\\label\{([^}#]+)\}", body))
        labels |= set(re.findall(r"\\figfile\{[^}]+\}\{.*?\}\{(fig:[^}]+)\}", body, re.S))
    check("balanced braces in every .tex", not bad_braces, bad_braces)
    check("balanced environments in every .tex", not bad_env, bad_env)
    check("every \\Ev... and \\N... macro is defined", not undefined, sorted(undefined)[:30])
    check("every \\input exists", not bad_inputs, bad_inputs)
    check("every \\figfile exists as a PDF", not bad_figs, bad_figs)
    check("every \\ref has a \\label", used_labels <= labels, sorted(used_labels - labels))
    left = literal_digits()
    check("no number typed by hand in the evaluation text", not left, left[:30])


def unescaped_underscores() -> list[str]:
    out = []
    for f in [*(REPORT / n for n in EVAL_TEX), *sorted((REPORT / "generated").glob("*.tex"))]:
        text = strip_comments(f.read_text(encoding="utf-8"))
        text = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text, flags=re.S)
        text = re.sub(r"\\(input|ref|label|figfile)\{[^{}]*\}", "", text)
        text = re.sub(r"\\newcommand\{[^{}]*\}", "", text)
        text = re.sub(r"\}\{fig:[^{}]*\}", "", text)
        out += [f"{f.name}: {m[:40]}" for m in re.findall(r"[^\n]{0,20}(?<!\\)_[^\n]{0,20}", text)]
    return out


def literal_digits() -> list[str]:
    """Digits left in the evaluation text after removing macros, names, ids, code and references."""
    out = []
    for name in EVAL_TEX:
        text = strip_comments((REPORT / name).read_text(encoding="utf-8"))
        text = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text, flags=re.S)
        text = re.sub(r"\\(texttt|ref|label|input|figfile\{[^}]*\})\{[^{}]*\}", "", text)
        text = re.sub(r"\\figfile\{[^}]*\}", "", text)
        text = re.sub(r"\\[A-Za-z]+", "", text)
        text = re.sub(r"\bv1\.1\b|\bv1\b|\bv2\b|Hit@6|[Éé]tapes?[ ~]1( et[ ~]2)?\b|[Éé]tapes?[ ~]2\b|\bH[1-4]\b|\bA[12]\b", "", text)
        text = re.sub(r"\bn?_?ac\d+\b|\bc\d+\b", "", text)
        for lineno, line in enumerate(text.splitlines(), 1):
            for num in re.findall(r"\d[\d.,]*", line):
                if num.rstrip(".,") not in ALLOWED_DIGITS:
                    out.append(f"{name}:{lineno}: {num} in: {line.strip()[:90]}")
    return out


def main():
    nb, _ = mnr.compute()
    test_summaries(nb)
    test_noisy(nb)
    test_audit_latency(nb)
    test_formats()
    test_claims(nb)
    test_latex()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
