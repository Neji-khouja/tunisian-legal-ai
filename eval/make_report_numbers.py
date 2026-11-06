"""Numbers of the evaluation part of the report, from the result files. No LLM call, no model; result files are only read.

    python eval/make_report_numbers.py        # writes report/generated/numbers.tex, *.csv and the t_*.tex tables

Rows are read and scored with the harness's own functions (run_answer_eval.read_jsonl, metric_values, boot_ci;
noisy_vs_clean.pairs, diffs, did; audit_agreement.read_rows, label, kappa): same definitions and same bootstrap as
--summarize and eval/noisy_vs_clean.py (95% percentile interval, 5000 resamples over questions, seed 20261022).
eval/test_make_report_numbers.py compares the output with the saved .txt files.

Stages stay separate. The only pooled values are in pooled_descriptive.csv (labelled descriptive, used in no claim) and
the counts of the pre-registered thresholds written on all the items (10 unscripted ambiguous, 70 answerable, 15 out of
scope), which pre_registered.csv labels as such. Agreement kappa intervals and latency intervals are new here (the saved
outputs have none); they use the same bootstrap. An optional input that is missing gives \\acompleter, never a guess.
"""
import re
import csv
import sys
import json
import random
import statistics
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import harness_core as hc
import run_answer_eval as rae
import noisy_vs_clean as nvc
import audit_agreement as aa

RESULTS = ROOT / "results"
OUT = ROOT / "report" / "generated"
SYSTEMS = ["v1", "v1.1", "v2"]
SYS_MACRO = {"v1": "Vone", "v1.1": "Voneone", "v2": "Vtwo"}
SETS = {"S1": ("Sone", "étape 1"), "S2": ("Stwo", "étape 2"), "N": ("Snoisy", "jeu bruité")}
COMPARISONS = [("v2", "v1.1"), ("v1.1", "v1"), ("v2", "v1")]
TOKENS = "tokens per question"
# harness metric name, macro name, French label, kind (rate: shown in %; else: (decimals, unit))
METRICS = [("Hit@6 (gold in evidence)", "Hit", "Hit@6", "rate"), ("correct", "Correct", "Correct", "rate"),
           ("partial", "Partial", "Partiel", "rate"), ("incorrect", "Incorrect", "Incorrect", "rate"),
           ("hallucination (answer)", "Halluc", "Hallucination (réponse)", "rate"),
           ("numeric check ok (answer)", "Numeric", "Contrôle numérique réussi", "rate"),
           ("false refuse", "FalseRefuse", "Faux refus", "rate"), ("false clarify", "FalseClarify", "Fausse clarification", "rate"),
           ("LLM calls", "Calls", "Appels LLM", (2, "")), ("seconds", "Seconds", "Secondes", (1, "s")),
           (TOKENS, "Tokens", "Tokens", (0, "")), ("429 responses", "RateLimit", "Réponses 429", (2, "")),
           ("client retry waits (s)", "Waits", "Attentes annoncées (s)", (1, "s")),
           ("clarify w/o reply: asked for clarification (status)", "ClarAsked", "Clarification demandée", "rate"),
           ("clarify w/o reply: answered directly", "ClarDirect", "Réponse directe", "rate"),
           ("clarify w/o reply: enumerated alternatives", "ClarEnum", "Énumération des lectures", "rate"),
           ("clarify then answer: Hit@6 after the reply", "ClarHit", "Hit@6 après la réponse scriptée", "rate"),
           ("refuse: explicit refusal status", "RefExplicit", "Refus explicite", "rate"),
           ("refuse: no substantive answer", "RefNoSubst", "Pas de réponse de fond", "rate"),
           ("gate accuracy (explicit status)", "GateAcc", "Décision correcte (statut explicite)", "rate"),
           ("claims flagged on 1st pass", "Flagged", "Affirmations signalées au premier passage", "rate"),
           ("repaired (repairs == 1)", "Repaired", "Réponses réparées", "rate"),
           ("dropped claims (mean)", "Dropped", "Affirmations retirées (moyenne)", (2, "")),
           ("no supported claim left", "NoClaim", "Plus aucune affirmation appuyée", "rate"),
           ("numeric miss, draft claims (answer)", "MissDraft", "Nombre absent, brouillon", "rate"),
           ("numeric miss, final claims (answer)", "MissFinal", "Nombre absent, réponse finale", "rate")]
BY_NAME = {m[0]: m for m in METRICS}
RESULT_METRICS = ["Hit@6 (gold in evidence)", "correct", "partial", "incorrect", "hallucination (answer)",
                  "numeric check ok (answer)", "false refuse", "LLM calls", "seconds"]
GROUPS = {"noise_type": {"typos": "Typos", "wrong_term": "WrongTerm", "describe": "Describe", "informal": "Informal"},
          "language": {"fr": "Fr", "ar": "Ar", "en": "En"}}
GROUP_FR = {"typos": "fautes de frappe", "wrong_term": "terme erroné", "describe": "description sans nommer",
            "informal": "registre familier", "fr": "français", "ar": "arabe", "en": "anglais"}
COMMITS = {"Addendum": "0b83d32", "StageOneDone": "1ce24a6", "FallbackFlags": "e76f31e", "Sensitivity": "32a822f",
           "NoisyDraft": "18a63dc", "BackupKeys": "310fd11", "JudgeRetry": "78ade16", "StageTwoDone": "b4e49e9",
           "LockFile": "4889400", "NoisyLock": "09365f8", "NoisyDone": "e34e412", "NoisyAnalysis": "8c9e792",
           "AuditExport": "1f17a93", "AuditDone": "5048622", "LatencyScript": "902b348", "LatencyDone": "b3f4c6d"}
MISSING = r"\acompleter{}"


# ---------------------------------------------------------------- formats

def number(x: float, nd: int) -> str:
    return f"{x:.{nd}f}".replace(".", "{,}")


def is_zero(x: float, nd: int) -> bool:
    return float(f"{abs(x):.{nd}f}") == 0


def signed(x: float, nd: int) -> str:
    if is_zero(x, nd):
        return number(0.0, nd)
    return ("+" if x > 0 else "$-$") + number(abs(x), nd)


def tokens_text(x: float) -> str:
    return f"{x:,.0f}".replace(",", "\\,")


def tex_id(s: str) -> str:
    return str(s).replace("_", r"\_")


def breakable(h: str, every: int = 8) -> str:
    """A long hash that can break across lines."""
    return r"llowbreak ".join(h[i:i + every] for i in range(0, len(h), every))


def scale(kind) -> tuple[float, int]:
    return (100.0, 1) if kind == "rate" else (1.0, kind[0])


def show(x, kind) -> str:
    if x is None:
        return "n.d."
    f, nd = scale(kind)
    return tokens_text(x) if nd == 0 else number(x * f, nd)


def show_signed(x, kind) -> str:
    if x is None:
        return "n.d."
    f, nd = scale(kind)
    return signed(x * f, nd)


def ci_text(lo, hi, kind, sign=False) -> str:
    if lo is None:
        return "n.d."
    f = show_signed if sign else show
    return f"[{f(lo, kind)}~; {f(hi, kind)}]"


def stat(xs: list) -> dict:
    """mean and interval, as print_block and noisy_vs_clean.line compute them."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    lo, hi = rae.boot_ci(xs, random.Random(rae.SEED))
    return {"n": len(xs), "mean": sum(xs) / len(xs), "lo": lo, "hi": hi}


def excludes_zero(s: dict) -> bool:
    return s["lo"] is not None and (s["lo"] > 0 or s["hi"] < 0)


# ---------------------------------------------------------------- reading

def set_path(key: str, system: str) -> Path:
    if key == "N":
        return RESULTS / "noisy" / f"answer_eval_noisy_set_{system}.jsonl"
    return RESULTS / f"answer_eval_answer_set_stage{key[1]}_{system}.jsonl"


def latest_rows(key: str, system: str) -> dict:
    return {r["id"]: r for r in rae.read_jsonl(set_path(key, system))}


def values(rows: list) -> dict:
    """metric_values, plus the tokens of the system per answerable question (same rows: not degraded, error or failed)."""
    vals = rae.metric_values(rows)
    vals[TOKENS] = [(r.get("tokens") or {}).get("total") for r in rows
                    if r["expected"] == "answer" and r.get("status") not in rae.NO_ANSWER]
    return vals


def paired(A: dict, B: dict, name: str) -> list:
    """B - A per question, as summarize does it."""
    ids = sorted(i for i in set(A) & set(B) if not {A[i].get("status"), B[i].get("status")} & set(rae.NO_ANSWER))
    va, vb = values([A[i] for i in ids]), values([B[i] for i in ids])
    return [y - x for x, y in zip(va[name], vb.get(name, [])) if x is not None and y is not None]


# ---------------------------------------------------------------- parsing the saved outputs

LINE = re.compile(r"^  (?P<name>\S.*?)\s+n=(?P<n>\d+)\s+(?P<mean>[-+]?[\d.]+)\s+\[(?P<lo>[-+]?[\d.]+), (?P<hi>[-+]?[\d.]+)\]\s*$")


def parse_blocks(text: str) -> dict:
    """Saved --summarize or noisy_vs_clean text -> {block title: {metric: (n, mean, lo, hi)}}."""
    out, block = {}, None
    for line in text.splitlines():
        if line.startswith(("=== ", "--- ")):
            block = line.strip("=- ").strip()
            out[block] = {}
        m = LINE.match(line)
        if m and block is not None:
            out[block][m["name"]] = (int(m["n"]), float(m["mean"]), float(m["lo"]), float(m["hi"]))
    return out


AGREE = re.compile(r"^(?P<name>[a-z].*?): n=(?P<n>\d+)  agreement (?P<a>[\d.]+)%  95% CI \[(?P<lo>[\d.]+)%, (?P<hi>[\d.]+)%\]"
                   r"  Cohen's kappa (?P<k>[\d.]+)")


def parse_agreement(text: str) -> dict:
    return {m["name"]: {"n": int(m["n"]), "agreement": float(m["a"]) / 100, "lo": float(m["lo"]) / 100,
                        "hi": float(m["hi"]) / 100, "kappa": float(m["k"])}
            for m in map(AGREE.match, text.splitlines()) if m}


# ---------------------------------------------------------------- computations

class Numbers:
    def __init__(self):
        self.macros, self.csv = {}, defaultdict(list)
        self.stats, self.diff, self.deg, self.did = {}, {}, {}, {}
        self.warnings = []

    def m(self, name: str, value):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"not a LaTeX macro name: {name}")
        if name in self.macros and self.macros[name] != str(value):
            raise ValueError(f"macro defined twice with two values: {name}")
        self.macros[name] = str(value)

    def triple(self, stem: str, s: dict, kind, sign=False, count=False):
        self.m(stem, (show_signed if sign else show)(s["mean"], kind))
        self.m(stem + "CI", ci_text(s["lo"], s["hi"], kind, sign))
        self.m(stem + "N", s["n"])
        if count and s["n"]:
            self.m(stem + "Count", f"{round(s['mean'] * s['n'])}/{s['n']}")


def per_system(nb: Numbers, data: dict):
    for key, (smac, label) in SETS.items():
        for s in SYSTEMS:
            vals = values(list(data[key][s].values()))
            for name, mac, fr, kind in METRICS:
                st = stat(vals.get(name, []))
                nb.stats[key, s, name] = st
                nb.triple(f"Ev{smac}{SYS_MACRO[s]}{mac}", st, kind, count=kind == "rate")
                nb.csv["per_system"].append({"set": key, "set_label": label, "system": s, "metric": name, "label": fr, **st})


def differences(nb: Numbers, data: dict):
    for key, (smac, label) in SETS.items():
        for b, a in COMPARISONS:
            for name, mac, fr, kind in METRICS:
                st = stat(paired(data[key][a], data[key][b], name))
                nb.diff[key, b, a, name] = st
                nb.triple(f"EvDiff{smac}{SYS_MACRO[b]}{SYS_MACRO[a]}{mac}", st, kind, sign=True)
                nb.csv["paired"].append({"set": key, "set_label": label, "a": b, "b": a, "metric": name, "label": fr, **st})


def noisy(nb: Numbers):
    mapping = json.loads((ROOT / "eval" / "noisy_set.json").read_text(encoding="utf-8"))
    P = {}
    for s in SYSTEMS:
        clean, _ = nvc.latest(RESULTS / f"answer_eval_answer_set_stage2_{s}.jsonl")
        z, _ = nvc.latest(RESULTS / "noisy" / f"answer_eval_noisy_set_{s}.jsonl")
        P[s], excluded = nvc.pairs(clean, z, mapping)
        nb.m(f"EvNoisyPairs{SYS_MACRO[s]}", len(P[s]))
        nb.m(f"EvNoisyExcluded{SYS_MACRO[s]}", len(excluded))
        for name in nvc.METRICS:
            _, mac, fr, kind = BY_NAME[name]
            st = stat(nvc.diffs(P[s], name))
            nb.deg[s, name] = st
            nb.triple(f"EvDeg{SYS_MACRO[s]}{mac}", st, kind, sign=True)
            nb.csv["noisy_degradation"].append({"system": s, "metric": name, "label": fr, **st})
    for a, b, tag in (("v2", "v1.1", "primary"), ("v2", "v1", "secondary")):
        for name in nvc.DID_METRICS:
            _, mac, fr, kind = BY_NAME[name]
            st = stat(nvc.did(P[a], P[b], name))
            nb.did[a, b, name] = st
            nb.triple(f"EvDid{SYS_MACRO[a]}{SYS_MACRO[b]}{mac}", st, kind, sign=True)
            nb.csv["noisy_did"].append({"a": a, "b": b, "role": tag, "metric": name, "label": fr, **st})
    nb.m("EvMinNci", nvc.MIN_N_CI)
    for field, groups in GROUPS.items():
        for g, gmac in groups.items():
            ids = {n["original_id"] for n in mapping if n[field] == g}
            nb.m(f"EvGroup{gmac}N", len(ids))
            for s in SYSTEMS:
                for name in nvc.METRICS:
                    _, mac, fr, kind = BY_NAME[name]
                    xs = nvc.diffs(P[s], name, ids)
                    st = stat(xs) if len(xs) >= nvc.MIN_N_CI else {"n": len(xs), "mean": sum(xs) / len(xs) if xs else None, "lo": None, "hi": None}
                    nb.triple(f"EvDeg{gmac}{SYS_MACRO[s]}{mac}", st, kind, sign=True)
                    nb.csv["noisy_exploratory"].append({"field": field, "group": g, "system": s, "metric": name, "label": fr, **st})
    nb.m("EvNoisyN", len(mapping))
    return mapping


def audit(nb: Numbers):
    sheet = aa.read_rows(RESULTS / "audit_sheet.csv")
    key = aa.read_rows(RESULTS / "audit_key.csv")
    cv = [i for i, r in sheet.items() if aa.label(r.get("human_cannot_verify")) == 1]
    used = {i: r for i, r in sheet.items() if i not in cv}
    nb.m("EvAuditRows", len(sheet))
    nb.m("EvAuditCannot", len(cv))
    nb.m("EvAuditUsed", len(used))
    nb.m("EvAuditThreshold", number(aa.THRESHOLD * 100, 0))
    cv_system = Counter(key[i]["system"].strip() for i in cv)
    cv_lang = Counter(sheet[i]["language"].strip() for i in cv)
    for s in SYSTEMS:
        nb.m(f"EvAuditPerSystem{SYS_MACRO[s]}", sum(key[i]["system"].strip() == s for i in key))
        nb.m(f"EvAuditCannot{SYS_MACRO[s]}", cv_system[s])
    for lang, mac in GROUPS["language"].items():
        nb.m(f"EvAuditCannot{mac}", cv_lang[lang])
    macs = {"correct": "Correct", "hallucination (answer)": "Halluc", "hallucination (LLM judge only)": "Judge"}
    results = {}
    for name, jcol, what in aa.MEASURES:
        pairs = [(int(key[i][jcol]), aa.label(r.get(f"human_{what}"))) for i, r in used.items()
                 if aa.label(r.get(f"human_{what}")) is not None]
        match = [int(j == h) for j, h in pairs]
        st = stat(match)
        rng, ks = random.Random(rae.SEED), []
        for _ in range(rae.N_BOOT):
            k = aa.kappa([pairs[rng.randrange(len(pairs))] for _ in pairs])
            if k is not None:
                ks.append(k)
        ks.sort()
        k_lo, k_hi = ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks)) - 1]
        k = aa.kappa(pairs)
        cells = {f"judge{a}_human{b}": sum(p == (a, b) for p in pairs) for a in (1, 0) for b in (1, 0)}
        results[name] = {"n": st["n"], "agreement": st["mean"], "lo": st["lo"], "hi": st["hi"], "kappa": k, "kappa_lo": k_lo,
                         "kappa_hi": k_hi, **cells, "threshold": aa.THRESHOLD, "met": st["mean"] >= aa.THRESHOLD}
        nb.csv["audit"].append({"measure": name, **results[name]})
        mac = macs[name]
        nb.m(f"EvAudit{mac}Agree", number(st["mean"] * 100, 1))
        nb.m(f"EvAudit{mac}CI", ci_text(st["lo"], st["hi"], "rate"))
        nb.m(f"EvAudit{mac}N", st["n"])
        nb.m(f"EvAudit{mac}Kappa", number(k, 3))
        nb.m(f"EvAudit{mac}KappaCI", f"[{signed(k_lo, 3)}~; {signed(k_hi, 3)}]")
        nb.m(f"EvAudit{mac}Met", "atteint" if st["mean"] >= aa.THRESHOLD else "non atteint")
        nb.m(f"EvAudit{mac}Disagree", len(match) - sum(match))
        by = Counter(key[i]["system"].strip() for i, r in used.items() if aa.label(r.get(f"human_{what}")) is not None
                     and int(key[i][jcol]) != aa.label(r.get(f"human_{what}")))
        for s in SYSTEMS:
            nb.m(f"EvAudit{mac}Disagree{SYS_MACRO[s]}", by[s])
    ai = RESULTS / "audit_claude_agreement.txt"
    if ai.exists():
        parsed = parse_agreement(ai.read_text(encoding="utf-8"))
        for name, mac in macs.items():
            nb.m(f"EvAuditAi{mac}Agree", number(parsed[name]["agreement"] * 100, 1))
            nb.m(f"EvAuditAi{mac}Kappa", number(parsed[name]["kappa"], 3))
        nb.m("EvAuditAiN", parsed["correct"]["n"])
    else:
        nb.warnings.append("results/audit_claude_agreement.txt is missing")
        for name, mac in macs.items():
            nb.m(f"EvAuditAi{mac}Agree", MISSING)
            nb.m(f"EvAuditAi{mac}Kappa", MISSING)
        nb.m("EvAuditAiN", MISSING)
    return results


def latency(nb: Numbers):
    path = RESULTS / "latency_benchmark.json"
    if not path.exists():
        nb.warnings.append("results/latency_benchmark.json is missing: latency parts are \\acompleter")
        nb.m("EvLatAvailable", 0)
        for name in ("Questions", "Ids", "WarmupId", "Pause", "Fr", "Ar", "En", "RamTotal", "Cpus", "Python", "Os", "TokensTotal"):
            nb.m(f"EvLat{name}", MISSING)
        for mac in SYS_MACRO.values():
            for name in ("Finished", "Planned", "Calls", "Tokens", "RateLimit", "MeanWait", "RamAvail", "WallCI", "ExclCI",
                         *(f"{stem}{f}" for stem in ("Wall", "Excl") for f in ("Median", "Mean", "Min", "Max"))):
                nb.m(f"EvLat{mac}{name}", MISSING)
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    nb.m("EvLatQuestions", len(doc["selection"]["question_ids"]))
    nb.m("EvLatIds", ", ".join(doc["selection"]["question_ids"]))
    nb.m("EvLatWarmupId", doc["selection"]["warmup_id"])
    nb.m("EvLatPause", doc["pause_seconds"])
    by_lang = doc["selection"]["questions_by_language"]
    for lang, mac in GROUPS["language"].items():
        nb.m(f"EvLat{mac}", by_lang[lang])
    mach = doc["machine"]
    nb.m("EvLatRamTotal", number(mach["ram_total_gb"], 1))
    nb.m("EvLatCpus", mach["cpu_count"])
    nb.m("EvLatPython", mach["python"])
    nb.m("EvLatOs", tex_id(mach["os"]))
    total_tokens = 0
    for s in SYSTEMS:
        sd, x, mac = doc["systems"][s], doc["summary"][s], SYS_MACRO[s]
        rows = [r for r in sd["questions"].values() if r["status"] not in ("degraded", "error")]
        wall, excl = stat([r["wall_seconds"] for r in rows]), stat([r["wall_excluding_retry_waits"] for r in rows])
        spent = sum(r["tokens_total"] for r in sd["questions"].values()) + sum((run["warmup"] or {}).get("tokens_total", 0) for run in sd["runs"])
        total_tokens += spent
        for stem, d, boot in (("Wall", x["wall_seconds"], wall), ("Excl", x["wall_excluding_retry_waits"], excl)):
            for f in ("median", "mean", "min", "max"):
                nb.m(f"EvLat{mac}{stem}{f.capitalize()}", number(d[f], 1))
            nb.m(f"EvLat{mac}{stem}CI", ci_text(boot["lo"], boot["hi"], (1, "s")))
        nb.m(f"EvLat{mac}Finished", x["finished"])
        nb.m(f"EvLat{mac}Planned", x["planned"])
        nb.m(f"EvLat{mac}Calls", number(x["mean_llm_calls"], 2))
        nb.m(f"EvLat{mac}Tokens", tokens_text(x["mean_tokens"]))
        nb.m(f"EvLat{mac}RateLimit", number(x["mean_429"], 2))
        nb.m(f"EvLat{mac}MeanWait", number(x["wall_seconds"]["mean"] - x["wall_excluding_retry_waits"]["mean"], 1))
        nb.m(f"EvLat{mac}RamAvail", number(sd["runs"][-1]["ram_available_gb"], 1))
        nb.csv["latency"].append({"system": s, "n": x["finished"], "wall_median": x["wall_seconds"]["median"],
                                  "wall_mean": x["wall_seconds"]["mean"], "wall_min": x["wall_seconds"]["min"],
                                  "wall_max": x["wall_seconds"]["max"], "wall_lo": wall["lo"], "wall_hi": wall["hi"],
                                  "excl_median": x["wall_excluding_retry_waits"]["median"],
                                  "excl_mean": x["wall_excluding_retry_waits"]["mean"],
                                  "excl_min": x["wall_excluding_retry_waits"]["min"],
                                  "excl_max": x["wall_excluding_retry_waits"]["max"], "excl_lo": excl["lo"], "excl_hi": excl["hi"],
                                  "mean_llm_calls": x["mean_llm_calls"], "mean_tokens": x["mean_tokens"], "mean_429": x["mean_429"],
                                  "ram_available_gb": sd["runs"][-1]["ram_available_gb"], "tokens_spent_with_warmup": spent})
    nb.m("EvLatTokensTotal", tokens_text(total_tokens))
    nb.m("EvLatAvailable", 1)
    return doc


def exam_facts(nb: Numbers):
    items = json.loads((ROOT / "eval" / "answer_set.json").read_text(encoding="utf-8"))
    stages = json.loads((ROOT / "eval" / "stages.json").read_text(encoding="utf-8"))
    nb.m("EvExamN", len(items))
    for k, mac in (("answer", "Answer"), ("clarify", "Clarify"), ("refuse", "Refuse")):
        nb.m(f"EvExam{mac}", sum(i["expected"] == k for i in items))
    for lang, mac in GROUPS["language"].items():
        nb.m(f"EvExam{mac}", sum(i["language"] == lang for i in items))
    for st, smac in (("stage1", "Sone"), ("stage2", "Stwo")):
        ids = set(stages[st])
        its = [i for i in items if i["id"] in ids]
        nb.m(f"Ev{smac}Items", len(its))
        for k, mac in (("answer", "Answer"), ("clarify", "Clarify"), ("refuse", "Refuse")):
            nb.m(f"Ev{smac}{mac}", sum(i["expected"] == k for i in its))
        nb.m(f"Ev{smac}Replies", sum(bool(i.get("reply")) for i in its))
        for lang, mac in GROUPS["language"].items():
            nb.m(f"Ev{smac}{mac}", sum(i["language"] == lang and i["expected"] == "answer" for i in its))
    nb.m("EvBoot", rae.N_BOOT)
    nb.m("EvSeed", rae.SEED)
    nb.m("EvLevel", 95)
    nb.m("EvTpd", tokens_text(rae.TPD))
    return items


def addendum(nb: Numbers) -> dict:
    """Numbers written in agent/LEARNING.md: the addendum, the earlier [edit] draft, the withdrawn claims."""
    text = (ROOT / "agent" / "LEARNING.md").read_text(encoding="utf-8")
    add = text[text.index("## PRE-REGISTRATION ADDENDUM"):]
    def find(pattern, source=add):
        m = re.search(pattern, source, re.S)
        if not m:
            raise ValueError(f"not found in agent/LEARNING.md: {pattern}")
        return m
    fp = find(r"count\):\s*([0-9a-f]{64}) \(index (\S+), (\d+) chunks\)")
    nb.m("EvPreDate", find(r"written (\d{4}-\d{2}-\d{2})")[1])
    nb.m("EvPreFingerprint", breakable(fp[1]))
    nb.m("EvPreFingerprintShort", fp[1][:12])
    nb.m("EvPreIndex", tex_id(fp[2]))
    nb.m("EvPreChunks", fp[3])
    exam = find(r"answer_set\.json sha256 ([0-9a-f]{64})")[1]
    stages = find(r"stages\.json sha256 ([0-9a-f]{64})")[1]
    nb.m("EvPreExamHash", breakable(exam))
    nb.m("EvPreStagesHash", breakable(stages))
    judge = find(r"Judge: (\S+), thinking (\w+)")
    nb.m("EvPreJudge", tex_id(judge[1]))
    nb.m("EvPreJudgeThinking", {"minimal": "minimal"}.get(judge[2], judge[2]))
    rule = find(r"\(>=(\d)/(\d) successes, median latency <(\d+) s")
    nb.m("EvJudgeRuleOk", rule[1]); nb.m("EvJudgeRuleN", rule[2]); nb.m("EvJudgeRuleSeconds", rule[3])
    t = {"audit": int(find(r"agreement >= (\d+)%")[1]), "clarify": int(find(r"clarification on >= (\d+)% of the (\d+)")[1]),
         "clarify_n": int(find(r"clarification on >= (\d+)% of the (\d+)")[2]), "over": int(find(r"over-clarification .{0,40}?<= (\d+)%")[1]),
         "refuse": int(find(r">= (\d+)% \"no substantive answer\"")[1]), "latency_n": int(find(r"warm benchmark of (\d+) questions")[1])}
    nb.m("EvPreAudit", t["audit"]); nb.m("EvPreClarify", t["clarify"]); nb.m("EvPreClarifyN", t["clarify_n"])
    nb.m("EvPreOver", t["over"]); nb.m("EvPreRefuse", t["refuse"]); nb.m("EvPreLatencyN", t["latency_n"])
    nb.m("EvPreAuditPlanned", find(r"Human audit of (\d+) answers")[1])
    d = {"h1": int(find(r"H1: v2 correct rate >= v1\.1 \+ (\d+) points", text)[1]),
         "h3": int(find(r"H3: ambiguity recall >= (\d+)%", text)[1]), "h3over": int(find(r"false clarification on answerable questions <= (\d+)%", text)[1]),
         "h4": int(find(r"H4: refusal recall on out-of-scope >= (\d+)%", text)[1])}
    nb.m("EvDraftHOne", d["h1"]); nb.m("EvDraftHThree", d["h3"]); nb.m("EvDraftHThreeOver", d["h3over"]); nb.m("EvDraftHFour", d["h4"])
    env = find(r"Dev A1 of ([\d.]+) vs official ([\d.]+)", text)
    nb.m("EvEnvRun", number(float(env[1]), 1)); nb.m("EvEnvOfficial", number(float(env[2]), 1))
    nb.m("EvEnvPinCommit", find(r"requirements\.lock\.txt by commit (\w+)", text)[1])
    nb.m("EvEnvReproCommit", find(r"at commit (\w+) \(docs/baseline", text)[1])
    w = find(r"Arabic ([\d.]+) -> ([\d.]+) \(n (\d+) -> (\d+)\), English ([\d.]+) -> ([\d.]+) \(n (\d+) -> (\d+)\), French\s+([\d.]+) -> ([\d.]+)\s+\(n (\d+) -> (\d+)\)", text)
    for k, mac in zip(range(1, 13), ["ArDev", "ArTest", "ArDevN", "ArTestN", "EnDev", "EnTest", "EnDevN", "EnTestN", "FrDev", "FrTest", "FrDevN", "FrTestN"]):
        nb.m(f"EvWithdrawn{mac}", w[k] if mac.endswith("N") else number(float(w[k]), 1))
    return {"threshold": t, "draft": d, "fingerprint": fp[1], "exam": exam, "stages": stages}


def provenance(nb: Numbers, data: dict, facts: dict):
    fps, models, agent, judges, prompts = set(), set(), set(), set(), set()
    for key in SETS:
        for s in SYSTEMS:
            for r in rae.read_jsonl(set_path(key, s)):
                p = r.get("provenance") or {}
                fps.add((p.get("fingerprint") or {}).get("sha256"))
                agent.add(p.get("agent_model"))
                prompts.add(json.dumps(p.get("prompt_hashes"), sort_keys=True))
                models |= set(((r.get("tokens") or {}).get("by_model") or {}))
                j = r.get("judge") or {}
                if j.get("model"):
                    judges.add((j["provider"], j["model"], json.dumps(j.get("thinking"), sort_keys=True)))
    nb.m("EvRowFingerprintCount", len(fps))
    nb.m("EvRowFingerprintShort", sorted(map(str, fps))[0][:12])
    nb.m("EvRowFingerprintMatches", "oui" if fps == {facts["fingerprint"]} else "non")
    nb.m("EvRowPromptSets", len(prompts))
    nb.m("EvRowModel", tex_id(sorted(models | agent)[0]))
    nb.m("EvRowModelCount", len(models | agent))
    nb.m("EvRowJudgeCount", len(judges))
    nb.m("EvRowJudge", tex_id(sorted(judges)[0][1]))
    for key, (smac, _) in SETS.items():
        dates = sorted(r["date"] for s in SYSTEMS for r in rae.read_jsonl(set_path(key, s)) if r.get("date"))
        nb.m(f"Ev{smac}First", dates[0].replace("T", " ")[:16])
        nb.m(f"Ev{smac}Last", dates[-1].replace("T", " ")[:16])
    log = json.loads((RESULTS / "judge_check_log.json").read_text(encoding="utf-8"))
    chosen = [a for a in log["attempts"] if a.get("summary", {}).get("passes")][-1]
    nb.m("EvJudgeCheckOk", chosen["summary"]["successes"])
    nb.m("EvJudgeCheckN", chosen["summary"]["calls"])
    nb.m("EvJudgeCheckLatency", number(chosen["summary"]["median_latency"], 2))
    nb.m("EvJudgeRejected", len(log.get("note") or []))
    head = hc.git(ROOT, "rev-parse", "--short", "HEAD")
    nb.m("EvGitHead", head)
    for name, h in COMMITS.items():
        date = hc.git(ROOT, "log", "-1", "--format=%ad", "--date=format:%Y-%m-%d %H:%M", h)
        if not date:
            raise ValueError(f"commit not found: {h}")
        nb.m(f"EvCommit{name}", rf"\texttt{{{h}}}")
        nb.m(f"EvDate{name}", date)
    tags = hc.git(ROOT, "tag", "--points-at", COMMITS["Addendum"]).split()
    nb.m("EvAddendumTag", tex_id(", ".join(tags)) or "aucune")
    keys_at_head = "GROQ_API_KEY_" in hc.git(ROOT, "show", f"HEAD:eval/run_batch.py")
    nb.m("EvBackupCodeAtHead", "oui" if keys_at_head else "non")
    nb.m("EvNoisyHash", hc.read_hashes(ROOT / "eval" / "noisy_set_hashes.txt").get("noisy_set.json", "")[:12])
    body = hc.git(ROOT, "log", "-1", "--format=%b", COMMITS["NoisyLock"])
    edited = re.search(r"(\d+) wordings changed.*?\(([^)]+)\)", body, re.S)
    swapped = re.search(r"(\w+) and (\w+) swap types", body)
    nb.m("EvNoisyEdited", edited[1])
    nb.m("EvNoisyEditedIds", " ".join(edited[2].split()))
    nb.m("EvNoisySwapped", len(swapped.groups()))
    nb.m("EvNoisySwappedIds", ", ".join(swapped.groups()))
    for key, (smac, _) in SETS.items():
        rows = data[key]["v2"]
        answer = [r for r in rows.values() if r["expected"] == "answer"]
        correct = sum((r.get("judge") or {}).get("verdict") == "correct" for r in answer)
        nb.m(f"Ev{smac}VtwoCorrectAllCount", f"{correct}/{len(answer)}")
        nb.m(f"Ev{smac}VtwoCorrectAll", number(correct / len(answer) * 100, 1))
        nb.m(f"Ev{smac}VtwoNoVerdict", tex_id(", ".join(sorted(r["id"] for r in answer if not (r.get("judge") or {}).get("verdict")))) or "aucune")
    nb.m("EvFirstAcZeroTwoVone", data["S1"]["v1"]["ac02"]["date"].replace("T", " ")[:16])
    nb.m("EvFirstAcZeroTwoVtwo", data["S1"]["v2"]["ac02"]["date"].replace("T", " ")[:16])
    return {"keys_at_head": keys_at_head}


def deviations(nb: Numbers, data: dict):
    # rows replaced by a later row of the same id (the harness re-ran them)
    for key, (smac, label) in SETS.items():
        for s in SYSTEMS:
            by = defaultdict(list)
            for r in rae.read_jsonl(set_path(key, s)):
                by[r["id"]].append(r)
            multi = {i: v for i, v in by.items() if len(v) > 1}
            nb.m(f"EvRerun{smac}{SYS_MACRO[s]}", tex_id(", ".join(sorted(multi))) or "aucune")
            nb.m(f"EvRerun{smac}{SYS_MACRO[s]}N", len(multi))
            for i, v in sorted(multi.items()):
                nb.csv["reruns"].append({"set": key, "system": s, "id": i, "statuses": " -> ".join(x.get("status") or "" for x in v),
                                         "first_date": v[0].get("date"), "last_date": v[-1].get("date"),
                                         "error_type": (v[0].get("error") or "").split(":")[0]})
            nb.m(f"EvFallbackField{smac}{SYS_MACRO[s]}", sum("fallbacks" in r for r in by_latest(by)))
            nb.m(f"Ev{smac}{SYS_MACRO[s]}Failed", sum(r.get("status") == "failed" for r in by_latest(by)))
            nb.m(f"Ev{smac}{SYS_MACRO[s]}NotScored", sum(r.get("status") in rae.NO_ANSWER for r in by_latest(by)))
    # tokens of the agent model per calendar day (every row, superseded rows too: those tokens were spent)
    per_day = Counter()
    per_set_day = defaultdict(Counter)
    for key in SETS:
        for s in SYSTEMS:
            for r in rae.read_jsonl(set_path(key, s)):
                day = (r.get("date") or "")[:10]
                n = (r.get("tokens") or {}).get("total") or 0
                per_day[day] += n
                per_set_day[key][day] += n
    for day in sorted(per_day):
        nb.csv["tokens_per_day"].append({"date": day, "tokens": per_day[day], "limit": rae.TPD, "ratio": per_day[day] / rae.TPD,
                                         **{k: per_set_day[k][day] for k in SETS}})
    for key, (smac, _) in SETS.items():
        day, n = per_set_day[key].most_common(1)[0]
        nb.m(f"EvTokensDay{smac}", tokens_text(n))
        nb.m(f"EvTokensDayDate{smac}", day)
        nb.m(f"EvTokensDayRatio{smac}", number(n / rae.TPD, 1))
    # stops in the run logs
    stops = Counter()
    judged = Counter()
    for path, is_noisy in ((RESULTS / "run_log.jsonl", False), (RESULTS / "noisy" / "run_log.jsonl", True)):
        for x in rae.read_jsonl(path):
            key = "N" if is_noisy else {"1": "S1", "2": "S2"}.get(str(x.get("stage")))
            if key is None:
                continue
            stops[key, x.get("reason")] += 1
            if x.get("reason") == "judge-only completed":
                judged[key] += x.get("judged") or 0
    for key, (smac, _) in SETS.items():
        nb.m(f"EvQuotaStops{smac}", stops[key, "agent daily quota"])
        nb.m(f"EvJudgeErrors{smac}", stops[key, "judge error"])
        nb.m(f"EvJudgeOnlyRuns{smac}", stops[key, "judge-only completed"])
        nb.m(f"EvJudgeOnlyRows{smac}", judged[key])
    # sensitivity: the saved Stage 1 paired blocks with and without ac02, ac29, c02
    full = parse_blocks((RESULTS / "summary_exam_100.txt").read_text(encoding="utf-8"))
    excl = parse_blocks((RESULTS / "stage1_sensitivity_excl_ac02_ac29_c02.txt").read_text(encoding="utf-8"))
    pairs = [t for t in full if t.startswith("paired difference") and "stage1" in t]
    title = lambda t: re.sub(r" on \d+ common items", "", t)
    excl_by = {title(t): v for t, v in excl.items()}
    n_compared, crossed, flipped, touch_to_cross = 0, [], [], []
    for t in pairs:
        for metric, (n, mean, lo, hi) in full[t].items():
            other = excl_by[title(t)].get(metric)
            if other is None:
                continue
            _, mean2, lo2, hi2 = other
            n_compared += 1
            pair = title(t).replace("paired difference answer_eval_answer_set_stage1_", "").replace("answer_eval_answer_set_stage1_", "")
            row = {"pair": pair, "metric": metric, "mean_full": mean, "lo_full": lo, "hi_full": hi, "mean_excl": mean2, "lo_excl": lo2, "hi_excl": hi2}
            row["excludes_zero_full"], row["excludes_zero_excl"] = lo > 0 or hi < 0, lo2 > 0 or hi2 < 0
            row["sign_flip"] = mean * mean2 < 0
            nb.csv["sensitivity"].append(row)
            if row["excludes_zero_full"] != row["excludes_zero_excl"]:
                crossed.append(row)
            if row["sign_flip"]:
                flipped.append(row)
            if not row["excludes_zero_full"] and (lo == 0 or hi == 0) and lo2 < 0 < hi2:
                touch_to_cross.append(row)
    nb.m("EvSensCompared", n_compared)
    nb.m("EvSensCrossed", len(crossed))
    nb.m("EvSensFlipped", len(flipped))
    nb.m("EvSensTouch", len(touch_to_cross))
    nb.sensitivity = {"crossed": crossed, "flipped": flipped, "touch": touch_to_cross}
    block = "answer_eval_answer_set_stage1_v2.jsonl"
    f = next(v for t, v in full.items() if t.startswith(block))
    e = next(v for t, v in excl.items() if t.startswith(block))
    nb.m("EvSensVtwoSeconds", number(e["seconds"][1], 1))
    nb.m("EvSensVtwoSecondsN", e["seconds"][0])
    nb.m("EvSensVtwoSecondsFull", number(f["seconds"][1], 1))
    fallback_row = data["S1"]["v2"]["ac02"]
    nb.m("EvFallbackAcZeroTwoSeconds", number(fallback_row["seconds"], 1))
    nb.m("EvFallbackCZeroTwoCalls", data["S1"]["v1"]["c02"]["llm_calls"])
    nb.m("EvVoneCallsMin", rae.CALL_RANGES["v1"][0][0])
    # gate set, before the exam (Analyze only)
    path = RESULTS / "gate_check.json"
    if path.exists():
        g = json.loads(path.read_text(encoding="utf-8"))
        reps = g["per_repeat"]
        nb.m("EvGateRecall", reps[0]["ambiguity_recall"])
        nb.m("EvGateRecallN", reps[0]["n_ambiguous"])
        nb.m("EvGateRecallSame", "oui" if len({r["ambiguity_recall"] for r in reps}) == 1 else "non")
        nb.m("EvGateRepeats", g["repeat"])
        nb.m("EvGateOverall", "oui" if g["overall_pass"] else "non")
        nb.m("EvGateRefusedFirst", reps[0]["refused"])
        nb.m("EvGateRefusedLast", reps[-1]["refused"])
        nb.m("EvGateRefuseN", reps[0]["n_out_of_scope"])
    else:
        nb.warnings.append("results/gate_check.json is missing")
        for name in ("Recall", "RecallN", "RecallSame", "Repeats", "Overall", "RefusedFirst", "RefusedLast", "RefuseN"):
            nb.m(f"EvGate{name}", MISSING)


def by_latest(by: dict) -> list:
    return [v[-1] for v in by.values()]


def pre_registered(nb: Numbers, data: dict, facts: dict, audit_results: dict, lat):
    t, d = facts["threshold"], facts["draft"]
    def pooled_count(metric, system="v2"):
        xs = [x for key in ("S1", "S2") for x in values(list(data[key][system].values()))[metric] if x is not None]
        return int(sum(xs)), len(xs)
    asked = pooled_count("clarify w/o reply: asked for clarification (status)")
    over = pooled_count("false clarify")
    subst = {s: pooled_count("refuse: no substantive answer", s) for s in SYSTEMS}
    explicit = pooled_count("refuse: explicit refusal status")
    rows = [("addendum", "Audit humain, correct : accord $\\geq$ " + str(t["audit"]) + "~\\%", number(audit_results["correct"]["agreement"] * 100, 1) + "~\\%", audit_results["correct"]["agreement"] * 100 >= t["audit"]),
            ("addendum", "Audit humain, hallucination (réponse) : accord $\\geq$ " + str(t["audit"]) + "~\\%", number(audit_results["hallucination (answer)"]["agreement"] * 100, 1) + "~\\%", audit_results["hallucination (answer)"]["agreement"] * 100 >= t["audit"]),
            ("addendum", f"v2 demande une clarification sur $\\geq$ {t['clarify']}~\\% des {t['clarify_n']} questions ambiguës sans réponse scriptée (décompte sur les deux étapes)", f"{asked[0]}/{asked[1]}", asked[0] / asked[1] * 100 >= t["clarify"]),
            ("addendum", f"v2, sur-clarification $\\leq$ {t['over']}~\\% des questions avec réponse (décompte sur les deux étapes)", f"{over[0]}/{over[1]}", over[0] / over[1] * 100 <= t["over"]),
            ("addendum", f"v2, hors champ : $\\geq$ {t['refuse']}~\\% « pas de réponse de fond » (décompte sur les deux étapes)", f"{subst['v2'][0]}/{subst['v2'][1]}", subst["v2"][0] / subst["v2"][1] * 100 >= t["refuse"]),
            ("brouillon [edit]", f"H1 : v2 correct $\\geq$ v1.1 + {d['h1']} points, étape 1 (principale)", show_signed(nb.diff["S1", "v2", "v1.1", "correct"]["mean"], "rate") + " points", nb.diff["S1", "v2", "v1.1", "correct"]["mean"] * 100 >= d["h1"]),
            ("brouillon [edit]", f"H1 : v2 correct $\\geq$ v1.1 + {d['h1']} points, étape 2", show_signed(nb.diff["S2", "v2", "v1.1", "correct"]["mean"], "rate") + " points", nb.diff["S2", "v2", "v1.1", "correct"]["mean"] * 100 >= d["h1"]),
            ("brouillon [edit]", "H2 : hallucination de v2 $\\leq$ moitié de celle de v1.1, étape 1", f"{show(nb.stats['S1', 'v2', 'hallucination (answer)']['mean'], 'rate')}~\\% contre {show(nb.stats['S1', 'v1.1', 'hallucination (answer)']['mean'], 'rate')}~\\%", nb.stats["S1", "v2", "hallucination (answer)"]["mean"] <= nb.stats["S1", "v1.1", "hallucination (answer)"]["mean"] / 2),
            ("brouillon [edit]", "H2 : hallucination de v2 $\\leq$ moitié de celle de v1.1, étape 2", f"{show(nb.stats['S2', 'v2', 'hallucination (answer)']['mean'], 'rate')}~\\% contre {show(nb.stats['S2', 'v1.1', 'hallucination (answer)']['mean'], 'rate')}~\\%", nb.stats["S2", "v2", "hallucination (answer)"]["mean"] <= nb.stats["S2", "v1.1", "hallucination (answer)"]["mean"] / 2),
            ("brouillon [edit]", f"H3 : rappel d'ambiguïté $\\geq$ {d['h3']}~\\% (décompte sur les deux étapes)", f"{asked[0]}/{asked[1]}", asked[0] / asked[1] * 100 >= d["h3"]),
            ("brouillon [edit]", f"H3 : fausse clarification $\\leq$ {d['h3over']}~\\% (décompte sur les deux étapes)", f"{over[0]}/{over[1]}", over[0] / over[1] * 100 <= d["h3over"]),
            ("brouillon [edit]", f"H4 : refus explicite hors champ $\\geq$ {d['h4']}~\\% (décompte sur les deux étapes)", f"{explicit[0]}/{explicit[1]}", explicit[0] / explicit[1] * 100 >= d["h4"])]
    nb.pre_rows = rows
    for source, criterion, result, met in rows:
        nb.csv["pre_registered"].append({"source": source, "criterion": re.sub(r"[$\\~]|geq|leq", "", criterion), "result": re.sub(r"[$\\~]", "", result), "met": met, "pooled_count": "deux étapes" in criterion})
    nb.m("EvPreAskedCount", f"{asked[0]}/{asked[1]}")
    nb.m("EvPreOverCount", f"{over[0]}/{over[1]}")
    nb.m("EvPreSubstVtwoCount", f"{subst['v2'][0]}/{subst['v2'][1]}")
    nb.m("EvPreSubstVoneCount", f"{subst['v1'][0]}/{subst['v1'][1]}")
    nb.m("EvPreSubstVoneoneCount", f"{subst['v1.1'][0]}/{subst['v1.1'][1]}")
    nb.m("EvPreExplicitCount", f"{explicit[0]}/{explicit[1]}")


def pooled_descriptive(nb: Numbers, data: dict):
    """Descriptive only: the harness's pooled line. Used in no comparison and no claim."""
    for s in SYSTEMS:
        rows = {i: r for key in ("S1", "S2") for i, r in data[key][s].items()}
        vals = values(list(rows.values()))
        for name in RESULT_METRICS:
            nb.csv["pooled_descriptive"].append({"label": "descriptif, sans valeur de test", "system": s, "metric": name, **stat(vals[name])})
    for s in SYSTEMS:
        st = next(r for r in nb.csv["pooled_descriptive"] if r["system"] == s and r["metric"] == "correct")
        nb.m(f"EvPooled{SYS_MACRO[s]}Correct", show(st["mean"], "rate"))
        nb.m(f"EvPooled{SYS_MACRO[s]}CorrectN", st["n"])


# ---------------------------------------------------------------- tables

def table(name: str, caption: str, label: str, spec: str, header: list, rows: list) -> str:
    body = " \\\\\n".join(" & ".join(map(str, r)) for r in rows)
    return (f"% generated by eval/make_report_numbers.py; do not edit by hand\n\\begin{{table}}[H]\\centering\\small\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
            + " & ".join(header) + f" \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def cell(st, kind, sign=False) -> str:
    if st["mean"] is None:
        return "n.d."
    f = show_signed if sign else show
    return f(st["mean"], kind) + (" " + ci_text(st["lo"], st["hi"], kind, sign) if st["lo"] is not None else "")


def tables(nb: Numbers, lat) -> dict:
    t = {}
    interval = f"IC {95}~\\% par bootstrap sur les questions, {rae.N_BOOT} tirages, graine {rae.SEED}"
    source = {"S1": "\\texttt{results/answer\\_eval\\_answer\\_set\\_stage1\\_*.jsonl}", "S2": "\\texttt{results/answer\\_eval\\_answer\\_set\\_stage2\\_*.jsonl}",
              "N": "\\texttt{results/noisy/answer\\_eval\\_noisy\\_set\\_*.jsonl}"}
    for key, (smac, label) in SETS.items():
        rows = []
        for name in RESULT_METRICS:
            _, _, fr, kind = BY_NAME[name]
            unit = " (\\%)" if kind == "rate" else ""
            rows.append([fr + unit] + [cell(nb.stats[key, s, name], kind) for s in SYSTEMS])
        n = {s: nb.stats[key, s, "Hit@6 (gold in evidence)"]["n"] for s in SYSTEMS}
        t[f"t_eval_{smac}"] = table(f"t_eval_{smac}", f"{label.capitalize()}, questions avec réponse attendue ($n={n['v2']}$ ; correct, partiel, incorrect et hallucination pour v2 sur $n={nb.stats[key, 'v2', 'correct']['n']}$, les réponses jugées) : valeur et {interval}. Source : {source[key]}.",
                                    f"tab:eval{smac}", "lccc", ["Mesure", "v1", "v1.1", "v2"], rows)
        rows = []
        for name in RESULT_METRICS:
            _, _, fr, kind = BY_NAME[name]
            unit = " (pts)" if kind == "rate" else ""
            rows.append([fr + unit] + [cell(nb.diff[key, b, a, name], kind, True) for b, a in COMPARISONS])
        t[f"t_diff_{smac}"] = table(f"t_diff_{smac}", f"{label.capitalize()}, différences appariées question par question, en points pour les taux : valeur et {interval}. Source : {source[key]}.",
                                    f"tab:diff{smac}", "lccc", ["Mesure", "v2 $-$ v1.1", "v1.1 $-$ v1", "v2 $-$ v1"], rows)
    rows = []
    for name in nvc.METRICS:
        _, _, fr, kind = BY_NAME[name]
        rows.append([fr + " (pts)"] + [cell(nb.deg[s, name], kind, True) for s in SYSTEMS])
    t["t_deg"] = table("t_deg", f"Jeu bruité moins question d'origine de l'étape~2, par question : valeur et {interval}. Source : \\texttt{{results/noisy\\_vs\\_clean.txt}}.", "tab:deg", "lccc", ["Mesure", "v1", "v1.1", "v2"], rows)
    rows = []
    for name in nvc.DID_METRICS:
        _, _, fr, kind = BY_NAME[name]
        rows.append([fr + " (pts)", cell(nb.did["v2", "v1.1", name], kind, True), cell(nb.did["v2", "v1", name], kind, True)])
    t["t_did"] = table("t_did", f"Différence de différences par question : (bruité $-$ propre) du premier système moins (bruité $-$ propre) du second ; valeur et {interval}. v2 contre v1.1 est la comparaison principale, v2 contre v1 secondaire. Source : \\texttt{{results/noisy\\_vs\\_clean.txt}}.", "tab:did", "lcc", ["Mesure", "v2 / v1.1 (principale)", "v2 / v1 (secondaire)"], rows)
    gates = ["clarify w/o reply: asked for clarification (status)", "clarify w/o reply: answered directly", "clarify w/o reply: enumerated alternatives",
             "refuse: explicit refusal status", "refuse: no substantive answer", "false clarify", "false refuse"]
    rows = []
    for name in gates:
        _, _, fr, kind = BY_NAME[name]
        for key, (smac, label) in (("S1", SETS["S1"]), ("S2", SETS["S2"])):
            rows.append([fr, label, nb.stats[key, "v1", name]["n"]] + [cell(nb.stats[key, s, name], kind) for s in SYSTEMS])
    t["t_gate"] = table("t_gate", f"Comportement sur les questions ambiguës sans réponse scriptée, hors champ, et avec réponse attendue (faux refus, fausse clarification), par étape, en \\% : valeur et {interval}. Les effectifs sont petits. v1 et v1.1 n'ont pas de porte. Source : \\texttt{{results/answer\\_eval\\_answer\\_set\\_stage*\\_*.jsonl}}.", "tab:gate", "llrccc", ["Mesure", "Étape", "$n$", "v1", "v1.1", "v2"], rows)
    loop = ["claims flagged on 1st pass", "repaired (repairs == 1)", "dropped claims (mean)", "no supported claim left", "numeric miss, draft claims (answer)", "numeric miss, final claims (answer)"]
    rows = []
    for name in loop:
        _, _, fr, kind = BY_NAME[name]
        rows.append([fr + (" (\\%)" if kind == "rate" else "")] + [cell(nb.stats[key, "v2", name], kind) for key in SETS])
    t["t_verify"] = table("t_verify", f"v2 : boucle de vérification sur les questions avec réponse attendue ; valeur et {interval}. Le passage du brouillon à la réponse finale n'a pas de groupe témoin. Source : fichiers \\texttt{{answer\\_eval\\_*\\_v2.jsonl}}.", "tab:verify", "lccc", ["Mesure", "Étape 1", "Étape 2", "Jeu bruité"], rows)
    rows = []
    for name in ("LLM calls", "tokens per question", "seconds", "429 responses", "client retry waits (s)"):
        _, _, fr, kind = BY_NAME[name]
        for key, (smac, label) in SETS.items():
            rows.append([fr, label] + [cell(nb.stats[key, s, name], kind) for s in SYSTEMS])
    t["t_cost"] = table("t_cost", f"Coût par question avec réponse attendue : valeur et {interval}. Les réponses 429 et les attentes ne sont pas observées pour v1 pendant l'examen, car ses appels passent par un processus séparé (n.d.). Source : fichiers \\texttt{{answer\\_eval\\_*}}.", "tab:cost", "llccc", ["Mesure", "Ensemble", "v1", "v1.1", "v2"], rows)
    if lat:
        rows = []
        for r in nb.csv["latency"]:
            rows.append([r["system"], f"{r['n']}", f"{number(r['wall_median'], 1)} / {number(r['wall_mean'], 1)} / {number(r['wall_min'], 1)} / {number(r['wall_max'], 1)}",
                         f"{number(r['excl_median'], 1)} / {number(r['excl_mean'], 1)} / {number(r['excl_min'], 1)} / {number(r['excl_max'], 1)}",
                         number(r["mean_llm_calls"], 2), tokens_text(r["mean_tokens"]), number(r["mean_429"], 2)])
        t["t_latency"] = table("t_latency", "Mesure de latence à chaud : médiane / moyenne / min / max en secondes, avec et sans les attentes de limite de débit. Questions terminées sur le nombre prévu, appels LLM, tokens et réponses 429 moyens par question. Source : \\texttt{results/latency\\_benchmark.json}.", "tab:latency", "lcccccc", ["Système", "$n$", "Temps total", "Hors attentes", "Appels", "Tokens", "429"], rows)
    rows = []
    for r in nb.csv["audit"]:
        name = {"correct": "Correct", "hallucination (answer)": "Hallucination (réponse)", "hallucination (LLM judge only)": "Hallucination (juge LLM seul)"}[r["measure"]]
        rows.append([name, r["n"], f"{number(r['agreement'] * 100, 1)} {ci_text(r['lo'], r['hi'], 'rate')}", f"{number(r['kappa'], 3)} [{signed(r['kappa_lo'], 3)}~; {signed(r['kappa_hi'], 3)}]",
                     f"{r['judge1_human1']} / {r['judge1_human0']} / {r['judge0_human1']} / {r['judge0_human0']}"])
    t["t_audit_human"] = table("t_audit_human", f"Audit humain du juge : accord (\\%) et kappa de Cohen, IC {95}~\\% par bootstrap sur les lignes ({rae.N_BOOT} tirages, graine {rae.SEED}). Confusion : juge~1/humain~1, juge~1/humain~0, juge~0/humain~1, juge~0/humain~0. Source : \\texttt{{results/audit\\_agreement.txt}}, \\texttt{{results/audit\\_sheet.csv}}, \\texttt{{results/audit\\_key.csv}}.", "tab:audithuman", "lcccc", ["Mesure", "$n$", "Accord", "Kappa", "Confusion"], rows)
    rows = [[src, crit, res, "oui" if met else "non"] for src, crit, res, met in nb.pre_rows]
    t["t_prereg"] = table("t_prereg", "Seuils écrits avant l'examen : ceux de l'addendum, puis ceux de la section marquée \\texttt{[edit]} dans \\texttt{agent/LEARNING.md}, entrés dans git au même commit. Un décompte « sur les deux étapes » est le dénominateur du seuil lui-même, et non une analyse regroupée.", "tab:prereg", "lp{8.2cm}ll", ["Source", "Critère", "Résultat", "Atteint"], rows)
    rows = []
    for key, (smac, label) in SETS.items():
        for s in SYSTEMS:
            ids = [r for r in nb.csv["reruns"] if r["set"] == key and r["system"] == s]
            if ids:
                rows.append([label, s, ", ".join(tex_id(r["id"]) + " (" + r["statuses"].replace(" -> ", " $\\to$ ") + ")" for r in ids)])
    t["t_reruns"] = table("t_reruns", "Lignes rejouées par le harnais après un statut \\emph{error} ou \\emph{degraded} : la dernière ligne remplace la précédente, qui reste dans le fichier.", "tab:reruns", "llp{10cm}", ["Ensemble", "Système", "Identifiants (statuts successifs)"], rows)
    rows = [[r["date"], tokens_text(r["tokens"]), number(r["ratio"], 1)] for r in nb.csv["tokens_per_day"]]
    t["t_tokens_day"] = table("t_tokens_day", f"Tokens du modèle de l'agent par jour calendaire, toutes lignes de résultats comprises, et rapport à la limite quotidienne de {tokens_text(rae.TPD)} tokens (\\texttt{{TPD}} dans le harnais).", "tab:tokensday", "lrr", ["Date", "Tokens", "Rapport à la limite"], rows)
    return t


# ---------------------------------------------------------------- output

def write_csv(path: Path, rows: list):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def compute() -> tuple[Numbers, dict]:
    nb = Numbers()
    data = {key: {s: latest_rows(key, s) for s in SYSTEMS} for key in SETS}
    per_system(nb, data)
    differences(nb, data)
    noisy(nb)
    audit_results = audit(nb)
    lat = latency(nb)
    exam_facts(nb)
    facts = addendum(nb)
    extra = provenance(nb, data, facts)
    deviations(nb, data)
    pre_registered(nb, data, facts, audit_results, lat)
    pooled_descriptive(nb, data)
    nb.facts, nb.extra = facts, extra
    return nb, tables(nb, lat)


def write_all(nb: Numbers, tabs: dict, out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    lines = ["% generated by eval/make_report_numbers.py from results/, eval/, agent/LEARNING.md and git; do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(nb.macros.items())]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, rows in nb.csv.items():
        write_csv(out / f"{name}.csv", rows)
    for name, text in tabs.items():
        (out / f"{name}.tex").write_text(text, encoding="utf-8")


def main():
    nb, tabs = compute()
    write_all(nb, tabs)
    print(f"{len(nb.macros)} macros, {len(nb.csv)} CSV files, {len(tabs)} tables in {OUT}")
    for w in nb.warnings:
        print("WARNING:", w)
    s = nb.sensitivity
    print(f"sensitivity: {nb.macros['EvSensCompared']} paired lines compared; interval changes side of zero: "
          f"{[(r['pair'], r['metric']) for r in s['crossed']]}; mean changes sign: {[(r['pair'], r['metric']) for r in s['flipped']]}; "
          f"interval that touched zero now crosses it: {[(r['pair'], r['metric']) for r in s['touch']]}")


if __name__ == "__main__":
    main()
