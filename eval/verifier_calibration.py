"""Verifier calibration with ground truth by construction (calls Groq: one verify() call per article).

    python eval/verifier_calibration.py --n 24

Sample: N articles from the PDF text (verified sources only, eval/pdf_truth.py), seed 20261022, at most ceil(N/6)
per source, each with a chunk in the index and a sentence of 50-300 characters containing a number. The sentence is
a verbatim substring of normalize(PDF article text). Four claims per article, shuffled into one Answer:
  good            the sentence, cited to its own chunk                     truth: supported
  number_changed  the first number replaced by another number              truth: not supported
  wrong_article   the sentence cited to another chunk of the same source   truth: not supported
  added_condition the sentence + " à condition que la personne concernée soit mineure"   truth: not supported
The evidence given to verify() is the index chunk text (what the system really sees) of both cited chunks.
Choices: "un"/"une" are counted as numbers for eligibility only together with another number, and are never the
number that is changed (they are mostly articles in French); only "fact" numbers count (see fact_numbers: not a
reference to an article/alinéa/section, not a list marker like "2)", "1°", "3-", not part of a date);
sentences with "n°" (law references) are skipped.
"""
import sys
import json
import math
import random
import sqlite3
import hashlib
import time
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "eval"))
import re
from pdf_truth import load, normalize, ROOT as _R
from answer import Answer, Claim
import verify

WAITS = (10, 30, 60)                 # retries of transient errors (connection, timeout, 429) left after the client's
HTTP = {"200": 0, "429": 0, "other": 0}


class _CountGroq(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "api.groq.com" in msg:
            HTTP["200" if " 200 " in msg else "429" if " 429 " in msg else "other"] += 1


logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpx").addHandler(_CountGroq())
logging.getLogger("httpx").propagate = False


def transient(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}"
    return any(k in s for k in ("RateLimit", "429", "APIConnection", "Connect", "Timeout", "RemoteProtocol"))


def verify_retry(answer, evidence):
    for attempt in range(len(WAITS) + 1):
        try:
            return verify.verify(answer, evidence)
        except Exception as e:
            if not transient(e) or attempt == len(WAITS):
                raise
            print(f"  transient error ({type(e).__name__}); retry in {WAITS[attempt]} s", file=sys.stderr, flush=True)
            time.sleep(WAITS[attempt])

SEED = 20261022
DB = ROOT / "db_v1_1" / "chroma.sqlite3"
OUT = ROOT / "results" / "verifier_calibration.json"
CONDITION = " à condition que la personne concernée soit mineure"
TYPES = ["good", "number_changed", "wrong_article", "added_condition"]
WORDS = ["un", "une", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze",
         "treize", "quatorze", "quinze", "seize", "vingt", "trente", "quarante", "cinquante", "soixante", "cent", "mille"]
NUMBER = re.compile(r"\d+(?:[.,]\d+)?(?![)°\d])|\b(?:" + "|".join(WORDS) + r")\b", re.I)   # "2)" / "1°" list markers excluded
NEXT_WORD = {w: WORDS[(i + 1) % len(WORDS)] if WORDS[(i + 1) % len(WORDS)] not in ("un", "une") else "deux"
             for i, w in enumerate(WORDS)}
MONTHS = r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
REF_BEFORE = re.compile(r"(?:articles?|art\.|alinéas?|numéros?|section|chapitre|titre|livre|paragraphe|parag\.)"
                        r"\s+[\d\s,etoàu-]*$", re.I)
MARKER_AFTER = re.compile(r"\s*(?:[-–/)°(]|\.\s*-|\.\s*$)")     # "2)", "1°", "3-", "1/", ")1(" footnote, final "2."
DATE_AROUND = re.compile(rf"^\s+(?:{MONTHS})\b|^\s*(?:er\s+)?(?:{MONTHS})\b", re.I)
YEAR_BEFORE = re.compile(rf"(?:{MONTHS})\s+$", re.I)


def fact_numbers(s: str) -> list:
    """Number tokens that state a fact (a duration, an amount, a count, a rate): not "un"/"une", not a reference
    to an article/alinéa/section, not a list marker ("2.-", "3-", "1/"), not part of a date."""
    out = []
    for m in NUMBER.finditer(s):
        tok = m.group(0).lower()
        if tok in ("un", "une") or REF_BEFORE.search(s[:m.start()]) or YEAR_BEFORE.search(s[:m.start()]):
            continue
        if tok[0].isdigit() and (MARKER_AFTER.match(s[m.end():]) or DATE_AROUND.match(s[m.end():])):
            continue
        out.append(m)
    return out


def flat(t: str) -> str:
    return normalize(t).replace(" ", "")


def index_chunks() -> dict:
    """cid -> {source, article, law_type, text} from the sqlite index (no model)."""
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    meta = defaultdict(dict)
    for i, k, v in con.execute("select id, key, string_value from embedding_metadata where key in "
                               "('chunk_id','source','article','law_type','chroma:document')"):
        meta[i][k] = v
    con.close()
    return {m["chunk_id"]: {"source": m["source"], "article": m["article"], "law_type": m["law_type"],
                            "text": m["chroma:document"]} for m in meta.values()}


def candidate_sentence(text: str, rng: random.Random) -> str | None:
    t = normalize(text)
    sents = [s.strip() for s in re.split(r"(?<=[.;:])\s+", t)]
    ok = [s for s in sents if 50 <= len(s) <= 300 and not s.lower().startswith(("article", "art."))
          and "n°" not in s and fact_numbers(s)]
    return rng.choice(ok) if ok else None


def change_first_number(s: str) -> str:
    m = fact_numbers(s)[0]
    tok = m.group(0)
    if tok[0].isdigit():
        new = str(int(re.match(r"\d+", tok).group(0)) + 1) + tok[len(re.match(r"\d+", tok).group(0)):]
    else:
        new = NEXT_WORD[tok.lower()]
        new = new.capitalize() if tok[0].isupper() else new
    return s[:m.start()] + new + s[m.end():]


def add_condition(s: str) -> str:
    return s[:-1] + CONDITION + s[-1] if s[-1] in ".;:" else s + CONDITION + "."


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def build_sample(n: int, rng: random.Random, chunks: dict):
    pdf = load()
    cap = math.ceil(n / 6)
    cands = sorted((stem, sec, label, text) for stem, d in pdf.items() if d["status"] == "verified"
                   for label, occ in d["articles"].items() for sec, text in occ)
    rng.shuffle(cands)
    by_src, picked, chunk_mismatch, no_sentence = Counter(), [], [], 0
    for stem, sec, label, text in cands:
        if len(picked) == n:
            break
        cid = f"{stem}/s{sec:02d}/{label}"
        if by_src[stem] >= cap or cid not in chunks:
            continue
        sent = candidate_sentence(text, rng)
        if sent is None:
            no_sentence += 1
            continue
        if flat(sent) not in flat(chunks[cid]["text"]):
            chunk_mismatch.append(cid)
            continue
        others = sorted(c for c in chunks if c.startswith(stem + "/") and c != cid and flat(sent) not in flat(chunks[c]["text"]))
        picked.append({"cid": cid, "stem": stem, "section": sec, "label": label, "sentence": sent,
                       "other": rng.choice(others)})
        by_src[stem] += 1
    return picked, chunk_mismatch, no_sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    rng = random.Random(SEED)
    chunks = index_chunks()
    sample, chunk_mismatch, no_sentence = build_sample(args.n, rng, chunks)
    prompt_sha = hashlib.sha1(verify.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    print(f"verifier calibration: n={args.n} (got {len(sample)}), seed {SEED}, verify.SYSTEM_PROMPT sha1 {prompt_sha}")
    print(f"skipped: chunk_mismatch {len(chunk_mismatch)} {chunk_mismatch}, articles without an eligible sentence {no_sentence}")
    print(f"per source: {dict(Counter(s['stem'] for s in sample))}\n")

    rows = []
    for k, s in enumerate(sample, 1):
        claims = {"good": Claim(text=s["sentence"], cited_ids=[s["cid"]]),
                  "number_changed": Claim(text=change_first_number(s["sentence"]), cited_ids=[s["cid"]]),
                  "wrong_article": Claim(text=s["sentence"], cited_ids=[s["other"]]),
                  "added_condition": Claim(text=add_condition(s["sentence"]), cited_ids=[s["cid"]])}
        order = TYPES[:]
        rng.shuffle(order)
        evidence = {c: chunks[c] for c in (s["cid"], s["other"])}
        print(f"  [{k}/{len(sample)}] {s['cid']}", file=sys.stderr, flush=True)
        try:
            v = verify_retry(Answer(claims=[claims[t] for t in order]), evidence)
        except Exception as e:
            print(f"\nSTOPPED: verify() failed on {s['cid']} after the client's retries and 3 more: {type(e).__name__}: {e}")
            break
        by_index = {x.claim_index: x for x in v.verdicts}
        for pos, t in enumerate(order, 1):
            x = by_index.get(pos)
            rows.append({"type": t, "cid": s["cid"], "stem": s["stem"], "label": s["label"], "cited": claims[t].cited_ids,
                         "claim": claims[t].text, "position": pos,
                         "verdict": x.verdict if x else "MISSING", "problem": x.problem if x else ""})

    done = len({r["cid"] for r in rows})
    print(f"\nverified articles: {done}/{len(sample)}\n")
    print(f"  {'type':<17}{'n':>4}{'flagged':>9}{'rate':>8}   95% Wilson      supported  partly  unsupported  missing")
    summary = {}
    for t in TYPES:
        rs = [r for r in rows if r["type"] == t]
        c = Counter(r["verdict"] for r in rs)
        flagged = sum(r["verdict"] != "supported" for r in rs)
        lo, hi = wilson(flagged, len(rs))
        summary[t] = {"n": len(rs), "flagged": flagged, "wilson95": [lo, hi], "verdicts": dict(c)}
        rate = f"{flagged / len(rs):.1%}" if rs else "-"
        print(f"  {t:<17}{len(rs):>4}{flagged:>9}{rate:>8}   [{lo:.1%}, {hi:.1%}]   {c['supported']:>9}{c['partly']:>8}"
              f"{c['unsupported']:>13}{c['MISSING']:>9}")
    print("\n  'flagged' = verdict != supported (for 'good' a flag is a false alarm; for the others it is a detection)")

    misses = [r for r in rows if (r["type"] == "good") == (r["verdict"] != "supported")]
    print(f"\nMISSES ({len(misses)}): good claims not judged supported, mutated claims judged supported")
    for r in misses:
        print(f"  - {r['type']:<16} {r['stem']} art. {r['label']} (cites {', '.join(r['cited'])})  verdict={r['verdict']}"
              f"\n      claim:   {r['claim']}\n      problem: {r['problem'] or '-'}")

    print(f"\nGroq HTTP responses: {HTTP}   (verify.VERIFY_CHARS = {verify.VERIFY_CHARS})")
    OUT.write_text(json.dumps({"seed": SEED, "n": args.n, "verify_prompt_sha1": prompt_sha,
                               "verify_chars": verify.VERIFY_CHARS, "http": HTTP,
                               "chunk_mismatch": chunk_mismatch, "no_eligible_sentence": no_sentence,
                               "sample": sample, "rows": rows, "summary": summary}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nSaved {OUT.relative_to(ROOT)}")
    if done < len(sample):
        sys.exit(1)


if __name__ == "__main__":
    main()
