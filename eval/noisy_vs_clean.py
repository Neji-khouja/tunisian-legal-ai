"""Noisy vs clean: paired analysis of the noisy-wording set against the same items in Stage 2. No LLM call, read-only.

    python eval/noisy_vs_clean.py [--clean-dir results] [--noisy-dir results/noisy] [--out results/noisy_vs_clean.txt]
"""
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_answer_eval as rae

SYSTEMS = ["v1", "v1.1", "v2"]
METRICS = ["Hit@6 (gold in evidence)", "correct", "partial", "incorrect", "hallucination (answer)",
           "numeric check ok (answer)", "false refuse"]
DID_METRICS = ["Hit@6 (gold in evidence)", "correct", "incorrect", "hallucination (answer)"]
MIN_N_CI = 8                                    # section C: interval only from this group size


def latest(path: Path) -> tuple[dict, int]:
    """-> ({id: latest row}, number of older rows replaced by a later row of the same id)."""
    rows = rae.read_jsonl(path)
    by_id = {r["id"]: r for r in rows}
    return by_id, len(rows) - len(by_id)


def values(row) -> dict:
    """metric -> 0/1 (or None = not applicable) for one valid row, as --summarize scores it."""
    v = rae.metric_values([row])
    return {m: (v[m][0] if v.get(m) else None) for m in METRICS}


def why_out(row) -> str | None:
    if row is None:
        return "no row"
    if row.get("status") in rae.NO_ANSWER:
        return f"status {row.get('status')}"
    return None


def pairs(clean: dict, noisy: dict, mapping: list[dict]) -> tuple[dict, list]:
    """-> ({original id: (clean values, noisy values)}, [(original id, side, reason)] for the items left out)."""
    out, excluded = {}, []
    for n in mapping:
        o = n["original_id"]
        c, z = clean.get(o), noisy.get(n["id"])
        reasons = [(side, why_out(r)) for side, r in (("clean", c), ("noisy", z)) if why_out(r)]
        if c is not None and c.get("expected") != "answer":
            reasons.append(("clean", f"expected {c.get('expected')}"))
        if reasons:
            excluded += [(o, side, why) for side, why in reasons]
            continue
        out[o] = (values(c), values(z))
    return out, excluded


def diffs(p: dict, metric: str, ids=None) -> list:
    """noisy - clean per question, items where both sides have a value."""
    return [z[metric] - c[metric] for o, (c, z) in p.items()
            if (ids is None or o in ids) and c[metric] is not None and z[metric] is not None]


def did(pa: dict, pb: dict, metric: str) -> list:
    """per question: (noisy - clean) of system a minus (noisy - clean) of system b, items with all four values."""
    out = []
    for o in sorted(set(pa) & set(pb)):
        (ca, za), (cb, zb) = pa[o], pb[o]
        if None not in (ca[metric], za[metric], cb[metric], zb[metric]):
            out.append((za[metric] - ca[metric]) - (zb[metric] - cb[metric]))
    return out


def line(name: str, xs: list, ci=True) -> str:
    if not xs:
        return f"  {name:<34} n=0   -"
    mean = sum(xs) / len(xs)
    if ci:
        lo, hi = rae.boot_ci(xs, random.Random(rae.SEED))
        return f"  {name:<34} n={len(xs):<4} {mean:+8.3f}   [{lo:+.3f}, {hi:+.3f}]"
    c = Counter(round(x) for x in xs)
    return (f"  {name:<34} n={len(xs):<4} {mean:+8.3f}   (n < {MIN_N_CI}: no interval; per question "
            f"worse {c[-1]}, same {c[0]}, better {c[1]})")


def fingerprints(rows: dict) -> list:
    return sorted({str(((r.get("provenance") or {}).get("fingerprint") or {}).get("sha256"))[:12] for r in rows.values()})


def report(clean_dir: Path, noisy_dir: Path, set_path: Path) -> list[str]:
    mapping = json.loads(set_path.read_text(encoding="utf-8"))
    L = [f"Noisy vs clean (eval/noisy_vs_clean.py): {len(mapping)} noisy items vs their Stage 2 originals; "
         f"noisy minus clean per question; 95% bootstrap CI over questions, {rae.N_BOOT} resamples, seed {rae.SEED}; "
         f"metrics as in --summarize; degraded/error/failed rows left out"]
    P, audit = {}, {}
    for s in SYSTEMS:
        cp = clean_dir / f"answer_eval_answer_set_stage2_{s}.jsonl"
        npth = noisy_dir / f"answer_eval_noisy_set_{s}.jsonl"
        clean, c_old = latest(cp)
        noisy, n_old = latest(npth)
        P[s], excluded = pairs(clean, noisy, mapping)
        audit[s] = dict(clean=clean, noisy=noisy, c_old=c_old, n_old=n_old, excluded=excluded, cp=cp, np=npth)

    L += ["", "=== A. Degradation per system: noisy - clean per question (paired on the same item)"]
    for s in SYSTEMS:
        L.append(f"--- {s}: {len(P[s])} items valid on both sides")
        L += [line(m, diffs(P[s], m)) for m in METRICS]

    L += ["", "=== B. PRE-REGISTERED: difference in differences per question, (noisy - clean) of one system minus "
              "(noisy - clean) of the other"]
    for a, b, tag in (("v2", "v1.1", "primary"), ("v2", "v1", "secondary")):
        L.append(f"--- {a} vs {b} ({tag}): {len(set(P[a]) & set(P[b]))} items valid on both sides for both systems")
        L += [line(m, did(P[a], P[b], m)) for m in DID_METRICS]

    L += ["", f"=== C. EXPLORATORY (n about 10 per group, descriptive only): noisy - clean per question by noise_type "
              f"and by language; interval only when n >= {MIN_N_CI}"]
    for field, groups in (("noise_type", ["typos", "wrong_term", "describe", "informal"]), ("language", ["fr", "ar", "en"])):
        for g in groups:
            ids = {n["original_id"] for n in mapping if n[field] == g}
            for s in SYSTEMS:
                k = len(ids & set(P[s]))
                L.append(f"--- {field} {g}, {s}: {k} items")
                for m in METRICS:
                    xs = diffs(P[s], m, ids)
                    L.append(line(m, xs, ci=len(xs) >= MIN_N_CI))

    L += ["", "=== D. Data audit: rows used, items left out and why, false refuse ids, system fingerprints"]
    for s in SYSTEMS:
        a = audit[s]
        L.append(f"--- {s}: clean {a['cp'].name}: {len(a['clean'])} items ({sum(r.get('expected') == 'answer' for r in a['clean'].values())} "
                 f"answerable), {a['c_old']} older rows superseded; noisy {a['np'].name}: {len(a['noisy'])} items, "
                 f"{a['n_old']} older rows superseded; pairs used: {len(P[s])}")
        L.append(f"  left out: {', '.join(f'{o} ({side}: {why})' for o, side, why in a['excluded']) or 'none'}")
    fr = {side: sorted(o for o, (c, z) in P["v2"].items() if (c if side == "clean" else z)["false refuse"] == 1)
          for side in ("clean", "noisy")}
    same = sorted(set(fr["clean"]) & set(fr["noisy"]))
    L.append(f"--- v2 false refuse: clean {fr['clean'] or 'none'}; noisy (original ids) {fr['noisy'] or 'none'}; "
             f"same item on both sides: {same or 'none'}")
    found = {}
    for s in SYSTEMS:
        for side in ("clean", "noisy"):
            found[f"{s} {side}"] = fingerprints(audit[s][side])
    allfp = sorted({f for v in found.values() for f in v})
    L.append("--- system fingerprints (sha256 prefix) per file: " + "; ".join(f"{k} {v}" for k, v in found.items()))
    L.append(f"  all rows share one fingerprint: {allfp[0]}" if len(allfp) == 1 else
             f"  FLAG: more than one fingerprint: {allfp}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clean-dir", default=str(ROOT / "results"))
    ap.add_argument("--noisy-dir", default=str(ROOT / "results" / "noisy"))
    ap.add_argument("--set", default=str(ROOT / "eval" / "noisy_set.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "noisy_vs_clean.txt"))
    args = ap.parse_args(argv)
    text = "\n".join(report(Path(args.clean_dir), Path(args.noisy_dir), Path(args.set))) + "\n"
    print(text, end="")
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
