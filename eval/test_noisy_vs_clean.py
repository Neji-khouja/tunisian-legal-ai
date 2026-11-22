"""Offline tests of eval/noisy_vs_clean.py with stub rows (temp folders; no model, no network).

    python eval/test_noisy_vs_clean.py
"""
import io
import sys
import json
import tempfile
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import noisy_vs_clean as nvc

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   {detail}"))


def row(i, verdict, hit, status="done", expected="answer", refuse=0, hall=None):
    return {"id": i, "status": status, "expected": expected, "seconds": 1.0, "llm_calls": 2,
            "provenance": {"fingerprint": {"sha256": "abc123def456" + "0" * 52}},
            "metrics": {"gold_hit": hit, "gold_cited": None, "false_clarify": False, "false_refuse": bool(refuse),
                        "pdf_numeric_ok": True, "pdf_numeric_claims": []},
            "judge": {"verdict": verdict, "unsupported_statements": []},
            "hallucinated": bool(hall if hall is not None else verdict == "incorrect")}


def write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def setup(tmp: Path):
    (tmp / "clean").mkdir()
    (tmp / "noisy").mkdir()
    ids = ["o1", "o2", "o3", "o4", "o5"]
    mapping = [{"id": f"n_{o}", "original_id": o, "noise_type": t, "language": "fr"}
               for o, t in zip(ids, ["typos", "wrong_term", "describe", "informal", "typos"])]
    (tmp / "noisy_set.json").write_text(json.dumps(mapping), encoding="utf-8")
    base_clean = {"o1": ("incorrect", 1), "o2": ("correct", 1), "o3": ("correct", 1), "o4": ("correct", 1),
                  "o5": ("correct", 1)}
    base_noisy = {"o1": ("correct", 1), "o2": ("correct", 1), "o3": ("correct", 1), "o4": ("correct", 1),
                  "o5": ("correct", 1)}
    for s in ("v1", "v1.1"):
        write(tmp / "clean" / f"answer_eval_answer_set_stage2_{s}.jsonl",
              [row(o, *base_clean[o]) for o in ids] + [row("c9", None, None, expected="refuse")])
        write(tmp / "noisy" / f"answer_eval_noisy_set_{s}.jsonl", [row(f"n_{o}", *base_noisy[o]) for o in ids])
    write(tmp / "clean" / "answer_eval_answer_set_stage2_v2.jsonl",
          [row("o1", "correct", 1), row("o2", "correct", 1), row("o3", "correct", 1),
           row("o4", None, 1, status="degraded"), row("o5", "correct", 1, refuse=1)])
    write(tmp / "noisy" / "answer_eval_noisy_set_v2.jsonl",
          [row("n_o1", "incorrect", 0), row("n_o3", None, None, status="error"),
           row("n_o4", "correct", 1), row("n_o5", "correct", 1),
           dict(row("n_o5", "incorrect", 1, refuse=1), rerun=True, supersedes={"status": "done"})])


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup(tmp)
        out = tmp / "out.txt"
        with contextlib.redirect_stdout(io.StringIO()):
            nvc.main(["--clean-dir", str(tmp / "clean"), "--noisy-dir", str(tmp / "noisy"),
                      "--set", str(tmp / "noisy_set.json"), "--out", str(out)])
        text = out.read_text(encoding="utf-8")
        lines = text.splitlines()

        def block(head):
            k = next(i for i, l in enumerate(lines) if l.startswith(head))
            return lines[k:k + 8]

        mapping = json.loads((tmp / "noisy_set.json").read_text(encoding="utf-8"))
        clean, _ = nvc.latest(tmp / "clean" / "answer_eval_answer_set_stage2_v2.jsonl")
        noisy, n_old = nvc.latest(tmp / "noisy" / "answer_eval_noisy_set_v2.jsonl")
        p2, excl = nvc.pairs(clean, noisy, mapping)
        check("missing twin, error row and degraded row are left out, with the side and the reason",
              sorted(p2) == ["o1", "o5"] and sorted(excl) == [("o2", "noisy", "no row"), ("o3", "noisy", "status error"),
                                                             ("o4", "clean", "status degraded")], f"{sorted(p2)} {excl}")
        check("superseded row: the later row wins and is counted as one replaced row",
              p2["o5"][1]["correct"] == 0 and n_old == 1, f"{p2['o5']} {n_old}")
        c11, _ = nvc.latest(tmp / "clean" / "answer_eval_answer_set_stage2_v1.1.jsonl")
        z11, _ = nvc.latest(tmp / "noisy" / "answer_eval_noisy_set_v1.1.jsonl")
        p11, _ = nvc.pairs(c11, z11, mapping)
        dc, dh = nvc.did(p2, p11, "correct"), nvc.did(p2, p11, "Hit@6 (gold in evidence)")
        check("difference in differences, hand-computed: correct [-2, -1] (mean -1.5), Hit@6 [-1, 0] (mean -0.5)",
              dc == [-2, -1] and dh == [-1, 0], f"{dc} {dh}")
        check("degradation per system (A): v1.1 correct +1 on o1 only -> mean +0.200 over 5 items; v2 correct -1.000 on 2",
              any(l.startswith("  correct") and "n=5" in l and "+0.200" in l for l in block("--- v1.1:"))
              and any(l.startswith("  correct") and "n=2" in l and "-1.000" in l for l in block("--- v2:")),
              "\n".join(block("--- v1.1:") + block("--- v2:")))
        b = block("--- v2 vs v1.1 (primary)")
        check("B in the file: correct -1.500, Hit@6 -0.500, n=2",
              any(l.startswith("  correct") and "n=2" in l and "-1.500" in l for l in b)
              and any(l.startswith("  Hit@6") and "n=2" in l and "-0.500" in l for l in b), "\n".join(b))
        check("C: a group of n < 8 has no interval, only counts",
              "(n < 8: no interval; per question worse" in text and "PRE-REGISTERED" in text and "EXPLORATORY" in text)
        check("D: the audit names each left-out item, the v2 false refuse ids and one fingerprint",
              "o2 (noisy: no row)" in text and "o4 (clean: status degraded)" in text
              and "v2 false refuse: clean ['o5']; noisy (original ids) ['o5']; same item on both sides: ['o5']" in text
              and "all rows share one fingerprint: abc123def456" in text, text[-800:])
        check("D: a clean clarify/refuse item outside the noisy set is not used", "c9" not in text)
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed: {'all checks passed' if not n_fail else 'FAILURES'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
