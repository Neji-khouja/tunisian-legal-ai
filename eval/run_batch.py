"""Run an exam stage in batches of K questions, one process per system (v1, then v1.1, then v2): this machine's RAM does
not hold the v1 process and the v1.1/v2 models in one process.

    python eval/run_batch.py --stage 1 --judge-provider gemini --judge-model <id> --v1-worktree ../RAG_PDF_v1 --dry-run
    python eval/run_batch.py --stage 1 [--questions-per-run 10] --judge-provider gemini --judge-model <id> \
           --v1-worktree ../RAG_PDF_v1 [--min-gap 3] [--daily-budget-tokens N] [other run_answer_eval.py flags]

Picks the next K question ids of the stage (exam order, question-major) that lack a finished row in at least one system,
then runs eval/run_answer_eval.py --stage S --system X --only-ids <the K ids> for X = v1, v1.1, v2, one after another
(a system whose rows for these ids are all finished is skipped). Exit code 75 (daily quota or token budget) is
forwarded at once and the batch stops; any other failure is reported and the next system runs (exit code 1 at the
end). Flags not listed here are passed to every run. Prints done/remaining per system and the token projection.
--loop: batch after batch, one progress line each, until exit 75 (stop, exit 75), nothing left (exit 0), or two
consecutive failed batches (exit 1). A batch fails when a system exits with a code other than 0 and 75, or when it
finishes no row at all (e.g. the same items degraded again), so a loop cannot spin. --dry-run shows one batch only.
"""
import sys
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import harness_core as hc
import run_answer_eval as rae

RUN = subprocess.run                     # tests replace it


def stage_items(set_path: Path, stages_path: Path, stage: str) -> list[dict]:
    ids = rae.stage_ids(stages_path, stage)
    return [i for i in json.loads(set_path.read_text(encoding="utf-8")) if i["id"] in ids]


def result_files(args) -> dict:
    return {s: hc.ResultFile(rae.out_path(Path(args.set), s, args.stage, Path(args.results_dir))) for s in rae.SYSTEM_ORDER}


def progress(items, files) -> dict:
    ids = {i["id"] for i in items}
    todo = {s: len(ids - f.finished()) for s, f in files.items()}
    print(f"\nstage progress ({len(ids)} questions):")
    print(f"  {'system':<6} {'done':>5} {'remaining':>10}")
    for s, f in files.items():
        print(f"  {s:<6} {len(ids & f.finished()):>5} {todo[s]:>10}")
    rae.projection(files, todo)
    return todo


def run_one_batch(args, rest, items) -> tuple[str, list, list]:
    """-> (outcome "nothing" | "ok" | "failed", batch ids, failures). Exit code 75 of a run: sys.exit(75) at once."""
    files = result_files(args)
    batch = [i["id"] for i in items if any(i["id"] not in f.finished() for f in files.values())][:args.questions_per_run]
    if not batch:
        return "nothing", [], []
    print(f"\nbatch of {len(batch)} question(s): {','.join(batch)}")
    failures = []
    for s in rae.SYSTEM_ORDER:
        if set(batch) <= files[s].finished():
            print(f"\n--- {s}: every question of the batch is finished; skipped")
            continue
        cmd = [sys.executable, str(ROOT / "eval" / "run_answer_eval.py"), "--set", args.set, "--stages-file",
               args.stages_file, "--results-dir", args.results_dir, "--stage", args.stage, "--system", s,
               "--only-ids", ",".join(batch), "--judge-provider", args.judge_provider, *rest] + \
              (["--dry-run"] if args.dry_run else [])
        print(f"\n--- {s}: " + " ".join(cmd), flush=True)
        code = RUN(cmd, cwd=ROOT).returncode
        if code == hc.EXIT_QUOTA:
            print(f"\nSTOPPED: {s} exited with {hc.EXIT_QUOTA} (daily quota or token budget); the batch stops here. "
                  f"Re-run the same command later to continue.", file=sys.stderr)
            sys.exit(hc.EXIT_QUOTA)
        if code:
            failures.append((s, code))
            print(f"\n{s} FAILED (exit code {code}); going on to the next system.", file=sys.stderr)
    done_before = sum(len(set(batch) & f.finished()) for f in files.values())
    done_after = sum(len(set(batch) & f.finished()) for f in result_files(args).values())
    if args.loop and not args.dry_run and not failures and done_after == done_before:
        failures.append(("all", "no row finished"))
    return ("failed" if failures else "ok"), batch, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["1", "2"], required=True)
    ap.add_argument("--questions-per-run", type=int, default=10, metavar="K")
    ap.add_argument("--judge-provider", choices=["groq", "gemini"], default=hc.env_judge_provider())
    ap.add_argument("--set", default=str(rae.EXAM_DIR / "answer_set.json"))
    ap.add_argument("--stages-file", default=str(rae.EXAM_DIR / "stages.json"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--dry-run", action="store_true", help="every run with --dry-run (no LLM call)")
    ap.add_argument("--loop", action="store_true", help="repeat batches until quota (75), done (0) or 2 failed batches (1)")
    args, rest = ap.parse_known_args(argv)
    clash = [a for a in rest if a.split("=")[0] in ("--system", "--only-ids", "--limit", "--judge-only", "--summarize")]
    if clash:
        sys.exit(f"run_batch.py sets these itself or does not batch them: {clash}")
    items = stage_items(Path(args.set), Path(args.stages_file), args.stage)
    progress(items, result_files(args))
    n, failed_in_a_row = 0, 0
    while True:
        outcome, batch, failures = run_one_batch(args, rest, items)
        if outcome == "nothing":
            print("\nnothing left to run in this stage.")
            return
        n += 1
        ids = {i["id"] for i in items}
        done = ", ".join(f"{s} {len(ids & f.finished())}/{len(ids)}" for s, f in result_files(args).items())
        print(f"\nBATCH {n} {outcome.upper()} ({batch[0]}..{batch[-1]}, {len(batch)} questions); done {done}"
              + (f"; failures: {', '.join(f'{s} ({c})' for s, c in failures)}" if failures else ""), flush=True)
        failed_in_a_row = failed_in_a_row + 1 if outcome == "failed" else 0
        if not args.loop or args.dry_run:
            progress(items, result_files(args))
            if failures:
                sys.exit(1)
            return
        if failed_in_a_row == 2:
            progress(items, result_files(args))
            print("\nSTOPPED: two consecutive batches failed (not a quota stop).", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
