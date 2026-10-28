"""Regenerates the report inputs from files, checks the LaTeX sources, and compiles if a LaTeX engine exists.

    python report/build_report.py

1. make_numbers.py -> numbers.tex, tables/*.tex, data.json        (this interpreter)
2. make_figures.py -> figures/*.pdf                                 (this interpreter if it has matplotlib,
                                                                     otherwise the one in env FIG_PYTHON)
3. static checks: macros defined, \\input files present, \\ref labels defined, braces and environments balanced,
   no Arabic script, no local paths; lists every "à compléter" and "[à vérifier]"
4. pdflatex/latexmk twice if available; otherwise says so (the folder compiles on Overleaf as is)
"""
import os
import re
import sys
import shutil
import importlib.util
import subprocess
from pathlib import Path

REP = Path(__file__).resolve().parent
ROOT = REP.parent


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=kw.pop("cwd", ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out)
    return r.returncode


def tex_files():
    return [REP / "main.tex"] + sorted((REP / "sections").glob("*.tex"))


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def checks() -> int:
    problems = []
    defined = set(re.findall(r"\\newcommand\{\\(N\w+)\}", (REP / "numbers.tex").read_text(encoding="utf-8")))
    labels, refs = set(), set()
    all_tex = tex_files() + sorted((REP / "tables").glob("*.tex"))
    for f in all_tex:
        s = strip_comments(f.read_text(encoding="utf-8"))
        rel = f.relative_to(REP)
        for m in set(re.findall(r"\\(N[A-Z]\w*)", s)) - defined:
            problems.append(f"{rel}: macro \\{m} is not defined in numbers.tex")
        for inp in re.findall(r"\\input\{([^}]+)\}", s):
            p = REP / (inp if inp.endswith(".tex") else inp + ".tex")
            if not p.exists():
                problems.append(f"{rel}: \\input{{{inp}}} not found")
        labels |= set(re.findall(r"\\label\{([^}]+)\}", s))
        labels |= set(re.findall(r"\\figfile\{[^}]*\}\{.*?\}\{(fig:[^}]+)\}", s, re.S))     # label = 3rd argument
        refs |= set(re.findall(r"\\ref\{([^}]+)\}", s))
        body = s.replace("\\\\", "").replace(r"\{", "").replace(r"\}", "")              # "\\{..." is a line break + group
        if body.count("{") != body.count("}"):
            problems.append(f"{rel}: unbalanced braces ({body.count('{')} open, {body.count('}')} close)")
        for env in set(re.findall(r"\\begin\{(\w+\*?)\}", s)) | set(re.findall(r"\\end\{(\w+\*?)\}", s)):
            b, e = s.count(f"\\begin{{{env}}}"), s.count(f"\\end{{{env}}}")
            if b != e:
                problems.append(f"{rel}: environment {env} begin {b} / end {e}")
        if re.search(r"[\u0600-\u06FF]", s):
            problems.append(f"{rel}: contains Arabic script")
        if re.search(r"[A-Za-z]:\\\\?Users|/Users/|AppData", s):
            problems.append(f"{rel}: contains a local path")
        if re.search(r"gsk_[A-Za-z0-9]{10,}|api[_-]?key\s*=", s, re.I):
            problems.append(f"{rel}: looks like an API key")
    for r in sorted(refs - labels):
        problems.append(f"\\ref{{{r}}} has no \\label")
    for fig in set(re.findall(r"\\figfile\{([^}]+)\}", "".join(f.read_text(encoding="utf-8") for f in tex_files()))):
        if not (REP / "figures" / f"{fig}.pdf").exists():
            print(f"  note: figures/{fig}.pdf absent -> 'Figure à générer' box")
    print(f"\nstatic checks: {len(problems)} problem(s)")
    for p in problems:
        print("  -", p)
    # what is still open
    num = (REP / "numbers.tex").read_text(encoding="utf-8")
    missing = sorted(re.findall(r"\\newcommand\{\\(N\w+)\}\{\\textit\{à compléter\}\}", num))
    used_missing = []
    for f in all_tex:
        s = strip_comments(f.read_text(encoding="utf-8"))
        used_missing += [f"{f.relative_to(REP)}: \\{m}" for m in missing if re.search(rf"\\{m}(?![A-Za-z])", s)]
        if r"\acompleter" in s and f.name != "main.tex":
            used_missing.append(f"{f.relative_to(REP)}: \\acompleter (text)")
        if "à compléter" in s and f.parent.name == "tables":
            used_missing.append(f"{f.relative_to(REP)}: table/listing to complete")
    print(f"\n'à compléter' in the document ({len(used_missing)}):")
    for u in used_missing:
        print("  -", u)
    verif = []
    for f in tex_files():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if r"\averifier" in line and "newcommand" not in line:
                verif.append(f"{f.relative_to(REP)}:{i}: {line.strip()[:110]}")
    print(f"\n'[à vérifier]' in the document ({len(verif)}):")
    for v in verif:
        print("  -", v)
    return len(problems)


def main():
    if run([sys.executable, str(REP / "make_numbers.py")]):
        sys.exit("make_numbers.py failed")
    fig_py = sys.executable if importlib.util.find_spec("matplotlib") else os.environ.get("FIG_PYTHON")
    if fig_py:
        run([fig_py, str(REP / "make_figures.py")])
    else:
        print("matplotlib not available here and FIG_PYTHON not set: figures not regenerated "
              "(existing figures/*.pdf are kept; missing ones show 'Figure à générer').")
    n = checks()
    engine = shutil.which("latexmk") or shutil.which("pdflatex")
    if not engine:
        print("\nNo pdflatex/latexmk found: not compiled here. The report/ folder is meant to compile on Overleaf as is "
              "(main.tex, pdflatex).")
        sys.exit(1 if n else 0)
    for _ in range(2):
        cmd = [engine, "-pdf", "-interaction=nonstopmode", "main.tex"] if "latexmk" in engine else \
              [engine, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        if run(cmd, cwd=REP):
            sys.exit("LaTeX compilation failed (see main.log)")
    print("compiled report/main.pdf")


if __name__ == "__main__":
    main()
