"""Read-only: extract example questions from the agent prompts, compare gate/candidate (and extra) questions to them.
    python eval/prompt_overlap.py [--extra "question|lang" ...]
"""
import re, sys, json, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import analyze, plan, grade, reformulate, verify, answer, condense, codes

PROMPTS = {"analyze.py": analyze.SYSTEM_PROMPT, "plan.py": plan.PLAN_PROMPT, "codes.py": codes.codes_block(),
           "grade.py": grade.GRADE_PROMPT, "reformulate.py": reformulate.REFORMULATE_PROMPT,
           "verify.py": verify.SYSTEM_PROMPT, "answer.py": answer.SYSTEM_PROMPT, "condense.py": condense.CONDENSE_PROMPT}
STOP = {
 "en": set("a an the of to in on for at by with from and or but if is are was be been do does did i my me you your he she it "
           "its we our they them their this that what which who whom how when where why can could may might must should "
           "would will have has had not no any some there here as about into than then so get".split()),
 "fr": set("le la les l un une des de du d et ou mais si est sont etre ai a as avons ont je j me m mon ma mes tu te ton ta tes "
           "il elle on nous vous ils elles se s son sa ses leur leurs ce cet cette ces qui que qu quoi quel quelle quels quelles "
           "comment quand ou pourquoi dans en sur pour par avec sans au aux ne pas plus y faut peut puis-je dois doit".split()),
 "ar": set("ما ماذا هل من في على إلى الى عن مع هو هي أن ان كم كيف متى أين لماذا التي الذي هذا هذه ذلك تلك أو او و ثم لا "
           "لم لن قد كان يكون يجب يمكن أي اي عند بعد قبل".split()),
}


def lang_of(q):
    if re.search(r"[\u0600-\u06FF]", q):
        return "ar"
    toks = set(re.findall(r"\w+", q.lower()))
    return "fr" if toks & {"le", "la", "les", "de", "du", "est", "quelle", "mon", "je", "pour", "que", "une", "des"} else "en"


def words(q, lang):
    q = unicodedata.normalize("NFKD", q.lower().replace("’", "'"))
    q = "".join(c for c in q if not unicodedata.combining(c)) if lang != "ar" else q
    out = set()
    for t in re.findall(r"\w+", q.replace("'", " ")):
        if t in STOP[lang] or len(t) < 2:
            continue
        if lang != "ar" and len(t) > 3 and t.endswith("s"):
            t = t[:-1]                                   # light plural folding
        out.add(t)
    return out


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def extract():
    ex = []
    for f, p in PROMPTS.items():
        found = []
        for line in p.splitlines():                       # Question: "..." / User: ... / New message: ...
            m = re.match(r'\s*(Question|User|New message):\s*"?(.+?)"?\s*$', line)
            if m:
                found.append(m.group(2))
        flat = " ".join(p.split())                        # quotes may span a line break
        found += [q for q in re.findall(r'"([^"]*)"', flat) if re.search(r"[?؟]\s*$", q)]   # quoted questions
        for q in dict.fromkeys(found):
            ex.append({"file": f, "question": q, "language": lang_of(q)})
    return ex


def load_set(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    ex = extract()
    (ROOT / "results" / "prompt_examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(ex)} prompt examples -> results/prompt_examples.json")
    for e in ex:
        print(f"  {e['file']:<15} {e['language']}  {e['question']}")
    items = [(it["id"], it["question"], it["language"]) for it in load_set("eval/gate_set.json")]
    items += [(it["id"], it["question"], it["language"]) for it in load_set("eval/answer_set_candidates.json")]
    extra = sys.argv[sys.argv.index("--extra") + 1:] if "--extra" in sys.argv else []
    items += [(f"new{k}", s.split("|")[0], s.split("|")[1]) for k, s in enumerate(extra, 1)]
    print("\nPairs with content-word Jaccard >= 0.4 (same language):")
    n = 0
    for iid, q, lang in items:
        for e in ex:
            if e["language"] == lang:
                j = jac(words(q, lang), words(e["question"], lang))
                if j >= 0.4:
                    n += 1
                    print(f"  {j:.2f}  {iid} {q!r}  <->  {e['file']} {e['question']!r}")
    print(f"  ({n} pairs)")
    for iid, q, lang in items:
        if iid == "g05" or iid.startswith("new"):
            close = sorted(((jac(words(q, lang), words(e["question"], lang)), e) for e in ex if e["language"] == lang),
                           key=lambda x: -x[0])[:3]
            print(f"\n3 closest prompt examples to {iid} {q!r}  words {sorted(words(q, lang))}:")
            for j, e in close:
                print(f"  {j:.2f}  {e['file']} {e['question']!r}  words {sorted(words(e['question'], lang))}")
