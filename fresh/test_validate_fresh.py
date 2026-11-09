"""Stub tests of fresh/validate_fresh.py (no model: a bag-of-words embedding; temporary sqlite indexes).

    python fresh/test_validate_fresh.py
"""
import sys
import json
import sqlite3
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import validate_fresh as V

FAILS, CNT = [], {"n": 0}


def check(name, ok, detail=""):
    CNT["n"] += 1
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"   {detail}"))
    if not ok:
        FAILS.append(name)


def bow_embed(texts):
    vocab = {}
    toks = [V.norm_text(t).split() for t in texts]
    for ts in toks:
        for w in ts:
            vocab.setdefault(w, len(vocab))
    out = []
    for ts in toks:
        v = [0.0] * max(1, len(vocab))
        for w in ts:
            v[vocab[w]] += 1
        out.append(v)
    return out


def shared_embed(all_texts):
    """Bag-of-words vectors over one fixed vocabulary (all texts known up front)."""
    vocab = {}
    for s in all_texts:
        for w in V.norm_text(s).split():
            vocab.setdefault(w, len(vocab))

    def embed(texts):
        out = []
        for s in texts:
            v = [0.0] * len(vocab)
            for w in V.norm_text(s).split():
                v[vocab[w]] += 1
            out.append(v)
        return out
    return embed


def entry(**kw):
    e = {"id": "F001", "language": "fr", "type": "clear", "question": "Quelle est la peine du faux témoignage ?",
         "gold_articles": ["PenalFr/s02/243"], "gold_notes": "", "labeller": "NK", "labelled_on": "2026-11-20",
         "anonymised": True}
    return e | kw


IDS = {"db_v1_2": {"PenalFr/s02/243", "COC/s02/94", "PenalFr/s02/240-bis"}, "db_v1_1": {"PenalFr/s02/243", "COC/s02/94"}}


def errors_of(e):
    return V.check_entry(e, IDS)[0]


def make_db(rows, with_text=False):
    d = Path(tempfile.mkdtemp())
    con = sqlite3.connect(d / "chroma.sqlite3")
    con.execute("create table embedding_metadata (id integer, key text, string_value text)")
    for i, (cid, doc, text) in enumerate(rows):
        con.execute("insert into embedding_metadata values (?, 'chunk_id', ?)", (i, cid))
        con.execute("insert into embedding_metadata values (?, 'chroma:document', ?)", (i, doc))
        if with_text:
            con.execute("insert into embedding_metadata values (?, 'article_text', ?)", (i, text))
    con.commit()
    con.close()
    return d


def main():
    check("valid entry: no error", errors_of(entry()) == [])
    check("missing field", any("missing field gold_notes" in m for m in errors_of({k: v for k, v in entry().items() if k != "gold_notes"})))
    check("language not fr/en/ar", any("language" in m for m in errors_of(entry(language="de"))))
    check("unknown type", any("type" in m for m in errors_of(entry(type="other"))))
    check("empty question", any("question" in m for m in errors_of(entry(question="  "))))
    check("out_of_scope with gold -> error", any("out_of_scope" in m for m in errors_of(entry(type="out_of_scope"))))
    check("out_of_scope without gold -> ok", errors_of(entry(type="out_of_scope", gold_articles=[])) == [])
    check("clear without gold -> error", any("at least one gold" in m for m in errors_of(entry(gold_articles=[]))))
    check("vague without clarification note -> error", any("clarification" in m for m in errors_of(entry(type="vague"))))
    check("vague with a note -> ok", errors_of(entry(type="vague", gold_notes="civil ou pénal ?")) == [])
    check("gold id format", any("format" in m for m in errors_of(entry(gold_articles=["Penal 243"]))))
    check("gold id missing from db_v1_1 only", errors_of(entry(gold_articles=["PenalFr/s02/240-bis"])) == ["gold id PenalFr/s02/240-bis not in db_v1_1"])
    check("gold id missing from both indexes", len(errors_of(entry(gold_articles=["PenalFr/s02/999"]))) == 2)
    check("duplicate gold article", any("duplicate gold" in m for m in errors_of(entry(gold_articles=["COC/s02/94", "COC/s02/94"]))))
    check("anonymised must be true", any("anonymised" in m for m in errors_of(entry(anonymised=False))))
    check("anonymised as a string is refused", any("anonymised" in m for m in errors_of(entry(anonymised="true"))))
    check("labelled_on date format", any("labelled_on" in m for m in errors_of(entry(labelled_on="20/11/2026"))))
    check("empty labeller", any("labeller" in m for m in errors_of(entry(labeller=""))))

    es = [entry(), entry(id="F001", question="autre question sur le divorce"), entry(id="F003", question="QUELLE est la peine du  faux témoignage?"),
          entry(id="EX", example=True)]
    r = V.validate(es, IDS)
    check("duplicate ids reported", r["duplicate_ids"] == ["F001"])
    check("same question (case, spaces, punctuation) reported", r["duplicate_questions"] == [["F001", "F003"]], r["duplicate_questions"])
    check("examples ignored and reported", r["examples_ignored"] == ["EX"] and r["entries"] == 3)
    check("errors counted -> exit code 1 condition", r["errors"] >= 2)

    refs = [("answer_set.json", "ac01", "Quelle peine pour le vol simple commis la nuit ?"),
            ("noisy_set.json", "n_ac01", "jai signe un contrat d arbitrage international"),
            ("prompt normalize.py", "question", "un temoin a raconté des mensonges a l'audiance")]
    fresh = [entry(id="F1", question="Quelle peine pour le vol simple commis la nuit ?"),
             entry(id="F2", question="un temoin a raconté des mensonges a l'audiance, il risque quoi"),
             entry(id="F3", question="succession du conjoint survivant partage"),
             entry(id="F4", question="succession du conjoint survivant partage des biens")]
    nd = V.near_duplicates(fresh, refs, shared_embed([r[2] for r in refs] + [e['question'] for e in fresh]), 0.68)
    hits = {(i, src) for i, s, src, rid, t in nd["reference"]}
    check("near-duplicate of an exam question flagged with its source", ("F1", "answer_set.json") in hits, nd["reference"])
    check("near-duplicate of a normalization-prompt example flagged", ("F2", "prompt normalize.py") in hits)
    check("unrelated question not flagged", not any(i == "F3" for i, *_ in nd["reference"]))
    check("near-duplicates within the file", [(a, b) for a, b, s in nd["within"]] == [("F3", "F4")], nd["within"])
    check("threshold respected", all(s >= 0.68 for _, s, *_ in nd["reference"]))
    check("default threshold = calibrated 0.68", V.NEAR_DUP_THRESHOLD == 0.68)

    many = [entry(id=f"A{k}", language=("fr", "en", "ar")[k % 3], question=f"q {k}") for k in range(150)]
    r = V.validate(many, IDS)
    check("minimums: 150 total, 50 per language -> met", all(m["met"] for m in r["minimums"].values()), r["minimums"])
    few = [entry(id=f"B{k}", language="fr" if k < 140 else "ar", question=f"q {k}") for k in range(150)]
    r = V.validate(few, IDS)
    check("minimums: a language under 30 -> not met (warning, not error)",
          not r["minimums"]["language ar"]["met"] and not r["minimums"]["language en"]["met"] and r["minimums"]["total"]["met"]
          and r["errors"] == 0 and r["warnings"] >= 2)
    check("counts per language and type", r["by_language"] == {"fr": 140, "en": 0, "ar": 10} and r["by_type"]["clear"] == 150)

    d = make_db([("PenalFr/s02/243", "Code pénal > Titre ...\nArticle 243.- Est puni", "Article 243.- Est puni"),
                 ("PenalFr/s02/243", "dup sub-chunk", "Article 243.- Est puni"), ("COC/s02/94", "Article 94.- Chacun", "Article 94.- Chacun"),
                 ("PenalFr/s03/243", "Article 243 annexe", "Article 243 annexe")], with_text=True)
    check("index ids read from sqlite", V.index_ids(d) == {"PenalFr/s02/243", "COC/s02/94", "PenalFr/s03/243"})
    lk = V.lookup("PenalFr", "243", d)
    check("lookup: every section with this label, article text (not the header)",
          [c for c, _ in lk] == ["PenalFr/s02/243", "PenalFr/s03/243"] and lk[0][1].startswith("Article 243"), lk)
    d1 = make_db([("COC/s02/94", "Article 94.- Chacun", None)])
    check("lookup: v1.1 index (no article_text) uses the document", V.lookup("COC", "94", d1) == [("COC/s02/94", "Article 94.- Chacun")])
    before = (d / "chroma.sqlite3").read_bytes()
    V.index_ids(d), V.lookup("COC", "94", d)
    check("index read-only: file bytes unchanged", (d / "chroma.sqlite3").read_bytes() == before)

    check("threshold rule: largest step flagging >= 90% of positives",
          V.near_dup_rule([0.95] * 9 + [0.5], [0.3]) == 0.95 and V.near_dup_rule([0.9, 0.8, 0.7, 0.6, 0.85, 0.82, 0.81, 0.83, 0.84, 0.86], []) == 0.7)

    tpl = V.load_entries(HERE / "questions_template.json")
    real_ids = {n: V.index_ids(p) for n, p in V.INDEXES.items() if (p / "chroma.sqlite3").exists()}
    r = V.validate(tpl, real_ids)
    check("template: 3 examples, ignored, 0 errors", r["examples_ignored"] == ["EXAMPLE-1", "EXAMPLE-2", "EXAMPLE-3"] and r["errors"] == 0)
    r = V.validate([dict(e, example=False) for e in tpl], real_ids)
    check("template examples are valid entries (gold ids in both indexes)", r["errors"] == 0 and len(real_ids) == 2, r["per_entry"])
    check("template examples: one per type clear/messy/vague, three languages",
          sorted(e["type"] for e in tpl) == ["clear", "messy", "vague"] and sorted(e["language"] for e in tpl) == ["ar", "en", "fr"])
    print(f"\n{CNT['n'] - len(FAILS)}/{CNT['n']} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
