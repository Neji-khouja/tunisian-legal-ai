"""Draws report/figures/*.pdf from report/data.json (written by make_numbers.py). Needs matplotlib only.
A figure whose data is missing is not drawn; the LaTeX shows "Figure à générer" instead.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = Path(__file__).resolve().parent
FIG = REP / "figures"
DATA = json.loads((REP / "data.json").read_text(encoding="utf-8"))
W = 14 / 2.54                                     # 14 cm
# Okabe-Ito (colour-blind safe); one colour per variant, used in every figure
C = {"v1": "#999999", "v11": "#0072B2", "A2": "#0072B2", "D": "#E69F00", "G": "#009E73",
     "Boucle depuis A2": "#56B4E9", "Boucle depuis G": "#CC79A7", "A1 (oracle)": "#555555",
     "good": "#009E73", "number_changed": "#E69F00", "wrong_article": "#D55E00", "added_condition": "#CC79A7"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def fr(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def labels(ax, bars, vals, nd=1, suffix="", inside=False):
    """Value labels: on top of the bar, or inside at its base (when error bars would overlap the label)."""
    for b, v in zip(bars, vals):
        if v is not None:
            x = b.get_x() + b.get_width() / 2
            if inside:
                ax.text(x, 2, fr(v, nd) + suffix, ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
            else:
                ax.text(x, b.get_height(), fr(v, nd) + suffix, ha="center", va="bottom", fontsize=8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", f"figures/{name}.pdf")


def f2():
    d = DATA.get("f2") or {}
    if not (d.get("v1") and d.get("v11")):
        return
    metrics = [("hit", "Hit@6"), ("hit1", "Hit@1"), ("recall", "Rappel"), ("mrr", "MRR (×100)")]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    x = range(len(metrics))
    for k, (sys_, lab) in enumerate((("v1", "v1"), ("v11", "v1.1 (A1)"))):
        vals = [d[sys_][m] * 100 for m, _ in metrics]
        bars = ax.bar([i + (k - 0.5) * 0.38 for i in x], vals, 0.38, color=C[sys_], label=lab)
        labels(ax, bars, vals)
    ax.set_xticks(list(x), [l for _, l in metrics]); ax.set_ylabel("%"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right"); ax.set_title(f"Jeu de test ({d['v11']['n']} questions)", fontsize=9)
    save(fig, "f2_v1_vs_v11")


def f3():
    d = DATA.get("f3")
    if not d or d["test"].get("A2") is None:
        return
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for k, v in enumerate(("A2", "D")):
        vals = [d["dev"][v] * 100, d["test"][v] * 100]
        bars = ax.bar([0 + (k - 0.5) * 0.38, 1 + (k - 0.5) * 0.38], vals, 0.38, color=C[v],
                      label="A2 (question brute)" if v == "A2" else "D (plan, filtre doux)")
        labels(ax, bars, vals)
    ax.set_xticks([0, 1], [f"dev ({d['n']['dev']} questions)", f"test ({d['n']['test']} questions)"])
    ax.set_ylabel("Hit@6 (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "f3_dev_vs_test")


def f4():
    d = DATA.get("f4")
    if not d:
        return
    names = list(d)
    fig, ax = plt.subplots(figsize=(W, 2.8))
    vals = [d[n]["hit"] * 100 for n in names]
    err = [[(d[n]["hit"] - d[n]["ci"][0]) * 100 for n in names], [(d[n]["ci"][1] - d[n]["hit"]) * 100 for n in names]]
    bars = ax.bar(range(len(names)), vals, 0.6, color=[C.get(n, C["A2"] if n == "A2" else "#777") for n in names],
                  yerr=err, capsize=3, ecolor="#333")
    labels(ax, bars, vals, inside=True)
    ax.set_xticks(range(len(names)), [n.replace(" depuis ", "\ndepuis ") for n in names], fontsize=8)
    ax.set_ylabel("Hit@6 (%)"); ax.set_ylim(0, 105)
    ax.set_title("Dev ; IC 95 % par bootstrap sur les concepts", fontsize=9)
    save(fig, "f4_ablation")


def f5():
    d = DATA.get("f5")
    if not d:
        return
    names, langs = list(d["by_lang"]), ("fr", "ar", "en")
    fig, ax = plt.subplots(figsize=(W, 2.8))
    w = 0.8 / len(names)
    for k, n in enumerate(names):
        vals = [d["by_lang"][n][l] * 100 for l in langs]
        bars = ax.bar([i + (k - (len(names) - 1) / 2) * w for i in range(3)], vals, w, color=C.get(n, "#777"), label=n)
        labels(ax, bars, vals, nd=0)
    ax.set_xticks(range(3), [f"{l.upper()} (n={d['n'][l]})" for l in langs]); ax.set_ylabel("Hit@6 (%)"); ax.set_ylim(0, 115)
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper center")
    save(fig, "f5_by_language")


def f6():
    d = DATA.get("f6")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4))
    for ax, start in zip(axes, ("A2", "G")):
        g = d[start]
        m = [[g["enough|True"], g["enough|False"]], [g["weak|True"], g["weak|False"]]]
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i][j]), ha="center", va="center", fontsize=10,
                        color="white" if m[i][j] > max(max(r) for r in m) / 2 else "black")
        ax.set_xticks([0, 1], ["réf. trouvée", "réf. absente"]); ax.set_yticks([0, 1], ["suffisant", "faible"])
        ax.set_title(f"Départ {start}", fontsize=9)
    fig.supxlabel("Tour 0 : article de référence dans le top 6", fontsize=8)
    save(fig, "f6_grade_confusion")


def f7():
    d = DATA.get("f7")
    if not d:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.5))
    for k, start in enumerate(("A2", "G")):
        r = d[start]["retries"]
        vals = [r.get(str(i), r.get(i, 0)) for i in range(3)]
        bars = a1.bar([i + (k - 0.5) * 0.38 for i in range(3)], vals, 0.38, color=C[start], label=f"départ {start}")
        labels(a1, bars, vals, nd=0)
        reqs = d[start]["llm_requests"]
        counts = [reqs.count(i) for i in range(max(reqs) + 1)]
        bars = a2.bar([i + (k - 0.5) * 0.38 for i in range(len(counts))], counts, 0.38, color=C[start], label=f"départ {start}")
        labels(a2, bars, counts, nd=0)
    a1.set_xticks(range(3), ["0", "1", "2"]); a1.set_xlabel("reprises"); a1.set_ylabel("questions")
    a2.set_xlabel("appels LLM (Évaluer + Reformuler)"); a1.legend(frameon=False, fontsize=7)
    save(fig, "f7_retries_calls")


def f8():
    d = DATA.get("f8")
    if not d:
        return
    names = list(d)
    lab = {"good": "correcte\n(fausse alarme)", "number_changed": "nombre\nmodifié", "wrong_article": "mauvais\narticle",
           "added_condition": "condition\najoutée"}
    vals = [d[n]["rate"] * 100 for n in names]
    err = [[(d[n]["rate"] - d[n]["ci"][0]) * 100 for n in names], [(d[n]["ci"][1] - d[n]["rate"]) * 100 for n in names]]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    bars = ax.bar(range(len(names)), vals, 0.6, color=[C[n] for n in names], yerr=err, capsize=3, ecolor="#333")
    labels(ax, bars, vals, inside=True)
    ax.set_xticks(range(len(names)), [lab[n] for n in names], fontsize=8); ax.set_ylabel("signalées (%)"); ax.set_ylim(0, 110)
    ax.set_title(f"n = {d[names[0]]['n']} par type ; IC 95 % de Wilson", fontsize=9)
    save(fig, "f8_calibration")


def f9():
    d = DATA.get("f9")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(W, 2.4))
    vals = [x["jaccard"] if x["jaccard"] is not None else 0 for x in d]
    bars = ax.bar(range(len(d)), vals, 0.6, color=["#009E73" if x["same"] else "#D55E00" for x in d])
    labels(ax, bars, vals, nd=2)
    ax.set_xticks(range(len(d)), [x["id"] for x in d]); ax.set_ylabel("Jaccard moyen des citations"); ax.set_ylim(0, 1.15)
    ax.set_title("vert : 6 preuves identiques entre exécutions ; orange : différentes", fontsize=8)
    save(fig, "f9_stability")


def f10_f11():
    d = DATA.get("final_eval")
    if not d:
        return
    systems = [s for s in ("v1", "v1.1", "v2") if s in d]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for k, (m, lab) in enumerate((("correct", "correctes"), ("hallucination (answer)", "hallucinées"))):
        vals = [(d[s].get(m) or 0) * 100 for s in systems]
        bars = ax.bar([i + (k - 0.5) * 0.38 for i in range(len(systems))], vals, 0.38,
                      color=["#009E73", "#D55E00"][k], label=lab)
        labels(ax, bars, vals)
    ax.set_xticks(range(len(systems)), systems); ax.set_ylabel("%"); ax.legend(frameon=False)
    save(fig, "f10_final_eval")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.4))
    for ax, m, lab in ((a1, "seconds", "secondes / question"), (a2, "LLM calls", "appels LLM / question")):
        vals = [d[s].get(m) or 0 for s in systems]
        labels(ax, ax.bar(systems, vals, 0.6, color="#0072B2"), vals)
        ax.set_ylabel(lab)
    save(fig, "f11_latency_calls")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (f2, f3, f4, f5, f6, f7, f8, f9, f10_f11):
        f()
