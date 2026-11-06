"""Draws report/figures/f1*.pdf from report/generated/*.csv only (written by eval/make_report_numbers.py).

    <python with matplotlib> eval/make_report_figures.py

Needs matplotlib and the standard library, no project module: like report/make_figures.py, it can run in another
interpreter than the rag-app environment, which has no matplotlib. One colour and one marker per system in every
figure (Okabe-Ito, colour-blind safe); all intervals are the 95% bootstrap intervals of the CSV files.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
W = 14 / 2.54
COLOR = {"v1": "#999999", "v1.1": "#0072B2", "v2": "#D55E00"}
MARK = {"v1": "s", "v1.1": "o", "v2": "D"}
SYSTEMS = ["v1", "v1.1", "v2"]
SETS = [("S1", "Étape 1"), ("S2", "Étape 2"), ("N", "Jeu bruité")]
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.grid": True, "axes.grid.axis": "x", "grid.color": "#dddddd", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})
COMMA = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",").replace("-", "\N{MINUS SIGN}"))


def read(name: str) -> list[dict]:
    with (GEN / f"{name}.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(r: dict, k: str):
    return float(r[k]) if r.get(k) not in ("", None) else None


def lookup(rows, **kw) -> dict:
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def dot(ax, y, r, color, marker, f=100.0, label=None, hollow=False):
    """One dot with its interval on a horizontal axis. An empty line (metric not observed) draws nothing."""
    m = num(r, "mean")
    if m is None:
        return
    lo, hi = num(r, "lo"), num(r, "hi")
    if lo is not None:
        ax.plot([lo * f, hi * f], [y, y], color=color, lw=1.2, solid_capstyle="round", zorder=2)
    ax.plot(m * f, y, marker, color=color, ms=4.5, mfc="white" if hollow else color, mec=color, label=label, zorder=3)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"report/figures/{name}.pdf")


def system_legend(fig, y=1.04):
    handles = [plt.Line2D([], [], color=COLOR[s], marker=MARK[s], ms=4.5, lw=1.2, label=s) for s in SYSTEMS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, y))


def f10_hit_correct():
    rows = read("per_system")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True, constrained_layout=True)
    for ax, (metric, title) in zip(axes, (("Hit@6 (gold in evidence)", "Hit@6"), ("correct", "Correct (juge)"))):
        for k, (key, _) in enumerate(SETS):
            for j, s in enumerate(SYSTEMS):
                dot(ax, k + (j - 1) * 0.22, lookup(rows, set=key, system=s, metric=metric), COLOR[s], MARK[s])
        ax.set_title(title); ax.set_xlim(0, 100); ax.set_xlabel("Part des questions (%)"); ax.xaxis.set_major_formatter(COMMA)
    axes[0].set_yticks(range(3), [lab for _, lab in SETS]); axes[0].invert_yaxis()
    system_legend(fig, 1.1)
    save(fig, "f10_hit_correct")


def f11_paired():
    rows = read("paired")
    metrics = [("Hit@6 (gold in evidence)", "Hit@6"), ("correct", "Correct"), ("incorrect", "Incorrect"), ("hallucination (answer)", "Hallucination")]
    comps = [("v2", "v1.1", "v2 \N{MINUS SIGN} v1.1"), ("v1.1", "v1", "v1.1 \N{MINUS SIGN} v1")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True, sharex=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, SETS):
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        for k, (metric, _) in enumerate(metrics):
            for j, (a, b, _) in enumerate(comps):
                dot(ax, k + (j - 0.5) * 0.3, lookup(rows, set=key, a=a, b=b, metric=metric), COLOR[a], MARK[a])
        ax.set_title(title); ax.set_xlabel("Différence (points de %)"); ax.xaxis.set_major_formatter(COMMA)
    axes[0].set_yticks(range(4), [m[1] for m in metrics]); axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=COLOR[a], marker=MARK[a], ms=4.5, lw=1.2, label=lab) for a, _, lab in comps]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    save(fig, "f11_paired")


def f12_noisy():
    deg, did = read("noisy_degradation"), read("noisy_did")
    metrics = [("Hit@6 (gold in evidence)", "Hit@6"), ("correct", "Correct"), ("incorrect", "Incorrect"), ("hallucination (answer)", "Hallucination")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True, constrained_layout=True)
    a1.axvline(0, color="black", lw=0.8, zorder=1); a2.axvline(0, color="black", lw=0.8, zorder=1)
    for k, (metric, _) in enumerate(metrics):
        for j, s in enumerate(SYSTEMS):
            dot(a1, k + (j - 1) * 0.22, lookup(deg, system=s, metric=metric), COLOR[s], MARK[s])
        dot(a2, k, lookup(did, a="v2", b="v1.1", metric=metric), COLOR["v2"], MARK["v2"])
    a1.set_title("Dégradation : bruité \N{MINUS SIGN} propre", fontsize=8); a2.set_title("Différence de différences : v2 / v1.1", fontsize=8)
    a1.set_xlabel("Différence (points de %)"); a2.set_xlabel("Différence de différences (points de %)")
    a1.xaxis.set_major_formatter(COMMA); a2.xaxis.set_major_formatter(COMMA)
    a1.set_yticks(range(4), [m[1] for m in metrics]); a1.invert_yaxis()
    system_legend(fig, 1.1)
    save(fig, "f12_noisy")


def f13_behaviour():
    rows = read("per_system")
    metrics = [("clarify w/o reply: asked for clarification (status)", "Clarification\ndemandée"), ("clarify w/o reply: answered directly", "Réponse\ndirecte"),
               ("refuse: explicit refusal status", "Refus\nexplicite"), ("refuse: no substantive answer", "Pas de réponse\nde fond")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True, sharex=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, SETS[:2]):
        for k, (metric, _) in enumerate(metrics):
            for j, s in enumerate(SYSTEMS):
                dot(ax, k + (j - 1) * 0.22, lookup(rows, set=key, system=s, metric=metric), COLOR[s], MARK[s])
        n = [lookup(rows, set=key, system="v2", metric=metrics[i][0])["n"] for i in (0, 2)]
        ax.set_title(f"{title}\nambiguës n = {n[0]}, hors champ n = {n[1]}", fontsize=8)
        ax.set_xlim(-3, 103); ax.set_xlabel("Part des questions (%)"); ax.xaxis.set_major_formatter(COMMA)
    axes[0].set_yticks(range(4), [m[1] for m in metrics]); axes[0].invert_yaxis()
    system_legend(fig, 1.09)
    save(fig, "f13_behaviour")


def f14_cost():
    rows = read("per_system")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True, constrained_layout=True)
    for ax, (metric, label) in zip(axes, (("LLM calls", "Appels LLM par question"), ("tokens per question", "Tokens par question"))):
        for k, (key, _) in enumerate(SETS):
            for j, s in enumerate(SYSTEMS):
                dot(ax, k + (j - 1) * 0.22, lookup(rows, set=key, system=s, metric=metric), COLOR[s], MARK[s], f=1.0)
        ax.set_xlabel(label); ax.set_xlim(left=0); ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "\N{THIN SPACE}")))
    axes[0].set_yticks(range(3), [lab for _, lab in SETS]); axes[0].invert_yaxis()
    system_legend(fig, 1.1)
    save(fig, "f14_cost")


def f15_latency():
    path = GEN / "latency.csv"
    if not path.exists():
        return
    rows = read("latency")
    fig, ax = plt.subplots(figsize=(W, 2.1), constrained_layout=True)
    for k, s in enumerate(SYSTEMS):
        r = lookup(rows, system=s)
        for j, (stem, hollow) in enumerate((("wall", False), ("excl", True))):
            y = k + (j - 0.5) * 0.34
            ax.plot([num(r, f"{stem}_lo"), num(r, f"{stem}_hi")], [y, y], color=COLOR[s], lw=1.2, zorder=2)
            ax.plot(num(r, f"{stem}_mean"), y, MARK[s], color=COLOR[s], mfc="white" if hollow else COLOR[s], ms=4.5, zorder=3)
            ax.plot(num(r, f"{stem}_median"), y, "|", color="black", ms=7, zorder=4)
    ax.set_yticks(range(3), SYSTEMS); ax.invert_yaxis(); ax.set_xlim(left=0)
    ax.set_xlabel("Secondes par question"); ax.xaxis.set_major_formatter(COMMA)
    handles = [plt.Line2D([], [], color="#444444", marker="o", ms=4.5, lw=1.2, label="moyenne, temps total"),
               plt.Line2D([], [], color="#444444", marker="o", mfc="white", ms=4.5, lw=1.2, label="moyenne, hors attentes de limite de débit"),
               plt.Line2D([], [], color="black", marker="|", ms=7, lw=0, label="médiane")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.12))
    save(fig, "f15_latency")


def f16_audit():
    rows = read("audit")
    names = {"correct": "Correct", "hallucination (answer)": "Hallucination\n(réponse)", "hallucination (LLM judge only)": "Hallucination\n(juge LLM seul)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True, constrained_layout=True)
    for k, r in enumerate(rows):
        a1.plot([num(r, "lo") * 100, num(r, "hi") * 100], [k, k], color="#444444", lw=1.2, zorder=2)
        a1.plot(num(r, "agreement") * 100, k, "o", color="#444444", ms=4.5, zorder=3)
        a2.plot([num(r, "kappa_lo"), num(r, "kappa_hi")], [k, k], color="#444444", lw=1.2, zorder=2)
        a2.plot(num(r, "kappa"), k, "o", color="#444444", ms=4.5, zorder=3)
    threshold = num(rows[0], "threshold") * 100
    a1.axvline(threshold, color="black", lw=0.8, ls="--", zorder=1)
    a1.text(threshold, 1.02, f"seuil {threshold:.0f} %", ha="center", va="bottom", fontsize=7, transform=a1.get_xaxis_transform())
    a2.axvline(0, color="black", lw=0.8, zorder=1)
    a1.set_xlim(0, 100); a1.set_xlabel("Accord juge / humain (%)"); a2.set_xlabel("Kappa de Cohen (sans unité)")
    a1.xaxis.set_major_formatter(COMMA); a2.xaxis.set_major_formatter(COMMA)
    a1.set_yticks(range(len(rows)), [names[r["measure"]] for r in rows]); a1.invert_yaxis()
    save(fig, "f16_audit")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (f10_hit_correct, f11_paired, f12_noisy, f13_behaviour, f14_cost, f15_latency, f16_audit):
        f()
