"""ROUGH power estimate for fresh/FRESH_PREREG_DRAFT.md (simulation; no data, no model, no LLM).

    python fresh/power_estimate.py      -> fresh/power_estimate.json
"""
import json
from pathlib import Path
import numpy as np

SEED = 20261022
N_SIM = 2000
N_BOOT = 1000            # per simulated study (the real analysis uses 5000)
NS = (100, 150, 200)
DISC_C = 22 / 110
DISC_B = 19 / 110


def probs(disc, net):
    """gain / loss probabilities with discordance disc and expected net difference net."""
    return (disc + net) / 2, (disc - net) / 2


def simulate(n, p_gain, p_loss, rule, rng):
    langs = np.arange(n) % 3
    ok = 0
    for _ in range(N_SIM):
        u = rng.random(n)
        d = np.where(u < p_gain, 1, np.where(u < p_gain + p_loss, -1, 0))
        lang_ok = all(d[langs == l].sum() > -3 for l in range(3))
        if rule == "C":
            idx = rng.integers(0, n, size=(N_BOOT, n))
            means = np.sort(d[idx].mean(axis=1))
            passed = means[int(0.025 * N_BOOT)] > 0 and lang_ok
        else:
            passed = d.mean() >= 0 and lang_ok
        ok += passed
    return ok / N_SIM


def main():
    rng = np.random.default_rng(SEED)
    scen_c = {"dev effect (+16 / -6 per 110: net +9.1 pp)": probs(DISC_C, 10 / 110),
              "half the dev effect (net +4.5 pp)": probs(DISC_C, 5 / 110),
              "no effect (net 0)": probs(DISC_C, 0.0)}
    scen_b = {"dev effect (+12 / -7 per 110: net +4.5 pp)": probs(DISC_B, 5 / 110),
              "no effect (net 0)": probs(DISC_B, 0.0),
              "a loss (net -4.5 pp)": probs(DISC_B, -5 / 110)}
    out = {"label": "ROUGH ESTIMATE (simulation under assumed effects; the dev effects are likely optimistic)",
           "seed": SEED, "n_sim": N_SIM, "n_boot": N_BOOT, "discordance_C": DISC_C, "discordance_B": DISC_B,
           "C_adopted_probability": {s: {n: simulate(n, *p, "C", rng) for n in NS} for s, p in scen_c.items()},
           "B_adopted_probability": {s: {n: simulate(n, *p, "B", rng) for n in NS} for s, p in scen_b.items()},
           "minimal_net_difference_80pct_normal_approx": {n: None for n in NS}}
    for n in NS:
        d = 0.0
        while d < DISC_C and d - 1.96 * np.sqrt((DISC_C - d * d) / n) < 0.84 * np.sqrt((DISC_C - d * d) / n):
            d += 0.001
        out["minimal_net_difference_80pct_normal_approx"][n] = round(d, 3)
    Path(__file__).with_name("power_estimate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
