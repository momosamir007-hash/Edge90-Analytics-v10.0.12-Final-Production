"""Leakage-safe paired significance tests for Edge90 decision policies.

All tests compare policies on the exact same OOS rows. No model training or
probability generation occurs here.
"""
from __future__ import annotations
import math
import random
from decision_policy import classify


def _preds(rows, policy):
    return [classify(r["probs"], policy, context=r.get("context")) for r in rows]


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mcnemar_exact(rows, challenger, baseline="argmax"):
    """Exact two-sided McNemar p-value for paired classification accuracy."""
    a = _preds(rows, baseline); c = _preds(rows, challenger)
    y = [r["actual"] for r in rows]
    b = sum(x != yy and z == yy for x, z, yy in zip(a, c, y))  # base wrong, challenger right
    cc = sum(x == yy and z != yy for x, z, yy in zip(a, c, y))
    n = b + cc
    if n == 0:
        return {"b": b, "c": cc, "discordant": 0, "p_value": 1.0}
    k = min(b, cc)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": cc, "discordant": n, "p_value": p}


def paired_bootstrap_accuracy(rows, challenger, baseline="argmax", n_boot=4000, seed=20260907):
    """Deterministic paired bootstrap CI for accuracy difference (challenger-base)."""
    if not rows:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    y = [r["actual"] for r in rows]
    base = _preds(rows, baseline); chal = _preds(rows, challenger)
    diffs = [int(c == yy) - int(b == yy) for b, c, yy in zip(base, chal, y)]
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(max(100, int(n_boot))):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    # Conservative two-sided bootstrap sign probability.
    p = min(1.0, 2.0 * min(sum(x <= 0 for x in samples), sum(x >= 0 for x in samples)) / len(samples))
    return {"difference": obs, "ci_low": lo, "ci_high": hi, "p_value": p, "n": n}


def compare_policies(rows, challenger, baseline="argmax", alpha=0.05, n_boot=4000):
    exact = mcnemar_exact(rows, challenger, baseline)
    boot = paired_bootstrap_accuracy(rows, challenger, baseline, n_boot=n_boot)
    significant = exact["p_value"] < alpha and boot["ci_low"] > 0.0
    return {
        "baseline": baseline,
        "challenger": challenger,
        "mcnemar": exact,
        "bootstrap": boot,
        "alpha": alpha,
        "significant_accuracy_gain": bool(significant),
    }
