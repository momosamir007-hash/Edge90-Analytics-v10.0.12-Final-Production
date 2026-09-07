"""Edge90 v10.0.12 dual-objective decision layer.

Probability generation is deliberately out of scope.  This module only
selects a classification policy from leakage-safe OOS evidence and keeps
Argmax as the canonical safety baseline unless a challenger proves a real,
stable advantage.
"""
from __future__ import annotations
from decision_policy_audit import _aggregate_policy
from statistical_significance import compare_policies

VERSION = "v10.0.12"
BASELINE = "argmax"


def _eligible_accuracy(m, base, sig, min_win_rate=0.60, max_macro_f1_std=0.08):
    return (
        m["accuracy"] > base["accuracy"]
        and m["win_rate_vs_argmax"] >= min_win_rate
        and m["worst_fold_accuracy_ratio"] >= 0.95
        and m["macro_f1_std"] <= max_macro_f1_std
        and sig["significant_accuracy_gain"]
        and not m.get("behaviorally_equivalent_to_argmax", False)
    )


def _eligible_balanced(
    m, base, sig, accuracy_tolerance=0.01, min_win_rate=0.60,
    max_macro_f1_std=0.08, min_macro_f1_delta=0.005,
):
    # Balanced is deliberately strict: a challenger must improve Macro-F1 by
    # a material amount, remain accuracy-noninferior, and be stable over time.
    return (
        m["macro_f1"] >= base["macro_f1"] + min_macro_f1_delta
        and m["accuracy"] >= base["accuracy"] - accuracy_tolerance
        and m["win_rate_vs_argmax"] >= min_win_rate
        and m["worst_fold_accuracy_ratio"] >= 0.95
        and m["macro_f1_std"] <= max_macro_f1_std
        and sig["bootstrap"]["ci_low"] >= -accuracy_tolerance
        and not m.get("behaviorally_equivalent_to_argmax", False)
    )


def dual_objective_selection(
    audits, folds_rows, alpha=0.05, accuracy_tolerance=0.01,
    min_macro_f1_delta=0.005,
):
    if not audits or len(audits) < 3:
        return {
            "version": VERSION, "accuracy_first": BASELINE, "balanced": BASELINE,
            "baseline": BASELINE, "reason": "insufficient temporal evidence", "policies": {},
        }
    aggregates = {n: _aggregate_policy(n, audits) for n in audits[0]}
    base = aggregates[BASELINE]
    diagnostics = {}
    accuracy_candidates = []
    balanced_candidates = []
    all_rows = [r for fold in folds_rows for r in fold]
    for name, m in aggregates.items():
        if name == BASELINE:
            continue
        sig = compare_policies(all_rows, name, alpha=alpha)
        fold_base = [a[BASELINE]["accuracy"] for a in audits]
        fold_cur = [a[name]["accuracy"] for a in audits]
        wins = sum(c >= b for c, b in zip(fold_cur, fold_base))
        m["wins_vs_argmax"] = wins
        m["win_rate_vs_argmax"] = wins / len(audits)
        m["worst_fold_accuracy_ratio"] = min(
            (c / b if b else (1.0 if c >= b else 0.0))
            for c, b in zip(fold_cur, fold_base)
        )
        # Exact behavioural equivalence is stronger than equal aggregate metrics.
        m["behaviorally_equivalent_to_argmax"] = all(
            a[name].get("accuracy") == a[BASELINE].get("accuracy")
            and a[name].get("predicted_draw_rate") == a[BASELINE].get("predicted_draw_rate")
            for a in audits
        )
        diagnostics[name] = {
            "metrics": m,
            "significance": sig,
            "accuracy_first_eligible": _eligible_accuracy(m, base, sig),
            "balanced_eligible": _eligible_balanced(
                m, base, sig, accuracy_tolerance, min_macro_f1_delta=min_macro_f1_delta
            ),
        }
        if diagnostics[name]["accuracy_first_eligible"]:
            accuracy_candidates.append(name)
        if diagnostics[name]["balanced_eligible"]:
            balanced_candidates.append(name)

    accuracy_first = max(
        accuracy_candidates,
        key=lambda n: (diagnostics[n]["metrics"]["accuracy"], diagnostics[n]["metrics"]["macro_f1"]),
        default=BASELINE,
    )
    balanced = max(
        balanced_candidates,
        key=lambda n: (diagnostics[n]["metrics"]["macro_f1"], diagnostics[n]["metrics"]["accuracy"]),
        default=BASELINE,
    )
    return {
        "version": VERSION,
        "accuracy_first": accuracy_first,
        "balanced": balanced,
        "baseline": BASELINE,
        "alpha": alpha,
        "accuracy_tolerance": accuracy_tolerance,
        "min_macro_f1_delta": min_macro_f1_delta,
        "policies": diagnostics,
    }
