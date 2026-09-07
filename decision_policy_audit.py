"""Leakage-safe temporal decision-policy evaluation and selection for Edge90 v10.0.9."""
from __future__ import annotations

from statistics import pstdev
from decision_policy import POLICIES, classification_metrics

VERSION = "v10.0.12"


def audit_fold(rows, candidate_params=None):
    """Evaluate every policy on exactly one historical/out-of-sample fold."""
    candidate_params = candidate_params or {}
    return {
        name: classification_metrics(rows, name, **candidate_params.get(name, {}))
        for name in POLICIES
    }


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _safe_pstdev(values):
    return pstdev(values) if len(values) > 1 else 0.0


def _aggregate_policy(name, audits):
    vals = [a[name] for a in audits]
    keys = [k for k in vals[0] if k != "policy"]
    out = {k: _mean([float(v[k]) for v in vals]) for k in keys}
    out.update({
        "folds": len(vals),
        "accuracy_std": _safe_pstdev([float(v["accuracy"]) for v in vals]),
        "macro_f1_std": _safe_pstdev([float(v["macro_f1"]) for v in vals]),
        "draw_recall_std": _safe_pstdev([float(v["draw_recall"]) for v in vals]),
        "draw_precision_std": _safe_pstdev([float(v["draw_precision"]) for v in vals]),
    })
    return out


def select_policy(
    audits,
    min_folds=3,
    min_accuracy_ratio=0.98,
    max_draw_rate_delta=0.15,
    min_win_rate=0.60,
    min_macro_f1_delta=0.0,
    max_macro_f1_std=0.08,
    min_worst_fold_accuracy_ratio=0.95,
):
    """Select a policy only from historical temporal folds.

    Argmax is the immutable safety baseline. A challenger must satisfy all
    safety/stability gates across folds; otherwise Argmax is selected.
    """
    if not audits:
        return {"selected": "argmax", "reason": "no temporal folds supplied", "stable": False}
    if len(audits) < min_folds:
        return {
            "selected": "argmax",
            "reason": f"insufficient temporal folds ({len(audits)} < {min_folds})",
            "stable": False,
            "folds": len(audits),
        }

    required = set(POLICIES)
    if any(set(a) != required for a in audits):
        return {"selected": "argmax", "reason": "incomplete policy audit fold", "stable": False}

    aggregates = {name: _aggregate_policy(name, audits) for name in POLICIES}
    base = aggregates["argmax"]
    base_fold_acc = [float(a["argmax"]["accuracy"]) for a in audits]
    base_mean_draw = base["predicted_draw_rate"]

    candidates = {}
    diagnostics = {}
    for name, m in aggregates.items():
        if name == "argmax":
            continue
        fold_acc = [float(a[name]["accuracy"]) for a in audits]
        wins = sum(x >= b for x, b in zip(fold_acc, base_fold_acc))
        win_rate = wins / len(audits)
        worst_ratio = min(
            (x / b) if b > 0 else (1.0 if x >= b else 0.0)
            for x, b in zip(fold_acc, base_fold_acc)
        )
        acc_gate = m["accuracy"] >= base["accuracy"] * min_accuracy_ratio
        draw_gate = abs(m["predicted_draw_rate"] - base_mean_draw) <= max_draw_rate_delta
        f1_gate = m["macro_f1"] >= base["macro_f1"] + min_macro_f1_delta
        stability_gate = (
            win_rate >= min_win_rate
            and worst_ratio >= min_worst_fold_accuracy_ratio
            and m["macro_f1_std"] <= max_macro_f1_std
        )
        diagnostics[name] = {
            **m,
            "win_rate_vs_argmax": win_rate,
            "worst_fold_accuracy_ratio": worst_ratio,
            "accuracy_gate": acc_gate,
            "draw_rate_gate": draw_gate,
            "macro_f1_gate": f1_gate,
            "stability_gate": stability_gate,
            "eligible": bool(acc_gate and draw_gate and f1_gate and stability_gate),
        }
        if diagnostics[name]["eligible"]:
            candidates[name] = diagnostics[name]

    if not candidates:
        return {
            "selected": "argmax",
            "reason": "no challenger passed all temporal safety and stability gates",
            "stable": True,
            "folds": len(audits),
            "baseline": base,
            "policies": diagnostics,
        }

    # Never promote a challenger that is behaviorally equivalent to Argmax.
    candidates = {
        n: v for n, v in candidates.items()
        if not (v.get("accuracy") == base.get("accuracy") and
                v.get("predicted_draw_rate") == base.get("predicted_draw_rate"))
    }
    if not candidates:
        return {
            "selected": "argmax",
            "reason": "no non-equivalent challenger passed temporal gates",
            "stable": True, "folds": len(audits), "baseline": base,
            "policies": diagnostics,
        }

    selected = max(
        candidates,
        key=lambda n: (
            candidates[n]["macro_f1"],
            candidates[n]["draw_recall"],
            candidates[n]["accuracy"],
            candidates[n]["win_rate_vs_argmax"],
        ),
    )
    return {
        "selected": selected,
        "metrics": candidates[selected],
        "reason": "best challenger after temporal safety and stability gates",
        "stable": True,
        "folds": len(audits),
        "baseline": base,
        "policies": diagnostics,
    }


def audit_temporal_folds(folds, candidate_params=None, **selection_kwargs):
    """Audit a sequence of chronologically ordered folds without cross-fold leakage."""
    audits = [audit_fold(rows, candidate_params) for rows in folds if rows]
    selection = select_policy(audits, **selection_kwargs)
    try:
        from decision_layer import dual_objective_selection
        dual = dual_objective_selection(audits, folds)
    except Exception as exc:
        dual = {"accuracy_first":"argmax", "balanced":"argmax", "error": str(exc)}
    return {
        "version": VERSION,
        "fold_count": len(audits),
        "fold_audits": audits,
        "selection": selection,
        "dual_objective": dual,
        "probability_engine_unchanged": True,
    }
