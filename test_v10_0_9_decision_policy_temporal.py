from decision_policy_audit import audit_fold, audit_temporal_folds, select_policy


def row(actual, probs):
    return {"actual": actual, "probs": probs}

# Three folds where draw-threshold is consistently at least as accurate as argmax
# and improves macro-F1 by recovering otherwise missed draws.
folds = [
    [row("DRAW", (.34,.34,.32)), row("HOME", (.46,.30,.24)), row("AWAY", (.25,.29,.46)), row("DRAW", (.31,.36,.33))],
    [row("DRAW", (.33,.35,.32)), row("HOME", (.47,.29,.24)), row("AWAY", (.24,.30,.46)), row("DRAW", (.32,.35,.33))],
    [row("DRAW", (.34,.34,.32)), row("HOME", (.45,.31,.24)), row("AWAY", (.24,.31,.45)), row("DRAW", (.30,.37,.33))],
]

result = audit_temporal_folds(folds, min_folds=3)
assert result["fold_count"] == 3
assert result["selection"]["selected"] in {"argmax", "draw_threshold", "margin_based", "contextual", "expected_cost"}
assert result["probability_engine_unchanged"] is True

# One fold is not allowed to promote anything.
one = select_policy([audit_fold(folds[0])])
assert one["selected"] == "argmax"
assert one["stable"] is False

# Deliberately unstable challenger: strict stability gates must force Argmax.
unstable = select_policy([audit_fold(f) for f in [folds[0], folds[1], [row("HOME", (.60,.20,.20)), row("HOME", (.60,.20,.20)), row("AWAY", (.20,.20,.60))]]])
assert unstable["selected"] == "argmax"

print("V10.0.9 TEMPORAL DECISION POLICY PASS")
