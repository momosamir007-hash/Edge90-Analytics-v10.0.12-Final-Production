"""Final release acceptance tests for Edge90 v10.0.12."""
from decision_policy import classify
from decision_policy_audit import audit_temporal_folds
from statistical_significance import compare_policies


def row(actual, probs):
    return {"actual": actual, "probs": probs}

# Policy equivalence: symmetric unit costs must be exactly Argmax.
probs = [(0.70,0.20,0.10),(0.20,0.55,0.25),(0.10,0.20,0.70),(0.34,0.33,0.33)]
assert all(classify(p, "expected_cost") == classify(p, "argmax") for p in probs)

fold = [row("HOME", (.70,.20,.10)), row("DRAW", (.20,.55,.25)),
        row("AWAY", (.10,.20,.70)), row("HOME", (.60,.25,.15))]
folds = [fold, fold, fold]
result = audit_temporal_folds(folds)
assert result["version"] == "v10.0.12"
assert result["selection"]["selected"] == "argmax"
assert result["dual_objective"]["accuracy_first"] == "argmax"
assert result["dual_objective"]["balanced"] == "argmax"

rows = fold * 20
sig = compare_policies(rows, "expected_cost")
assert sig["mcnemar"]["p_value"] == 1.0
assert sig["bootstrap"]["difference"] == 0.0
print("V10.0.12 FINAL ACCEPTANCE PASS")
