from decision_policy_audit import audit_fold, audit_temporal_folds
from statistical_significance import mcnemar_exact, paired_bootstrap_accuracy


def row(actual, probs): return {"actual": actual, "probs": probs}

fold = [row("HOME",(.70,.20,.10)), row("HOME",(.60,.25,.15)), row("AWAY",(.10,.20,.70)), row("DRAW",(.20,.55,.25))]
rows = fold * 10
assert mcnemar_exact(rows, "argmax")["p_value"] == 1.0
b = paired_bootstrap_accuracy(rows, "argmax")
assert b["n"] == len(rows)
res = audit_temporal_folds([fold, fold, fold])
assert res["version"] == "v10.0.12"
assert set(res["dual_objective"]) >= {"accuracy_first", "balanced", "policies"}
print("V10.0.11 SIGNIFICANCE + DUAL OBJECTIVE PASS")
