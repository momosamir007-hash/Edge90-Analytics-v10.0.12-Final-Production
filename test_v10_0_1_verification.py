"""Reproducible v10.0.1 integration verification."""
import json
import app

# Keep the smoke test deterministic and practical while exercising the
# isolated temporal Engine path that previously crashed.
app.ML_AVAILABLE = False
league = app.LeagueApp("PL", "")
assert league.init(), "League initialization failed"
result = league.bt.run(league.raw[:500], league.resources, split=0.70)
assert not result.get("error"), result
assert result.get("eval_size", 0) >= 10
assert result.get("logloss", 0) > 0
print(json.dumps({
    "status": "PASS",
    "matches_loaded": len(league.raw),
    "eval_size": result.get("eval_size"),
    "logloss": result.get("logloss"),
    "brier": result.get("brier"),
    "result_acc": result.get("result_acc"),
}, indent=2))
