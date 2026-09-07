"""Smoke tests for v10.0.3 draw telemetry and acceptance metrics."""
from app import Pred, FinalBenchmarkAcceptanceEngine
p = Pred()
assert isinstance(p.draw_trace, dict)
wf = {
    'fold_count': 4, 'total_eval': 100, 'logloss': 1.0, 'brier': .30,
    'actual_draw_rate': .25, 'predicted_draw_rate': .01,
    'draw_recall': .02, 'mean_draw_probability': .24,
    'draw_probability_on_actual_draw': .27, 'draw_brier': .18,
}
assert FinalBenchmarkAcceptanceEngine._metric_status(wf) == []
print('V10.0.3 DRAW AUDIT SMOKE PASS')
