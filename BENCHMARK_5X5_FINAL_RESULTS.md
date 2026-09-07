# Edge90 v10.0.12 — Final 5×5 Benchmark Record

## Scope
Five leagues × five chronological folds = **25 OOS folds** and **3,821 OOS matches**.
All policies were evaluated on the same OOS probability vectors.

## Global results

| Policy | Accuracy | Macro-F1 | Draw Recall | Draw Precision |
|---|---:|---:|---:|---:|
| Argmax | **51.26%** | 0.4334 | 14.38% | 27.06% |
| Expected Cost | **51.26%** | 0.4334 | 14.38% | 27.06% |
| Margin Based | 49.69% | **0.4406** | 22.78% | **30.22%** |
| Contextual | 49.13% | **0.4486** | **28.00%** | 26.87% |
| Draw Threshold | 46.72% | 0.4299 | **34.60%** | 26.14% |

## Paired statistical comparison with Argmax

| Challenger | McNemar p-value | Bootstrap 95% CI of accuracy difference | Decision |
|---|---:|---:|---|
| Draw Threshold | 2.95e-13 | -5.65 to -3.25 pp | Reject |
| Margin Based | 5.38e-05 | -2.33 to -0.76 pp | Reject |
| Contextual | 6.97e-06 | -3.09 to -1.18 pp | Reject |
| Expected Cost | 1.000 | 0.00 to 0.00 pp | Equivalent |

## Production conclusion
- **Accuracy-First:** Argmax.
- **Balanced:** Argmax in the tested configuration; no challenger met the strict material-improvement gate.
- **Expected Cost:** behaviorally equivalent to Argmax under the symmetric unit-cost matrix.
- No alternative policy demonstrated a statistically supported accuracy advantage over Argmax.

These results are evidence from historical OOS evaluation, not a guarantee of future performance.
