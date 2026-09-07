# Edge90 Analytics v10.0.12 — Final Production Release

## Status
**FINAL / DELIVERY READY**

This release is the final hardened package following the temporal 5×5 benchmark and the v10.0.11 statistical-significance review.

## Production decision policy
**Argmax is the canonical production policy.**

The system keeps Argmax as the immutable safety baseline. A challenger can only be promoted after temporal stability, effect-size, accuracy, and statistical gates are satisfied.

## v10.0.12 final fixes
- Policy-equivalence guard: a policy that produces the same behavior as Argmax is never promoted as a separate winner.
- Expected Cost with the symmetric unit-cost matrix is treated as equivalent to Argmax.
- Accuracy-First requires a strict accuracy improvement plus paired statistical significance.
- Balanced requires a material Macro-F1 improvement (default +0.005), accuracy non-inferiority within the configured tolerance, temporal stability, and statistical evidence.
- Probability generation is unchanged by the decision layer.
- Deterministic paired bootstrap and exact McNemar diagnostics remain available.
- All five master datasets are included.
- Historical league datasets are included for reproducible retraining/backtesting.

## Included
- `app.py` — application entry point.
- `data/` — five league master datasets.
- `historical_data/` — historical league datasets.
- `config/` — aliases, team maps, and rivalries.
- `decision_policy.py` — classification policies.
- `decision_policy_audit.py` — temporal policy audit and safe selection.
- `decision_layer.py` — dual-objective selection.
- `statistical_significance.py` — paired significance diagnostics.
- `benchmark_v10_0_10.py` — benchmark runner.
- `resumable_walk_forward.py` — resumable temporal validation support.
- `audit_project.py` — integrity audit.
- `test_*.py` — regression and release tests.
- `requirements.txt` and `requirements-lock.txt` — dependency specifications.
- Release notes and historical change logs.

## Validation performed
- Python syntax compilation: PASS.
- Project integrity audit: PASS — 0 warnings / 0 errors.
- Historical regression tests: PASS.
- v10.0.12 final acceptance tests: PASS.
- Five league master datasets validated for required columns, non-negative goals, consistent FTR, and duplicate raw keys.
- Policy equivalence behavior verified.

## Installation

```bash
python -m pip install -r requirements.txt
python audit_project.py
python test_v10_0_12_final.py
streamlit run app.py
```

For a fully reproducible environment, use the pinned core versions in `requirements-lock.txt` and install the Streamlit runtime constraint from `requirements.txt`.

## Important
A statistical test does not guarantee future accuracy. It is used here as one conservative gate together with chronological OOS evaluation, stability, and safety fallbacks.
