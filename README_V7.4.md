# v7.4 Deep Error Mining

Run the normal application/backtest. The returned backtest dictionary now contains `deep_error_mining` with:
- `reliability`: confidence-bin calibration diagnostics
- `high_confidence_errors`: the strongest wrong predictions
- `component_agreement`: which component models agree/disagree with the final ensemble
- `by_actual`: distribution by realised outcome

This version is diagnostic-first. Do not tune thresholds on the evaluation window; use a later calibration window to validate any changes.
