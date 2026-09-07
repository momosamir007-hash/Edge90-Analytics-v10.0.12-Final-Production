# v10.0 Release Readiness Checklist

Before declaring a league live:

1. Load the intended league data successfully.
2. Run `LeagueApp.run_final_benchmark_acceptance()`.
3. Confirm the report says `PASS` and `safe_for_final_release: true`.
4. Review data-quality and leakage-risk fields.
5. Confirm production hardening does not block automatic actions.
6. Verify model artifacts can load in the target runtime.
7. Keep the previous champion artifact available for rollback.

A syntax or ZIP integrity pass alone is not a substitute for empirical acceptance.
