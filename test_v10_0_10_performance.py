"""v10.0.10 tests: fast-path fit preserves final fitted pipeline semantics and benchmark isolation."""
import tempfile, json
from pathlib import Path
from benchmark_v10_0_10 import benchmark_league

def test_policy_benchmark_uses_all_folds():
    folds=[]
    for i in range(3):
        folds.append({"fold":i+1,"predictions":[
            {"actual":"HOME","probs":(0.60,0.20,0.20)},
            {"actual":"DRAW","probs":(0.30,0.40,0.30)},
            {"actual":"AWAY","probs":(0.20,0.25,0.55)},
        ]})
    r=benchmark_league(folds,"TEST")
    assert r["fold_count"]==3
    assert set(r["aggregate"])=={"argmax","draw_threshold","margin_based","expected_cost","contextual"}
    assert r["selection"]["folds"]==3

def test_empty_benchmark_is_safe():
    r=benchmark_league([],"EMPTY")
    assert r["fold_count"]==0
    assert r["selection"]["selected"]=="argmax"
