from app import normalize_probs, validate_probability_simplex
import math

cases=[(0.4,0.3,0.3),(float("nan"),0.2,0.8),(0.7,-0.1,0.2),(0,0,0),(1e300,1e300,1e300)]
for x in cases:
    p=validate_probability_simplex(x)
    assert all(math.isfinite(v) and v>=0 for v in p), p
    assert abs(sum(p)-1.0)<1e-10, p
print("V10.0.5 PROBABILITY INTEGRITY PASS")
