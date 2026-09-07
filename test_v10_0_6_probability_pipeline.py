import numpy as np
from app import align_probability_matrix, temporal_neg_log_loss_scorer

p=align_probability_matrix([[0.7,0.3],[0.2,0.8]],[0,2])
assert p.shape==(2,3)
assert np.allclose(p.sum(axis=1),1.0)
assert np.allclose(p[:,1],0.0)

class E:
    classes_=np.array([0,2])
    def predict_proba(self,X): return np.array([[.6,.4],[.1,.9]])

score=temporal_neg_log_loss_scorer(E(), np.zeros((2,1)), np.array([0,2]))
assert score < 0
print('V10.0.6 PROBABILITY PIPELINE PASS')
