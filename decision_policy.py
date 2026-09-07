"""Edge90 v10.0.9 decision-policy engine. Probability metrics remain untouched."""
from __future__ import annotations
import numpy as np

LABELS=("HOME","DRAW","AWAY")

def argmax_policy(p, **_): return LABELS[int(np.argmax(p))]
def draw_threshold_policy(p, threshold=0.30, **_): return "DRAW" if p[1]>=threshold else argmax_policy(p)
def margin_draw_policy(p, threshold=0.22, margin=0.06, **_):
    return "DRAW" if p[1]>=threshold and max(p[0],p[2])-p[1]<=margin else argmax_policy(p)
def expected_cost_policy(p, cost=None, **_):
    C=np.asarray(cost if cost is not None else [[0,1,1],[1,0,1],[1,1,0]],dtype=float)
    return LABELS[int(np.argmin(np.asarray(p) @ C))]
def contextual_policy(p, context=None, **kw):
    context=context or {}; conf=float(max(p)); close=abs(float(p[0])-float(p[2]))
    threshold=float(context.get("draw_threshold",0.24)); margin=float(context.get("margin",0.07))
    if p[1]>=threshold and close<=margin and conf<0.62: return "DRAW"
    return margin_draw_policy(p, threshold=threshold, margin=margin, **kw)

POLICIES={"argmax":argmax_policy,"draw_threshold":draw_threshold_policy,"margin_based":margin_draw_policy,"expected_cost":expected_cost_policy,"contextual":contextual_policy}

def classify(probabilities, policy="argmax", **kwargs):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(3,) or not np.all(np.isfinite(p)) or np.any(p<0): raise ValueError("Invalid probability vector")
    s=p.sum()
    if s<=0: raise ValueError("Probability vector has zero mass")
    p=p/s
    return POLICIES[policy](p, **kwargs)

def classification_metrics(rows, policy="argmax", **kwargs):
    y=[r["actual"] for r in rows]; pred=[classify(r["probs"],policy,context=r.get("context"),**kwargs) for r in rows]
    n=max(1,len(y)); labels=LABELS
    recalls={c:sum(a==c and b==c for a,b in zip(y,pred))/max(1,sum(a==c for a in y)) for c in labels}
    precision=sum(a=="DRAW" and b=="DRAW" for a,b in zip(y,pred))/max(1,sum(b=="DRAW" for b in pred))
    f1=[]
    for c in labels:
        pr=sum(a==c and b==c for a,b in zip(y,pred))/max(1,sum(b==c for b in pred)); rc=recalls[c]
        f1.append(0.0 if pr+rc==0 else 2*pr*rc/(pr+rc))
    return {"policy":policy,"accuracy":sum(a==b for a,b in zip(y,pred))/n,"draw_recall":recalls["DRAW"],"draw_precision":precision,"predicted_draw_rate":sum(b=="DRAW" for b in pred)/n,"home_recall":recalls["HOME"],"away_recall":recalls["AWAY"],"macro_f1":float(np.mean(f1))}
