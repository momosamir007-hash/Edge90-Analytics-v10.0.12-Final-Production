"""Edge90 v10.0.10 — temporal decision-policy benchmark.

Policy evaluation is deliberately separated from probability generation: each
fold is generated once by the leakage-safe backtester, then every policy is
scored on the exact same out-of-sample probabilities.
"""
from __future__ import annotations
import json, math, statistics, argparse
from pathlib import Path
from decision_policy import POLICIES
from decision_policy_audit import audit_fold, select_policy

VERSION = "v10.0.10"
STAGES = ("draw_model","weighted_ensemble","after_draw_preservation","after_draw_correction","after_calibration","after_meta","final")

def _finite(x, default=0.0):
    try:
        x=float(x)
        return x if math.isfinite(x) else default
    except Exception:
        return default

def prediction_rows(predictions):
    rows=[]
    for p in predictions or []:
        probs=p.get("probs", (0,0,0))
        if not isinstance(probs,(list,tuple)) or len(probs)!=3: continue
        actual=p.get("actual")
        if actual not in ("HOME","DRAW","AWAY"): continue
        rows.append({"actual":actual,"probs":tuple(_finite(v) for v in probs),"context":p.get("context",{}) or {}})
    return rows

def fold_policy_audit(predictions):
    return audit_fold(prediction_rows(predictions))

def benchmark_league(folds, league_name=""):
    audits=[]
    fold_reports=[]
    for i, fold in enumerate(folds, 1):
        rows=prediction_rows(fold.get("predictions", fold) if isinstance(fold,dict) else fold)
        if not rows: continue
        audit=audit_fold(rows)
        audits.append(audit)
        fold_reports.append({"fold":int(fold.get("fold",i)) if isinstance(fold,dict) else i,
                             "eval_size":len(rows),"policies":audit})
    selection=select_policy(audits)
    aggregate={}
    if audits:
        names=list(POLICIES)
        for name in names:
            vals=[a[name] for a in audits]
            aggregate[name]={k:float(statistics.fmean([_finite(v.get(k)) for v in vals])) for k in ("accuracy","macro_f1","draw_recall","draw_precision","predicted_draw_rate")}
            aggregate[name]["accuracy_std"]=float(statistics.pstdev([_finite(v.get("accuracy")) for v in vals])) if len(vals)>1 else 0.0
            aggregate[name]["macro_f1_std"]=float(statistics.pstdev([_finite(v.get("macro_f1")) for v in vals])) if len(vals)>1 else 0.0
            aggregate[name]["draw_recall_std"]=float(statistics.pstdev([_finite(v.get("draw_recall")) for v in vals])) if len(vals)>1 else 0.0
            base=[_finite(a["argmax"].get("accuracy")) for a in audits]
            cur=[_finite(a[name].get("accuracy")) for a in audits]
            aggregate[name]["wins_vs_argmax"]=sum(c>=b for c,b in zip(cur,base))
            aggregate[name]["win_rate_vs_argmax"]=aggregate[name]["wins_vs_argmax"]/len(audits)
            aggregate[name]["worst_fold_accuracy_ratio"]=min(((c/b) if b>0 else (1.0 if c>=b else 0.0)) for c,b in zip(cur,base))
    return {"version":VERSION,"league":league_name,"fold_count":len(audits),"folds":fold_reports,"aggregate":aggregate,"selection":selection}

def run_from_checkpoints(root):
    root=Path(root); out={}
    for league_dir in sorted(root.glob("*")):
        if not league_dir.is_dir(): continue
        checkpoint=league_dir/"walk_forward_v10.0.10.json"
        if not checkpoint.exists(): continue
        try:
            data=json.loads(checkpoint.read_text(encoding="utf-8"))
            folds=list(data.get("completed",{}).values())
            out[league_dir.name]=benchmark_league(folds, league_dir.name)
        except Exception as exc:
            out[league_dir.name]={"version":VERSION,"error":str(exc)}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoints",default="models/walk_forward")
    ap.add_argument("--output",default="reports/benchmark_v10.0.10.json")
    args=ap.parse_args()
    report=run_from_checkpoints(args.checkpoints)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:{"fold_count":v.get("fold_count",0),"selected":v.get("selection",{}).get("selected")} for k,v in report.items()},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
