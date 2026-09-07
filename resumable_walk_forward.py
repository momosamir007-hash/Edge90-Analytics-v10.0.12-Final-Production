"""Edge90 v10.0.7 - resumable, leakage-safe walk-forward runner."""
from __future__ import annotations
import json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

class ResumableWalkForwardRunner:
    VERSION = "v10.0.7"
    def __init__(self, checkpoint_path):
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic(self, payload):
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        fd, tmp = tempfile.mkstemp(prefix=self.path.name, dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def load(self):
        if not self.path.exists():
            return {"version": self.VERSION, "completed": {}, "failed": {}, "created_at": datetime.now(timezone.utc).isoformat()}
        with self.path.open(encoding="utf-8") as f: return json.load(f)

    def completed_ids(self): return set(self.load().get("completed", {}))

    def save_fold(self, fold_id, metrics):
        state=self.load(); state.setdefault("completed", {})[str(fold_id)] = metrics
        state.setdefault("failed", {}).pop(str(fold_id), None); self._atomic(state)

    def save_failure(self, fold_id, error):
        state=self.load(); state.setdefault("failed", {})[str(fold_id)]={"error":str(error),"at":datetime.now(timezone.utc).isoformat()}; self._atomic(state)

    def aggregate(self):
        rows=list(self.load().get("completed", {}).values())
        if not rows: return {"fold_count":0,"total_eval":0}
        weights=np.array([max(1,float(r.get("eval_size",1))) for r in rows])
        out={"version":self.VERSION,"fold_count":len(rows),"total_eval":int(weights.sum()),"folds":sorted(rows,key=lambda r:r.get("fold",0))}
        numeric=set().union(*(r.keys() for r in rows)) - {"fold","train_size","eval_size","draw_trace"}
        for k in numeric:
            vals=[]; ws=[]
            for r,w in zip(rows,weights):
                if isinstance(r.get(k),(int,float)) and np.isfinite(r[k]): vals.append(float(r[k])); ws.append(w)
            if vals: out[k]=float(np.average(vals,weights=ws))
        stages=set().union(*(r.get("draw_trace",{}).keys() for r in rows))
        out["draw_trace"]={s:float(np.average([r.get("draw_trace",{}).get(s,0.0) for r in rows],weights=weights)) for s in stages}
        return out

    def run(self, matches, fold_plan, evaluate_fold):
        done=self.completed_ids()
        for spec in fold_plan:
            fid=str(spec["fold"])
            if fid in done: continue
            try:
                metrics=evaluate_fold(spec)
                if metrics.get("error"): raise RuntimeError(metrics["error"])
                self.save_fold(fid, metrics)
            except Exception as exc:
                self.save_failure(fid, exc)
                raise
        return self.aggregate()
