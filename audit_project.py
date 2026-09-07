#!/usr/bin/env python3
"""Edge90 v7 integrity audit: data quality, duplicates, feature count, and syntax."""
from pathlib import Path
import ast, zipfile, hashlib, json
import pandas as pd
ROOT=Path(__file__).parent
errors=[]; warnings=[]
for p in ROOT.rglob('*.py'):
    try: ast.parse(p.read_text(encoding='utf-8'))
    except Exception as e: errors.append(f"Syntax: {p}: {e}")
for p in sorted((ROOT/'data').glob('*_Master.csv')):
    try:
        df=pd.read_csv(p)
        required={'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'}
        miss=required-set(df.columns)
        if miss: errors.append(f"{p.name}: missing {sorted(miss)}")
        if df.duplicated(['Date','HomeTeam','AwayTeam']).sum(): warnings.append(f"{p.name}: duplicate raw keys")
        if df[['FTHG','FTAG']].lt(0).any().any(): errors.append(f"{p.name}: negative goals")
        bad=df.apply(lambda r: (r.FTR=='H' and r.FTHG<=r.FTAG) or (r.FTR=='D' and r.FTHG!=r.FTAG) or (r.FTR=='A' and r.FTHG>=r.FTAG),axis=1).sum()
        if bad: errors.append(f"{p.name}: {bad} inconsistent FTR rows")
        print(f"OK {p.name}: {len(df)} rows")
    except Exception as e: errors.append(f"Data: {p}: {e}")
# Verify 126-feature declaration and no legacy v5 paths in active config
cfg=json.loads((ROOT/'leagues_config.json').read_text(encoding='utf-8'))
if cfg.get('global_settings',{}).get('n_features')!=126: errors.append('Config n_features != 126')
for code,v in cfg.get('leagues',{}).items():
    if '_v5' in str(v): errors.append(f'{code}: legacy v5 artifact configured')
print(f"Warnings: {len(warnings)} | Errors: {len(errors)}")
for x in warnings: print('WARN',x)
for x in errors: print('ERROR',x)
raise SystemExit(1 if errors else 0)
