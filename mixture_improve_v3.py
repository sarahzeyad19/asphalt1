# -*- coding: utf-8 -*-
"""
IMPROVE v3 — mine existing data harder (unique-mix honest metric)
================================================================================
Three doable-now levers, evaluated on the SAME per-mix test set so results are
directly comparable, plus a lab-noise ceiling estimate.

  (1) ADDITIVE features parsed from Binder_Additive_Name / _PercentMix
      (anti-strip, WMA, fiber, polymer, additive count & dosage) + gradation
      shape (from v2).
  (2) REPLICATE-AUGMENTED training: train on ALL rows (grouped by Mix_ID so a
      test mix is never seen) but SCORE one prediction per unique test mix.
      Compared head-to-head with UNIQUE training (one row/mix) on the same
      test mixes -> isolates whether the extra rows actually help.
  (3) MONOTONIC constraints (LightGBM) encoding pavement physics for LWT
      (higher PG-high -> less rutting; more air voids -> more rutting).

  NOISE FLOOR: from mixes measured more than once with DIFFERENT target values,
  estimate the irreducible lab-test variance -> the max R2 any model could reach.

Run: python mixture_improve_v3.py
"""
import os, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False

from mixture_unique_model import build_X, num, TARGETS
from mixture_improve_v2 import add_gradation

RANDOM=42

# ---------------- additive parsing -----------------------------------------
def add_additives(df):
    name=df["Binder_Additive_Name"].astype(str)
    pct =df["Binder_Additive_PercentMix"].astype(str)
    up=name.str.upper()
    A=pd.DataFrame(index=df.index)
    A["add_antistrip"]=up.str.contains("ANTI.?STRIP|AD.?HERE|PERMA.?TAC|LA.?2|AD-HERE",regex=True).astype(float)
    A["add_wma"]      =up.str.contains("WMA|EVOTHERM|ZYCO|THERMA|WARM",regex=True).astype(float)
    A["add_fiber"]    =up.str.contains("FIBER|CELLULOSE",regex=True).astype(float)
    A["add_polymer"]  =(up.str.contains("LATEX|SBS|POLYMER|RUBBER",regex=True)
                        | df["Binder_Modification"].astype(str).str.upper().str.contains("MODIF")
                        | df["PG_Grade"].astype(str).str.contains(r"\d2m|\d2M|rm|RM",regex=True)).astype(float)
    A["add_count"]=name.apply(lambda s: max(0,str(s).count(",")))
    def doses(s):
        parts=[p.strip() for p in str(s).split(",")]
        vals=[]
        for p in parts[1:]:                      # skip first (binder %)
            v=pd.to_numeric(p,errors="coerce")
            if not pd.isna(v): vals.append(float(v))
        return (max(vals) if vals else 0.0, sum(vals) if vals else 0.0)
    d=pct.apply(doses)
    A["add_dose_max"]=d.apply(lambda t:t[0]); A["add_dose_sum"]=d.apply(lambda t:t[1])
    return A

def features(df_rows):
    X=build_X(df_rows)
    Gd=add_gradation(df_rows); Gd.index=X.index
    Ad=add_additives(df_rows); Ad.index=X.index
    X=pd.concat([X,Gd,Ad],axis=1)
    X=X.loc[:,~X.columns.duplicated()]
    X=X.replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return X

# ---------------- models ----------------------------------------------------
def make_models(mono=None):
    m={"LightGBM":lgb.LGBMRegressor(n_estimators=700,learning_rate=0.03,num_leaves=63,
          min_child_samples=15,subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,
          random_state=RANDOM,n_jobs=-1,verbose=-1),
       "ExtraTrees":ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,max_features=0.6,
          n_jobs=-1,random_state=RANDOM)}
    if HAS_XGB:
        m["XGBoost"]=xgb.XGBRegressor(n_estimators=700,learning_rate=0.03,max_depth=6,
          subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,min_child_weight=3,
          random_state=RANDOM,n_jobs=-1,verbosity=0)
    if mono is not None:
        mm=lgb.LGBMRegressor(n_estimators=700,learning_rate=0.03,num_leaves=63,
          min_child_samples=15,subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,
          monotone_constraints=mono,random_state=RANDOM,n_jobs=-1,verbose=-1)
        m={"LightGBM_mono":mm,**m}
    return m

def score(models,Xtr,ytr,Xte,yte,log):
    yfit=np.log1p(ytr) if log else ytr; inv=(np.expm1 if log else (lambda p:p))
    preds={}
    for nm,md in models.items():
        md.fit(Xtr,yfit); preds[nm]=inv(md.predict(Xte))
    preds["BLEND"]=np.mean([preds[k] for k in preds if k!="BLEND"],axis=0)
    return {k:r2_score(yte,v) for k,v in preds.items()},preds

def mono_vector(cols):
    v=[]
    for c in cols:
        if c in ("PG_High_Temp_C","PG_span"): v.append(-1)   # higher PG -> less rutting
        elif c in ("%Voids","Va"):            v.append(1)    # more voids -> more rutting
        else:                                  v.append(0)
    return v

def run(df,name,cfg):
    print("\n"+"="*74+f"\n  {name}\n"+"="*74)
    log=(name=="LWT")
    d=df[num(df[cfg["col"]]).notna()].copy(); d[cfg["col"]]=num(d[cfg["col"]])
    d=d[(d[cfg["col"]]>=cfg["lo"])&(d[cfg["col"]]<=cfg["hi"])].reset_index(drop=True)
    grp=d["Mix_ID"].astype(str).values
    X=features(d).reset_index(drop=True); y=d[cfg["col"]].values
    # per-mix median target + split MIXES (test mix never seen in training)
    mix=pd.Series(y,index=grp).groupby(level=0).median()
    mb=pd.qcut(mix.rank(method="first"),10,labels=False,duplicates="drop")
    dev_m,test_m=train_test_split(mix.index.values,test_size=0.15,random_state=RANDOM,stratify=mb)
    dev_rows=np.isin(grp,dev_m); test_rows=np.isin(grp,test_m)
    # test = one row per test mix (median features), y = median target
    Xte=X[test_rows].groupby(grp[test_rows]).median(); yte=mix.loc[Xte.index].values
    # UNIQUE training: one row per dev mix
    Xdu=X[dev_rows].groupby(grp[dev_rows]).median(); ydu=mix.loc[Xdu.index].values
    # AUGMENTED training: all dev rows
    Xda=X[dev_rows]; yda=y[dev_rows]
    print(f"  mixes: dev={len(Xdu)}  test={len(Xte)}   |  augmented train rows={len(Xda)}   feats={X.shape[1]}")

    r_u,_=score(make_models(),Xdu,ydu,Xte,yte,log)
    r_a,_=score(make_models(),Xda,yda,Xte,yte,log)
    mono=mono_vector(list(X.columns)) if log else None
    r_m,_=score(make_models(mono=mono),Xda,yda,Xte,yte,log) if mono else (None,None)

    def show(tag,r):
        if r is None: return
        best=max(r,key=r.get); print(f"  [{tag}]  best={best} R2={r[best]:.3f}   "
              +"  ".join(f"{k}={r[k]:.3f}" for k in r))
    show("UNIQUE train + new feats",r_u)
    show("AUGMENTED train (all rows)",r_a)
    if r_m: show("AUGMENTED + monotonic",r_m)
    cand={"UNIQUE":max(r_u.values()),"AUGMENTED":max(r_a.values())}
    if r_m: cand["AUG+MONO"]=max(r_m.values())
    return name,cand

def noise_floor(df,cfg):
    d=df[num(df[cfg["col"]]).notna()].copy(); d[cfg["col"]]=num(d[cfg["col"]])
    d=d[(d[cfg["col"]]>=cfg["lo"])&(d[cfg["col"]]<=cfg["hi"])]
    g=d.groupby("Mix_ID")[cfg["col"]]
    within=g.var()                      # per-mix variance
    multi=g.apply(lambda s: s.nunique()>1)
    wv=within[multi].mean()             # mean within-mix var among truly-repeated mixes
    total=d[cfg["col"]].var()
    frac_multi=multi.mean()
    ceil=1-(wv/total) if (wv==wv and total>0) else np.nan
    print(f"  {cfg['col']}: mixes remeasured w/ different values = {frac_multi*100:.0f}%  "
          f"| mean within-mix var={wv:.4f}  total var={total:.4f}  -> est. R2 ceiling ≈ {ceil:.3f}")

if __name__=="__main__":
    print(f"[v3] XGBoost={HAS_XGB}")
    df=pd.read_csv("mixture_dataset.csv")
    print("\n--- lab-test noise floor (max achievable R2) ---")
    for name,cfg in TARGETS.items(): noise_floor(df,cfg)
    res=[run(df,n,c) for n,c in TARGETS.items()]
    print("\n"+"="*74+"\n  SUMMARY — best per-mix TEST R2\n"+"="*74)
    print("  baseline (v1/v2): LWT 0.744   SCB 0.792")
    for name,cand in res:
        print(f"  {name:5s}  " + "   ".join(f"{k}={v:.3f}" for k,v in cand.items()))
