# -*- coding: utf-8 -*-
"""
IMPROVE v4 — UNIQUE mixes only (NO augmentation) + extra engineered features
================================================================================
Per user: keep one-row-per-mix (no replicate-augmented training), keep the
additive + gradation features (which took LWT->0.809, SCB->0.854), and TRY
ADDING MORE features. Clean A/B on the SAME locked split:

   CONFIG A = build_X + gradation + additives            (the 0.809 / 0.854 set)
   CONFIG B = A + EXTRA:
        * 0.45-power maximum-density-line deviation (Superpave gradation quality)
        * per-band retained fractions across the sieve ladder
        * volumetric/binder interactions (VMA-Va, VFA*Pbe, Pbe*Dust, AFT*Dust,
          Gmm*Va, PG*Pbe, PG*Dust, MixTemp-PG, Polymer*PG, PG*RBR)

Same split, same models -> the delta is purely the new features.
LWT uses log1p target. Run: python mixture_improve_v4.py
"""
import os, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False

from mixture_unique_model import build_X, num, collapse_unique, TARGETS
from mixture_improve_v2 import add_gradation
from mixture_improve_v3 import add_additives

RANDOM=42
SIEVE_MM={'Pass 1 1/2"':37.5,'Pass 1"':25,'Pass 3/4"':19,'Pass 1/2"':12.5,
          'Pass 3/8"':9.5,'Pass No.4':4.75,'Pass No.8':2.36,'Pass No.16':1.18,
          'Pass No.30':0.60,'Pass No.50':0.30,'Pass No.100':0.15,'Pass No.200':0.075}

def add_extra(g):
    E=pd.DataFrame(index=g.index)
    P={s:num(g[s]) for s in SIEVE_MM if s in g.columns}
    order=[s for s in SIEVE_MM if s in P]
    # per-band retained fractions
    for i in range(len(order)-1):
        E[f"Ret_{i}"]=(P[order[i]]-P[order[i+1]]).clip(lower=0)
    # 0.45-power maximum-density-line deviation
    nmas=(num(g['NMAS']) if 'NMAS' in g else pd.Series(12.5,index=g.index)).replace(0,np.nan)
    devs=[]
    for s in order:
        mdl=(100*(SIEVE_MM[s]/nmas)**0.45).clip(upper=100)
        dev=P[s]-mdl; devs.append(dev); E[f"MDLdev_{re.sub(r'[^0-9A-Za-z]+','_',s)}"]=dev
    if devs:
        D=pd.concat(devs,axis=1)
        E["MDL_meanabs"]=D.abs().mean(axis=1); E["MDL_max"]=D.max(axis=1); E["MDL_area"]=D.sum(axis=1)
    c=lambda k: num(g[k]) if k in g.columns else None
    VMA,Va,VFA,Pbe,Dust,AFT,Gmm=c('VMA'),c('%Voids'),c('VFA'),c('Pbe'),c('Dust/Pbeff'),c('AFT'),c('Gmm')
    PGH,MixT,RBR=c('PG_High_Temp_C'),c('Mix_Temp'),c('RBR')
    if VMA is not None and Va is not None: E["VMA_minus_Va"]=VMA-Va
    if VFA is not None and Pbe is not None: E["VFA_x_Pbe"]=VFA*Pbe
    if Pbe is not None and Dust is not None: E["Pbe_x_Dust"]=Pbe*Dust
    if AFT is not None and Dust is not None: E["AFT_x_Dust"]=AFT*Dust
    if Gmm is not None and Va is not None: E["Gmm_x_Va"]=Gmm*Va
    if PGH is not None:
        if MixT is not None: E["MixT_minus_PG"]=MixT-PGH
        if Pbe is not None:  E["PG_x_Pbe"]=PGH*Pbe
        if Dust is not None: E["PG_x_Dust"]=PGH*Dust
        if RBR is not None:  E["PG_x_RBR2"]=PGH*RBR
        if 'Binder_Modification' in g:
            poly=g['Binder_Modification'].astype(str).str.upper().str.contains('MODIF').astype(float)
            E["Polymer_x_PG"]=poly*PGH
    return E.replace([np.inf,-np.inf],np.nan).fillna(0.0)

def features(g, extra):
    X=build_X(g)
    for blk in (add_gradation(g), add_additives(g)) + ((add_extra(g),) if extra else ()):
        blk.index=X.index; X=pd.concat([X,blk],axis=1)
    X=X.loc[:,~X.columns.duplicated()]
    X.columns=[re.sub(r"[^0-9A-Za-z_]+","_",str(cc)).strip("_") for cc in X.columns]
    X=X.loc[:,~X.columns.duplicated()]
    return X.replace([np.inf,-np.inf],np.nan).fillna(0.0)

def models():
    m={"LightGBM":lgb.LGBMRegressor(n_estimators=700,learning_rate=0.03,num_leaves=63,
          min_child_samples=15,subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,
          random_state=RANDOM,n_jobs=-1,verbose=-1),
       "ExtraTrees":ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,max_features=0.6,
          n_jobs=-1,random_state=RANDOM)}
    if HAS_XGB:
        m["XGBoost"]=xgb.XGBRegressor(n_estimators=700,learning_rate=0.03,max_depth=6,
          subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,min_child_weight=3,
          random_state=RANDOM,n_jobs=-1,verbosity=0)
    return m

def evaluate(g,y,tr,te,log,label):
    Xtr,Xte=g.iloc[tr],g.iloc[te]; ytr,yte=y[tr],y[te]
    yfit=np.log1p(ytr) if log else ytr; inv=(np.expm1 if log else (lambda p:p))
    preds={}
    for nm,md in models().items():
        md.fit(Xtr,yfit); preds[nm]=inv(md.predict(Xte))
    preds["BLEND"]=np.mean([preds[k] for k in preds if k!="BLEND"],axis=0)
    r={k:r2_score(yte,v) for k,v in preds.items()}
    best=max(r,key=r.get)
    print(f"  [{label}]  feats={g.shape[1]}  best={best} R2={r[best]:.3f}   "
          +"  ".join(f"{k}={r[k]:.3f}" for k in r),flush=True)
    return best,r[best]

def run(df,name,cfg):
    print("\n"+"="*74+f"\n  {name}\n"+"="*74)
    log=(name=="LWT")
    gc=collapse_unique(df,cfg["col"]); y=num(gc[cfg["col"]]).values
    keep=(y>=cfg["lo"])&(y<=cfg["hi"]); gc=gc[keep].reset_index(drop=True); y=y[keep]
    yb=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    tr,te=train_test_split(np.arange(len(y)),test_size=0.15,random_state=RANDOM,stratify=yb)
    XA=features(gc,extra=False); XB=features(gc,extra=True)
    bA=evaluate(XA,y,tr,te,log,"A: additive+gradation (0.809/0.854 set)")
    bB=evaluate(XB,y,tr,te,log,"B: A + EXTRA features")
    return name,bA,bB

if __name__=="__main__":
    print(f"[v4] XGBoost={HAS_XGB}  (unique mixes, no augmentation, same-split A/B)")
    df=pd.read_csv("mixture_dataset.csv")
    out=[run(df,n,c) for n,c in TARGETS.items()]
    print("\n"+"="*74+"\n  SUMMARY — best per-mix TEST R2 (same locked split)\n"+"="*74)
    print(f"  {'target':6s} {'A: add+grad':>16s} {'B: +EXTRA':>16s}  delta")
    for name,bA,bB in out:
        print(f"  {name:6s} {bA[0]+' '+format(bA[1],'.3f'):>16s} {bB[0]+' '+format(bB[1],'.3f'):>16s}  {bB[1]-bA[1]:+.3f}")
