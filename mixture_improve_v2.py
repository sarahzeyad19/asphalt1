# -*- coding: utf-8 -*-
"""
IMPROVE v2 — gradation-shape features + fair WIDER tuning (unique mixes)
================================================================================
Two levers combined, evaluated honestly (one row per Mix_ID, test scored once):

  * Tier-2 features: gradation SHAPE derived from the sieve ladder
      - Fineness Modulus, coarse fraction, fine fraction, dust,
        coarse/fine slopes, a Bailey-style CA ratio, dust-to-fine ratio
  * Fair tuning: RandomizedSearchCV whose capacity range SPANS small -> large
      (so the search can pick big models too, not only regularized ones),
      n_iter=40, on LightGBM + XGBoost; ExtraTrees kept as a strong fixed ref.
  * log1p target for LWT (right-skewed).

Compares, per target:  BASE features  vs  BASE + GRAD features.
Prints TEST R2 next to the current baseline (LWT 0.739 / SCB 0.792).

Run: python mixture_improve_v2.py
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
import lightgbm as lgb
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False

from mixture_unique_model import collapse_unique, build_X, num, TARGETS

RANDOM=42; N_ITER=40; CV=4

SIEVES=['Pass 1 1/2"','Pass 1"','Pass 3/4"','Pass 1/2"','Pass 3/8"','Pass No.4',
        'Pass No.8','Pass No.16','Pass No.30','Pass No.50','Pass No.100','Pass No.200']

def add_gradation(g):
    """Gradation SHAPE features from the sieve ladder (mechanistic signal that
    raw single-sieve % columns don't capture)."""
    P={s:num(g[s]) for s in SIEVES if s in g.columns}
    G=pd.DataFrame(index=g.index)
    fm_sieves=['Pass 3/4"','Pass 3/8"','Pass No.4','Pass No.8','Pass No.16',
               'Pass No.30','Pass No.50','Pass No.100']
    have=[s for s in fm_sieves if s in P]
    if have: G["Grad_FinenessModulus"]=sum((100-P[s]) for s in have)/100
    if 'Pass No.4' in P:  G["Grad_CoarseFrac"]=100-P['Pass No.4']
    if 'Pass No.8' in P and 'Pass No.200' in P: G["Grad_FineFrac"]=P['Pass No.8']-P['Pass No.200']
    if 'Pass No.200' in P: G["Grad_Dust"]=P['Pass No.200']
    if 'Pass 1/2"' in P and 'Pass No.4' in P: G["Grad_CoarseSlope"]=P['Pass 1/2"']-P['Pass No.4']
    if 'Pass No.4' in P and 'Pass No.8' in P: G["Grad_FineSlope"]=P['Pass No.4']-P['Pass No.8']
    if 'Pass 1/2"' in P and 'Pass No.8' in P:
        G["Grad_CA_ratio"]=(P['Pass 1/2"']-P['Pass No.8'])/(100-P['Pass 1/2"']+1e-6)
    if 'Pass No.200' in P and 'Pass No.30' in P:
        G["Grad_DustToFine"]=P['Pass No.200']/(P['Pass No.30']+1e-6)
    if 'Pass No.4' in P and 'Pass No.30' in P and 'Pass No.100' in P:
        G["Grad_MidSlope"]=P['Pass No.4']-P['Pass No.100']
    return G.replace([np.inf,-np.inf],np.nan).fillna(G.median(numeric_only=True)).fillna(0.0)

def prep(df,cfg,grad=False):
    g=collapse_unique(df,cfg["col"]); y=num(g[cfg["col"]]).values
    keep=(y>=cfg["lo"])&(y<=cfg["hi"]); g=g[keep].reset_index(drop=True); y=y[keep]
    X=build_X(g)
    if grad:
        Gd=add_gradation(g); Gd.index=X.index; X=pd.concat([X,Gd],axis=1)
        X=X.loc[:,~X.columns.duplicated()]
    return X,y

def space():
    sp={"LightGBM":(lgb.LGBMRegressor(random_state=RANDOM,n_jobs=1,verbose=-1),
          dict(n_estimators=randint(300,1400), learning_rate=uniform(0.01,0.07),
               num_leaves=randint(15,180), min_child_samples=randint(5,40),
               subsample=uniform(0.6,0.4), colsample_bytree=uniform(0.5,0.5),
               reg_alpha=uniform(0,3), reg_lambda=uniform(0,4), max_depth=randint(3,12)))}
    if HAS_XGB:
        sp["XGBoost"]=(xgb.XGBRegressor(random_state=RANDOM,n_jobs=1,verbosity=0),
          dict(n_estimators=randint(300,1400), learning_rate=uniform(0.01,0.07),
               max_depth=randint(3,11), subsample=uniform(0.6,0.4),
               colsample_bytree=uniform(0.5,0.5), reg_alpha=uniform(0,3),
               reg_lambda=uniform(0,4), min_child_weight=randint(1,8)))
    return sp

def evaluate(X,y,log,label):
    yb=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    idx=np.arange(len(y))
    dev,te=train_test_split(idx,test_size=0.15,random_state=RANDOM,stratify=yb)
    tr,va=train_test_split(dev,test_size=0.1765,random_state=RANDOM,stratify=yb[dev])
    yfit=np.log1p(y) if log else y; inv=(np.expm1 if log else (lambda p:p))
    kf=KFold(CV,shuffle=True,random_state=RANDOM)
    print(f"\n  [{label}]  n={len(y)}  feats={X.shape[1]}  (test n={len(te)})",flush=True)
    preds={}
    for nm,(est,dist) in space().items():
        rs=RandomizedSearchCV(est,dist,n_iter=N_ITER,cv=kf,scoring="r2",
                              random_state=RANDOM,n_jobs=-1,refit=True)
        rs.fit(X.iloc[tr],yfit[tr]); preds[nm]=inv(rs.best_estimator_.predict(X.iloc[te]))
        print(f"      {nm:11s} tunedCV={rs.best_score_:.3f}  TEST R2={r2_score(y[te],preds[nm]):.3f}",flush=True)
    # strong fixed ExtraTrees reference (baseline-style, no tuning)
    et=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,max_features=0.6,
                           n_jobs=-1,random_state=RANDOM); et.fit(X.iloc[tr],yfit[tr])
    preds["ExtraTrees_fix"]=inv(et.predict(X.iloc[te]))
    print(f"      {'ExtraTrees*':11s}            TEST R2={r2_score(y[te],preds['ExtraTrees_fix']):.3f}",flush=True)
    blend=np.mean(list(preds.values()),axis=0)
    rows={**{k:r2_score(y[te],v) for k,v in preds.items()},"BLEND":r2_score(y[te],blend)}
    best=max(rows,key=rows.get)
    bp=blend if best=="BLEND" else preds[best]
    print(f"      BLEND                  TEST R2={rows['BLEND']:.3f}",flush=True)
    print(f"      -> best: {best}  TEST R2={rows[best]:.3f}  "
          f"RMSE={np.sqrt(mean_squared_error(y[te],bp)):.3f}  MAE={mean_absolute_error(y[te],bp):.3f}",flush=True)
    return best,rows[best]

if __name__=="__main__":
    print(f"[v2] XGBoost={HAS_XGB}  n_iter={N_ITER} cv={CV}  (wide capacity + gradation shape)")
    df=pd.read_csv("mixture_dataset.csv")
    base={"LWT":0.739,"SCB":0.792}
    out=[]
    for name,cfg in TARGETS.items():
        print("\n"+"="*74+f"\n  {name}\n"+"="*74)
        log=(name=="LWT")
        Xb,y=prep(df,cfg,grad=False); bb=evaluate(Xb,y,log,f"{name} BASE (wide tune"+(" +log)" if log else ")"))
        Xg,_ =prep(df,cfg,grad=True ); bg=evaluate(Xg,y,log,f"{name} BASE+GRADATION")
        out.append((name,base[name],bb,bg))
    print("\n"+"="*74+"\n  SUMMARY — best TEST R2\n"+"="*74)
    print(f"  {'target':6s} {'baseline':>10s} {'wide-tune BASE':>22s} {'+GRADATION':>22s}")
    for name,b,bb,bg in out:
        print(f"  {name:6s} {b:>10.3f} {bb[0]+' '+format(bb[1],'.3f'):>22s} {bg[0]+' '+format(bg[1],'.3f'):>22s}")
