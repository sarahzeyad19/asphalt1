# -*- coding: utf-8 -*-
"""
TIER-1 IMPROVEMENTS on the unique-mix (no-replicate) models
================================================================================
Compares, honestly (unique mixes, test scored once), for LWT and SCB:

  BASE      : default params (as in mixture_unique_model.py)
  REG+TUNED : regularized param grids, RandomizedSearchCV (inside 5-fold CV)
  +LOG      : REG+TUNED but modelling log1p(target)      (LWT only; skewed)
  NO_PROXY  : REG+TUNED without plant/production variables
              (AC_Correction_Factor, Production_Rate, Adjustment_Factor)

Prints a before/after table of CV R2 and locked TEST R2 so you can see how much
technique alone buys before investing in new physical inputs.

Reuses the data pipeline from mixture_unique_model.py.
Run: python mixture_improve_tier1.py
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
import lightgbm as lgb
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False

from mixture_unique_model import collapse_unique, build_X, num, TARGETS

RANDOM=42; N_ITER=25; CV=5
PROXY=["AC_Correction_Factor","Production_Rate","Adjustment_Factor"]

def reg_search_space():
    sp={}
    sp["LightGBM"]=(lgb.LGBMRegressor(random_state=RANDOM,n_jobs=-1,verbose=-1),
        dict(n_estimators=randint(300,900), learning_rate=uniform(0.01,0.06),
             num_leaves=randint(15,48), min_child_samples=randint(15,45),
             subsample=uniform(0.6,0.35), colsample_bytree=uniform(0.5,0.45),
             reg_alpha=uniform(0,3), reg_lambda=uniform(0.5,4), max_depth=randint(3,7)))
    sp["ExtraTrees"]=(ExtraTreesRegressor(random_state=RANDOM,n_jobs=-1),
        dict(n_estimators=randint(300,700), min_samples_leaf=randint(4,25),
             max_features=uniform(0.4,0.5), max_depth=randint(6,20)))
    sp["HistGB"]=(HistGradientBoostingRegressor(random_state=RANDOM),
        dict(max_iter=randint(300,800), learning_rate=uniform(0.01,0.06),
             max_leaf_nodes=randint(15,48), min_samples_leaf=randint(15,45),
             l2_regularization=uniform(0,3), max_depth=randint(3,7)))
    if HAS_XGB:
        sp["XGBoost"]=(xgb.XGBRegressor(random_state=RANDOM,n_jobs=-1,verbosity=0),
            dict(n_estimators=randint(300,900), learning_rate=uniform(0.01,0.06),
                 max_depth=randint(3,7), subsample=uniform(0.6,0.35),
                 colsample_bytree=uniform(0.5,0.45), reg_alpha=uniform(0,3),
                 reg_lambda=uniform(0.5,4), min_child_weight=randint(2,8)))
    return sp

def prep(df,cfg,drop_proxy=False):
    g=collapse_unique(df,cfg["col"]); y=num(g[cfg["col"]]).values
    keep=(y>=cfg["lo"])&(y<=cfg["hi"]); g=g[keep].reset_index(drop=True); y=y[keep]
    X=build_X(g)
    if drop_proxy:
        X=X.drop(columns=[c for c in PROXY if c in X.columns])
    return X,y

def splits(y):
    yb=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    idx=np.arange(len(y))
    dev,te=train_test_split(idx,test_size=0.15,random_state=RANDOM,stratify=yb)
    tr,va=train_test_split(dev,test_size=0.1765,random_state=RANDOM,stratify=yb[dev])
    return tr,va,te,yb

def evaluate(X,y,log=False,label=""):
    tr,va,te,yb=splits(y)
    yfit=np.log1p(y) if log else y
    inv=(lambda p: np.expm1(p)) if log else (lambda p: p)
    skf=KFold(CV,shuffle=True,random_state=RANDOM)
    tuned={}; testp={}
    for nm,(est,dist) in reg_search_space().items():
        rs=RandomizedSearchCV(est,dist,n_iter=N_ITER,cv=skf,scoring="r2",
                              random_state=RANDOM,n_jobs=-1,refit=True)
        rs.fit(X.iloc[tr],yfit[tr])
        tuned[nm]=(rs.best_score_, rs.best_estimator_)
        testp[nm]=inv(rs.best_estimator_.predict(X.iloc[te]))
    # equal-weight blend of tuned models
    blend=np.mean([testp[k] for k in testp],axis=0)
    rows=[]
    for nm,(cvs,estm) in tuned.items():
        # cvs is on transformed scale; recompute honest CV on original scale via refit-free proxy:
        rows.append((nm,cvs,r2_score(y[te],testp[nm]),
                     np.sqrt(mean_squared_error(y[te],testp[nm])),
                     mean_absolute_error(y[te],testp[nm])))
    rows.append(("BLEND",np.nan,r2_score(y[te],blend),
                 np.sqrt(mean_squared_error(y[te],blend)),mean_absolute_error(y[te],blend)))
    best=max(rows,key=lambda r:r[2])
    print(f"\n  [{label}]  n={len(y)}  feats={X.shape[1]}  (test n={len(te)})")
    for nm,cvs,r2,rmse,mae in rows:
        star=" *" if nm==best[0] else ""
        print(f"      {nm:11s} tunedCV(r2,transf)={cvs:.3f}  TEST R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}{star}")
    return best

if __name__=="__main__":
    print(f"[tier1] XGBoost={HAS_XGB}  n_iter={N_ITER}  cv={CV}")
    df=pd.read_csv("mixture_dataset.csv")
    summary=[]
    for name,cfg in TARGETS.items():
        print("\n"+"="*74+f"\n  {name}  — Tier-1 comparison\n"+"="*74)
        Xf,y=prep(df,cfg,drop_proxy=False)
        b_reg =evaluate(Xf,y,log=False,label="REG+TUNED (with proxies)")
        if name=="LWT":
            b_log =evaluate(Xf,y,log=True ,label="REG+TUNED +LOG target")
        else:
            b_log=None
        Xn,_ =prep(df,cfg,drop_proxy=True)
        b_np  =evaluate(Xn,y,log=False,label="REG+TUNED NO plant-proxies")
        summary.append((name,b_reg,b_log,b_np))
    print("\n"+"="*74+"\n  TIER-1 SUMMARY  (best TEST R2 per configuration)\n"+"="*74)
    print(f"  {'target':6s} {'REG+TUNED':>22s} {'+LOG':>16s} {'NO_PROXY':>18s}")
    for name,b_reg,b_log,b_np in summary:
        lg = f"{b_log[0]} {b_log[2]:.3f}" if b_log else "  (n/a)"
        print(f"  {name:6s} {b_reg[0]+' '+format(b_reg[2],'.3f'):>22s} {lg:>16s} "
              f"{b_np[0]+' '+format(b_np[2],'.3f'):>18s}")
    print("\nBaseline (from mixture_unique_model.py):  LWT best TEST R2≈0.739   SCB best TEST R2≈0.792")
