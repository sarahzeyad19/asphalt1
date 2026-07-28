# -*- coding: utf-8 -*-
"""
GROUPED 10-FOLD CV on the per-target files (replicates KEPT, no leakage)
================================================================================
Per user: keep replicate rows and split with StratifiedGroupKFold(10) grouped by
Mix_ID, so a mix's replicates never span train and validation -> no leakage while
using every row.

Files (trimmed 43-col schema; NO binder-additive columns, so additive features
from earlier trials are unavailable here):
    mixture_dataset_lwt.csv   target LWT   (rutting, mm)
    mixture_dataset_scb.csv   target SCB   (fracture)

Features (from columns present): volumetrics + aggregate + full gradation +
gradation-SHAPE (fineness modulus, coarse/fine fractions, slopes, Bailey CA,
0.45-power max-density deviation) + clean PG (High/Low/span) + PG interactions
+ polymer flag (Binder_Modification) + one-hot Design_Level/Binder_Modification.

Reports per-model grouped 10-fold OOF R2/RMSE/MAE + fold mean +/- std, a weighted
ensemble, SHAP, and an OOF measured-vs-predicted figure. LWT uses log1p target.

Run: python mixture_grouped_cv.py
"""
import os, re, itertools, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False
try: import shap; HAS_SHAP=True
except Exception: HAS_SHAP=False

RANDOM=42; FOLDS=10; OUTDIR="mixture_grouped_cv_outputs"; os.makedirs(OUTDIR,exist_ok=True)
num=lambda s: pd.to_numeric(s,errors="coerce")
FILES={"LWT":("/root/.claude/uploads/326cb6b5-2bf6-5192-84c2-c14c3d42c3df/3f7a56c0-mixture_dataset_lwt.csv"," (mm)"),
       "SCB":("/root/.claude/uploads/326cb6b5-2bf6-5192-84c2-c14c3d42c3df/1a633dba-mixture_dataset_scb.csv","")}
CAT=["Design_Level","Binder_Modification","Mix_Type","Spec_Edition"]
IDS=["Project_ID","Mix_ID","IsVerification","Date_Approved"]
SIEVE_MM={'Pass 1 1/2"':37.5,'Pass 1"':25,'Pass 3/4"':19,'Pass 1/2"':12.5,'Pass 3/8"':9.5,
          'Pass No.4':4.75,'Pass No.8':2.36,'Pass No.16':1.18,'Pass No.30':0.60,
          'Pass No.50':0.30,'Pass No.100':0.15,'Pass No.200':0.075}

def gradation(df):
    P={s:num(df[s]) for s in SIEVE_MM if s in df.columns}; order=[s for s in SIEVE_MM if s in P]
    G=pd.DataFrame(index=df.index)
    fm=['Pass 3/4"','Pass 3/8"','Pass No.4','Pass No.8','Pass No.16','Pass No.30','Pass No.50','Pass No.100']
    have=[s for s in fm if s in P]
    if have: G["Grad_FM"]=sum((100-P[s]) for s in have)/100
    if 'Pass No.4' in P: G["Grad_CoarseFrac"]=100-P['Pass No.4']
    if 'Pass No.8' in P and 'Pass No.200' in P: G["Grad_FineFrac"]=P['Pass No.8']-P['Pass No.200']
    if 'Pass No.200' in P: G["Grad_Dust"]=P['Pass No.200']
    if 'Pass 1/2"' in P and 'Pass No.4' in P: G["Grad_CoarseSlope"]=P['Pass 1/2"']-P['Pass No.4']
    if 'Pass No.4' in P and 'Pass No.8' in P: G["Grad_FineSlope"]=P['Pass No.4']-P['Pass No.8']
    if 'Pass 1/2"' in P and 'Pass No.8' in P: G["Grad_CA"]=(P['Pass 1/2"']-P['Pass No.8'])/(100-P['Pass 1/2"']+1e-6)
    nmas=(num(df['NMAS']) if 'NMAS' in df else pd.Series(12.5,index=df.index)).replace(0,np.nan)
    devs=[]
    for s in order:
        dev=P[s]-(100*(SIEVE_MM[s]/nmas)**0.45).clip(upper=100); devs.append(dev)
    if devs:
        D=pd.concat(devs,axis=1); G["MDL_meanabs"]=D.abs().mean(axis=1); G["MDL_area"]=D.sum(axis=1)
    return G

def build_features(df, target):
    X=pd.DataFrame(index=df.index)
    skip=set(IDS+CAT+[target]+list(SIEVE_MM))
    for c in df.columns:                     # numeric mix-design columns
        if c in skip: continue
        v=num(df[c])
        if v.notna().sum()>0: X[c]=v
    for s in SIEVE_MM:                        # raw sieves (kept too)
        if s in df.columns: X[s]=num(df[s])
    X=pd.concat([X,gradation(df)],axis=1)
    # clean PG + interactions
    if "PG_High_Temp_C" in df:
        ph=num(df["PG_High_Temp_C"]); X["PG_High"]=ph
        if "PG_Low_Temp_C" in df: X["PG_span"]=ph-num(df["PG_Low_Temp_C"])
        if "%Voids" in df: X["PG_x_Voids"]=ph*num(df["%Voids"])
        if "RBR" in df:    X["PG_x_RBR"]=ph*num(df["RBR"])
        if "AFT" in df:    X["PG_x_AFT"]=ph*num(df["AFT"])
        if "Pbe" in df:    X["PG_x_Pbe"]=ph*num(df["Pbe"])
    if "Binder_Modification" in df:
        X["Polymer"]=df["Binder_Modification"].astype(str).str.upper().str.contains("MODIF").astype(float)
    for c in CAT:
        if c in df:
            X=pd.concat([X,pd.get_dummies(df[c].astype(str),prefix=c).astype(float).set_index(X.index)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]
    X.columns=[re.sub(r"[^0-9A-Za-z_]+","_",str(c)).strip("_") for c in X.columns]
    X=X.loc[:,~X.columns.duplicated()]
    return X

def zoo():
    m={"LightGBM":lgb.LGBMRegressor(n_estimators=700,learning_rate=0.03,num_leaves=63,
          min_child_samples=15,subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,
          random_state=RANDOM,n_jobs=-1,verbose=-1),
       "ExtraTrees":ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,max_features=0.6,
          n_jobs=-1,random_state=RANDOM),
       "HistGB":HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=15,l2_regularization=0.1,random_state=RANDOM)}
    if HAS_XGB:
        m["XGBoost"]=xgb.XGBRegressor(n_estimators=700,learning_rate=0.03,max_depth=6,
          subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,min_child_weight=3,
          random_state=RANDOM,n_jobs=-1,verbosity=0)
    return m

def met(y,p): return r2_score(y,p),np.sqrt(mean_squared_error(y,p)),mean_absolute_error(y,p)
def opt_ens(oof,y):
    ks=list(oof); best=(-9,None); grid=np.arange(0,1.01,0.25)
    for w in itertools.product(grid,repeat=len(ks)):
        if abs(sum(w)-1)>1e-6: continue
        s=r2_score(y,sum(w[i]*oof[ks[i]] for i in range(len(ks))))
        if s>best[0]: best=(s,dict(zip(ks,w)))
    return best[1]

def run(name):
    path,unit=FILES[name]; log=(name=="LWT")
    df=pd.read_csv(path)
    y=num(df[name]).values; grp=df["Mix_ID"].astype(str).values
    X=build_features(df,name)
    print("\n"+"="*74+f"\n  {name}  — StratifiedGroupKFold({FOLDS}), replicates KEPT (grouped by Mix_ID)\n"+"="*74)
    print(f"  rows(with replicates)={len(X)}  unique mixes={len(np.unique(grp))}  features={X.shape[1]}  "
          f"target=[{y.min():.3f},{y.max():.3f}]")
    yb=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    sgkf=StratifiedGroupKFold(FOLDS,shuffle=True,random_state=RANDOM)
    folds=list(sgkf.split(X,yb,groups=grp))
    inv=(np.expm1 if log else (lambda p:p)); yfit=np.log1p(y) if log else y
    oof_all={}; foldr2={}
    for nm in zoo():
        oof=np.zeros(len(y)); fr=[]
        for a,b in folds:
            mm=zoo()[nm]; mm.fit(X.iloc[a],yfit[a]); oof[b]=inv(mm.predict(X.iloc[b])); fr.append(r2_score(y[b],oof[b]))
        r2,rmse,mae=met(y,oof); oof_all[nm]=oof; foldr2[nm]=np.array(fr)
        print(f"    {nm:11s} OOF R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  | fold={np.mean(fr):.3f}±{np.std(fr):.3f}")
    W=opt_ens(oof_all,y); ens=sum(W[k]*oof_all[k] for k in W)
    r2,rmse,mae=met(y,ens)
    fr=[r2_score(y[b],ens[b]) for _,b in folds]
    print(f"    {'ENSEMBLE':11s} OOF R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  | fold={np.mean(fr):.3f}±{np.std(fr):.3f}")
    print(f"    weights: { {k:round(W[k],2) for k in W if W[k]>0} }")
    # SHAP (fold-0 train -> its val)
    top=None
    if HAS_SHAP:
        a,b=folds[0]; base=zoo()["LightGBM"]; base.fit(X.iloc[a],yfit[a])
        try:
            sv=shap.TreeExplainer(base).shap_values(X.iloc[b])
            top=pd.Series(np.abs(sv).mean(0),index=X.columns).sort_values(ascending=False)
            plt.figure(figsize=(9,6)); shap.summary_plot(sv,X.iloc[b],max_display=15,show=False)
            plt.title(f"{name}: SHAP (grouped-CV fold)"); plt.tight_layout()
            plt.savefig(f"{OUTDIR}/{name}_shap.png",dpi=200,bbox_inches="tight"); plt.close()
        except Exception as e: print("   [shap skipped]",e)
    # OOF scatter (ensemble)
    plt.figure(figsize=(6.5,6.5)); lo,hi=y.min(),y.max()
    plt.scatter(y,ens,s=14,c="#2E6DA4",alpha=.5,edgecolor="k",lw=.2)
    plt.plot([lo,hi],[lo,hi],"r--",lw=1.4,label="y=x")
    a2,b2=np.polyfit(y,ens,1); xs=np.array([lo,hi]); plt.plot(xs,a2*xs+b2,"k-",lw=1.2,label=f"fit R²={r2:.3f}")
    plt.xlabel(f"Actual{unit}"); plt.ylabel(f"Predicted{unit}"); plt.legend()
    plt.title(f"{name}: grouped 10-fold OOF (replicates kept)")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/{name}_oof.png",dpi=200,bbox_inches="tight"); plt.close()
    return name,r2,rmse,np.mean(fr),np.std(fr),top

if __name__=="__main__":
    print(f"[grouped-cv] XGBoost={HAS_XGB} SHAP={HAS_SHAP}  folds={FOLDS}")
    res=[run(n) for n in FILES]
    print("\n"+"="*74+"\n  SUMMARY — grouped 10-fold OOF (replicates kept, no leakage)\n"+"="*74)
    for name,r2,rmse,fm,fs,top in res:
        print(f"  {name:5s} ENSEMBLE OOF R2={r2:.3f}  RMSE={rmse:.3f}  fold={fm:.3f}±{fs:.3f}")
        if top is not None: print(f"        top SHAP: {list(top.head(8).index)}")
    print(f"\nFigures in ./{OUTDIR}/")
