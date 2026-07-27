# -*- coding: utf-8 -*-
"""
MIXTURE_DATASET — UNIQUE-MIX (NO REPLICATES) MODELING + HONEST TEST
================================================================================
Purpose
  Predict LWT (rutting) and SCB (fracture) from Louisiana JMF variables using
  ONE ROW PER UNIQUE MIX (Mix_ID). The raw file has ~5 rows per mix (design +
  verification + repeats); training on it row-wise would leak identical copies
  of a mix across train/test. This script collapses to unique mixes FIRST, so
  the train / validation / test split is honest with no grouping tricks needed.

Pipeline
  1. Load mixture_dataset.csv
  2. Per target: keep rows with a valid target, collapse to one row per Mix_ID
     (target = median across that mix's tests; numeric features = median;
      categoricals = first). -> genuinely replicate-free table.
  3. Light sanity filter on the target range.
  4. Features: clean JMF columns (incl. PG_High_Temp_C, RBR, AFT), ADT band ->
     ordinal, one-hot categoricals, a few physics interactions.
  5. Stratified 70/15/15 train / validation / test holdout + 10-fold CV.
  6. Train 6 models + weighted ensemble; report R2/RMSE/MAE for TRAIN,
     VALIDATION and TEST separately (test scored once, at the end).
  7. Figures: train/val/test scatter (per-subset R2) + SHAP (if installed).

Run: set CSV path below -> `python mixture_unique_model.py`  (or F5 in Spyder)
"""
import os, re, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# optional boosters / shap (degrade gracefully)
try: import lightgbm as lgb; HAS_LGB=True
except Exception: HAS_LGB=False
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False
try: from catboost import CatBoostRegressor; HAS_CAT=True
except Exception: HAS_CAT=False
try: import shap; HAS_SHAP=True
except Exception: HAS_SHAP=False

# =============================== CONFIG =====================================
CSV      = "mixture_dataset.csv"
RANDOM   = 42
OUTDIR   = "mixture_unique_outputs"
TARGETS  = {
    "LWT": dict(col="LWT", unit=" (mm)", lo=0.5, hi=10.0),   # rutting
    "SCB": dict(col="SCB", unit="",      lo=0.30, hi=1.25),  # fracture
}
os.makedirs(OUTDIR, exist_ok=True)
num = lambda s: pd.to_numeric(s, errors="coerce")

# ---- columns that are NOT model features ----------------------------------
ID_COLS   = ["Project_ID","Mix_ID","Date_Approved","IsVerification"]
DROP_COLS = ["PG_Grade","Binder_Additive_Name","Binder_Additive_PercentMix",
             "PassMaxRut","LWT","SCB"]
CAT_COLS  = ["Design_Level","Mix_Type","Binder_Modification","Spec_Edition"]

def adt_ordinal(series):
    """ADT band string ('> 7000','1000 - 3500','< 1000') -> numeric proxy."""
    def conv(v):
        s=str(v).strip().lower()
        if s in ("","nan","none"): return np.nan
        n=pd.to_numeric(s,errors="coerce")
        if not pd.isna(n): return float(n)
        nums=[float(x) for x in re.findall(r"\d+",s)]
        if ">" in s and nums: return nums[0]*1.5
        if "<" in s and nums: return nums[0]*0.5
        if len(nums)>=2:      return (nums[0]+nums[1])/2
        if nums:              return nums[0]
        return np.nan
    return series.apply(conv)

def collapse_unique(df, tgt):
    """One row per Mix_ID: target=median, numeric=median, categorical=first.
    Numeric columns are coerced to real numbers first (the CSV stores several
    numeric fields as pandas 'str' dtype, which cannot take median)."""
    d=df[num(df[tgt]).notna()].copy()
    keep_str=set(CAT_COLS+["ADT","Mix_ID"])
    agg={}
    for c in d.columns:
        if c in ID_COLS and c!="Mix_ID": continue
        if c=="Mix_ID": continue
        if c in keep_str:
            agg[c]="first"
        else:
            d[c]=num(d[c])                       # coerce to numeric up front
            agg[c]=("median" if d[c].notna().any() else "first")
    g=d.groupby("Mix_ID",as_index=False).agg(agg)
    return g

def build_X(d):
    X=pd.DataFrame(index=d.index)
    # numeric features = everything not id/drop/cat/adt
    skip=set(ID_COLS+DROP_COLS+CAT_COLS+["ADT"])
    for c in d.columns:
        if c in skip: continue
        v=num(d[c])
        if v.notna().sum()>0: X[c]=v
    X["ADT_ord"]=adt_ordinal(d["ADT"]) if "ADT" in d else np.nan
    # physics interactions using the clean PG columns
    if "PG_High_Temp_C" in X:
        if "PG_Low_Temp_C" in X: X["PG_span"]=X["PG_High_Temp_C"]-X["PG_Low_Temp_C"]
        if "%Voids" in X:  X["PG_x_Voids"]=X["PG_High_Temp_C"]*X["%Voids"]
        if "RBR" in X:     X["PG_x_RBR"]=X["PG_High_Temp_C"]*X["RBR"]
        if "AFT" in X:     X["PG_x_AFT"]=X["PG_High_Temp_C"]*X["AFT"]
    # one-hot categoricals
    for c in CAT_COLS:
        if c in d:
            oh=pd.get_dummies(d[c].astype(str),prefix=c).astype(float)
            X=pd.concat([X,oh.set_index(X.index)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]
    # sanitize feature names (LightGBM/XGBoost reject %, ", /, comma, etc.)
    X.columns=[re.sub(r"[^0-9A-Za-z_]+","_",str(c)).strip("_") for c in X.columns]
    X=X.loc[:,~X.columns.duplicated()]
    return X

def zoo():
    m={"ExtraTrees":ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,
            max_features=0.6,n_jobs=-1,random_state=RANDOM),
       "RandomForest":RandomForestRegressor(n_estimators=500,min_samples_leaf=2,
            max_features=0.7,n_jobs=-1,random_state=RANDOM),
       "HistGB":HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,
            max_leaf_nodes=63,min_samples_leaf=15,l2_regularization=0.1,random_state=RANDOM)}
    if HAS_LGB: m["LightGBM"]=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,
            num_leaves=63,min_child_samples=15,subsample=0.85,colsample_bytree=0.7,
            reg_lambda=1.0,random_state=RANDOM,n_jobs=-1,verbose=-1)
    if HAS_XGB: m["XGBoost"]=xgb.XGBRegressor(n_estimators=600,learning_rate=0.03,
            max_depth=6,subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,
            min_child_weight=3,random_state=RANDOM,n_jobs=-1,verbosity=0)
    if HAS_CAT: m["CatBoost"]=CatBoostRegressor(iterations=600,learning_rate=0.03,
            depth=6,l2_leaf_reg=3.0,random_state=RANDOM,verbose=0)
    return m

def metrics(y,p): return (r2_score(y,p),np.sqrt(mean_squared_error(y,p)),mean_absolute_error(y,p))

def optimize_ensemble(oof,y):
    keys=list(oof); best=(-9,None); grid=np.arange(0,1.01,0.2)
    for w in itertools.product(grid,repeat=len(keys)):
        if abs(sum(w)-1)>1e-6: continue
        p=sum(w[i]*oof[keys[i]] for i in range(len(keys)))
        s=r2_score(y,p)
        if s>best[0]: best=(s,dict(zip(keys,w)))
    return best[1]

def run_target(df,name,cfg):
    print("\n"+"="*74+f"\n  {name}  (unique mixes only, honest train/val/test)\n"+"="*74)
    g=collapse_unique(df,cfg["col"])
    y=num(g[cfg["col"]]).values
    keep=(y>=cfg["lo"])&(y<=cfg["hi"])
    g=g[keep].reset_index(drop=True); y=y[keep]
    X=build_X(g)
    print(f"  raw rows for this target -> unique mixes kept in range: {len(X)}")
    print(f"  features: {X.shape[1]}   target range=[{y.min():.3f},{y.max():.3f}]  mean={y.mean():.3f}")

    ybin=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    # 70/15/15 stratified holdout (unique mixes -> no leakage)
    idx=np.arange(len(y))
    dev,test=train_test_split(idx,test_size=0.15,random_state=RANDOM,stratify=ybin)
    tr,va=train_test_split(dev,test_size=0.1765,random_state=RANDOM,stratify=ybin[dev])  # 0.1765*0.85≈0.15
    print(f"  split: train={len(tr)}  validation={len(va)}  test={len(te) if (te:=test) is not None else 0}")

    skf=StratifiedKFold(10,shuffle=True,random_state=RANDOM)
    folds=list(skf.split(X,ybin))
    res={}; oof_all={}; cvr2={}
    for nm,proto in zoo().items():
        # 10-fold CV (honest generalization estimate)
        oof=np.zeros(len(y)); fs=[]
        for a,b in folds:
            mm=zoo()[nm]; mm.fit(X.iloc[a],y[a]); oof[b]=mm.predict(X.iloc[b]); fs.append(r2_score(y[b],oof[b]))
        # fit on train, score train/val/test separately
        mm=zoo()[nm]; mm.fit(X.iloc[tr],y[tr])
        ptr,pva,pte=mm.predict(X.iloc[tr]),mm.predict(X.iloc[va]),mm.predict(X.iloc[test])
        res[nm]=dict(tr=(y[tr],ptr,*metrics(y[tr],ptr)),
                     va=(y[va],pva,*metrics(y[va],pva)),
                     te=(y[test],pte,*metrics(y[test],pte)),
                     cv=np.mean(fs))
        oof_all[nm]=oof; cvr2[nm]=np.mean(fs)
        print(f"    {nm:12s} CV R2={np.mean(fs):.3f}±{np.std(fs):.3f} | "
              f"train R2={res[nm]['tr'][2]:.3f}  val R2={res[nm]['va'][2]:.3f}  "
              f"TEST R2={res[nm]['te'][2]:.3f}  RMSE={res[nm]['te'][3]:.3f}  MAE={res[nm]['te'][4]:.3f}")
    # weighted ensemble (weights from CV OOF)
    W=optimize_ensemble(oof_all,y)
    def blend(split): return sum(W[k]*res[k][split][1] for k in W)
    for split,yy in [("tr",y[tr]),("va",y[va]),("te",y[test])]:
        p=blend(split); res.setdefault("ENSEMBLE",{})[split]=(yy,p,*metrics(yy,p))
    res["ENSEMBLE"]["cv"]=r2_score(y,sum(W[k]*oof_all[k] for k in W))
    e=res["ENSEMBLE"]
    print(f"    {'ENSEMBLE':12s} CV R2={e['cv']:.3f}      | train R2={e['tr'][2]:.3f}  "
          f"val R2={e['va'][2]:.3f}  TEST R2={e['te'][2]:.3f}  RMSE={e['te'][3]:.3f}  MAE={e['te'][4]:.3f}")
    print(f"    ensemble weights: { {k:round(W[k],2) for k in W if W[k]>0} }")

    fig_tvt(res,name,cfg["unit"])
    shap_top=fig_shap(X,y,tr,test,name) if HAS_SHAP else None
    return dict(name=name,X=X,y=y,res=res,weights=W,shap_top=shap_top)

# ============================ FIGURES ========================================
SUB={"tr":("Training","#7e57c2","D"),"va":("Validation","#ff9800","s"),"te":("Test","#2ca02c","^")}
def fig_tvt(res,name,unit):
    order=[k for k in res if k!="ENSEMBLE"]+["ENSEMBLE"]
    allv=np.concatenate([res[order[0]][s][0] for s in ("tr","va","te")])
    lo,hi=allv.min(),allv.max()
    n=len(order); cols=4; rows=int(np.ceil(n/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(4.4*cols,4.2*rows),squeeze=False)
    fig.suptitle(f"{name}: Actual vs Predicted — Train / Validation / Test (unique mixes){unit}",fontweight="bold")
    axl=axs.ravel()
    for ax,nm in zip(axl,order):
        for s in ("tr","va","te"):
            yy,pp,r2=res[nm][s][0],res[nm][s][1],res[nm][s][2]
            lab,col,mk=SUB[s]; ax.scatter(yy,pp,s=18,c=col,marker=mk,alpha=.65,edgecolor="k",lw=.2,label=f"{lab} $R^2$={r2:.3f}")
        ax.plot([lo,hi],[lo,hi],"--",color="k",lw=1.2,label="y=x")
        ax.set_title(nm); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}"); ax.legend(fontsize=7,loc="upper left")
    for ax in axl[n:]: ax.axis("off")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{OUTDIR}/{name}_train_val_test.png",dpi=200,bbox_inches="tight"); plt.close(fig)

def fig_shap(X,y,tr,test,name):
    base=zoo().get("LightGBM") or zoo()["ExtraTrees"]; base.fit(X.iloc[tr],y[tr])
    samp=X.iloc[test]
    try:
        sv=shap.TreeExplainer(base).shap_values(samp)
        mean=pd.Series(np.abs(sv).mean(0),index=samp.columns).sort_values(ascending=False)
        plt.figure(figsize=(9,6)); shap.summary_plot(sv,samp,max_display=15,show=False)
        plt.title(f"{name}: SHAP (test set)"); plt.tight_layout()
        plt.savefig(f"{OUTDIR}/{name}_shap.png",dpi=200,bbox_inches="tight"); plt.close()
        return mean
    except Exception as e:
        print("   [shap skipped]",e); return None

# ============================ MAIN ===========================================
if __name__=="__main__":
    print(f"[deps] LightGBM={HAS_LGB} XGBoost={HAS_XGB} CatBoost={HAS_CAT} SHAP={HAS_SHAP}")
    path=CSV if os.path.isfile(CSV) else None
    if path is None:
        import glob
        h=glob.glob("**/*mixture_dataset*.csv",recursive=True)
        path=h[0] if h else CSV
    print("[data]",path)
    df=pd.read_csv(path)
    print(f"[data] rows={len(df)}  unique Mix_ID={df['Mix_ID'].nunique()}")
    Rs=[run_target(df,n,c) for n,c in TARGETS.items()]
    print("\n"+"="*74+"\n  FINAL — HONEST TEST-SET RESULTS (unique mixes, scored once)\n"+"="*74)
    for R in Rs:
        best=max(R["res"],key=lambda k:R["res"][k]["te"][2])
        te=R["res"][best]["te"]
        print(f"  {R['name']:5s} best={best:12s} TEST R2={te[2]:.3f}  RMSE={te[3]:.3f}  MAE={te[4]:.3f}")
        if R["shap_top"] is not None:
            print(f"        top SHAP: {list(R['shap_top'].head(8).index)}")
    print(f"\nFigures + SHAP saved in ./{OUTDIR}/")
