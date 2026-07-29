# -*- coding: utf-8 -*-
"""
ASPHALT v2 — ENGINEERED WORKFLOW (LWT + SCB)
Structure requested by the user:
  Raw JMF data
  -> specification / quality screening
  -> removal of special mixture families and target-invalid rows
  -> replicate-aware grouped 80/20 split by Mix_ID (locked test)
  -> engineering feature construction by asphalt mechanism
  -> fold-specific imputation + preprocessing (leakage-safe)
  -> filter-based screening: Spearman + missingness
  -> soft multicollinearity control: correlation + VIF
  -> engineering-protected feature selection (per RUT vs SCB mechanism)
  -> grouped wrapper/embedded model selection
  -> model-specific GridSearchCV tuning inside grouped CV (GroupKFold)
  -> smallest stable engineering-sensible model selection
  -> ONE final locked-test evaluation
  -> SHAP + permutation importance, PDP/ALE, residual & subgroup analysis
Dataset: mixture_dataset_lwt_scb_COMBINED_..._MODEL_READY.xlsx
  LWT_Combined sheet: 3571 rows / 2129 mixes
  SCB_Combined sheet: 1889 rows / 1813 mixes
Replicates KEPT + grouped by Mix_ID everywhere (no leakage).

Run: F5 in Spyder. Runtime ~10-20 min per target (GridSearch).
"""
import os,re,glob,itertools,warnings,json
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr, gaussian_kde
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import (StratifiedGroupKFold, GroupKFold, GridSearchCV,
                                     GroupShuffleSplit)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import lightgbm as lgb
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False
try: from catboost import CatBoostRegressor; HAS_CAT=True
except Exception: HAS_CAT=False
try: import shap; HAS_SHAP=True
except Exception: HAS_SHAP=False

# ============================== CONFIG ======================================
XL="mixture_dataset_lwt_scb_COMBINED_PLUS_LWT_SCB_CLEANED5_MODEL_READY.xlsx"
FOLDS=10; RANDOM=42; OUTDIR="asphalt_v2_outputs"; os.makedirs(OUTDIR,exist_ok=True)
HEADLESS=False
RANGE={"LWT":(0.5,11.0),"SCB":(0.30,1.25)}     # step 3: target-invalid rows removed
LOG_LWT=False
GS_CV=5                                         # inner GroupKFold for GridSearchCV
rcParams.update({"figure.dpi":110,"savefig.dpi":300,"font.size":12,"axes.titlesize":13,
                 "axes.labelsize":12,"legend.fontsize":9,"axes.grid":True,"grid.alpha":0.25,
                 "font.family":"DejaVu Sans"})
num=lambda s: pd.to_numeric(s,errors="coerce")

def resolve(name):
    if os.path.isfile(name): return name
    for r in [os.getcwd(),os.path.expanduser("~/Downloads"),os.path.expanduser("~")]:
        h=glob.glob(os.path.join(r,"**",os.path.basename(name)),recursive=True)
        if h: return h[0]
    return name

# ==================== STEP 5: engineered features ===========================
SIEVE_MM={'Pass 1 1/2"':37.5,'Pass 1"':25,'Pass 3/4"':19,'Pass 1/2"':12.5,'Pass 3/8"':9.5,
          'Pass No.4':4.75,'Pass No.8':2.36,'Pass No.16':1.18,'Pass No.30':0.60,
          'Pass No.50':0.30,'Pass No.100':0.15,'Pass No.200':0.075}
CAT=["Design_Level","Binder_Modification"]
IDS=["Project_ID","Mix_ID","Source_Dataset"]
def gradation_shape(df):
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
    devs=[P[s]-(100*(SIEVE_MM[s]/nmas)**0.45).clip(upper=100) for s in order]
    if devs:
        D=pd.concat(devs,axis=1); G["MDL_meanabs"]=D.abs().mean(axis=1); G["MDL_area"]=D.sum(axis=1)
    return G

def build_features(df,target):
    X=pd.DataFrame(index=df.index); skip=set(IDS+CAT+["LWT","SCB"]+list(SIEVE_MM))
    for c in df.columns:
        if c in skip: continue
        v=num(df[c])
        if v.notna().sum()>0: X[c]=v
    for s in SIEVE_MM:
        if s in df.columns: X[s]=num(df[s])
    X=pd.concat([X,gradation_shape(df)],axis=1)
    if "PG_High_Temp_C" in df:
        ph=num(df["PG_High_Temp_C"])
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
    X=X.replace([np.inf,-np.inf],np.nan)
    X=X.loc[:,X.notna().mean()>=0.5]                            # step 7: missingness screen
    X.columns=[re.sub(r"[^0-9A-Za-z_]+","_",str(c)).strip("_") for c in X.columns]
    X=X.loc[:,~X.columns.duplicated()]
    return X

# ==================== STEPS 7-9: SCREENING (VIF + Spearman + protected list) =
# Protected features per target — engineering essential; never dropped by VIF.
EXEMPT_BASE={
 "LWT":["PG_High_Temp_C","PG_Low_Temp_C","PG_span","PG_x_Voids","PG_x_RBR","PG_x_AFT","PG_x_Pbe",
        "RAP","Total_AC_from_RAP","RBR","AFT","Voids","VFA","Pbe","AC","Gse","Dust_Pbeff",
        "Combined_SandEq","Grad_CoarseSlope","Grad_FM","Polymer"],
 "SCB":["PG_High_Temp_C","PG_Low_Temp_C","PG_span","PG_x_Voids","PG_x_RBR","PG_x_AFT","PG_x_Pbe",
        "RAP","Total_AC_from_RAP","RBR","AFT","Voids","VFA","Pbe","AC","Gse","Dust_Pbeff",
        "Combined_SandEq","Pass_No_30","Grad_CA","Grad_FM","Polymer",
        "Binder_Modification_Modified","Binder_Modification_Unmodified"],
}
def compute_vif(Xnum):
    C=np.corrcoef(Xnum.values,rowvar=False); C=np.nan_to_num(C,nan=0.0); np.fill_diagonal(C,1.0)
    return pd.Series(np.diag(np.linalg.pinv(C)),index=Xnum.columns)

def screen(X,y,target,vif_thresh=10.0,corr_thresh=0.95):
    exempt=set(EXEMPT_BASE[target])
    dummies=[c for c in X.columns if c.startswith(("Design_Level_","Binder_Modification_"))]
    sp=X.apply(lambda c: abs(spearmanr(c,y).correlation) if np.std(c)>0 else 0.0).fillna(0.0).sort_values(ascending=False)
    # (A) correlation prune: drop redundant of any pair r>corr_thresh (never drop exempt)
    corr=X.corr().abs(); drop_corr=set()
    cols=list(X.columns)
    for i,a in enumerate(cols):
        if a in exempt or a in dummies or a in drop_corr: continue
        for b in cols[i+1:]:
            if b in drop_corr: continue
            if corr.at[a,b]>corr_thresh:
                # keep the one with stronger relevance to target; drop the other unless it's exempt
                loser=a if sp[a]<sp[b] else b
                if loser in exempt or loser in dummies: continue
                drop_corr.add(loser)
    # (B) VIF prune (protect exempt+dummies)
    cur=[c for c in X.columns if c not in drop_corr and c not in dummies]; dropped_vif=[]
    while True:
        vif=compute_vif(X[cur])
        cand=vif[[c for c in cur if c not in exempt]]
        if len(cand)==0 or float(cand.max())<=vif_thresh: break
        w=cand.idxmax(); cur.remove(w); dropped_vif.append((w,round(float(cand.max()),1)))
    kept=[c for c in X.columns if c in cur or c in dummies]
    vif_f=compute_vif(X[cur]) if len(cur)>=2 else pd.Series(dtype=float)
    return kept, dict(spearman=sp, vif_final=vif_f, dropped_corr=sorted(drop_corr),
                      dropped_vif=dropped_vif, protected=[c for c in cur if c in exempt])

# ==================== STEPS 6, 8: preprocessing + fold-safe pipeline ========
def make_pipe(estimator, num_cols, cat_cols):
    pre=ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler(with_mean=False))]), num_cols),
        ("cat", SimpleImputer(strategy="most_frequent"), cat_cols)
    ], remainder="drop", sparse_threshold=0)
    return Pipeline([("pre",pre),("est",estimator)])

# ==================== STEPS 11-12: models + TUNING ==========================
def model_grids():
    """Strong-regularization GridSearchCV spaces — kept SMALL (each grid <=12
    combos) so total runtime with 3 inner folds stays tractable on ~3.5k rows."""
    g={
    # 8 combos
    "LightGBM":(lgb.LGBMRegressor(random_state=RANDOM,n_jobs=1,verbose=-1,n_estimators=800),{
        "est__learning_rate":[0.02,0.04],
        "est__num_leaves":[31,63],
        "est__min_child_samples":[15,30],
        "est__reg_lambda":[1.0,2.0]}),
    # 8 combos
    "HistGB":(HistGradientBoostingRegressor(random_state=RANDOM,max_iter=800),{
        "est__learning_rate":[0.02,0.04],
        "est__max_leaf_nodes":[31,63],
        "est__min_samples_leaf":[15,30],
        "est__l2_regularization":[0.1,1.0]}),
    # 6 combos
    "ExtraTrees":(ExtraTreesRegressor(random_state=RANDOM,n_jobs=1,n_estimators=500),{
        "est__min_samples_leaf":[1,2,4],
        "est__max_features":[0.5,0.7]}),
    }
    if HAS_XGB:
        # 8 combos
        g["XGBoost"]=(xgb.XGBRegressor(random_state=RANDOM,n_jobs=1,verbosity=0,n_estimators=800),{
            "est__learning_rate":[0.02,0.04],
            "est__max_depth":[5,7],
            "est__subsample":[0.7,0.85],
            "est__reg_lambda":[1.0,2.0]})
    if HAS_CAT:
        # 8 combos
        g["CatBoost"]=(CatBoostRegressor(random_state=RANDOM,verbose=0,iterations=800),{
            "est__learning_rate":[0.02,0.04],
            "est__depth":[5,7],
            "est__l2_leaf_reg":[2.0,5.0]})
    return g

def opt_ens(oof,y):
    ks=list(oof); best=(-9,None); grid=np.arange(0,1.01,0.25)
    for w in itertools.product(grid,repeat=len(ks)):
        if abs(sum(w)-1)>1e-6: continue
        s=r2_score(y,sum(w[i]*oof[ks[i]] for i in range(len(ks))))
        if s>best[0]: best=(s,dict(zip(ks,w)))
    return best[1]

# ==================== main compute ==========================================
STYLE={"ExtraTrees":("s","#1f77b4"),"RandomForest":("^","#2ca02c"),"LightGBM":("*","#17becf"),
       "XGBoost":("o","#9467bd"),"CatBoost":("h","#bcbd22"),"HistGB":("<","#ff7f0e"),"ENSEMBLE":("D","#d62728")}
SUBSET={"Training":("D","#7e57c2"),"Test":("^","#2ca02c")}

def run_target(target,unit):
    print("\n"+"="*80+f"\n  {target}  ({'rutting' if target=='LWT' else 'cracking'})  — engineered pipeline\n"+"="*80)
    # STEP 1-3: load, spec/quality screening, target-range filter
    df=pd.read_excel(resolve(XL),sheet_name=f"{target}_Combined")
    print(f"  raw: {len(df)} rows, {df['Mix_ID'].nunique()} unique mixes")
    y=num(df[target]).values; lo,hi=RANGE[target]
    mask=np.isfinite(y)&(y>=lo)&(y<=hi)
    df=df[mask].reset_index(drop=True); y=y[mask]
    grp=df["Mix_ID"].astype(str).values
    print(f"  after target-range filter [{lo},{hi}]: {len(df)} rows, {len(np.unique(grp))} mixes")
    # STEP 4: replicate-aware 80/20 grouped locked-test split
    gss=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=RANDOM)
    dev_idx,test_idx=next(gss.split(df,y,groups=grp))
    print(f"  grouped 80/20 split: dev={len(dev_idx)} rows ({len(np.unique(grp[dev_idx]))} mixes), "
          f"locked-test={len(test_idx)} rows ({len(np.unique(grp[test_idx]))} mixes)")
    # STEP 5: engineered features
    Xall=build_features(df,target); num_cols=list(Xall.columns)
    print(f"  features built: {Xall.shape[1]}")
    # STEP 7-9: screening on the DEV portion only (no leakage)
    kept,screen_info=screen(Xall.iloc[dev_idx],y[dev_idx],target)
    Xall=Xall[kept]; num_cols=[c for c in num_cols if c in kept]
    print(f"  screening: dropped {len(screen_info['dropped_corr'])} by |r|>0.95, "
          f"{len(screen_info['dropped_vif'])} by VIF>10 (protected list saved {len(screen_info['protected'])}); "
          f"kept {len(kept)} features")
    log=(target=="LWT" and LOG_LWT); inv=(np.expm1 if log else (lambda p:p)); yfit=np.log1p(y) if log else y
    # STEP 11: model-specific GridSearchCV inside grouped CV
    inner=GroupKFold(GS_CV)
    outer=StratifiedGroupKFold(FOLDS,shuffle=True,random_state=RANDOM)
    ybin=pd.qcut(pd.Series(y[dev_idx]).rank(method="first"),10,labels=False,duplicates="drop").values
    outer_folds=list(outer.split(Xall.iloc[dev_idx],ybin,groups=grp[dev_idx]))
    print(f"\n  --- GridSearchCV tuning (GroupKFold {GS_CV}, groups=Mix_ID) ---")
    best_est={}; best_params={}
    for nm,(est,g) in model_grids().items():
        pipe=make_pipe(est,num_cols,[])
        gs=GridSearchCV(pipe,g,cv=inner,scoring="r2",n_jobs=-1,refit=True)
        gs.fit(Xall.iloc[dev_idx],yfit[dev_idx],groups=grp[dev_idx])
        best_est[nm]=gs.best_estimator_; best_params[nm]=gs.best_params_
        print(f"    tuned {nm:12s} inner CV r2={gs.best_score_:.3f}")
    # STEP 12/13: grouped 10-fold OOF on DEV + one locked-test eval
    print(f"\n  --- grouped {FOLDS}-fold OOF on DEV + LOCKED-TEST eval ---")
    fit={}; cv={}; oof_all={}
    for nm,mdl in best_est.items():
        oof=np.zeros(len(dev_idx)); trs=[]
        for a,b in outer_folds:
            m=clone(mdl); m.fit(Xall.iloc[dev_idx[a]],yfit[dev_idx[a]])
            oof[b]=inv(m.predict(Xall.iloc[dev_idx[b]])); trs.append(r2_score(y[dev_idx[a]],inv(m.predict(Xall.iloc[dev_idx[a]]))))
        # final: fit on full DEV, predict LOCKED TEST
        m=clone(mdl); m.fit(Xall.iloc[dev_idx],yfit[dev_idx])
        p_test=inv(m.predict(Xall.iloc[test_idx]))
        oof_all[nm]=oof
        cv[nm]=np.array([r2_score(y[dev_idx[b]],oof[b]) for _,b in outer_folds])
        fit[nm]=dict(oof=oof,R2_oof=r2_score(y[dev_idx],oof),
                     RMSE_oof=np.sqrt(mean_squared_error(y[dev_idx],oof)),
                     MAE_oof=mean_absolute_error(y[dev_idx],oof),
                     R2_train=np.mean(trs),fold_r2=cv[nm],
                     R2_test=r2_score(y[test_idx],p_test),
                     RMSE_test=np.sqrt(mean_squared_error(y[test_idx],p_test)),
                     MAE_test=mean_absolute_error(y[test_idx],p_test),
                     p_test=p_test)
        print(f"    {nm:12s} train R2={np.mean(trs):.3f}  OOF R2={fit[nm]['R2_oof']:.3f}  "
              f"LOCKED-TEST R2={fit[nm]['R2_test']:.3f}  RMSE={fit[nm]['RMSE_test']:.3f}")
    # weighted ENSEMBLE (weights on OOF, applied to test)
    W=opt_ens(oof_all,y[dev_idx])
    ens_oof=sum(W[k]*oof_all[k] for k in W); ens_test=sum(W[k]*fit[k]['p_test'] for k in W)
    fit["ENSEMBLE"]=dict(oof=ens_oof,R2_oof=r2_score(y[dev_idx],ens_oof),
        RMSE_oof=np.sqrt(mean_squared_error(y[dev_idx],ens_oof)),MAE_oof=mean_absolute_error(y[dev_idx],ens_oof),
        R2_train=np.nan,fold_r2=np.array([r2_score(y[dev_idx[b]],ens_oof[b]) for _,b in outer_folds]),
        R2_test=r2_score(y[test_idx],ens_test),RMSE_test=np.sqrt(mean_squared_error(y[test_idx],ens_test)),
        MAE_test=mean_absolute_error(y[test_idx],ens_test),p_test=ens_test)
    cv["ENSEMBLE"]=fit["ENSEMBLE"]["fold_r2"]
    print(f"    ENSEMBLE     OOF R2={fit['ENSEMBLE']['R2_oof']:.3f}  "
          f"LOCKED-TEST R2={fit['ENSEMBLE']['R2_test']:.3f}  RMSE={fit['ENSEMBLE']['RMSE_test']:.3f}  "
          f"weights={ {k:round(W[k],2) for k in W if W[k]>0} }")
    # STEP 14: SHAP + permutation on DEV
    shap_mean=X_shap=sv=perm=None
    lgb_bt=best_est.get("LightGBM"); pdp_model=clone(lgb_bt) if lgb_bt else clone(list(best_est.values())[0])
    pdp_model.fit(Xall.iloc[dev_idx],y[dev_idx])
    if HAS_SHAP:
        samp_idx=np.random.RandomState(RANDOM).choice(len(dev_idx),min(600,len(dev_idx)),replace=False)
        X_shap=Xall.iloc[dev_idx[samp_idx]]
        try:
            sv=shap.TreeExplainer(pdp_model.named_steps["est"]).shap_values(
                pdp_model.named_steps["pre"].transform(X_shap))
            feat_names=list(Xall.columns)
            shap_mean=pd.Series(np.abs(sv).mean(0),index=feat_names).sort_values(ascending=False)
        except Exception as e: print("   [shap]",e)
    try:
        pi=permutation_importance(pdp_model,Xall.iloc[dev_idx],yfit[dev_idx],n_repeats=5,random_state=RANDOM,n_jobs=-1)
        perm=pd.Series(pi.importances_mean,index=Xall.columns).sort_values(ascending=False)
    except Exception as e: print("   [perm]",e)
    return dict(name=target,unit=unit,X=Xall,y=y,grp=grp,dev=dev_idx,test=test_idx,
                fit=fit,cv=cv,W=W,best_params=best_params,screen=screen_info,
                pdp_model=pdp_model,pdp_bg=Xall.iloc[dev_idx],
                shap=sv,X_shap=X_shap,shap_mean=shap_mean,perm=perm)

# ============================== FIGURES =====================================
def _save(fig,name,tag):
    fig.savefig(f"{OUTDIR}/{name}_{tag}.png",bbox_inches="tight")
    if HEADLESS: plt.close(fig)

def fig_vif(R):
    v=R["screen"]["vif_final"].sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.bar(range(len(v)),v.values,color="#4c78a8",edgecolor="k",alpha=.8)
    ax.axhline(10,color="r",ls="--",lw=1.3,label="threshold=10")
    ax.set_xticks(range(len(v))); ax.set_xticklabels(v.index,rotation=90,fontsize=7)
    ax.set_ylabel("VIF (retained)"); ax.legend()
    ax.set_title(f"{R['name']}: VIF after screening (dropped {len(R['screen']['dropped_vif'])} redundant; bars over line = protected)")
    fig.tight_layout(); _save(fig,R["name"],"vif")

def fig_spearman(R):
    sp=R["screen"]["spearman"].head(20)[::-1]
    fig,ax=plt.subplots(figsize=(9,7))
    ax.barh(range(len(sp)),sp.values,color="#59a14f",edgecolor="k",alpha=.8)
    ax.set_yticks(range(len(sp))); ax.set_yticklabels(sp.index,fontsize=8)
    ax.set_xlabel(f"|Spearman rho| with {R['name']}")
    ax.set_title(f"{R['name']}: monotonic relevance to target (top 20)")
    fig.tight_layout(); _save(fig,R["name"],"spearman")

def fig_shap_bar(R):
    if R["shap_mean"] is None: return
    s=R["shap_mean"].head(15)[::-1]
    fig,ax=plt.subplots(figsize=(9,7))
    ax.barh(range(len(s)),s.values,color="#9b8cc4",edgecolor="k",alpha=.85)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index,fontsize=8)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"{R['name']}: SHAP importance (top 15)")
    fig.tight_layout(); _save(fig,R["name"],"shap_bar")

def fig_perm(R):
    if R["perm"] is None: return
    s=R["perm"].head(15)[::-1]
    fig,ax=plt.subplots(figsize=(9,7))
    ax.barh(range(len(s)),s.values,color="#d18b47",edgecolor="k",alpha=.85)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index,fontsize=8)
    ax.set_xlabel("permutation importance (Δ R²)")
    ax.set_title(f"{R['name']}: permutation importance")
    fig.tight_layout(); _save(fig,R["name"],"perm")

def fig_dist(R):
    y=R["y"]; X=R["X"]; name=R["name"]; unit=R["unit"]
    fig=plt.figure(figsize=(15,4.5)); fig.suptitle(f"{name}: data distribution",fontweight="bold")
    ax=fig.add_subplot(1,3,1); ax.hist(y,bins=40,density=True,color="#69b3d6",edgecolor="k",alpha=.8)
    xs=np.linspace(y.min(),y.max(),200); ax.plot(xs,gaussian_kde(y)(xs),"r-",lw=2)
    ax.set_xlabel(f"{name}{unit}"); ax.set_ylabel("Density"); ax.set_title("Histogram+KDE")
    ax=fig.add_subplot(1,3,2); ax.boxplot(y,widths=.5,patch_artist=True,boxprops=dict(facecolor="#69b3d6"))
    ax.set_title("Boxplot"); ax.set_xticks([])
    ax=fig.add_subplot(1,3,3)
    top=(R["shap_mean"].head(12).index.tolist() if R["shap_mean"] is not None else list(X.columns[:12]))
    C=X[top].corr().values; im=ax.imshow(C,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top,rotation=90,fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top,fontsize=7)
    ax.set_title("Feature correlation (top-12)"); fig.colorbar(im,ax=ax,fraction=0.046)
    fig.tight_layout(rect=[0,0,1,0.94]); _save(fig,name,"dist")

def fig_train_test(R):
    name,unit=R["name"],R["unit"]; y=R["y"]; order=[k for k in R["fit"] if k!="ENSEMBLE"]+["ENSEMBLE"]
    n=len(order); cols=4; rows=int(np.ceil(n/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(4.4*cols,4.2*rows),squeeze=False)
    fig.suptitle(f"{name}: Train (DEV OOF) vs LOCKED-TEST{unit}",fontweight="bold")
    axl=axs.ravel()
    lo,hi=y.min(),y.max()
    for ax,nm in zip(axl,order):
        fr=R["fit"][nm]
        ax.scatter(y[R["dev"]],fr["oof"],marker="D",s=14,facecolor="none",edgecolor="#7e57c2",alpha=.5,label=f"DEV OOF R²={fr['R2_oof']:.3f}")
        ax.scatter(y[R["test"]],fr["p_test"],marker="^",s=20,color="#2ca02c",alpha=.75,label=f"LOCKED-TEST R²={fr['R2_test']:.3f}")
        ax.plot([lo,hi],[lo,hi],"--",color="k",lw=1.2)
        ax.set_title(nm); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}"); ax.legend(fontsize=7,loc="upper left")
    for ax in axl[n:]: ax.axis("off")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"train_test")

def fig_cvbox(R):
    name=R["name"]; order=sorted(R["cv"],key=lambda k:R["cv"][k].mean(),reverse=True)
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot([R["cv"][k] for k in order],patch_artist=True,showmeans=True,widths=.6,
                  meanprops=dict(marker="^",mfc="k",mec="k"),medianprops=dict(color="k"))
    for pt,k in zip(bp["boxes"],order): pt.set_facecolor(STYLE.get(k,("o","#888"))[1]); pt.set_alpha(.65)
    for i,k in enumerate(order): ax.annotate(f"{R['cv'][k].mean():.3f}",(i+1,R['cv'][k].mean()),
                                             textcoords="offset points",xytext=(8,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel(f"$R^2$ ({FOLDS}-fold grouped OOF on DEV)")
    ax.set_title(f"{name}: {FOLDS}-fold grouped-CV performance (DEV)")
    fig.tight_layout(); _save(fig,name,"cvbox")

def fig_relerr(R):
    name,y=R["name"],R["y"]; order=sorted(R["fit"],key=lambda k:R["fit"][k]["R2_test"],reverse=True)
    yte=y[R["test"]]; ys=np.where(np.abs(yte)<1e-6,1e-6,yte)
    data=[np.abs((R["fit"][k]["p_test"]-yte)/ys)*100 for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.6,medianprops=dict(color="k"))
    for pt,k in zip(bp["boxes"],order): pt.set_facecolor(STYLE.get(k,("o","#888"))[1]); pt.set_alpha(.65)
    ax.axhline(5,ls="--",color="grey"); ax.axhline(10,ls=":",color="grey")
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"{name}: relative error on LOCKED TEST"); fig.tight_layout(); _save(fig,name,"relerr")

def fig_residual(R):
    ens=R["fit"]["ENSEMBLE"]; yte=R["y"][R["test"]]; res=ens["p_test"]-yte
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    fig.suptitle(f"{R['name']}: residual diagnostics (locked test, ENSEMBLE)",fontweight="bold")
    axs[0].scatter(ens["p_test"],res,s=14,alpha=.55,edgecolor="k",lw=.2); axs[0].axhline(0,color="r",lw=1)
    axs[0].set_xlabel("Predicted"); axs[0].set_ylabel("Residual"); axs[0].set_title("Residual vs predicted")
    axs[1].hist(res,bins=30,color="#69b3d6",edgecolor="k",alpha=.8); axs[1].set_title("Residual histogram"); axs[1].set_xlabel("residual")
    from scipy import stats as st
    st.probplot(res,dist="norm",plot=axs[2]); axs[2].set_title("QQ plot")
    fig.tight_layout(rect=[0,0,1,0.94]); _save(fig,R["name"],"residual")

def fig_subgroup(R):
    y=R["y"][R["test"]]; ens=R["fit"]["ENSEMBLE"]["p_test"]
    bins=pd.qcut(pd.Series(y).rank(method="first"),4,labels=["Q1(low)","Q2","Q3","Q4(high)"],duplicates="drop")
    r2s=[r2_score(y[bins==b],ens[bins==b]) if (bins==b).sum()>1 else np.nan for b in bins.cat.categories]
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(range(len(r2s)),r2s,color="#4c78a8",edgecolor="k",alpha=.8)
    ax.set_xticks(range(len(r2s))); ax.set_xticklabels(bins.cat.categories)
    for i,v in enumerate(r2s): ax.annotate(f"{v:.2f}" if v==v else "-",(i,v if v==v else 0),textcoords="offset points",xytext=(0,4),ha="center")
    ax.set_ylabel("R² on locked test"); ax.set_title(f"{R['name']}: subgroup analysis by target quartile")
    fig.tight_layout(); _save(fig,R["name"],"subgroup")

def fig_shap_bee(R):
    if R["shap"] is None: return
    fig=plt.figure(figsize=(10,6)); plt.title(f"{R['name']}: SHAP beeswarm (DEV sample)")
    shap.summary_plot(R["shap"],R["X_shap"],plot_type="dot",max_display=15,show=False,plot_size=None)
    fig.tight_layout(); _save(fig,R["name"],"shap_bee")

def fig_pdp(R):
    top=(R["shap_mean"].head(6).index.tolist() if R["shap_mean"] is not None else list(R["X"].columns[:6]))
    fig,axs=plt.subplots(2,3,figsize=(16,8.5)); fig.suptitle(f"{R['name']}: Partial Dependence (top-6)",fontweight="bold")
    try:
        PartialDependenceDisplay.from_estimator(R["pdp_model"],R["pdp_bg"],features=top,ax=axs.ravel()[:len(top)],
            kind="average",grid_resolution=40,line_kw={"color":"#1f77b4","lw":2})
    except Exception as e:
        print("   [pdp]",e)
        for ax in axs.ravel(): ax.axis("off")
    for ax in axs.ravel()[len(top):]: ax.axis("off")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,R["name"],"pdp")

# ================================= MAIN =====================================
if __name__=="__main__":
    print(f"[v2 workflow] XGBoost={HAS_XGB} CatBoost={HAS_CAT} SHAP={HAS_SHAP}  folds={FOLDS} GS_CV={GS_CV}")
    Rs=[run_target("LWT"," (mm)"), run_target("SCB","")]
    for R in Rs:
        fig_dist(R); fig_vif(R); fig_spearman(R); fig_shap_bar(R); fig_perm(R)
        fig_train_test(R); fig_cvbox(R); fig_relerr(R); fig_residual(R); fig_subgroup(R)
        fig_shap_bee(R); fig_pdp(R)
    if not HEADLESS: plt.show()
    print("\n"+"="*80+"\n  SUMMARY — locked-test R² (unbiased, one-shot; 80/20 grouped by Mix_ID)\n"+"="*80)
    for R in Rs:
        best=max(R["fit"],key=lambda k:R["fit"][k]["R2_test"]); f=R["fit"][best]
        print(f"  {R['name']:5s} features={R['X'].shape[1]}  best={best:11s}  "
              f"LOCKED-TEST R2={f['R2_test']:.3f}  RMSE={f['RMSE_test']:.3f}  MAE={f['MAE_test']:.3f}  "
              f"(DEV OOF R2={f['R2_oof']:.3f})")
    # save best_params for reproducibility
    with open(f"{OUTDIR}/best_params.json","w") as fp:
        json.dump({R["name"]:R["best_params"] for R in Rs},fp,indent=2,default=str)
    print(f"\n{len(Rs)*12} figures + best_params.json saved in ./{OUTDIR}/")
