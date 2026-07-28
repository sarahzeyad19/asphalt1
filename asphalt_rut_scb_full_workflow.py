# -*- coding: utf-8 -*-
"""
ASPHALT RUT (LWT) + SCB — FULL WORKFLOW
  VIF/Spearman screening  ->  hyperparameter-TUNED models  ->  grouped 10-fold CV
================================================================================
Pipeline (in order):
  0. Load per-target CSV, keep ALL rows (replicates KEPT).
  1. Build features from the JMF columns present.
  2. FEATURE SCREENING (preliminary step, before any model):
        a. Spearman |rho| of each feature with the target  (relevance)
        b. VARIANCE INFLATION FACTOR (VIF) multicollinearity pruning:
           iteratively drop the numeric feature with the highest VIF until all
           VIF <= VIF_THRESH (default 10). One-hot dummies are kept as-is.
        -> the reduced, low-collinearity feature set is what the models see.
  3. HYPERPARAMETER TUNING: every model is tuned with RandomizedSearchCV using
     GroupKFold (grouped by Mix_ID) so tuning is leakage-safe.
  4. EVALUATION: StratifiedGroupKFold(10) grouped by Mix_ID (replicates never
     span folds). Reports TRAIN + OOF(test) R2 per model + weighted ensemble.
  5. FEATURE IMPORTANCE: SHAP (beeswarm, polar, bar) shows how important each
     feature is to rutting (LWT) and cracking (SCB).
  6. All diagnostic figures: distribution, actual-vs-pred, train/val/test,
     CV box, overlay, relative error, SHAP, PDP, ranking consistency, OOF,
     + NEW: VIF bar and Spearman-with-target bar.

HOW TO RUN IN SPYDER
  Put mixture_dataset_lwt.csv and mixture_dataset_scb.csv next to this file
  (or in Downloads), or edit FILE_LWT/FILE_SCB. Press F5.
  Runtime: tuning makes it ~5-10 min per target (TUNE_N_ITER controls it).

Requirements: pandas numpy scikit-learn lightgbm xgboost shap scipy matplotlib
              (catboost optional).
"""
import os, re, glob, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr, gaussian_kde, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
import lightgbm as lgb
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False
try: from catboost import CatBoostRegressor; HAS_CAT=True
except Exception: HAS_CAT=False
try: import shap; HAS_SHAP=True
except Exception: HAS_SHAP=False

# =============================== CONFIG =====================================
FILE_LWT   = "mixture_dataset_lwt.csv"
FILE_SCB   = "mixture_dataset_scb.csv"
FOLDS      = 10
RANDOM     = 42
LOG_LWT    = False      # model log1p(LWT)? marginal; keep False for real-unit SHAP/PDP
VIF_THRESH = 10.0       # drop numeric features until every VIF <= this
# Features VIF must NEVER drop, even if collinear — physically important to
# rutting/cracking and shown to enhance the model in earlier trials. VIF still
# prunes the OTHER redundant columns around them.
EXEMPT_VIF = [
    # binder / PG grade
    "PG_High_Temp_C","PG_Low_Temp_C","PG_span","Polymer",
    "PG_x_Voids","PG_x_RBR","PG_x_AFT","PG_x_Pbe",
    # RAP family (RAP %, AC-from-RAP, RAP binder ratio, film thickness)
    "RAP","Total_AC_from_RAP","RBR","AFT",
    # core volumetrics / binder content that drive performance
    "Voids","VFA","Pbe","AC","Gse","Dust_Pbeff",
    # aggregate cleanliness + key gradation-shape terms
    "Combined_SandEq","Pass_No_30","Grad_CA","Grad_CoarseSlope","Grad_FM",
    # binder-additive chemistry (biggest enhancer in earlier trials)
    "add_antistrip","add_wma","add_fiber","add_polymer","add_count","add_dose_max","add_dose_sum",
]
SPEARMAN_MIN = 0.0      # drop features whose |Spearman rho| with target < this (0 = keep all, report only)
TUNE       = True       # hyperparameter-tune every model (RandomizedSearchCV, grouped)
TUNE_N_ITER= 30         # search iterations per model
TUNE_CV    = 5          # inner GroupKFold folds for tuning
HEADLESS   = False
OUTDIR     = "asphalt_full_outputs"
os.makedirs(OUTDIR, exist_ok=True)
rcParams.update({"figure.dpi":110,"savefig.dpi":300,"font.size":12,"axes.titlesize":13,
                 "axes.labelsize":12,"legend.fontsize":9,"axes.grid":True,"grid.alpha":0.25,
                 "font.family":"DejaVu Sans"})
num=lambda s: pd.to_numeric(s,errors="coerce")

def resolve(name):
    if os.path.isfile(name): return name
    for root in [os.getcwd(),os.path.expanduser("~/Downloads"),os.path.expanduser("~")]:
        hit=glob.glob(os.path.join(root,"**",os.path.basename(name)),recursive=True)
        if hit: return hit[0]
    return name

# =========================== FEATURES =======================================
CAT=["Design_Level","Binder_Modification","Mix_Type","Spec_Edition"]
IDS=["Project_ID","Mix_ID","IsVerification","Date_Approved"]
SIEVE_MM={'Pass 1 1/2"':37.5,'Pass 1"':25,'Pass 3/4"':19,'Pass 1/2"':12.5,'Pass 3/8"':9.5,
          'Pass No.4':4.75,'Pass No.8':2.36,'Pass No.16':1.18,'Pass No.30':0.60,
          'Pass No.50':0.30,'Pass No.100':0.15,'Pass No.200':0.075}
DUMMY_PREFIX=("Design_Level_","Binder_Modification_","Mix_Type_","Spec_Edition_")

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
    devs=[P[s]-(100*(SIEVE_MM[s]/nmas)**0.45).clip(upper=100) for s in order]
    if devs:
        D=pd.concat(devs,axis=1); G["MDL_meanabs"]=D.abs().mean(axis=1); G["MDL_area"]=D.sum(axis=1)
    return G

def add_additives(df):
    """Binder-additive chemistry features (only if the columns are present):
    anti-strip, WMA, fiber, polymer flags + additive count & dosage. These were
    the single biggest enhancer in earlier trials (~+0.06 R2)."""
    A=pd.DataFrame(index=df.index)
    if "Binder_Additive_Name" not in df.columns: return A
    up=df["Binder_Additive_Name"].astype(str).str.upper()
    A["add_antistrip"]=up.str.contains("ANTI.?STRIP|AD.?HERE|PERMA.?TAC|LA.?2",regex=True).astype(float)
    A["add_wma"]      =up.str.contains("WMA|EVOTHERM|ZYCO|THERMA|WARM",regex=True).astype(float)
    A["add_fiber"]    =up.str.contains("FIBER|CELLULOSE",regex=True).astype(float)
    A["add_polymer"]  =(up.str.contains("LATEX|SBS|POLYMER|RUBBER",regex=True)
                        | df.get("Binder_Modification",pd.Series("",index=df.index)).astype(str).str.upper().str.contains("MODIF")
                        | df.get("PG_Grade",pd.Series("",index=df.index)).astype(str).str.contains(r"\d2m|\d2M|rm|RM",regex=True)).astype(float)
    A["add_count"]=df["Binder_Additive_Name"].apply(lambda s:max(0,str(s).count(",")))
    if "Binder_Additive_PercentMix" in df.columns:
        def doses(s):
            vals=[pd.to_numeric(p.strip(),errors="coerce") for p in str(s).split(",")[1:]]
            vals=[float(v) for v in vals if not pd.isna(v)]
            return (max(vals) if vals else 0.0, sum(vals) if vals else 0.0)
        d=df["Binder_Additive_PercentMix"].apply(doses)
        A["add_dose_max"]=d.apply(lambda t:t[0]); A["add_dose_sum"]=d.apply(lambda t:t[1])
    return A

def build_features(df, target):
    X=pd.DataFrame(index=df.index); skip=set(IDS+CAT+[target]+list(SIEVE_MM))
    for c in df.columns:
        if c in skip: continue
        v=num(df[c])
        if v.notna().sum()>0: X[c]=v
    for s in SIEVE_MM:
        if s in df.columns: X[s]=num(df[s])
    X=pd.concat([X,gradation(df)],axis=1)
    if "PG_High_Temp_C" in df:
        ph=num(df["PG_High_Temp_C"])
        if "PG_Low_Temp_C" in df: X["PG_span"]=ph-num(df["PG_Low_Temp_C"])
        if "%Voids" in df: X["PG_x_Voids"]=ph*num(df["%Voids"])
        if "RBR" in df: X["PG_x_RBR"]=ph*num(df["RBR"])
        if "AFT" in df: X["PG_x_AFT"]=ph*num(df["AFT"])
        if "Pbe" in df: X["PG_x_Pbe"]=ph*num(df["Pbe"])
    if "Binder_Modification" in df:
        X["Polymer"]=df["Binder_Modification"].astype(str).str.upper().str.contains("MODIF").astype(float)
    add=add_additives(df)                      # binder-additive chemistry (if columns present)
    if add.shape[1]: X=pd.concat([X,add.set_index(X.index)],axis=1)
    for c in CAT:
        if c in df:
            X=pd.concat([X,pd.get_dummies(df[c].astype(str),prefix=c).astype(float).set_index(X.index)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]
    X.columns=[re.sub(r"[^0-9A-Za-z_]+","_",str(c)).strip("_") for c in X.columns]
    X=X.loc[:,~X.columns.duplicated()]
    return X

# ================= FEATURE SCREENING: Spearman + VIF ========================
def compute_vif(Xnum):
    """VIF via the diagonal of the inverse correlation matrix (pinv-robust)."""
    C=np.corrcoef(Xnum.values,rowvar=False)
    C=np.nan_to_num(C,nan=0.0); np.fill_diagonal(C,1.0)
    vif=np.diag(np.linalg.pinv(C))
    return pd.Series(vif,index=Xnum.columns)

def screen_features(X,y):
    """(1) Spearman relevance filter, (2) VIF multicollinearity pruning.
    One-hot dummies are always kept; VIF prunes only continuous features."""
    dummies=[c for c in X.columns if c.startswith(DUMMY_PREFIX)]
    # (1) Spearman |rho| with target
    sp=X.apply(lambda col: abs(spearmanr(col,y).correlation) if np.std(col)>0 else 0.0)
    sp=sp.fillna(0.0).sort_values(ascending=False)
    weak=[c for c in X.columns if c not in dummies and c not in EXEMPT_VIF and sp.get(c,0)<SPEARMAN_MIN]
    Xs=X.drop(columns=weak)
    # (2) VIF prune numeric — but NEVER drop exempt (physically-important) or dummy cols.
    exempt=set(EXEMPT_VIF)
    cur=[c for c in Xs.columns if c not in dummies]; dropped=[]
    while True:
        vif=compute_vif(Xs[cur])
        droppable=vif[[c for c in cur if c not in exempt]]
        if len(droppable)==0 or float(droppable.max())<=VIF_THRESH: break
        worst=droppable.idxmax(); cur.remove(worst); dropped.append((worst,round(float(droppable.max()),1)))
    kept=[c for c in X.columns if c in cur or c in dummies]
    vif_final=compute_vif(Xs[cur]) if len(cur)>=2 else pd.Series(dtype=float)
    kept_exempt=[c for c in cur if c in exempt]
    print(f"  screening: Spearman dropped {len(weak)} (|rho|<{SPEARMAN_MIN}); "
          f"VIF dropped {len(dropped)} (>{VIF_THRESH}, non-exempt only); kept {len(kept)} features "
          f"({len(kept_exempt)} protected/exempt)")
    if dropped: print("    VIF-dropped:", ", ".join(f"{c}({v})" for c,v in dropped[:12])+(" ..." if len(dropped)>12 else ""))
    hi_exempt=[(c,round(float(vif_final[c]),1)) for c in kept_exempt if c in vif_final.index and vif_final[c]>VIF_THRESH]
    if hi_exempt: print("    kept despite high VIF (protected):", ", ".join(f"{c}({v})" for c,v in hi_exempt))
    return kept, dict(spearman=sp, vif_final=vif_final, vif_dropped=dropped, spearman_dropped=weak)

# ============================ MODELS ========================================
def base_models():
    m={"ExtraTrees":ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,max_features=0.6,n_jobs=-1,random_state=RANDOM),
       "RandomForest":RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.7,n_jobs=-1,random_state=RANDOM),
       "LightGBM":lgb.LGBMRegressor(n_estimators=700,learning_rate=0.03,num_leaves=63,min_child_samples=15,
          subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,random_state=RANDOM,n_jobs=-1,verbose=-1),
       "HistGB":HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=15,l2_regularization=0.1,random_state=RANDOM)}
    if HAS_XGB: m["XGBoost"]=xgb.XGBRegressor(n_estimators=700,learning_rate=0.03,max_depth=6,subsample=0.85,
          colsample_bytree=0.7,reg_lambda=1.5,min_child_weight=3,random_state=RANDOM,n_jobs=-1,verbosity=0)
    if HAS_CAT: m["CatBoost"]=CatBoostRegressor(iterations=600,learning_rate=0.03,depth=6,l2_leaf_reg=3.0,random_state=RANDOM,verbose=0)
    return m

def search_spaces():
    sp={"ExtraTrees":(ExtraTreesRegressor(n_jobs=1,random_state=RANDOM),
          dict(n_estimators=randint(300,700),min_samples_leaf=randint(1,20),
               max_features=uniform(0.3,0.6),max_depth=randint(6,26))),
        "RandomForest":(RandomForestRegressor(n_jobs=1,random_state=RANDOM),
          dict(n_estimators=randint(300,700),min_samples_leaf=randint(1,15),
               max_features=uniform(0.3,0.6),max_depth=randint(6,26))),
        "LightGBM":(lgb.LGBMRegressor(random_state=RANDOM,n_jobs=1,verbose=-1),
          dict(n_estimators=randint(300,1200),learning_rate=uniform(0.01,0.07),num_leaves=randint(15,150),
               min_child_samples=randint(5,40),subsample=uniform(0.6,0.4),colsample_bytree=uniform(0.5,0.5),
               reg_alpha=uniform(0,3),reg_lambda=uniform(0,4),max_depth=randint(3,12))),
        "HistGB":(HistGradientBoostingRegressor(random_state=RANDOM),
          dict(max_iter=randint(300,900),learning_rate=uniform(0.01,0.07),max_leaf_nodes=randint(15,80),
               min_samples_leaf=randint(10,40),l2_regularization=uniform(0,3),max_depth=randint(3,12)))}
    if HAS_XGB: sp["XGBoost"]=(xgb.XGBRegressor(random_state=RANDOM,n_jobs=1,verbosity=0),
          dict(n_estimators=randint(300,1200),learning_rate=uniform(0.01,0.07),max_depth=randint(3,11),
               subsample=uniform(0.6,0.4),colsample_bytree=uniform(0.5,0.5),reg_alpha=uniform(0,3),
               reg_lambda=uniform(0,4),min_child_weight=randint(1,8)))
    if HAS_CAT: sp["CatBoost"]=(CatBoostRegressor(random_state=RANDOM,verbose=0),
          dict(iterations=randint(300,900),learning_rate=uniform(0.01,0.07),depth=randint(4,9),l2_leaf_reg=uniform(1,6)))
    return sp

def tune_models(X,y,grp,yfit):
    gkf=GroupKFold(TUNE_CV); tuned={}; params={}
    print(f"  tuning {len(search_spaces())} models (RandomizedSearch n_iter={TUNE_N_ITER}, GroupKFold {TUNE_CV})...")
    for nm,(est,dist) in search_spaces().items():
        rs=RandomizedSearchCV(est,dist,n_iter=TUNE_N_ITER,cv=gkf,scoring="r2",
                              random_state=RANDOM,n_jobs=-1,refit=True)
        rs.fit(X,yfit,groups=grp); tuned[nm]=rs.best_estimator_; params[nm]=rs.best_params_
        print(f"    tuned {nm:12s} inner-CV r2={rs.best_score_:.3f}")
    return tuned,params

STYLE={"ExtraTrees":("s","#1f77b4"),"RandomForest":("^","#2ca02c"),"LightGBM":("*","#17becf"),
       "XGBoost":("o","#9467bd"),"CatBoost":("h","#bcbd22"),"HistGB":("<","#ff7f0e"),"ENSEMBLE":("D","#d62728")}
SUBSET={"Training":("D","#7e57c2"),"Validation":("s","#ff9800"),"Test":("^","#2ca02c")}
def met(y,p): return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p))
def opt_ens(oof,y):
    ks=list(oof); best=(-9,None); grid=np.arange(0,1.01,0.25)
    for w in itertools.product(grid,repeat=len(ks)):
        if abs(sum(w)-1)>1e-6: continue
        s=r2_score(y,sum(w[i]*oof[ks[i]] for i in range(len(ks))))
        if s>best[0]: best=(s,dict(zip(ks,w)))
    return best[1]
def grouped_three_way(X,y,grp,seed=RANDOM):
    ng=len(np.unique(grp)); k1=max(3,min(7,ng//2)); k2=max(3,min(6,ng//3))
    nb=max(2,min(10,ng//max(1,k1)))
    yb=pd.qcut(pd.Series(y).rank(method="first"),nb,labels=False,duplicates="drop")
    dev,te=next(iter(StratifiedGroupKFold(k1,shuffle=True,random_state=seed).split(X,yb,groups=grp)))
    d_tr,d_va=next(iter(StratifiedGroupKFold(k2,shuffle=True,random_state=seed).split(X.iloc[dev],yb[dev],groups=grp[dev])))
    return np.sort(dev[d_tr]),np.sort(dev[d_va]),np.sort(te)

# ============================ COMPUTE =======================================
def compute(path,name,target,unit):
    print("\n"+"="*74+f"\n  {name}  — VIF screen + tuned models + grouped {FOLDS}-fold CV\n"+"="*74)
    df=pd.read_csv(resolve(path)); y=num(df[target]).values; keep=np.isfinite(y)
    df=df[keep].reset_index(drop=True); y=y[keep]; grp=df["Mix_ID"].astype(str).values
    Xfull=build_features(df,target)
    kept,screen=screen_features(Xfull,y); X=Xfull[kept]
    log=(name=="LWT" and LOG_LWT); inv=(np.expm1 if log else (lambda p:p)); yfit=np.log1p(y) if log else y
    print(f"  rows={len(X)}  unique mixes={len(np.unique(grp))}  features(after screen)={X.shape[1]}  "
          f"target=[{y.min():.3f},{y.max():.3f}]")
    models,params = (tune_models(X,y,grp,yfit) if TUNE else (base_models(),{}))
    yb=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    folds=list(StratifiedGroupKFold(FOLDS,shuffle=True,random_state=RANDOM).split(X,yb,groups=grp))
    tr3,va3,te3=grouped_three_way(X,y,grp)
    fit={}; cv={}; oof_all={}; tvt={}
    for nm in models:
        oof=np.zeros(len(y)); fr=[]; trs=[]
        for a,b in folds:
            mm=clone(models[nm]); mm.fit(X.iloc[a],yfit[a])
            oof[b]=inv(mm.predict(X.iloc[b])); trs.append(r2_score(y[a],inv(mm.predict(X.iloc[a])))); fr.append(r2_score(y[b],oof[b]))
        om=met(y,oof); oof_all[nm]=oof; cv[nm]=np.array(fr)
        m3=clone(models[nm]); m3.fit(X.iloc[tr3],yfit[tr3])
        ptr,pva,pte=inv(m3.predict(X.iloc[tr3])),inv(m3.predict(X.iloc[va3])),inv(m3.predict(X.iloc[te3]))
        fit[nm]=dict(oof=oof,R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],R2_train=np.mean(trs),
                     fold_r2=np.array(fr),ptr=ptr,pva=pva,pte=pte,
                     R2_te=r2_score(y[te3],pte),R2_va=r2_score(y[va3],pva),R2_tr3=r2_score(y[tr3],ptr))
        tvt[nm]=dict(tr=(y[tr3],ptr,fit[nm]["R2_tr3"]),va=(y[va3],pva,fit[nm]["R2_va"]),te=(y[te3],pte,fit[nm]["R2_te"]))
        print(f"    {nm:12s} train R2={np.mean(trs):.3f}  OOF(test) R2={om['R2']:.3f}  RMSE={om['RMSE']:.3f}  fold={np.mean(fr):.3f}±{np.std(fr):.3f}")
    W=opt_ens(oof_all,y); ens=sum(W[k]*oof_all[k] for k in W); om=met(y,ens)
    cv["ENSEMBLE"]=np.array([r2_score(y[b],ens[b]) for _,b in folds])
    e_tr=sum(W[k]*fit[k]["ptr"] for k in W); e_va=sum(W[k]*fit[k]["pva"] for k in W); e_te=sum(W[k]*fit[k]["pte"] for k in W)
    fit["ENSEMBLE"]=dict(oof=ens,R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],R2_train=np.nan,
        fold_r2=cv["ENSEMBLE"],ptr=e_tr,pva=e_va,pte=e_te,R2_te=r2_score(y[te3],e_te),
        R2_va=r2_score(y[va3],e_va),R2_tr3=r2_score(y[tr3],e_tr))
    tvt["ENSEMBLE"]=dict(tr=(y[tr3],e_tr,fit["ENSEMBLE"]["R2_tr3"]),va=(y[va3],e_va,fit["ENSEMBLE"]["R2_va"]),te=(y[te3],e_te,fit["ENSEMBLE"]["R2_te"]))
    print(f"    ENSEMBLE     OOF(test) R2={om['R2']:.3f}  RMSE={om['RMSE']:.3f}  weights={ {k:round(W[k],2) for k in W if W[k]>0} }")
    shap_mean=X_shap=sv=None
    pdp_model=clone(models["LightGBM"]); pdp_model.fit(X.iloc[tr3],y[tr3])
    if HAS_SHAP:
        X_shap=X.iloc[te3] if len(te3)<=800 else X.iloc[te3].sample(800,random_state=RANDOM)
        try:
            sv=shap.TreeExplainer(pdp_model).shap_values(X_shap)
            shap_mean=pd.Series(np.abs(sv).mean(0),index=X_shap.columns).sort_values(ascending=False)
        except Exception as ex: print("   [shap skipped]",ex)
    return dict(name=name,unit=unit,X=X,y=y,grp=grp,folds=folds,fit=fit,cv=cv,weights=W,params=params,
                tvt=tvt,tr3=tr3,va3=va3,te3=te3,shap=sv,X_shap=X_shap,shap_mean=shap_mean,
                pdp_model=pdp_model,pdp_bg=X.iloc[tr3],screen=screen)

# ============================ FIGURES =======================================
def _save(fig,name,tag):
    fig.savefig(f"{OUTDIR}/{name}_{tag}.png",bbox_inches="tight")
    if HEADLESS: plt.close(fig)

def fig_vif(R):
    name=R["name"]; vif=R["screen"]["vif_final"].sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.bar(range(len(vif)),vif.values,color="#4c78a8",edgecolor="k",alpha=.8)
    ax.axhline(VIF_THRESH,color="r",ls="--",lw=1.3,label=f"threshold={VIF_THRESH}")
    ax.set_xticks(range(len(vif))); ax.set_xticklabels(vif.index,rotation=90,fontsize=7)
    ax.set_ylabel("VIF (retained features)")
    drop=R["screen"]["vif_dropped"]
    ax.set_title(f"{name}: VIF after pruning (dropped {len(drop)} redundant; bars over line = protected/exempt features kept on purpose)")
    ax.legend(); fig.tight_layout(); _save(fig,name,"vif")

def fig_spearman_target(R):
    name,unit=R["name"],R["unit"]; sp=R["screen"]["spearman"].head(20)[::-1]
    fig,ax=plt.subplots(figsize=(9,7))
    ax.barh(range(len(sp)),sp.values,color="#59a14f",edgecolor="k",alpha=.8)
    ax.set_yticks(range(len(sp))); ax.set_yticklabels(sp.index,fontsize=8)
    ax.set_xlabel(f"|Spearman rho| with {name}"); ax.set_title(f"{name}: monotonic relevance of features to target (top 20)")
    fig.tight_layout(); _save(fig,name,"spearman_target")

def fig_importance(R):
    if R["shap_mean"] is None: return
    name=R["name"]; s=R["shap_mean"].head(15)[::-1]
    fig,ax=plt.subplots(figsize=(9,7))
    ax.barh(range(len(s)),s.values,color="#9b8cc4",edgecolor="k",alpha=.85)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index,fontsize=8)
    ax.set_xlabel("mean |SHAP| (importance)")
    ax.set_title(f"{name}: feature importance to {'rutting' if name=='LWT' else 'cracking'} (SHAP)")
    fig.tight_layout(); _save(fig,name,"importance")

def fig_distribution(R):
    name,unit,y,X=R["name"],R["unit"],R["y"],R["X"]
    fig=plt.figure(figsize=(15,4.5)); fig.suptitle(f"{name}: data distribution",fontweight="bold")
    ax=fig.add_subplot(1,3,1); ax.hist(y,bins=40,density=True,color="#69b3d6",edgecolor="k",alpha=.8)
    xs=np.linspace(y.min(),y.max(),200); ax.plot(xs,gaussian_kde(y)(xs),"r-",lw=2)
    ax.set_xlabel(f"{name}{unit}"); ax.set_ylabel("Density"); ax.set_title("Histogram + KDE")
    ax=fig.add_subplot(1,3,2); ax.boxplot(y,widths=.5,patch_artist=True,boxprops=dict(facecolor="#69b3d6"))
    ax.set_ylabel(f"{name}{unit}"); ax.set_title("Boxplot"); ax.set_xticks([])
    ax=fig.add_subplot(1,3,3)
    top=(R["shap_mean"].head(12).index.tolist() if R["shap_mean"] is not None else list(X.columns[:12]))
    C=X[top].corr().values; im=ax.imshow(C,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top,rotation=90,fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top,fontsize=7)
    ax.set_title("Feature correlation (top-12)"); fig.colorbar(im,ax=ax,fraction=0.046)
    fig.tight_layout(rect=[0,0,1,0.94]); _save(fig,name,"dist")

def fig_actual_pred(R):
    name,unit=R["name"],R["unit"]; order=[k for k in R["fit"] if k!="ENSEMBLE"]+["ENSEMBLE"]
    ytr,yte=R["y"][R["tr3"]],R["y"][R["te3"]]; lo,hi=R["y"].min(),R["y"].max()
    rows=int(np.ceil(len(order)/4)); fig,axs=plt.subplots(rows,4,figsize=(18,4.2*rows),squeeze=False)
    fig.suptitle(f"{name}: Actual vs Predicted — Train vs Test{unit}",fontweight="bold")
    axl=axs.ravel()
    for ax,nm in zip(axl,order):
        fr=R["fit"][nm]
        ax.scatter(ytr,fr["ptr"],marker="D",s=15,facecolor="none",edgecolor="#7e57c2",alpha=.5,label=f"Train R²={fr['R2_tr3']:.2f}")
        ax.scatter(yte,fr["pte"],marker="^",s=20,color="#2ca02c",alpha=.7,label=f"Test R²={fr['R2_te']:.2f}")
        ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.2); ax.set_title(nm); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}"); ax.legend(fontsize=7,loc="upper left")
    for ax in axl[len(order):]: ax.axis("off")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"actual_pred")

def fig_train_val_test(R):
    name,unit=R["name"],R["unit"]; order=[k for k in R["fit"] if k!="ENSEMBLE"]+["ENSEMBLE"]
    allv=np.concatenate([R["y"][R["tr3"]],R["y"][R["va3"]],R["y"][R["te3"]]]); lo,hi=allv.min(),allv.max()
    rows=int(np.ceil(len(order)/4)); fig,axs=plt.subplots(rows,4,figsize=(18,4.2*rows),squeeze=False)
    fig.suptitle(f"{name}: Train / Validation / Test (grouped, no leakage){unit}",fontweight="bold")
    axl=axs.ravel()
    for ax,nm in zip(axl,order):
        d=R["tvt"][nm]
        for key,(ya,pa,r2) in [("Training",d["tr"]),("Validation",d["va"]),("Test",d["te"])]:
            mk,col=SUBSET[key]; ax.scatter(ya,pa,marker=mk,s=18,color=col,alpha=.6,edgecolor="k",lw=.2,label=f"{key} $R^2$={r2:.3f}")
        ax.plot([lo,hi],[lo,hi],"--",color="k",lw=1.2); ax.set_title(nm); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}"); ax.legend(fontsize=6.5,loc="upper left")
    for ax in axl[len(order):]: ax.axis("off")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"train_val_test")

def fig_cv_box(R):
    name=R["name"]; order=sorted(R["cv"],key=lambda k:R["cv"][k].mean(),reverse=True)
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot([R["cv"][k] for k in order],patch_artist=True,showmeans=True,widths=.6,
                  meanprops=dict(marker="^",mfc="k",mec="k"),medianprops=dict(color="k"))
    for pt,k in zip(bp["boxes"],order): pt.set_facecolor(STYLE.get(k,("o","#888"))[1]); pt.set_alpha(.65)
    for i,k in enumerate(order): ax.annotate(f"{R['cv'][k].mean():.3f}",(i+1,R['cv'][k].mean()),textcoords="offset points",xytext=(8,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel(f"$R^2$ ({FOLDS}-fold grouped OOF)")
    ax.set_title(f"{name}: {FOLDS}-fold grouped CV performance"); fig.tight_layout(); _save(fig,name,"cvbox")

def fig_overlay(R):
    name,unit,y=R["name"],R["unit"],R["y"]; fig,ax=plt.subplots(figsize=(7.5,7.5)); lo,hi=y.min(),y.max()
    for nm,fr in R["fit"].items():
        mk,col=STYLE.get(nm,("o","#888")); ax.scatter(y,fr["oof"],marker=mk,s=14,color=col,alpha=.45,label=nm,edgecolor="k",lw=.2)
    ax.plot([lo,hi],[lo,hi],"r-",lw=1.6,label="y=x")
    ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted (OOF){unit}"); ax.set_title(f"{name}: all models (grouped OOF)")
    ax.legend(fontsize=8,ncol=2); fig.tight_layout(); _save(fig,name,"overlay")

def fig_relerr(R):
    name,y=R["name"],R["y"]; order=sorted(R["fit"],key=lambda k:R["fit"][k]["R2_oof"],reverse=True)
    ys=np.where(np.abs(y)<1e-6,1e-6,y); data=[np.abs((R["fit"][k]["oof"]-y)/ys)*100 for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.6,medianprops=dict(color="k"))
    for pt,k in zip(bp["boxes"],order): pt.set_facecolor(STYLE.get(k,("o","#888"))[1]); pt.set_alpha(.65)
    ax.axhline(5,ls="--",color="grey",lw=1); ax.axhline(10,ls=":",color="grey",lw=1)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"{name}: relative error (grouped OOF)"); fig.tight_layout(); _save(fig,name,"relerr")

def fig_shap(R):
    if R["shap"] is None: return
    name=R["name"]; fig=plt.figure(figsize=(15,7)); fig.suptitle(f"{name}: SHAP summary & importance (LightGBM)",fontweight="bold")
    ax1=fig.add_subplot(1,2,1); plt.sca(ax1)
    shap.summary_plot(R["shap"],R["X_shap"],plot_type="dot",max_display=15,show=False,plot_size=None); ax1.set_title("SHAP beeswarm")
    ax2=fig.add_subplot(1,2,2,polar=True); top=R["shap_mean"].head(14)[::-1]; N=len(top)
    ang=np.linspace(0,2*np.pi,N,endpoint=False)
    ax2.bar(ang,top.values,width=2*np.pi/N*0.9,color="#9b8cc4",edgecolor="k",alpha=.8)
    ax2.set_xticks(ang); ax2.set_xticklabels(top.index,fontsize=7); ax2.set_title("Mean |SHAP| (polar)")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"shap")

def fig_pdp(R):
    name=R["name"]; top=(R["shap_mean"].head(6).index.tolist() if R["shap_mean"] is not None else list(R["X"].columns[:6]))
    fig,axs=plt.subplots(2,3,figsize=(16,8.5)); fig.suptitle(f"{name}: Partial Dependence (LightGBM, top-6)",fontweight="bold")
    try:
        PartialDependenceDisplay.from_estimator(R["pdp_model"],R["pdp_bg"],features=top,ax=axs.ravel()[:len(top)],
            kind="average",grid_resolution=40,line_kw={"color":"#1f77b4","lw":2})
    except Exception:
        for ax,f in zip(axs.ravel(),top):
            xs=np.linspace(R["pdp_bg"][f].quantile(.02),R["pdp_bg"][f].quantile(.98),40)
            base=R["pdp_bg"].median(); pr=[R["pdp_model"].predict(pd.DataFrame([{**base,f:v}]))[0] for v in xs]
            ax.plot(xs,pr,color="#1f77b4",lw=2); ax.set_xlabel(f); ax.set_ylabel("partial dependence")
    for ax in axs.ravel()[len(top):]: ax.axis("off")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"pdp")

def fig_spearman(R):
    name=R["name"]; models=list(R["fit"])
    xv=np.array([R["cv"][m].mean() for m in models]); yv=np.array([R["fit"][m]["R2_te"] for m in models])
    rho=spearmanr(xv,yv).correlation; fig,ax=plt.subplots(figsize=(7,6.5))
    for m in models:
        mk,col=STYLE.get(m,("o","#888")); ax.scatter(R["cv"][m].mean(),R["fit"][m]["R2_te"],marker=mk,s=90,color=col,edgecolor="k",label=m)
    lo=min(xv.min(),yv.min())-.05; hi=max(xv.max(),yv.max())+.05
    ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.2); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel(f"{FOLDS}-fold CV mean $R^2$"); ax.set_ylabel("Held-out (3-way test) $R^2$")
    ax.set_title(f"{name}: ranking consistency  Spearman $\\rho$={rho:.3f}"); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); _save(fig,name,"spearman")

def fig_oof(R):
    name,unit,y=R["name"],R["unit"],R["y"]; ens=R["fit"]["ENSEMBLE"]
    fig,ax=plt.subplots(figsize=(6.8,6.8)); lo,hi=y.min(),y.max()
    ax.scatter(y,ens["oof"],s=14,c="#2E6DA4",alpha=.5,edgecolor="k",lw=.2); ax.plot([lo,hi],[lo,hi],"r--",lw=1.4,label="y=x")
    a,b=np.polyfit(y,ens["oof"],1); xs=np.array([lo,hi]); ax.plot(xs,a*xs+b,"k-",lw=1.2,label=f"fit R²={ens['R2_oof']:.3f}\nRMSE={ens['RMSE_oof']:.3f}")
    ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted (OOF){unit}"); ax.set_title(f"{name}: grouped {FOLDS}-fold OOF (ENSEMBLE)")
    ax.legend(); fig.tight_layout(); _save(fig,name,"oof")

# ============================ MAIN ==========================================
if __name__=="__main__":
    print(f"[full] XGBoost={HAS_XGB} CatBoost={HAS_CAT} SHAP={HAS_SHAP}  TUNE={TUNE}  VIF_THRESH={VIF_THRESH}")
    Rs=[compute(FILE_LWT,"LWT","LWT"," (mm)"), compute(FILE_SCB,"SCB","SCB","")]
    for R in Rs:
        fig_vif(R); fig_spearman_target(R); fig_importance(R)
        fig_distribution(R); fig_actual_pred(R); fig_train_val_test(R); fig_cv_box(R)
        fig_overlay(R); fig_relerr(R); fig_shap(R); fig_pdp(R); fig_spearman(R); fig_oof(R)
    if not HEADLESS: plt.show()
    print("\n"+"="*74+"\n  SUMMARY (VIF-screened, tuned, grouped 10-fold OOF, replicates kept)\n"+"="*74)
    for R in Rs:
        e=R["fit"]["ENSEMBLE"]; tr=max(R['fit'][k]['R2_train'] for k in R['fit'] if k!='ENSEMBLE')
        print(f"  {R['name']:5s} features={R['X'].shape[1]}  ENSEMBLE train R2={tr:.3f}  test(OOF) R2={e['R2_oof']:.3f}  "
              f"RMSE={e['RMSE_oof']:.3f}  fold={e['fold_r2'].mean():.3f}±{e['fold_r2'].std():.3f}")
    print(f"\nAll figures (13 per target incl. VIF, Spearman, importance) in ./{OUTDIR}/")
