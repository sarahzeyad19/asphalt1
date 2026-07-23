# -*- coding: utf-8 -*-
"""
RUTTING & SCB — HIERARCHICAL JMF-FAMILY-AWARE GROUPED VALIDATION
================================================================
Implements the mixture-family (LOCO-style) validation plan:

  * DESIGN mixes only (Rutting_Design / SCB_Design). All report rows are KEPT.
  * Build a mixture-family tree without ever using the target:
        Exact_JMF_ID      = JMF_Record_Key
        Base_Mix_ID       = Project_ID | Plant_Code | Mix_ID-without-version
        Mixture_Family_ID = engineering-similarity family:
            Stage 1 (hard rules): same MixType + equivalent NMAS class  (+ binder/RAP class if present)
            Stage 2 (similarity): Ward agglomerative clustering on STANDARDIZED design features
                                  (AC, gradation, VMA, VFA, Gmm, Gmb, Va, CAA, FAA, Pbe, ...),
                                  cut at FAMILY_DISTANCE.  Target (SCB / Rut_20k) is NEVER used.
  * Locked test = GroupShuffleSplit(test_size=0.20) at the Mixture_Family_ID level
    (≈20% of FAMILIES, not rows). 5-fold GroupKFold inside the 80% development set.
  * Assertions: family overlap = 0 AND exact-JMF overlap = 0 across train/test.
  * Three reported metrics:
        (1) Unweighted report-level R2       (each row weight 1)
        (2) Family-balanced report-level R2  (w_i = 1 / reports-in-family)
        (3) Family-averaged R2               (one averaged prediction per family)
  * Models: SVR-RBF + CatBoost / LightGBM / XGBoost / HistGB / Huber; best by grouped CV.
  * Plots: best-fit, residuals, importance, SHAP, and family-size distribution.
Identifiers are never model predictors.
"""
from __future__ import annotations
import warnings, re
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
try: matplotlib.use("TkAgg")
except Exception: pass
import matplotlib.pyplot as plt
try: import shap; HAS_SHAP=True
except Exception: HAS_SHAP=False

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVR
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try: from lightgbm import LGBMRegressor; HAS_LGBM=True
except Exception: HAS_LGBM=False
try: from xgboost import XGBRegressor; HAS_XGB=True
except Exception: HAS_XGB=False
try: from catboost import CatBoostRegressor; HAS_CAT=True
except Exception: HAS_CAT=False

RANDOM_STATE=42
TEST_SIZE=0.20
CV_FOLDS=5
FAMILY_DISTANCE=3.0     # Ward cut in standardized-feature space (larger -> fewer, broader families)
SHOW_PLOTS=True
HOME=Path.home()
DOWNLOADS=Path(r"C:\Users\lenovo\Downloads")
if not DOWNLOADS.exists():
    DOWNLOADS=HOME/"Downloads" if (HOME/"Downloads").exists() else Path.cwd()
DATA="updated data extraction.xlsx"
OUT=DOWNLOADS/"Rut_SCB_FamilyAware_outputs"/"figures"; OUT.mkdir(parents=True,exist_ok=True)

CATEG=["MixType","DesignLev"]
RENAME={"Design_Submission__Percent_Voids":"Va","Design_Submission__VMA":"VMA","Design_Submission__VFA":"VFA",
"Design_Submission__Gmm":"Gmm","Design_Submission__Gmb_Nd":"Gmb","Design_Submission__Percent_AC":"AsphaltContent_Design",
"Design_Submission__Gse":"Gse","Design_Submission__Pba":"Pba_pct","Design_Submission__Pbe":"Pbe_pct",
"Design_Submission__Dust_Pbeff":"Dust_Binder","Design_Submission__Percent_Gmm_Ni":"Gmm_Nini","Design_Submission__Percent_Gmm_Nm":"Gmm_Nmax",
"Combined_Aggregate_Bulk_Gravity":"Gsb","Combined_Aggregate_Absorption":"Absorption","Combined_Aggregate_FAA":"FAA",
"Combined_Aggregate_CAA":"CAA","Combined_Aggregate_Sand_Equivalent":"SandEq","Mix_Type":"MixType","Design_Level":"DesignLev",
"Design_Submission__Pass_No_4":"Grad_No4","Design_Submission__Pass_No_8":"Grad_No8","Design_Submission__Pass_No_16":"Grad_No16",
"Design_Submission__Pass_No_30":"Grad_No30","Design_Submission__Pass_No_50":"Grad_No50","Design_Submission__Pass_No_100":"Grad_No100",
"Design_Submission__Pass_No_200":"Grad_No200"}
# design features used for BOTH modelling and family similarity (never the target)
FAMILY_FEATS=["AsphaltContent_Design","Va","VMA","VFA","Gmm","Gmb","Gse","CAA","FAA","SandEq","Absorption","Pbe_pct",
              "Grad_No4","Grad_No8","Grad_No30","Grad_No50","Grad_No100","Grad_No200"]
RUT_FEATURES=FAMILY_FEATS+["Gsb","Dust_Binder","Gmm_Nini","Gmm_Nmax","NMAS_mm","MixType","DesignLev"]
SCB_FEATURES=FAMILY_FEATS+["Dust_Binder","NMAS_mm","MixType","DesignLev"]

def M(a,b): return dict(R2=r2_score(a,b),RMSE=float(np.sqrt(mean_squared_error(a,b))),MAE=mean_absolute_error(a,b))
def nz(d,c): return pd.to_numeric(d[c],errors="coerce") if c in d.columns else pd.Series(np.nan,index=d.index)
def parse_nmas(v):
    if pd.isna(v): return np.nan
    m=re.search(r"(\d+(?:\.\d+)?)",str(v))
    if not m: return np.nan
    x=float(m.group(1)); return round(x*25.4,1) if x<=3 else x

def build_families(d):
    """Mixture_Family_ID from hard-rule blocks + Ward similarity clustering (NO target)."""
    nb=pd.cut(nz(d,"NMAS_mm"),[0,10,13,20,100],labels=["9","12","19","25"]).astype(str)
    block=d.get("MixType","NA").astype(str)+"|"+nb
    feats=[c for c in FAMILY_FEATS if c in d.columns]
    fam=np.empty(len(d),dtype=object); nxt=0
    for b,idx in d.groupby(block).groups.items():
        idx=np.array(list(idx)); sub=d.loc[idx,feats].apply(pd.to_numeric,errors="coerce")
        sub=sub.fillna(sub.median()).fillna(0.0)   # median per column, then 0 for all-NaN columns
        if len(idx)<3 or sub.std().sum()==0:
            for i in idx: fam[d.index.get_loc(i)]=f"F{nxt}"; nxt+=1
            continue
        Z=StandardScaler().fit_transform(sub.values)
        lab=AgglomerativeClustering(n_clusters=None,distance_threshold=FAMILY_DISTANCE,linkage="ward").fit_predict(Z)
        for i,l in zip(idx,lab): fam[d.index.get_loc(i)]=f"F{nxt+l}"
        nxt+=lab.max()+1
    return pd.Series(fam,index=d.index).astype(str)

def load(sheet,target_col,target_name):
    path=DOWNLOADS/DATA
    if not path.exists():
        alt=Path.cwd()/DATA
        if alt.exists(): path=alt
    d=pd.read_excel(path,sheet).rename(columns=RENAME)
    d[target_name]=pd.to_numeric(d[target_col],errors="coerce")
    d=d[d[target_name].notna() & (d[target_name]>0)].reset_index(drop=True)
    d["NMAS_mm"]=d["Nominal_Aggregate_Size"].apply(parse_nmas) if "Nominal_Aggregate_Size" in d.columns else np.nan
    def basemix(r):
        mid=re.sub(r'v\d*$','',str(r.get("Mix_ID",""))).rstrip('-_ ')
        return f'{r.get("Project_ID","")}|{r.get("Plant_Code","")}|{mid}'
    d["Exact_JMF_ID"]=d["JMF_Record_Key"].astype(str) if "JMF_Record_Key" in d.columns else d.index.astype(str)
    d["Base_Mix_ID"]=d.apply(basemix,axis=1)
    d["Mixture_Family_ID"]=build_families(d)
    return d

def get_X(d,feats):
    cols=[c for c in feats if c in d.columns]; X=d[cols].copy(); num,cat=[],[]
    for c in cols:
        if c in CATEG: cat.append(c); X[c]=X[c].astype("object").where(X[c].notna(),"NA").astype(str)
        else: num.append(c); X[c]=pd.to_numeric(X[c],errors="coerce")
    return X,num,cat

def prune(X,y):
    numX=X.select_dtypes("number"); corr=numX.corr("spearman").abs(); tc=numX.apply(lambda s:abs(s.corr(y,"spearman")))
    kept,drop=[],[]
    for f in tc.sort_values(ascending=False).index:
        tw=next((k for k in kept if corr.loc[f,k]>=0.90),None)
        if tw: drop.append((f,tw));
        else: kept.append(f)
    return kept+[c for c in X.columns if c in CATEG],drop

def prep(est,num,cat,scale=False):
    steps=[("num",Pipeline([("i",SimpleImputer(strategy="median"))]+([("s",RobustScaler())] if scale else [])),num)]
    if cat: steps.append(("cat",Pipeline([("i",SimpleImputer(strategy="constant",fill_value="NA")),("o",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat))
    return Pipeline([("prep",ColumnTransformer(steps)),("model",est)])

def zoo(num,cat):
    z={"SVR_RBF":prep(SVR(kernel="rbf",C=10,gamma="scale",epsilon=0.2),num,cat,True),
       "Huber":prep(HuberRegressor(max_iter=2000),num,cat,True),
       "HistGB":prep(HistGradientBoostingRegressor(learning_rate=0.05,max_iter=500,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.0,random_state=RANDOM_STATE),num,cat)}
    if HAS_LGBM: z["LightGBM"]=prep(LGBMRegressor(n_estimators=700,learning_rate=0.02,num_leaves=31,min_child_samples=30,subsample=0.85,colsample_bytree=0.8,reg_lambda=10,random_state=RANDOM_STATE,verbose=-1),num,cat)
    if HAS_XGB: z["XGBoost"]=prep(XGBRegressor(objective="reg:squarederror",tree_method="hist",n_estimators=700,learning_rate=0.02,max_depth=3,min_child_weight=15,subsample=0.8,colsample_bytree=0.8,reg_lambda=20,random_state=RANDOM_STATE,n_jobs=-1),num,cat)
    if HAS_CAT: z["CatBoost"]=prep(CatBoostRegressor(iterations=700,learning_rate=0.03,depth=5,l2_leaf_reg=10,verbose=0,random_seed=RANDOM_STATE),num,cat)
    return z

def grouped_oof(est,X,y,groups,w=None):
    oof=np.full(len(y),np.nan)
    for tr,va in GroupKFold(CV_FOLDS).split(X,y,groups):
        e=clone(est)
        try: e.fit(X.iloc[tr],y.iloc[tr],model__sample_weight=(None if w is None else w[tr]))
        except Exception: e.fit(X.iloc[tr],y.iloc[tr])
        oof[va]=e.predict(X.iloc[va])
    return oof

def _save(name):
    plt.tight_layout(); plt.savefig(OUT/name,dpi=200,bbox_inches="tight")
    if SHOW_PLOTS:
        try: plt.show()
        except Exception: pass
    plt.close()

def run(target_name,sheet,target_col,feats):
    print("\n"+"#"*90+f"\n{target_name} — Hierarchical JMF-Family-Aware Grouped Validation\n"+"#"*90)
    d=load(sheet,target_col,target_name)
    y=d[target_name].reset_index(drop=True)
    X,num,cat=get_X(d,feats); keep,dropped=prune(X,y); X=X[keep]
    num=[c for c in keep if c not in CATEG]; cat=[c for c in keep if c in CATEG]
    fam=d["Mixture_Family_ID"].astype(str).values
    jmf=d["Exact_JMF_ID"].astype(str).values
    # ---- family-level locked split ----
    gss=GroupShuffleSplit(n_splits=1,test_size=TEST_SIZE,random_state=RANDOM_STATE)
    dev,te=next(iter(gss.split(X,y,groups=fam)))
    fam_tr,fam_te=set(fam[dev]),set(fam[te]); jmf_tr,jmf_te=set(jmf[dev]),set(jmf[te])
    assert fam_tr.isdisjoint(fam_te), "FAMILY overlap!"
    assert jmf_tr.isdisjoint(jmf_te), "Exact-JMF overlap!"
    print("Partition summary")
    print(f"  Total rows: {len(d)} | Mixture families: {pd.Series(fam).nunique()} | Exact JMFs: {pd.Series(jmf).nunique()}")
    print(f"  Development rows: {len(dev)}  families: {len(fam_tr)}")
    print(f"  Locked-test rows: {len(te)}   families: {len(fam_te)}")
    print(f"  Family overlap: {len(fam_tr & fam_te)}  |  Exact-JMF overlap: {len(jmf_tr & jmf_te)}")
    print(f"  Dropped same-effect features: {', '.join(a+'~'+b for a,b in dropped) or 'none'}")
    # ---- model selection by grouped CV on dev ----
    Xd,Xt=X.iloc[dev],X.iloc[te]; yd,yt=y.iloc[dev],y.iloc[te]; fam_dev=fam[dev]
    # family-balanced weights on dev
    wser=pd.Series(1.0,index=np.arange(len(dev))).groupby(fam_dev).transform(lambda s:1.0/len(s)).values
    best=None; bo=-9; rows=[]
    for name,est in zoo(num,cat).items():
        try: oof=grouped_oof(est,Xd,yd,fam_dev,wser)
        except Exception as e: print(f"  {name} failed: {e}"); continue
        r=r2_score(yd,oof); rows.append((name,r))
        if r>bo: bo,best=r,(name,est)
    rows.sort(key=lambda x:-x[1]); print("  grouped-CV OOF R2 (dev): "+" | ".join(f"{n}:{r:.2f}" for n,r in rows))
    bname,bpipe=best; fin=clone(bpipe)
    try: fin.fit(Xd,yd,model__sample_weight=wser)
    except Exception: fin.fit(Xd,yd)
    pte=fin.predict(Xt)
    # ---- three reported metrics on the locked test ----
    r_unw=M(yt,pte)
    wte=pd.Series(1.0,index=np.arange(len(te))).groupby(fam[te]).transform(lambda s:1.0/len(s)).values
    ss_res=np.sum(wte*(yt.values-pte)**2); ybar=np.average(yt.values,weights=wte); ss_tot=np.sum(wte*(yt.values-ybar)**2)
    r_fambal=1-ss_res/ss_tot
    fdf=pd.DataFrame({"fam":fam[te],"y":yt.values,"p":pte}).groupby("fam").mean()
    r_famavg=r2_score(fdf["y"],fdf["p"])
    print(f"\n  BEST MODEL = {bname} | dev grouped-CV R2={bo:.3f}")
    print(f"  LOCKED TEST — (1) Unweighted report R2 = {r_unw['R2']:.3f} (RMSE {r_unw['RMSE']:.3f}, MAE {r_unw['MAE']:.3f})")
    print(f"               (2) Family-balanced report R2 = {r_fambal:.3f}")
    print(f"               (3) Family-averaged R2 = {r_famavg:.3f} (n_families_test={fdf.shape[0]})")
    # ---- plots ----
    unit="mm" if target_name=="Rut_20k" else ""
    m=r_unw; sl,ic=np.polyfit(yt.values,pte,1)
    plt.figure(figsize=(6,5.6)); plt.scatter(yt,pte,alpha=0.55,s=20)
    lo,hi=float(min(yt.min(),pte.min())),float(max(yt.max(),pte.max())); xs=np.linspace(lo,hi,100)
    plt.plot([lo,hi],[lo,hi],"--",lw=2,label="Ideal 1:1"); plt.plot(xs,sl*xs+ic,lw=2,label=f"Fit y={sl:.2f}x+{ic:.2f}")
    plt.xlabel(f"Measured {target_name} {unit}"); plt.ylabel(f"Predicted {target_name} {unit}")
    plt.title(f"{target_name} locked test — family-aware — {bname}\nR2={m['R2']:.3f} RMSE={m['RMSE']:.3f}")
    plt.legend(); plt.grid(alpha=0.3); _save(f"{target_name}_family_bestfit.png")
    plt.figure(figsize=(6,5)); plt.scatter(pte,pte-yt.values,alpha=0.55,s=20); plt.axhline(0,ls="--",lw=2)
    plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title(f"{target_name} residuals — family-aware"); plt.grid(alpha=0.3)
    _save(f"{target_name}_family_residuals.png")
    sizes=pd.Series(fam).value_counts()
    plt.figure(figsize=(6,4)); plt.hist(sizes.values,bins=range(1,sizes.max()+2),align="left")
    plt.xlabel("Reports per mixture family"); plt.ylabel("Number of families")
    plt.title(f"{target_name} family-size distribution ({pd.Series(fam).nunique()} families)"); plt.grid(alpha=0.3)
    _save(f"{target_name}_family_sizes.png")
    try:
        model=fin.named_steps["model"]; names=list(fin.named_steps["prep"].get_feature_names_out())
        imp=getattr(model,"feature_importances_",None)
        if imp is not None:
            s=pd.Series(imp,index=names).sort_values(ascending=False).head(16)[::-1]
            plt.figure(figsize=(7,max(4,len(s)*0.32))); plt.barh(s.index,s.values); plt.xlabel("Importance")
            plt.title(f"{target_name} importance — family-aware"); plt.grid(axis="x",alpha=0.3); _save(f"{target_name}_family_importance.png")
    except Exception: pass
    if HAS_SHAP:
        try:
            pre=fin.named_steps["prep"]; model=fin.named_steps["model"]; Xtt=pre.transform(Xt); names=list(pre.get_feature_names_out())
            sv=shap.TreeExplainer(model)(Xtt); plt.figure(); shap.summary_plot(sv.values,Xtt,feature_names=names,show=False,max_display=16)
            _save(f"{target_name}_family_shap.png")
        except Exception: pass
    print(f"  Figures saved to: {OUT}")

def main():
    run("Rut_20k","Rutting_Design","LWT_Design_Result",RUT_FEATURES)
    run("SCB","SCB_Design","SCB_Result",SCB_FEATURES)

if __name__=="__main__":
    main()
