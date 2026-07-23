# -*- coding: utf-8 -*-
"""
RUTTING & SCB — DESIGN-ONLY, UNIQUE-MIX, BASE-MIX-GROUPED PIPELINE
=================================================================
Implements the reviewed data-integrity + modeling plan on the updated extraction:

  DATA  : updated data extraction.xlsx, DESIGN sheets ONLY (Rutting_Design / SCB_Design).
          Validation sheets are NOT used (design-only scenario).
  UNIQUE: reduced to content-unique mixes (removes duplicate JMF records of the same mix;
          keeps genuinely distinct material/version rows).
  GROUPS: Base_Mix_ID = Project_ID | Plant_Code | Mix_ID-without-version-suffix.
          80/20 LOCKED-TEST split is GROUPED by Base_Mix_ID (no base-mix crosses the split);
          5-fold grouped CV inside the 80% dev set for honest model selection.
          A stricter Engineering-cluster (near-duplicate) grouped test is reported alongside.
  WEIGHT: JMF-frequency weights w_i = 1 / (#rows sharing the same JMF_Record_Key).
  FEATS : engineering features + correlation pruning (drop same-effect twins, |rho|>=0.90).
          (This file has NO PG / NO RAP, but HAS gyratory %Gmm_Ni/Nm densification.)
  MODELS: SVR-RBF benchmark (SimpleImputer->RobustScaler->SVR) + CatBoost, LightGBM, XGBoost,
          HistGradientBoosting, Huber; plus 10/50/90 QUANTILE boosting for uncertainty bands.
  Identifiers (JMF, Mix_ID, Base_Mix_ID, cluster) are NEVER model predictors.
"""
from __future__ import annotations
import warnings, re
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.svm import SVR
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
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
DBSCAN_EPS=1.6
HOME=Path.home()
DOWNLOADS=Path(r"C:\Users\lenovo\Downloads")
if not DOWNLOADS.exists():
    DOWNLOADS = HOME/"Downloads" if (HOME/"Downloads").exists() else Path.cwd()
DATA="updated data extraction.xlsx"

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
RUT_FEATURES=["Va","VMA","VFA","Pbe_pct","CAA","FAA","SandEq","Absorption","AsphaltContent_Design","Gmm","Gse","Gsb",
 "NMAS_mm","Grad_No4","Grad_No8","Grad_No30","Grad_No50","Grad_No100","Grad_No200","Dust_Binder","Gmm_Nini","Gmm_Nmax",
 "Fines_to_Pbe","VMA_Filled_Index","Coarse_Fraction","Intermediate_Fraction","PostDes_Densif","MixType","DesignLev"]
SCB_FEATURES=["Pbe_pct","Va","VMA","VFA","Absorption","AsphaltContent_Design","SandEq","FAA","CAA","Gse","Gmm","Dust_Binder",
 "Grad_No30","Grad_No50","Grad_No100","Grad_No200","Fines_to_Pbe","VMA_Filled_Index","MixType","DesignLev"]
CLUSTER_FEATS=["AsphaltContent_Design","Va","VMA","VFA","Gmm","CAA","FAA","Pbe_pct","Grad_No4","Grad_No200"]

def M(a,b): return dict(R2=r2_score(a,b),RMSE=float(np.sqrt(mean_squared_error(a,b))),MAE=mean_absolute_error(a,b))
def nz(d,c): return pd.to_numeric(d[c],errors="coerce") if c in d.columns else pd.Series(np.nan,index=d.index)
def parse_nmas(v):
    if pd.isna(v): return np.nan
    m=re.search(r"(\d+(?:\.\d+)?)",str(v))
    if not m: return np.nan
    x=float(m.group(1)); return round(x*25.4,1) if x<=3 else x
def base_mix(row):
    mid=re.sub(r'v\d*$','',str(row.get("Mix_ID",""))).rstrip('-_ ')
    return f'{row.get("Project_ID","")}|{row.get("Plant_Code","")}|{mid}'

def load(sheet, target_col, target_name):
    path=DOWNLOADS/DATA
    if not path.exists():
        alt=Path.cwd()/DATA
        if alt.exists(): path=alt
    d=pd.read_excel(path,sheet).rename(columns=RENAME)
    d[target_name]=pd.to_numeric(d[target_col],errors="coerce")
    d=d[d[target_name].notna() & (d[target_name]>0)].reset_index(drop=True)
    d["NMAS_mm"]=d["Nominal_Aggregate_Size"].apply(parse_nmas) if "Nominal_Aggregate_Size" in d.columns else np.nan
    d["Base_Mix_ID"]=d.apply(base_mix,axis=1)
    # JMF-frequency weight = 1 / (#rows sharing the same JMF_Record_Key)
    if "JMF_Record_Key" in d.columns:
        d["w"]=d.groupby("JMF_Record_Key")["JMF_Record_Key"].transform("size").rdiv(1.0)
    else: d["w"]=1.0
    # engineered features
    d["Fines_to_Pbe"]=nz(d,"Grad_No200")/nz(d,"Pbe_pct").replace(0,np.nan)
    d["VMA_Filled_Index"]=nz(d,"VMA")*nz(d,"VFA")/100.0
    d["Coarse_Fraction"]=100-nz(d,"Grad_No4"); d["Intermediate_Fraction"]=nz(d,"Grad_No4")-nz(d,"Grad_No8")
    d["PostDes_Densif"]=nz(d,"Gmm_Nmax")-(100-nz(d,"Va"))
    # UNIQUE MIXES: drop content-duplicate rows (same features+target)
    sigcols=[c for c in CLUSTER_FEATS if c in d.columns]
    sig=pd.Series(["|".join(map(str,r)) for r in d[sigcols].round(3).fillna(-9).values])+"|"+d[target_name].round(3).astype(str)
    before=len(d); d=d.loc[~sig.duplicated()].reset_index(drop=True)
    print(f"{target_name}: {sheet} {before} rows -> {len(d)} UNIQUE mixes "
          f"| Base_Mix families={d['Base_Mix_ID'].nunique()} (NO PG/RAP in this file; HAS gyratory Ni/Nm)")
    return d

def get_X(d,feats):
    cols=[c for c in feats if c in d.columns]; X=d[cols].copy()
    num,cat=[],[]
    for c in cols:
        if c in CATEG: cat.append(c); X[c]=X[c].astype("object").where(X[c].notna(),"NA").astype(str)
        else: num.append(c); X[c]=pd.to_numeric(X[c],errors="coerce")
    return X,num,cat

def prune(X,y):
    num=X.select_dtypes("number"); corr=num.corr("spearman").abs(); tc=num.apply(lambda s:abs(s.corr(y,"spearman")))
    kept,drop=[],[]
    for f in tc.sort_values(ascending=False).index:
        tw=next((k for k in kept if corr.loc[f,k]>=0.90),None)
        if tw: drop.append((f,tw,round(corr.loc[f,tw],2)))
        else: kept.append(f)
    return kept+[c for c in X.columns if c in CATEG], drop

def clusters(d):
    cid=np.array([f"c{i}" for i in range(len(d))],dtype=object); nxt=0
    feats=[c for c in CLUSTER_FEATS if c in d.columns]
    nb=pd.cut(nz(d,"NMAS_mm"),[0,10,13,20,100],labels=["9","12","19","25"]).astype(str)
    blk=d.get("MixType","NA").astype(str)+"|"+nb
    for b,idx in d.groupby(blk).groups.items():
        idx=np.array(list(idx)); sub=d.loc[idx,feats].apply(pd.to_numeric,errors="coerce"); sub=sub.fillna(sub.median())
        if len(idx)<2 or sub.std().sum()==0:
            for i in idx: cid[d.index.get_loc(i)]=f"C{nxt}"; nxt+=1
            continue
        lab=DBSCAN(eps=DBSCAN_EPS,min_samples=2).fit_predict(StandardScaler().fit_transform(sub.values))
        for i,l in zip(idx,lab): cid[d.index.get_loc(i)]=f"C{nxt+l}" if l>=0 else f"C{nxt+9000+i}"
        nxt+=(lab.max()+1 if lab.max()>=0 else 0)+1
    return cid.astype(str)

def prep(est,num,cat,scale=False):
    steps=[("num", Pipeline([("i",SimpleImputer(strategy="median"))]+([("s",RobustScaler())] if scale else [])), num)]
    if cat: steps.append(("cat",Pipeline([("i",SimpleImputer(strategy="constant",fill_value="NA")),("o",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat))
    return Pipeline([("prep",ColumnTransformer(steps)),("model",est)])

def model_zoo(num,cat):
    z={}
    z["SVR_RBF"]=prep(SVR(kernel="rbf",C=10,gamma="scale",epsilon=0.2),num,cat,scale=True)
    z["Huber"]=prep(HuberRegressor(max_iter=2000),num,cat,scale=True)
    z["HistGB"]=prep(HistGradientBoostingRegressor(learning_rate=0.05,max_iter=500,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.0,random_state=RANDOM_STATE),num,cat)
    if HAS_LGBM: z["LightGBM"]=prep(LGBMRegressor(n_estimators=700,learning_rate=0.02,num_leaves=31,min_child_samples=30,subsample=0.85,colsample_bytree=0.8,reg_lambda=10,random_state=RANDOM_STATE,verbose=-1),num,cat)
    if HAS_XGB: z["XGBoost"]=prep(XGBRegressor(objective="reg:squarederror",tree_method="hist",n_estimators=700,learning_rate=0.02,max_depth=3,min_child_weight=15,subsample=0.8,colsample_bytree=0.8,reg_lambda=20,random_state=RANDOM_STATE,n_jobs=-1),num,cat)
    if HAS_CAT: z["CatBoost"]=prep(CatBoostRegressor(iterations=700,learning_rate=0.03,depth=5,l2_leaf_reg=10,verbose=0,random_seed=RANDOM_STATE),num,cat)
    return z

def grouped_oof(est,X,y,groups,w=None):
    oof=np.full(len(y),np.nan)
    for tr,va in GroupKFold(CV_FOLDS).split(X,y,groups):
        e=clone(est)
        try: e.fit(X.iloc[tr],y.iloc[tr], model__sample_weight=(w[tr] if w is not None else None))
        except Exception: e.fit(X.iloc[tr],y.iloc[tr])
        oof[va]=e.predict(X.iloc[va])
    return oof

def split_groups(y,groups):
    b=pd.qcut(y,5,labels=False,duplicates="drop")
    dev,te=next(iter(StratifiedGroupKFold(max(2,round(1/TEST_SIZE)),shuffle=True,random_state=RANDOM_STATE).split(np.zeros(len(y)),b,groups)))
    return dev,te

def run(target_name, sheet, target_col, feats):
    print("\n"+"#"*88+f"\n{target_name}\n"+"#"*88)
    d=load(sheet,target_col,target_name)
    y=d[target_name].reset_index(drop=True)
    X,num,cat=get_X(d,feats)
    keep,dropped=prune(X,y); X=X[keep]; num=[c for c in keep if c not in CATEG]; cat=[c for c in keep if c in CATEG]
    print("Dropped same-effect features:", ", ".join(f"{a}~{b}({r})" for a,b,r in dropped) or "none")
    w=d["w"].values
    for gname,groups in [("Base-Mix (primary)", d["Base_Mix_ID"].astype(str).values),
                         ("Eng-Cluster (strict)", clusters(d))]:
        dev,te=split_groups(y,groups)
        Xd,Xt=X.iloc[dev],X.iloc[te]; yd,yt=y.iloc[dev],y.iloc[te]; wd=w[dev]; gd=groups[dev]
        leak=len(set(groups[dev])&set(groups[te]))
        print(f"\n  [{gname}]  dev={len(dev)} test={len(te)} groups(dev)={pd.Series(gd).nunique()} leak={leak}")
        rows=[]; best=None; bestoof=-9
        for name,est in model_zoo(num,cat).items():
            try: oof=grouped_oof(est,Xd,yd,gd,wd)
            except Exception as e: print(f"    {name} failed: {e}"); continue
            r2=r2_score(yd,oof); rows.append((name,r2))
            if r2>bestoof: bestoof,best=r2,(name,est)
        rows.sort(key=lambda x:-x[1])
        print("    grouped-CV OOF R2 (dev): "+" | ".join(f"{n}:{r:.2f}" for n,r in rows))
        bname,bpipe=best; fin=clone(bpipe)
        try: fin.fit(Xd,yd,model__sample_weight=wd)
        except Exception: fin.fit(Xd,yd)
        tm=M(yt,fin.predict(Xt))
        print(f"    BEST={bname} | dev grouped-CV R2={bestoof:.3f} | LOCKED-TEST R2={tm['R2']:.3f} RMSE={tm['RMSE']:.3f} MAE={tm['MAE']:.3f}")
        # 10/50/90 quantile uncertainty band (GradientBoosting) on the primary grouping only
        if gname.startswith("Base-Mix"):
            try:
                qp={}
                for a in (0.1,0.5,0.9):
                    qm=prep(GradientBoostingRegressor(loss="quantile",alpha=a,n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.85,random_state=RANDOM_STATE),num,cat)
                    qm.fit(Xd,yd); qp[a]=qm.predict(Xt)
                cover=float(np.mean((yt.values>=qp[0.1])&(yt.values<=qp[0.9])))
                width=float(np.mean(qp[0.9]-qp[0.1]))
                print(f"    Quantile band: 80% PI coverage={cover:.2f} (target 0.80), mean width={width:.2f} {'mm' if target_name=='Rut_20k' else ''}")
            except Exception as e: print("    quantile band failed:",e)

def main():
    run("Rut_20k","Rutting_Design","LWT_Design_Result",RUT_FEATURES)
    run("SCB","SCB_Design","SCB_Result",SCB_FEATURES)

if __name__=="__main__":
    main()
