# -*- coding: utf-8 -*-
"""
MODEL DIAGNOSTICS v6  --  RUT + SCB
================================================================================
Adds, on top of v5:

  A) JMF DATA-INTEGRITY AUDIT  (audit_jmf_alignment)
        Prints exactly which raw column every model feature came from, flags
        parsing failures (PG grade, ADT band, Polymer/Latex), and writes a
        per-mix table  <NAME>_MODEL_INPUT_AUDIT.xlsx  so you can cross-check the
        numbers the model actually saw against the source LaPave JMF PDF
        (e.g. Mix 00780-0003v3 / JMF11 L1WCR).

  B) SAFER PG / ADT / POLYMER PARSING  (resolve_pg_high, adt_ordinal, is_polymer)
        Prefers a clean explicit column when the workbook has one
        (PG_HighTemp / "PG High" / ...), and only falls back to Custom_Name
        regex + LA-DOTD heuristic when it does not.  Prints which source was
        used for every row so nothing is silently invented.

  C) TRAIN / VALIDATION / TEST SEPARATE PLOTS  (fig_train_val_test)
        Grouped three-way split (by Exact_JMF_Group, no replicate leakage).
        One panel per model, purple = train, orange = validation, green = test,
        black dashed 1:1 line, red best-fit line, and a per-subset R2 box --
        the same layout as Fig. 7 in the reference paper.

  D) PARTIAL DEPENDENCE PLOTS  (fig_pdp)
        1-D PDP for the top SHAP features (paper Fig. 10 style).

Everything from v5 (distribution, actual-vs-pred, CV box, overlay, relative
error, SHAP beeswarm + polar, Spearman) is kept.

Run in Spyder: set FILE at top -> press F5.
Requirements: pandas numpy scikit-learn lightgbm xgboost catboost shap scipy
              matplotlib openpyxl
"""
import os, glob, re, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr, gaussian_kde
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import shap

# =============================== CONFIG =====================================
FILE     = r"Book1_filtered_RUT05_10_SCB030_125_KEEP_REPLICATES.xlsx"
CV_FOLDS = 10
RANDOM   = 42
FAST     = False
HEADLESS = False
OUTDIR   = "."
AUDIT    = True     # write per-mix MODEL_INPUT_AUDIT.xlsx and print alignment report

rcParams.update({"figure.dpi":110,"savefig.dpi":300,"font.size":12,
                 "axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":9,
                 "axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})

def resolve_file(name):
    if os.path.isfile(name): return name
    base=os.path.basename(name)
    for root in [os.getcwd(),os.path.expanduser("~/Downloads"),
                 os.path.expanduser("~"),"C:/Users"]:
        for pat in [base,"*KEEP_REPLICATES*.xlsx","*filtered*RUT*.xlsx"]:
            hit=glob.glob(os.path.join(root,"**",pat),recursive=True)
            if hit: print(f"[info] using workbook: {hit[0]}"); return hit[0]
    raise FileNotFoundError(f"Set FILE at top of script to the full path of '{base}'.")

# =========================== FEATURES ========================================
NMAS_MAP={"1/2 in.":12.5,"3/4 in.":19.0,"1 in.":25.0,"3/8 in.":9.5,"1/4 in.":6.25}
SA_FACTOR={"Pass_No_4":0.41,"Pass_No_8":0.82,"Pass_No_16":1.64,"Pass_No_30":2.87,
           "Pass_No_50":6.14,"Pass_No_100":12.29,"Pass_No_200":32.77}
num=lambda s: pd.to_numeric(s,errors="coerce")

def pick(df,*c):
    for x in c:
        if x in df.columns: return num(df[x])
    return pd.Series(np.nan,index=df.index)

def first_col(df,*names):
    """Return the first column name present in df (case/space-insensitive)."""
    norm={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in df.columns}
    for n in names:
        k=re.sub(r'[^a-z0-9]','',str(n).lower())
        if k in norm: return norm[k]
    return None

# ---- PG / ADT / Polymer: prefer a clean column, fall back transparently -----
# These three are the fields most likely to be mis-parsed, because on the JMF
# the PG grade and the "+Latex" tag live in the binder ROW ("PG 67-22+Latex"),
# NOT in the mix Custom_Name ("JMF11 L1WCR"); and ADT is often a band (">7000").
PG_COL_ALIASES = ["PG_HighTemp","PG High","PG_High","PGHigh","PG_High_Temp",
                  "PGHighTemp","Binder_PG_High"]
PG_LOW_ALIASES = ["PG_LowTemp","PG Low","PG_Low","PGLow","Binder_PG_Low"]
BINDER_NAME_ALIASES = ["Binder_Name","PS_Name","Binder","AsphaltBinder",
                       "PG_Grade","Grade","Material_Name"]

def _parse_pg(text):
    m=re.search(r'(\d{2,3})\s*[-\u2013]\s*(\d{2})',str(text).upper())
    if m: return float(m.group(1)),-float(m.group(2))
    return np.nan,np.nan

def fill_pg_high_heuristic(df):
    """LA-DOTD binder-selection fallback (only for rows with no parsed PG)."""
    dl=df["Design_Level"].astype(str).str.upper() if "Design_Level" in df else pd.Series("",index=df.index)
    mt=df["Mix_Type"].astype(str).str.upper() if "Mix_Type" in df else pd.Series("",index=df.index)
    adt_raw=df["ADT"] if "ADT" in df else pd.Series("",index=df.index)
    adt=adt_ordinal(adt_raw)  # numeric band midpoint, see below
    est=pd.Series(np.nan,index=df.index)
    est[dl.str.contains("LOW ADT")]=64
    est[dl.eq("1")|dl.eq("1F")]=67
    est[dl.eq("2")|dl.eq("2F")]=76
    est[dl.eq("A")]=76; est[dl.eq("SMA")]=76; est[dl.eq("OGFC")]=76
    est[dl.eq("THIN LIFT")]=70
    est[adt>7000]=76
    est[adt>15000]=82
    est[mt.str.contains("BASE") & est.isna()]=64
    return est.fillna(67)

def adt_ordinal(series):
    """Convert LA-DOTD ADT which may be numeric OR a band string ('>7000',
    '3000-7000', '< 3000') into a numeric proxy so it is never silently 0."""
    def conv(v):
        s=str(v).strip().lower()
        if s in ("","nan","none"): return np.nan
        n=pd.to_numeric(s,errors="coerce")
        if not pd.isna(n): return float(n)
        # band strings -> representative value
        nums=re.findall(r'\d+',s)
        nums=[float(x) for x in nums]
        if ">" in s and nums:  return nums[0]*1.5      # ">7000" -> 10500
        if "<" in s and nums:  return nums[0]*0.5
        if len(nums)>=2:       return (nums[0]+nums[1])/2
        if nums:               return nums[0]
        return np.nan
    return series.apply(conv)

def resolve_pg_high(df, report):
    """Return (PG_High, PG_Low, source_tag). Preference order:
       1) explicit clean PG column   2) Custom_Name regex
       3) binder-name column regex   4) LA-DOTD heuristic."""
    n=len(df)
    pg_hi=pd.Series(np.nan,index=df.index); pg_lo=pd.Series(np.nan,index=df.index)
    src=pd.Series("",index=df.index)

    col_hi=first_col(df,*PG_COL_ALIASES)
    col_lo=first_col(df,*PG_LOW_ALIASES)
    if col_hi is not None:
        v=num(df[col_hi]); m=v.notna()
        pg_hi[m]=v[m]; src[m]="explicit_col:"+col_hi
    if col_lo is not None:
        v=num(df[col_lo]); m=v.notna()&pg_lo.isna()
        pg_lo[m]=v[m]

    # Custom_Name regex (only where still missing)
    if "Custom_Name" in df:
        parsed=df["Custom_Name"].apply(_parse_pg)
        ph=parsed.apply(lambda t:t[0]); pl=parsed.apply(lambda t:t[1])
        m=pg_hi.isna()&ph.notna(); pg_hi[m]=ph[m]; src[m]="Custom_Name_regex"
        m=pg_lo.isna()&pl.notna(); pg_lo[m]=pl[m]

    # binder-name column regex (this is where "PG 67-22+Latex" usually lives)
    bcol=first_col(df,*BINDER_NAME_ALIASES)
    if bcol is not None:
        parsed=df[bcol].apply(_parse_pg)
        ph=parsed.apply(lambda t:t[0]); pl=parsed.apply(lambda t:t[1])
        m=pg_hi.isna()&ph.notna(); pg_hi[m]=ph[m]; src[m]="binder_col:"+bcol
        m=pg_lo.isna()&pl.notna(); pg_lo[m]=pl[m]

    # heuristic fallback
    est=fill_pg_high_heuristic(df)
    m=pg_hi.isna(); pg_hi[m]=est[m]; src[m]="LA-DOTD_heuristic"
    pg_lo=pg_lo.fillna(-22.0)

    counts=src.value_counts().to_dict()
    report.append(("PG_High source breakdown (rows)", counts))
    report.append(("PG_High from a REAL value (not heuristic) %",
                   round(100*(src!="LA-DOTD_heuristic").mean(),1)))
    return pg_hi, pg_lo, src

def is_polymer(df, report):
    """Latex/SBS flag. On the JMF the modifier is in the binder row
    ('PG 67-22+Latex' / 'Latex' line), so check binder-name column first,
    then Custom_Name."""
    pat=r'SBS|POLY|LATEX|MODIF|ELVALOY'
    flag=pd.Series(0.0,index=df.index); hit_src="none"
    bcol=first_col(df,*BINDER_NAME_ALIASES)
    if bcol is not None:
        f=df[bcol].astype(str).str.upper().str.contains(pat,regex=True).astype(float)
        flag=np.maximum(flag,f); hit_src=bcol
    if "Custom_Name" in df:
        f=df["Custom_Name"].astype(str).str.upper().str.contains(pat,regex=True).astype(float)
        flag=np.maximum(flag,f)
    report.append(("Polymer/Latex flagged rows (%)", round(100*flag.mean(),1)))
    report.append(("Polymer source column", hit_src))
    return pd.Series(flag,index=df.index)

# ---- provenance map: model feature -> JMF field it should equal -------------
FEATURE_TO_JMF = {
 "Va":"DESIGN %Voids (Submittal)", "VMA":"DESIGN VMA", "VFA":"DESIGN VFA",
 "Pbe":"DESIGN Pbe", "Pba":"DESIGN Pba", "Gse":"DESIGN Gse",
 "AC":"DESIGN % AC", "Dust_Pbe":"DESIGN Dust/Pbeff",
 "Gmm_Ni":"DESIGN %Gmm,Ni", "Gmm_Nm":"DESIGN %Gmm,Nm",
 "Absorption":"Combined Aggregate Absp", "FAA":"Combined Aggregate FAA",
 "CAA":"Combined Aggregate CAA", "SandEq":"Combined Aggregate Sand Eq",
 "FlatElong":"Combined Aggregate Flat/Elng", "Gsb":"Combined Aggregate Bulk Grav.",
 "Pass_No_4":"Gradation Pass No.4","Pass_No_8":"Gradation Pass No.8",
 "Pass_No_16":"Gradation Pass No.16","Pass_No_30":"Gradation Pass No.30",
 "Pass_No_50":"Gradation Pass No.50","Pass_No_100":"Gradation Pass No.100",
 "Pass_No_200":"Gradation Pass No.200",
 "ADT":"Header ADT (band -> numeric)", "MixTemp":"Header Mix Temp",
 "AC_from_RAP":"Total %AC from RAP", "Production_Rate":"Header Prod.Rate",
 "NMAS_mm":"Header Nom.Agg.Size", "PG_High":"Binder row PG high temp",
 "PG_Low":"Binder row PG low temp", "Polymer":"Binder row modifier (+Latex/SBS)",
}

def build_features(df, report=None):
    if report is None: report=[]
    P,A="Design_Submission__","Combined_Aggregate_"; X=pd.DataFrame(index=df.index)
    X["Va"]=pick(df,P+"Percent_Voids");X["VMA"]=pick(df,P+"VMA");X["VFA"]=pick(df,P+"VFA")
    X["Pbe"]=pick(df,P+"Pbe");X["Pba"]=pick(df,P+"Pba");X["Gse"]=pick(df,P+"Gse")
    X["AC"]=pick(df,P+"Percent_AC");X["Dust_Pbe"]=pick(df,P+"Dust_Pbeff")
    X["Gmm_Ni"]=pick(df,P+"Percent_Gmm_Ni");X["Gmm_Nm"]=pick(df,P+"Percent_Gmm_Nm")
    X["Absorption"]=pick(df,A+"Absorption");X["FAA"]=pick(df,A+"FAA");X["CAA"]=pick(df,A+"CAA")
    X["SandEq"]=pick(df,A+"Sand_Equivalent");X["FlatElong"]=pick(df,A+"Flat_Elongated")
    X["Gsb"]=pick(df,A+"Bulk_Gravity")
    for s in ["Pass_No_4","Pass_No_8","Pass_No_16","Pass_No_30","Pass_No_50","Pass_No_100","Pass_No_200"]:
        X[s]=pick(df,P+s)
    # ADT: band-safe ordinal instead of raw num() (which turned '>7000' into NaN->0)
    X["ADT"]=adt_ordinal(df["ADT"]) if "ADT" in df else np.nan
    X["MixTemp"]=pick(df,"Mix_Temperature")
    X["AC_from_RAP"]=pick(df,"Total_AC_From_RAP");X["Production_Rate"]=pick(df,"Production_Rate")
    if "Nominal_Aggregate_Size" in df:
        v=df["Nominal_Aggregate_Size"]
        X["NMAS_mm"]=v.map(NMAS_MAP) if v.dtype==object else num(v)
    # PG grade (clean-column-first, transparent fallback)
    pg_hi,pg_lo,pg_src=resolve_pg_high(df,report)
    X["PG_High"]=pg_hi; X["PG_Low"]=pg_lo
    X["PG_span"]=X["PG_High"]-X["PG_Low"]
    X["PG_from_real_value"]=(pg_src!="LA-DOTD_heuristic").astype(float)
    X["Polymer"]=is_polymer(df,report).values
    # engineered material properties
    X["RAP_Binder_Ratio"]=X["AC_from_RAP"]/X["AC"].replace(0,np.nan)
    X["Fine_Fraction"]=X["Pass_No_8"]-X["Pass_No_200"]
    X["Intermediate_Frac"]=X["Pass_No_4"]-X["Pass_No_8"]
    X["VMA_Filled_Index"]=X["VMA"]*X["VFA"]/100
    sa=sum(X[k]*v for k,v in SA_FACTOR.items())/100+0.41
    X["AFT_micron"]=(X["Pbe"]/100)/(sa*X["Gsb"].replace(0,np.nan))*1000
    # PG interactions
    X["PG_x_Va"]  = X["PG_High"]*X["Va"]
    X["PG_x_ADT"] = X["PG_High"]*np.log1p(X["ADT"].fillna(0))
    X["PG_x_RBR"] = X["PG_High"]*X["RAP_Binder_Ratio"]
    X["PG_x_AFT"] = X["PG_High"]*X["AFT_micron"]
    # one-hot categoricals
    for c in ["Design_Level","Mix_Type"]:
        if c in df:
            d=pd.get_dummies(df[c].astype(str),prefix=c).astype(float)
            X=pd.concat([X,d.set_index(X.index)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]
    return X, report

# ============================ DATA AUDIT =====================================
def audit_jmf_alignment(df, X, y, grp, name, report):
    """Print how faithfully the model inputs match the source JMF and dump a
    per-mix table for manual cross-checking against the JMF PDF."""
    print("\n"+"-"*72+f"\n  [AUDIT] {name}: JMF -> model-input alignment\n"+"-"*72)
    print(f"  raw workbook columns ({df.shape[1]}):")
    for c in df.columns: print("      ", c)
    print(f"\n  model features built ({X.shape[1]}): {list(X.columns)}")

    # parsing-health checks (the fields most likely to be wrong)
    print("\n  parsing-health checks:")
    for label,val in report:
        print(f"      - {label}: {val}")
    if "ADT" in df:
        raw=df["ADT"].astype(str)
        nonnum=raw[~raw.str.match(r'^\s*-?\d+(\.\d+)?\s*$',na=False)].unique()[:8]
        print(f"      - ADT non-numeric tokens seen (band strings): {list(nonnum)}")
    if "Custom_Name" in df:
        pg_in_cn=df["Custom_Name"].astype(str).str.contains(r'\d{2,3}\s*[-\u2013]\s*\d{2}').mean()
        print(f"      - Custom_Name rows that literally contain a PG grade: {round(100*pg_in_cn,1)}%")
        print(f"      - sample Custom_Name values: {list(df['Custom_Name'].astype(str).unique()[:5])}")

    # provenance table: feature -> JMF field -> value seen for the first mix
    prov=pd.DataFrame({
        "model_feature":list(FEATURE_TO_JMF.keys()),
        "expected_JMF_field":list(FEATURE_TO_JMF.values())})
    prov["in_X"]=prov["model_feature"].isin(X.columns)
    prov["first_row_value"]=[round(float(X[f].iloc[0]),4) if f in X.columns else None
                             for f in prov["model_feature"]]
    print("\n  feature -> JMF field provenance (value = what the model saw for row 0):")
    with pd.option_context("display.max_rows",None,"display.width",120):
        print(prov.to_string(index=False))

    if AUDIT:
        idcols=[c for c in ["Exact_JMF_Group","Custom_Name","Mix_ID","Mix_Type",
                            "Design_Level","ADT"] if c in df.columns]
        out=pd.concat([df[idcols].reset_index(drop=True),
                       X.reset_index(drop=True),
                       pd.Series(y,name="TARGET")],axis=1)
        # one row per unique mix keeps it easy to line up with a single JMF PDF
        path=f"{OUTDIR}/{name}_MODEL_INPUT_AUDIT.xlsx"
        try:
            with pd.ExcelWriter(path) as xl:
                out.to_excel(xl,"all_rows",index=False)
                prov.to_excel(xl,"feature_provenance",index=False)
                if "Exact_JMF_Group" in df.columns:
                    out.drop_duplicates("Exact_JMF_Group").to_excel(xl,"one_row_per_mix",index=False)
            print(f"\n  [AUDIT] wrote {path}  <- open this and compare a row to your JMF PDF")
        except Exception as e:
            print(f"  [AUDIT] could not write xlsx ({e}); CSV instead")
            out.to_csv(f"{OUTDIR}/{name}_MODEL_INPUT_AUDIT.csv",index=False)

def load_target(path,sheet,tgt):
    df=pd.read_excel(path,sheet_name=sheet)
    y=num(df[tgt]).values
    grp=df["Exact_JMF_Group"].astype(str).values
    X,report=build_features(df)
    if AUDIT: audit_jmf_alignment(df,X,y,grp,sheet.split("_")[0],report)
    if FAST:
        rng=np.random.RandomState(RANDOM); idx=rng.choice(len(X),min(1200,len(X)),replace=False)
        X,y,grp=X.iloc[idx].reset_index(drop=True),y[idx],grp[idx]
    return X,y,grp

# ============================ MODELS (tuned) =================================
def zoo():
    T=300 if FAST else 1000; L=300 if FAST else 800
    return {
        "ExtraTrees":  ExtraTreesRegressor(n_estimators=T,min_samples_leaf=1,max_features=0.6,
                          n_jobs=-1,random_state=RANDOM),
        "RandomForest":RandomForestRegressor(n_estimators=int(T*0.8),min_samples_leaf=2,max_features=0.7,
                          n_jobs=-1,random_state=RANDOM),
        "LightGBM":    lgb.LGBMRegressor(n_estimators=L,learning_rate=0.03,num_leaves=63,
                          min_child_samples=15,subsample=0.85,colsample_bytree=0.7,
                          reg_lambda=1.0,random_state=RANDOM,n_jobs=-1,verbose=-1),
        "XGBoost":     xgb.XGBRegressor(n_estimators=L,learning_rate=0.03,max_depth=6,
                          subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,min_child_weight=3,
                          random_state=RANDOM,n_jobs=-1,verbosity=0),
        "CatBoost":    CatBoostRegressor(iterations=L,learning_rate=0.03,depth=6,l2_leaf_reg=3.0,
                          random_state=RANDOM,verbose=0),
        "HistGB":      HistGradientBoostingRegressor(max_iter=L,learning_rate=0.03,max_leaf_nodes=63,
                          min_samples_leaf=15,l2_regularization=0.1,random_state=RANDOM),
    }
STYLE={"ExtraTrees":("s","#1f77b4"),"RandomForest":("^","#2ca02c"),"LightGBM":("*","#17becf"),
       "XGBoost":("o","#9467bd"),"CatBoost":("h","#bcbd22"),"HistGB":("<","#ff7f0e"),
       "ENSEMBLE":("D","#d62728")}
# paper Fig.7 subset colours
SUBSET_STYLE={"Training":("D","#7e57c2"),"Validation":("s","#ff9800"),"Test":("^","#2ca02c")}

def metrics(y,p):
    return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p))

def optimize_ensemble(oof_dict, y):
    keys=list(oof_dict); best=(-9,None); grid=np.arange(0,1.01,0.1)
    for w in itertools.product(grid,repeat=len(keys)):
        if abs(sum(w)-1)>1e-6: continue
        p=sum(w[i]*oof_dict[keys[i]] for i in range(len(keys)))
        s=r2_score(y,p)
        if s>best[0]: best=(s,dict(zip(keys,w)))
    return best

def grouped_three_way(X,y,grp,seed=RANDOM):
    """Grouped train/validation/test split (~70/15/15) with NO mix straddling
    two subsets. Uses two StratifiedGroupKFold passes on target bins.
    Split counts are capped by the number of unique mixes so small datasets
    (or a smoke test) never trip StratifiedGroupKFold's members-per-class rule."""
    ng=len(np.unique(grp))
    k1=max(3,min(7,ng//2)); k2=max(3,min(6,ng//3))          # ~1/k1 -> test, ~1/k2 of dev -> val
    nbin=max(2,min(10,ng//max(1,k1)))
    ybin=pd.qcut(pd.Series(y).rank(method="first"),nbin,labels=False,duplicates="drop")
    s1=StratifiedGroupKFold(n_splits=k1,shuffle=True,random_state=seed)
    dev_idx,test_idx=next(iter(s1.split(X,ybin,groups=grp)))
    s2=StratifiedGroupKFold(n_splits=k2,shuffle=True,random_state=seed)
    d_tr,d_va=next(iter(s2.split(X.iloc[dev_idx],ybin[dev_idx],groups=grp[dev_idx])))
    tr_idx=dev_idx[d_tr]; va_idx=dev_idx[d_va]
    return np.sort(tr_idx),np.sort(va_idx),np.sort(test_idx)

# ============================ CORE ===========================================
def compute(path,name,sheet,tgt,unit):
    print("\n"+"="*72+f"\n  {name}   (v6: audit + PG-safe + train/val/test + PDP)\n"+"="*72)
    X,y,grp=load_target(path,sheet,tgt); feats=list(X.columns)
    print(f"  rows={len(X)}  unique mixes={len(np.unique(grp))}  features={X.shape[1]}  "
          f"target range=[{y.min():.2f},{y.max():.2f}]")
    ybin=pd.qcut(y,10,labels=False,duplicates="drop")
    sgkf=StratifiedGroupKFold(CV_FOLDS,shuffle=True,random_state=RANDOM)
    folds=list(sgkf.split(X,ybin,groups=grp))
    tr0,te0=folds[0]
    # grouped three-way split for the paper-style train/val/test figure
    tr3,va3,te3=grouped_three_way(X,y,grp)
    print(f"  three-way split  train={len(tr3)}  val={len(va3)}  test={len(te3)} rows "
          f"({len(np.unique(grp[tr3]))}/{len(np.unique(grp[va3]))}/{len(np.unique(grp[te3]))} mixes)")

    fit_res={}; cv_r2={}; oof_all={}; tvt={}
    for nm in zoo():
        oof=np.zeros(len(y)); fold_scores=[]
        for a,b in folds:
            m=zoo()[nm]; m.fit(X.iloc[a],y[a]); oof[b]=m.predict(X.iloc[b])
            fold_scores.append(r2_score(y[b],oof[b]))
        m=zoo()[nm]; m.fit(X.iloc[tr0],y[tr0])
        ptr=m.predict(X.iloc[tr0]); pte=m.predict(X.iloc[te0])
        om=metrics(y,oof); tm=metrics(y[te0],pte); trm=metrics(y[tr0],ptr)
        fit_res[nm]=dict(oof=oof,ptr=ptr,pte=pte,fold_r2=fold_scores,
                         R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],
                         R2_te=tm["R2"],RMSE_te=tm["RMSE"],MAE_te=tm["MAE"],R2_tr=trm["R2"])
        cv_r2[nm]=np.array(fold_scores); oof_all[nm]=oof
        # ---- three-way: fit on TRAIN only, predict train/val/test separately
        m3=zoo()[nm]; m3.fit(X.iloc[tr3],y[tr3])
        p_tr=m3.predict(X.iloc[tr3]); p_va=m3.predict(X.iloc[va3]); p_te=m3.predict(X.iloc[te3])
        tvt[nm]=dict(tr=(y[tr3],p_tr,r2_score(y[tr3],p_tr)),
                     va=(y[va3],p_va,r2_score(y[va3],p_va)),
                     te=(y[te3],p_te,r2_score(y[te3],p_te)))
        print(f"    {nm:12s} OOF R2={om['R2']:.4f}  RMSE={om['RMSE']:.4f}  "
              f"{CV_FOLDS}-fold={np.mean(fold_scores):.4f}+/-{np.std(fold_scores):.3f}  "
              f"| 3way tr/va/te R2={tvt[nm]['tr'][2]:.3f}/{tvt[nm]['va'][2]:.3f}/{tvt[nm]['te'][2]:.3f}")
    # ---- weighted ensemble ----
    best_r2,W = optimize_ensemble(oof_all,y)
    ens_oof=sum(W[k]*oof_all[k] for k in W)
    ens_te =sum(W[k]*fit_res[k]["pte"] for k in W)
    ens_tr =sum(W[k]*fit_res[k]["ptr"] for k in W)
    om=metrics(y,ens_oof); tm=metrics(y[te0],ens_te); trm=metrics(y[tr0],ens_tr)
    fit_res["ENSEMBLE"]=dict(oof=ens_oof,ptr=ens_tr,pte=ens_te,fold_r2=[],
                             R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],
                             R2_te=tm["R2"],RMSE_te=tm["RMSE"],MAE_te=tm["MAE"],R2_tr=trm["R2"])
    cv_r2["ENSEMBLE"]=np.array([r2_score(y[b],ens_oof[b]) for _,b in folds])
    tvt["ENSEMBLE"]=dict(
        tr=(y[tr3],sum(W[k]*tvt[k]['tr'][1] for k in W),r2_score(y[tr3],sum(W[k]*tvt[k]['tr'][1] for k in W))),
        va=(y[va3],sum(W[k]*tvt[k]['va'][1] for k in W),r2_score(y[va3],sum(W[k]*tvt[k]['va'][1] for k in W))),
        te=(y[te3],sum(W[k]*tvt[k]['te'][1] for k in W),r2_score(y[te3],sum(W[k]*tvt[k]['te'][1] for k in W))))
    print(f"    ENSEMBLE     OOF R2={om['R2']:.4f}  RMSE={om['RMSE']:.4f}  "
          f"weights={ {k:round(W[k],2) for k in W if W[k]>0} }")
    # ---- SHAP ----
    print("  computing SHAP on LightGBM ...")
    lgbm=zoo()["LightGBM"]; lgbm.fit(X.iloc[tr0],y[tr0])
    samp=X.iloc[te0].sample(min(600,len(te0)),random_state=RANDOM)
    sv=shap.TreeExplainer(lgbm).shap_values(samp)
    shap_mean=pd.Series(np.abs(sv).mean(0),index=samp.columns).sort_values(ascending=False)
    return dict(name=name,unit=unit,X=X,y=y,tr=tr0,te=te0,feats=feats,
                fit=fit_res,cv=cv_r2,shap=sv,X_shap=samp,shap_mean=shap_mean,weights=W,
                tvt=tvt,tr3=tr3,va3=va3,te3=te3,pdp_model=lgbm,pdp_bg=X.iloc[tr0])

# ============================ FIGURES ========================================
def fig_distribution(R):
    name,unit,y,X=R["name"],R["unit"],R["y"],R["X"]
    fig=plt.figure(figsize=(15,4.5)); fig.suptitle(f"{name}: data distribution",fontweight="bold")
    ax=fig.add_subplot(1,3,1); ax.hist(y,bins=40,density=True,color="#69b3d6",edgecolor="k",alpha=.8)
    xs=np.linspace(y.min(),y.max(),200); ax.plot(xs,gaussian_kde(y)(xs),"r-",lw=2)
    ax.set_xlabel(f"{name} target{unit}"); ax.set_ylabel("Density"); ax.set_title("Target histogram + KDE")
    ax=fig.add_subplot(1,3,2); ax.boxplot(y,vert=True,widths=.5,patch_artist=True,
        boxprops=dict(facecolor="#69b3d6")); ax.set_ylabel(f"{name} target{unit}"); ax.set_title("Target boxplot"); ax.set_xticks([])
    ax=fig.add_subplot(1,3,3)
    top=R["shap_mean"].head(12).index.tolist()
    C=X[top].corr().values
    im=ax.imshow(C,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top,rotation=90,fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top,fontsize=7)
    ax.set_title("Feature correlation (top-12)"); fig.colorbar(im,ax=ax,fraction=0.046)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(f"{OUTDIR}/{name}_dist.png",bbox_inches="tight"); return fig

def fig_actual_pred(R):
    name,unit=R["name"],R["unit"]; ytr,yte=R["y"][R["tr"]],R["y"][R["te"]]
    order=[k for k in R["fit"] if k!="ENSEMBLE"][:6]+["ENSEMBLE"]
    fig,axs=plt.subplots(2,4,figsize=(18,8.5))
    fig.suptitle(f"{name}: Actual vs Predicted by model{unit}",fontweight="bold")
    lo=min(ytr.min(),yte.min()); hi=max(ytr.max(),yte.max())
    for ax,nm in zip(axs.ravel(),order+[None]*(8-len(order))):
        if nm is None: ax.axis("off"); continue
        fr=R["fit"][nm]
        ax.scatter(ytr,fr["ptr"],marker="D",s=16,facecolor="none",edgecolor="#7e57c2",alpha=.6,label="Training")
        ax.scatter(yte,fr["pte"],marker="^",s=22,color="#2ca02c",alpha=.7,label="Test")
        ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.3,label="y=x")
        if len(yte)>=2 and np.std(yte)>0:
            a,b=np.polyfit(yte,fr["pte"],1); xs=np.array([lo,hi])
            ax.plot(xs,a*xs+b,"r-",lw=1.5,label="Fit")
            ax.text(0.05,0.92,f"y={a:.3f}x+{b:.2f}\n$R^2$={fr['R2_te']:.3f}",transform=ax.transAxes,
                    va="top",fontsize=9,bbox=dict(fc="white",ec="grey",alpha=.7))
        ax.set_title(nm+(" (blend)" if nm=="ENSEMBLE" else "")); ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    axs.ravel()[0].legend(loc="lower right",fontsize=7)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_actual_pred.png",bbox_inches="tight"); return fig

def fig_train_val_test(R):
    """Paper Fig.7 style, but with TRAIN, VALIDATION and TEST shown separately:
       purple = train, orange = validation, green = test, dashed 1:1 line, red
       best-fit, and a per-subset R2 box. One panel per model."""
    name,unit=R["name"],R["unit"]
    order=[k for k in R["fit"] if k!="ENSEMBLE"][:6]+["ENSEMBLE"]
    allv=np.concatenate([R["y"][R["tr3"]],R["y"][R["va3"]],R["y"][R["te3"]]])
    lo,hi=allv.min(),allv.max()
    fig,axs=plt.subplots(2,4,figsize=(18,8.5))
    fig.suptitle(f"{name}: Actual vs Predicted — Train / Validation / Test (grouped, no leakage){unit}",
                 fontweight="bold")
    for ax,nm in zip(axs.ravel(),order+[None]*(8-len(order))):
        if nm is None or nm not in R["tvt"]: ax.axis("off"); continue
        d=R["tvt"][nm]
        for key,(ya,pa,r2) in [("Training",d["tr"]),("Validation",d["va"]),("Test",d["te"])]:
            mk,col=SUBSET_STYLE[key]
            ax.scatter(ya,pa,marker=mk,s=20,color=col,alpha=.65,edgecolor="k",lw=.2,
                       label=f"{key}  $R^2$={r2:.3f}")
        ax.plot([lo,hi],[lo,hi],"--",color="k",lw=1.3,label="y = x")
        # best-fit on all three combined
        ya=np.concatenate([d["tr"][0],d["va"][0],d["te"][0]])
        pa=np.concatenate([d["tr"][1],d["va"][1],d["te"][1]])
        if np.std(ya)>0:
            a,b=np.polyfit(ya,pa,1); xs=np.array([lo,hi])
            ax.plot(xs,a*xs+b,"r-",lw=1.4,label=f"fit y={a:.2f}x+{b:.2f}")
        ax.set_title(nm+(" (blend)" if nm=="ENSEMBLE" else "")); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}")
        ax.legend(fontsize=6.5,loc="upper left")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_train_val_test.png",bbox_inches="tight"); return fig

def fig_cv_box(R):
    name=R["name"]; order=sorted([k for k in R["cv"] if len(R["cv"][k])>0],
                                 key=lambda k:R["cv"][k].mean(),reverse=True)
    data=[R["cv"][k] for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showmeans=True,widths=.6,
                  meanprops=dict(marker="^",mfc="k",mec="k"),medianprops=dict(color="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE.get(k,("o","#888"))[1]); patch.set_alpha(.65)
    for i,k in enumerate(order):
        ax.annotate(f"{R['cv'][k].mean():.3f}",(i+1,R['cv'][k].mean()),
                    textcoords="offset points",xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right")
    ax.set_ylabel(f"$R^2$ ({CV_FOLDS}-fold grouped OOF, replicates kept)")
    ax.set_title(f"{name}: {CV_FOLDS}-fold LOPOCV-style performance")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_cvbox.png",bbox_inches="tight"); return fig

def fig_overlay(R):
    name,unit,yte=R["name"],R["unit"],R["y"][R["te"]]
    fig,ax=plt.subplots(figsize=(7.5,7.5))
    lo,hi=yte.min(),yte.max()
    for nm,fr in R["fit"].items():
        mk,col=STYLE.get(nm,("o","#888"))
        ax.scatter(yte,fr["pte"],marker=mk,s=30,color=col,alpha=.7,label=nm,edgecolor="k",lw=.3)
    ax.plot([lo,hi],[lo,hi],"r-",lw=1.6,label="y = x")
    ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}")
    ax.set_title(f"{name}: prediction results of different models")
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_overlay.png",bbox_inches="tight"); return fig

def fig_relerr(R):
    name,yte=R["name"],R["y"][R["te"]]
    order=sorted(R["fit"],key=lambda k:R["fit"][k]["R2_te"],reverse=True)
    yte_safe=np.where(np.abs(yte)<1e-6,1e-6,yte)
    data=[np.abs((R["fit"][k]["pte"]-yte)/yte_safe)*100 for k in order]
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.6,medianprops=dict(color="k"))
    for patch,k in zip(bp["boxes"],order): patch.set_facecolor(STYLE.get(k,("o","#888"))[1]); patch.set_alpha(.65)
    ax.axhline(5,ls="--",color="grey",lw=1.2,label="5%"); ax.axhline(10,ls=":",color="grey",lw=1.2,label="10%")
    for i,k in enumerate(order):
        ax.annotate(f"{np.median(data[i]):.1f}",(i+1,np.median(data[i])),
                    textcoords="offset points",xytext=(10,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"{name}: relative error (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_relerr.png",bbox_inches="tight"); return fig

def fig_shap(R):
    name=R["name"]; fig=plt.figure(figsize=(15,7))
    fig.suptitle(f"{name}: SHAP summary & importance (LightGBM, test set)",fontweight="bold")
    ax1=fig.add_subplot(1,2,1); plt.sca(ax1)
    shap.summary_plot(R["shap"],R["X_shap"],plot_type="dot",max_display=15,show=False,plot_size=None)
    ax1.set_title("SHAP beeswarm")
    ax2=fig.add_subplot(1,2,2,polar=True)
    top=R["shap_mean"].head(14)[::-1]; N=len(top)
    ang=np.linspace(0,2*np.pi,N,endpoint=False); width=2*np.pi/N*0.9
    ax2.bar(ang,top.values,width=width,color="#9b8cc4",edgecolor="k",alpha=.8)
    ax2.set_xticks(ang); ax2.set_xticklabels(top.index,fontsize=7)
    ax2.set_title("Mean |SHAP| (polar)")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_shap.png",bbox_inches="tight"); return fig

def fig_pdp(R):
    """Partial dependence for the top SHAP features (paper Fig.10 style)."""
    name=R["name"]; top=R["shap_mean"].head(6).index.tolist()
    fig,axs=plt.subplots(2,3,figsize=(16,8.5))
    fig.suptitle(f"{name}: Partial Dependence Plots (LightGBM, top-6 SHAP features)",fontweight="bold")
    try:
        PartialDependenceDisplay.from_estimator(
            R["pdp_model"],R["pdp_bg"],features=top,ax=axs.ravel()[:len(top)],
            kind="average",grid_resolution=40,line_kw={"color":"#1f77b4","lw":2})
    except Exception as e:
        # manual fallback if the sklearn version dislikes the ax array
        for ax,f in zip(axs.ravel(),top):
            xs=np.linspace(R["pdp_bg"][f].quantile(.02),R["pdp_bg"][f].quantile(.98),40)
            base=R["pdp_bg"].median(); preds=[]
            for v in xs:
                row=base.copy(); row[f]=v
                preds.append(R["pdp_model"].predict(pd.DataFrame([row]))[0])
            ax.plot(xs,preds,color="#1f77b4",lw=2); ax.set_xlabel(f); ax.set_ylabel("partial dependence")
    for ax in axs.ravel()[len(top):]: ax.axis("off")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUTDIR}/{name}_pdp.png",bbox_inches="tight"); return fig

def fig_spearman(R):
    name=R["name"]; models=list(R["fit"])
    xv=np.array([R["cv"][m].mean() if len(R["cv"][m])>0 else R["fit"][m]["R2_oof"] for m in models])
    yv=np.array([R["fit"][m]["R2_te"] for m in models])
    rho=spearmanr(xv,yv).correlation
    fig,ax=plt.subplots(figsize=(7,6.5))
    for m in models:
        mk,col=STYLE.get(m,("o","#888"))
        ax.scatter(xv[models.index(m)],R["fit"][m]["R2_te"],marker=mk,s=90,color=col,edgecolor="k",label=m)
    lo=min(xv.min(),yv.min())-0.05; hi=max(xv.max(),yv.max())+0.05
    ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.2)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel(f"{CV_FOLDS}-fold CV mean $R^2$"); ax.set_ylabel("Locked-test $R^2$")
    ax.set_title(f"{name}: model ranking consistency\nSpearman $\\rho$ = {rho:.3f}")
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/{name}_spearman.png",bbox_inches="tight"); return fig

# ============================ MAIN ===========================================
if __name__=="__main__":
    print("[v6] audit + PG-safe parsing + train/val/test + PDP.")
    path=resolve_file(FILE)
    Rs=[]
    Rs.append(compute(path,"RUT","Rutting_Filtered","LWT_Design_Result"," (mm)"))
    Rs.append(compute(path,"SCB","SCB_Filtered","SCB_Result",""))
    for R in Rs:
        fig_distribution(R); fig_actual_pred(R); fig_train_val_test(R); fig_cv_box(R)
        fig_overlay(R); fig_relerr(R); fig_shap(R); fig_pdp(R); fig_spearman(R)
    if not HEADLESS: plt.show()
    print("\n"+"="*72+"\n  FINAL SUMMARY  (grouped 10-fold OOF, replicates KEPT, PG-safe)\n"+"="*72)
    for R in Rs:
        best=max(R["fit"],key=lambda k:R["fit"][k]["R2_oof"])
        f=R["fit"][best]
        print(f"  {R['name']}   best={best}   OOF R2={f['R2_oof']:.4f}   RMSE={f['RMSE_oof']:.4f}")
        if best=="ENSEMBLE":
            w=R["weights"]; print(f"       weights: { {k:round(w[k],2) for k in w if w[k]>0} }")
    print("\nDONE - 9 figures per target + MODEL_INPUT_AUDIT.xlsx saved.")
