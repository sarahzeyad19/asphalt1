# -*- coding: utf-8 -*-
"""
FEATURE-GROUP STUDY — which engineered features help RUTTING vs CRACKING (separately)
Grouped 10-fold CV (by Mix_ID, replicates kept). For each target, measures the
contribution of each engineered feature GROUP by ablation (full R2 minus R2
without the group) + SHAP importance. Tells us which features to keep per target.
"""
import re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb
try: import shap; HAS_SHAP=True
except Exception: HAS_SHAP=False
RANDOM=42; FOLDS=10
XL="/root/.claude/uploads/326cb6b5-2bf6-5192-84c2-c14c3d42c3df/5f3188c0-mixture_dataset_clean_lwtscb.xlsx"
num=lambda s: pd.to_numeric(s,errors="coerce")
SIEVE_MM={'Pass 1 1/2"':37.5,'Pass 1"':25,'Pass 3/4"':19,'Pass 1/2"':12.5,'Pass 3/8"':9.5,
          'Pass No.4':4.75,'Pass No.8':2.36,'Pass No.16':1.18,'Pass No.30':0.60,
          'Pass No.50':0.30,'Pass No.100':0.15,'Pass No.200':0.075}
RANGE={"LWT":(0.5,11.0),"SCB":(0.30,1.25)}

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

CAT=["Design_Level","Binder_Modification"]
IDS=["Project_ID","Mix_ID"]
def build(df,target):
    X=pd.DataFrame(index=df.index); skip=set(IDS+CAT+["LWT","SCB"]+list(SIEVE_MM))
    for c in df.columns:
        if c in skip: continue
        v=num(df[c])
        if v.notna().sum()>0: X[c]=v
    for s in SIEVE_MM:
        if s in df.columns: X[s]=num(df[s])
    grad=gradation(df); X=pd.concat([X,grad],axis=1)
    groups={"GRAD_SHAPE":list(grad.columns)}
    ph=num(df["PG_High_Temp_C"])
    pgi={}
    pgi["PG_span"]=ph-num(df["PG_Low_Temp_C"]); pgi["PG_x_Voids"]=ph*num(df["%Voids"])
    pgi["PG_x_RBR"]=ph*num(df["RBR"]); pgi["PG_x_AFT"]=ph*num(df["AFT"]); pgi["PG_x_Pbe"]=ph*num(df["Pbe"])
    for k,v in pgi.items(): X[k]=v
    groups["PG_INTERACT"]=list(pgi.keys())
    # additive flags already in file
    add=[c for c in ["AntiStrip_add","WMA_add","Latex_add","Fiber_add"] if c in df.columns]
    for c in add: X[c]=num(df[c])
    if "Binder_Modification" in df: X["Polymer"]=df["Binder_Modification"].astype(str).str.upper().str.contains("MODIF").astype(float)
    groups["ADDITIVES"]=add+["Polymer"]
    groups["RAP_FAMILY"]=[c for c in ["RAP","%RAP","Total_AC_from_RAP","RBR","AFT"] if c in X.columns]
    for c in CAT:
        if c in df: X=pd.concat([X,pd.get_dummies(df[c].astype(str),prefix=c).astype(float).set_index(X.index)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X=X.loc[:,X.nunique()>1]
    X.columns=[re.sub(r"[^0-9A-Za-z_]+","_",str(c)).strip("_") for c in X.columns]
    X=X.loc[:,~X.columns.duplicated()]
    groups={g:[re.sub(r"[^0-9A-Za-z_]+","_",str(c)).strip("_") for c in cols] for g,cols in groups.items()}
    groups={g:[c for c in cols if c in X.columns] for g,cols in groups.items()}
    return X,groups

def oof_r2(X,y,grp,log):
    yb=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    folds=StratifiedGroupKFold(FOLDS,shuffle=True,random_state=RANDOM).split(X,yb,groups=grp)
    inv=(np.expm1 if log else (lambda p:p)); yfit=np.log1p(y) if log else y
    oof=np.zeros(len(y))
    for a,b in folds:
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=63,min_child_samples=15,
            subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,random_state=RANDOM,n_jobs=-1,verbose=-1)
        m.fit(X.iloc[a],yfit[a]); oof[b]=inv(m.predict(X.iloc[b]))
    return r2_score(y,oof)

def study(target):
    df=pd.read_excel(XL,sheet_name="in")
    y=num(df[target]).values; lo,hi=RANGE[target]
    keep=(y>=lo)&(y<=hi); df=df[keep].reset_index(drop=True); y=y[keep]
    grp=df["Mix_ID"].astype(str).values
    X,groups=build(df,target); log=(target=="LWT")
    print("\n"+"="*70+f"\n  {target}  ({'rutting' if target=='LWT' else 'cracking'})  "
          f"rows={len(X)} mixes={len(np.unique(grp))} feats={X.shape[1]} range=[{lo},{hi}]\n"+"="*70)
    full=oof_r2(X,y,grp,log); print(f"  FULL set OOF R2 = {full:.3f}")
    print("  group ablation (contribution = FULL - without_group; + = helps this target):")
    contrib={}
    for g,cols in groups.items():
        cols=[c for c in cols if c in X.columns]
        if not cols: continue
        r=oof_r2(X.drop(columns=cols),y,grp,log); contrib[g]=full-r
        print(f"    {g:12s} ({len(cols):2d} feats)  without={r:.3f}   contribution={full-r:+.3f}")
    if HAS_SHAP:
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=63,min_child_samples=15,
            subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,random_state=RANDOM,n_jobs=-1,verbose=-1)
        m.fit(X,np.log1p(y) if log else y)
        sv=shap.TreeExplainer(m).shap_values(X)
        imp=pd.Series(np.abs(sv).mean(0),index=X.columns).sort_values(ascending=False)
        print("  top-12 SHAP importance:", ", ".join(f"{k}({imp[k]:.3f})" for k in imp.head(12).index))
    return target,full,contrib

if __name__=="__main__":
    print("[feature study] grouped 10-fold, replicates kept")
    res=[study("LWT"),study("SCB")]
    print("\n"+"="*70+"\n  SUMMARY — group contribution per target (+ helps, - hurts)\n"+"="*70)
    allg=sorted({g for _,_,c in res for g in c})
    print(f"  {'group':13s} " + "  ".join(f"{t:>8s}" for t,_,_ in res))
    for g in allg:
        print(f"  {g:13s} " + "  ".join(f"{c.get(g,0):+8.3f}" for _,_,c in res))
