# -*- coding: utf-8 -*-
"""
ASPHALT RUT (LWT) + SCB — FULL GROUPED-CV WORKFLOW WITH ALL DIAGNOSTIC PLOTS
================================================================================
Leakage-safe, replicates KEPT. StratifiedGroupKFold(10) grouped by Mix_ID so a
mix's replicate rows never span folds. One script, both targets, every figure we
built:

    1. Data distribution        (histogram+KDE, boxplot, correlation heatmap)
    2. Actual vs Predicted      (per model: TRAIN vs TEST)
    3. Train / Validation / Test (grouped 3-way split, per model, paper Fig.7)
    4. 10-fold CV box plot      (R2 per model)
    5. Overlay                  (all models, OOF actual-vs-pred)
    6. Relative error           (per model, test)
    7. SHAP                     (beeswarm + polar mean|SHAP|)
    8. PDP                      (partial dependence, top-6 features)
    9. Ranking consistency      (CV-mean vs held-out, Spearman rho)
   10. Grouped 10-fold OOF      (ensemble measured-vs-predicted)

HOW TO RUN IN SPYDER
   1. Put mixture_dataset_lwt.csv and mixture_dataset_scb.csv next to this file
      (or in Downloads), OR edit FILE_LWT / FILE_SCB below.
   2. Press F5. Figures are shown and saved to ./asphalt_full_outputs/ at 300 dpi.

Requirements: pandas numpy scikit-learn lightgbm xgboost shap scipy matplotlib
              (catboost optional). CatBoost/XGBoost/SHAP degrade gracefully.
"""
import os, re, glob, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr, gaussian_kde
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
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
FILE_LWT = "mixture_dataset_lwt.csv"
FILE_SCB = "mixture_dataset_scb.csv"
FOLDS    = 10
RANDOM   = 42
LOG_LWT  = False        # model log1p(LWT)? marginal gain; keep False for real-unit SHAP/PDP
HEADLESS = False        # True = save only, don't show
OUTDIR   = "asphalt_full_outputs"
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
    return name  # let pandas raise a clear error

# =========================== FEATURES =======================================
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
    devs=[P[s]-(100*(SIEVE_MM[s]/nmas)**0.45).clip(upper=100) for s in order]
    if devs:
        D=pd.concat(devs,axis=1); G["MDL_meanabs"]=D.abs().mean(axis=1); G["MDL_area"]=D.sum(axis=1)
    return G

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
        ph=num(df["PG_High_Temp_C"]); X["PG_High"]=ph
        if "PG_Low_Temp_C" in df: X["PG_span"]=ph-num(df["PG_Low_Temp_C"])
        if "%Voids" in df: X["PG_x_Voids"]=ph*num(df["%Voids"])
        if "RBR" in df: X["PG_x_RBR"]=ph*num(df["RBR"])
        if "AFT" in df: X["PG_x_AFT"]=ph*num(df["AFT"])
        if "Pbe" in df: X["PG_x_Pbe"]=ph*num(df["Pbe"])
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

# ============================ MODELS ========================================
def zoo():
    m={"ExtraTrees":ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,max_features=0.6,
          n_jobs=-1,random_state=RANDOM),
       "RandomForest":RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.7,
          n_jobs=-1,random_state=RANDOM),
       "LightGBM":lgb.LGBMRegressor(n_estimators=700,learning_rate=0.03,num_leaves=63,
          min_child_samples=15,subsample=0.85,colsample_bytree=0.7,reg_lambda=1.0,
          random_state=RANDOM,n_jobs=-1,verbose=-1),
       "HistGB":HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=15,l2_regularization=0.1,random_state=RANDOM)}
    if HAS_XGB: m["XGBoost"]=xgb.XGBRegressor(n_estimators=700,learning_rate=0.03,max_depth=6,
          subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,min_child_weight=3,
          random_state=RANDOM,n_jobs=-1,verbosity=0)
    if HAS_CAT: m["CatBoost"]=CatBoostRegressor(iterations=600,learning_rate=0.03,depth=6,
          l2_leaf_reg=3.0,random_state=RANDOM,verbose=0)
    return m
STYLE={"ExtraTrees":("s","#1f77b4"),"RandomForest":("^","#2ca02c"),"LightGBM":("*","#17becf"),
       "XGBoost":("o","#9467bd"),"CatBoost":("h","#bcbd22"),"HistGB":("<","#ff7f0e"),
       "ENSEMBLE":("D","#d62728")}
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
    d_tr,d_va=next(iter(StratifiedGroupKFold(k2,shuffle=True,random_state=seed).split(
        X.iloc[dev],yb[dev],groups=grp[dev])))
    return np.sort(dev[d_tr]),np.sort(dev[d_va]),np.sort(te)

# ============================ COMPUTE =======================================
def compute(path,name,target,unit):
    print("\n"+"="*74+f"\n  {name}  — grouped {FOLDS}-fold CV (replicates kept)\n"+"="*74)
    df=pd.read_csv(resolve(path))
    y=num(df[target]).values; keep=np.isfinite(y)
    df=df[keep].reset_index(drop=True); y=y[keep]
    grp=df["Mix_ID"].astype(str).values
    X=build_features(df,target)
    log=(name=="LWT" and LOG_LWT); inv=(np.expm1 if log else (lambda p:p)); yfit=np.log1p(y) if log else y
    print(f"  rows={len(X)}  unique mixes={len(np.unique(grp))}  features={X.shape[1]}  "
          f"target=[{y.min():.3f},{y.max():.3f}]")
    yb=pd.qcut(pd.Series(y).rank(method="first"),10,labels=False,duplicates="drop")
    folds=list(StratifiedGroupKFold(FOLDS,shuffle=True,random_state=RANDOM).split(X,yb,groups=grp))
    tr3,va3,te3=grouped_three_way(X,y,grp)

    fit={}; cv={}; oof_all={}; tvt={}
    for nm in zoo():
        oof=np.zeros(len(y)); fr=[]; trs=[]; trp={}
        for fi,(a,b) in enumerate(folds):
            mm=zoo()[nm]; mm.fit(X.iloc[a],yfit[a])
            oof[b]=inv(mm.predict(X.iloc[b])); trp[fi]=inv(mm.predict(X.iloc[a]))
            fr.append(r2_score(y[b],oof[b])); trs.append(r2_score(y[a],trp[fi]))
        om=met(y,oof); oof_all[nm]=oof; cv[nm]=np.array(fr)
        # 3-way split fit for scatter/SHAP/PDP
        m3=zoo()[nm]; m3.fit(X.iloc[tr3],yfit[tr3])
        ptr,pva,pte=inv(m3.predict(X.iloc[tr3])),inv(m3.predict(X.iloc[va3])),inv(m3.predict(X.iloc[te3]))
        fit[nm]=dict(oof=oof,R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],
                     R2_train=np.mean(trs),fold_r2=np.array(fr),
                     ptr=ptr,pva=pva,pte=pte,R2_te=r2_score(y[te3],pte),
                     R2_va=r2_score(y[va3],pva),R2_tr3=r2_score(y[tr3],ptr))
        tvt[nm]=dict(tr=(y[tr3],ptr,fit[nm]["R2_tr3"]),va=(y[va3],pva,fit[nm]["R2_va"]),
                     te=(y[te3],pte,fit[nm]["R2_te"]))
        print(f"    {nm:12s} train R2={np.mean(trs):.3f}  OOF(test) R2={om['R2']:.3f}  "
              f"RMSE={om['RMSE']:.3f}  fold={np.mean(fr):.3f}±{np.std(fr):.3f}")
    W=opt_ens(oof_all,y); ens=sum(W[k]*oof_all[k] for k in W)
    om=met(y,ens); cv["ENSEMBLE"]=np.array([r2_score(y[b],ens[b]) for _,b in folds])
    ens_tr3=sum(W[k]*fit[k]["ptr"] for k in W); ens_va=sum(W[k]*fit[k]["pva"] for k in W); ens_te=sum(W[k]*fit[k]["pte"] for k in W)
    fit["ENSEMBLE"]=dict(oof=ens,R2_oof=om["R2"],RMSE_oof=om["RMSE"],MAE_oof=om["MAE"],
        R2_train=np.nan,fold_r2=cv["ENSEMBLE"],ptr=ens_tr3,pva=ens_va,pte=ens_te,
        R2_te=r2_score(y[te3],ens_te),R2_va=r2_score(y[va3],ens_va),R2_tr3=r2_score(y[tr3],ens_tr3))
    tvt["ENSEMBLE"]=dict(tr=(y[tr3],ens_tr3,fit["ENSEMBLE"]["R2_tr3"]),
                         va=(y[va3],ens_va,fit["ENSEMBLE"]["R2_va"]),te=(y[te3],ens_te,fit["ENSEMBLE"]["R2_te"]))
    print(f"    ENSEMBLE     OOF(test) R2={om['R2']:.3f}  RMSE={om['RMSE']:.3f}  "
          f"weights={ {k:round(W[k],2) for k in W if W[k]>0} }")
    # SHAP + PDP model (fit on 3-way train, raw scale for interpretability)
    shap_mean=X_shap=sv=None; pdp_model=zoo()["LightGBM"]; pdp_model.fit(X.iloc[tr3],y[tr3])
    if HAS_SHAP:
        X_shap=X.iloc[te3] if len(te3)<=800 else X.iloc[te3].sample(800,random_state=RANDOM)
        try:
            sv=shap.TreeExplainer(pdp_model).shap_values(X_shap)
            shap_mean=pd.Series(np.abs(sv).mean(0),index=X_shap.columns).sort_values(ascending=False)
        except Exception as e: print("   [shap skipped]",e)
    return dict(name=name,unit=unit,X=X,y=y,grp=grp,folds=folds,fit=fit,cv=cv,weights=W,
                tvt=tvt,tr3=tr3,va3=va3,te3=te3,shap=sv,X_shap=X_shap,shap_mean=shap_mean,
                pdp_model=pdp_model,pdp_bg=X.iloc[tr3])

# ============================ FIGURES =======================================
def _save(fig,name,tag):
    fig.savefig(f"{OUTDIR}/{name}_{tag}.png",bbox_inches="tight")
    if HEADLESS: plt.close(fig)

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
    n=len(order); fig,axs=plt.subplots(2,4,figsize=(18,8.5))
    fig.suptitle(f"{name}: Actual vs Predicted — Train vs Test{unit}",fontweight="bold")
    for ax,nm in zip(axs.ravel(),order+[None]*(8-n)):
        if nm is None: ax.axis("off"); continue
        fr=R["fit"][nm]
        ax.scatter(ytr,fr["ptr"],marker="D",s=15,facecolor="none",edgecolor="#7e57c2",alpha=.5,label=f"Train R²={fr['R2_tr3']:.2f}")
        ax.scatter(yte,fr["pte"],marker="^",s=20,color="#2ca02c",alpha=.7,label=f"Test R²={fr['R2_te']:.2f}")
        ax.plot([lo,hi],[lo,hi],"--",color="grey",lw=1.2)
        ax.set_title(nm); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}"); ax.legend(fontsize=7,loc="upper left")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"actual_pred")

def fig_train_val_test(R):
    name,unit=R["name"],R["unit"]; order=[k for k in R["fit"] if k!="ENSEMBLE"]+["ENSEMBLE"]
    allv=np.concatenate([R["y"][R["tr3"]],R["y"][R["va3"]],R["y"][R["te3"]]]); lo,hi=allv.min(),allv.max()
    fig,axs=plt.subplots(2,4,figsize=(18,8.5))
    fig.suptitle(f"{name}: Train / Validation / Test (grouped, no leakage){unit}",fontweight="bold")
    for ax,nm in zip(axs.ravel(),order+[None]*(8-len(order))):
        if nm is None: ax.axis("off"); continue
        d=R["tvt"][nm]
        for key,(ya,pa,r2) in [("Training",d["tr"]),("Validation",d["va"]),("Test",d["te"])]:
            mk,col=SUBSET[key]; ax.scatter(ya,pa,marker=mk,s=18,color=col,alpha=.6,edgecolor="k",lw=.2,
                                           label=f"{key} $R^2$={r2:.3f}")
        ax.plot([lo,hi],[lo,hi],"--",color="k",lw=1.2)
        ax.set_title(nm); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted{unit}"); ax.legend(fontsize=6.5,loc="upper left")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"train_val_test")

def fig_cv_box(R):
    name=R["name"]; order=sorted(R["cv"],key=lambda k:R["cv"][k].mean(),reverse=True)
    fig,ax=plt.subplots(figsize=(12,5.5))
    bp=ax.boxplot([R["cv"][k] for k in order],patch_artist=True,showmeans=True,widths=.6,
                  meanprops=dict(marker="^",mfc="k",mec="k"),medianprops=dict(color="k"))
    for pt,k in zip(bp["boxes"],order): pt.set_facecolor(STYLE.get(k,("o","#888"))[1]); pt.set_alpha(.65)
    for i,k in enumerate(order): ax.annotate(f"{R['cv'][k].mean():.3f}",(i+1,R['cv'][k].mean()),
                    textcoords="offset points",xytext=(8,0),fontsize=8)
    ax.set_xticklabels(order,rotation=30,ha="right"); ax.set_ylabel(f"$R^2$ ({FOLDS}-fold grouped OOF)")
    ax.set_title(f"{name}: {FOLDS}-fold grouped CV performance"); fig.tight_layout(); _save(fig,name,"cvbox")

def fig_overlay(R):
    name,unit,y=R["name"],R["unit"],R["y"]; fig,ax=plt.subplots(figsize=(7.5,7.5)); lo,hi=y.min(),y.max()
    for nm,fr in R["fit"].items():
        mk,col=STYLE.get(nm,("o","#888")); ax.scatter(y,fr["oof"],marker=mk,s=14,color=col,alpha=.45,label=nm,edgecolor="k",lw=.2)
    ax.plot([lo,hi],[lo,hi],"r-",lw=1.6,label="y=x")
    ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted (OOF){unit}")
    ax.set_title(f"{name}: all models (grouped OOF)"); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); _save(fig,name,"overlay")

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
    name=R["name"]; fig=plt.figure(figsize=(15,7))
    fig.suptitle(f"{name}: SHAP summary & importance (LightGBM)",fontweight="bold")
    ax1=fig.add_subplot(1,2,1); plt.sca(ax1)
    shap.summary_plot(R["shap"],R["X_shap"],plot_type="dot",max_display=15,show=False,plot_size=None)
    ax1.set_title("SHAP beeswarm")
    ax2=fig.add_subplot(1,2,2,polar=True); top=R["shap_mean"].head(14)[::-1]; N=len(top)
    ang=np.linspace(0,2*np.pi,N,endpoint=False)
    ax2.bar(ang,top.values,width=2*np.pi/N*0.9,color="#9b8cc4",edgecolor="k",alpha=.8)
    ax2.set_xticks(ang); ax2.set_xticklabels(top.index,fontsize=7); ax2.set_title("Mean |SHAP| (polar)")
    fig.tight_layout(rect=[0,0,1,0.95]); _save(fig,name,"shap")

def fig_pdp(R):
    name=R["name"]
    top=(R["shap_mean"].head(6).index.tolist() if R["shap_mean"] is not None else list(R["X"].columns[:6]))
    fig,axs=plt.subplots(2,3,figsize=(16,8.5))
    fig.suptitle(f"{name}: Partial Dependence (LightGBM, top-6)",fontweight="bold")
    try:
        PartialDependenceDisplay.from_estimator(R["pdp_model"],R["pdp_bg"],features=top,
            ax=axs.ravel()[:len(top)],kind="average",grid_resolution=40,line_kw={"color":"#1f77b4","lw":2})
    except Exception as e:
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
    ax.set_title(f"{name}: ranking consistency  Spearman $\\rho$={rho:.3f}"); ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); _save(fig,name,"spearman")

def fig_oof(R):
    name,unit,y=R["name"],R["unit"],R["y"]; ens=R["fit"]["ENSEMBLE"]
    fig,ax=plt.subplots(figsize=(6.8,6.8)); lo,hi=y.min(),y.max()
    ax.scatter(y,ens["oof"],s=14,c="#2E6DA4",alpha=.5,edgecolor="k",lw=.2)
    ax.plot([lo,hi],[lo,hi],"r--",lw=1.4,label="y=x")
    a,b=np.polyfit(y,ens["oof"],1); xs=np.array([lo,hi]); ax.plot(xs,a*xs+b,"k-",lw=1.2,
        label=f"fit  R²={ens['R2_oof']:.3f}\nRMSE={ens['RMSE_oof']:.3f}")
    ax.set_xlabel(f"Actual{unit}"); ax.set_ylabel(f"Predicted (OOF){unit}")
    ax.set_title(f"{name}: grouped {FOLDS}-fold OOF (ENSEMBLE)"); ax.legend(); fig.tight_layout(); _save(fig,name,"oof")

# ============================ MAIN ==========================================
if __name__=="__main__":
    print(f"[full] XGBoost={HAS_XGB} CatBoost={HAS_CAT} SHAP={HAS_SHAP}  folds={FOLDS}")
    Rs=[compute(FILE_LWT,"LWT","LWT"," (mm)"), compute(FILE_SCB,"SCB","SCB","")]
    for R in Rs:
        fig_distribution(R); fig_actual_pred(R); fig_train_val_test(R); fig_cv_box(R)
        fig_overlay(R); fig_relerr(R); fig_shap(R); fig_pdp(R); fig_spearman(R); fig_oof(R)
    if not HEADLESS: plt.show()
    print("\n"+"="*74+"\n  SUMMARY (grouped 10-fold OOF, replicates kept, no leakage)\n"+"="*74)
    for R in Rs:
        e=R["fit"]["ENSEMBLE"]
        print(f"  {R['name']:5s} ENSEMBLE  train R2={max(R['fit'][k]['R2_train'] for k in R['fit'] if k!='ENSEMBLE'):.3f}"
              f"  test(OOF) R2={e['R2_oof']:.3f}  RMSE={e['RMSE_oof']:.3f}  fold={e['fold_r2'].mean():.3f}±{e['fold_r2'].std():.3f}")
    print(f"\nAll figures saved in ./{OUTDIR}/ (10 per target).")
