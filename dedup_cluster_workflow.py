# -*- coding: utf-8 -*-
"""
RUTTING & SCB — DUPLICATE-AWARE, ENGINEERING-CLUSTER GROUPED WORKFLOW
====================================================================
Implements the reviewed duplicate-management + sampling plan (DagsHub / PDDM-AL / CLRN):

  1. Keep every LEGITIMATE repeated report; remove ONLY exact extraction copies
     (byte-identical feature+target fingerprint).
  2. Build grouping levels:
       - Exact_Mix   = MixDesignKey          (operational: unseen exact version)
       - Eng_Cluster = DBSCAN on standardized engineering features, blocked by
                       MixType + NMAS class  (strict: unseen engineering-similar mixture)
  3. 70/10/20 split GROUPED so an entire mix/cluster stays in ONE subset (no leakage).
  4. Inverse Exact-Mix report weights: every mix gets ~equal total influence.
  5. Target-specific feature sets, then CORRELATION PRUNING drops same-effect twins
     (|Spearman| >= 0.90, keep the one more correlated with the target).
  6. Superior models compared: XGBoost, CatBoost, LightGBM, ExtraTrees, HistGB + Ridge stack.
  7. Report BOTH grouping levels for each target (operational vs strict).

Data: Design_Validation_Separated_Complete_Cleaned.xlsx (Design_Complete + Validation_Complete;
both carry Rut_20k and SCB; has PG_HighTemp — the dominant rutting feature).
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
    from lightgbm import LGBMRegressor; HAS_LGBM = True
except Exception: HAS_LGBM = False
try:
    from xgboost import XGBRegressor; HAS_XGB = True
except Exception: HAS_XGB = False
try:
    from catboost import CatBoostRegressor; HAS_CAT = True
except Exception: HAS_CAT = False

RANDOM_STATE = 42
TEST_SIZE, VAL_SIZE = 0.20, 0.10
DBSCAN_EPS = 1.6            # cluster radius in standardized-feature space (tune 1.2-2.0)
DBSCAN_MIN = 2

HOME = Path.home()
DOWNLOADS = Path(r"C:\Users\lenovo\Downloads")
if not DOWNLOADS.exists():
    DOWNLOADS = HOME / "Downloads" if (HOME / "Downloads").exists() else Path.cwd()
DATA = "Design_Validation_Separated_Complete_Cleaned.xlsx"

# ---- Target-specific candidate features (engineering-mechanism driven) ----
RUT_FEATURES = ["PG_HighTemp","Va","VMA","VFA","Pbe_pct","CAA","FAA","SandEq","Absorption",
                "RAP_pct","ACinRAP","AsphaltContent_Design","Gmm","Gse","Gsb","NMAS (mm)",
                "Grad_No4","Grad_No8","Grad_No30","Grad_No50","Grad_No100","Grad_No200","Dust_Binder","MixType","DesignLev"]
SCB_FEATURES = ["Pbe_pct","Va","VMA","VFA","Absorption","RAP_pct","ACinRAP","AsphaltContent_Design",
                "SandEq","FAA","CAA","Gse","Gmm","Dust_Binder","Grad_No30","Grad_No50","Grad_No100","Grad_No200","MixType","DesignLev"]
CLUSTER_FEATS = ["AsphaltContent_Design","RAP_pct","Va","VMA","VFA","Gmm","CAA","FAA","Pbe_pct","Grad_No4","Grad_No200","PG_HighTemp"]
CATEG = ["MixType","DesignLev"]

def M(a,b): return dict(R2=r2_score(a,b), RMSE=float(np.sqrt(mean_squared_error(a,b))), MAE=mean_absolute_error(a,b))
def nz(d,c): return pd.to_numeric(d[c],errors="coerce") if c in d.columns else pd.Series(np.nan,index=d.index)

def load():
    path = DOWNLOADS / DATA
    if not path.exists():
        alt = Path.cwd() / DATA
        if alt.exists(): path = alt
    frames = []
    for s in ["Design_Complete","Validation_Complete"]:
        p = pd.read_excel(path, sheet_name=s); p["Stage"] = s; frames.append(p)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df.columns = [str(c).strip() for c in df.columns]
    # ---- exact-copy removal (byte-identical numeric fingerprint incl. both targets) ----
    numcols = [c for c in df.select_dtypes("number").columns]
    sig = pd.Series(["|".join(map(str,r)) for r in df[numcols].round(4).fillna(-9e9).values])
    before = len(df); df = df.loc[~sig.duplicated()].reset_index(drop=True)
    print(f"Exact-copy removal: {before} -> {len(df)} rows (removed {before-len(df)} byte-identical copies)")
    return df

def engineer(df):
    df = df.copy()
    df["Fines_to_Pbe"] = nz(df,"Grad_No200")/nz(df,"Pbe_pct").replace(0,np.nan)
    df["VMA_Filled_Index"] = nz(df,"VMA")*nz(df,"VFA")/100.0
    df["Coarse_Fraction"] = 100-nz(df,"Grad_No4")
    df["Intermediate_Fraction"] = nz(df,"Grad_No4")-nz(df,"Grad_No8")
    df["RAP_Binder_Load"] = nz(df,"RAP_pct")*nz(df,"ACinRAP")/100.0
    df["PG_x_RAPload"] = nz(df,"PG_HighTemp")*df["RAP_Binder_Load"]
    return df

def make_clusters(df):
    """DBSCAN engineering clusters within MixType + NMAS blocks (standardized features)."""
    # init every row to its own singleton cluster (string), so any row not touched by DBSCAN
    # still gets a valid, unique group id.
    cid = np.array([f"Cinit_{i}" for i in range(len(df))], dtype=object); nxt = 0
    feats = [c for c in CLUSTER_FEATS if c in df.columns]
    nmas_bin = pd.cut(nz(df,"NMAS (mm)"), bins=[0,10,13,20,100], labels=["9.5","12.5","19","25+"]).astype(str)
    blk = df.get("MixType","NA").astype(str) + "|" + nmas_bin
    for b, idx in df.groupby(blk).groups.items():
        idx = np.array(list(idx)); sub = df.loc[idx, feats].apply(pd.to_numeric, errors="coerce")
        sub = sub.fillna(sub.median())
        if len(idx) < DBSCAN_MIN or sub.std().sum() == 0:
            for i in idx: cid[df.index.get_loc(i)] = f"C{nxt}"; nxt += 1
            continue
        Z = StandardScaler().fit_transform(sub.values)
        lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(Z)
        for i, l in zip(idx, lab):
            pos = df.index.get_loc(i)
            cid[pos] = f"C{nxt+l}" if l >= 0 else f"C{nxt+1000+i}"   # noise -> singleton
        nxt += (lab.max()+1 if lab.max() >= 0 else 0) + 1
    return pd.Series(cid, index=df.index).astype(str)

def prune(X, y):
    """Drop same-effect features: |Spearman| >= 0.90, keep the one more correlated with target."""
    num = X.select_dtypes("number")
    corr = num.corr("spearman").abs(); tcorr = num.apply(lambda s: abs(s.corr(y,"spearman")))
    kept, dropped = [], []
    for f in tcorr.sort_values(ascending=False).index:
        twin = next((k for k in kept if corr.loc[f,k] >= 0.90), None)
        if twin: dropped.append((f, twin, round(corr.loc[f,twin],2)))
        else: kept.append(f)
    keep_cols = kept + [c for c in X.columns if c in CATEG]
    return keep_cols, dropped

def models():
    m = {}
    if HAS_XGB: m["XGBoost"] = XGBRegressor(objective="reg:squarederror",tree_method="hist",n_estimators=700,
        learning_rate=0.02,max_depth=3,min_child_weight=15,subsample=0.8,colsample_bytree=0.8,reg_lambda=20,random_state=RANDOM_STATE,n_jobs=-1)
    if HAS_CAT: m["CatBoost"] = CatBoostRegressor(iterations=700,learning_rate=0.03,depth=5,l2_leaf_reg=10,verbose=0,random_seed=RANDOM_STATE)
    if HAS_LGBM: m["LightGBM"] = LGBMRegressor(n_estimators=700,learning_rate=0.02,num_leaves=31,min_child_samples=30,
        subsample=0.85,colsample_bytree=0.8,reg_lambda=10,random_state=RANDOM_STATE,verbose=-1)
    m["ExtraTrees"] = ExtraTreesRegressor(n_estimators=600,min_samples_leaf=3,max_features=0.8,random_state=RANDOM_STATE,n_jobs=-1)
    m["HistGB"] = HistGradientBoostingRegressor(learning_rate=0.05,max_iter=500,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.0,random_state=RANDOM_STATE)
    return m

def pipe(est, num, cat):
    t = [("num", SimpleImputer(strategy="median"), num)]
    if cat:
        t.append(("cat", Pipeline([("i",SimpleImputer(strategy="constant",fill_value="NA")),
                                   ("o",__import__("sklearn.preprocessing",fromlist=["OneHotEncoder"]).OneHotEncoder(handle_unknown="ignore",sparse_output=False))]), cat))
    return Pipeline([("prep", ColumnTransformer(t)), ("model", est)])

def gsplit(y, groups):
    groups = np.asarray([str(g) for g in groups])   # uniform string groups (avoid mixed-type errors)
    b = pd.qcut(y, 5, labels=False, duplicates="drop")
    dev, te = next(iter(StratifiedGroupKFold(max(2,round(1/TEST_SIZE)),shuffle=True,random_state=RANDOM_STATE).split(np.zeros(len(y)), b, groups)))
    vb, vg = b.iloc[dev].values, groups[dev]
    tr, va = next(iter(StratifiedGroupKFold(max(2,round((1-TEST_SIZE)/VAL_SIZE)),shuffle=True,random_state=RANDOM_STATE).split(np.zeros(len(dev)), vb, vg)))
    return dev[tr], dev[va], te

def run_target(df, target, feats):
    d = df[pd.to_numeric(df[target],errors="coerce").notna()].copy()
    if target == "SCB": d = d[pd.to_numeric(d[target],errors="coerce")>0].copy()
    d = d.reset_index(drop=True)
    y = pd.to_numeric(d[target], errors="coerce").reset_index(drop=True)
    cols = [c for c in feats if c in d.columns]
    X = d[cols].copy()
    for c in X.columns:
        if c not in CATEG:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = X[c].astype("object").where(X[c].notna(), "NA").astype(str)
    keep, dropped = prune(X, y)
    X = X[keep]
    num = [c for c in keep if c not in CATEG]; cat = [c for c in keep if c in CATEG]
    exact = d["MixDesignKey"].astype(str).values
    clusters = make_clusters(d).values
    w_map = pd.Series(1.0, index=d.index).groupby(exact).transform(lambda s: 1.0/len(s)).values  # inverse exact-mix
    print(f"\n{'='*84}\nTARGET={target} | rows={len(d)} | exact mixes={pd.Series(exact).nunique()} | eng-clusters={pd.Series(clusters).nunique()}")
    print(f"Correlation pruning dropped {len(dropped)} same-effect features: " + ", ".join(f"{a}~{b}({r})" for a,b,r in dropped))
    print(f"Kept {len(keep)} features: {keep}")
    out = {}
    for gname, groups in [("Exact-Mix (operational)", exact), ("Eng-Cluster (strict)", clusters)]:
        tr, va, te = gsplit(y, groups)
        Xtr,Xva,Xte = X.iloc[tr],X.iloc[va],X.iloc[te]; ytr,yva,yte = y.iloc[tr],y.iloc[va],y.iloc[te]
        leak = len(set(groups[tr])&set(groups[te])) + len(set(groups[va])&set(groups[te]))
        wtr = w_map[tr]
        best_name, best_val, best_fit = None, -9, None
        rows = []
        for name, est in models().items():
            p = pipe(clone(est), num, cat)
            try: p.fit(Xtr, ytr, model__sample_weight=wtr)
            except Exception: p.fit(Xtr, ytr)
            vm = M(yva, p.predict(Xva)); rows.append((name, vm["R2"]))
            if vm["R2"] > best_val: best_name, best_val, best_fit = name, vm["R2"], p
        # stacking of the tree models
        try:
            ests = [(n, pipe(clone(models()[n]), num, cat)) for n in ["LightGBM","ExtraTrees","HistGB"] if n in models()]
            st = StackingRegressor(ests, final_estimator=RidgeCV(), cv=5, n_jobs=-1); st.fit(Xtr, ytr)
            vm = M(yva, st.predict(Xva)); rows.append(("Stacking", vm["R2"]))
            if vm["R2"] > best_val: best_name, best_val, best_fit = "Stacking", vm["R2"], st
        except Exception as e: print("  stacking failed:", e)
        final = clone(best_fit);
        try: final.fit(pd.concat([Xtr,Xva]), pd.concat([ytr,yva]))
        except Exception: final.fit(pd.concat([Xtr,Xva]), pd.concat([ytr,yva]))
        tem = M(yte, final.predict(Xte)); trm = M(ytr, best_fit.predict(Xtr))
        board = " | ".join(f"{n}:{r:.2f}" for n,r in sorted(rows,key=lambda x:-x[1]))
        print(f"\n  [{gname}]  split {len(tr)}/{len(va)}/{len(te)}  leak={leak}")
        print(f"    val leaderboard: {board}")
        print(f"    BEST={best_name} -> Train R2={trm['R2']:.3f} | Val R2={best_val:.3f} | TEST R2={tem['R2']:.3f} RMSE={tem['RMSE']:.3f} MAE={tem['MAE']:.3f}")
        out[gname] = (best_name, tem)
    return out

def main():
    df = engineer(load())
    print("\n" + "#"*84 + "\nRUTTING" + "\n" + "#"*84); run_target(df, "Rut_20k", RUT_FEATURES)
    print("\n" + "#"*84 + "\nSCB" + "\n" + "#"*84);     run_target(df, "SCB", SCB_FEATURES)

if __name__ == "__main__":
    main()
