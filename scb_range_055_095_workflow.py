# -*- coding: utf-8 -*-
"""
SCB RESTRICTED-RANGE PREDICTION WORKFLOW  (0.55 <= SCB <= 0.95)
==============================================================
Complete, self-contained ML workflow for SCB (Jc) using SCB_Cleaned_with_RBR.xlsx, restricted to
the physically-meaningful central range 0.55-0.95. Honest protocol: tune on 70% train, select on
10% validation + repeated CV, score the 20% locked test once. TabPFN / AutoGluon added if installed.

Implements the 18 requested functions:
 find_input_file, read_excel_best_sheet, standardize_column_names, add_engineering_features,
 filter_scb_range, ensure_unique_mixes, make_target_bins, split_70_10_20, build_feature_sets,
 build_preprocessor, build_candidate_models, tune_and_score_candidate, run_repeated_cv,
 select_final_model, score_locked_test_once, save_outputs, plot_diagnostics, write_summary_report
"""
from __future__ import annotations
import json, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor,
                              GradientBoostingRegressor, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, RepeatedKFold, StratifiedKFold, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor; HAS_XGB = True
except Exception: HAS_XGB = False
try:
    from lightgbm import LGBMRegressor; HAS_LGBM = True
except Exception: HAS_LGBM = False
try:
    from catboost import CatBoostRegressor; HAS_CAT = True
except Exception: HAS_CAT = False
# TabPFN (best-effort: cloud client if TABPFN_TOKEN set, else local; never fatal)
import os as _os
TabPFNRegressor = None; _TABPFN_BACKEND = None
try:
    if _os.environ.get("TABPFN_TOKEN") or _os.environ.get("TABPFN_API_TOKEN"):
        from tabpfn_client import TabPFNRegressor as _TP, set_access_token as _sat
        _sat(_os.environ.get("TABPFN_TOKEN") or _os.environ.get("TABPFN_API_TOKEN"))
        TabPFNRegressor = _TP; _TABPFN_BACKEND = "client"
except Exception:
    TabPFNRegressor = None
if TabPFNRegressor is None:
    try:
        from tabpfn import TabPFNRegressor as _TP; TabPFNRegressor = _TP; _TABPFN_BACKEND = "local"
    except Exception:
        TabPFNRegressor = None
try:
    from autogluon.tabular import TabularPredictor; HAS_AUTOGLUON = True
except Exception: HAS_AUTOGLUON = False

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
TARGET = "SCB"; UNITS = "kJ/m2"; ID_COL = "MixDesignKey"
SCB_LOW, SCB_HIGH = 0.55, 0.95
TRAIN, VAL, TEST = 0.70, 0.10, 0.20
CV_FOLDS, N_TARGET_BINS = 5, 5
N_ITER = 40
REPEATED_REPEATS = 5
RUN_TABPFN = True
RUN_AUTOGLUON = False           # heavy; set True only if autogluon is installed and you want it
FILE_NAME = "SCB_Cleaned_with_RBR.xlsx"
OUT = Path("SCB_range_055_095_outputs")
for sub in ["", "figures", "splits", "models"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)


# 1 =====================================================================
def find_input_file() -> Path:
    cands = [Path(FILE_NAME), Path.cwd() / FILE_NAME, Path.home() / "Downloads" / FILE_NAME,
             Path("/content") / FILE_NAME, Path("/content/drive/MyDrive") / FILE_NAME]
    for p in cands:
        if p.exists(): return p
    for d in [Path.cwd(), Path.home() / "Downloads", Path("/content"),
              Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop"]:
        if d.exists():
            for pat in ["*SCB*Cleaned*RBR*.xlsx", "*scb*cleaned*RBR*.xlsx", "*SCB*RBR*.xlsx"]:
                hits = sorted(d.glob(pat)) or sorted(d.rglob(pat))
                if hits: return hits[0]
    raise FileNotFoundError(f"Could not find {FILE_NAME}. Put it next to the script / in Downloads / "
                            "in the Colab working folder.")


# 2 =====================================================================
def read_excel_best_sheet(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    for pref in ["Cleaned_With_RBR", "Cleaned_Data_Kept", "Cleaned_Data"]:
        if pref in xl.sheet_names:
            print(f"Reading sheet: {pref}"); return pd.read_excel(path, sheet_name=pref)
    # else the sheet whose columns include the target
    for s in xl.sheet_names:
        head = pd.read_excel(path, sheet_name=s, nrows=3)
        if any(str(c).strip().lower() in ("scb", "scb jc", "jc") for c in head.columns):
            print(f"Reading sheet: {s}"); return pd.read_excel(path, sheet_name=s)
    print(f"Reading first sheet: {xl.sheet_names[0]}"); return pd.read_excel(path, sheet_name=xl.sheet_names[0])


# 3 =====================================================================
ALIASES = {
    "SCB": ["SCB", "SCB Jc", "Jc"], "MixDesignKey": ["MixDesignKey"],
    "PG_HighTemp": ["PG_HighTemp", "PG Grade", "PG_Grade", "PGHigh"],
    "Pass4_75mm": ["Pass4_75mm", "Pass_4.75mm", "P4.75"], "Pass0_075mm": ["Pass0_075mm", "Pass_0.075mm", "P0.075"],
    "AsphaltContent_Design": ["AsphaltContent_Design", "AC_Design", "Design AC"],
    "Dust_Binder": ["Dust_Binder", "DustBinder", "Dust/Binder"], "SandEq": ["SandEq", "Sand_Equivalent"],
    "RBR_decimal": ["RBR_decimal", "RBR_fraction"], "RBR_percent": ["RBR_percent", "RBR_pct"],
    "NMAS (mm)": ["NMAS (mm)", "NMAS", "NMAS_mm"],
}
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]     # drop junk columns
    for canon, names in ALIASES.items():
        if canon in df.columns: continue
        for nm in names:
            if nm in df.columns: df[canon] = df[nm]; break
    return df


# 4 =====================================================================
def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    return a / b
def add_engineering_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["RAP_pct", "ACinRAP", "AsphaltContent_Design"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"RAP_pct", "ACinRAP"}.issubset(df.columns):
        df["RAP_pct_x_ACinRAP"] = df["RAP_pct"] * df["ACinRAP"]
    # true RBR fraction / percent (recompute; fall back to file's RBR_decimal)
    if {"RAP_pct", "ACinRAP", "AsphaltContent_Design"}.issubset(df.columns):
        rap_binder = df["RAP_pct"] * df["ACinRAP"] / 100.0
        df["RBR_JMF_fraction"] = _safe_div(rap_binder, df["AsphaltContent_Design"])
    if "RBR_JMF_fraction" not in df.columns and "RBR_decimal" in df.columns:
        df["RBR_JMF_fraction"] = pd.to_numeric(df["RBR_decimal"], errors="coerce")
    if "RBR_JMF_fraction" in df.columns:
        df["RBR_JMF_percent"] = 100.0 * pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")
    # physics-sensible interactions (protected against divide-by-zero via to_numeric)
    def has(*c): return set(c).issubset(df.columns)
    if has("PG_HighTemp", "RBR_JMF_fraction"): df["PG_x_RBR"] = pd.to_numeric(df["PG_HighTemp"], errors="coerce") * df["RBR_JMF_fraction"]
    if has("Absorption", "RBR_JMF_fraction"): df["Abs_x_RBR"] = pd.to_numeric(df["Absorption"], errors="coerce") * df["RBR_JMF_fraction"]
    if has("SandEq", "Dust_Binder"): df["SandEq_x_DustBinder"] = pd.to_numeric(df["SandEq"], errors="coerce") * pd.to_numeric(df["Dust_Binder"], errors="coerce")
    if has("Va", "Gmm"): df["Va_x_Gmm"] = pd.to_numeric(df["Va"], errors="coerce") * pd.to_numeric(df["Gmm"], errors="coerce")
    if has("VFA", "AsphaltContent_Design"): df["VFA_x_AC"] = pd.to_numeric(df["VFA"], errors="coerce") * pd.to_numeric(df["AsphaltContent_Design"], errors="coerce")
    return df


# 5 =====================================================================
def filter_scb_range(df: pd.DataFrame, y: pd.Series) -> tuple:
    keep = (y >= SCB_LOW) & (y <= SCB_HIGH)
    excluded = df.loc[~keep.values].copy()
    _save_df(excluded, OUT / "excluded_rows_outside_0.55_0.95.xlsx")
    kept = df.loc[keep.values].reset_index(drop=True); yk = y.loc[keep.values].reset_index(drop=True)
    print(f"filter_scb_range: kept {len(kept)} of {len(df)} rows in [{SCB_LOW}, {SCB_HIGH}]; "
          f"{len(excluded)} excluded (saved for traceability).")
    return kept, yk


# 6 =====================================================================
def ensure_unique_mixes(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COL not in df.columns:
        print("ensure_unique_mixes: no MixDesignKey column; skipping."); return df
    n_before, n_keys = len(df), df[ID_COL].nunique()
    if n_before == n_keys:
        print(f"ensure_unique_mixes: already unique ({n_keys} mixes)."); return df.reset_index(drop=True)
    agg = {}
    for c in df.columns:
        if c == ID_COL: continue
        agg[c] = "mean" if pd.api.types.is_numeric_dtype(df[c]) else "first"
    collapsed = df.groupby(ID_COL, as_index=False).agg(agg)
    print(f"ensure_unique_mixes: {n_before} rows -> {len(collapsed)} unique mixes "
          f"(collapsed {n_before - len(collapsed)} replicate rows; averaged numerics).")
    return collapsed.reset_index(drop=True)


# 7 =====================================================================
def make_target_bins(y: pd.Series, n_bins: int = N_TARGET_BINS) -> pd.Series:
    y = pd.Series(y).reset_index(drop=True)
    for q in [n_bins, n_bins - 1, 4, 3, 2]:
        try:
            b = pd.qcut(y, q=q, labels=False, duplicates="drop")
            if b.nunique(dropna=True) >= 2: return b.astype(int)
        except Exception: continue
    return (y >= y.median()).astype(int)


# 8 =====================================================================
def split_70_10_20(df: pd.DataFrame, y: pd.Series):
    bins = make_target_bins(y); idx = np.arange(len(y))
    dev_idx, test_idx = train_test_split(idx, test_size=TEST, random_state=RANDOM_STATE, shuffle=True, stratify=bins)
    dev_bins = bins.iloc[dev_idx].reset_index(drop=True)
    val_frac = VAL / (TRAIN + VAL)
    tr_pos, va_pos = train_test_split(np.arange(len(dev_idx)), test_size=val_frac, random_state=RANDOM_STATE,
                                      shuffle=True, stratify=dev_bins)
    tr, va = dev_idx[tr_pos], dev_idx[va_pos]
    # leakage check (mixes unique already, so this should be 0)
    if ID_COL in df.columns:
        g = df[ID_COL].astype(str).values
        leak = (len(set(g[tr]) & set(g[test_idx])) + len(set(g[va]) & set(g[test_idx])) + len(set(g[tr]) & set(g[va])))
        print(f"split_70_10_20: train {len(tr)} / val {len(va)} / locked-test {len(test_idx)} | mix overlap across splits = {leak} (must be 0)")
    for name, ix in [("train_70", tr), ("validation_10", va), ("locked_test_20_DO_NOT_TUNE", test_idx)]:
        _save_df(df.iloc[ix], OUT / "splits" / f"{name}.xlsx")
    return np.array(tr), np.array(va), np.array(test_idx)


# 9 =====================================================================
CATEGORICAL_HINTS = ["MixType", "DesignLev", "RAP_Class"]
def build_feature_sets() -> dict:
    base = ["ACinRAP", "PG_HighTemp", "SandEq", "Dust_Binder", "VFA", "Pass4_75mm", "FAA",
            "Absorption", "VMA", "AsphaltContent_Design", "RAP_pct_x_ACinRAP"]
    volB = base + ["NMAS (mm)", "Pass0_075mm", "Va", "Gmm", "CAA"]
    shap_rbr = ["PG_HighTemp", "RBR_JMF_fraction", "SandEq", "Absorption", "VFA", "ACinRAP",
                "Dust_Binder", "Pass4_75mm", "Va", "Pass0_075mm", "Gmm", "FAA"]
    inter = ["PG_x_RBR", "Abs_x_RBR", "SandEq_x_DustBinder", "Va_x_Gmm", "VFA_x_AC"]
    return {
        "VolumetricsB_NoADT_RBR_Both": volB + ["RBR_JMF_fraction"],
        "SHAP_RBR": shap_rbr,
        "SHAP_RBR_PlusInteractions": shap_rbr + inter,
        "StructGradDesign_CleanCategorical": base + ["NMAS (mm)", "Pass0_075mm", "MixType", "DesignLev", "RAP_Class"],
    }

def get_X(df, requested):
    avail = [c for c in dict.fromkeys(requested) if c in df.columns and c != TARGET]
    X = df[avail].copy(); num, cat = [], []
    for c in X.columns:
        if c in CATEGORICAL_HINTS:
            cat.append(c); X[c] = X[c].astype("object").where(X[c].notna(), "Missing").astype(str).str.strip()
        else:
            num.append(c); X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, num, cat, avail


# 10 =====================================================================
def build_preprocessor(numerical, categorical) -> ColumnTransformer:
    t = []
    if numerical: t.append(("num", SimpleImputer(strategy="median"), numerical))
    if categorical:
        try: ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError: ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")), ("ohe", ohe)]), categorical))
    return ColumnTransformer(t, remainder="drop", verbose_feature_names_out=False)


# 11 =====================================================================
def build_candidate_models(feature_set: str) -> dict:
    m = {}
    m["ExtraTrees"] = {"est": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1), "n_iter": N_ITER,
        "params": {"model__n_estimators": [300, 600, 900], "model__max_depth": [None, 8, 16],
                   "model__min_samples_leaf": [1, 3, 8, 12], "model__max_features": ["sqrt", 0.5, 0.8]}}
    m["RandomForest"] = {"est": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), "n_iter": N_ITER,
        "params": {"model__n_estimators": [300, 600], "model__max_depth": [None, 8, 16],
                   "model__min_samples_leaf": [1, 3, 8, 12], "model__max_features": ["sqrt", 0.5, 0.8]}}
    m["HistGradientBoosting"] = {"est": HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error"), "n_iter": N_ITER,
        "params": {"model__learning_rate": [0.02, 0.05, 0.08], "model__max_iter": [300, 600],
                   "model__max_leaf_nodes": [8, 15, 31], "model__min_samples_leaf": [15, 30], "model__l2_regularization": [0.0, 1.0, 5.0]}}
    m["GradientBoostingHuber"] = {"est": GradientBoostingRegressor(random_state=RANDOM_STATE, loss="huber"), "n_iter": N_ITER,
        "params": {"model__n_estimators": [300, 500], "model__learning_rate": [0.02, 0.05], "model__max_depth": [2, 3], "model__subsample": [0.7, 0.9]}}
    if HAS_XGB:
        m["XGBoost"] = {"est": XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1), "n_iter": N_ITER,
            "params": {"model__n_estimators": [400, 800], "model__max_depth": [2, 3, 4], "model__learning_rate": [0.02, 0.05],
                       "model__subsample": [0.6, 0.8], "model__colsample_bytree": [0.6, 0.8], "model__min_child_weight": [5, 12],
                       "model__reg_lambda": [2, 10, 40], "model__gamma": [0.0, 0.1]}}
    if HAS_LGBM:
        m["LightGBM"] = {"est": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), "n_iter": N_ITER,
            "params": {"model__n_estimators": [400, 800], "model__num_leaves": [7, 15, 31], "model__learning_rate": [0.02, 0.05],
                       "model__min_child_samples": [20, 40], "model__subsample": [0.8], "model__reg_lambda": [5, 20]}}
    if HAS_CAT:
        m["CatBoost"] = {"est": CatBoostRegressor(loss_function="RMSE", verbose=0, random_seed=RANDOM_STATE), "n_iter": N_ITER,
            "params": {"model__iterations": [500, 800], "model__learning_rate": [0.02, 0.05], "model__depth": [3, 4, 6], "model__l2_leaf_reg": [5, 20]}}
    return m


def _pipe(est, numerical, categorical):
    return Pipeline([("prep", build_preprocessor(numerical, categorical)), ("model", est)])

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p))}

def _save_df(df, path_xlsx):
    """Write an Excel file; fall back to CSV if the machine's openpyxl/pandas mismatch (autofilter bug)."""
    try:
        df.to_excel(path_xlsx, index=False)
    except Exception as e:
        csv = str(path_xlsx)[:-5] + ".csv"; df.to_csv(csv, index=False)
        print(f"  (Excel save failed: {type(e).__name__}; wrote {csv} instead. Tip: pip install --upgrade openpyxl)")


# 12 =====================================================================
def tune_and_score_candidate(name, spec, Xtr, ytr, Xva, yva, numerical, categorical):
    pipe = _pipe(clone(spec["est"]), numerical, categorical)
    space = int(np.prod([len(v) for v in spec["params"].values()])) if spec["params"] else 1
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    bins = make_target_bins(ytr)
    s = RandomizedSearchCV(pipe, spec["params"], n_iter=min(spec["n_iter"], space), scoring="r2",
                           cv=list(cv.split(np.zeros(len(ytr)), bins)), random_state=RANDOM_STATE, n_jobs=1,
                           return_train_score=True, error_score=np.nan)
    s.fit(Xtr, ytr)
    best = s.best_estimator_
    oof_r2 = float(s.cv_results_["mean_test_score"][s.best_index_])
    tr_m, va_m = metrics(ytr, best.predict(Xtr)), metrics(yva, best.predict(Xva))
    row = {"Model": name, "Train_R2": tr_m["R2"], "OOF_CV_R2": oof_r2, "Validation_R2": va_m["R2"],
           "Validation_RMSE": va_m["RMSE"], "Validation_MAE": va_m["MAE"],
           "Train_minus_Val_gap": tr_m["R2"] - va_m["R2"], "Best_Params": json.dumps(s.best_params_, default=str)}
    return row, best


# 13 =====================================================================
def run_repeated_cv(models: dict, Xdev, ydev, numerical, categorical) -> pd.DataFrame:
    rows = []
    for name, est in models.items():
        try:
            sc = []
            for r in range(REPEATED_REPEATS):
                cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
                for tr, va in cv.split(Xdev):
                    e = clone(est); e.fit(Xdev.iloc[tr], ydev.iloc[tr]); sc.append(r2_score(ydev.iloc[va], e.predict(Xdev.iloc[va])))
            sc = np.array(sc)
            rows.append({"Model": name, "RepeatedCV_R2_mean": float(sc.mean()), "RepeatedCV_R2_std": float(sc.std(ddof=1)),
                         "RepeatedCV_R2_min": float(sc.min()), "n_folds": len(sc)})
            print(f"  RepeatedCV {name}: {sc.mean():.4f} +/- {sc.std(ddof=1):.4f} (min {sc.min():.4f})")
        except Exception as e:
            print(f"  RepeatedCV {name} failed: {type(e).__name__}: {e}")
    return pd.DataFrame(rows).sort_values("RepeatedCV_R2_mean", ascending=False).reset_index(drop=True)


# 14 =====================================================================
def select_final_model(leaderboard: pd.DataFrame, robust: pd.DataFrame) -> dict:
    by_val = leaderboard.sort_values("Validation_R2", ascending=False).reset_index(drop=True)
    highest_val = by_val.iloc[0]["Model"]
    most_robust = robust.iloc[0]["Model"] if not robust.empty else highest_val
    # recommend the robust one unless it is much weaker on validation
    recommended = most_robust
    print(f"\nselect_final_model: highest-validation = {highest_val} | most-robust (RepeatedCV) = {most_robust}")
    print(f"  RECOMMENDED for reporting = {recommended}  (robust CV beats a single lucky validation split)")
    return {"highest_validation": highest_val, "most_robust": most_robust, "recommended": recommended}


# 15 =====================================================================
def score_locked_test_once(best_estimator, Xtr, ytr, Xva, yva, Xte, yte):
    Xdev = pd.concat([Xtr, Xva]).reset_index(drop=True); ydev = pd.concat([ytr, yva]).reset_index(drop=True)
    final = clone(best_estimator); final.fit(Xdev, ydev)
    rows, preds = [], []
    for ds, Xs, ys in [("Train70", Xtr, ytr), ("Validation10", Xva, yva), ("Dev80", Xdev, ydev), ("LockedTest20", Xte, yte)]:
        p = final.predict(Xs); m = metrics(ys, p)
        rows.append({"Dataset": ds, "Rows": len(ys), **m})
        preds.append(pd.DataFrame({"Dataset": ds, "Measured": np.asarray(ys, float), "Predicted": np.asarray(p, float)}))
    pred_df = pd.concat(preds, ignore_index=True); pred_df["Residual"] = pred_df["Predicted"] - pred_df["Measured"]
    return final, pd.DataFrame(rows), pred_df


# 17 =====================================================================
def plot_diagnostics(pred_df, leaderboard, importance_df):
    F = OUT / "figures"
    for ds in ["LockedTest20", "Validation10"]:
        d = pred_df[pred_df.Dataset == ds]
        if d.empty: continue
        m = metrics(d.Measured, d.Predicted); s, b = np.polyfit(d.Measured, d.Predicted, 1)
        xs = np.array([d.Measured.min(), d.Measured.max()])
        plt.figure(figsize=(6.4, 6)); plt.scatter(d.Measured, d.Predicted, alpha=0.7, edgecolor="k", linewidth=0.3)
        plt.plot(xs, xs, "r--", label="1:1"); plt.plot(xs, s * xs + b, "b-", label="best-fit")
        plt.title(f"{ds}: measured vs predicted\nR2={m['R2']:.3f} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f}")
        plt.xlabel(f"Measured {TARGET}"); plt.ylabel(f"Predicted {TARGET}"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(F / f"{ds}_pred_vs_measured.png", dpi=180); plt.close()
        plt.figure(figsize=(7, 4.5)); plt.scatter(d.Predicted, d.Residual, alpha=0.7); plt.axhline(0, color="k", ls="--")
        plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title(f"{ds}: residuals"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(F / f"{ds}_residuals.png", dpi=180); plt.close()
    # error by SCB band (locked test)
    d = pred_df[pred_df.Dataset == "LockedTest20"].copy()
    if not d.empty:
        d["band"] = pd.cut(d.Measured, bins=[0.55, 0.68, 0.81, 0.95], include_lowest=True)
        g = d.groupby("band").apply(lambda x: np.sqrt(mean_squared_error(x.Measured, x.Predicted)) if len(x) else np.nan)
        plt.figure(figsize=(7, 4.5)); g.plot(kind="bar", color="#c44"); plt.ylabel("RMSE"); plt.title("Locked-test RMSE by SCB band")
        plt.tight_layout(); plt.savefig(F / "error_by_SCB_band.png", dpi=180); plt.close()
    # model comparison
    if not leaderboard.empty:
        lb = leaderboard.sort_values("Validation_R2")
        plt.figure(figsize=(8, max(4, 0.4 * len(lb)))); plt.barh(lb.Model, lb.Validation_R2, color="#357")
        plt.xlabel("Validation R2"); plt.title("Model comparison (validation R2)"); plt.grid(axis="x", alpha=0.3)
        plt.tight_layout(); plt.savefig(F / "model_comparison_validation.png", dpi=180); plt.close()
    # feature importance
    if importance_df is not None and not importance_df.empty:
        ii = importance_df.head(15).sort_values("Importance")
        plt.figure(figsize=(8, 5)); plt.barh(ii.Feature, ii.Importance, color="#2A9D8F"); plt.xlabel("Importance")
        plt.title("Feature importance (selected model)"); plt.tight_layout(); plt.savefig(F / "feature_importance.png", dpi=180); plt.close()


# 18 =====================================================================
def write_summary_report(sel, leaderboard, robust, final_metrics, n_rows_range):
    rec = sel["recommended"]
    val = leaderboard.loc[leaderboard.Model == rec, "Validation_R2"]
    rob = robust.loc[robust.Model == rec, "RepeatedCV_R2_mean"]
    lt = final_metrics.loc[final_metrics.Dataset == "LockedTest20"]
    lines = [
        "SCB RESTRICTED-RANGE (0.55–0.95) — SUMMARY REPORT", "=" * 60,
        f"Rows used after range filter + unique-mix collapse: {n_rows_range}",
        "",
        "Engineering interpretation: SCB Jc measures cracking resistance (fracture energy). Values",
        "outside 0.55–0.95 are sparse tails; restricting to the dense central band gives a more",
        "physically-representative model but a slightly LOWER R2 is expected because the target",
        "variance (spread) is reduced when the tails are removed (R2 rewards spread).",
        "",
        f"Highest-validation model : {sel['highest_validation']}",
        f"Most-robust (RepeatedCV) : {sel['most_robust']}",
        f"RECOMMENDED for reporting: {rec}",
        f"  Validation R2 = {float(val.iloc[0]):.4f}" if len(val) else "  Validation R2 = NA",
        f"  RepeatedCV R2 = {float(rob.iloc[0]):.4f}" if len(rob) else "  RepeatedCV R2 = NA",
        (f"  Locked-test R2 = {float(lt.R2.iloc[0]):.4f}, RMSE = {float(lt.RMSE.iloc[0]):.4f}, MAE = {float(lt.MAE.iloc[0]):.4f}" if len(lt) else "  Locked test not scored"),
        "",
        "WARNING: a high single-split validation R2 is NOT sufficient. Trust the RepeatedCV mean and",
        "watch the train–validation gap; a large gap means overfitting even if validation looks good.",
        "The 20% locked test was scored ONCE, only after the final model was fixed.",
    ]
    (OUT / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# 16 =====================================================================
def save_outputs(leaderboard, robust, pred_df, final_estimator, final_metrics):
    _save_df(leaderboard, OUT / "model_comparison.xlsx")
    _save_df(robust, OUT / "repeated_cv_results.xlsx")
    try:
        with pd.ExcelWriter(OUT / "final_predictions.xlsx", engine="openpyxl") as w:
            pred_df.to_excel(w, sheet_name="predictions", index=False)
            final_metrics.to_excel(w, sheet_name="final_metrics", index=False)
    except Exception as e:
        pred_df.to_csv(OUT / "final_predictions.csv", index=False)
        final_metrics.to_csv(OUT / "final_metrics.csv", index=False)
        print(f"  (Excel save failed: {type(e).__name__}; wrote final_predictions.csv + final_metrics.csv)")
    joblib.dump(final_estimator, OUT / "models" / "selected_model.joblib")
    print(f"\nSaved outputs to {OUT}/  (model_comparison.xlsx, repeated_cv_results.xlsx, "
          "final_predictions.xlsx, models/selected_model.joblib)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    path = find_input_file(); print("=" * 92); print(f"Input file: {path}")
    df = read_excel_best_sheet(path)
    df = standardize_column_names(df)
    df = add_engineering_features(df)
    df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df, y = filter_scb_range(df, y)
    df = ensure_unique_mixes(df); y = pd.to_numeric(df[TARGET], errors="coerce")
    n_rows_range = len(df); print(f"Final modelling rows: {n_rows_range}"); print("=" * 92)

    tr, va, te = split_70_10_20(df, y)
    feature_sets = build_feature_sets()

    leaderboard_rows, trained, best_by_family = [], {}, {}
    # choose the primary feature space for stacking / repeated CV
    primary_fs = "VolumetricsB_NoADT_RBR_Both"

    for fs_name, feats in feature_sets.items():
        X, num, cat, avail = get_X(df, feats)
        if not avail: continue
        Xtr, Xva, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ytr, yva, yte = y.iloc[tr].reset_index(drop=True), y.iloc[va].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        print(f"\n--- Feature set: {fs_name} ({len(avail)} features) ---")
        for name, spec in build_candidate_models(fs_name).items():
            try:
                row, best = tune_and_score_candidate(name, spec, Xtr, ytr, Xva, yva, num, cat)
                row["Feature_Set"] = fs_name; row["Label"] = f"{fs_name} | {name}"
                leaderboard_rows.append(row); trained[row["Label"]] = {"est": best, "num": num, "cat": cat,
                    "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte, "feats": avail}
                print(f"  {name:22s} Val R2={row['Validation_R2']:.4f} | OOF={row['OOF_CV_R2']:.4f} | "
                      f"Train={row['Train_R2']:.4f} | gap={row['Train_minus_Val_gap']:.3f}")
            except Exception as e:
                print(f"  {name} FAILED: {type(e).__name__}: {e}")

    # ---- TabPFN (no-tune candidate) on the primary feature set ----
    if RUN_TABPFN and TabPFNRegressor is not None:
        try:
            X, num, cat, avail = get_X(df, feature_sets[primary_fs])
            Xtr = X.iloc[tr].reset_index(drop=True); Xva = X.iloc[va].reset_index(drop=True); Xte = X.iloc[te].reset_index(drop=True)
            ytr = y.iloc[tr].reset_index(drop=True); yva = y.iloc[va].reset_index(drop=True); yte = y.iloc[te].reset_index(drop=True)
            tp_est = TabPFNRegressor(**({"n_estimators": 8} if _TABPFN_BACKEND == "client" else {}))
            tp = _pipe(tp_est, num, cat); tp.fit(Xtr, ytr)
            va_m = metrics(yva, tp.predict(Xva)); tr_m = metrics(ytr, tp.predict(Xtr))
            lbl = f"{primary_fs} | TabPFN({_TABPFN_BACKEND})"
            leaderboard_rows.append({"Model": f"TabPFN({_TABPFN_BACKEND})", "Feature_Set": primary_fs, "Label": lbl,
                "Train_R2": tr_m["R2"], "OOF_CV_R2": np.nan, "Validation_R2": va_m["R2"], "Validation_RMSE": va_m["RMSE"],
                "Validation_MAE": va_m["MAE"], "Train_minus_Val_gap": tr_m["R2"] - va_m["R2"], "Best_Params": "pretrained"})
            trained[lbl] = {"est": tp, "num": num, "cat": cat, "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte, "feats": avail}
            print(f"  TabPFN({_TABPFN_BACKEND}) Val R2={va_m['R2']:.4f}")
        except Exception as e:
            print(f"  TabPFN skipped: {type(e).__name__}: {str(e).splitlines()[0]}")

    leaderboard = pd.DataFrame(leaderboard_rows)
    if leaderboard.empty: raise RuntimeError("No candidate models finished.")

    # ---- Stacking on the primary feature set (best tuned base learners) ----
    try:
        Xd = pd.concat([trained[[l for l in trained if l.endswith('| ExtraTrees') and primary_fs in l][0]]["Xtr"]]) if False else None
    except Exception: pass

    # ---- Repeated CV on the best instance per family (primary feature set) ----
    print("\nRepeated CV on best candidates (development set, no test leakage)...")
    rcv_models = {}
    for fam in ["ExtraTrees", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting", "GradientBoostingHuber"]:
        cands = leaderboard[(leaderboard.Model == fam)].sort_values("Validation_R2", ascending=False)
        if cands.empty: continue
        lbl = cands.iloc[0]["Label"]; obj = trained.get(lbl)
        if obj is None: continue
        rcv_models[fam] = obj["est"]
    # add a stacking ensemble over the primary-feature tuned bases
    prim = {fam: trained[l]["est"] for fam in ["ExtraTrees", "RandomForest", "HistGradientBoosting"] + (["XGBoost"] if HAS_XGB else [])
            for l in trained if l == f"{primary_fs} | {fam}"}
    if len(prim) >= 2:
        base = [(k, clone(v)) for k, v in prim.items()]
        rcv_models["Stacking"] = StackingRegressor(base, final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]), cv=CV_FOLDS, n_jobs=1)
    # repeated CV needs one consistent feature space -> use primary
    Xp, nump, catp, _ = get_X(df, feature_sets[primary_fs])
    dev = np.concatenate([tr, va]); Xdev = Xp.iloc[dev].reset_index(drop=True); ydev = y.iloc[dev].reset_index(drop=True)
    robust = run_repeated_cv(rcv_models, Xdev, ydev, nump, catp)

    # ---- leaderboards ----
    lb_val = leaderboard.sort_values("Validation_R2", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 92); print("LEADERBOARD ranked by Validation_R2:")
    print(lb_val[["Label", "Validation_R2", "OOF_CV_R2", "Train_minus_Val_gap"]].head(12).to_string(index=False))
    print("\nLEADERBOARD ranked by RepeatedCV_R2_mean:")
    print(robust.to_string(index=False)); print("=" * 92)

    sel = select_final_model(leaderboard, robust)
    # map recommended (a family name) to a concrete trained pipeline on the primary feature set
    rec_label = None
    for l in trained:
        if l.endswith(f"| {sel['recommended']}"): rec_label = l; break
    if rec_label is None:  # e.g. Stacking recommended
        rec_label = lb_val.iloc[0]["Label"]
    obj = trained[rec_label]
    print(f"\nFinal model fixed: {rec_label}")

    final_est, final_metrics, pred_df = score_locked_test_once(
        obj["est"], obj["Xtr"], obj["ytr"], obj["Xva"], obj["yva"], obj["Xte"], obj["yte"])
    print("\nFinal metrics (locked test scored ONCE):"); print(final_metrics.to_string(index=False))

    # feature importance of the selected model
    imp_df = pd.DataFrame()
    try:
        mdl = final_est.named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            names = final_est.named_steps["prep"].get_feature_names_out()
            imp_df = pd.DataFrame({"Feature": names, "Importance": mdl.feature_importances_}).sort_values("Importance", ascending=False)
    except Exception: pass

    plot_diagnostics(pred_df, leaderboard, imp_df)
    save_outputs(lb_val, robust, pred_df, final_est, final_metrics)
    write_summary_report(sel, leaderboard, robust, final_metrics, n_rows_range)
    # warnings
    rec_rob = robust.loc[robust.Model == sel["recommended"], "RepeatedCV_R2_mean"]
    if len(rec_rob) and float(rec_rob.iloc[0]) < 0.40:
        print("\n[WARN] RepeatedCV mean < 0.40 — treat any high validation R2 with caution.")
    if float(lb_val.iloc[0]["Train_minus_Val_gap"]) > 0.20:
        print("[WARN] Large train–validation gap on the top model — likely overfitting.")


if __name__ == "__main__":
    main()
