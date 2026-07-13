# -*- coding: utf-8 -*-
"""
RUT_20k ENHANCED WORKFLOW — CLEANED DATA FILE (New_Data_SCB_LWT_Cleaned_Modeling_Files.xlsx,
sheet LWT_Clean_Modeling, 1890 rows)
===========================================================================
Data notes (from the file's own Feature_Removal_Log): Pass4_75mm/Pass0_075mm are replaced by
Grad_No4/Grad_No200; Pba_pct (negative values -> physically invalid) and Gmb_specimen_AC are
removed; additive columns are the cleaned Additive_Type_clean/Additive_Rate_clean; RBR_percent
is shipped directly. FLAG_* columns are row-quality flags and are NEVER used as predictors.

Mirrors the code that got the HIGH rutting result (rut_tabpfn_workflow.py: TabPFN test 0.612,
XGBoost 0.601, TabPFN+XGB blend 0.613) and ENHANCES it with the NEW calculated features shipped
in the updated data file:

    Pba_pct (absorbed binder)  <- strongest new Rut predictor
    Dust_Pbe_ratio, Gmb_specimen_AC, Additive_Rate, AFT_micron,
    Grad_3_8in / Grad_1_2in / Grad_3_4in (coarse gradation = grain skeleton -> rutting),
    Grad_No4, Grad_No200

Models: tuned XGBoost (advisor regularized grid) + TabPFN (cloud client if TABPFN_TOKEN is set,
else local install, else skipped) + a validation-weighted blend of the two.
Protocol (honest): 70/10/20 stratified split, tune on train only, RepeatedCV on dev,
locked test scored ONCE at the end.
"""
from __future__ import annotations
import json, os, warnings
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, StratifiedKFold, StratifiedGroupKFold,
                                     RandomizedSearchCV, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor; HAS_XGB = True
except Exception: HAS_XGB = False

# TabPFN: cloud client if a token is set (n_estimators max 8), else local, else skip.
TabPFNRegressor = None; _TABPFN_BACKEND = None
TABPFN_ENSEMBLE = 8
try:
    if os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_API_TOKEN"):
        from tabpfn_client import TabPFNRegressor as _TP, set_access_token as _sat
        _sat(os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_API_TOKEN"))
        TabPFNRegressor = _TP; _TABPFN_BACKEND = "client"
except Exception:
    TabPFNRegressor = None
if TabPFNRegressor is None:
    try:
        from tabpfn import TabPFNRegressor as _TP; TabPFNRegressor = _TP; _TABPFN_BACKEND = "local"
    except Exception:
        TabPFNRegressor = None

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
TARGET = "Rut_20k"; UNITS = "mm"; ID_COL = "MixDesignKey"
SHEET = "LWT_Clean_Modeling"         # cleaned LWT/rutting modeling sheet
FILE_NAME = "New_Data_SCB_LWT_Cleaned_Modeling_Files.xlsx"
TRAIN, VAL, TEST = 0.70, 0.10, 0.20
CV_FOLDS, N_TARGET_BINS = 5, 5
N_ITER_XGB = 120                     # the winner's tuning budget style
REPEATED_REPEATS = 5
# Use ALL test rows (replicates kept). Leakage is prevented by the GROUP-aware split below:
# every replicate of a mix goes to the SAME split, so no mix straddles train/val/test.
UNIQUE_MIXES = False
OUT = Path("Rut_Enhanced_NewData_outputs")
for sub in ["", "figures", "splits", "models"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)

# =============================================================================
# FEATURES: old winner set + the NEW calculated features (chosen by their effect on Rut)
# =============================================================================
# Pass4_75mm -> Grad_No4 and Pass0_075mm -> Grad_No200 (same sieves; per Feature_Removal_Log).
# Pba_pct and Gmb_specimen_AC were removed by the cleaning; Pbe_pct/Gse carry the binder-
# absorption signal instead.
OLD_WINNER = ["ACinRAP", "PG_HighTemp", "SandEq", "Dust_Binder", "VFA", "Grad_No4", "FAA",
              "Absorption", "VMA", "AsphaltContent_Design", "RAP_pct_x_ACinRAP",
              "NMAS (mm)", "Grad_No200", "Va", "Gmm", "CAA", "RBR_JMF_fraction"]
NEW_RUT = ["Pbe_pct", "Gse", "Dust_Pbe_ratio", "Additive_Rate_clean", "AFT_micron",
           "Grad_3_8in", "Grad_1_2in", "Grad_3_4in", "SurfaceArea_m2kg", "MixTemperature_F_clean"]
CATEGORICAL_HINTS = ["MixType", "DesignLev", "Additive_Type_clean", "Has_Additive"]

def build_feature_sets() -> dict:
    return {
        "Winner_OldFeatures": OLD_WINNER,
        "Winner_PlusNewFeatures": OLD_WINNER + NEW_RUT,
        "PlusNew_WithAdditiveType": OLD_WINNER + NEW_RUT + ["Additive_Type_clean", "Has_Additive"],
    }

# =============================================================================
# DATA
# =============================================================================
def find_input_file() -> Path:
    home = Path.home()
    cands = [Path(FILE_NAME), Path.cwd() / FILE_NAME, home / "Downloads" / FILE_NAME,
             Path("/content") / FILE_NAME]
    for p in cands:
        if p.exists(): return p
    for d in [Path.cwd(), home / "Downloads", home / "Desktop", home / "OneDrive" / "Desktop",
              Path("/content"), Path("/root/.claude/uploads")]:
        if d.exists():
            # also matches copies like "New_Data_SCB_LWT_Cleaned_Modeling_Files (1).xlsx"
            for pat in ["*New_Data*SCB_LWT*Cleaned*Modeling*.xlsx", "*New_Data_SCB_LWT*.xlsx",
                        "*SCB_LWT*Cleaned*.xlsx"]:
                hits = sorted(d.glob(pat)) or sorted(d.rglob(pat))
                if hits: return hits[0]
    raise FileNotFoundError(f"Put {FILE_NAME} next to this script or in Downloads.")

def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    return a / b

def load_data() -> tuple:
    path = find_input_file()
    print("=" * 92); print(f"Input file: {path}  (sheet {SHEET})")
    df = pd.read_excel(path, sheet_name=SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for c in ["RAP_pct", "ACinRAP", "AsphaltContent_Design"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"RAP_pct", "ACinRAP"}.issubset(df.columns):
        df["RAP_pct_x_ACinRAP"] = df["RAP_pct"] * df["ACinRAP"]
    # RBR: prefer the file's own RBR_percent; recompute only if it is absent
    if "RBR_percent" in df.columns:
        df["RBR_JMF_fraction"] = pd.to_numeric(df["RBR_percent"], errors="coerce") / 100.0
    elif {"RAP_pct", "ACinRAP", "AsphaltContent_Design"}.issubset(df.columns):
        df["RBR_JMF_fraction"] = _safe_div(df["RAP_pct"] * df["ACinRAP"] / 100.0, df["AsphaltContent_Design"])
    df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
    if UNIQUE_MIXES and ID_COL in df.columns and df[ID_COL].nunique() < len(df):
        n0 = len(df)
        agg = {c: ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "first")
               for c in df.columns if c != ID_COL}
        df = df.groupby(ID_COL, as_index=False).agg(agg)
        print(f"UNIQUE-MIX DATA: {n0} rows -> {len(df)} unique mixes (replicates averaged; no leakage).")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    print(f"Rows: {len(df)} | {TARGET}: min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f} {UNITS}")
    print("=" * 92)
    return df, y

# =============================================================================
# SPLIT / PIPELINE HELPERS
# =============================================================================
def make_target_bins(y, n_bins=N_TARGET_BINS):
    y = pd.Series(y).reset_index(drop=True)
    for q in [n_bins, n_bins - 1, 4, 3, 2]:
        try:
            b = pd.qcut(y, q=q, labels=False, duplicates="drop")
            if b.nunique(dropna=True) >= 2: return b.astype(int)
        except Exception: continue
    return (y >= y.median()).astype(int)

def _group_holdout(bins, groups, holdout_size, seed):
    """Target-stratified holdout that keeps every replicate of a mix together."""
    n_splits = max(2, int(round(1.0 / holdout_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep, hold = next(iter(sgkf.split(np.zeros(len(bins)), bins, groups)))
    return np.array(keep), np.array(hold)

def split_70_10_20(df, y):
    bins = make_target_bins(y); idx = np.arange(len(y))
    groups = df[ID_COL].astype(str).values if ID_COL in df.columns else None
    if groups is not None and len(set(groups)) < len(groups):
        # ALL rows kept -> GROUP-aware split so replicates never straddle splits
        dev, te = _group_holdout(bins, groups, TEST, RANDOM_STATE)
        vfrac = VAL / (TRAIN + VAL)
        trp, vap = _group_holdout(bins.iloc[dev].reset_index(drop=True), groups[dev], vfrac, RANDOM_STATE)
        tr, va = dev[trp], dev[vap]
    else:
        dev, te = train_test_split(idx, test_size=TEST, random_state=RANDOM_STATE, shuffle=True, stratify=bins)
        vfrac = VAL / (TRAIN + VAL)
        trp, vap = train_test_split(np.arange(len(dev)), test_size=vfrac, random_state=RANDOM_STATE,
                                    shuffle=True, stratify=bins.iloc[dev].reset_index(drop=True))
        tr, va = dev[trp], dev[vap]
    if groups is not None:
        g = groups
        leak = len(set(g[tr]) & set(g[te])) + len(set(g[va]) & set(g[te])) + len(set(g[tr]) & set(g[va]))
        print(f"Split: train {len(tr)} / val {len(va)} / locked-test {len(te)} rows | mix overlap = {leak} (must be 0)")
        print(f"       mixes: train {len(set(g[tr]))} / val {len(set(g[va]))} / test {len(set(g[te]))}")
    for name, ix in [("train_70", tr), ("validation_10", va), ("locked_test_20_DO_NOT_TUNE", te)]:
        _save_df(df.iloc[ix], OUT / "splits" / f"{name}.xlsx")
    return tr, va, te

def get_X(df, requested):
    avail = [c for c in dict.fromkeys(requested) if c in df.columns and c != TARGET]
    X = df[avail].copy(); num, cat = [], []
    for c in X.columns:
        if c in CATEGORICAL_HINTS:
            cat.append(c); X[c] = X[c].astype("object").where(X[c].notna(), "Missing").astype(str).str.strip()
        else:
            num.append(c); X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, num, cat, avail

def build_preprocessor(numerical, categorical):
    t = []
    if numerical: t.append(("num", SimpleImputer(strategy="median"), numerical))
    if categorical:
        try: ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError: ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                                   ("ohe", ohe)]), categorical))
    return ColumnTransformer(t, remainder="drop", verbose_feature_names_out=False)

def _pipe(est, num, cat):
    return Pipeline([("prep", build_preprocessor(num, cat)), ("model", est)])

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}

def _save_df(df, path_xlsx):
    try: df.to_excel(path_xlsx, index=False)
    except Exception as e:
        csv = str(path_xlsx)[:-5] + ".csv"; df.to_csv(csv, index=False)
        print(f"  (Excel save failed: {type(e).__name__}; wrote {csv}. Tip: pip install --upgrade openpyxl)")

# =============================================================================
# MODELS: tuned XGBoost (the advisor regularized grid that scored 0.60) + TabPFN + blend
# =============================================================================
XGB_GRID = {"model__n_estimators": [500, 900, 1400], "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "model__subsample": [0.6, 0.7, 0.8], "model__colsample_bytree": [0.5, 0.6, 0.8],
            "model__min_child_weight": [5, 10, 20], "model__reg_alpha": [0.1, 0.5, 1.0, 2.0],
            "model__reg_lambda": [5, 10, 30, 60], "model__gamma": [0.0, 0.1, 0.2]}

def _train_cv_splits(ytr, gtr):
    """Group-aware CV inside training when replicates exist (keeps a mix in one fold)."""
    bins = make_target_bins(ytr)
    if gtr is not None and len(set(gtr)) < len(gtr):
        cv = StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        return list(cv.split(np.zeros(len(ytr)), bins, gtr))
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(ytr)), bins))

def tune_xgb(Xtr, ytr, Xva, yva, num, cat, gtr=None):
    est = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                       random_state=RANDOM_STATE, n_jobs=-1)
    pipe = _pipe(est, num, cat)
    s = RandomizedSearchCV(pipe, XGB_GRID, n_iter=N_ITER_XGB, scoring="r2",
                           cv=_train_cv_splits(ytr, gtr), random_state=RANDOM_STATE,
                           n_jobs=1, return_train_score=True, error_score=np.nan)
    s.fit(Xtr, ytr)
    best = s.best_estimator_
    oof = float(s.cv_results_["mean_test_score"][s.best_index_])
    trm, vam = metrics(ytr, best.predict(Xtr)), metrics(yva, best.predict(Xva))
    row = {"Model": "XGBoost", "Train_R2": trm["R2"], "OOF_CV_R2": oof, "Validation_R2": vam["R2"],
           "Validation_RMSE": vam["RMSE"], "Validation_MAE": vam["MAE"],
           "Train_minus_Val_gap": trm["R2"] - vam["R2"], "Best_Params": json.dumps(s.best_params_, default=str)}
    return row, best

def fit_tabpfn(Xtr, ytr, num, cat):
    if TabPFNRegressor is None: return None
    kw = {"n_estimators": TABPFN_ENSEMBLE} if _TABPFN_BACKEND == "client" else {}
    tp = _pipe(TabPFNRegressor(**kw), num, cat)
    tp.fit(Xtr, ytr)
    return tp

def repeated_cv(est, Xdev, ydev, tag, gdev=None):
    """Repeated CV; GROUP-aware when replicates exist (each repeat randomly re-assigns whole
    mixes to folds, so replicates of a mix are never split between train and validation)."""
    sc = []
    for r in range(REPEATED_REPEATS):
        if gdev is not None and len(set(gdev)) < len(gdev):
            rng = np.random.RandomState(RANDOM_STATE + 59 * r)
            ug = np.unique(gdev); fold_of = dict(zip(ug, rng.randint(0, CV_FOLDS, size=len(ug))))
            f = np.array([fold_of[g] for g in gdev])
            splits = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(CV_FOLDS)]
        else:
            cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
            splits = list(cv.split(Xdev))
        for tr, va in splits:
            if len(va) == 0 or len(tr) == 0: continue
            e = clone(est); e.fit(Xdev.iloc[tr], ydev.iloc[tr])
            sc.append(r2_score(ydev.iloc[va], e.predict(Xdev.iloc[va])))
    sc = np.array(sc)
    print(f"  RepeatedCV {tag}: {sc.mean():.4f} +/- {sc.std(ddof=1):.4f} (min {sc.min():.4f})")
    return float(sc.mean()), float(sc.std(ddof=1)), float(sc.min())

# =============================================================================
# MAIN
# =============================================================================
def main():
    if not HAS_XGB: raise RuntimeError("xgboost is required: pip install xgboost")
    df, y = load_data()
    tr, va, te = split_70_10_20(df, y)
    groups = df[ID_COL].astype(str).values if ID_COL in df.columns else None
    gtr = groups[tr] if groups is not None else None
    gdev = groups[np.concatenate([tr, va])] if groups is not None else None
    rows, trained = [], {}

    for fs_name, feats in build_feature_sets().items():
        X, num, cat, avail = get_X(df, feats)
        Xtr, Xva, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ytr, yva, yte = y.iloc[tr].reset_index(drop=True), y.iloc[va].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        print(f"\n--- Feature set: {fs_name} ({len(avail)} features) ---")
        # tuned XGBoost
        try:
            row, best = tune_xgb(Xtr, ytr, Xva, yva, num, cat, gtr=gtr)
            row["Feature_Set"] = fs_name; row["Label"] = f"{fs_name} | XGBoost"
            rows.append(row)
            trained[row["Label"]] = {"est": best, "kind": "xgb", "Xtr": Xtr, "ytr": ytr, "Xva": Xva,
                                     "yva": yva, "Xte": Xte, "yte": yte, "num": num, "cat": cat}
            print(f"  XGBoost        Val R2={row['Validation_R2']:.4f} | OOF={row['OOF_CV_R2']:.4f} | gap={row['Train_minus_Val_gap']:.3f}")
        except Exception as e:
            print(f"  XGBoost FAILED: {type(e).__name__}: {e}")
        # TabPFN (no tuning; pretrained transformer)
        try:
            tp = fit_tabpfn(Xtr, ytr, num, cat)
            if tp is not None:
                vam = metrics(yva, tp.predict(Xva)); trm = metrics(ytr, tp.predict(Xtr))
                lbl = f"{fs_name} | TabPFN({_TABPFN_BACKEND})"
                rows.append({"Model": f"TabPFN({_TABPFN_BACKEND})", "Feature_Set": fs_name, "Label": lbl,
                             "Train_R2": trm["R2"], "OOF_CV_R2": np.nan, "Validation_R2": vam["R2"],
                             "Validation_RMSE": vam["RMSE"], "Validation_MAE": vam["MAE"],
                             "Train_minus_Val_gap": trm["R2"] - vam["R2"], "Best_Params": "pretrained"})
                trained[lbl] = {"est": tp, "kind": "tabpfn", "Xtr": Xtr, "ytr": ytr, "Xva": Xva,
                                "yva": yva, "Xte": Xte, "yte": yte, "num": num, "cat": cat}
                print(f"  TabPFN({_TABPFN_BACKEND})  Val R2={vam['R2']:.4f}")
            else:
                print("  TabPFN not available (no token / not installed) — skipped.")
        except Exception as e:
            print(f"  TabPFN skipped: {type(e).__name__}: {str(e).splitlines()[0]}")

    lb = pd.DataFrame(rows)
    if lb.empty: raise RuntimeError("No models finished.")
    lb_val = lb.sort_values("Validation_R2", ascending=False).reset_index(drop=True)
    print("\nLEADERBOARD by Validation_R2:")
    print(lb_val[["Label", "Validation_R2", "OOF_CV_R2", "Train_minus_Val_gap"]].to_string(index=False))

    # ---- blend (TabPFN + XGB) on the best NEW feature set, weight chosen on validation ----
    best_fs = "Winner_PlusNewFeatures"
    xgb_lbl, tp_lbl = f"{best_fs} | XGBoost", f"{best_fs} | TabPFN({_TABPFN_BACKEND})"
    blend = None
    if xgb_lbl in trained and tp_lbl in trained:
        ox, ot = trained[xgb_lbl], trained[tp_lbl]
        pv_x, pv_t = ox["est"].predict(ox["Xva"]), ot["est"].predict(ot["Xva"])
        ws = np.linspace(0, 1, 21)
        r2s = [r2_score(ox["yva"], w * pv_t + (1 - w) * pv_x) for w in ws]
        w = float(ws[int(np.argmax(r2s))])
        print(f"\nBlend weight (TabPFN share) chosen on validation: {w:.2f} | blend Val R2={max(r2s):.4f}")
        blend = (w, ox, ot)

    # ---- RepeatedCV (XGBoost only; TabPFN-cloud is too slow/expensive to re-fit 25x) ----
    print("\nRepeatedCV (5x5) on the dev set (XGBoost per feature set):")
    rcv_rows = []
    for fs_name in build_feature_sets():
        lbl = f"{fs_name} | XGBoost"
        if lbl not in trained: continue
        obj = trained[lbl]
        Xdev = pd.concat([obj["Xtr"], obj["Xva"]]).reset_index(drop=True)
        ydev = pd.concat([obj["ytr"], obj["yva"]]).reset_index(drop=True)
        m, s, mn = repeated_cv(obj["est"], Xdev, ydev, lbl, gdev=gdev)
        rcv_rows.append({"Label": lbl, "RepeatedCV_R2_mean": m, "RepeatedCV_R2_std": s, "RepeatedCV_R2_min": mn})
    rcv = pd.DataFrame(rcv_rows).sort_values("RepeatedCV_R2_mean", ascending=False).reset_index(drop=True)

    # ---- final scoring: XGB (best RepeatedCV), TabPFN and blend on the SAME locked test ----
    final_rows, frames = [], []
    xbest_lbl = rcv.iloc[0]["Label"] if not rcv.empty else lb_val.iloc[0]["Label"]
    ox = trained[xbest_lbl]
    Xdev = pd.concat([ox["Xtr"], ox["Xva"]]).reset_index(drop=True)
    ydev = pd.concat([ox["ytr"], ox["yva"]]).reset_index(drop=True)
    fx = clone(ox["est"]); fx.fit(Xdev, ydev)
    px = fx.predict(ox["Xte"]); mx = metrics(ox["yte"], px)
    final_rows.append({"Final_Model": f"XGBoost [{xbest_lbl}]", **mx})
    frames.append(pd.DataFrame({"Model": "XGBoost", "Measured": np.asarray(ox["yte"], float), "Predicted": px}))
    print(f"\nLocked test XGBoost [{xbest_lbl}]: R2={mx['R2']:.4f} RMSE={mx['RMSE']:.4f}")
    joblib.dump(fx, OUT / "models" / "final_xgboost.joblib")

    if tp_lbl in trained:
        ot = trained[tp_lbl]
        # TabPFN refit on dev (train+val) then locked test once
        tp_dev = fit_tabpfn(pd.concat([ot["Xtr"], ot["Xva"]]).reset_index(drop=True),
                            pd.concat([ot["ytr"], ot["yva"]]).reset_index(drop=True), ot["num"], ot["cat"])
        pt = tp_dev.predict(ot["Xte"]); mt = metrics(ot["yte"], pt)
        final_rows.append({"Final_Model": f"TabPFN({_TABPFN_BACKEND}) [{best_fs}]", **mt})
        frames.append(pd.DataFrame({"Model": "TabPFN", "Measured": np.asarray(ot["yte"], float), "Predicted": pt}))
        print(f"Locked test TabPFN [{best_fs}]: R2={mt['R2']:.4f} RMSE={mt['RMSE']:.4f}")
        if blend is not None and xgb_lbl in trained:
            w = blend[0]
            fx2 = clone(trained[xgb_lbl]["est"]); fx2.fit(Xdev, ydev)
            pb = w * pt + (1 - w) * fx2.predict(trained[xgb_lbl]["Xte"])
            mb = metrics(ot["yte"], pb)
            final_rows.append({"Final_Model": f"Blend w={w:.2f} TabPFN+XGB [{best_fs}]", **mb})
            frames.append(pd.DataFrame({"Model": "Blend", "Measured": np.asarray(ot["yte"], float), "Predicted": pb}))
            print(f"Locked test Blend (w={w:.2f}): R2={mb['R2']:.4f} RMSE={mb['RMSE']:.4f}")

    fm = pd.DataFrame(final_rows)
    pred = pd.concat(frames, ignore_index=True); pred["Residual"] = pred.Predicted - pred.Measured
    print("\nFINAL LOCKED-TEST TABLE (scored once):"); print(fm.to_string(index=False))

    # parity plot for the best final model
    top = fm.sort_values("R2", ascending=False).iloc[0]
    d = pred[pred.Model == ("Blend" if "Blend" in top.Final_Model else
                            "TabPFN" if "TabPFN" in top.Final_Model else "XGBoost")]
    xs = np.array([d.Measured.min(), d.Measured.max()]); slope, b = np.polyfit(d.Measured, d.Predicted, 1)
    plt.figure(figsize=(6.4, 6)); plt.scatter(d.Measured, d.Predicted, alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.plot(xs, xs, "r--", label="1:1"); plt.plot(xs, slope * xs + b, "b-", label="best-fit")
    plt.title(f"Locked test ({top.Final_Model}):\nR2={top.R2:.3f} RMSE={top.RMSE:.3f} {UNITS}")
    plt.xlabel(f"Measured Rut_20k ({UNITS})"); plt.ylabel(f"Predicted ({UNITS})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "figures" / "locked_test_parity.png", dpi=180); plt.close()

    _save_df(lb_val, OUT / "model_comparison.xlsx")
    _save_df(rcv, OUT / "repeated_cv_results.xlsx")
    _save_df(pred, OUT / "final_predictions.xlsx")
    _save_df(fm, OUT / "final_metrics.xlsx")

    old = rcv[rcv.Label.str.startswith("Winner_OldFeatures")]
    new = rcv[rcv.Label.str.startswith("Winner_PlusNewFeatures")]
    gain = (float(new.RepeatedCV_R2_mean.iloc[0]) - float(old.RepeatedCV_R2_mean.iloc[0])) if len(old) and len(new) else np.nan
    lines = ["RUT_20k ENHANCED (NEW DATA) — SUMMARY", "=" * 60,
             f"Best locked-test model: {top.Final_Model}  R2={top.R2:.4f} RMSE={top.RMSE:.4f} MAE={top.MAE:.4f}",
             f"RepeatedCV gain from NEW features (old -> old+new, XGBoost): {gain:+.4f}" if np.isfinite(gain) else "",
             "New features used: " + ", ".join(NEW_RUT),
             "TabPFN backend: " + (_TABPFN_BACKEND or "not available"),
             "Selection by RepeatedCV; the 20% locked test was scored once after the models were fixed."]
    (OUT / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved outputs to {OUT}/")


if __name__ == "__main__":
    main()
