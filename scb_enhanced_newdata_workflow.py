# -*- coding: utf-8 -*-
"""
SCB ENHANCED WORKFLOW — CLEANED DATA FILE (New_Data_SCB_LWT_Cleaned_Modeling_Files.xlsx,
sheet SCB_Clean_Modeling, 760 rows)
=======================================================================
Data notes (from the file's own Feature_Removal_Log): Pass4_75mm/Pass0_075mm are replaced by
Grad_No4/Grad_No200; Pba_pct and Gmb_specimen_AC are removed; additive columns are the cleaned
Additive_Type_clean/Additive_Rate_clean; RBR_percent is shipped directly. FLAG_* columns are
row-quality flags and are NEVER used as predictors.

Mirrors the code that got the HIGH SCB result (validation ~0.68: tuned ExtraTrees on the full
range, 70/10/20 stratified split) and ENHANCES it with the NEW calculated features shipped in
the updated data file:

    AFT_micron (asphalt film thickness)  <- strongest new SCB predictor
    Additive_Rate, Pbe_pct, Dust_Pbe_ratio, Gse, Gmb_specimen_AC,
    Grad_No8 / Grad_No50 / Grad_No100 (fine gradation), SurfaceArea_m2kg

Protocol (honest): tune on 70% train with 5-fold stratified CV, check on 10% validation,
RepeatedCV (5x5) on the 80% dev set, score the 20% locked test ONCE at the end.
Feature sets compared: the old winner set vs old+new so the gain from the new data is measured.
"""
from __future__ import annotations
import json, warnings
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
TARGET = "SCB"; ID_COL = "MixDesignKey"
SHEET = "SCB_Clean_Modeling"         # cleaned SCB modeling sheet
FILE_NAME = "New_Data_SCB_LWT_Cleaned_Modeling_Files.xlsx"
TRAIN, VAL, TEST = 0.70, 0.10, 0.20  # the split that produced the 0.68 result
CV_FOLDS, N_TARGET_BINS = 5, 5
N_ITER = 80                          # ExtraTrees tuning budget (the winner model)
REPEATED_REPEATS = 5                 # 5 folds x 5 repeats RepeatedCV
UNIQUE_MIXES = True                  # collapse replicates by MixDesignKey (no replicate leakage)
OUT = Path("SCB_Enhanced_NewData_outputs")
for sub in ["", "figures", "splits", "models"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)

# =============================================================================
# FEATURES: old winner set + the NEW calculated features (chosen by their effect on SCB)
# =============================================================================
# Pass4_75mm -> Grad_No4 and Pass0_075mm -> Grad_No200 (same sieves; per Feature_Removal_Log)
OLD_WINNER = ["ACinRAP", "PG_HighTemp", "SandEq", "Dust_Binder", "VFA", "Grad_No4", "FAA",
              "Absorption", "VMA", "AsphaltContent_Design", "RAP_pct_x_ACinRAP",
              "NMAS (mm)", "Grad_No200", "Va", "Gmm", "CAA", "RBR_JMF_fraction"]
NEW_SCB = ["AFT_micron", "Additive_Rate_clean", "Pbe_pct", "Dust_Pbe_ratio", "Gse",
           "Grad_No8", "Grad_No50", "Grad_No100", "SurfaceArea_m2kg", "MixTemperature_F_clean"]
CATEGORICAL_HINTS = ["MixType", "DesignLev", "Additive_Type_clean", "Has_Additive"]

def build_feature_sets() -> dict:
    return {
        "Winner_OldFeatures": OLD_WINNER,                                  # baseline = the 0.68 set
        "Winner_PlusNewFeatures": OLD_WINNER + NEW_SCB,                    # the enhancement
        "PlusNew_WithAdditiveType": OLD_WINNER + NEW_SCB + ["Additive_Type_clean", "Has_Additive"],
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
    # engineered features (same as the winning code)
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
    print(f"Rows: {len(df)} | {TARGET}: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f} (FULL range)")
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

def split_70_10_20(df, y):
    bins = make_target_bins(y); idx = np.arange(len(y))
    dev, te = train_test_split(idx, test_size=TEST, random_state=RANDOM_STATE, shuffle=True, stratify=bins)
    vfrac = VAL / (TRAIN + VAL)
    trp, vap = train_test_split(np.arange(len(dev)), test_size=vfrac, random_state=RANDOM_STATE,
                                shuffle=True, stratify=bins.iloc[dev].reset_index(drop=True))
    tr, va = dev[trp], dev[vap]
    if ID_COL in df.columns:
        g = df[ID_COL].astype(str).values
        leak = len(set(g[tr]) & set(g[te])) + len(set(g[va]) & set(g[te])) + len(set(g[tr]) & set(g[va]))
        print(f"Split: train {len(tr)} / val {len(va)} / locked-test {len(te)} | mix overlap = {leak} (must be 0)")
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
# MODEL: tuned ExtraTrees — the family that produced the 0.68 validation result
# (same broad grid as the winning scb_70_10_20 code, tuned on train only)
# =============================================================================
ET_GRID = {"model__n_estimators": [400, 800, 1200],
           "model__max_depth": [None, 10, 14, 18],
           "model__min_samples_leaf": [1, 2, 4],
           "model__min_samples_split": [2, 5, 10],
           "model__max_features": ["sqrt", 0.5, 0.7, 1.0]}
RF_GRID = {"model__n_estimators": [400, 800],
           "model__max_depth": [None, 10, 16],
           "model__min_samples_leaf": [1, 2, 4],
           "model__max_features": ["sqrt", 0.5, 0.7]}

def tune(name, est, grid, n_iter, Xtr, ytr, Xva, yva, num, cat):
    pipe = _pipe(clone(est), num, cat)
    space = int(np.prod([len(v) for v in grid.values()]))
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    bins = make_target_bins(ytr)
    s = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, space), scoring="r2",
                           cv=list(cv.split(np.zeros(len(ytr)), bins)), random_state=RANDOM_STATE,
                           n_jobs=1, return_train_score=True, error_score=np.nan)
    s.fit(Xtr, ytr)
    best = s.best_estimator_
    oof = float(s.cv_results_["mean_test_score"][s.best_index_])
    trm, vam = metrics(ytr, best.predict(Xtr)), metrics(yva, best.predict(Xva))
    row = {"Model": name, "Train_R2": trm["R2"], "OOF_CV_R2": oof, "Validation_R2": vam["R2"],
           "Validation_RMSE": vam["RMSE"], "Validation_MAE": vam["MAE"],
           "Train_minus_Val_gap": trm["R2"] - vam["R2"], "Best_Params": json.dumps(s.best_params_, default=str)}
    return row, best

def repeated_cv(est, Xdev, ydev):
    sc = []
    for r in range(REPEATED_REPEATS):
        cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
        for tr, va in cv.split(Xdev):
            e = clone(est); e.fit(Xdev.iloc[tr], ydev.iloc[tr])
            sc.append(r2_score(ydev.iloc[va], e.predict(Xdev.iloc[va])))
    sc = np.array(sc)
    return float(sc.mean()), float(sc.std(ddof=1)), float(sc.min())

# =============================================================================
# MAIN
# =============================================================================
def main():
    df, y = load_data()
    tr, va, te = split_70_10_20(df, y)
    rows, trained = [], {}

    for fs_name, feats in build_feature_sets().items():
        X, num, cat, avail = get_X(df, feats)
        Xtr, Xva, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ytr, yva, yte = y.iloc[tr].reset_index(drop=True), y.iloc[va].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        print(f"\n--- Feature set: {fs_name} ({len(avail)} features) ---")
        for mname, est, grid, ni in [("ExtraTrees", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1), ET_GRID, N_ITER),
                                     ("RandomForest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), RF_GRID, max(20, N_ITER // 2))]:
            try:
                row, best = tune(mname, est, grid, ni, Xtr, ytr, Xva, yva, num, cat)
                row["Feature_Set"] = fs_name; row["Label"] = f"{fs_name} | {mname}"
                rows.append(row)
                trained[row["Label"]] = {"est": best, "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
                                         "Xte": Xte, "yte": yte, "num": num, "cat": cat}
                print(f"  {mname:14s} Val R2={row['Validation_R2']:.4f} | OOF={row['OOF_CV_R2']:.4f} | gap={row['Train_minus_Val_gap']:.3f}")
            except Exception as e:
                print(f"  {mname} FAILED: {type(e).__name__}: {e}")

    lb = pd.DataFrame(rows)
    # honest selection = OOF CV (not the lucky single validation slice)
    lb_oof = lb.sort_values("OOF_CV_R2", ascending=False).reset_index(drop=True)
    print("\nLEADERBOARD by OOF_CV_R2 (honest):")
    print(lb_oof[["Label", "OOF_CV_R2", "Validation_R2", "Train_minus_Val_gap"]].to_string(index=False))

    # RepeatedCV on the top instance of each feature set (ExtraTrees) — dev set only
    print("\nRepeatedCV (5x5) on the dev set:")
    rcv_rows = []
    for fs_name in build_feature_sets():
        cands = lb[(lb.Feature_Set == fs_name) & (lb.Model == "ExtraTrees")]
        if cands.empty: continue
        lbl = cands.iloc[0]["Label"]; obj = trained[lbl]
        Xdev = pd.concat([obj["Xtr"], obj["Xva"]]).reset_index(drop=True)
        ydev = pd.concat([obj["ytr"], obj["yva"]]).reset_index(drop=True)
        m, s, mn = repeated_cv(obj["est"], Xdev, ydev)
        rcv_rows.append({"Label": lbl, "RepeatedCV_R2_mean": m, "RepeatedCV_R2_std": s, "RepeatedCV_R2_min": mn})
        print(f"  {lbl:45s} {m:.4f} +/- {s:.4f} (min {mn:.4f})")
    rcv = pd.DataFrame(rcv_rows).sort_values("RepeatedCV_R2_mean", ascending=False).reset_index(drop=True)

    # final = best RepeatedCV label
    final_label = rcv.iloc[0]["Label"] if not rcv.empty else lb_oof.iloc[0]["Label"]
    obj = trained[final_label]
    print(f"\nFinal model fixed: {final_label}")
    Xdev = pd.concat([obj["Xtr"], obj["Xva"]]).reset_index(drop=True)
    ydev = pd.concat([obj["ytr"], obj["yva"]]).reset_index(drop=True)
    final = clone(obj["est"]); final.fit(Xdev, ydev)

    frames, mrows = [], []
    for ds, Xs, ys in [("Train70", obj["Xtr"], obj["ytr"]), ("Validation10", obj["Xva"], obj["yva"]),
                       ("LockedTest20", obj["Xte"], obj["yte"])]:
        p = final.predict(Xs); m = metrics(ys, p)
        mrows.append({"Dataset": ds, "Rows": len(ys), **m})
        frames.append(pd.DataFrame({"Dataset": ds, "Measured": np.asarray(ys, float), "Predicted": np.asarray(p, float)}))
    fm = pd.DataFrame(mrows); pred = pd.concat(frames, ignore_index=True)
    pred["Residual"] = pred.Predicted - pred.Measured
    print("\nFinal metrics (locked test scored ONCE):"); print(fm.to_string(index=False))

    # plots
    d = pred[pred.Dataset == "LockedTest20"]
    m = metrics(d.Measured, d.Predicted); xs = np.array([d.Measured.min(), d.Measured.max()])
    slope, b = np.polyfit(d.Measured, d.Predicted, 1)
    plt.figure(figsize=(6.4, 6)); plt.scatter(d.Measured, d.Predicted, alpha=0.7, edgecolor="k", linewidth=0.3)
    plt.plot(xs, xs, "r--", label="1:1"); plt.plot(xs, slope * xs + b, "b-", label="best-fit")
    plt.title(f"Locked test: R2={m['R2']:.3f} RMSE={m['RMSE']:.3f}")
    plt.xlabel("Measured SCB Jc"); plt.ylabel("Predicted SCB Jc"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "figures" / "locked_test_parity.png", dpi=180); plt.close()
    try:
        mdl = final.named_steps["model"]
        names = final.named_steps["prep"].get_feature_names_out()
        imp = pd.DataFrame({"Feature": names, "Importance": mdl.feature_importances_}).sort_values("Importance", ascending=False)
        _save_df(imp, OUT / "feature_importance.xlsx")
        ii = imp.head(15).sort_values("Importance")
        plt.figure(figsize=(8, 5)); plt.barh(ii.Feature, ii.Importance, color="#2A9D8F")
        plt.title("Feature importance (final model)"); plt.tight_layout()
        plt.savefig(OUT / "figures" / "feature_importance.png", dpi=180); plt.close()
    except Exception: pass

    _save_df(lb_oof, OUT / "model_comparison.xlsx")
    _save_df(rcv, OUT / "repeated_cv_results.xlsx")
    _save_df(pred, OUT / "final_predictions.xlsx")
    _save_df(fm, OUT / "final_metrics.xlsx")
    joblib.dump(final, OUT / "models" / "final_model.joblib")

    old = rcv[rcv.Label.str.startswith("Winner_OldFeatures")]
    new = rcv[rcv.Label.str.startswith("Winner_PlusNewFeatures")]
    gain = (float(new.RepeatedCV_R2_mean.iloc[0]) - float(old.RepeatedCV_R2_mean.iloc[0])) if len(old) and len(new) else np.nan
    lt = fm[fm.Dataset == "LockedTest20"]
    lines = ["SCB ENHANCED (NEW DATA) — SUMMARY", "=" * 60,
             f"Final model: {final_label}",
             f"Locked-test R2 = {float(lt.R2.iloc[0]):.4f}  RMSE = {float(lt.RMSE.iloc[0]):.4f}  MAE = {float(lt.MAE.iloc[0]):.4f}",
             f"RepeatedCV gain from NEW features (old -> old+new): {gain:+.4f}" if np.isfinite(gain) else "",
             "New features used: " + ", ".join(NEW_SCB),
             "Selection by OOF CV + RepeatedCV (not a lucky single validation slice); the 20%",
             "locked test was scored once after the model was fixed."]
    (OUT / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved outputs to {OUT}/")


if __name__ == "__main__":
    main()
