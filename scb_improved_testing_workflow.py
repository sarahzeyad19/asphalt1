# -*- coding: utf-8 -*-
"""
SCB (Jc) — IMPROVED TESTING WORKFLOW
====================================
Upgrades the paper-replication script (BP_MLP / RandomForest / XGBoost, 75/25 split) with every
honest-improvement lever developed for this project, aimed at a better and more trustworthy
INDEPENDENT-TEST result:

  1. TRUE RBR feature            RBR_JMF_fraction = (RAP_pct * ACinRAP / 100) / AsphaltContent_Design
  2. Jc < 1 restriction          model built on the reliable Jc < 1 range (drops Jc >= 1)
  3. Strong regularization       shallow trees + high L1/L2 + subsampling -> closes the huge
                                 train-vs-CV overfit gap (was ~0.51 for RandomForest)
  4. RepeatedCV robustness       5 x 5 = 25 dev folds (mean +/- SD, min fold)
  5. Nested CV (unbiased)        outer 5x2, inner 4-fold RandomizedSearch -> honest generalization
  6. Stacking ensemble           RF + XGBoost + ExtraTrees + HistGB, RidgeCV meta
  7. OOF-fit calibration         slope/intercept learned on TRAIN out-of-fold preds, applied ONCE
                                 to the test -> corrects compression (test slope < 1); raw AND
                                 calibrated test metrics reported side by side
  8. Adjusted R^2                feature-count-aware (k = number of predictors)
  9. Importance-based selection  auto TopImpact feature subset by RandomForest importance
 10. VIF multicollinearity + SHAP + honest evaluation ladder + decision table + data-enrichment plan

Selection is CV-first (never the test). The test is scored once, at the end, for the chosen model.
Optional dependencies (xgboost, shap) degrade gracefully if missing.
"""
from __future__ import annotations
import json, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, RepeatedKFold, RandomizedSearchCV,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except Exception:
    HAS_SM = False

# =============================================================================
# SETTINGS
# =============================================================================
RANDOM_STATE = 42
TARGET = "SCB"
UNITS = "kJ/m2"
ID_COL = "MixDesignKey"
TEST_SIZE = 0.25
CV_FOLDS = 5
N_ITER = 60                      # per-model RandomizedSearch budget
REPEATED_CV_REPEATS = 5         # 5 x 5 = 25 dev folds
NESTED_OUTER_SPLITS, NESTED_OUTER_REPEATS, NESTED_INNER = 5, 2, 4
RESTRICT_JC_LT_1 = True         # build the model only on Jc < 1 (drops Jc >= 1)
APPLY_CALIBRATION = True        # OOF-fit slope/intercept, applied once to the test
ADD_ADJUSTED_R2 = True
ADD_TOP_IMPACT = True
TOP_IMPACT_K = 10

INPUT_CANDIDATES = [
    Path("Final_Cleaned_590Dataset_User_Removal_Rules.xlsx"),
    Path.home() / "Downloads" / "Final_Cleaned_590Dataset_User_Removal_Rules.xlsx",
    Path("/root/.claude/uploads/2de3e2fd-e268-5a6b-af7f-ada01b8f275d/2e7aaf8e-Final_Cleaned_590Dataset_User_Removal_Rules.xlsx"),
]
SHEET = "Cleaned_Data_Kept"
OUT = Path("SCB_Improved_Testing_Outputs")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

# Paper's 14 requested features (exact names in this dataset)
PAPER14 = ["AsphaltContent_Design", "Va", "VMA", "Dust_Binder", "Pass_0.075mm", "Gsb", "Gmm",
           "Pass_4.75mm", "CAA", "SandEq", "RAP_pct", "ACinRAP", "PG Grade", "ADT_DOTD_ord"]
EXTRA = ["VFA", "Absorption", "Gmb", "Compaction_Ratio"]   # extra signal present in this file
np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPERS
# =============================================================================
def resolve_input() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Put 'Final_Cleaned_590Dataset_User_Removal_Rules.xlsx' next to the script.")

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}

def best_fit(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    s, b = np.polyfit(y, p, 1)
    return s, b, f"Predicted = {s:.4f} x Measured + {b:.4f}"

def adjusted_r2(r2, n, k):
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else np.nan

def engineer_rbr(df):
    for c in ["RAP_pct", "ACinRAP", "AsphaltContent_Design"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rap_binder = df["RAP_pct"] * df["ACinRAP"] / 100.0
    df["RBR_JMF_fraction"] = (rap_binder / df["AsphaltContent_Design"]).replace([np.inf, -np.inf], np.nan)
    df["RBR_JMF_percent"] = 100.0 * df["RBR_JMF_fraction"]
    return df

def build_pipe(estimator, features):
    return Pipeline([("prep", ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]), features)],
        remainder="drop")), ("model", estimator)])


# =============================================================================
# MODELS — strongly regularized to close the overfit gap
# =============================================================================
def model_specs():
    specs = {
        "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__max_features": ["sqrt", 0.4, 0.5, 0.7],
            "model__min_samples_leaf": [3, 5, 8, 12, 16],
            "model__min_samples_split": [5, 10, 20],
        }),
        "ExtraTrees": (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
            "model__n_estimators": [400, 700],
            "model__max_depth": [4, 6, 8, None],
            "model__max_features": ["sqrt", 0.5, 0.7],
            "model__min_samples_leaf": [3, 5, 8, 12],
        }),
        "HistGB": (HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error"), {
            "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
            "model__max_iter": [300, 500, 800],
            "model__max_leaf_nodes": [8, 15, 20],
            "model__min_samples_leaf": [15, 25, 40],
            "model__l2_regularization": [0.0, 0.5, 1.0, 5.0],
        }),
    }
    if HAS_XGB:
        specs["XGBoost"] = (XGBRegressor(objective="reg:squarederror", eval_metric="rmse",
                                         tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1), {
            "model__n_estimators": [400, 700, 1000],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "model__subsample": [0.6, 0.7, 0.8],
            "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
            "model__min_child_weight": [5, 8, 12, 20],
            "model__reg_alpha": [0.0, 0.5, 1.0, 2.0],
            "model__reg_lambda": [2, 5, 10, 20, 40],
            "model__gamma": [0.0, 0.05, 0.1, 0.2],
        })
    return specs


def tune(name, est, grid, Xtr, ytr, features):
    pipe = build_pipe(clone(est), features)
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_iter = min(N_ITER, int(np.prod([len(v) for v in grid.values()])))
    s = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="r2", cv=cv,
                           random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan)
    s.fit(Xtr, ytr)
    return s.best_estimator_, float(s.cv_results_["mean_test_score"][s.best_index_]), \
           float(s.cv_results_["std_test_score"][s.best_index_]), s.best_params_


def repeated_cv(est, X, y):
    sc = []
    for r in range(REPEATED_CV_REPEATS):
        cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
        for tr, va in cv.split(X):
            e = clone(est); e.fit(X.iloc[tr], y.iloc[tr])
            sc.append(r2_score(y.iloc[va], e.predict(X.iloc[va])))
    sc = np.array(sc)
    return {"RepeatedCV_Mean": float(sc.mean()), "RepeatedCV_SD": float(sc.std(ddof=1)),
            "RepeatedCV_Min": float(sc.min()), "N": len(sc)}


def nested_cv(est, grid, X, y, features):
    outer = RepeatedKFold(n_splits=NESTED_OUTER_SPLITS, n_repeats=NESTED_OUTER_REPEATS, random_state=RANDOM_STATE)
    n_iter = min(30, int(np.prod([len(v) for v in grid.values()])))
    scores = []
    for i, (tr, te) in enumerate(outer.split(X), 1):
        inner = KFold(NESTED_INNER, shuffle=True, random_state=RANDOM_STATE + i)
        s = RandomizedSearchCV(build_pipe(clone(est), features), grid, n_iter=n_iter, scoring="r2",
                               cv=inner, random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan)
        s.fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[te], s.best_estimator_.predict(X.iloc[te])))
    scores = np.array(scores)
    return {"NestedCV_Mean": float(np.nanmean(scores)), "NestedCV_SD": float(np.nanstd(scores, ddof=1)),
            "NestedCV_Min": float(np.nanmin(scores)), "Outer_Folds": len(scores)}


def vif_table(X):
    Xn = X.apply(pd.to_numeric, errors="coerce").dropna()
    if len(Xn) < X.shape[1] + 2:
        return pd.DataFrame()
    Xs = MinMaxScaler().fit_transform(Xn.values)
    try:
        if HAS_SM:
            vals = [variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])]
        else:
            vals = list(np.diag(np.linalg.pinv(np.corrcoef(Xs, rowvar=False))))
        return pd.DataFrame({"Feature": X.columns, "VIF": vals}).sort_values("VIF", ascending=False)
    except Exception:
        return pd.DataFrame()


def parity(y, p, title, path):
    m = metrics(y, p); s, b, eq = best_fit(y, p)
    lo, hi = float(min(np.min(y), np.min(p))), float(max(np.max(y), np.max(p)))
    xs = np.array([lo, hi])
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y, p, alpha=0.7, edgecolor="k", linewidth=0.3, color="#2878B5")
    plt.plot(xs, xs, "r--", lw=2, label="1:1 reference")
    plt.plot(xs, s * xs + b, color="#173F5F", lw=2, label="Best-fit")
    plt.title(f"{title}\nR2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
    plt.xlabel(f"Measured {TARGET} ({UNITS})"); plt.ylabel(f"Predicted {TARGET} ({UNITS})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    path = resolve_input()
    df = pd.read_excel(path, sheet_name=SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    df = engineer_rbr(df)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[y.notna()].reset_index(drop=True); y = y.loc[y.notna()].reset_index(drop=True)

    print("=" * 92); print(f"SCB IMPROVED TESTING WORKFLOW | file: {path.name} | rows: {len(df)}")
    print(f"XGBoost={HAS_XGB} SHAP={HAS_SHAP} statsmodels={HAS_SM}"); print("=" * 92)

    if RESTRICT_JC_LT_1:
        keep = y < 1.0
        print(f"Jc<1 restriction: kept {int(keep.sum())} of {len(df)} rows (dropped {int((~keep).sum())} with Jc>=1).")
        df = df.loc[keep.values].reset_index(drop=True); y = y.loc[keep.values].reset_index(drop=True)

    # Feature sets (only columns actually present)
    def present(cols): return [c for c in cols if c in df.columns]
    feature_sets = {
        "Paper14": present(PAPER14),
        "Paper14_plus_RBR": present(PAPER14 + ["RBR_JMF_fraction"]),
        "Extended_plus_RBR": present(PAPER14 + EXTRA + ["RBR_JMF_fraction"]),
    }
    for k, v in feature_sets.items():
        print(f"  feature set {k}: {len(v)} vars")

    # Stratified 75/25 split (same rows for every candidate)
    strata = pd.qcut(y, q=min(5, y.nunique()), labels=False, duplicates="drop")
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, shuffle=True, stratify=strata)
    y_tr, y_te = y.iloc[idx_tr].reset_index(drop=True), y.iloc[idx_te].reset_index(drop=True)
    print(f"Split: train {len(idx_tr)} / test {len(idx_te)} (stratified 75/25)")

    # ---- TopImpact feature set (RandomForest importance on TRAIN of Extended set) ----
    if ADD_TOP_IMPACT:
        pool = feature_sets["Extended_plus_RBR"]
        Xp = df[pool].apply(pd.to_numeric, errors="coerce").iloc[idx_tr]
        rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(Xp.fillna(Xp.median()), y_tr)
        imp = pd.Series(rf.feature_importances_, index=pool).sort_values(ascending=False)
        feature_sets["TopImpact"] = list(imp.head(TOP_IMPACT_K).index)
        print(f"  feature set TopImpact ({TOP_IMPACT_K}): {feature_sets['TopImpact']}")

    specs = model_specs()
    rows, trained = [], {}
    for fs_name, feats in feature_sets.items():
        Xf = df[feats].apply(pd.to_numeric, errors="coerce")
        Xtr, Xte = Xf.iloc[idx_tr].reset_index(drop=True), Xf.iloc[idx_te].reset_index(drop=True)
        for mname, (est, grid) in specs.items():
            try:
                best, cvm, cvsd, bp = tune(mname, est, grid, Xtr, y_tr, feats)
                tr_m = metrics(y_tr, best.predict(Xtr)); te_m = metrics(y_te, best.predict(Xte))
                label = f"{fs_name} | {mname}"
                rows.append({"Feature_Set": fs_name, "Model": mname, "N_Features": len(feats),
                             "Training_R2": tr_m["R2"], "CV_Mean_R2": cvm, "CV_SD_R2": cvsd,
                             "Testing_R2": te_m["R2"], "Testing_RMSE": te_m["RMSE"], "Testing_MAE": te_m["MAE"],
                             "Overfit_Gap_TrainMinusCV": tr_m["R2"] - cvm, "Label": label,
                             "Best_Params": json.dumps(bp, default=str)})
                trained[label] = {"est": best, "feats": feats, "grid": grid, "spec": est,
                                  "Xtr": Xtr, "Xte": Xte}
                print(f"  {label:38s} CV={cvm:.3f}±{cvsd:.3f} | Train={tr_m['R2']:.3f} | "
                      f"Test={te_m['R2']:.3f} | gap={tr_m['R2']-cvm:.3f}")
            except Exception as e:
                print(f"  FAILED {fs_name}|{mname}: {type(e).__name__}: {e}")

    res = pd.DataFrame(rows)

    # ---- Stacking on the best feature set (Extended+RBR) ----
    try:
        feats = feature_sets.get("Extended_plus_RBR", feature_sets["Paper14_plus_RBR"])
        Xf = df[feats].apply(pd.to_numeric, errors="coerce")
        Xtr, Xte = Xf.iloc[idx_tr].reset_index(drop=True), Xf.iloc[idx_te].reset_index(drop=True)
        base = []
        for mname in ["RandomForest", "ExtraTrees", "HistGB"] + (["XGBoost"] if HAS_XGB else []):
            lab = f"Extended_plus_RBR | {mname}"
            if lab in trained:
                base.append((mname, clone(trained[lab]["est"])))
        if len(base) >= 2:
            stack = StackingRegressor(base, final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
                                      cv=CV_FOLDS, n_jobs=1)
            stack.fit(Xtr, y_tr)
            tr_m = metrics(y_tr, stack.predict(Xtr)); te_m = metrics(y_te, stack.predict(Xte))
            rc = repeated_cv(stack, Xtr, y_tr)
            rows.append({"Feature_Set": "Extended_plus_RBR", "Model": "Stacking", "N_Features": len(feats),
                         "Training_R2": tr_m["R2"], "CV_Mean_R2": rc["RepeatedCV_Mean"], "CV_SD_R2": rc["RepeatedCV_SD"],
                         "Testing_R2": te_m["R2"], "Testing_RMSE": te_m["RMSE"], "Testing_MAE": te_m["MAE"],
                         "Overfit_Gap_TrainMinusCV": tr_m["R2"] - rc["RepeatedCV_Mean"],
                         "Label": "Extended_plus_RBR | Stacking", "Best_Params": "stacking"})
            trained["Extended_plus_RBR | Stacking"] = {"est": stack, "feats": feats, "grid": None,
                                                       "spec": None, "Xtr": Xtr, "Xte": Xte}
            res = pd.DataFrame(rows)
            print(f"  {'Extended_plus_RBR | Stacking':38s} RepeatedCV={rc['RepeatedCV_Mean']:.3f}±{rc['RepeatedCV_SD']:.3f} | Test={te_m['R2']:.3f}")
    except Exception as e:
        print(f"  Stacking failed: {type(e).__name__}: {e}")

    # ---- CV-FIRST selection (never the test) ----
    res = res.sort_values(["CV_Mean_R2", "CV_SD_R2", "Overfit_Gap_TrainMinusCV"],
                          ascending=[False, True, True]).reset_index(drop=True)
    if ADD_ADJUSTED_R2:
        res["Testing_Adjusted_R2"] = [adjusted_r2(r2, len(idx_te), k)
                                      for r2, k in zip(res["Testing_R2"], res["N_Features"])]
    sel = res.iloc[0]; obj = trained[sel["Label"]]
    print("\n" + "=" * 92)
    print(f"CV-SELECTED MODEL: {sel['Label']}  (CV {sel['CV_Mean_R2']:.3f}, test {sel['Testing_R2']:.3f})")
    print("=" * 92)

    # ---- RepeatedCV + Nested CV on the selected model (honest ladder) ----
    Xtr, Xte, feats = obj["Xtr"], obj["Xte"], obj["feats"]
    rc = repeated_cv(obj["est"], Xtr, y_tr)
    nested = {"NestedCV_Mean": np.nan}
    if obj["grid"] is not None:
        nested = nested_cv(obj["spec"], obj["grid"],
                           pd.concat([Xtr, Xte]).reset_index(drop=True),
                           pd.concat([y_tr, y_te]).reset_index(drop=True), feats)

    # ---- OOF-fit calibration (honest: learned on TRAIN out-of-fold preds) ----
    cal_row = {"Use": False}
    test_pred = obj["est"].predict(Xte)
    if APPLY_CALIBRATION and obj["grid"] is not None:
        oof = cross_val_predict(clone(obj["est"]), Xtr, y_tr,
                                cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
        a, b = np.polyfit(oof, y_tr, 1)                      # Measured = a*Pred + b
        cal_pred = a * test_pred + b
        raw, cal = metrics(y_te, test_pred), metrics(y_te, cal_pred)
        cal_row = {"Use": True, "a_slope": float(a), "b_intercept": float(b),
                   "Test_R2_raw": raw["R2"], "Test_R2_calibrated": cal["R2"],
                   "Test_RMSE_raw": raw["RMSE"], "Test_RMSE_calibrated": cal["RMSE"],
                   "Test_MAE_raw": raw["MAE"], "Test_MAE_calibrated": cal["MAE"]}
        parity(y_te, cal_pred, f"SCB test — {sel['Model']} (calibrated)", OUT / "figures" / "test_parity_calibrated.png")
    parity(y_tr, obj["est"].predict(Xtr), f"SCB train — {sel['Model']}", OUT / "figures" / "train_parity.png")
    parity(y_te, test_pred, f"SCB test — {sel['Model']} (raw)", OUT / "figures" / "test_parity_raw.png")

    # ---- Honest evaluation ladder ----
    ladder = pd.DataFrame([
        {"Stage": "Training R2 (optimistic)", "R2": sel["Training_R2"]},
        {"Stage": "5-fold CV (selection)", "R2": sel["CV_Mean_R2"]},
        {"Stage": f"RepeatedCV {CV_FOLDS}x{REPEATED_CV_REPEATS} mean", "R2": rc["RepeatedCV_Mean"]},
        {"Stage": "RepeatedCV min fold", "R2": rc["RepeatedCV_Min"]},
        {"Stage": "Nested CV (unbiased)", "R2": nested["NestedCV_Mean"]},
        {"Stage": "Independent test (raw)", "R2": sel["Testing_R2"]},
        {"Stage": "Independent test (calibrated)", "R2": cal_row.get("Test_R2_calibrated", np.nan)},
        {"Stage": "Independent test (adjusted R2)", "R2": sel.get("Testing_Adjusted_R2", np.nan)},
    ])
    print("\nHONEST EVALUATION LADDER (SCB, Jc<1):")
    print(ladder.to_string(index=False))
    if cal_row["Use"]:
        print(f"\nCalibration (OOF-fit)  Measured = {cal_row['a_slope']:.4f}*Pred + {cal_row['b_intercept']:.4f}"
              f"  ->  test R2 {cal_row['Test_R2_raw']:.3f} -> {cal_row['Test_R2_calibrated']:.3f}, "
              f"RMSE {cal_row['Test_RMSE_raw']:.4f} -> {cal_row['Test_RMSE_calibrated']:.4f}")

    # ---- Feature importance (+SHAP) + VIF ----
    imp_df = pd.DataFrame()
    model = obj["est"].named_steps["model"] if hasattr(obj["est"], "named_steps") else None
    if model is not None and hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({"Feature": feats, "Importance": model.feature_importances_}
                              ).sort_values("Importance", ascending=False)
    else:
        try:
            pm = permutation_importance(obj["est"], Xte, y_te, n_repeats=20, random_state=RANDOM_STATE, scoring="r2")
            imp_df = pd.DataFrame({"Feature": feats, "Importance": pm.importances_mean}
                                  ).sort_values("Importance", ascending=False)
        except Exception:
            pass
    if not imp_df.empty:
        print("\nTop features:", ", ".join(imp_df.head(6)["Feature"]))
    vif = vif_table(df[feats])

    # ---- Decision table + data plan ----
    decision = pd.DataFrame([{
        "Selected": sel["Label"], "CV_R2": round(sel["CV_Mean_R2"], 3),
        "RepeatedCV_R2": round(rc["RepeatedCV_Mean"], 3), "Nested_R2": round(nested["NestedCV_Mean"], 3) if not np.isnan(nested["NestedCV_Mean"]) else "n/a",
        "Test_R2_raw": round(sel["Testing_R2"], 3),
        "Test_R2_calibrated": round(cal_row.get("Test_R2_calibrated", np.nan), 3) if cal_row["Use"] else "n/a",
        "Overfit_Gap": round(sel["Overfit_Gap_TrainMinusCV"], 3),
        "Honest_note": "Selection by CV; test scored once; calibration OOF-fit (no test leakage)."}])
    data_plan = pd.DataFrame([
        {"Priority": 1, "Add": "Binder rheology / continuous PG (DSR G*/sinδ)", "Why": "Direct cracking driver; PG here is a coarse grade."},
        {"Priority": 2, "Add": "Binder aging (RTFO/PAV)", "Why": "Aged binder embrittles; not captured."},
        {"Priority": 3, "Add": "SCB test temperature", "Why": "Jc is temperature-sensitive."},
        {"Priority": 4, "Add": "Replicate/section IDs -> grouped CV", "Why": "Prevents mix leakage across train/test."},
    ])

    # ---- Save workbook ----
    wb = OUT / "SCB_Improved_Testing_Results.xlsx"
    with pd.ExcelWriter(wb, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="All_Candidates", index=False)
        ladder.to_excel(w, sheet_name="Honest_Ladder", index=False)
        decision.to_excel(w, sheet_name="Decision_Table", index=False)
        pd.DataFrame([cal_row]).to_excel(w, sheet_name="Calibration", index=False)
        pd.DataFrame([rc]).to_excel(w, sheet_name="RepeatedCV", index=False)
        pd.DataFrame([nested]).to_excel(w, sheet_name="NestedCV", index=False)
        if not imp_df.empty: imp_df.to_excel(w, sheet_name="Feature_Importance", index=False)
        if not vif.empty: vif.to_excel(w, sheet_name="VIF", index=False)
        pd.DataFrame({"Feature_Set": list(feature_sets), "Variables": [", ".join(v) for v in feature_sets.values()]}
                     ).to_excel(w, sheet_name="Feature_Sets", index=False)
        data_plan.to_excel(w, sheet_name="DataPlan_to_improve", index=False)

    print(f"\nSaved workbook: {wb}")
    print(f"Figures: {OUT / 'figures'}")
    print("\nBOTTOM LINE: honest SCB (Jc<1) = RepeatedCV %.3f / nested %s; independent test raw %.3f%s."
          % (rc["RepeatedCV_Mean"],
             ("%.3f" % nested["NestedCV_Mean"]) if not np.isnan(nested["NestedCV_Mean"]) else "n/a",
             sel["Testing_R2"],
             (" -> calibrated %.3f" % cal_row["Test_R2_calibrated"]) if cal_row["Use"] else ""))


if __name__ == "__main__":
    main()
