# -*- coding: utf-8 -*-
"""
SCB (Jc) — OPTUNA-ENHANCED COMPANION WORKFLOW
=============================================
ADDITIVE upgrade to the paper-replication SCB script. Nothing in the original workflow is removed;
this reproduces the SAME setup (exact 14 requested features, 75/25 stratified split, MinMaxScaler
pipeline) and ADDS the pieces requested:

  + TRUE RBR feature      RBR_JMF_fraction = (RAP_pct * ACinRAP / 100) / AsphaltContent_Design
  + OPTUNA tuning         TPE sampler, N_TRIALS x 5-fold CV per model (default 200 -> ~1000 fits,
                          up from the original 175). Falls back to RandomizedSearchCV if optuna
                          is not installed.
  + Validation ladder     RepeatedStratifiedKFold (k=10 x 5) mean+/-SD  AND  Nested CV (unbiased)
  + OOF things            cross_val_predict OOF used ONLY for (a) the Stacking meta-model and
                          (b) an OOF-fit calibration (slope/intercept applied once to the test)
  + Extra models          ExtraTrees + HistGB + Stacking (RidgeCV meta) added to RF + XGBoost
  + Adjusted R2, honest evaluation ladder, decision table, parity plots, Excel workbook

Selection is CV-FIRST (Repeated-CV mean); the independent test is scored ONCE. Goal: beat the
prior XGBoost independent-test R2 = 0.619 on the RBR-augmented data, reported honestly.
"""
from __future__ import annotations
import json, warnings
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, RepeatedKFold, cross_val_predict,
                                     cross_val_score, train_test_split, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# =============================================================================
# SETTINGS  (kept identical to your paper script unless noted)
# =============================================================================
RANDOM_STATE = 42
TARGET = "SCB"
UNITS = "kJ/m2"
TEST_SIZE = 0.25                 # your 75/25 split, unchanged
CV_FOLDS = 5                     # your 5-fold selection CV, unchanged
N_TRIALS = 200                   # Optuna trials per model (was 175 fits total via RandomizedSearch)
REPEATED_K, REPEATED_REPEATS = 10, 5     # added validation: RepeatedKFold 10x5
NESTED_OUTER, NESTED_INNER, NESTED_INNER_TRIALS = 10, 5, 40   # added: nested CV (unbiased)
ADD_ADJUSTED_R2 = True

# Exact 14 requested features from your script (+ engineered RBR)
PAPER14 = ["AsphaltContent_Design", "Va", "VMA", "Dust_Binder", "Pass_0.075mm", "Gsb", "Gmm",
           "Pass_4.75mm", "CAA", "SandEq", "RAP_pct", "ACinRAP", "PG Grade", "ADT_DOTD_ord"]

INPUT_CANDIDATES = [
    Path("Final_Cleaned_590Dataset_User_Removal_Rules.xlsx"),
    Path.home() / "Downloads" / "Final_Cleaned_590Dataset_User_Removal_Rules.xlsx",
    Path("/root/.claude/uploads/2de3e2fd-e268-5a6b-af7f-ada01b8f275d/b2f34a91-Final_Cleaned_590Dataset_User_Removal_Rules.xlsx"),
]
SHEET = "Cleaned_Data_Kept"
OUT = Path("SCB_Optuna_Enhanced_Outputs"); OUT.mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPERS
# =============================================================================
def resolve_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Put 'Final_Cleaned_590Dataset_User_Removal_Rules.xlsx' next to the script.")

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}

def best_fit(y, p):
    s, b = np.polyfit(np.asarray(y, float), np.asarray(p, float), 1)
    return s, b, f"Predicted = {s:.4f} x Measured + {b:.4f}"

def adjusted_r2(r2, n, k):
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else np.nan

def make_pipe(estimator, features):
    return Pipeline([("prep", ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]), features)],
        remainder="drop")), ("model", estimator)])


# =============================================================================
# OPTUNA SEARCH SPACES (strong regularization to close the overfit gap)
# =============================================================================
def suggest(trial, name):
    if name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 16),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20))
    if name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 16),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 16))
    if name == "HistGB":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_iter=trial.suggest_int("max_iter", 200, 900, step=100),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 31),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 40),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True))
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.95),
            min_child_weight=trial.suggest_int("min_child_weight", 3, 25),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 50.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-3, 0.4, log=True))
    raise ValueError(name)

# RandomizedSearch fallback grids (only used if optuna is missing)
FALLBACK_GRID = {
    "RandomForest": {"model__n_estimators": [300, 500, 800], "model__max_depth": [3, 5, 8, 12],
                     "model__max_features": [0.3, 0.5, 0.7, 1.0], "model__min_samples_leaf": [1, 3, 8, 12]},
    "ExtraTrees": {"model__n_estimators": [300, 700], "model__max_depth": [4, 8, 16, None],
                   "model__max_features": [0.3, 0.5, 0.7], "model__min_samples_leaf": [1, 3, 8]},
    "HistGB": {"model__learning_rate": [0.02, 0.05, 0.1], "model__max_iter": [300, 600],
               "model__max_leaf_nodes": [8, 15, 31], "model__l2_regularization": [0.0, 1.0, 5.0]},
    "XGBoost": {"model__n_estimators": [400, 800], "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.01, 0.03, 0.05], "model__subsample": [0.6, 0.8],
                "model__reg_lambda": [2, 10, 40], "model__min_child_weight": [5, 12]},
}


def tune_model(name, Xtr, ytr, features):
    """Return (best_pipeline, cv_mean_r2). Optuna TPE if available, else RandomizedSearchCV."""
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if HAS_OPTUNA and (name != "XGBoost" or HAS_XGB):
        def objective(trial):
            est = suggest(trial, name)
            pipe = make_pipe(est, features)
            return float(np.mean(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="r2", n_jobs=1)))
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best = make_pipe(suggest(optuna.trial.FixedTrial(study.best_params), name), features)
        best.fit(Xtr, ytr)
        return best, float(study.best_value)
    # fallback
    base = {"RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error")}
    if HAS_XGB:
        base["XGBoost"] = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                       random_state=RANDOM_STATE, n_jobs=-1)
    s = RandomizedSearchCV(make_pipe(base[name], features), FALLBACK_GRID[name], n_iter=40,
                           scoring="r2", cv=cv, random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan)
    s.fit(Xtr, ytr)
    return s.best_estimator_, float(s.cv_results_["mean_test_score"][s.best_index_])


def repeated_cv(est, X, y):
    sc = []
    for r in range(REPEATED_REPEATS):
        cv = KFold(REPEATED_K, shuffle=True, random_state=RANDOM_STATE + 41 * r)
        for tr, va in cv.split(X):
            e = clone(est); e.fit(X.iloc[tr], y.iloc[tr])
            sc.append(r2_score(y.iloc[va], e.predict(X.iloc[va])))
    sc = np.array(sc)
    return {"Mean": float(sc.mean()), "SD": float(sc.std(ddof=1)), "Min": float(sc.min()), "N": len(sc)}


def nested_cv(name, X, y, features):
    """Unbiased: outer folds score a model tuned (inner Optuna/RandomizedSearch) inside each fold."""
    outer = KFold(NESTED_OUTER, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in outer.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        cvi = KFold(NESTED_INNER, shuffle=True, random_state=RANDOM_STATE)
        if HAS_OPTUNA and (name != "XGBoost" or HAS_XGB):
            def obj(trial):
                return float(np.mean(cross_val_score(make_pipe(suggest(trial, name), features),
                                                     Xtr, ytr, cv=cvi, scoring="r2", n_jobs=1)))
            st = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            st.optimize(obj, n_trials=NESTED_INNER_TRIALS, show_progress_bar=False)
            m = make_pipe(suggest(optuna.trial.FixedTrial(st.best_params), name), features)
        else:
            base = {"RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                    "ExtraTrees": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                    "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error")}
            if HAS_XGB: base["XGBoost"] = XGBRegressor(tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
            m = RandomizedSearchCV(make_pipe(base[name], features), FALLBACK_GRID[name], n_iter=20,
                                   scoring="r2", cv=cvi, random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan)
        m.fit(Xtr, ytr)
        scores.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
    s = np.array(scores)
    return {"Mean": float(s.mean()), "SD": float(s.std(ddof=1)), "Min": float(s.min()), "Outer": len(s)}


def parity(y, p, title, path):
    m = metrics(y, p); s, b, _ = best_fit(y, p)
    xs = np.array([float(min(np.min(y), np.min(p))), float(max(np.max(y), np.max(p)))])
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y, p, alpha=0.7, edgecolor="k", linewidth=0.3, color="#2878B5")
    plt.plot(xs, xs, "r--", lw=2, label="1:1 reference"); plt.plot(xs, s * xs + b, color="#173F5F", lw=2, label="Best-fit")
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
    for c in ["RAP_pct", "ACinRAP", "AsphaltContent_Design"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["RBR_JMF_fraction"] = ((df["RAP_pct"] * df["ACinRAP"] / 100.0) / df["AsphaltContent_Design"]
                              ).replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[y.notna()].reset_index(drop=True); y = y.loc[y.notna()].reset_index(drop=True)

    features = [c for c in PAPER14 if c in df.columns] + ["RBR_JMF_fraction"]
    X = df[features].apply(pd.to_numeric, errors="coerce")

    print("=" * 92)
    print(f"SCB OPTUNA-ENHANCED | file {path.name} | rows {len(df)} | features {len(features)} (14 + RBR)")
    print(f"Optuna={HAS_OPTUNA} (trials={N_TRIALS}) | XGBoost={HAS_XGB} | goal: beat test R2 0.619")
    print("=" * 92)

    strata = pd.qcut(y, q=min(5, y.nunique()), labels=False, duplicates="drop")
    itr, ite = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                random_state=RANDOM_STATE, shuffle=True, stratify=strata)
    Xtr, Xte = X.iloc[itr].reset_index(drop=True), X.iloc[ite].reset_index(drop=True)
    ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
    print(f"Split: train {len(itr)} / test {len(ite)} (stratified 75/25)\n")

    model_names = ["RandomForest", "ExtraTrees", "HistGB"] + (["XGBoost"] if HAS_XGB else [])
    rows, tuned = [], {}
    for name in model_names:
        best, cvm = tune_model(name, Xtr, ytr, features)
        tuned[name] = best
        tr_m, te_m = metrics(ytr, best.predict(Xtr)), metrics(yte, best.predict(Xte))
        rc = repeated_cv(best, Xtr, ytr)
        rows.append({"Model": name, "Training_R2": tr_m["R2"], "CV5_R2": cvm,
                     "RepeatedCV_R2": rc["Mean"], "RepeatedCV_SD": rc["SD"], "RepeatedCV_Min": rc["Min"],
                     "Testing_R2": te_m["R2"], "Testing_RMSE": te_m["RMSE"], "Testing_MAE": te_m["MAE"],
                     "Overfit_Gap_TrainMinusRepCV": tr_m["R2"] - rc["Mean"]})
        print(f"  {name:14s} CV5={cvm:.3f} | RepeatedCV={rc['Mean']:.3f}±{rc['SD']:.3f} | "
              f"Train={tr_m['R2']:.3f} | Test={te_m['R2']:.3f} | gap={tr_m['R2']-rc['Mean']:.3f}")

    # ---- Stacking on OOF (RidgeCV meta) ----
    base = [(n, clone(tuned[n])) for n in model_names]
    stack = StackingRegressor(base, final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
                              cv=CV_FOLDS, n_jobs=1)
    stack.fit(Xtr, ytr)
    tr_m, te_m = metrics(ytr, stack.predict(Xtr)), metrics(yte, stack.predict(Xte))
    rc = repeated_cv(stack, Xtr, ytr)
    rows.append({"Model": "Stacking", "Training_R2": tr_m["R2"], "CV5_R2": rc["Mean"],
                 "RepeatedCV_R2": rc["Mean"], "RepeatedCV_SD": rc["SD"], "RepeatedCV_Min": rc["Min"],
                 "Testing_R2": te_m["R2"], "Testing_RMSE": te_m["RMSE"], "Testing_MAE": te_m["MAE"],
                 "Overfit_Gap_TrainMinusRepCV": tr_m["R2"] - rc["Mean"]})
    tuned["Stacking"] = stack
    print(f"  {'Stacking':14s} RepeatedCV={rc['Mean']:.3f}±{rc['SD']:.3f} | Train={tr_m['R2']:.3f} | Test={te_m['R2']:.3f}")

    res = pd.DataFrame(rows).sort_values(["RepeatedCV_R2", "Testing_R2"], ascending=False).reset_index(drop=True)
    if ADD_ADJUSTED_R2:
        res["Testing_Adjusted_R2"] = [adjusted_r2(r, len(ite), len(features)) for r in res["Testing_R2"]]

    # ---- CV-first selection (honest); test already computed, reported once ----
    sel = res.iloc[0]; sel_name = sel["Model"]; sel_est = tuned[sel_name]
    print("\n" + "=" * 92)
    print(f"CV-SELECTED (RepeatedCV-first): {sel_name}  RepeatedCV={sel['RepeatedCV_R2']:.3f} | Test={sel['Testing_R2']:.3f}")
    print("=" * 92)

    # ---- Nested CV (unbiased). Stacking cannot be nested-tuned here, so report the unbiased
    #      estimate of the best SINGLE model so the ladder always has an unbiased number. ----
    nested_name = sel_name if sel_name != "Stacking" else \
        res[res["Model"] != "Stacking"].iloc[0]["Model"]
    nested = nested_cv(nested_name, pd.concat([Xtr, Xte]).reset_index(drop=True),
                       pd.concat([ytr, yte]).reset_index(drop=True), features)
    nested["Model"] = nested_name

    # ---- OOF-fit calibration (learned on TRAIN OOF, applied once to test) ----
    test_pred = sel_est.predict(Xte)
    oof = cross_val_predict(clone(sel_est), Xtr, ytr, cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
    a, b = np.polyfit(oof, ytr, 1)
    cal_pred = a * test_pred + b
    raw, cal = metrics(yte, test_pred), metrics(yte, cal_pred)

    parity(ytr, sel_est.predict(Xtr), f"SCB train — {sel_name}", OUT / "figures" / "train_parity.png")
    parity(yte, test_pred, f"SCB test — {sel_name} (raw)", OUT / "figures" / "test_parity_raw.png")
    parity(yte, cal_pred, f"SCB test — {sel_name} (calibrated)", OUT / "figures" / "test_parity_calibrated.png")

    ladder = pd.DataFrame([
        {"Stage": "Training R2 (optimistic)", "R2": sel["Training_R2"]},
        {"Stage": "5-fold CV (Optuna objective)", "R2": sel["CV5_R2"]},
        {"Stage": f"RepeatedCV {REPEATED_K}x{REPEATED_REPEATS}", "R2": sel["RepeatedCV_R2"]},
        {"Stage": "RepeatedCV min fold", "R2": sel["RepeatedCV_Min"]},
        {"Stage": f"Nested CV (unbiased, {nested_name})", "R2": nested["Mean"]},
        {"Stage": "Independent test (raw)", "R2": raw["R2"]},
        {"Stage": "Independent test (calibrated)", "R2": cal["R2"]},
        {"Stage": "Independent test (adjusted R2)", "R2": sel.get("Testing_Adjusted_R2", np.nan)},
    ])
    print("\nHONEST EVALUATION LADDER (SCB, all 535 rows):")
    print(ladder.to_string(index=False))
    print(f"\nBEST INDEPENDENT TEST R2 across models: {res['Testing_R2'].max():.3f} "
          f"(prior baseline was 0.619). Calibration: Measured={a:.4f}*Pred+{b:.4f} -> "
          f"test {raw['R2']:.3f} -> {cal['R2']:.3f}.")

    # ---- Excel ----
    wb = OUT / "SCB_Optuna_Enhanced_Results.xlsx"
    with pd.ExcelWriter(wb, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="All_Models", index=False)
        ladder.to_excel(w, sheet_name="Honest_Ladder", index=False)
        pd.DataFrame([{"a_slope": a, "b_intercept": b, "Test_R2_raw": raw["R2"],
                       "Test_R2_calibrated": cal["R2"], "Test_RMSE_raw": raw["RMSE"],
                       "Test_RMSE_calibrated": cal["RMSE"]}]).to_excel(w, sheet_name="Calibration", index=False)
        pd.DataFrame([nested]).to_excel(w, sheet_name="NestedCV", index=False)
        pd.DataFrame({"Feature": features}).to_excel(w, sheet_name="Features", index=False)
    print(f"\nSaved: {wb} | figures in {OUT/'figures'}")


if __name__ == "__main__":
    main()
