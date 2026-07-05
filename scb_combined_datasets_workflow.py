# -*- coding: utf-8 -*-
"""
SCB (Jc) — COMBINE OLD + NEW DATASETS  (honest union, de-duplicated)
====================================================================
Uses BOTH SCB files together:
  OLD = SCB_Cleaned_with_RBR.xlsx              (763 rows, has replicates)
  NEW = Final_Cleaned_590Dataset_...xlsx       (535 rows, cleaned subset)

IMPORTANT: 534 of the 535 new rows share a MixDesignKey with the old file — the NEW file is a
cleaned subset of the OLD one. Blindly concatenating would DUPLICATE ~534 mixes and inflate R²
(leakage). So this script builds an HONEST union:
  * harmonizes the different column names in the two files to one schema,
  * engineers true RBR on both,
  * de-duplicates by MixDesignKey, KEEPING the cleaned NEW row where a mix appears in both,
    and adding the OLD-only mixes,
  * runs the models (Optuna-tuned RF / ExtraTrees / HistGB / XGBoost + Stacking) on the combined
    set with the honest ladder (RepeatedCV + Nested CV) and a one-time stratified test.

A "Source" column (new / old_only) is kept so you can see the composition. Set COMBINE_MODE to
"new_only" or "old_only" to reproduce either single-file result for comparison.
"""
from __future__ import annotations
import warnings
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
from sklearn.model_selection import (KFold, RepeatedKFold, cross_val_score,
                                     cross_val_predict, train_test_split, RandomizedSearchCV)
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

# =============================================================================
# SETTINGS
# =============================================================================
RANDOM_STATE = 42
TARGET, UNITS = "SCB", "kJ/m2"
ID_COL = "MixDesignKey"
TEST_SIZE, CV_FOLDS = 0.25, 5
N_TRIALS = 150
REPEATED_K, REPEATED_REPEATS = 10, 5
NESTED_OUTER, NESTED_INNER, NESTED_INNER_TRIALS = 10, 5, 40
COMBINE_MODE = "union_dedup"     # "union_dedup" (both), or "new_only", or "old_only"
RESTRICT_JC_LT_1 = False         # keep all rows by default (set True to drop Jc >= 1)

# 14 canonical features (NEW-file naming) + engineered RBR
PAPER14 = ["AsphaltContent_Design", "Va", "VMA", "Dust_Binder", "Pass_0.075mm", "Gsb", "Gmm",
           "Pass_4.75mm", "CAA", "SandEq", "RAP_pct", "ACinRAP", "PG Grade", "ADT_DOTD_ord"]

# canonical name -> possible column names across the two files
ALIASES = {
    "AsphaltContent_Design": ["AsphaltContent_Design", "AC_Design", "Design AC"],
    "Va": ["Va", "AirVoids", "VTM"],
    "VMA": ["VMA"],
    "Dust_Binder": ["Dust_Binder", "DustBinder", "Dust/Binder"],
    "Pass_0.075mm": ["Pass_0.075mm", "Pass0_075mm", "P0.075"],
    "Gsb": ["Gsb"],
    "Gmm": ["Gmm"],
    "Pass_4.75mm": ["Pass_4.75mm", "Pass4_75mm", "P4.75"],
    "CAA": ["CAA"],
    "SandEq": ["SandEq", "Sand_Equivalent"],
    "RAP_pct": ["RAP_pct", "RAP", "RAP_Percent"],
    "ACinRAP": ["ACinRAP", "AC_in_RAP"],
    "PG Grade": ["PG Grade", "PG_HighTemp", "PGHigh", "PG_Grade"],
    "ADT_DOTD_ord": ["ADT_DOTD_ord", "ADT_ord", "ADT_enc"],
    "SCB": ["SCB", "SCB Jc", "Jc"],
    ID_COL: [ID_COL, "MixDesignKey"],
}

OLD_CANDIDATES = [Path("SCB_Cleaned_with_RBR.xlsx"), Path.home() / "Downloads" / "SCB_Cleaned_with_RBR.xlsx",
                  Path("/tmp/claude-0/-home-user-asphalt1/2de3e2fd-e268-5a6b-af7f-ada01b8f275d/scratchpad/SCB_Cleaned_with_RBR.xlsx")]
NEW_CANDIDATES = [Path("Final_Cleaned_590Dataset_User_Removal_Rules.xlsx"),
                  Path.home() / "Downloads" / "Final_Cleaned_590Dataset_User_Removal_Rules.xlsx",
                  Path("/root/.claude/uploads/2de3e2fd-e268-5a6b-af7f-ada01b8f275d/b2f34a91-Final_Cleaned_590Dataset_User_Removal_Rules.xlsx")]
OUT = Path("SCB_Combined_Datasets_Outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)


def resolve(cands):
    for p in cands:
        if p.exists():
            return p
    return None

def read_best_sheet(path):
    xl = pd.ExcelFile(path)
    for s in xl.sheet_names:
        if "lean" in s.lower():          # Cleaned_With_RBR / Cleaned_Data_Kept
            return pd.read_excel(path, sheet_name=s)
    return pd.read_excel(path, sheet_name=xl.sheet_names[0])

def harmonize(df, source):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    out = {}
    for canon, names in ALIASES.items():
        for nm in names:
            if nm in df.columns:
                out[canon] = df[nm]; break
    h = pd.DataFrame(out)
    for c in ["RAP_pct", "ACinRAP", "AsphaltContent_Design", "SCB"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    # engineer true RBR
    if {"RAP_pct", "ACinRAP", "AsphaltContent_Design"}.issubset(h.columns):
        h["RBR_JMF_fraction"] = ((h["RAP_pct"] * h["ACinRAP"] / 100.0) / h["AsphaltContent_Design"]
                                 ).replace([np.inf, -np.inf], np.nan)
    h["Source"] = source
    return h

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}

def make_pipe(est, feats):
    return Pipeline([("prep", ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]), feats)],
        remainder="drop")), ("model", est)])


def suggest(trial, name):
    if name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 16))
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
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 15, 45),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-2, 20.0, log=True))
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.95),
            min_child_weight=trial.suggest_int("min_child_weight", 5, 30),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 60.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-2, 0.5, log=True))
    raise ValueError(name)

FALLBACK = {
    "RandomForest": {"model__n_estimators": [300, 600], "model__max_depth": [4, 8, 12], "model__min_samples_leaf": [1, 5, 12]},
    "ExtraTrees": {"model__n_estimators": [400, 700], "model__max_depth": [6, 12, None], "model__min_samples_leaf": [1, 5, 12]},
    "HistGB": {"model__learning_rate": [0.03, 0.08], "model__max_iter": [400, 700], "model__max_leaf_nodes": [15, 31]},
    "XGBoost": {"model__n_estimators": [400, 800], "model__max_depth": [3, 4], "model__learning_rate": [0.02, 0.05], "model__reg_lambda": [5, 20]},
}

def tune(name, Xtr, ytr, feats):
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if HAS_OPTUNA and (name != "XGBoost" or HAS_XGB):
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        st.optimize(lambda t: float(np.mean(cross_val_score(make_pipe(suggest(t, name), feats),
                    Xtr, ytr, cv=cv, scoring="r2", n_jobs=1))), n_trials=N_TRIALS, show_progress_bar=False)
        best = make_pipe(suggest(optuna.trial.FixedTrial(st.best_params), name), feats); best.fit(Xtr, ytr)
        return best
    base = {"RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error")}
    if HAS_XGB: base["XGBoost"] = XGBRegressor(tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
    s = RandomizedSearchCV(make_pipe(base[name], feats), FALLBACK[name], n_iter=20, scoring="r2", cv=cv,
                           random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan); s.fit(Xtr, ytr)
    return s.best_estimator_

def repeated_cv(est, X, y):
    sc = []
    for r in range(REPEATED_REPEATS):
        cv = KFold(REPEATED_K, shuffle=True, random_state=RANDOM_STATE + 41 * r)
        for tr, va in cv.split(X):
            e = clone(est); e.fit(X.iloc[tr], y.iloc[tr]); sc.append(r2_score(y.iloc[va], e.predict(X.iloc[va])))
    sc = np.array(sc); return {"Mean": float(sc.mean()), "SD": float(sc.std(ddof=1)), "Min": float(sc.min())}

def nested_cv(name, X, y, feats):
    outer = KFold(NESTED_OUTER, shuffle=True, random_state=RANDOM_STATE); sc = []
    for tr, te in outer.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]; cvi = KFold(NESTED_INNER, shuffle=True, random_state=RANDOM_STATE)
        if HAS_OPTUNA and (name != "XGBoost" or HAS_XGB):
            st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            st.optimize(lambda t: float(np.mean(cross_val_score(make_pipe(suggest(t, name), feats),
                        Xtr, ytr, cv=cvi, scoring="r2", n_jobs=1))), n_trials=NESTED_INNER_TRIALS, show_progress_bar=False)
            m = make_pipe(suggest(optuna.trial.FixedTrial(st.best_params), name), feats)
        else:
            base = {"RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                    "ExtraTrees": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                    "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error")}
            if HAS_XGB: base["XGBoost"] = XGBRegressor(tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
            m = RandomizedSearchCV(make_pipe(base[name], feats), FALLBACK[name], n_iter=15, scoring="r2", cv=cvi,
                                   random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan)
        m.fit(Xtr, ytr); sc.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
    sc = np.array(sc); return {"Mean": float(sc.mean()), "SD": float(sc.std(ddof=1)), "Min": float(sc.min())}

def parity(y, p, title, path):
    m = metrics(y, p); s, b = np.polyfit(np.asarray(y, float), np.asarray(p, float), 1)
    xs = np.array([float(min(np.min(y), np.min(p))), float(max(np.max(y), np.max(p)))])
    plt.figure(figsize=(6.5, 6)); plt.scatter(y, p, alpha=0.7, edgecolor="k", linewidth=0.3, color="#2878B5")
    plt.plot(xs, xs, "r--", lw=2, label="1:1"); plt.plot(xs, s * xs + b, color="#173F5F", lw=2, label="Best-fit")
    plt.title(f"{title}\nR2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
    plt.xlabel(f"Measured {TARGET}"); plt.ylabel(f"Predicted {TARGET}"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# =============================================================================
def main():
    old_p, new_p = resolve(OLD_CANDIDATES), resolve(NEW_CANDIDATES)
    if old_p is None or new_p is None:
        raise FileNotFoundError(f"Need both files. old={old_p} new={new_p}. "
                                "Put SCB_Cleaned_with_RBR.xlsx and Final_Cleaned_590Dataset_User_Removal_Rules.xlsx next to the script.")
    old = harmonize(read_best_sheet(old_p), "old")
    new = harmonize(read_best_sheet(new_p), "new")
    print("=" * 92)
    print(f"OLD {old_p.name}: {len(old)} rows | NEW {new_p.name}: {len(new)} rows")

    # ---- honest combine: dedup by MixDesignKey, prefer the cleaned NEW row ----
    if COMBINE_MODE == "new_only":
        combined = new.copy(); mode = "NEW only"
    elif COMBINE_MODE == "old_only":
        combined = old.copy(); mode = "OLD only"
    else:
        if ID_COL in old.columns and ID_COL in new.columns:
            new_keys = set(new[ID_COL].astype(str))
            old_only = old[~old[ID_COL].astype(str).isin(new_keys)].copy()
            # within the old-only slice, drop duplicate mixes (old file has replicates)
            old_only = old_only.drop_duplicates(subset=[ID_COL])
            combined = pd.concat([new, old_only], ignore_index=True)
            mode = f"UNION dedup by {ID_COL} (NEW {len(new)} + OLD-only {len(old_only)})"
        else:
            combined = pd.concat([new, old], ignore_index=True).drop_duplicates()
            mode = "UNION (no key; row-dedup)"
    print(f"Combine mode: {mode} -> {len(combined)} rows")
    print("Source composition:", combined["Source"].value_counts().to_dict())

    y = pd.to_numeric(combined[TARGET], errors="coerce")
    combined = combined.loc[y.notna()].reset_index(drop=True); y = y.loc[y.notna()].reset_index(drop=True)
    if RESTRICT_JC_LT_1:
        keep = y < 1.0; combined = combined.loc[keep.values].reset_index(drop=True); y = y.loc[keep.values].reset_index(drop=True)
        print(f"Jc<1 restriction -> {len(combined)} rows")

    feats = [c for c in PAPER14 if c in combined.columns] + (["RBR_JMF_fraction"] if "RBR_JMF_fraction" in combined.columns else [])
    X = combined[feats].apply(pd.to_numeric, errors="coerce")
    print(f"Features ({len(feats)}): {feats}"); print("=" * 92)

    strata = pd.qcut(y, q=min(5, y.nunique()), labels=False, duplicates="drop")
    itr, ite = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=strata)
    Xtr, Xte = X.iloc[itr].reset_index(drop=True), X.iloc[ite].reset_index(drop=True)
    ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
    print(f"Split: train {len(itr)} / test {len(ite)} (stratified 75/25)\n")

    names = ["RandomForest", "ExtraTrees", "HistGB"] + (["XGBoost"] if HAS_XGB else [])
    tuned, rows = {}, []
    for nm in names:
        best = tune(nm, Xtr, ytr, feats); tuned[nm] = best
        tr_m, te_m = metrics(ytr, best.predict(Xtr)), metrics(yte, best.predict(Xte)); rc = repeated_cv(best, Xtr, ytr)
        rows.append({"Model": nm, "Train_R2": tr_m["R2"], "RepeatedCV_R2": rc["Mean"], "RepeatedCV_SD": rc["SD"],
                     "Test_R2": te_m["R2"], "Test_RMSE": te_m["RMSE"], "Overfit_Gap": tr_m["R2"] - rc["Mean"]})
        print(f"  {nm:14s} Train={tr_m['R2']:.3f} | RepeatedCV={rc['Mean']:.3f}±{rc['SD']:.3f} | Test={te_m['R2']:.3f} | gap={tr_m['R2']-rc['Mean']:.3f}")

    stack = StackingRegressor([(n, clone(tuned[n])) for n in names], final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]), cv=CV_FOLDS, n_jobs=1)
    stack.fit(Xtr, ytr); rc = repeated_cv(stack, Xtr, ytr)
    tr_m, te_m = metrics(ytr, stack.predict(Xtr)), metrics(yte, stack.predict(Xte))
    rows.append({"Model": "Stacking", "Train_R2": tr_m["R2"], "RepeatedCV_R2": rc["Mean"], "RepeatedCV_SD": rc["SD"],
                 "Test_R2": te_m["R2"], "Test_RMSE": te_m["RMSE"], "Overfit_Gap": tr_m["R2"] - rc["Mean"]}); tuned["Stacking"] = stack
    print(f"  {'Stacking':14s} Train={tr_m['R2']:.3f} | RepeatedCV={rc['Mean']:.3f}±{rc['SD']:.3f} | Test={te_m['R2']:.3f}")

    res = pd.DataFrame(rows).sort_values("RepeatedCV_R2", ascending=False).reset_index(drop=True)
    sel = res.iloc[0]["Model"]; est = tuned[sel]
    print("\n" + "=" * 92); print(f"CV-SELECTED: {sel} | RepeatedCV={res.iloc[0]['RepeatedCV_R2']:.3f} | Test={res.iloc[0]['Test_R2']:.3f}"); print("=" * 92)

    nested_name = sel if sel != "Stacking" else res[res.Model != "Stacking"].iloc[0]["Model"]
    nested = nested_cv(nested_name, pd.concat([Xtr, Xte]).reset_index(drop=True), pd.concat([ytr, yte]).reset_index(drop=True), feats)

    test_pred = est.predict(Xte)
    oof = cross_val_predict(clone(est), Xtr, ytr, cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
    a, b = np.polyfit(oof, ytr, 1); cal = metrics(yte, a * test_pred + b)
    parity(ytr, est.predict(Xtr), f"SCB combined train — {sel}", OUT / "figures" / "train_parity.png")
    parity(yte, test_pred, f"SCB combined test — {sel} (raw)", OUT / "figures" / "test_parity_raw.png")

    ladder = pd.DataFrame([
        {"Stage": "Training (optimistic)", "R2": res.iloc[0]["Train_R2"]},
        {"Stage": f"RepeatedCV {REPEATED_K}x{REPEATED_REPEATS}", "R2": res.iloc[0]["RepeatedCV_R2"]},
        {"Stage": f"Nested CV ({nested_name})", "R2": nested["Mean"]},
        {"Stage": "Independent test (raw)", "R2": res.iloc[0]["Test_R2"]},
        {"Stage": "Independent test (calibrated)", "R2": cal["R2"]},
    ])
    print("\nHONEST LADDER (combined data):"); print(ladder.to_string(index=False))
    print(f"\nBest test across models: {res['Test_R2'].max():.3f}")

    with pd.ExcelWriter(OUT / "SCB_Combined_Results.xlsx", engine="openpyxl") as w:
        res.to_excel(w, sheet_name="Models", index=False)
        ladder.to_excel(w, sheet_name="Honest_Ladder", index=False)
        combined["Source"].value_counts().rename_axis("Source").reset_index(name="Rows").to_excel(w, sheet_name="Source_Composition", index=False)
        pd.DataFrame({"Feature": feats}).to_excel(w, sheet_name="Features", index=False)
    print(f"\nSaved: {OUT/'SCB_Combined_Results.xlsx'} | Combine mode: {mode}")


if __name__ == "__main__":
    main()
