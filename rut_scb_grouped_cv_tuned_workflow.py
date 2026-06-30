"""
====================================================================================================
RUT_20K AND SCB — GROUPED CROSS-VALIDATION TUNED MODELING WORKFLOW
====================================================================================================
Predicts two asphalt mixture performance targets (Rut_20k, SCB) from routine JMF variables using
grouped cross-validation by MixDesignKey (so repeated/replicate tests of the same mix never leak
across folds). All preprocessing + tuning happen inside leakage-safe sklearn Pipelines.

Outputs:
  - Rut_SCB_Tuned_Model_Results.xlsx  (multi-sheet results workbook)
  - best_Rut_20k_model.joblib / best_SCB_model.joblib  (best pipeline per target)
  - saved_models/  (every trained model pipeline, both targets)
  - figures/  (OOF measured-vs-predicted, residuals, feature importance)

Note: GroupKFold has no random_state parameter in scikit-learn (group assignment is deterministic
given the groups array); RANDOM_STATE still seeds every model and RandomizedSearchCV.

Note: these are GROUPED CROSS-VALIDATION results, not a locked external test-set evaluation. No
row from the dataset is held out from all modeling; the grouping only guarantees a given mix's
replicate rows never span train and validation within a fold.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LIGHTGBM = False


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Tuning. "random" = RandomizedSearchCV (fast, recommended). "grid" = GridSearchCV (exhaustive,
# can be slow for the boosting models — a warning is printed if selected).
TUNER = "random"          # "random" or "grid"
N_ITER = 40                # RandomizedSearchCV iterations (capped per-model to its grid size)
CV_FOLDS = 5
SCORING = "r2"
N_JOBS = -1
QUICK_SMOKE_TEST = False   # True = tiny n_iter / fewer models, for a fast end-to-end check

if QUICK_SMOKE_TEST:
    N_ITER = 4
    CV_FOLDS = 3

# ---- File locations ----
HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
SHEET_NAME = "Cleaned_Dataset"
SHEET_NAME_FALLBACKS = ["Cleaned_Dataset", "Cleaned_With_RBR", "Cleaned_Data_Kept", "Sheet1"]

RUT_FILE = DOWNLOADS / "RUT_ModelReady_Cleaned_Dataset.xlsx"
SCB_FILE = DOWNLOADS / "SCB_ModelReady_Cleaned_Dataset.xlsx"
# Extra filenames to try automatically if the exact name above isn't found (e.g. files from an
# earlier stage of this project, or Excel's "(1)" duplicate-download suffix).
RUT_FILE_FALLBACKS = [
    "RUT_ModelReady_Cleaned_Dataset (1).xlsx",
    "Rutting_Cleaned_with_RBR.xlsx",
    "Rutting_Cleaned_SpecBased.xlsx",
]
SCB_FILE_FALLBACKS = [
    "SCB_ModelReady_Cleaned_Dataset (1).xlsx",
    "SCB_Cleaned_with_RBR.xlsx",
]

OUTPUT_DIR = DOWNLOADS / "Rut_SCB_Tuned_Modeling_Outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "saved_models"
EXCEL_PATH = OUTPUT_DIR / "Rut_SCB_Tuned_Model_Results.xlsx"

for d in [OUTPUT_DIR, FIG_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ID_COL = "MixDesignKey"
RUT_TARGET = "Rut_20k"
SCB_TARGET = "SCB"

# Columns that must NEVER be used as predictors, even if accidentally listed in a feature set.
DROP_ALWAYS = [
    "MixDesignKey", "Rut_20k", "SCB", "LWT_Record_ID", "SCB_Record_ID",
    "LWT_LastUpdated", "SCB_LastUpdated", "Aggregate_components_used", "Gmb_specimen_AC",
]

# ---- Feature sets (exactly as specified) ----
RUT_FEATURES = [
    "ADT_ordinal", "PG_HighTemp", "NMAS (mm)", "MixType", "DesignLev",
    "RAP_pct", "ACinRAP", "Pass4_75mm", "Pass0_075mm", "Va", "VMA", "VFA",
    "Dust_Binder", "Gmm", "Gmb", "SandEq", "FAA", "Absorption", "AsphaltContent_Design",
]
RUT_FEATURES = [f for f in RUT_FEATURES if f not in DROP_ALWAYS]

SCB_FEATURES = [
    "AsphaltContent_Design", "Va", "VMA", "VFA", "Dust_Binder", "Pass0_075mm",
    "Pass4_75mm", "Gsb", "Gmm", "Gmb", "CAA", "FAA", "SandEq", "RAP_pct",
    "ACinRAP", "PG_HighTemp", "ADT_ordinal", "NMAS (mm)", "MixType", "DesignLev",
]
SCB_FEATURES = [f for f in SCB_FEATURES if f not in DROP_ALWAYS]

# ---- Variable treatment ----
NUMERICAL_COLS_MASTER = [
    "Va", "VMA", "VFA", "Gmb", "Gmm", "Gsb", "AsphaltContent_Design", "RAP_pct",
    "ACinRAP", "PG_HighTemp", "NMAS (mm)", "Pass4_75mm", "Pass0_075mm", "Dust_Binder",
    "SandEq", "CAA", "FAA", "Absorption", "ADT_ordinal",
]
CATEGORICAL_COLS_MASTER = ["MixType", "DesignLev"]

# Tree models for which we extract / plot feature importance.
TREE_MODELS = ["RandomForest", "ExtraTrees", "XGBoost", "CatBoost", "LightGBM"]

# Overfit-gap warning threshold for the final printed recommendation.
OVERFIT_GAP_WARN = 0.15


# =============================================================================
# DATA LOADING + VALIDATION
# =============================================================================

def resolve_file_path(primary: Path, fallback_names: List[str]) -> Path:
    """Find the data file: try the exact primary path, then each fallback filename in
    Downloads, then anywhere matching nearby (script directory, current working directory).
    Raises a clear error listing the .xlsx files actually present in Downloads if none match,
    so a wrong/renamed file is a one-glance fix instead of a guessing game."""
    candidates = [primary] + [primary.parent / name for name in fallback_names]
    try:
        script_dir = Path(__file__).resolve().parent
        candidates += [script_dir / c.name for c in list(candidates)]
    except Exception:
        pass
    candidates += [Path.cwd() / c.name for c in list(candidates)]
    seen = set()
    for c in candidates:
        if str(c) in seen:
            continue
        seen.add(str(c))
        if c.exists():
            if c != primary:
                print(f"Note: using fallback file {c} (primary name {primary.name} not found).")
            return c

    present = sorted(p.name for p in primary.parent.glob("*.xlsx")) if primary.parent.exists() else []
    raise FileNotFoundError(
        f"Could not find {primary.name} (or any fallback name) in {primary.parent}.\n"
        f"  Tried: {[c.name for c in candidates[:8]]}\n"
        f"  .xlsx files actually present in {primary.parent}: {present or '(none found)'}\n"
        f"  Fix: rename your file to {primary.name!r} and place it in {primary.parent}, "
        f"or add its current name to the *_FILE_FALLBACKS list at the top of this script."
    )


def load_dataset(path: Path, fallback_names: Optional[List[str]] = None,
                  sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Resolve the file (with fallback names), load its modeling sheet (with sheet-name
    fallback), and normalise column names."""
    resolved = resolve_file_path(path, fallback_names or [])
    xls = pd.ExcelFile(resolved)
    chosen_sheet = sheet_name if sheet_name in xls.sheet_names else None
    if chosen_sheet is None:
        for candidate in SHEET_NAME_FALLBACKS:
            if candidate in xls.sheet_names:
                chosen_sheet = candidate
                break
    if chosen_sheet is None:
        chosen_sheet = xls.sheet_names[0]
    if chosen_sheet != sheet_name:
        print(f"Note: sheet {sheet_name!r} not found in {resolved.name}; using {chosen_sheet!r} "
              f"(available sheets: {xls.sheet_names}).")
    df = pd.read_excel(resolved, sheet_name=chosen_sheet)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded: {resolved}  (sheet={chosen_sheet!r}, rows={len(df)}, cols={df.shape[1]})")
    return df


def check_columns(df: pd.DataFrame, target: str, features: List[str], id_col: str = ID_COL) -> None:
    """Verify the target, every requested predictor, and the group column all exist."""
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found.\n  Available columns: {list(df.columns)}")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(
            f"Missing predictor columns for target {target!r}: {missing}\n"
            f"  Available columns in the file: {list(df.columns)}"
        )
    if id_col not in df.columns:
        raise KeyError(f"Group column {id_col!r} not found.\n  Available columns: {list(df.columns)}")


def assert_no_missing_values(df: pd.DataFrame, target: str, features: List[str], id_col: str = ID_COL) -> None:
    """Hard-fail if the target, any predictor, or the group column has missing values.
    The workflow does not silently impute — the Cleaned_Dataset sheet is expected to already
    be complete for these columns."""
    cols = [target, id_col] + features
    n_missing = df[cols].isna().sum()
    bad = n_missing[n_missing > 0]
    if not bad.empty:
        raise ValueError(
            f"Missing values found in required columns for target {target!r}:\n{bad.to_string()}\n"
            "Clean these values before running the workflow (no implicit imputation is performed)."
        )


def _parse_adt_to_ordinal(value: Any) -> float:
    """Convert a raw ADT value (numeric, or text like '3500 - 7000' / 'Low'/'Medium'/'High')
    into a single numeric ordinal. Used only as a fallback when the file has a raw 'ADT'
    column but not a pre-built 'ADT_ordinal' column."""
    import re as _re
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().lower().replace(",", "")
    nums = [float(x) for x in _re.findall(r"\d+(?:\.\d+)?", s)]
    if len(nums) >= 2:
        return float(np.mean(nums[:2]))
    if len(nums) == 1:
        return nums[0]
    if "low" in s:
        return 1.0
    if "medium" in s or "med" in s:
        return 2.0
    if "high" in s:
        return 3.0
    return np.nan


def derive_adt_ordinal_if_missing(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """If a script's feature list needs 'ADT_ordinal' but the file only has a raw 'ADT'
    column (e.g. older cleaned files), derive it automatically instead of failing."""
    if "ADT_ordinal" in features and "ADT_ordinal" not in df.columns:
        candidates = [c for c in ["ADT_DOTD_ord", "ADT_ord", "ADT"] if c in df.columns]
        if candidates:
            src = candidates[0]
            df = df.copy()
            df["ADT_ordinal"] = df[src].apply(_parse_adt_to_ordinal)
            print(f"Note: derived 'ADT_ordinal' from raw column {src!r} "
                  "('ADT_ordinal' was not present in the file).")
    return df


def coerce_numeric(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
    """Coerce numeric-typed predictor columns to float so the assert-no-missing check also
    catches any non-numeric/garbage values that would otherwise pass silently as text."""
    df = df.copy()
    for c in numerical_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def print_dataset_summary(df: pd.DataFrame, target: str, label: str) -> Dict[str, Any]:
    """Print and return basic dataset diagnostics: size, unique mixes, target stats."""
    y = pd.to_numeric(df[target], errors="coerce")
    n_mixes = df[ID_COL].nunique()
    summary = {
        "Target": label,
        "Rows": int(len(df)),
        "Unique_Mixes": int(n_mixes),
        "Avg_Replicates_per_Mix": round(len(df) / max(n_mixes, 1), 2),
        "Target_Mean": float(y.mean()),
        "Target_SD": float(y.std()),
        "Target_Min": float(y.min()),
        "Target_Max": float(y.max()),
    }
    print("\n" + "=" * 100)
    print(f"Dataset summary — {label}")
    print("=" * 100)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


# =============================================================================
# PREPROCESSING
# =============================================================================

def build_preprocessor(numerical: List[str], categorical: List[str]) -> ColumnTransformer:
    """Numerical -> MinMaxScaler; Categorical -> OneHotEncoder(handle_unknown='ignore').
    Wrapped inside a Pipeline downstream so fitting always happens INSIDE each CV fold
    (no leakage of scaling parameters or category levels across folds)."""
    transformers = []
    if numerical:
        transformers.append(("num", MinMaxScaler(), numerical))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Clean post-encoding feature names (numeric columns keep their name; one-hot columns
    become '<col>_<category>')."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"x{i}" for i in range(preprocessor.transform.__self__.n_features_in_)]


# =============================================================================
# MODELS + HYPER-PARAMETER SEARCH SPACES
# =============================================================================

def get_models_and_param_spaces() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """Return {model_name: (estimator, param_distributions)}. Optional libraries are skipped
    gracefully with an install hint if not present. Hyper-parameter spaces are intentionally
    moderate in size so RandomizedSearchCV stays efficient; GridSearchCV will use the same
    grids (set TUNER='grid' only if you are prepared for the longer runtime)."""
    models: Dict[str, Tuple[Any, Dict[str, list]]] = {}

    models["RandomForest"] = (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 6, 10, 16],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", 0.5, 0.8],
        },
    )

    models["ExtraTrees"] = (
        ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 8, 16, 24],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", 0.5, 0.8],
        },
    )

    if HAS_XGBOOST:
        models["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=1,
            ),
            {
                "n_estimators": [300, 600, 900],
                "max_depth": [2, 3, 4, 6],
                "learning_rate": [0.01, 0.03, 0.05, 0.10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0.0, 0.5, 1.0, 2.0],
                "reg_lambda": [1.0, 5.0, 10.0, 30.0],
            },
        )
    else:
        print("XGBoost not installed. Skipping XGBoost. Install with: pip install xgboost")

    if HAS_CATBOOST:
        models["CatBoost"] = (
            CatBoostRegressor(
                loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0, thread_count=1,
            ),
            {
                "iterations": [300, 600, 900],
                "depth": [4, 6, 8],
                "learning_rate": [0.02, 0.05, 0.10],
                "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
            },
        )
    else:
        print("CatBoost not installed. Skipping CatBoost. Install with: pip install catboost")

    if HAS_LIGHTGBM:
        models["LightGBM"] = (
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
            {
                "n_estimators": [300, 600, 900],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.01, 0.03, 0.05, 0.10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0.0, 0.5, 1.0],
                "reg_lambda": [0.0, 1.0, 5.0],
            },
        )
    else:
        print("LightGBM not installed. Skipping LightGBM. Install with: pip install lightgbm")

    models["HistGradientBoosting"] = (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "max_iter": [200, 400, 600],
            "max_depth": [None, 4, 6, 8],
            "learning_rate": [0.02, 0.05, 0.10],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
    )

    models["SVR_RBF"] = (
        SVR(kernel="rbf"),
        {
            "C": [0.1, 1, 10, 50],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "epsilon": [0.01, 0.05, 0.1, 0.2],
        },
    )

    models["KNN"] = (
        KNeighborsRegressor(n_jobs=1),
        {
            "n_neighbors": [3, 5, 7, 10, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    )

    return models


def _param_space_size(param_dist: Dict[str, list]) -> int:
    size = 1
    for v in param_dist.values():
        size *= len(v)
    return size


def make_search(pipe: Pipeline, param_dist: Dict[str, list], cv) -> Any:
    """RandomizedSearchCV or GridSearchCV depending on TUNER, with n_iter capped to the
    actual grid size (avoids sklearn's duplicate-sampling warning on small grids)."""
    if TUNER == "grid":
        return GridSearchCV(
            estimator=pipe, param_grid=param_dist, scoring=SCORING, cv=cv,
            n_jobs=N_JOBS, refit=True,
        )
    n_iter_eff = min(N_ITER, _param_space_size(param_dist))
    return RandomizedSearchCV(
        estimator=pipe, param_distributions=param_dist, n_iter=n_iter_eff,
        scoring=SCORING, cv=cv, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        refit=True, verbose=0,
    )


# =============================================================================
# GROUPED MODELING
# =============================================================================

def run_grouped_modeling(
    df: pd.DataFrame, target: str, features: List[str], target_label: str,
) -> Dict[str, Any]:
    """Train + tune every available model for one target using GroupKFold(MixDesignKey),
    compute true pooled out-of-fold metrics, training metrics, and the overfit gap.
    Returns a dict with: results (leaderboard), best_params, oof (per-model OOF predictions),
    models (fitted pipelines), importance (tree-model feature importance)."""
    numerical = [c for c in NUMERICAL_COLS_MASTER if c in features]
    categorical = [c for c in CATEGORICAL_COLS_MASTER if c in features]

    X = df[features].reset_index(drop=True)
    y = pd.to_numeric(df[target], errors="coerce").reset_index(drop=True)
    groups = df[ID_COL].reset_index(drop=True)

    # GroupKFold has no random_state (deterministic group assignment); RANDOM_STATE governs
    # the model seeds and the RandomizedSearchCV sampling instead.
    cv = GroupKFold(n_splits=CV_FOLDS)
    preprocessor = build_preprocessor(numerical, categorical)
    model_specs = get_models_and_param_spaces()

    if TUNER == "grid":
        print("WARNING: TUNER='grid' uses GridSearchCV (exhaustive). This can be slow for the "
              "boosting models' full parameter grids.")

    results_rows: List[Dict[str, Any]] = []
    best_params_rows: List[Dict[str, Any]] = []
    oof_frames: List[pd.DataFrame] = []
    fitted_pipelines: Dict[str, Pipeline] = {}
    importance_frames: List[pd.DataFrame] = []

    print("\n" + "=" * 100)
    print(f"Grouped-CV modeling — {target_label} ({len(model_specs)} models, "
          f"{CV_FOLDS}-fold GroupKFold by {ID_COL})")
    print("=" * 100)

    for name, (estimator, param_dist) in model_specs.items():
        t0 = time.time()
        pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
        prefixed = {f"model__{k}": v for k, v in param_dist.items()}
        search = make_search(pipe, prefixed, cv)
        try:
            search.fit(X, y, groups=groups)
        except Exception as e:
            print(f"  {name}: FAILED — {type(e).__name__}: {e}")
            continue

        best_est = search.best_estimator_
        best_idx = search.best_index_
        cv_r2_mean = float(search.cv_results_["mean_test_score"][best_idx])
        cv_r2_sd = float(search.cv_results_["std_test_score"][best_idx])

        # ---- True pooled out-of-fold predictions using the best hyper-parameters ----
        # Re-running the SAME GroupKFold(deterministic) split and refitting per fold gives
        # honest OOF predictions for every row exactly once.
        oof = np.full(len(y), np.nan, dtype=float)
        for tr_idx, va_idx in cv.split(X, y, groups):
            fold_est = clone(best_est)
            fold_est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[va_idx] = fold_est.predict(X.iloc[va_idx])
        oof_r2 = r2_score(y, oof)
        oof_rmse = float(np.sqrt(mean_squared_error(y, oof)))
        oof_mae = float(mean_absolute_error(y, oof))

        # ---- Training metrics (best estimator refit on the FULL dataset) ----
        best_est.fit(X, y)
        train_pred = best_est.predict(X)
        train_r2 = float(r2_score(y, train_pred))
        overfit_gap = train_r2 - oof_r2

        elapsed = time.time() - t0
        print(f"  {name:<22} OOF R2={oof_r2:.4f}  RMSE={oof_rmse:.4f}  MAE={oof_mae:.4f}  "
              f"Train R2={train_r2:.4f}  Gap={overfit_gap:.4f}  ({elapsed:.1f}s)")

        results_rows.append({
            "Target": target_label, "Model": name,
            "CV_R2_Mean": cv_r2_mean, "CV_R2_SD": cv_r2_sd,
            "OOF_R2": oof_r2, "OOF_RMSE": oof_rmse, "OOF_MAE": oof_mae,
            "Train_R2": train_r2, "Overfit_Gap": overfit_gap,
            "N_Predictors": len(features), "Tuning_Seconds": round(elapsed, 1),
        })
        best_params_rows.append({
            "Target": target_label, "Model": name,
            "Best_Params": json.dumps(search.best_params_, default=str),
            "CV_R2_Mean": cv_r2_mean, "CV_R2_SD": cv_r2_sd,
        })

        oof_df = pd.DataFrame({
            ID_COL: groups.values, "Target": target_label, "Model": name,
            "Measured": y.values, "Predicted": oof,
        })
        oof_df["Residual"] = oof_df["Predicted"] - oof_df["Measured"]
        oof_frames.append(oof_df)

        fitted_pipelines[name] = best_est
        joblib.dump(best_est, MODELS_DIR / f"{target_label}_{name}_pipeline.joblib")

        # ---- Feature importance for tree models ----
        if name in TREE_MODELS:
            try:
                fi_names = get_feature_names(best_est.named_steps["preprocess"])
                fi_values = best_est.named_steps["model"].feature_importances_
                fi_df = pd.DataFrame({"Feature": fi_names, "Importance": fi_values})
                fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
                fi_df.insert(0, "Model", name)
                fi_df.insert(0, "Target", target_label)
                importance_frames.append(fi_df)
            except Exception as e:
                print(f"    Feature importance extraction failed for {name}: {type(e).__name__}: {e}")

    results_df = pd.DataFrame(results_rows)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["OOF_R2", "OOF_RMSE", "Overfit_Gap"], ascending=[False, True, True]
        ).reset_index(drop=True)
        results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    return {
        "results": results_df,
        "best_params": pd.DataFrame(best_params_rows),
        "oof": pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame(),
        "models": fitted_pipelines,
        "importance": pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
        "numerical": numerical, "categorical": categorical,
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_oof_predictions(y_true, y_pred, target_label: str, model_name: str, save_path: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=22)
    plt.plot([mn, mx], [mn, mx], "--", linewidth=2, color="black", label="Ideal 1:1")
    plt.xlabel(f"Measured {target_label}")
    plt.ylabel(f"Predicted {target_label} (out-of-fold)")
    plt.title(f"{target_label} — {model_name}\nGrouped-CV OOF: R2={r2:.3f}, RMSE={rmse:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, target_label: str, model_name: str, save_path: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residual, alpha=0.6, s=22)
    plt.axhline(0, linestyle="--", linewidth=2, color="black")
    plt.xlabel(f"Predicted {target_label} (out-of-fold)")
    plt.ylabel("Residual: predicted - measured")
    plt.title(f"{target_label} — {model_name} grouped-CV residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, target_label: str, model_name: str, save_path: Path,
                             top_n: int = 15) -> None:
    if fi_df.empty:
        return
    d = fi_df.sort_values("Importance", ascending=True).tail(top_n)
    plt.figure(figsize=(8, max(4, 0.35 * len(d))))
    plt.barh(d["Feature"], d["Importance"], color="#4f81bd")
    plt.xlabel("Feature importance")
    plt.title(f"{target_label} — {model_name} feature importance (top {top_n})")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


# =============================================================================
# EXCEL OUTPUT
# =============================================================================

def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(text))


def save_results_to_excel(
    dataset_overview: pd.DataFrame,
    rut_out: Dict[str, Any], scb_out: Dict[str, Any],
    recommendation_df: pd.DataFrame, methodology_df: pd.DataFrame,
    path: Path,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dataset_overview.to_excel(writer, sheet_name="Dataset_Overview", index=False)
        rut_out["results"].to_excel(writer, sheet_name="Rut_Model_Results", index=False)
        scb_out["results"].to_excel(writer, sheet_name="SCB_Model_Results", index=False)

        best_params = pd.concat(
            [rut_out["best_params"], scb_out["best_params"]], ignore_index=True
        )
        best_params.to_excel(writer, sheet_name="Best_Hyperparameters", index=False)

        rut_out["oof"].to_excel(writer, sheet_name="Rut_OOF_Predictions", index=False)
        scb_out["oof"].to_excel(writer, sheet_name="SCB_OOF_Predictions", index=False)

        importance = pd.concat(
            [rut_out["importance"], scb_out["importance"]], ignore_index=True
        )
        if not importance.empty:
            importance.to_excel(writer, sheet_name="Feature_Importance", index=False)

        recommendation_df.to_excel(writer, sheet_name="Model_Recommendation", index=False)
        methodology_df.to_excel(writer, sheet_name="Methodology", index=False)

    print(f"\nResults workbook saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def process_target(file_path: Path, target: str, features: List[str], target_label: str,
                   fallback_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load, validate, and run the full grouped-modeling workflow for one target."""
    df = load_dataset(file_path, fallback_names)
    df = derive_adt_ordinal_if_missing(df, features)
    check_columns(df, target, features)
    numerical_in_set = [c for c in NUMERICAL_COLS_MASTER if c in features]
    df = coerce_numeric(df, numerical_in_set + [target])
    assert_no_missing_values(df, target, features)
    summary = print_dataset_summary(df, target, target_label)
    out = run_grouped_modeling(df, target, features, target_label)
    out["summary"] = summary
    return out


def build_recommendation(rut_out: Dict[str, Any], scb_out: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, out in [("Rut_20k", rut_out), ("SCB", scb_out)]:
        if out["results"].empty:
            continue
        best = out["results"].iloc[0]
        warn = "YES - investigate overfitting" if best["Overfit_Gap"] > OVERFIT_GAP_WARN else "No"
        rows.append({
            "Target": label,
            "Best_Model": best["Model"],
            "OOF_R2": best["OOF_R2"],
            "OOF_RMSE": best["OOF_RMSE"],
            "OOF_MAE": best["OOF_MAE"],
            "Train_R2": best["Train_R2"],
            "Overfit_Gap": best["Overfit_Gap"],
            "High_Overfit_Warning": warn,
            "Note": "Grouped cross-validation result (GroupKFold by MixDesignKey) — NOT a locked external test-set result.",
        })
    return pd.DataFrame(rows)


def build_methodology_sheet() -> pd.DataFrame:
    rows = [
        ("Grouping", f"All splitting uses GroupKFold(n_splits={CV_FOLDS}) on {ID_COL}; replicate "
                      "tests of the same mix can never span train and validation within a fold."),
        ("Random splits", "Plain random row-level train/test splitting was NOT used, per the leakage "
                           "requirement; GroupKFold has no random_state (deterministic group assignment)."),
        ("Reproducibility", f"RANDOM_STATE={RANDOM_STATE} seeds every model and RandomizedSearchCV."),
        ("Preprocessing", "ColumnTransformer: numerical -> MinMaxScaler, categorical -> "
                           "OneHotEncoder(handle_unknown='ignore'); fit INSIDE each Pipeline/CV fold."),
        ("Tuning", f"TUNER='{TUNER}', N_ITER={N_ITER} (capped per-model to its grid size), "
                    f"scoring='{SCORING}', n_jobs={N_JOBS}, groups passed to .fit()."),
        ("Ranking", "Models ranked by: 1) highest OOF R2, 2) lowest OOF RMSE, 3) smallest overfit gap."),
        ("Overfit gap", "Training R2 (refit on the full dataset) minus pooled out-of-fold R2."),
        ("Data quality", "No missing values are imputed; the workflow asserts the target, predictors, "
                          "and MixDesignKey are complete before modeling."),
        ("Limitation", "Results are grouped cross-validation estimates, not a locked external "
                        "test-set evaluation; no rows are held out from the entire modeling process."),
        ("Limitation 2", "GroupKFold does not stratify by target value; a target-bin-stratified "
                          "grouped split (StratifiedGroupKFold) could reduce fold-to-fold variance "
                          "for a skewed target if needed in a future iteration."),
    ]
    return pd.DataFrame(rows, columns=["Item", "Detail"])


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("RUT_20K + SCB — GROUPED CROSS-VALIDATION TUNED MODELING WORKFLOW")
    print("=" * 100)
    print(f"Tuner: {TUNER} | N_ITER={N_ITER} | CV_FOLDS={CV_FOLDS} | RANDOM_STATE={RANDOM_STATE}")
    print(f"XGBoost={HAS_XGBOOST} CatBoost={HAS_CATBOOST} LightGBM={HAS_LIGHTGBM}")
    print(f"Output folder: {OUTPUT_DIR}")

    rut_out = process_target(RUT_FILE, RUT_TARGET, RUT_FEATURES, "Rut_20k", RUT_FILE_FALLBACKS)
    scb_out = process_target(SCB_FILE, SCB_TARGET, SCB_FEATURES, "SCB", SCB_FILE_FALLBACKS)

    dataset_overview = pd.DataFrame([rut_out["summary"], scb_out["summary"]])

    # ---- Plots: best model gets OOF + residual plots; every tree model gets an importance plot ----
    for label, out in [("Rut_20k", rut_out), ("SCB", scb_out)]:
        if out["results"].empty:
            continue
        best_name = out["results"].iloc[0]["Model"]
        best_oof = out["oof"][out["oof"]["Model"] == best_name]
        plot_oof_predictions(
            best_oof["Measured"], best_oof["Predicted"], label, best_name,
            FIG_DIR / f"{safe_name(label)}_{safe_name(best_name)}_oof_predictions.png",
        )
        plot_residuals(
            best_oof["Measured"], best_oof["Predicted"], label, best_name,
            FIG_DIR / f"{safe_name(label)}_{safe_name(best_name)}_residuals.png",
        )
        if not out["importance"].empty:
            for model_name, fi_sub in out["importance"].groupby("Model"):
                plot_feature_importance(
                    fi_sub, label, model_name,
                    FIG_DIR / f"{safe_name(label)}_{safe_name(model_name)}_feature_importance.png",
                )

    # ---- Save the overall best model per target (top of the ranked leaderboard) ----
    if not rut_out["results"].empty:
        best_rut_name = rut_out["results"].iloc[0]["Model"]
        joblib.dump(rut_out["models"][best_rut_name], OUTPUT_DIR / "best_Rut_20k_model.joblib")
    if not scb_out["results"].empty:
        best_scb_name = scb_out["results"].iloc[0]["Model"]
        joblib.dump(scb_out["models"][best_scb_name], OUTPUT_DIR / "best_SCB_model.joblib")

    recommendation_df = build_recommendation(rut_out, scb_out)
    methodology_df = build_methodology_sheet()

    save_results_to_excel(
        dataset_overview, rut_out, scb_out, recommendation_df, methodology_df, EXCEL_PATH
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 100)
    print("FINAL RECOMMENDATION")
    print("=" * 100)
    print(recommendation_df.to_string(index=False))
    for _, r in recommendation_df.iterrows():
        if r["High_Overfit_Warning"] != "No":
            print(f"\nWARNING: {r['Target']} best model ({r['Best_Model']}) has overfit gap "
                  f"{r['Overfit_Gap']:.3f} > {OVERFIT_GAP_WARN} — investigate regularization/depth.")
    print("\nReminder: all reported metrics are GROUPED CROSS-VALIDATION results "
          "(GroupKFold by MixDesignKey), not a locked external test-set evaluation.")
    print(f"\nElapsed: {elapsed/60:.2f} minutes")
    print(f"Workbook: {EXCEL_PATH}")
    print(f"Best models: {OUTPUT_DIR / 'best_Rut_20k_model.joblib'}, {OUTPUT_DIR / 'best_SCB_model.joblib'}")
    print(f"All pipelines: {MODELS_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
