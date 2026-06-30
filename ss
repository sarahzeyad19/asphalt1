# -*- coding: utf-8 -*-
"""
V5 LEAKAGE-SAFE ASPHALT ML WORKFLOW
====================================
Primary correction from methodology review:
- Rows are NOT independent when multiple records share the same MixDesignKey.
- Primary analysis uses one independent row per MixDesignKey by aggregating target replicates.
- Split is made at MixDesignKey/group level: 80% development groups + 20% locked-test groups.
- Model selection is done by repeated outer grouped/stratified CV inside the development set.
- Hyperparameter tuning is nested inside each outer-training fold.
- The locked test is scored only when SCORE_LOCKED_TEST=True.

Default target: Rut_20k
To run SCB, change TARGET_CONFIG = "SCB".

Recommended first run:
    FAST_RUN = True
After code works and paths are correct:
    FAST_RUN = False
For final thesis run only:
    SCORE_LOCKED_TEST = True
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    joblib = None
    HAS_JOBLIB = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except Exception:
    variance_inflation_factor = None
    HAS_STATSMODELS = False


# =============================================================================
# 0) USER SETTINGS
# =============================================================================

RANDOM_STATE = 42

# Choose "Rut20k" or "SCB".
TARGET_CONFIG = "Rut20k"

# First run with FAST_RUN=True to confirm everything works.
# Set False for stronger model search.
FAST_RUN = True

# Locked test is protected by default. Set True only after the grouped-CV model choice is frozen.
SCORE_LOCKED_TEST = False

# Primary analysis: aggregate replicates to one independent row per MixDesignKey.
# Options: "mean" or "median".
TARGET_AGG = "mean"
RUN_MEDIAN_SENSITIVITY_NOTE = True

# 80% development groups, 20% locked-test groups.
LOCKED_TEST_GROUP_FRACTION = 0.20
N_TARGET_BINS = 5

# Nested CV controls.
OUTER_FOLDS = 5
INNER_FOLDS = 4
OUTER_REPEATS = 1 if FAST_RUN else 3
N_OPTUNA_TRIALS = 8 if FAST_RUN else 30
OPTUNA_TIMEOUT_SECONDS = None

# If True, compare optional stacking after single-model candidates.
RUN_STACKING = False if FAST_RUN else True

# Explainability/diagnostics controls.
RUN_SHAP = True
RUN_PERMUTATION_IMPORTANCE = True
RUN_PDP = True
MAX_SHAP_ROWS = 700
MAX_PDP_ROWS = 700
MAX_PDP_FEATURES = 8

# High-risk threshold for a screening-style diagnostic. Only used for plots/tables.
RUT_HIGH_RISK_THRESHOLD_MM = 6.0

# Windows-safe paths. Uses the current Windows user, not a hard-coded C:\Users\lenovo path.
HOME = Path.home()
DOWNLOADS = HOME / "Downloads"

CONFIGS = {
    "Rut20k": {
        "target": "Rut_20k",
        "id_col": "MixDesignKey",
        "input_candidates": [
            DOWNLOADS / "Rutting_Cleaned_with_RBR.xlsx",
            DOWNLOADS / "Rutting_Cleaned_SpecBased.xlsx",
            DOWNLOADS / "Rutting_Cleaned_with_RBR (1).xlsx",
            DOWNLOADS / "Rutting_Cleaned_SpecBased (1).xlsx",
            DOWNLOADS / "725e0ea2-Rutting_Cleaned_with_RBR.xlsx",
        ],
        "sheet_candidates": ["Cleaned_With_RBR", "Cleaned_Dataset", "Cleaned_Data_Kept", "Sheet1", 0],
        "output_name": "Rut20k_v5_GROUPED_MixDesignKey_NestedCV_outputs",
    },
    "SCB": {
        "target": "SCB",
        "id_col": "MixDesignKey",
        "input_candidates": [
            DOWNLOADS / "SCB_Cleaned_with_RBR.xlsx",
            DOWNLOADS / "SCB_Cleaned_with_RBR (1).xlsx",
            DOWNLOADS / "SCB_Cleaned_SpecBased.xlsx",
            DOWNLOADS / "SCB_Cleaned_SpecBased (1).xlsx",
        ],
        "sheet_candidates": ["Cleaned_With_RBR", "Cleaned_Dataset", "Cleaned_Data_Kept", "Sheet1", 0],
        "output_name": "SCB_v5_GROUPED_MixDesignKey_NestedCV_outputs",
    },
}

if TARGET_CONFIG not in CONFIGS:
    raise ValueError("TARGET_CONFIG must be either 'Rut20k' or 'SCB'.")

TARGET = CONFIGS[TARGET_CONFIG]["target"]
ID_COL = CONFIGS[TARGET_CONFIG]["id_col"]
OUTPUT_DIR = DOWNLOADS / CONFIGS[TARGET_CONFIG]["output_name"]


# =============================================================================
# 1) BASIC HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    return {
        prefix + "R2": float(r2_score(y_true, y_pred)),
        prefix + "RMSE": rmse(y_true, y_pred),
        prefix + "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def best_fit_equation(y_true, y_pred) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "equation": "NA"}
    slope, intercept = np.polyfit(y_true[ok], y_pred[ok], 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "equation": f"Predicted = {slope:.4f} x Measured + {intercept:.4f}",
    }


def save_fig(path: Path, show: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=250, bbox_inches="tight")
    if show:
        try:
            plt.show(block=False)
            plt.pause(0.2)
        except Exception:
            pass
    plt.close()


def resolve_input_file(candidates: List[Path]) -> Path:
    extra = []
    try:
        extra += list(Path.cwd().glob("*Rutting*RBR*.xlsx"))
        extra += list(Path.cwd().glob("*SCB*RBR*.xlsx"))
        extra += list(Path.cwd().glob("*.xlsx"))
    except Exception:
        pass
    for p in candidates + extra:
        if p.exists():
            return p
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find the input Excel file. Put it in your Downloads folder or edit CONFIGS. Tried:\n" + tried
    )


def read_excel_best_sheet(path: Path, sheet_candidates: List[Any]) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    for s in sheet_candidates:
        if s == 0:
            continue
        if str(s) in xls.sheet_names:
            print(f"Using sheet: {s}")
            return pd.read_excel(path, sheet_name=s)
    print(f"Using first sheet: {xls.sheet_names[0]}")
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])


# =============================================================================
# 2) COLUMN STANDARDIZATION + FEATURE ENGINEERING
# =============================================================================

ALIASES = {
    "MixDesignKey": ["MixDesignKey", "Mix_Design_Key", "MixDesign_Key", "JMF_Key", "JMF_ID"],
    "Rut_20k": ["Rut_20k", "Rut20k", "Rut_20k_mm", "LWT_Rut_20k", "Hamburg_Rut_20k"],
    "SCB": ["SCB", "SCB_Jc", "Jc", "SCB_Jc_kJ_m2", "SCB_Result"],
    "AsphaltContent_Design": ["AsphaltContent_Design", "AC_design", "AC_Design", "Asphalt_Content_Design", "Design_AC"],
    "Pass4_75mm": ["Pass4_75mm", "P4.75", "P4_75", "Pass_4_75mm", "Passing_4.75mm"],
    "Pass0_075mm": ["Pass0_075mm", "P0.075", "P0_075", "Pass_0_075mm", "Passing_0.075mm"],
    "NMAS_mm": ["NMAS_mm", "NMAS (mm)", "NMAS", "NMAS(mm)"],
    "PG_HighTemp": ["PG_HighTemp", "PG High", "PG_High", "PGHigh", "PG_High_Temp"],
    "PG_Grade": ["PG_Grade", "PG", "Binder_Grade"],
    "RAP_pct": ["RAP_pct", "RAP", "RAP%", "RAP_Percent", "RAP_Pct"],
    "ACinRAP": ["ACinRAP", "AC_in_RAP", "AC_RAP", "ACin_RAP"],
    "Dust_Binder": ["Dust_Binder", "DustBinder", "Dust_to_Binder", "Dust/Binder"],
    "SandEq": ["SandEq", "Sand_Equivalent", "SandEQ", "SE"],
    "DesignLev": ["DesignLev", "DesignLevel", "Design_Level", "TrafficLevel"],
    "MixType": ["MixType", "Mix_Type", "Mixture_Type", "Type"],
    "RAP_Class": ["RAP_Class", "RAPClass", "RAP class", "RAP_Classification"],
    "RBR_JMF_fraction": ["RBR_JMF_fraction", "RBR_decimal", "RBR_fraction", "RBR"],
    "RBR_JMF_percent": ["RBR_JMF_percent", "RBR_percent", "RBR_pct"],
    "VFA": ["VFA"],
    "VMA": ["VMA"],
    "Va": ["Va", "Air_Voids", "AirVoids"],
    "Gmm": ["Gmm"],
    "Gmb": ["Gmb"],
    "Gsb": ["Gsb"],
    "FAA": ["FAA"],
    "CAA": ["CAA"],
    "Absorption": ["Absorption", "Abs"],
}

DROP_ALWAYS = {
    "LWT_Record_ID", "SCB_Record_ID", "LWT_LastUpdated", "SCB_LastUpdated",
    "Rut_Replicate_Number", "SCB_Replicate_Number", "Predictor_Source",
    "Gmm_record_date", "Gmb_record_date", "Gmb_specimen_AC", "Review_Flags",
    "Aggregate_components_used", "IsLeft", "ADT", "RBR_band",
    "Flag_RBR_out_of_range", "Flag_missing_input", "Split", "Target_Std",
    "Target_Median", "Target_Mean", "Replicate_Count", "WithinMix_Target_Std",
}


def clean_raw_column_name(c: Any) -> str:
    return str(c).strip()


def normalize_col_for_matching(c: Any) -> str:
    s = str(c).strip().lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[\s\-\./()]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def copy_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_raw_column_name(c) for c in df.columns]
    normalized_lookup = {normalize_col_for_matching(c): c for c in df.columns}
    for canonical, candidates in ALIASES.items():
        if canonical in df.columns:
            continue
        for cand in candidates:
            key = normalize_col_for_matching(cand)
            if key in normalized_lookup:
                df[canonical] = df[normalized_lookup[key]]
                break
    return df


def parse_pg_high_temp(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    nums = re.findall(r"\d+", str(value))
    if not nums:
        return np.nan
    return float(nums[0])


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    return a / b


def create_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "PG_HighTemp" not in df.columns and "PG_Grade" in df.columns:
        df["PG_HighTemp"] = df["PG_Grade"].apply(parse_pg_high_temp)

    numeric_like = [
        "Rut_20k", "SCB", "ACinRAP", "PG_HighTemp", "SandEq", "Dust_Binder", "VFA",
        "RAP_pct", "Pass4_75mm", "Pass0_075mm", "FAA", "CAA", "Absorption", "VMA", "Va",
        "Gmb", "Gmm", "Gsb", "AsphaltContent_Design", "NMAS_mm",
        "RBR_JMF_fraction", "RBR_JMF_percent",
    ]
    for c in numeric_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # True RBR. Use fraction as the single model feature; percent is diagnostic only.
    if "RBR_JMF_fraction" not in df.columns and "RBR_JMF_percent" in df.columns:
        df["RBR_JMF_fraction"] = pd.to_numeric(df["RBR_JMF_percent"], errors="coerce") / 100.0
    if "RBR_JMF_fraction" not in df.columns and {"RAP_pct", "ACinRAP", "AsphaltContent_Design"}.issubset(df.columns):
        rap_binder = pd.to_numeric(df["RAP_pct"], errors="coerce") * pd.to_numeric(df["ACinRAP"], errors="coerce") / 100.0
        df["RBR_JMF_fraction"] = safe_divide(rap_binder, df["AsphaltContent_Design"])
    if "RBR_JMF_percent" not in df.columns and "RBR_JMF_fraction" in df.columns:
        df["RBR_JMF_percent"] = 100.0 * pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")

    if {"RAP_pct", "ACinRAP"}.issubset(df.columns):
        df["RAP_pct_x_ACinRAP"] = pd.to_numeric(df["RAP_pct"], errors="coerce") * pd.to_numeric(df["ACinRAP"], errors="coerce")
    if {"PG_HighTemp", "RBR_JMF_fraction"}.issubset(df.columns):
        df["PG_x_RBR"] = pd.to_numeric(df["PG_HighTemp"], errors="coerce") * pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")
    if {"PG_HighTemp", "RAP_pct_x_ACinRAP"}.issubset(df.columns):
        df["PG_x_RAPAC"] = pd.to_numeric(df["PG_HighTemp"], errors="coerce") * pd.to_numeric(df["RAP_pct_x_ACinRAP"], errors="coerce")
    if {"Absorption", "RBR_JMF_fraction"}.issubset(df.columns):
        df["Abs_x_RBR"] = pd.to_numeric(df["Absorption"], errors="coerce") * pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")
    if {"SandEq", "Dust_Binder"}.issubset(df.columns):
        df["SandEq_x_DustBinder"] = pd.to_numeric(df["SandEq"], errors="coerce") * pd.to_numeric(df["Dust_Binder"], errors="coerce")
    if {"Va", "Gmm"}.issubset(df.columns):
        df["Va_x_Gmm"] = pd.to_numeric(df["Va"], errors="coerce") * pd.to_numeric(df["Gmm"], errors="coerce")
    if {"VFA", "AsphaltContent_Design"}.issubset(df.columns):
        df["VFA_x_AC"] = pd.to_numeric(df["VFA"], errors="coerce") * pd.to_numeric(df["AsphaltContent_Design"], errors="coerce")
    if {"Pass0_075mm", "Dust_Binder"}.issubset(df.columns):
        df["P0075_x_DustBinder"] = pd.to_numeric(df["Pass0_075mm"], errors="coerce") * pd.to_numeric(df["Dust_Binder"], errors="coerce")
    return df


# =============================================================================
# 3) PRIMARY GROUP-LEVEL TABLE
# =============================================================================

def aggregate_one_row_per_mix(df: pd.DataFrame, target: str, id_col: str, target_agg: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create one independent row per MixDesignKey.

    Target is aggregated by mean or median. Numeric predictors are averaged; text predictors use first.
    Replicate count and within-mix target std are saved as diagnostics, not features.
    """
    if id_col not in df.columns:
        raise KeyError(f"{id_col!r} not found. Grouped leakage-safe analysis requires MixDesignKey.")
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not found.")

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.loc[df[target].notna()].reset_index(drop=True)
    df[id_col] = df[id_col].astype(str)

    diag = df.groupby(id_col)[target].agg(
        Replicate_Count="size",
        Target_Mean="mean",
        Target_Median="median",
        WithinMix_Target_Std="std",
        Target_Min="min",
        Target_Max="max",
    ).reset_index()

    agg_dict = {}
    for c in df.columns:
        if c == id_col:
            continue
        if c == target:
            agg_dict[c] = target_agg
        elif pd.api.types.is_numeric_dtype(df[c]):
            agg_dict[c] = "mean"
        else:
            agg_dict[c] = "first"
    group_df = df.groupby(id_col, as_index=False).agg(agg_dict)
    group_df = group_df.merge(diag, on=id_col, how="left")
    return group_df, diag


# =============================================================================
# 4) PREDECLARED FEATURE SETS
# =============================================================================

def available(df: pd.DataFrame, features: List[str]) -> List[str]:
    out = []
    for f in features:
        if f in df.columns and f not in DROP_ALWAYS and f not in [TARGET, ID_COL]:
            if f not in out:
                out.append(f)
    return out


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    # RBR_JMF_percent is intentionally not included because it duplicates RBR_JMF_fraction.
    shap12_rbr = [
        "PG_HighTemp", "RBR_JMF_fraction", "SandEq", "Absorption", "VFA", "ACinRAP",
        "Dust_Binder", "Pass4_75mm", "Va", "Pass0_075mm", "Gmm", "FAA",
    ]

    feature17_rbr = [
        "ACinRAP", "PG_HighTemp", "SandEq", "Dust_Binder", "VFA",
        "Pass4_75mm", "FAA", "Absorption", "VMA", "AsphaltContent_Design",
        "RAP_pct_x_ACinRAP", "NMAS_mm", "Pass0_075mm", "Va", "Gmm", "CAA",
        "RBR_JMF_fraction",
    ]

    physical_interactions = shap12_rbr + [
        "PG_x_RBR", "PG_x_RAPAC", "Abs_x_RBR", "SandEq_x_DustBinder",
        "Va_x_Gmm", "VFA_x_AC", "P0075_x_DustBinder",
    ]

    compact_no_rbr = [f for f in shap12_rbr if f != "RBR_JMF_fraction"]

    sets = {
        "SHAP12_RBR": available(df, shap12_rbr),
        "Feature17_RBR": available(df, feature17_rbr),
        "SHAP12_Plus_PhysicalInteractions": available(df, physical_interactions),
        "No_RBR_Sensitivity": available(df, compact_no_rbr),
    }

    # In FAST_RUN, keep the search smaller.
    if FAST_RUN:
        sets = {k: v for k, v in sets.items() if k in ["SHAP12_RBR", "Feature17_RBR"]}

    return {k: v for k, v in sets.items() if len(v) >= 4}


# =============================================================================
# 5) SPLITTING + CV FOLDS
# =============================================================================

def target_bins(y: pd.Series, n_bins: int = N_TARGET_BINS) -> np.ndarray:
    y = pd.Series(y).reset_index(drop=True)
    for q in range(n_bins, 1, -1):
        try:
            b = pd.qcut(y, q=q, labels=False, duplicates="drop")
            if b.nunique(dropna=True) >= 2:
                return b.fillna(0).astype(int).values
        except Exception:
            continue
    return (y >= y.median()).astype(int).values


def can_stratify(bins: np.ndarray, n_splits: Optional[int] = None) -> bool:
    u, c = np.unique(bins, return_counts=True)
    if len(u) < 2:
        return False
    if n_splits is not None and c.min() < n_splits:
        return False
    if c.min() < 2:
        return False
    return True


def split_group_holdout(group_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Split one-row-per-mix table into Dev80 and LockedTest20 by group rows."""
    out = group_df.copy().reset_index(drop=True)
    y = pd.to_numeric(out[target], errors="coerce")
    idx = np.arange(len(out))
    bins = target_bins(y)
    strat = bins if can_stratify(bins) else None
    dev_idx, test_idx = train_test_split(
        idx,
        test_size=LOCKED_TEST_GROUP_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=strat,
    )
    out["Split"] = ""
    out.loc[dev_idx, "Split"] = "Dev80"
    out.loc[test_idx, "Split"] = "LockedTest20"
    return out


def make_repeated_outer_splits(y_dev: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Same outer folds are reused for all model families and feature sets."""
    rows = []
    y_dev = np.asarray(y_dev, dtype=float)
    bins = target_bins(pd.Series(y_dev))
    for rep in range(OUTER_REPEATS):
        seed = RANDOM_STATE + rep * 1009
        if can_stratify(bins, OUTER_FOLDS):
            cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
            split_iter = cv.split(np.zeros(len(y_dev)), bins)
        else:
            cv = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
            split_iter = cv.split(np.zeros(len(y_dev)))
        for fold_id, (tr, va) in enumerate(split_iter, start=1):
            rows.append((rep + 1, np.array(tr), np.array(va)))
    return rows


def make_inner_splits(y_train_outer: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    y_train_outer = np.asarray(y_train_outer, dtype=float)
    bins = target_bins(pd.Series(y_train_outer))
    if can_stratify(bins, INNER_FOLDS):
        cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        return [(np.array(tr), np.array(va)) for tr, va in cv.split(np.zeros(len(y_train_outer)), bins)]
    cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return [(np.array(tr), np.array(va)) for tr, va in cv.split(np.zeros(len(y_train_outer)))]


# =============================================================================
# 6) MODELS + HYPERPARAMETERS
# =============================================================================

def model_available(name: str) -> bool:
    if name == "XGBoost":
        return HAS_XGBOOST
    if name == "LightGBM":
        return HAS_LIGHTGBM
    if name == "CatBoost":
        return HAS_CATBOOST
    return True


def model_names() -> List[str]:
    names = ["Dummy", "Ridge", "ElasticNet", "RandomForest", "ExtraTrees", "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
    if FAST_RUN:
        names = ["Dummy", "Ridge", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]
    return [n for n in names if model_available(n)]


def build_estimator(name: str, params: Optional[Dict[str, Any]] = None) -> Pipeline:
    params = params or {}

    if name == "Dummy":
        model = DummyRegressor(strategy="mean")
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]

    elif name == "Ridge":
        model = Ridge(**params)
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)]

    elif name == "ElasticNet":
        model = ElasticNet(random_state=RANDOM_STATE, max_iter=50000, **params)
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)]

    elif name == "RandomForest":
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]

    elif name == "ExtraTrees":
        model = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]

    elif name == "HistGradientBoosting":
        model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]

    elif name == "XGBoost":
        base = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        base.update(params)
        model = XGBRegressor(**base)
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]

    elif name == "LightGBM":
        base = dict(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        base.update(params)
        model = LGBMRegressor(**base)
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]

    elif name == "CatBoost":
        base = dict(loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
        base.update(params)
        model = CatBoostRegressor(**base)
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]

    else:
        raise ValueError(f"Unknown model {name}")

    return Pipeline(steps)


def default_params(name: str) -> Dict[str, Any]:
    if name == "Dummy":
        return {}
    if name == "Ridge":
        return {"alpha": 10.0}
    if name == "ElasticNet":
        return {"alpha": 0.05, "l1_ratio": 0.5}
    if name == "RandomForest":
        return {"n_estimators": 700, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.75}
    if name == "ExtraTrees":
        return {"n_estimators": 700, "max_depth": 12, "min_samples_leaf": 3, "max_features": 0.75}
    if name == "HistGradientBoosting":
        return {"max_iter": 600, "learning_rate": 0.035, "max_leaf_nodes": 20, "min_samples_leaf": 25, "l2_regularization": 1.0}
    if name == "XGBoost":
        return {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 10,
                "subsample": 0.80, "colsample_bytree": 0.75, "gamma": 0.2, "reg_alpha": 2.0, "reg_lambda": 30.0}
    if name == "LightGBM":
        return {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 3, "num_leaves": 15,
                "min_child_samples": 35, "subsample": 0.80, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 20.0}
    if name == "CatBoost":
        return {"iterations": 700, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 20.0, "random_strength": 1.0}
    return {}


def suggest_params(trial, name: str) -> Dict[str, Any]:
    if name == "Dummy":
        return {}
    if name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}
    if name == "ElasticNet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=250),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
            "max_features": trial.suggest_float("max_features", 0.45, 0.95),
        }
    if name == "ExtraTrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=250),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
            "max_features": trial.suggest_float("max_features", 0.45, 0.95),
        }
    if name == "HistGradientBoosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 300, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 40),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
        }
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 8, 40),
            "subsample": trial.suggest_float("subsample", 0.60, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.90),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 10.0, 120.0),
        }
    if name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample": trial.suggest_float("subsample", 0.60, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.90),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 80.0),
        }
    if name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 500, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 50.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        }
    return default_params(name)


# =============================================================================
# 7) FITTING, INNER TUNING, NESTED CV
# =============================================================================

def fit_predict(estimator: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame) -> np.ndarray:
    model = clone(estimator)
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    # rutting/SCB cannot be negative; clipping avoids nonsensical negative predictions.
    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 0, None)
    return pred


def inner_cv_score(name: str, params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, inner_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    estimator = build_estimator(name, params)
    rows = []
    for tr_idx, va_idx in inner_splits:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        pred_tr = fit_predict(estimator, X_tr, y_tr, X_tr)
        pred_va = fit_predict(estimator, X_tr, y_tr, X_va)
        rows.append({
            "Inner_RMSE": rmse(y_va, pred_va),
            "Inner_MAE": float(mean_absolute_error(y_va, pred_va)),
            "Inner_R2": float(r2_score(y_va, pred_va)),
            "Inner_Train_R2": float(r2_score(y_tr, pred_tr)),
        })
    d = pd.DataFrame(rows)
    return {
        "Inner_RMSE_Mean": float(d["Inner_RMSE"].mean()),
        "Inner_RMSE_SD": float(d["Inner_RMSE"].std()),
        "Inner_MAE_Mean": float(d["Inner_MAE"].mean()),
        "Inner_R2_Mean": float(d["Inner_R2"].mean()),
        "Inner_R2_SD": float(d["Inner_R2"].std()),
        "Inner_Train_R2_Mean": float(d["Inner_Train_R2"].mean()),
        "Inner_Gap_Mean": float(d["Inner_Train_R2"].mean() - d["Inner_R2"].mean()),
    }


def tune_inside_outer_fold(name: str, X_outer_train: pd.DataFrame, y_outer_train: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float], Optional[Any]]:
    if name == "Dummy":
        params = {}
        inner_splits = make_inner_splits(y_outer_train)
        return params, inner_cv_score(name, params, X_outer_train, y_outer_train, inner_splits), None

    inner_splits = make_inner_splits(y_outer_train)

    if not HAS_OPTUNA:
        params = default_params(name)
        return params, inner_cv_score(name, params, X_outer_train, y_outer_train, inner_splits), None

    def objective(trial):
        params = suggest_params(trial, name)
        s = inner_cv_score(name, params, X_outer_train, y_outer_train, inner_splits)
        # Optimize primarily RMSE. Add mild penalty for instability and overfit gap.
        objective_value = s["Inner_RMSE_Mean"] + 0.05 * s["Inner_RMSE_SD"] + 0.03 * max(0.0, s["Inner_Gap_Mean"])
        trial.set_user_attr("score", s)
        trial.set_user_attr("params", params)
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_SECONDS, show_progress_bar=False)
    best_params = study.best_trial.user_attrs["params"]
    best_inner = study.best_trial.user_attrs["score"]
    return best_params, best_inner, study


def evaluate_candidate_nested(
    feature_set_name: str,
    features: List[str],
    model_name: str,
    dev_df: pd.DataFrame,
    outer_splits: List[Tuple[int, np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_dev = dev_df[features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y_dev = pd.to_numeric(dev_df[TARGET], errors="coerce").values.astype(float)
    keys_dev = dev_df[ID_COL].astype(str).values

    outer_rows = []
    pred_rows = []

    for global_fold_id, (rep, tr_idx, va_idx) in enumerate(outer_splits, start=1):
        X_tr, X_va = X_dev.iloc[tr_idx], X_dev.iloc[va_idx]
        y_tr, y_va = y_dev[tr_idx], y_dev[va_idx]

        params, inner_score, _ = tune_inside_outer_fold(model_name, X_tr, y_tr)
        estimator = build_estimator(model_name, params)
        estimator.fit(X_tr, y_tr)
        pred_tr = np.clip(estimator.predict(X_tr), 0, None)
        pred_va = np.clip(estimator.predict(X_va), 0, None)

        m_va = metrics_dict(y_va, pred_va, prefix="Outer_")
        m_tr = metrics_dict(y_tr, pred_tr, prefix="OuterTrain_")
        bf = best_fit_equation(y_va, pred_va)

        outer_rows.append({
            "Feature_Set": feature_set_name,
            "Model": model_name,
            "Outer_Repeat": rep,
            "Outer_Fold": global_fold_id,
            "N_Train_Groups": len(tr_idx),
            "N_Validation_Groups": len(va_idx),
            "N_Features": len(features),
            **m_va,
            **m_tr,
            "Outer_Gap_TrainMinusVal_R2": m_tr["OuterTrain_R2"] - m_va["Outer_R2"],
            "Outer_BestFit_Slope": bf["slope"],
            "Outer_BestFit_Intercept": bf["intercept"],
            **inner_score,
            "Best_Params_JSON": json.dumps(params),
        })

        for k, yt, yp in zip(keys_dev[va_idx], y_va, pred_va):
            pred_rows.append({
                ID_COL: k,
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Outer_Repeat": rep,
                "Outer_Fold": global_fold_id,
                "Measured": yt,
                "Predicted": yp,
                "Residual_PredMinusMeasured": yp - yt,
                "Abs_Error": abs(yp - yt),
                "Abs_Rel_Error_pct": 100.0 * abs(yp - yt) / max(abs(yt), 1e-12),
            })

        print(
            f"  {feature_set_name} | {model_name} | outer fold {global_fold_id}: "
            f"R2={m_va['Outer_R2']:.3f}, RMSE={m_va['Outer_RMSE']:.4f}, MAE={m_va['Outer_MAE']:.4f}"
        )

    return pd.DataFrame(outer_rows), pd.DataFrame(pred_rows)


def summarize_leaderboard(outer_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fs, model), sub in outer_results.groupby(["Feature_Set", "Model"]):
        rows.append({
            "Feature_Set": fs,
            "Model": model,
            "Outer_RMSE_Mean": float(sub["Outer_RMSE"].mean()),
            "Outer_RMSE_SD": float(sub["Outer_RMSE"].std()),
            "Outer_MAE_Mean": float(sub["Outer_MAE"].mean()),
            "Outer_R2_Mean": float(sub["Outer_R2"].mean()),
            "Outer_R2_SD": float(sub["Outer_R2"].std()),
            "Outer_R2_Min": float(sub["Outer_R2"].min()),
            "Outer_Gap_Mean": float(sub["Outer_Gap_TrainMinusVal_R2"].mean()),
            "N_Features": int(sub["N_Features"].iloc[0]),
            "N_Outer_Folds": len(sub),
            "Robust_Rank_Score_LowerBetter": float(
                sub["Outer_RMSE"].mean()
                + 0.10 * sub["Outer_RMSE"].std()
                + 0.02 * max(0.0, sub["Outer_Gap_TrainMinusVal_R2"].mean())
                + 0.0005 * int(sub["N_Features"].iloc[0])
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["Robust_Rank_Score_LowerBetter", "Outer_RMSE_Mean", "Outer_R2_Mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


# =============================================================================
# 8) OPTIONAL STACKING USING GROUPED OUTER RESULTS
# =============================================================================

def evaluate_simple_stacking(best_candidates: pd.DataFrame, feature_sets: Dict[str, List[str]], dev_df: pd.DataFrame, outer_splits):
    """Optional secondary benchmark. Uses selected base model families from the leaderboard.
    Kept simple and not used in FAST_RUN.
    """
    if not RUN_STACKING or best_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Use the top 3 single candidates as base estimators, but fit them on the same feature set if possible.
    # To keep stacking valid and simple, use the best feature set from the top-ranked candidate.
    top = best_candidates.iloc[0]
    fs = top["Feature_Set"]
    features = feature_sets[fs]
    X_dev = dev_df[features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y_dev = pd.to_numeric(dev_df[TARGET], errors="coerce").values.astype(float)
    keys_dev = dev_df[ID_COL].astype(str).values

    base_names = [m for m in ["Ridge", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"] if model_available(m)]
    if len(base_names) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Tune each base once on all dev data by inner CV defaults/Optuna, then use cloned estimators in stacking.
    estimators = []
    for name in base_names[:4]:
        params, _, _ = tune_inside_outer_fold(name, X_dev, y_dev)
        estimators.append((name, build_estimator(name, params)))

    rows, pred_rows = [], []
    for global_fold_id, (rep, tr_idx, va_idx) in enumerate(outer_splits, start=1):
        X_tr, X_va = X_dev.iloc[tr_idx], X_dev.iloc[va_idx]
        y_tr, y_va = y_dev[tr_idx], y_dev[va_idx]
        stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=make_inner_splits(y_tr), n_jobs=-1)
        stack.fit(X_tr, y_tr)
        pred_tr = np.clip(stack.predict(X_tr), 0, None)
        pred_va = np.clip(stack.predict(X_va), 0, None)
        m_va = metrics_dict(y_va, pred_va, prefix="Outer_")
        m_tr = metrics_dict(y_tr, pred_tr, prefix="OuterTrain_")
        rows.append({
            "Feature_Set": fs,
            "Model": "StackingRegressor",
            "Outer_Repeat": rep,
            "Outer_Fold": global_fold_id,
            "N_Train_Groups": len(tr_idx),
            "N_Validation_Groups": len(va_idx),
            "N_Features": len(features),
            **m_va,
            **m_tr,
            "Outer_Gap_TrainMinusVal_R2": m_tr["OuterTrain_R2"] - m_va["Outer_R2"],
            "Best_Params_JSON": "stacking from tuned base estimators",
        })
        for k, yt, yp in zip(keys_dev[va_idx], y_va, pred_va):
            pred_rows.append({
                ID_COL: k,
                "Feature_Set": fs,
                "Model": "StackingRegressor",
                "Outer_Repeat": rep,
                "Outer_Fold": global_fold_id,
                "Measured": yt,
                "Predicted": yp,
                "Residual_PredMinusMeasured": yp - yt,
                "Abs_Error": abs(yp - yt),
                "Abs_Rel_Error_pct": 100.0 * abs(yp - yt) / max(abs(yt), 1e-12),
            })
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


# =============================================================================
# 9) DATA QUALITY, REDUNDANCY, SFR
# =============================================================================

def missingness_report(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for c in features:
        if c in df.columns:
            rows.append({
                "Feature": c,
                "Missing_Count": int(df[c].isna().sum()),
                "Missing_Percent": float(100.0 * df[c].isna().mean()),
                "Unique_Values": int(df[c].nunique(dropna=True)),
            })
    return pd.DataFrame(rows).sort_values(["Missing_Percent", "Feature"], ascending=[False, True])


def high_correlation_report(df: pd.DataFrame, features: List[str], threshold: float = 0.95) -> pd.DataFrame:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    corr = X.corr().abs()
    rows = []
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if j <= i:
                continue
            val = corr.loc[a, b]
            if np.isfinite(val) and val >= threshold:
                rows.append({"Feature_A": a, "Feature_B": b, "Abs_Correlation": float(val)})
    return pd.DataFrame(rows).sort_values("Abs_Correlation", ascending=False) if rows else pd.DataFrame(columns=["Feature_A", "Feature_B", "Abs_Correlation"])


def vif_report(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    if not HAS_STATSMODELS or len(features) < 2:
        return pd.DataFrame()
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=features)
    # Drop zero-variance features.
    X = X.loc[:, X.std() > 1e-12]
    rows = []
    for i, c in enumerate(X.columns):
        try:
            rows.append({"Feature": c, "VIF": float(variance_inflation_factor(X.values, i))})
        except Exception:
            rows.append({"Feature": c, "VIF": np.nan})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def sample_feature_ratio_table(n_groups: int, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    # Approximation from comments: innermost fit with 80% dev, 5 outer folds, 4 inner folds uses ~48% of all groups.
    inner_fraction = (1.0 - LOCKED_TEST_GROUP_FRACTION) * ((OUTER_FOLDS - 1) / OUTER_FOLDS) * ((INNER_FOLDS - 1) / INNER_FOLDS)
    rows = []
    for name, feats in feature_sets.items():
        p = max(len(feats), 1)
        rows.append({
            "Feature_Set": name,
            "Independent_Mixes": n_groups,
            "Features": len(feats),
            "Full_Group_SFR": n_groups / p,
            "Approx_Inner_Group_SFR": n_groups * inner_fraction / p,
            "Comment": "OK >=10 minimum; stronger practice is closer to 100" if (n_groups * inner_fraction / p) >= 10 else "Warning: below minimum SFR heuristic",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 10) PLOTS + EXPLAINABILITY
# =============================================================================

def plot_measured_predicted(y, pred, title: str, path: Path):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = metrics_dict(y, pred)
    bf = best_fit_equation(y, pred)
    minv = float(np.nanmin([y.min(), pred.min()]))
    maxv = float(np.nanmax([y.max(), pred.max()]))
    plt.figure(figsize=(7, 6))
    plt.scatter(y, pred, alpha=0.65)
    plt.plot([minv, maxv], [minv, maxv], "--", label="Ideal 1:1")
    plt.plot([minv, maxv], [bf["slope"] * minv + bf["intercept"], bf["slope"] * maxv + bf["intercept"]], label="Best fit")
    plt.xlabel(f"Measured {TARGET}")
    plt.ylabel(f"Predicted {TARGET}")
    plt.title(f"{title}\nR2={m['R2']:.3f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}\n{bf['equation']}")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(path)


def plot_residuals(y, pred, title: str, outdir: Path):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    res = pred - y
    plt.figure(figsize=(7, 5))
    plt.scatter(y, res, alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel(f"Measured {TARGET}")
    plt.ylabel("Residual = predicted - measured")
    plt.title(f"{title}: residuals vs measured")
    plt.grid(alpha=0.3)
    save_fig(outdir / f"{safe_name(title)}_residuals_vs_measured.png")

    plt.figure(figsize=(7, 5))
    plt.hist(res, bins=25)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"{title}: residual histogram")
    plt.grid(alpha=0.3)
    save_fig(outdir / f"{safe_name(title)}_residual_histogram.png")


def permutation_importance_table(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    if not RUN_PERMUTATION_IMPORTANCE:
        return pd.DataFrame()
    try:
        r = permutation_importance(model, X, y, n_repeats=10, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return pd.DataFrame({
            "Feature": X.columns,
            "Permutation_Importance_Mean_RMSE_Decrease": r.importances_mean,
            "Permutation_Importance_SD": r.importances_std,
        }).sort_values("Permutation_Importance_Mean_RMSE_Decrease", ascending=False)
    except Exception as e:
        print(f"Permutation importance failed: {e}")
        return pd.DataFrame()


def model_feature_importance(model: Pipeline, features: List[str]) -> pd.DataFrame:
    try:
        m = model.named_steps["model"]
        if hasattr(m, "feature_importances_"):
            return pd.DataFrame({"Feature": features, "Model_Importance": m.feature_importances_}).sort_values("Model_Importance", ascending=False)
        if hasattr(m, "coef_"):
            return pd.DataFrame({"Feature": features, "Coefficient": np.ravel(m.coef_)}).sort_values("Coefficient", key=np.abs, ascending=False)
    except Exception:
        pass
    return pd.DataFrame()


def make_shap_plots(model: Pipeline, X: pd.DataFrame, features: List[str], outdir: Path) -> pd.DataFrame:
    if not RUN_SHAP or not HAS_SHAP:
        return pd.DataFrame()
    try:
        if len(X) > MAX_SHAP_ROWS:
            X_use = X.sample(MAX_SHAP_ROWS, random_state=RANDOM_STATE)
        else:
            X_use = X.copy()
        imputer = model.named_steps["imputer"]
        X_imp = pd.DataFrame(imputer.transform(X_use), columns=features, index=X_use.index)
        tree_model = model.named_steps["model"]
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_imp)
        shap.summary_plot(shap_values, X_imp, plot_type="bar", show=False)
        plt.title("SHAP mean absolute importance")
        save_fig(outdir / "SHAP_bar.png")
        shap.summary_plot(shap_values, X_imp, show=False)
        plt.title("SHAP beeswarm")
        save_fig(outdir / "SHAP_beeswarm.png")
        return pd.DataFrame({"Feature": features, "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)}).sort_values("MeanAbsSHAP", ascending=False)
    except Exception as e:
        print(f"SHAP failed. This can happen for some model types. Error: {e}")
        return pd.DataFrame()


def make_pdp_1d(model: Pipeline, X: pd.DataFrame, features: List[str], outdir: Path) -> pd.DataFrame:
    if not RUN_PDP:
        return pd.DataFrame()
    if len(X) > MAX_PDP_ROWS:
        X_ref = X.sample(MAX_PDP_ROWS, random_state=RANDOM_STATE)
    else:
        X_ref = X.copy()
    rows = []
    for f in features[:MAX_PDP_FEATURES]:
        s = pd.to_numeric(X_ref[f], errors="coerce").dropna()
        if s.nunique() < 4:
            continue
        grid = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 25)))
        vals = []
        for g in grid:
            X_tmp = X_ref.copy()
            X_tmp[f] = g
            vals.append(float(np.mean(np.clip(model.predict(X_tmp), 0, None))))
            rows.append({"Feature": f, "Value": g, "PartialDependence": vals[-1]})
        plt.figure(figsize=(7, 5))
        plt.plot(grid, vals, marker="o")
        plt.xlabel(f)
        plt.ylabel(f"Predicted {TARGET}")
        plt.title(f"1D PDP: {f}")
        plt.grid(alpha=0.3)
        save_fig(outdir / f"PDP_1D_{safe_name(f)}.png")
    return pd.DataFrame(rows)


# =============================================================================
# 11) FINAL FIT + LOCKED TEST
# =============================================================================

def tune_on_all_dev(model_name: str, X_dev: pd.DataFrame, y_dev: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
    params, inner_score, _ = tune_inside_outer_fold(model_name, X_dev.reset_index(drop=True), np.asarray(y_dev, dtype=float))
    return params, inner_score


def select_one_standard_error_rule(leaderboard: pd.DataFrame) -> pd.Series:
    """Select simpler model if within one SE of the best RMSE.
    Since all candidates use same outer fold count, SE = SD/sqrt(n_folds).
    """
    lb = leaderboard.copy().reset_index(drop=True)
    best = lb.iloc[0]
    best_threshold = best["Outer_RMSE_Mean"] + best["Outer_RMSE_SD"] / np.sqrt(max(best["N_Outer_Folds"], 1))
    eligible = lb[lb["Outer_RMSE_Mean"] <= best_threshold].copy()
    simplicity_order = {"Dummy": 0, "Ridge": 1, "ElasticNet": 2, "RandomForest": 3, "ExtraTrees": 4,
                        "HistGradientBoosting": 5, "XGBoost": 6, "LightGBM": 7, "CatBoost": 8, "StackingRegressor": 9}
    eligible["Simplicity_Order"] = eligible["Model"].map(simplicity_order).fillna(99)
    eligible = eligible.sort_values(["Simplicity_Order", "N_Features", "Outer_RMSE_Mean"], ascending=True)
    chosen = eligible.iloc[0]
    chosen = chosen.copy()
    chosen["Selection_Note"] = f"One-SE rule used. Best RMSE threshold={best_threshold:.6f}."
    return chosen


# =============================================================================
# 12) MAIN
# =============================================================================

def main():
    t0 = time.time()
    ensure_dir(OUTPUT_DIR)
    fig_dir = OUTPUT_DIR / "figures"
    ensure_dir(fig_dir)

    print("=" * 100)
    print("V5 GROUPED MIXDESIGNKEY NESTED-CV ASPHALT ML WORKFLOW")
    print("=" * 100)
    print(f"Target config: {TARGET_CONFIG}")
    print(f"Target: {TARGET}")
    print(f"Current Windows user folder: {HOME}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"FAST_RUN={FAST_RUN} | SCORE_LOCKED_TEST={SCORE_LOCKED_TEST}")
    print("Primary analysis: one row per MixDesignKey; locked test split by groups.")

    path = resolve_input_file(CONFIGS[TARGET_CONFIG]["input_candidates"])
    raw = read_excel_best_sheet(path, CONFIGS[TARGET_CONFIG]["sheet_candidates"])
    raw = copy_alias_columns(raw)
    raw = create_engineered_columns(raw)

    group_df, replicate_diag = aggregate_one_row_per_mix(raw, TARGET, ID_COL, TARGET_AGG)
    group_df = create_engineered_columns(group_df)

    # Split groups, then never use locked groups in model selection.
    split_df = split_group_holdout(group_df, TARGET)
    dev_df = split_df[split_df["Split"] == "Dev80"].copy().reset_index(drop=True)
    test_df = split_df[split_df["Split"] == "LockedTest20"].copy().reset_index(drop=True)

    feature_sets = build_feature_sets(split_df)
    if not feature_sets:
        raise RuntimeError("No valid feature sets found. Check column names and target.")

    data_inventory = pd.DataFrame([{
        "Input_File": str(path),
        "Raw_Rows_With_Target": int(pd.to_numeric(raw[TARGET], errors="coerce").notna().sum()),
        "Unique_MixDesignKey": int(raw[ID_COL].astype(str).nunique()),
        "GroupRows_After_Aggregation": len(group_df),
        "Dev80_Groups": len(dev_df),
        "LockedTest20_Groups": len(test_df),
        "Target_Aggregation": TARGET_AGG,
        "Target_Min_Grouped": float(group_df[TARGET].min()),
        "Target_Max_Grouped": float(group_df[TARGET].max()),
        "RBR_Mean_Percent_Grouped": float(group_df["RBR_JMF_percent"].mean()) if "RBR_JMF_percent" in group_df.columns else np.nan,
    }])

    print("\nData inventory:")
    print(data_inventory.T)
    print("\nFeature sets:")
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)} features -> {v}")

    y_dev = pd.to_numeric(dev_df[TARGET], errors="coerce").values.astype(float)
    outer_splits = make_repeated_outer_splits(y_dev)

    # Save the fixed outer fold map by MixDesignKey, same for every model family.
    fold_map_rows = []
    for fold_global, (rep, tr_idx, va_idx) in enumerate(outer_splits, start=1):
        for idx in tr_idx:
            fold_map_rows.append({ID_COL: dev_df.loc[idx, ID_COL], "Outer_Repeat": rep, "Outer_Fold": fold_global, "Role": "OuterTrain"})
        for idx in va_idx:
            fold_map_rows.append({ID_COL: dev_df.loc[idx, ID_COL], "Outer_Repeat": rep, "Outer_Fold": fold_global, "Role": "OuterValidation"})
    fold_map_df = pd.DataFrame(fold_map_rows)

    all_outer = []
    all_preds = []

    models = model_names()
    print("\nModels to compare:", models)
    print(f"Outer folds: {len(outer_splits)} total = {OUTER_FOLDS} folds x {OUTER_REPEATS} repeat(s)")
    print(f"Inner folds: {INNER_FOLDS}; Optuna trials per inner search: {N_OPTUNA_TRIALS}")

    for fs_name, feats in feature_sets.items():
        for mname in models:
            print("\n" + "-" * 100)
            print(f"Evaluating {fs_name} | {mname}")
            print("-" * 100)
            try:
                out, pred = evaluate_candidate_nested(fs_name, feats, mname, dev_df, outer_splits)
                all_outer.append(out)
                all_preds.append(pred)
            except Exception as e:
                print(f"FAILED: {fs_name} | {mname}: {type(e).__name__}: {e}")

    if not all_outer:
        raise RuntimeError("No model completed. Check installed libraries and data columns.")

    outer_results = pd.concat(all_outer, ignore_index=True)
    outer_predictions = pd.concat(all_preds, ignore_index=True)
    leaderboard = summarize_leaderboard(outer_results)

    # Optional stacking benchmark.
    if RUN_STACKING:
        try:
            stack_out, stack_pred = evaluate_simple_stacking(leaderboard, feature_sets, dev_df, outer_splits)
            if not stack_out.empty:
                outer_results = pd.concat([outer_results, stack_out], ignore_index=True)
                outer_predictions = pd.concat([outer_predictions, stack_pred], ignore_index=True)
                leaderboard = summarize_leaderboard(outer_results)
        except Exception as e:
            print(f"Stacking skipped due to error: {e}")

    selected = select_one_standard_error_rule(leaderboard)
    selected_fs = selected["Feature_Set"]
    selected_model = selected["Model"]
    selected_features = feature_sets[selected_fs]

    print("\n" + "=" * 100)
    print("GROUPED-CV LEADERBOARD")
    print("=" * 100)
    print(leaderboard.head(12).to_string(index=False))
    print("\nSELECTED MODEL BY GROUPED OUTER CV + ONE-SE RULE")
    print(selected.to_string())

    # Tune final selected model on all Dev80 groups only.
    X_dev_final = dev_df[selected_features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y_dev_final = pd.to_numeric(dev_df[TARGET], errors="coerce").values.astype(float)
    final_params, final_inner_score = tune_on_all_dev(selected_model, X_dev_final, y_dev_final)
    final_model = build_estimator(selected_model, final_params)
    final_model.fit(X_dev_final, y_dev_final)
    pred_dev = np.clip(final_model.predict(X_dev_final), 0, None)
    dev_metrics = metrics_dict(y_dev_final, pred_dev, prefix="Dev80_FinalFit_")
    dev_bf = best_fit_equation(y_dev_final, pred_dev)

    final_summary = {
        "Run_Mode": "DEVELOPMENT_ONLY_LOCKED_TEST_NOT_SCORED" if not SCORE_LOCKED_TEST else "FINAL_LOCKED_TEST_SCORED_ONCE",
        "Target_Config": TARGET_CONFIG,
        "Target": TARGET,
        "Target_Aggregation": TARGET_AGG,
        "Selected_Feature_Set": selected_fs,
        "Selected_Model": selected_model,
        "N_Features": len(selected_features),
        "Features": ", ".join(selected_features),
        "GroupedCV_RMSE_Mean": selected["Outer_RMSE_Mean"],
        "GroupedCV_RMSE_SD": selected["Outer_RMSE_SD"],
        "GroupedCV_R2_Mean": selected["Outer_R2_Mean"],
        "GroupedCV_R2_SD": selected["Outer_R2_SD"],
        "GroupedCV_MAE_Mean": selected["Outer_MAE_Mean"],
        "GroupedCV_Gap_Mean": selected["Outer_Gap_Mean"],
        **dev_metrics,
        "Dev80_FinalFit_BestFit": dev_bf["equation"],
        "Final_Params_JSON": json.dumps(final_params),
        "Final_InnerCV_Score_JSON": json.dumps(final_inner_score),
    }

    locked_test_predictions = pd.DataFrame()
    if SCORE_LOCKED_TEST:
        X_test = test_df[selected_features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        y_test = pd.to_numeric(test_df[TARGET], errors="coerce").values.astype(float)
        pred_test = np.clip(final_model.predict(X_test), 0, None)
        test_metrics = metrics_dict(y_test, pred_test, prefix="LockedTest20_")
        test_bf = best_fit_equation(y_test, pred_test)
        final_summary.update(test_metrics)
        final_summary["LockedTest20_BestFit"] = test_bf["equation"]
        locked_test_predictions = pd.DataFrame({
            ID_COL: test_df[ID_COL].astype(str).values,
            "Measured": y_test,
            "Predicted": pred_test,
            "Residual_PredMinusMeasured": pred_test - y_test,
            "Abs_Error": np.abs(pred_test - y_test),
            "Abs_Rel_Error_pct": 100.0 * np.abs(pred_test - y_test) / np.maximum(np.abs(y_test), 1e-12),
        })
        plot_measured_predicted(y_test, pred_test, "LockedTest20 grouped unseen mixes", fig_dir / "LockedTest20_measured_vs_predicted.png")
        plot_residuals(y_test, pred_test, "LockedTest20", fig_dir)
    else:
        final_summary["LockedTest20_R2"] = "NOT SCORED - set SCORE_LOCKED_TEST=True only after final selection is frozen"
        final_summary["LockedTest20_RMSE"] = "NOT SCORED"
        final_summary["LockedTest20_MAE"] = "NOT SCORED"

    dev_predictions = pd.DataFrame({
        ID_COL: dev_df[ID_COL].astype(str).values,
        "Measured": y_dev_final,
        "Predicted": pred_dev,
        "Residual_PredMinusMeasured": pred_dev - y_dev_final,
        "Abs_Error": np.abs(pred_dev - y_dev_final),
        "Abs_Rel_Error_pct": 100.0 * np.abs(pred_dev - y_dev_final) / np.maximum(np.abs(y_dev_final), 1e-12),
    })
    plot_measured_predicted(y_dev_final, pred_dev, "Dev80 final fit grouped mixes", fig_dir / "Dev80_FinalFit_measured_vs_predicted.png")
    plot_residuals(y_dev_final, pred_dev, "Dev80_FinalFit", fig_dir)

    # Explainability on final model using Dev80 only, unless final test scoring is on.
    feature_importance_df = model_feature_importance(final_model, selected_features)
    perm_df = permutation_importance_table(final_model, X_dev_final, y_dev_final)
    shap_df = make_shap_plots(final_model, X_dev_final, selected_features, fig_dir)

    if not shap_df.empty:
        top_pdp_features = shap_df["Feature"].head(MAX_PDP_FEATURES).tolist()
    elif not feature_importance_df.empty and "Feature" in feature_importance_df.columns:
        top_pdp_features = feature_importance_df["Feature"].head(MAX_PDP_FEATURES).tolist()
    else:
        top_pdp_features = selected_features[:MAX_PDP_FEATURES]
    pdp_df = make_pdp_1d(final_model, X_dev_final, top_pdp_features, fig_dir)

    miss_df = missingness_report(split_df, selected_features)
    corr_df = high_correlation_report(split_df, selected_features, threshold=0.95)
    vif_df = vif_report(split_df, selected_features)
    sfr_df = sample_feature_ratio_table(len(group_df), feature_sets)

    # Map locked/dev split assignments back to raw replicate rows for audit.
    split_key_map = split_df[[ID_COL, "Split"]].copy()
    raw_with_split = raw.copy()
    raw_with_split[ID_COL] = raw_with_split[ID_COL].astype(str)
    raw_with_split = raw_with_split.merge(split_key_map, on=ID_COL, how="left")

    # Save model.
    model_path = OUTPUT_DIR / f"FINAL_MODEL_{TARGET_CONFIG}_{safe_name(selected_fs)}_{safe_name(selected_model)}.joblib"
    if HAS_JOBLIB:
        joblib.dump({
            "model": final_model,
            "selected_features": selected_features,
            "target": TARGET,
            "id_col": ID_COL,
            "target_config": TARGET_CONFIG,
            "target_agg": TARGET_AGG,
            "score_locked_test": SCORE_LOCKED_TEST,
            "final_summary": final_summary,
        }, model_path)

    final_summary["Model_File"] = str(model_path) if HAS_JOBLIB else "joblib not available"
    final_summary["Elapsed_Minutes"] = round((time.time() - t0) / 60.0, 2)

    # Save workbook.
    workbook = OUTPUT_DIR / f"{TARGET_CONFIG}_v5_GROUPED_MixDesignKey_NestedCV_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame([final_summary]).to_excel(writer, sheet_name="Final_Summary", index=False)
        data_inventory.to_excel(writer, sheet_name="Data_Inventory", index=False)
        leaderboard.to_excel(writer, sheet_name="GroupedCV_Leaderboard", index=False)
        outer_results.to_excel(writer, sheet_name="OuterCV_All_Folds", index=False)
        outer_predictions.to_excel(writer, sheet_name="OuterCV_Predictions", index=False)
        fold_map_df.to_excel(writer, sheet_name="OuterCV_FoldMap_ByKey", index=False)
        split_df[[ID_COL, "Split", TARGET, "Replicate_Count", "WithinMix_Target_Std"]].to_excel(writer, sheet_name="Group_Split_Assignments", index=False)
        raw_with_split.to_excel(writer, sheet_name="RawRows_With_GroupSplit", index=False)
        replicate_diag.to_excel(writer, sheet_name="Replicate_Diagnostics", index=False)
        pd.DataFrame({"Feature_Set": list(feature_sets.keys()), "Features": [", ".join(v) for v in feature_sets.values()]}).to_excel(writer, sheet_name="Feature_Sets", index=False)
        sfr_df.to_excel(writer, sheet_name="Sample_Feature_Ratio", index=False)
        miss_df.to_excel(writer, sheet_name="Missingness_Selected", index=False)
        corr_df.to_excel(writer, sheet_name="High_Correlation_Selected", index=False)
        vif_df.to_excel(writer, sheet_name="VIF_Selected", index=False)
        dev_predictions.to_excel(writer, sheet_name="Dev80_FinalFit_Pred", index=False)
        if not locked_test_predictions.empty:
            locked_test_predictions.to_excel(writer, sheet_name="LockedTest20_Pred", index=False)
        if not feature_importance_df.empty:
            feature_importance_df.to_excel(writer, sheet_name="Model_Feature_Importance", index=False)
        if not perm_df.empty:
            perm_df.to_excel(writer, sheet_name="Permutation_Importance", index=False)
        if not shap_df.empty:
            shap_df.to_excel(writer, sheet_name="SHAP_Importance", index=False)
        if not pdp_df.empty:
            pdp_df.to_excel(writer, sheet_name="PDP_1D", index=False)

    # README with methodology notes.
    readme = OUTPUT_DIR / "README_v5_grouped_methodology.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("V5 GROUPED MIXDESIGNKEY ASPHALT ML WORKFLOW\n")
        f.write("=" * 80 + "\n\n")
        f.write("Primary correction: rows sharing MixDesignKey are not independent.\n")
        f.write("This workflow aggregates target replicates to one row per MixDesignKey and creates an 80/20 group-level holdout.\n")
        f.write("The locked test is not scored unless SCORE_LOCKED_TEST=True.\n\n")
        f.write("Model selection is based on repeated outer grouped/stratified CV on Dev80 groups, using RMSE first, then R2/MAE/gap/stability.\n")
        f.write("Preprocessing uses sklearn Pipeline so imputation is fit inside each fold.\n")
        f.write("RBR_JMF_fraction is used; RBR_JMF_percent is excluded as a duplicate feature.\n")
        f.write("SHAP/PDP/permutation importance explain model behavior, not causal mechanisms.\n\n")
        f.write("Selected model:\n")
        f.write(json.dumps(final_summary, indent=2, default=str))

    print("\n" + "=" * 100)
    print("FINISHED V5 GROUPED WORKFLOW")
    print("=" * 100)
    print(f"Workbook: {workbook}")
    print(f"Figures: {fig_dir}")
    print(f"README: {readme}")
    print(f"Model: {model_path}")
    print("\nImportant: if SCORE_LOCKED_TEST=False, the locked test remains unscored.")
    print("Set SCORE_LOCKED_TEST=True only after you accept the grouped-CV selected model.")


if __name__ == "__main__":
    main()
