# -*- coding: utf-8 -*-
"""
RUT_20K MODELING WORKFLOW v2 — 70/10/20 LOCKED TEST + TRUE RBR + HIGH REGULARIZATION
                               + STACKING + REPEATED-CV ROBUSTNESS + FULL DIAGNOSTICS
                               + RBR-RANGE SENSITIVITY + APPLICABILITY DOMAIN
Author: Updated for Sarah Al-Jezawi

WHAT CHANGED vs THE PREVIOUS 80/10/10 SCRIPT (and WHY)
------------------------------------------------------
This version implements the reviewer / advisor improvement plan in `Executive_Summary_5.docx`
and the run history in `Rut_20k_Model_Runs_and_Split_Comparison.docx`, with the data split the
user requested.

1.  SPLIT CHANGED TO 70 / 10 / 20 (train / validation / locked-test).
    - Two-stage, target-bin–stratified, reproducible random split (fixed RANDOM_STATE).
    - The 20% test set is split off FIRST and is NEVER touched during tuning, feature
      selection, or model selection. It is scored exactly once, at the very end, after the
      selected model is refit on the combined 80% (train + validation).
    - Development = train(70%) + validation(10%) = 80% of the data.

2.  TRUE RBR (RAP Binder Ratio) IS A FIRST-CLASS FEATURE.
    - The cleaned data file already ships RBR columns:
          RAP_Binder_Contribution = RAP_pct * ACinRAP / 100
          RBR_percent             = 100 * RAP_Binder_Contribution / AsphaltContent_Design
          RBR_decimal             = RBR_percent / 100      (this is the JMF RBR fraction)
          RBR_band                = No RAP / Low / Moderate / High
    - We map RBR_decimal -> RBR_JMF_fraction and RBR_percent -> RBR_JMF_percent and also
      recompute them defensively if the columns are missing.
    - RBR replaces the less-physical RAP_pct_x_ACinRAP in one tested feature set, and is
      added alongside it in another, so we can measure how much RBR helps (advisor request).

3.  STRONGER REGULARIZATION + HIGHLY TUNED, ROBUST MODEL.
    - Shallow trees (max_depth 2-3), high reg_alpha / reg_lambda, low subsample /
      colsample, gamma, large min_child_weight — exactly the advisor grid.
    - RandomizedSearchCV with N_ITER_XGB (default 120) inside training-only 5-fold CV.
    - A repeated-CV robustness pass (RepeatedKFold, 5 folds x N repeats) on the 80% dev set
      gives a stable mean +/- std R^2 for the finalists, so the chosen model is not a lucky
      single split (advisor "repeated CV" request, made tractable).

4.  ENSEMBLE.
    - A StackingRegressor (XGB + LightGBM + HistGB + CatBoost when available, RidgeCV meta,
      OOF-safe cv=5) is trained and competes against the single models.

5.  HIGH-RUT SAMPLE WEIGHTING SENSITIVITY.
    - Optional up-weighting of high-rut samples (q80 -> 1.4, q90 -> 1.8) to attack the
      documented compression (low rut over-predicted, high rut under-predicted). Accepted
      only if it helps high-rut error without hurting overall validation R^2.

6.  RBR-RANGE / SENSITIVITY ANALYSIS + APPLICABILITY DOMAIN (advisor request).
    - Error tables by RBR_band and by rut range to see exactly where the model is weak.
    - SHAP mean|value| broken out by RBR_band.
    - Applicability-domain flags for sparse extreme regions (very high RBR / RAPxAC),
      with a comparison of error inside vs outside the flagged domain.

7.  FULL STATISTICAL + IMPORTANCE + EXPLAINABILITY GRAPH PACKAGE.
    - Descriptive stats, correlation heatmap, VIF multicollinearity table.
    - Best-fit-vs-45-degree plots for BOTH train and validation (and test only at the end).
    - Residual vs predicted / vs measured, residual histogram, Q-Q, relative-error plots.
    - Permutation importance, SHAP bar / beeswarm / waterfall.
    - 1D and 2D PDP for the key physical interactions.
    - Learning curve, bias-variance curves (max_depth, n_estimators), complexity elbow.

HONEST EXPECTATION (from the advisor doc)
-----------------------------------------
The requested target is Validation R^2 >= 0.80. With the current predictors the realistic
ceiling is ~0.60-0.65; the advisor notes that exceeding it almost certainly needs NEW physical
inputs (binder rheology, aging, test conditions). This script pushes as hard as is defensible
(regularization, RBR, interactions, weighting, stacking, repeated CV) and reports the honest
best-achieved metrics in a decision table against the targets, rather than overfitting to hit a
number. The locked 20% test is the final reality check.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg") if False else None  # keep interactive default; set Agg if headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import ElasticNet, HuberRegressor, RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    import scipy.stats as stats
    HAS_SCIPY = True
except Exception:
    stats = None
    HAS_SCIPY = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except Exception:
    variance_inflation_factor = None
    HAS_STATSMODELS = False

try:
    # Explainable Boosting Machine (glass-box GAM): pip install interpret
    from interpret.glassbox import ExplainableBoostingRegressor
    HAS_EBM = True
except Exception:
    ExplainableBoostingRegressor = None
    HAS_EBM = False


# =============================================================================
# USER SETTINGS
# =============================================================================

RANDOM_STATE = 42
TARGET = "Rut_20k"
ID_COL = "MixDesignKey"

# ---- DATA SPLIT (only thing that differs between the two delivered scripts) ----
TRAIN_SIZE = 0.65
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.20
# Development data = train + validation. Locked test is held out.
SPLIT_TAG = f"{int(TRAIN_SIZE*100)}_{int(VALIDATION_SIZE*100)}_{int(TEST_SIZE*100)}"

# ---- LOCKED-TEST SWITCH ----
# False (current request): DEVELOPMENT-ONLY run. Train on 70%, validate on 10%, check
#       stability / leakage. The 20% test is split off and saved to its own Excel file but
#       is NEVER read, scored, explained, or plotted. Results go to a separate DEV-ONLY workbook.
# True: reveal the locked 20% test ONCE — refit the selected model on the 80% dev set and
#       score + explain the test a single time (final-evaluation run).
SCORE_LOCKED_TEST = False

CV_FOLDS = 5
N_TARGET_BINS = 5

# Tuning budget. Lower these for a fast check; raise N_ITER_XGB to 120+ for final runs.
N_ITER_XGB = 120
N_ITER_OTHER = 50
N_JOBS_SEARCH = 1
N_JOBS_MODEL = -1

# Repeated-CV robustness (advisor "repeated CV"). Set REPEATS lower to save time.
RUN_REPEATED_CV = True
REPEATED_CV_REPEATS = 5          # 5 folds x 5 repeats = 25 dev scores
REPEATED_CV_TOP_K = 3            # legacy: retained for reference; RepeatedCV now runs the best
                                 # instance of EVERY model family + the stacking ensemble

# Ensemble.
RUN_STACKING = True

# High-rut sample weighting sensitivity. Turned OFF: it lowered validation R2 every time.
RUN_HIGH_RUT_WEIGHTING = False

# ---- Option B (Tier 2) data-quality levers: squeeze the most out of the current data ----
# AVERAGE_REPLICATES: the same mix (MixDesignKey) is tested several times with different
#   measured values (measurement noise). Collapsing each mix to ONE row with the AVERAGED
#   target gives a cleaner, more stable target -> usually higher R2 + smaller train/val gap.
#   It also removes the replicate leakage (after averaging, every mix is one unique row, so
#   no mix can appear in both train and test), which makes a group split unnecessary.
# LOG_TARGET: train on log1p(target). The target is right-skewed (many low values, few very
#   high ones); the model "compresses" the high tail (best-fit slope < 1). Modelling in log
#   space symmetrises the target -> better high-value behaviour and a smaller gap. All metrics,
#   best-fit lines, residuals and plots are reported back on the ORIGINAL scale.
AVERAGE_REPLICATES = True
LOG_TARGET = True

# ---- FORCE the final interpretable model (so SHAP/PDP run on a single tree model) ----
# The Stacking ensemble cannot be SHAP-explained, so we lock the headline model to a single
# tree model. Set both to None to fall back to automatic robust-score selection.
FORCE_FINAL_FEATURE_SET = "SHAP12_RBR"
FORCE_FINAL_MODEL = "XGBoost"

# ---- Nested cross-validation (unbiased robustness estimate; advisor request) ----
RUN_NESTED_CV = True
NESTED_CV_FEATURE_SET = "SHAP12_RBR"
NESTED_CV_MODEL = "XGBoost"
NESTED_CV_OUTER_SPLITS = 5
NESTED_CV_OUTER_REPEATS = 2     # 5x2 = 10 unbiased outer scores (raise to 3 for the paper run)
NESTED_CV_INNER_SPLITS = 4
NESTED_CV_INNER_NITER = 30

# GPU options. Default OFF (this machine has no usable CUDA driver -> avoids CatBoost errors).
USE_GPU = False
GPU_DEVICE = "0"
GPU_FALLBACK_TO_CPU = True
USE_LIGHTGBM_GPU = False

# Targets from the advisor decision table.
TARGET_VAL_R2 = 0.80
TARGET_GAP = 0.16
TARGET_RMSE = 1.03
TARGET_MAE = 0.73

# Applicability domain: flag sparse extreme regions.
AD_RBR_PERCENT_HIGH = None       # auto = 97.5th percentile if None
AD_RAPxAC_HIGH = None            # auto = 97.5th percentile if None

# Graph / analysis switches.
SHOW_PLOTS_IN_SPYDER = True
FIGURE_DPI = 200
RUN_PERMUTATION_IMPORTANCE = True
RUN_SHAP = True
RUN_PDP = True
RUN_LEARNING_CURVE = True
RUN_BIAS_VARIANCE_CURVES = True
RUN_ROC_RISK_SCREENING = True
RUT_RISK_THRESHOLD_MM = 6.0
MAX_SHAP_ROWS = 700
MAX_PDP_FEATURES = 8
MAX_PDP_PAIRS = 4

# Fast smoke test (verify the script runs end-to-end). Set False for the real run.
QUICK_SMOKE_TEST = False

# ---- Input file resolution ----
HOME = Path.home()
# Input data + outputs live here. Change this one line if your Downloads folder moves.
DOWNLOADS = Path(r"C:\Users\H0012066\Downloads")
RUT_FILENAME = "Rutting_Cleaned_with_RBR.xlsx"
RUT_FILE = DOWNLOADS / RUT_FILENAME
RUT_FILE_FALLBACKS = [
    DOWNLOADS / "725e0ea2-Rutting_Cleaned_with_RBR.xlsx",
    DOWNLOADS / "Rutting_Cleaned_with_RBR (1).xlsx",
    DOWNLOADS / "Rutting_Cleaned_SpecBased.xlsx",
]
SHEET_CANDIDATES = ["Cleaned_With_RBR", "Cleaned_Dataset", "Cleaned_Data_Kept", "Sheet1", 0]

OUTPUT_FOLDER = DOWNLOADS / f"Rut20k_v3_{SPLIT_TAG}_RBR_outputs"

if QUICK_SMOKE_TEST:
    N_ITER_XGB = 8
    N_ITER_OTHER = 6
    REPEATED_CV_REPEATS = 2
    MAX_SHAP_ROWS = 200
    NESTED_CV_OUTER_REPEATS = 1
    NESTED_CV_INNER_NITER = 6

if SHOW_PLOTS_IN_SPYDER:
    try:
        plt.ion()
    except Exception:
        pass
np.random.seed(RANDOM_STATE)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def make_dirs() -> Dict[str, Path]:
    dirs = {
        "root": OUTPUT_FOLDER,
        "figures": OUTPUT_FOLDER / "figures",
        "models": OUTPUT_FOLDER / "models",
        "tables": OUTPUT_FOLDER / "tables",
        "splits": OUTPUT_FOLDER / "splits",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for sub in ["best_fit", "diagnostics", "shap", "pdp", "learning",
                "bias_variance", "roc", "feature_sets", "statistics", "rbr_sensitivity"]:
        (dirs["figures"] / sub).mkdir(parents=True, exist_ok=True)
    return dirs


PATHS = make_dirs()


def gpu_status_string() -> str:
    if not USE_GPU:
        return "GPU disabled by setting."
    try:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return "GPU requested, but nvidia-smi not found. CUDA models will fall back to CPU."
        result = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "GPU detected: " + result.stdout.strip()
        return "GPU requested, but nvidia-smi returned nothing."
    except Exception as e:
        return f"GPU status check failed: {type(e).__name__}: {e}"


def is_gpu_error(err: Exception) -> bool:
    txt = str(err).lower()
    return any(k in txt for k in ["cuda", "gpu", "nvidia", "device", "driver", "no visible gpu"])


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if SHOW_PLOTS_IN_SPYDER:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def resolve_file_path(path: Path, fallbacks: List[Path]) -> Path:
    candidates = [path] + fallbacks
    try:
        script_dir = Path(__file__).resolve().parent
        candidates += [script_dir / p.name for p in list(candidates)]
        candidates += list(script_dir.glob("*Rutting_Cleaned_with_RBR*.xlsx"))
    except Exception:
        pass
    candidates += [Path.cwd() / p.name for p in list(candidates)]
    candidates += list(Path.cwd().glob("*Rutting_Cleaned_with_RBR*.xlsx"))
    seen, out = set(), []
    for p in candidates:
        if str(p) in seen:
            continue
        seen.add(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find the Rutting Excel file. Put 'Rutting_Cleaned_with_RBR.xlsx' in your "
        "Downloads folder or next to this script. Tried:\n" + "\n".join(str(p) for p in candidates[:12])
    )


def read_excel_best_sheet(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    for s in SHEET_CANDIDATES:
        try:
            if s == 0:
                continue
            if str(s) in xls.sheet_names:
                print(f"Using sheet: {s}")
                return pd.read_excel(path, sheet_name=s)
        except Exception:
            pass
    print(f"Using first sheet: {xls.sheet_names[0]}")
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])


# =============================================================================
# COLUMN ALIASES + FEATURE ENGINEERING
# =============================================================================

ALIASES = {
    "AsphaltContent_Design": ["AC_design", "AC_Design", "AsphaltContentDesign", "Design_AC"],
    "Pass4_75mm": ["P4.75", "P4_75", "Pass_4_75mm", "Passing_4.75mm"],
    "Pass0_075mm": ["P0.075", "P0_075", "Pass_0_075mm", "Passing_0.075mm"],
    "NMAS (mm)": ["NMAS", "NMAS_mm", "NMAS(mm)"],
    "PG_HighTemp": ["PG High", "PG_High", "PGHigh", "PG_High_Temp"],
    "RAP_pct": ["RAP", "RAP%", "RAP_Percent", "RAP_Pct"],
    "ACinRAP": ["AC_in_RAP", "AC_RAP", "ACin_RAP"],
    "Dust_Binder": ["DustBinder", "Dust_to_Binder", "Dust/Binder"],
    "SandEq": ["Sand_Equivalent", "SandEQ", "SE"],
    "DesignLev": ["DesignLevel", "Design_Level", "TrafficLevel"],
    "MixType": ["Mix_Type", "Mixture_Type", "Type"],
    "RAP_Class": ["RAPClass", "RAP class", "RAP_Classification"],
    # RBR from the cleaned file.
    "RBR_JMF_fraction": ["RBR_decimal", "RBR_fraction", "RBR"],
    "RBR_JMF_percent": ["RBR_percent", "RBR_pct"],
}

NUMERIC_HINTS = [
    "ACinRAP", "PG_HighTemp", "SandEq", "ADT_DOTD_ord", "Dust_Binder", "VFA",
    "RAP_pct", "Pass4_75mm", "Pass0_075mm", "FAA", "CAA", "Absorption", "VMA", "Va",
    "Gmb", "Gmm", "Gsb", "AsphaltContent_Design", "RAP_pct_x_ACinRAP", "NMAS (mm)",
    "RBR_JMF_fraction", "RBR_JMF_percent", "RAP_Binder_Contribution",
    "PG_x_RBR", "PG_x_RAPAC", "Abs_x_RBR", "SandEq_x_DustBinder",
    "Va_x_Gmm", "VFA_x_AC", "P0075_x_DustBinder",
]
CATEGORICAL_HINTS = ["MixType", "DesignLev", "RAP_Class"]

DROP_ALWAYS = [
    ID_COL, "Rut_20k", "SCB", "LWT_Record_ID", "SCB_Record_ID", "LWT_LastUpdated",
    "SCB_LastUpdated", "Rut_Replicate_Number", "SCB_Replicate_Number", "Predictor_Source",
    "Gmm_record_date", "Gmb_record_date", "Gmb_specimen_AC", "Review_Flags",
    "Aggregate_components_used", "IsLeft", "ADT", "RBR_band",
    "Flag_RBR_out_of_range", "Flag_missing_input",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def copy_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for canonical, aliases in ALIASES.items():
        if canonical in df.columns:
            continue
        for a in aliases:
            if a in df.columns:
                df[canonical] = df[a]
                break
    return df


def parse_adt_to_ordinal(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().lower().replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
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


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    return a / b


def create_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ADT" in df.columns and "ADT_DOTD_ord" not in df.columns:
        df["ADT_DOTD_ord"] = df["ADT"].apply(parse_adt_to_ordinal)

    if {"RAP_pct", "ACinRAP"}.issubset(df.columns):
        df["RAP_pct_x_ACinRAP"] = pd.to_numeric(df["RAP_pct"], errors="coerce") * pd.to_numeric(df["ACinRAP"], errors="coerce")

    # True RBR. Prefer the file's columns (mapped via aliases); recompute if absent.
    if "RBR_JMF_fraction" not in df.columns and {"RAP_pct", "ACinRAP", "AsphaltContent_Design"}.issubset(df.columns):
        rap_binder = pd.to_numeric(df["RAP_pct"], errors="coerce") * pd.to_numeric(df["ACinRAP"], errors="coerce") / 100.0
        df["RBR_JMF_fraction"] = safe_divide(rap_binder, df["AsphaltContent_Design"])
    if "RBR_JMF_percent" not in df.columns and "RBR_JMF_fraction" in df.columns:
        df["RBR_JMF_percent"] = 100.0 * pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")
    # Ensure numeric.
    for c in ["RBR_JMF_fraction", "RBR_JMF_percent"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Physically meaningful interactions (advisor: only physics-sensible terms).
    def has(*cols):
        return set(cols).issubset(df.columns)
    if has("PG_HighTemp", "RBR_JMF_fraction"):
        df["PG_x_RBR"] = pd.to_numeric(df["PG_HighTemp"], errors="coerce") * pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")
    if has("PG_HighTemp", "RAP_pct_x_ACinRAP"):
        df["PG_x_RAPAC"] = pd.to_numeric(df["PG_HighTemp"], errors="coerce") * pd.to_numeric(df["RAP_pct_x_ACinRAP"], errors="coerce")
    if has("Absorption", "RBR_JMF_fraction"):
        df["Abs_x_RBR"] = pd.to_numeric(df["Absorption"], errors="coerce") * pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")
    if has("SandEq", "Dust_Binder"):
        df["SandEq_x_DustBinder"] = pd.to_numeric(df["SandEq"], errors="coerce") * pd.to_numeric(df["Dust_Binder"], errors="coerce")
    if has("Va", "Gmm"):
        df["Va_x_Gmm"] = pd.to_numeric(df["Va"], errors="coerce") * pd.to_numeric(df["Gmm"], errors="coerce")
    if has("VFA", "AsphaltContent_Design"):
        df["VFA_x_AC"] = pd.to_numeric(df["VFA"], errors="coerce") * pd.to_numeric(df["AsphaltContent_Design"], errors="coerce")
    if has("Pass0_075mm", "Dust_Binder"):
        df["P0075_x_DustBinder"] = pd.to_numeric(df["Pass0_075mm"], errors="coerce") * pd.to_numeric(df["Dust_Binder"], errors="coerce")

    # Keep a clean RBR_band label for sensitivity grouping (not used as a model feature).
    if "RBR_band" not in df.columns and "RBR_JMF_fraction" in df.columns:
        f = pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce")
        df["RBR_band"] = pd.cut(
            f, bins=[-0.001, 0.0001, 0.15, 0.25, np.inf],
            labels=["No RAP (0)", "Low (>0-0.15)", "Moderate (>0.15-0.25)", "High (>0.25)"],
        ).astype("object")
    return df


# =============================================================================
# FEATURE SETS
# =============================================================================

# NOTE: RAP_pct and RBR_JMF_percent were REMOVED — permutation importance showed ~0 unique
# value (redundant with RAP_pct_x_ACinRAP and RBR_JMF_fraction).
RUT_BASE_NOADT = [
    "ACinRAP", "PG_HighTemp", "SandEq", "Dust_Binder", "VFA",
    "Pass4_75mm", "FAA", "Absorption", "VMA", "AsphaltContent_Design", "RAP_pct_x_ACinRAP",
]
VOLUMETRICS_B_NOADT = RUT_BASE_NOADT + ["NMAS (mm)", "Pass0_075mm", "Va", "Gmm", "CAA"]
VOLUMETRICS_B_RBR_BOTH = VOLUMETRICS_B_NOADT + ["RBR_JMF_fraction"]
VOLUMETRICS_B_RBR_REPLACE = [f for f in VOLUMETRICS_B_NOADT if f != "RAP_pct_x_ACinRAP"] + ["RBR_JMF_fraction"]

SHAP12_RBR = [
    "PG_HighTemp", "RBR_JMF_fraction", "SandEq", "Absorption", "VFA", "ACinRAP",
    "Dust_Binder", "Pass4_75mm", "Va", "Pass0_075mm", "Gmm", "FAA",
]
SHAP14_RBR = SHAP12_RBR + ["VMA", "AsphaltContent_Design"]

PHYSICAL_INTERACTIONS = [
    "PG_x_RBR", "PG_x_RAPAC", "Abs_x_RBR", "SandEq_x_DustBinder",
    "Va_x_Gmm", "VFA_x_AC", "P0075_x_DustBinder",
]

STRUCT_GRAD_DESIGN = RUT_BASE_NOADT + ["NMAS (mm)", "Pass0_075mm", "MixType", "DesignLev", "RAP_Class"]
FULL_EXPANDED_CLEAN = RUT_BASE_NOADT + [
    "NMAS (mm)", "Pass0_075mm", "MixType", "DesignLev", "RAP_Class",
    "CAA", "Gsb", "Va", "Gmm", "Gmb", "RBR_JMF_fraction",
]

FEATURE_SETS = {
    "VolumetricsB_NoADT_CurrentBest": VOLUMETRICS_B_NOADT,
    "VolumetricsB_NoADT_RBR_Both": VOLUMETRICS_B_RBR_BOTH,
    "VolumetricsB_NoADT_RBR_Replace_RAPAC": VOLUMETRICS_B_RBR_REPLACE,
    "SHAP12_RBR": SHAP12_RBR,
    "SHAP14_RBR": SHAP14_RBR,
    "SHAP12_RBR_PlusInteractions": SHAP12_RBR + PHYSICAL_INTERACTIONS,
    "SHAP14_RBR_PlusInteractions": SHAP14_RBR + PHYSICAL_INTERACTIONS,
    "StructGradDesign_CleanCategorical": STRUCT_GRAD_DESIGN,
    "FullExpanded_CleanCategorical": FULL_EXPANDED_CLEAN,
}

FEATURE_SETS_TO_RUN = [
    "VolumetricsB_NoADT_CurrentBest",
    "VolumetricsB_NoADT_RBR_Both",
    "VolumetricsB_NoADT_RBR_Replace_RAPAC",
    "SHAP12_RBR",
    "SHAP14_RBR",
    "SHAP12_RBR_PlusInteractions",
    "SHAP14_RBR_PlusInteractions",
    "StructGradDesign_CleanCategorical",
    "FullExpanded_CleanCategorical",
]
if QUICK_SMOKE_TEST:
    FEATURE_SETS_TO_RUN = ["VolumetricsB_NoADT_RBR_Both", "SHAP12_RBR_PlusInteractions"]


# =============================================================================
# DATA PREP + SPLITTING (70/10/20)
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.Series, Path]:
    path = resolve_file_path(RUT_FILE, RUT_FILE_FALLBACKS)
    df = read_excel_best_sheet(path)
    df = clean_column_names(df)
    df = copy_alias_columns(df)
    df = create_engineered_columns(df)
    if TARGET not in df.columns:
        raise KeyError(f"Target {TARGET!r} not found. Columns: {list(df.columns)[:30]}")
    # Drop rows with no target BEFORE averaging.
    df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)

    # ---- Option B: collapse replicate tests of the same mix into one averaged row ----
    # Each MixDesignKey may be tested multiple times; numeric columns (incl. the target) are
    # averaged, text columns take the first value. This denoises the target and guarantees one
    # unique mix per row (so the train/val/test split cannot leak the same mix across folds).
    if AVERAGE_REPLICATES and ID_COL in df.columns:
        before = len(df)
        agg = {}
        for c in df.columns:
            if c == ID_COL:
                continue
            agg[c] = "mean" if pd.api.types.is_numeric_dtype(df[c]) else "first"
        df = df.groupby(ID_COL, as_index=False).agg(agg)
        print(f"Replicate averaging ON: {before} test rows -> {len(df)} unique mixes "
              f"(one averaged target per mix; removes replicate leakage).")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    mask = y.notna()
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    print("\n" + "=" * 100)
    print(f"Loaded {TARGET} data")
    print("=" * 100)
    print(f"File: {path}")
    print(f"Rows: {len(df)} | Columns after engineering: {df.shape[1]}")
    print(f"Replicate averaging: {AVERAGE_REPLICATES} | Log-target: {LOG_TARGET}")
    has_rbr = "RBR_JMF_fraction" in df.columns
    print(f"True RBR present: {has_rbr}"
          + (f" | RBR_percent mean={df['RBR_JMF_percent'].mean():.2f}" if "RBR_JMF_percent" in df.columns else ""))
    return df, y, path


def make_target_bins(y: pd.Series, n_bins: int = N_TARGET_BINS) -> pd.Series:
    y = pd.Series(y).reset_index(drop=True)
    for q in [n_bins, n_bins - 1, 4, 3, 2]:
        if q < 2:
            break
        try:
            bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
            if bins.nunique(dropna=True) >= 2:
                return bins.astype(int)
        except Exception:
            continue
    return (y >= y.median()).astype(int)


def stratified_70_10_20_split(df: pd.DataFrame, y: pd.Series):
    """Two-stage stratified split. Hold out the 20% locked test FIRST, then carve dev into
    70% train + 10% validation (10/80 = 0.125 of dev)."""
    bins = make_target_bins(y)
    idx = np.arange(len(y))

    dev_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=bins,
    )
    dev_bins = bins.iloc[dev_idx].reset_index(drop=True)
    dev_positions = np.arange(len(dev_idx))
    val_fraction_of_dev = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)  # 0.10/0.80 = 0.125
    train_pos, val_pos = train_test_split(
        dev_positions, test_size=val_fraction_of_dev, random_state=RANDOM_STATE,
        shuffle=True, stratify=dev_bins,
    )
    train_idx = dev_idx[train_pos]
    val_idx = dev_idx[val_pos]
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def clean_categorical_series(s: pd.Series) -> pd.Series:
    return (s.astype("object").where(s.notna(), "Missing").astype(str).str.strip()
            .replace({"": "Missing", "nan": "Missing", "None": "Missing"}))


def get_X(df: pd.DataFrame, requested: List[str]):
    req = []
    for f in requested:
        if f not in req:
            req.append(f)
    available = [f for f in req if f in df.columns and f not in DROP_ALWAYS]
    missing = [f for f in req if f not in df.columns]
    X = df[available].copy()
    numerical, categorical = [], []
    for col in X.columns:
        if col in CATEGORICAL_HINTS:
            categorical.append(col)
            X[col] = clean_categorical_series(X[col])
        elif col in NUMERIC_HINTS:
            numerical.append(col)
            # Cast to float so PartialDependenceDisplay accepts integer-valued columns
            # (e.g. PG_HighTemp) without rounding errors.
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
        else:
            coerced = pd.to_numeric(X[col], errors="coerce")
            if coerced.notna().mean() >= 0.85:
                numerical.append(col)
                X[col] = coerced.astype("float64")
            else:
                categorical.append(col)
                X[col] = clean_categorical_series(X[col])
    return X, numerical, categorical, available, missing


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numerical: List[str], categorical: List[str]) -> ColumnTransformer:
    transformers = []
    if numerical:
        transformers.append(("num", SimpleImputer(strategy="median"), numerical))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", make_ohe()),
        ])
        transformers.append(("cat", cat_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_cv_splits_for_training(y_train: pd.Series):
    bins = make_target_bins(y_train)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(np.zeros(len(y_train)), bins))
    return splits, f"Target-bin StratifiedKFold inside 70% training set ({CV_FOLDS} folds, {N_TARGET_BINS} bins)"


# =============================================================================
# METRICS
# =============================================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {"R2": float(r2_score(y_true, y_pred)), "RMSE": rmse(y_true, y_pred),
            "MAE": float(mean_absolute_error(y_true, y_pred))}


def relative_error_summary(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    rel = 100.0 * (y_pred - y_true) / denom
    abs_rel = np.abs(rel)
    return {
        f"{prefix}Mean_Relative_Error_pct": float(np.nanmean(rel)),
        f"{prefix}Median_Relative_Error_pct": float(np.nanmedian(rel)),
        f"{prefix}Mean_Absolute_Relative_Error_pct": float(np.nanmean(abs_rel)),
        f"{prefix}Median_Absolute_Relative_Error_pct": float(np.nanmedian(abs_rel)),
    }


def inference_performance(estimator, X, model_path=None, n_repeats: int = 30) -> pd.DataFrame:
    """Measure real-time serving metrics for the final model on a held-out (non-test) sample:
    single-row latency (ms), batch throughput (rows/sec), and on-disk model size (a proxy for
    how amenable the model is to compression/quantization). These answer the "latency and
    throughput" evaluation criterion alongside accuracy."""
    import time
    X = X.reset_index(drop=True)
    n = len(X)
    if n == 0:
        return pd.DataFrame()
    # Warm-up (first call pays one-off JIT / thread-pool / allocation costs).
    estimator.predict(X.iloc[:1])
    estimator.predict(X)

    # --- Single-row latency: time one row at a time, repeated, report median + tail (p95). ---
    one = X.iloc[:1]
    single_ms = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        estimator.predict(one)
        single_ms.append((time.perf_counter() - t0) * 1e3)
    single_ms = np.array(single_ms)

    # --- Batch throughput: predict the whole sample, repeated; rows / total seconds. ---
    batch_s = []
    for _ in range(max(5, n_repeats // 3)):
        t0 = time.perf_counter()
        estimator.predict(X)
        batch_s.append(time.perf_counter() - t0)
    batch_s = np.array(batch_s)
    batch_throughput = n / float(np.median(batch_s))
    batch_latency_per_row_ms = (float(np.median(batch_s)) / n) * 1e3

    size_mb = np.nan
    if model_path is not None:
        try:
            size_mb = Path(model_path).stat().st_size / (1024 ** 2)
        except Exception:
            pass

    rows = [
        {"Metric": "Single-row latency median (ms)", "Value": round(float(np.median(single_ms)), 4),
         "Note": "Delay from one input row to its prediction (online serving)."},
        {"Metric": "Single-row latency p95 (ms)", "Value": round(float(np.percentile(single_ms, 95)), 4),
         "Note": "Tail latency — 95% of single predictions are faster than this."},
        {"Metric": "Batch latency per row (ms)", "Value": round(batch_latency_per_row_ms, 5),
         "Note": f"Amortized cost per row when scoring the whole {n}-row sample at once."},
        {"Metric": "Batch throughput (rows/sec)", "Value": round(batch_throughput, 1),
         "Note": "How many predictions per second in batch mode (vectorized)."},
        {"Metric": "Model size on disk (MB)", "Value": round(float(size_mb), 4) if np.isfinite(size_mb) else np.nan,
         "Note": "Serialized pipeline size; lower = easier to compress / quantize / deploy."},
        {"Metric": "Sample rows / hardware", "Value": f"{n} rows / CPU",
         "Note": "Measured on the validation sample (no test leakage), single-threaded CPU."},
    ]
    return pd.DataFrame(rows)


def add_error_columns(df: pd.DataFrame, pred_col: str = "Predicted") -> pd.DataFrame:
    out = df.copy()
    out["Residual"] = out[pred_col] - out["Measured"]
    out["Abs_Error"] = out["Residual"].abs()
    denom = out["Measured"].abs().replace(0, np.nan)
    out["Relative_Error_pct"] = 100.0 * out["Residual"] / denom
    out["Abs_Relative_Error_pct"] = out["Relative_Error_pct"].abs()
    return out


def best_fit(y_true, y_pred) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "equation": "NA"}
    slope, intercept = np.polyfit(y_true[ok], y_pred[ok], 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "equation": f"Predicted = {slope:.4f} x Measured + {intercept:.4f}"}


def high_rut_mae(y_true, y_pred, q: float = 0.80) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    thr = np.nanquantile(y_true, q)
    m = y_true >= thr
    if m.sum() < 2:
        return np.nan
    return float(mean_absolute_error(y_true[m], y_pred[m]))


def robust_score(validation_r2, oof_r2, gap, fold_sd, n_features) -> float:
    fold_sd = 0.0 if not np.isfinite(fold_sd) else fold_sd
    return float(0.55 * validation_r2 + 0.45 * oof_r2 - 0.45 * max(0.0, gap)
                 - 0.20 * fold_sd - 0.002 * n_features)


def rut_range_label(y: float) -> str:
    if y < 2:
        return "Low <2 mm"
    if y < 5:
        return "Medium 2-5 mm"
    if y < 7:
        return "High 5-7 mm"
    return "Very high >7 mm"


def error_by_group(pred_df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    rows = []
    d = pred_df.copy()
    for g, sub in d.groupby(group_col):
        if len(sub) < 2:
            continue
        m = metrics(sub["Measured"], sub["Predicted"])
        bias = float((sub["Predicted"] - sub["Measured"]).mean())
        rows.append({"Dataset": label, group_col: str(g), "Rows": len(sub),
                     **m, "Mean_Bias_PredMinusMeas": bias})
    return pd.DataFrame(rows)


# =============================================================================
# MODELS
# =============================================================================

def make_xgb(gpu: bool = USE_GPU):
    if not HAS_XGBOOST:
        return None
    params = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                  random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL)
    if gpu:
        params["device"] = f"cuda:{GPU_DEVICE}"
    return XGBRegressor(**params)


def make_lgbm():
    if not HAS_LIGHTGBM:
        return None
    params = dict(random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL, verbose=-1)
    if USE_LIGHTGBM_GPU:
        params.update({"device_type": "gpu"})
    return LGBMRegressor(**params)


def make_catboost(gpu: bool = USE_GPU):
    if not HAS_CATBOOST:
        return None
    params = dict(loss_function="RMSE", verbose=0, random_seed=RANDOM_STATE)
    params.update({"task_type": "GPU", "devices": GPU_DEVICE} if gpu else {"task_type": "CPU"})
    return CatBoostRegressor(**params)


def define_models(feature_set: str) -> Dict[str, Dict[str, Any]]:
    """High-regularization, shallow-tree grids per the advisor plan."""
    models: Dict[str, Dict[str, Any]] = {}
    if HAS_XGBOOST:
        # Stronger regularization to shrink the train-validation gap (low LR + more trees,
        # shallow depth, large min_child_weight, high L1/L2, gamma, aggressive subsampling).
        models["XGBoost"] = {
            "estimator": make_xgb(USE_GPU),
            "n_iter": N_ITER_XGB,
            "params": {
                "model__n_estimators": [600, 900, 1200],
                "model__learning_rate": [0.01, 0.015, 0.02],
                "model__max_depth": [2, 3],
                "model__min_child_weight": [15, 20, 30, 40],
                "model__subsample": [0.60, 0.70, 0.80],
                "model__colsample_bytree": [0.50, 0.60, 0.70],
                "model__gamma": [0.10, 0.20, 0.30],
                "model__reg_alpha": [1.0, 2.0, 4.0, 8.0],
                "model__reg_lambda": [30, 50, 80, 120],
                "model__max_bin": [128, 256],
            },
        }
    if HAS_LIGHTGBM and feature_set in ["VolumetricsB_NoADT_RBR_Both", "SHAP14_RBR_PlusInteractions"]:
        models["LightGBM"] = {
            "estimator": make_lgbm(), "n_iter": N_ITER_OTHER,
            "params": {
                "model__n_estimators": [500, 800, 1200],
                "model__learning_rate": [0.01, 0.015, 0.025, 0.04],
                "model__num_leaves": [7, 15, 31],
                "model__min_child_samples": [20, 40, 60, 80],
                "model__subsample": [0.70, 0.80, 0.90],
                "model__colsample_bytree": [0.60, 0.70, 0.85],
                "model__reg_alpha": [0.0, 0.5, 1.0, 3.0],
                "model__reg_lambda": [5.0, 10.0, 30.0, 60.0],
            },
        }
    if feature_set in ["VolumetricsB_NoADT_RBR_Both"]:
        models["HistGradientBoosting"] = {
            "estimator": HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error"),
            "n_iter": N_ITER_OTHER,
            "params": {
                "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
                "model__max_iter": [300, 500, 800],
                "model__max_leaf_nodes": [15, 20, 31],
                "model__min_samples_leaf": [20, 40, 60],
                "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
            },
        }
    if HAS_CATBOOST and feature_set in ["StructGradDesign_CleanCategorical", "FullExpanded_CleanCategorical"]:
        models["CatBoost"] = {
            "estimator": make_catboost(USE_GPU), "n_iter": N_ITER_OTHER,
            "params": {
                "model__iterations": [500, 800, 1200],
                "model__learning_rate": [0.015, 0.025, 0.04, 0.06],
                "model__depth": [3, 4, 5, 6],
                "model__l2_leaf_reg": [5, 10, 20, 40, 80],
                "model__random_strength": [0.5, 1.0, 2.0, 4.0],
            },
        }

    # ---- Additional TREE / BAGGING models (linear ElasticNet/Huber, SVR, EBM REMOVED:
    #      they added no value — ElasticNet/Huber R2~0.14 on this non-linear target, SVR
    #      unstable and slow, EBM below the boosters. Keeping a tree-only robust lineup.) ----
    rbr_or_shap = ["VolumetricsB_NoADT_RBR_Both", "SHAP12_RBR"]
    if feature_set in rbr_or_shap:
        # Extremely randomized trees: lowest-variance, hardest-to-overfit tree model.
        models["ExtraTrees"] = {
            "estimator": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL),
            "n_iter": N_ITER_OTHER, "scale": False,
            "params": {
                "model__n_estimators": [400, 800],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
        }
    if feature_set == "VolumetricsB_NoADT_RBR_Both":
        models["RandomForest"] = {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL),
            "n_iter": N_ITER_OTHER, "scale": False,
            "params": {
                "model__n_estimators": [400, 800],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
        }
        # Huber-loss boosting: down-weights the heavy-tail residuals (high-rut/low-rut extremes).
        models["GradientBoostingHuber"] = {
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE, loss="huber"),
            "n_iter": N_ITER_OTHER, "scale": False,
            "params": {
                "model__n_estimators": [300, 500, 800],
                "model__learning_rate": [0.02, 0.03, 0.05],
                "model__max_depth": [2, 3],
                "model__subsample": [0.70, 0.85],
                "model__min_samples_leaf": [10, 20],
            },
        }

    return {k: v for k, v in models.items() if v["estimator"] is not None}


# ---- Log-target helpers (Option B). When LOG_TARGET, the whole pipeline is wrapped in a
#      TransformedTargetRegressor so it TRAINS on log1p(y) but PREDICTS on the original scale.
#      That keeps every downstream metric/plot in original units with no other code changes,
#      except: (a) RandomizedSearch param keys gain a "regressor__" prefix, and (b) SHAP must
#      unwrap the inner pipeline to reach the tree model. ----

def maybe_log_wrap(estimator):
    if LOG_TARGET:
        return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    return estimator


def log_param_grid(params: Dict[str, Any]) -> Dict[str, Any]:
    if LOG_TARGET:
        return {f"regressor__{k}": v for k, v in params.items()}
    return params


def sw_key() -> str:
    """sample_weight fit-kwarg key, prefixed when the pipeline is target-log-wrapped."""
    return "regressor__model__sample_weight" if LOG_TARGET else "model__sample_weight"


def unwrap_pipeline(est):
    """Return the inner sklearn Pipeline whether or not it's wrapped in a TransformedTargetRegressor."""
    if isinstance(est, TransformedTargetRegressor):
        return getattr(est, "regressor_", est.regressor)
    return est


def build_pipeline(estimator, numerical, categorical, scale: bool = False, wrap: bool = True) -> Pipeline:
    # scale=True inserts StandardScaler after preprocessing for distance/penalty-based models.
    # wrap=False keeps a RAW pipeline (used for stacking base learners, so the log wrap is
    # applied ONCE around the whole ensemble instead of around every base learner).
    steps = [("preprocess", build_preprocessor(numerical, categorical))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    return maybe_log_wrap(pipe) if wrap else pipe


def fit_search_with_gpu_fallback(model_name, pipe, params, n_iter, X, y, cv_splits, sample_weight=None):
    params = log_param_grid(params)  # add "regressor__" prefix when the pipeline is log-wrapped
    space_size = int(np.prod([len(v) for v in params.values()])) if params else 1
    n_iter = min(n_iter, max(1, space_size))
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs[sw_key()] = sample_weight

    def _make_search(pp):
        return RandomizedSearchCV(estimator=pp, param_distributions=params, n_iter=n_iter,
                                  scoring="r2", cv=cv_splits, random_state=RANDOM_STATE,
                                  n_jobs=N_JOBS_SEARCH, verbose=0, return_train_score=True,
                                  error_score=np.nan)
    search = _make_search(pipe)
    try:
        search.fit(X, y, **fit_kwargs)
        return search, "GPU_or_requested"
    except Exception as e:
        if USE_GPU and GPU_FALLBACK_TO_CPU and model_name in ["XGBoost", "CatBoost"] and is_gpu_error(e):
            print(f"GPU failed for {model_name}; retrying on CPU. {type(e).__name__}: {e}")
            cpu_est = make_xgb(False) if model_name == "XGBoost" else make_catboost(False)
            inner = unwrap_pipeline(pipe)
            cpu_pipe = maybe_log_wrap(Pipeline([("preprocess", inner.named_steps["preprocess"]), ("model", cpu_est)]))
            search = _make_search(cpu_pipe)
            search.fit(X, y, **fit_kwargs)
            return search, "CPU_fallback"
        raise


def oof_predict(estimator, X, y, splits, sample_weight=None):
    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold, (tr, va) in enumerate(splits, start=1):
        est = clone(estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs[sw_key()] = np.asarray(sample_weight)[tr]
        est.fit(X.iloc[tr], y.iloc[tr], **fit_kwargs)
        pred = est.predict(X.iloc[va])
        oof[va] = pred
        m = metrics(y.iloc[va], pred)
        fold_rows.append({"Fold": fold, "Train_Rows": int(len(tr)), "OOF_Validation_Rows": int(len(va)),
                          "Fold_R2": m["R2"], "Fold_RMSE": m["RMSE"], "Fold_MAE": m["MAE"]})
    return oof, pd.DataFrame(fold_rows)


def rut_sample_weights(y) -> np.ndarray:
    yv = np.asarray(y, dtype=float)
    q80 = np.nanquantile(yv, 0.80)
    q90 = np.nanquantile(yv, 0.90)
    w = np.ones(len(yv), dtype=float)
    w[yv >= q80] = 1.4
    w[yv >= q90] = 1.8
    return w


def repeated_cv_robust(estimator, X, y, repeats=REPEATED_CV_REPEATS) -> Dict[str, float]:
    """Repeated KFold on the 80% dev set for a stable generalization estimate (no test leakage)."""
    scores = []
    for r in range(repeats):
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
        for tr, va in cv.split(X):
            est = clone(estimator)
            est.fit(X.iloc[tr], y.iloc[tr])
            scores.append(r2_score(y.iloc[va], est.predict(X.iloc[va])))
    scores = np.array(scores, dtype=float)
    return {"RepeatedCV_Mean_R2": float(scores.mean()), "RepeatedCV_SD_R2": float(scores.std(ddof=1)),
            "RepeatedCV_Min_R2": float(scores.min()), "RepeatedCV_N": int(len(scores))}


def nested_cv(feature_set, model_name, X_dev, y_dev, numerical, categorical):
    """Proper nested CV on the dev set: outer RepeatedKFold for an UNBIASED generalization
    estimate, inner RandomizedSearchCV for tuning inside each outer fold (no leakage)."""
    spec = define_models(feature_set).get(model_name)
    if spec is None:
        return pd.DataFrame(), pd.DataFrame()
    outer = RepeatedKFold(n_splits=NESTED_CV_OUTER_SPLITS, n_repeats=NESTED_CV_OUTER_REPEATS, random_state=RANDOM_STATE)
    space = int(np.prod([len(v) for v in spec["params"].values()])) if spec["params"] else 1
    n_iter = min(NESTED_CV_INNER_NITER, max(1, space))
    rows = []
    for i, (tr, te) in enumerate(outer.split(X_dev), start=1):
        Xtr, Xte = X_dev.iloc[tr], X_dev.iloc[te]
        ytr, yte = y_dev.iloc[tr], y_dev.iloc[te]
        inner = KFold(n_splits=NESTED_CV_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + i)
        pipe = build_pipeline(clone(spec["estimator"]), numerical, categorical, scale=spec.get("scale", False))
        try:
            search = RandomizedSearchCV(pipe, log_param_grid(spec["params"]), n_iter=n_iter, scoring="r2", cv=inner,
                                        random_state=RANDOM_STATE, n_jobs=N_JOBS_SEARCH, error_score=np.nan)
            search.fit(Xtr, ytr)
            r2 = r2_score(yte, search.best_estimator_.predict(Xte))
        except Exception as e:
            print(f"  Nested outer fold {i} failed: {type(e).__name__}: {e}")
            r2 = np.nan
        rows.append({"OuterFold": i, "Test_R2": r2, "Inner_n_iter": n_iter})
    d = pd.DataFrame(rows)
    s = d["Test_R2"].dropna()
    summary = pd.DataFrame([{
        "Feature_Set": feature_set, "Model": model_name,
        "NestedCV_Mean_R2": float(s.mean()) if len(s) else np.nan,
        "NestedCV_SD_R2": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "NestedCV_Min_R2": float(s.min()) if len(s) else np.nan,
        "NestedCV_Outer_Folds": int(len(d)),
        "Design": f"outer {NESTED_CV_OUTER_SPLITS}x{NESTED_CV_OUTER_REPEATS}, inner {NESTED_CV_INNER_SPLITS}-fold, n_iter={n_iter}",
    }])
    return d, summary


def train_candidate(feature_set, model_name, model_spec, X_train, y_train, X_val, y_val,
                    numerical, categorical, cv_splits, cv_name, weighted=False):
    wtxt = " | high-rut weighted" if weighted else ""
    print(f"  Training {feature_set} | {model_name}{wtxt}")
    pipe = build_pipeline(clone(model_spec["estimator"]), numerical, categorical, scale=model_spec.get("scale", False))
    weights = rut_sample_weights(y_train) if weighted and model_name == "XGBoost" else None
    search, device_used = fit_search_with_gpu_fallback(
        model_name, pipe, model_spec["params"], model_spec["n_iter"], X_train, y_train, cv_splits, sample_weight=weights)
    best_est = search.best_estimator_
    oof, folds = oof_predict(best_est, X_train, y_train, cv_splits, sample_weight=weights)

    candidate_est = clone(best_est)
    fit_kwargs = {sw_key(): weights} if weights is not None else {}
    candidate_est.fit(X_train, y_train, **fit_kwargs)

    train_pred = candidate_est.predict(X_train)
    val_pred = candidate_est.predict(X_val)
    tm, om, vm = metrics(y_train, train_pred), metrics(y_train, oof), metrics(y_val, val_pred)
    gap_train_val = tm["R2"] - vm["R2"]
    fold_sd = float(folds["Fold_R2"].std(ddof=1)) if len(folds) > 1 else np.nan
    bf_oof = best_fit(y_train, oof)
    bf_val = best_fit(y_val, val_pred)
    label = f"{feature_set} | {model_name}" + (" | WeightedHighRut" if weighted else "")

    row = {
        "Target": TARGET, "Feature_Set": feature_set, "Model": model_name,
        "Weighted_High_Rut": bool(weighted), "Label": label, "Device_Used": device_used,
        "N_Features_Raw": int(X_train.shape[1]), "Features_Used": ", ".join(X_train.columns),
        "Numerical_Features": ", ".join(numerical), "Categorical_Features": ", ".join(categorical),
        "CV_Method_on_Training70": cv_name,
        "Train_R2": tm["R2"], "Train_RMSE": tm["RMSE"], "Train_MAE": tm["MAE"],
        "TrainOOF_R2": om["R2"], "TrainOOF_RMSE": om["RMSE"], "TrainOOF_MAE": om["MAE"],
        "Validation_R2": vm["R2"], "Validation_RMSE": vm["RMSE"], "Validation_MAE": vm["MAE"],
        "Validation_HighRut_MAE_q80": high_rut_mae(y_val, val_pred, 0.80),
        "Gap_TrainMinusValidation_R2": float(gap_train_val),
        "Gap_TrainMinusTrainOOF_R2": float(tm["R2"] - om["R2"]),
        "TrainOOF_Fold_R2_SD": fold_sd, "TrainOOF_Fold_R2_Min": float(folds["Fold_R2"].min()) if len(folds) else np.nan,
        "Robust_Selection_Score": robust_score(vm["R2"], om["R2"], gap_train_val, fold_sd, X_train.shape[1]),
        "Validation_BestFit_Equation": bf_val["equation"], "TrainOOF_BestFit_Equation": bf_oof["equation"],
        "Meets_Val_R2_0.80": bool(vm["R2"] >= TARGET_VAL_R2),
        "Best_Params": json.dumps(search.best_params_, default=str),
    }
    row.update(relative_error_summary(y_val, val_pred, "Validation_"))

    pred_parts = []
    for split_name, ys, ps in [("Train70", y_train, train_pred), ("TrainOOF70", y_train, oof), ("Validation10", y_val, val_pred)]:
        d = pd.DataFrame({"Dataset": split_name, "Feature_Set": feature_set, "Model": model_name,
                          "Weighted_High_Rut": bool(weighted), "Measured": np.asarray(ys, float),
                          "Predicted": np.asarray(ps, float)})
        pred_parts.append(add_error_columns(d))
    pred_df = pd.concat(pred_parts, ignore_index=True)

    for c, v in [("Target", TARGET), ("Feature_Set", feature_set), ("Model", model_name), ("Weighted_High_Rut", bool(weighted))]:
        folds.insert(0, c, v)
    return row, pred_df, folds, best_est, candidate_est, search.best_params_


# =============================================================================
# STACKING ENSEMBLE
# =============================================================================

def build_stacking(best_params_by_model: Dict[str, Dict[str, Any]], numerical, categorical):
    """Build an OOF-safe stacking regressor from the tuned base models (RidgeCV meta)."""
    estimators = []
    for name, bp in best_params_by_model.items():
        # strip both the log-wrap prefix and the pipeline "model__" prefix to get raw estimator params
        clean = {k.replace("regressor__model__", "").replace("model__", ""): v for k, v in bp.items()}
        if name == "XGBoost" and HAS_XGBOOST:
            base = make_xgb(USE_GPU)
        elif name == "LightGBM" and HAS_LIGHTGBM:
            base = make_lgbm()
        elif name == "HistGradientBoosting":
            base = HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error")
        elif name == "CatBoost" and HAS_CATBOOST:
            base = make_catboost(USE_GPU)
        elif name == "ExtraTrees":
            base = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL)
        elif name == "RandomForest":
            base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL)
        elif name == "GradientBoostingHuber":
            base = GradientBoostingRegressor(random_state=RANDOM_STATE, loss="huber")
        else:
            # Penalized/robust-linear, SVR, EBM: kept as standalone candidates, not stacked
            # (they would need scaling inside the stack); skip to keep the ensemble tree-only.
            continue
        try:
            base.set_params(**clean)
        except Exception:
            pass
        # wrap=False: base learners stay RAW; the log transform is applied ONCE around the
        # whole ensemble below (so we don't log-transform the target twice).
        estimators.append((name, build_pipeline(base, numerical, categorical, wrap=False)))
    if len(estimators) < 2:
        return None
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
        cv=CV_FOLDS, n_jobs=N_JOBS_SEARCH, passthrough=False)
    return maybe_log_wrap(stack)


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def statistical_analysis(df: pd.DataFrame, y: pd.Series, feature_list: List[str], out_dir: Path):
    cols = [c for c in feature_list if c in df.columns]
    num = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.85]
    work = df[num].apply(pd.to_numeric, errors="coerce").copy()
    work[TARGET] = pd.to_numeric(y, errors="coerce").values

    desc = work.describe().T
    desc["skew"] = work.skew(numeric_only=True)
    desc["kurtosis"] = work.kurtosis(numeric_only=True)

    corr = work.corr(method="pearson")
    plt.figure(figsize=(min(1.0 + 0.55 * len(work.columns), 18), min(1.0 + 0.55 * len(work.columns), 18)))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Pearson correlation matrix (features + target)")
    save_fig(out_dir / "correlation_matrix.png")

    corr_with_target = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    plt.figure(figsize=(8, max(4, len(corr_with_target) * 0.3)))
    cc = corr_with_target.sort_values()
    plt.barh(cc.index, cc.values)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"Pearson correlation with {TARGET}")
    plt.title("Univariate correlation with target")
    save_fig(out_dir / "correlation_with_target.png")

    # VIF multicollinearity.
    vif_df = pd.DataFrame()
    Xn = work[num].dropna()
    if len(Xn) > len(num) + 2 and len(num) >= 2:
        try:
            Xs = StandardScaler().fit_transform(Xn.values)
            if HAS_STATSMODELS:
                vif_vals = [variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])]
            else:
                # fallback: VIF_i = 1/(1-R2_i) via correlation inverse diagonal
                inv = np.linalg.pinv(np.corrcoef(Xs, rowvar=False))
                vif_vals = list(np.diag(inv))
            vif_df = pd.DataFrame({"Feature": num, "VIF": vif_vals}).sort_values("VIF", ascending=False)
        except Exception as e:
            print(f"VIF computation failed: {type(e).__name__}: {e}")
    return desc.reset_index().rename(columns={"index": "Feature"}), corr_with_target.reset_index().rename(
        columns={"index": "Feature", TARGET: "Corr_with_target"}), vif_df


# =============================================================================
# DIAGNOSTIC GRAPHS
# =============================================================================

def plot_fit(y_true, y_pred, title, units, path) -> Dict[str, Any]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = metrics(y_true, y_pred)
    bf = best_fit(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=22)
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx], "--", linewidth=2, label="Ideal 1:1 (45 degrees)")
    if np.isfinite(bf["slope"]):
        xs = np.linspace(mn, mx, 100)
        plt.plot(xs, bf["slope"] * xs + bf["intercept"], linewidth=2, label="Best-fit line")
    plt.xlabel(f"Measured {TARGET} ({units})")
    plt.ylabel(f"Predicted {TARGET} ({units})")
    plt.title(f"{title}\nR2={m['R2']:.3f}, RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}\n{bf['equation']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(path)
    return {"Graph": path.name, "Type": "Best-fit", **m, **bf}


def plot_hist(values, title, xlabel, path, bins=35):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    save_fig(path)


def plot_residuals(pred_df, split_name, out_dir) -> List[Dict[str, Any]]:
    graphs = []
    d = pred_df[pred_df["Dataset"] == split_name].copy()
    if d.empty:
        return graphs
    base = safe_name(split_name)
    graphs.append(plot_fit(d["Measured"], d["Predicted"], f"{split_name} measured vs predicted", "mm",
                           out_dir / f"{base}_best_fit.png"))
    for xcol, xlab, suffix in [("Predicted", "Predicted Rut_20k (mm)", "vs_predicted"),
                                ("Measured", "Measured Rut_20k (mm)", "vs_measured")]:
        plt.figure(figsize=(7, 5))
        plt.scatter(d[xcol], d["Residual"], alpha=0.6, s=22)
        plt.axhline(0, ls="--", lw=2)
        plt.xlabel(xlab)
        plt.ylabel("Residual: predicted - measured (mm)")
        plt.title(f"{split_name} residuals {suffix.replace('_', ' ')}")
        plt.grid(True, alpha=0.3)
        p = out_dir / f"{base}_residuals_{suffix}.png"
        save_fig(p)
        graphs.append({"Graph": p.name, "Type": f"Residuals {suffix}"})
    p = out_dir / f"{base}_residual_histogram.png"
    plot_hist(d["Residual"], f"{split_name} residual distribution", "Residual (mm)", p)
    graphs.append({"Graph": p.name, "Type": "Residual histogram"})
    p = out_dir / f"{base}_abs_relative_error_hist.png"
    plot_hist(d["Abs_Relative_Error_pct"], f"{split_name} absolute relative error", "Absolute relative error (%)", p)
    graphs.append({"Graph": p.name, "Type": "Abs relative error hist"})
    plt.figure(figsize=(7, 5))
    plt.scatter(d["Measured"], d["Relative_Error_pct"], alpha=0.6, s=22)
    plt.axhline(0, ls="--", lw=2)
    plt.xlabel("Measured Rut_20k (mm)")
    plt.ylabel("Relative error (%)")
    plt.title(f"{split_name} relative error vs measured")
    plt.grid(True, alpha=0.3)
    p = out_dir / f"{base}_relative_error_vs_measured.png"
    save_fig(p)
    graphs.append({"Graph": p.name, "Type": "Relative error vs measured"})
    if HAS_SCIPY:
        plt.figure(figsize=(6, 6))
        stats.probplot(d["Residual"].dropna(), dist="norm", plot=plt)
        plt.title(f"{split_name} Q-Q plot of residuals")
        p = out_dir / f"{base}_qq_plot.png"
        save_fig(p)
        graphs.append({"Graph": p.name, "Type": "Q-Q residuals"})
    return graphs


def plot_fold_performance(folds, out_dir) -> List[Dict[str, Any]]:
    graphs = []
    if folds is None or folds.empty:
        return graphs
    for metric in ["Fold_R2", "Fold_RMSE", "Fold_MAE"]:
        plt.figure(figsize=(8, 5))
        plt.bar(folds["Fold"].astype(str), folds[metric])
        plt.xlabel("CV fold")
        plt.ylabel(metric)
        plt.title(f"Final selected model - {metric} by training CV fold")
        plt.grid(True, axis="y", alpha=0.3)
        p = out_dir / f"selected_model_{safe_name(metric)}_by_fold.png"
        save_fig(p)
        graphs.append({"Graph": p.name, "Type": f"Fold {metric}"})
    return graphs


def plot_model_comparison(results_df, out_dir) -> List[Dict[str, Any]]:
    graphs = []
    if results_df.empty:
        return graphs
    d = results_df.sort_values("Validation_R2", ascending=False).head(25).copy()
    d["Short_Label"] = d["Feature_Set"] + " | " + d["Model"] + np.where(d["Weighted_High_Rut"], " | W", "")
    for metric in ["Validation_R2", "TrainOOF_R2", "Gap_TrainMinusValidation_R2", "Robust_Selection_Score"]:
        if metric not in d.columns:
            continue
        plt.figure(figsize=(10, max(5, len(d) * 0.32)))
        dd = d.sort_values(metric, ascending=True)
        plt.barh(dd["Short_Label"], dd[metric])
        plt.xlabel(metric)
        plt.title(f"Candidate model comparison - {metric}")
        plt.grid(True, axis="x", alpha=0.3)
        p = out_dir / f"candidate_comparison_{safe_name(metric)}.png"
        save_fig(p)
        graphs.append({"Graph": p.name, "Type": f"Model comparison {metric}"})
    return graphs


def plot_feature_set_elbow(results_df, out_dir) -> List[Dict[str, Any]]:
    graphs = []
    if results_df.empty:
        return graphs
    d = results_df.groupby("Feature_Set", as_index=False).agg(
        N_Features_Raw=("N_Features_Raw", "first"),
        Best_Validation_R2=("Validation_R2", "max"),
        Best_TrainOOF_R2=("TrainOOF_R2", "max")).sort_values("N_Features_Raw")
    plt.figure(figsize=(8, 5))
    plt.plot(d["N_Features_Raw"], d["Best_Validation_R2"], marker="o", label="Best validation R2")
    plt.plot(d["N_Features_Raw"], d["Best_TrainOOF_R2"], marker="o", label="Best train-OOF R2")
    for _, r in d.iterrows():
        plt.annotate(str(r["Feature_Set"]), (r["N_Features_Raw"], r["Best_Validation_R2"]), fontsize=7, rotation=25)
    plt.xlabel("Number of raw features")
    plt.ylabel("R2")
    plt.title("Model-complexity / elbow curve by feature-set size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p = out_dir / "model_complexity_elbow.png"
    save_fig(p)
    graphs.append({"Graph": p.name, "Type": "Complexity elbow"})
    return graphs


def transformed_matrix(estimator, X):
    pre = unwrap_pipeline(estimator).named_steps.get("preprocess")
    X_t = pre.transform(X)
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"x{i}" for i in range(X_t.shape[1])]
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    return np.asarray(X_t), names


def plot_learning_curve(estimator, X, y, out_dir) -> List[Dict[str, Any]]:
    graphs = []
    if not RUN_LEARNING_CURVE:
        return graphs
    try:
        bins = make_target_bins(y)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_splits = list(cv.split(np.zeros(len(y)), bins))
        train_sizes, tr_scores, va_scores = learning_curve(
            estimator=clone(estimator), X=X, y=y, train_sizes=np.linspace(0.25, 1.0, 6),
            cv=cv_splits, scoring="r2", n_jobs=N_JOBS_SEARCH)
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, tr_scores.mean(1), marker="o", label="Training R2")
        plt.plot(train_sizes, va_scores.mean(1), marker="o", label="CV validation R2")
        plt.fill_between(train_sizes, tr_scores.mean(1) - tr_scores.std(1), tr_scores.mean(1) + tr_scores.std(1), alpha=0.15)
        plt.fill_between(train_sizes, va_scores.mean(1) - va_scores.std(1), va_scores.mean(1) + va_scores.std(1), alpha=0.15)
        plt.xlabel("Training examples")
        plt.ylabel("R2")
        plt.title("Learning curve for final selected model")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = out_dir / "learning_curve.png"
        save_fig(p)
        graphs.append({"Graph": p.name, "Type": "Learning curve"})
    except Exception as e:
        print(f"Learning curve failed: {type(e).__name__}: {e}")
    return graphs


def plot_bias_variance(best_params, numerical, categorical, X_train, y_train, X_val, y_val, out_dir):
    graphs = []
    if not RUN_BIAS_VARIANCE_CURVES or not HAS_XGBOOST:
        return graphs
    bp = {str(k).replace("regressor__model__", "").replace("model__", ""): v for k, v in best_params.items()}
    base = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL)
    if USE_GPU:
        base["device"] = f"cuda:{GPU_DEVICE}"
    base.update(bp)
    curves = {"max_depth": [1, 2, 3, 4, 5, 6], "n_estimators": [200, 400, 700, 900, 1200, 1500]}
    for param, values in curves.items():
        rows = []
        for v in values:
            p = base.copy()
            p[param] = v
            try:
                est = build_pipeline(XGBRegressor(**p), numerical, categorical)
                est.fit(X_train, y_train)
                tr = metrics(y_train, est.predict(X_train))["R2"]
                va = metrics(y_val, est.predict(X_val))["R2"]
                rows.append({"value": v, "Train_R2": tr, "Validation_R2": va, "Gap": tr - va})
            except Exception as e:
                if is_gpu_error(e):
                    p.pop("device", None)
                    try:
                        est = build_pipeline(XGBRegressor(**p), numerical, categorical)
                        est.fit(X_train, y_train)
                        tr = metrics(y_train, est.predict(X_train))["R2"]
                        va = metrics(y_val, est.predict(X_val))["R2"]
                        rows.append({"value": v, "Train_R2": tr, "Validation_R2": va, "Gap": tr - va})
                    except Exception as e2:
                        print(f"BV point failed {param}={v}: {e2}")
                else:
                    print(f"BV point failed {param}={v}: {e}")
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(d["value"], d["Train_R2"], marker="o", label="Training R2")
        plt.plot(d["value"], d["Validation_R2"], marker="o", label="Validation R2")
        plt.plot(d["value"], d["Gap"], marker="o", label="Gap")
        plt.xlabel(param)
        plt.ylabel("R2 / gap")
        plt.title(f"Bias-variance tradeoff curve: {param}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = out_dir / f"bias_variance_{param}.png"
        save_fig(p)
        d.to_csv(PATHS["tables"] / f"bias_variance_{param}.csv", index=False)
        graphs.append({"Graph": p.name, "Type": f"Bias-variance {param}"})
    return graphs


def run_permutation(estimator, X_val, y_val, out_dir) -> pd.DataFrame:
    if not RUN_PERMUTATION_IMPORTANCE:
        return pd.DataFrame()
    try:
        perm = permutation_importance(estimator, X_val, y_val, n_repeats=10,
                                      random_state=RANDOM_STATE, scoring="r2", n_jobs=1)
        imp = pd.DataFrame({"Feature": X_val.columns, "Importance_Mean": perm.importances_mean,
                            "Importance_SD": perm.importances_std}).sort_values("Importance_Mean", ascending=False)
        plt.figure(figsize=(8, max(5, len(imp.head(20)) * 0.30)))
        dd = imp.head(20).sort_values("Importance_Mean")
        plt.barh(dd["Feature"], dd["Importance_Mean"])
        plt.xlabel("Permutation importance (decrease in R2)")
        plt.title("Final model permutation importance on validation set")
        plt.grid(True, axis="x", alpha=0.3)
        save_fig(out_dir / "permutation_importance.png")
        return imp
    except Exception as e:
        print(f"Permutation importance failed: {type(e).__name__}: {e}")
        return pd.DataFrame()


def run_shap(estimator, X_explain, pred_df, out_dir):
    graphs = []
    if not RUN_SHAP or not HAS_SHAP:
        return pd.DataFrame(), graphs, None, None
    try:
        X_t, names = transformed_matrix(estimator, X_explain)
        idx_used = np.arange(X_t.shape[0])
        if X_t.shape[0] > MAX_SHAP_ROWS:
            rng = np.random.default_rng(RANDOM_STATE)
            idx_used = rng.choice(idx_used, size=MAX_SHAP_ROWS, replace=False)
            X_t = X_t[idx_used]
        model = unwrap_pipeline(estimator).named_steps["model"]
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_t)
        mean_abs = np.abs(sv.values).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": names, "MeanAbsSHAP": mean_abs}).sort_values("MeanAbsSHAP", ascending=False)

        plt.figure(figsize=(8, 7))
        shap.summary_plot(sv.values, X_t, feature_names=names, plot_type="bar", show=False, max_display=20)
        save_fig(out_dir / "shap_bar.png")
        graphs.append({"Graph": "shap_bar.png", "Type": "SHAP bar"})

        plt.figure(figsize=(8, 7))
        shap.summary_plot(sv.values, X_t, feature_names=names, show=False, max_display=20)
        save_fig(out_dir / "shap_beeswarm.png")
        graphs.append({"Graph": "shap_beeswarm.png", "Type": "SHAP beeswarm"})

        d = pred_df.reset_index(drop=True).copy()
        if len(d) > 0:
            cands = {"median_measured": int((d["Measured"] - d["Measured"].median()).abs().idxmin()),
                     "highest_measured": int(d["Measured"].idxmax()),
                     "lowest_abs_error": int(d["Abs_Error"].idxmin()),
                     "highest_abs_error": int(d["Abs_Error"].idxmax())}
            pos_map = {orig: i for i, orig in enumerate(idx_used)}
            for label, i in cands.items():
                if i not in pos_map:
                    continue
                plt.figure(figsize=(9, 6))
                shap.waterfall_plot(sv[pos_map[i]], max_display=16, show=False)
                save_fig(out_dir / f"shap_waterfall_{safe_name(label)}.png")
                graphs.append({"Graph": f"shap_waterfall_{safe_name(label)}.png", "Type": f"SHAP waterfall {label}"})
        return shap_df, graphs, sv, (names, idx_used)
    except Exception as e:
        print(f"SHAP failed: {type(e).__name__}: {e}")
        return pd.DataFrame(), graphs, None, None


def shap_by_rbr_band(sv, names_and_idx, rbr_band_series, out_dir) -> pd.DataFrame:
    """Break SHAP mean|value| out by RBR band to see which drivers dominate per RBR range."""
    if sv is None or names_and_idx is None:
        return pd.DataFrame()
    names, idx_used = names_and_idx
    bands = np.asarray(rbr_band_series.reset_index(drop=True).iloc[np.asarray(idx_used, dtype=int)])
    vals = np.abs(sv.values)
    rows = []
    for band in pd.unique(bands):
        mask = bands == band
        if mask.sum() < 3:
            continue
        m = vals[mask].mean(axis=0)
        top = pd.Series(m, index=names).sort_values(ascending=False).head(8)
        for feat, v in top.items():
            rows.append({"RBR_band": str(band), "Feature": feat, "MeanAbsSHAP": float(v), "N": int(mask.sum())})
    out = pd.DataFrame(rows)
    if not out.empty:
        try:
            piv = out.pivot_table(index="Feature", columns="RBR_band", values="MeanAbsSHAP", fill_value=0.0)
            piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
            piv.plot(kind="barh", figsize=(9, max(5, len(piv) * 0.4)))
            plt.xlabel("mean |SHAP|")
            plt.title("SHAP driver importance by RBR band")
            plt.gca().invert_yaxis()
            save_fig(out_dir / "shap_importance_by_rbr_band.png")
        except Exception as e:
            print(f"SHAP-by-band plot failed: {type(e).__name__}: {e}")
    return out


def run_pdp(estimator, X, top_features, out_dir) -> List[Dict[str, Any]]:
    graphs = []
    if not RUN_PDP:
        return graphs
    features = [f for f in top_features if f in X.columns][:MAX_PDP_FEATURES]
    for f in features:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(estimator, X, [f], ax=ax, grid_resolution=30)
            ax.set_title(f"1D PDP - {f}")
            p = out_dir / f"pdp_1d_{safe_name(f)}.png"
            save_fig(p)
            graphs.append({"Graph": p.name, "Type": f"1D PDP {f}"})
        except Exception as e:
            print(f"PDP failed for {f}: {type(e).__name__}: {e}")
    pair_candidates = [
        ("RBR_JMF_fraction", "PG_HighTemp"), ("RBR_JMF_fraction", "Absorption"),
        ("PG_HighTemp", "RAP_pct_x_ACinRAP"), ("SandEq", "Dust_Binder"), ("Va", "Gmm"),
    ]
    n = 0
    for a, b in pair_candidates:
        if a in X.columns and b in X.columns and n < MAX_PDP_PAIRS:
            try:
                fig, ax = plt.subplots(figsize=(7, 6))
                PartialDependenceDisplay.from_estimator(estimator, X, [(a, b)], ax=ax, grid_resolution=25)
                ax.set_title(f"2D PDP - {a} x {b}")
                p = out_dir / f"pdp_2d_{safe_name(a)}_x_{safe_name(b)}.png"
                save_fig(p)
                graphs.append({"Graph": p.name, "Type": f"2D PDP {a} x {b}"})
                n += 1
            except Exception as e:
                print(f"2D PDP failed for {a},{b}: {type(e).__name__}: {e}")
    return graphs


def plot_roc_risk(pred_df, out_dir) -> List[Dict[str, Any]]:
    graphs = []
    if not RUN_ROC_RISK_SCREENING:
        return graphs
    for ds in ["Validation10", "LockedTest20"]:
        d = pred_df[pred_df["Dataset"] == ds].copy()
        if d.empty:
            continue
        y_cls = (d["Measured"] >= RUT_RISK_THRESHOLD_MM).astype(int)
        if y_cls.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_cls, d["Predicted"].astype(float))
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"Threshold ROC: {ds}, high rut >= {RUT_RISK_THRESHOLD_MM} mm")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = out_dir / f"roc_{safe_name(ds)}.png"
        save_fig(p)
        graphs.append({"Graph": p.name, "Type": "ROC high-rut risk", "AUC": roc_auc})
    return graphs


# =============================================================================
# FINAL REFIT + LOCKED TEST (scored once)
# =============================================================================

def final_refit_and_test(final_estimator, X_train, y_train, X_val, y_val, X_test, y_test):
    X_dev = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_dev = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    final_est = clone(final_estimator)
    final_est.fit(X_dev, y_dev)

    pred_parts = []
    for ds, Xs, ys in [("Train70", X_train, y_train), ("Validation10", X_val, y_val),
                       ("Dev80_FinalFit", X_dev, y_dev), ("LockedTest20", X_test, y_test)]:
        ps = final_est.predict(Xs)
        d = pd.DataFrame({"Dataset": ds, "Measured": np.asarray(ys, float), "Predicted": np.asarray(ps, float)})
        pred_parts.append(add_error_columns(d))
    pred_df = pd.concat(pred_parts, ignore_index=True)

    rows = []
    for ds, sub in pred_df.groupby("Dataset"):
        m = metrics(sub["Measured"], sub["Predicted"])
        rel = relative_error_summary(sub["Measured"], sub["Predicted"], "")
        bf = best_fit(sub["Measured"], sub["Predicted"])
        rows.append({"Dataset": ds, "Rows": len(sub), **m,
                     "HighRut_MAE_q80": high_rut_mae(sub["Measured"], sub["Predicted"], 0.80),
                     **rel, "BestFit_Equation": bf["equation"]})
    return final_est, X_dev, y_dev, pred_df, pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("RUT_20K WORKFLOW v2 — 70/10/20 + TRUE RBR + HIGH REG + STACKING + FULL DIAGNOSTICS")
    print("=" * 100)
    print(f"Goal: push validation R2 toward {TARGET_VAL_R2:.2f}; keep the 20% test locked/hidden.")
    print(gpu_status_string())
    print(f"Split: {int(TRAIN_SIZE*100)}% train, {int(VALIDATION_SIZE*100)}% val, {int(TEST_SIZE*100)}% locked test")
    print(f"XGBoost={HAS_XGBOOST} LightGBM={HAS_LIGHTGBM} CatBoost={HAS_CATBOOST} SHAP={HAS_SHAP} | QUICK_SMOKE_TEST={QUICK_SMOKE_TEST}")
    print(f"Output: {OUTPUT_FOLDER}")

    df, y, input_path = load_data()
    train_idx, val_idx, test_idx = stratified_70_10_20_split(df, y)

    split_summary = pd.DataFrame([
        {"Split": "Train70", "Rows": len(train_idx), "Percent": 100 * len(train_idx) / len(df), "Purpose": "fit/tune (5-fold CV)"},
        {"Split": "Validation10", "Rows": len(val_idx), "Percent": 100 * len(val_idx) / len(df), "Purpose": "model/feature selection"},
        {"Split": "LockedTest20", "Rows": len(test_idx), "Percent": 100 * len(test_idx) / len(df), "Purpose": "final one-time test (HIDDEN)"},
    ])
    print("\nSplit summary:")
    print(split_summary.to_string(index=False))

    df.iloc[train_idx].to_excel(PATHS["splits"] / "train_70pct_rows.xlsx", index=False)
    df.iloc[val_idx].to_excel(PATHS["splits"] / "validation_10pct_rows.xlsx", index=False)
    df.iloc[test_idx].to_excel(PATHS["splits"] / "LOCKED_test_20pct_DO_NOT_USE_FOR_TUNING.xlsx", index=False)

    # Statistical analysis on DEV data only (no test leakage).
    dev_idx_all = np.concatenate([train_idx, val_idx])
    stat_features = sorted(set(VOLUMETRICS_B_RBR_BOTH + PHYSICAL_INTERACTIONS))
    desc_df, corr_df, vif_df = statistical_analysis(
        df.iloc[dev_idx_all].reset_index(drop=True), y.iloc[dev_idx_all].reset_index(drop=True),
        stat_features, PATHS["figures"] / "statistics")

    results_rows, pred_rows, folds_rows = [], [], []
    trained_objects: Dict[str, Dict[str, Any]] = {}
    best_params_for_stack: Dict[str, Dict[str, Any]] = {}
    stack_context = None
    graph_rows, feature_set_rows = [], []

    for fs_name in FEATURE_SETS_TO_RUN:
        requested = FEATURE_SETS[fs_name]
        X_all, numerical, categorical, available, missing = get_X(df, requested)
        feature_set_rows.append({"Feature_Set": fs_name, "Requested": ", ".join(requested),
                                 "Available": ", ".join(available), "Missing": ", ".join(missing) or "None",
                                 "N_Available": len(available), "Numerical": ", ".join(numerical),
                                 "Categorical": ", ".join(categorical)})
        if not available:
            print(f"Skipping {fs_name}: no available features.")
            continue

        X_train, X_val, X_test = (X_all.iloc[train_idx].reset_index(drop=True),
                                  X_all.iloc[val_idx].reset_index(drop=True),
                                  X_all.iloc[test_idx].reset_index(drop=True))
        y_train, y_val, y_test = (y.iloc[train_idx].reset_index(drop=True),
                                  y.iloc[val_idx].reset_index(drop=True),
                                  y.iloc[test_idx].reset_index(drop=True))
        cv_splits, cv_name = make_cv_splits_for_training(y_train)

        print("\n" + "=" * 100)
        print(f"Feature set: {fs_name} | available={len(available)} | missing={missing or 'None'}")
        print("=" * 100)

        for model_name, spec in define_models(fs_name).items():
            for weighted in ([False, True] if (RUN_HIGH_RUT_WEIGHTING and model_name == "XGBoost" and fs_name in
                              ["VolumetricsB_NoADT_RBR_Both", "SHAP12_RBR_PlusInteractions", "SHAP14_RBR_PlusInteractions"])
                              else [False]):
                try:
                    row, pred, folds, best_est, fitted, best_params = train_candidate(
                        fs_name, model_name, spec, X_train, y_train, X_val, y_val,
                        numerical, categorical, cv_splits, cv_name, weighted=weighted)
                    results_rows.append(row)
                    pred_rows.append(pred)
                    folds_rows.append(folds)
                    trained_objects[row["Label"]] = {
                        "best_estimator": best_est, "fitted": fitted, "cand_pred": pred,
                        "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
                        "X_test": X_test, "y_test": y_test, "numerical": numerical, "categorical": categorical,
                        "feature_set": fs_name, "model_name": model_name, "best_params": best_params, "folds": folds,
                    }
                    if not weighted and model_name not in best_params_for_stack:
                        # Use the RBR_Both set as the stacking feature space (consistent columns).
                        if fs_name == "VolumetricsB_NoADT_RBR_Both":
                            best_params_for_stack[model_name] = best_params
                            stack_context = {"numerical": numerical, "categorical": categorical,
                                             "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
                                             "X_test": X_test, "y_test": y_test, "feature_set": fs_name}
                    print(f"    -> Val R2={row['Validation_R2']:.4f} | OOF R2={row['TrainOOF_R2']:.4f} | "
                          f"Train R2={row['Train_R2']:.4f} | Gap={row['Gap_TrainMinusValidation_R2']:.4f} | "
                          f"Robust={row['Robust_Selection_Score']:.4f}")
                except Exception as e:
                    print(f"    FAILED {fs_name} | {model_name} weighted={weighted}: {type(e).__name__}: {e}")

    # ---- Stacking ensemble on the RBR_Both feature space ----
    if RUN_STACKING and stack_context is not None and len(best_params_for_stack) >= 2:
        try:
            print("\n" + "=" * 100)
            print(f"Stacking ensemble of: {list(best_params_for_stack.keys())} (RidgeCV meta)")
            print("=" * 100)
            sc = stack_context
            stack = build_stacking(best_params_for_stack, sc["numerical"], sc["categorical"])
            if stack is not None:
                stack.fit(sc["X_train"], sc["y_train"])
                train_pred = stack.predict(sc["X_train"])
                val_pred = stack.predict(sc["X_val"])
                tm, vm = metrics(sc["y_train"], train_pred), metrics(sc["y_val"], val_pred)
                gap = tm["R2"] - vm["R2"]
                row = {"Target": TARGET, "Feature_Set": sc["feature_set"], "Model": "StackingRegressor",
                       "Weighted_High_Rut": False, "Label": f"{sc['feature_set']} | StackingRegressor",
                       "Device_Used": "stacking", "N_Features_Raw": int(sc["X_train"].shape[1]),
                       "Features_Used": ", ".join(sc["X_train"].columns),
                       "Numerical_Features": ", ".join(sc["numerical"]), "Categorical_Features": ", ".join(sc["categorical"]),
                       "CV_Method_on_Training70": "StackingRegressor internal cv=5 OOF",
                       "Train_R2": tm["R2"], "Train_RMSE": tm["RMSE"], "Train_MAE": tm["MAE"],
                       "TrainOOF_R2": vm["R2"], "TrainOOF_RMSE": vm["RMSE"], "TrainOOF_MAE": vm["MAE"],
                       "Validation_R2": vm["R2"], "Validation_RMSE": vm["RMSE"], "Validation_MAE": vm["MAE"],
                       "Validation_HighRut_MAE_q80": high_rut_mae(sc["y_val"], val_pred, 0.80),
                       "Gap_TrainMinusValidation_R2": float(gap), "Gap_TrainMinusTrainOOF_R2": float(gap),
                       "TrainOOF_Fold_R2_SD": np.nan, "TrainOOF_Fold_R2_Min": np.nan,
                       "Robust_Selection_Score": robust_score(vm["R2"], vm["R2"], gap, 0.0, sc["X_train"].shape[1]),
                       "Validation_BestFit_Equation": best_fit(sc["y_val"], val_pred)["equation"],
                       "TrainOOF_BestFit_Equation": "NA (stacking)",
                       "Meets_Val_R2_0.80": bool(vm["R2"] >= TARGET_VAL_R2), "Best_Params": "stacking"}
                row.update(relative_error_summary(sc["y_val"], val_pred, "Validation_"))
                results_rows.append(row)
                pp = []
                for sn, ys, ps in [("Train70", sc["y_train"], train_pred), ("Validation10", sc["y_val"], val_pred)]:
                    d = pd.DataFrame({"Dataset": sn, "Feature_Set": sc["feature_set"], "Model": "StackingRegressor",
                                      "Weighted_High_Rut": False, "Measured": np.asarray(ys, float),
                                      "Predicted": np.asarray(ps, float)})
                    pp.append(add_error_columns(d))
                stack_pred_df = pd.concat(pp, ignore_index=True)
                pred_rows.append(stack_pred_df)
                trained_objects[row["Label"]] = {
                    "best_estimator": stack, "fitted": stack, "cand_pred": stack_pred_df,
                    "X_train": sc["X_train"], "y_train": sc["y_train"],
                    "X_val": sc["X_val"], "y_val": sc["y_val"], "X_test": sc["X_test"], "y_test": sc["y_test"],
                    "numerical": sc["numerical"], "categorical": sc["categorical"], "feature_set": sc["feature_set"],
                    "model_name": "StackingRegressor", "best_params": {}, "folds": pd.DataFrame()}
                print(f"    -> Stacking Val R2={vm['R2']:.4f} | Train R2={tm['R2']:.4f} | Gap={gap:.4f}")
        except Exception as e:
            print(f"Stacking failed: {type(e).__name__}: {e}")

    results_df = pd.DataFrame(results_rows).sort_values("Robust_Selection_Score", ascending=False) if results_rows else pd.DataFrame()
    predictions_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    folds_df = pd.concat(folds_rows, ignore_index=True) if folds_rows else pd.DataFrame()
    feature_sets_df = pd.DataFrame(feature_set_rows)
    if results_df.empty:
        raise RuntimeError("No candidate models finished successfully.")

    graph_rows += plot_model_comparison(results_df, PATHS["figures"] / "feature_sets")
    graph_rows += plot_feature_set_elbow(results_df, PATHS["figures"] / "feature_sets")

    # ---- Repeated-CV robustness: best instance of EVERY model family + the ensemble ----
    # Honest CV hierarchy: Validation R2 + OOF R2 (all candidates) -> RepeatedCV (one
    # finalist per family, here) -> Nested CV (single interpretable model, below).
    # We evaluate the best (top robust-score) instance of each model family so the leaderboard
    # compares boosters, bagging models AND the stacking ensemble on the SAME 25-fold protocol.
    robust_rows = []
    if RUN_REPEATED_CV:
        print("\nRepeated-CV robustness: best instance per model family + ensemble (80% dev set, no test leakage)...")
        core_families = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees",
                         "HistGradientBoosting", "GradientBoostingHuber", "StackingRegressor"]
        # Labels to evaluate: best-scoring instance of each family present, plus the forced final model.
        cv_labels = []
        for fam in core_families:
            fam_rows = results_df[results_df["Model"] == fam]
            if not fam_rows.empty:
                cv_labels.append(fam_rows.iloc[0]["Label"])  # results_df is sorted by Robust_Selection_Score
        if FORCE_FINAL_FEATURE_SET and FORCE_FINAL_MODEL:
            forced = f"{FORCE_FINAL_FEATURE_SET} | {FORCE_FINAL_MODEL}"
            if forced in set(results_df["Label"]) and forced not in cv_labels:
                cv_labels.append(forced)
        # de-dup, preserve order
        seen = set()
        cv_labels = [l for l in cv_labels if not (l in seen or seen.add(l))]
        for label in cv_labels:
            obj = trained_objects.get(label)
            if obj is None:
                continue
            meta = results_df[results_df["Label"] == label].iloc[0]
            try:
                X_dev = pd.concat([obj["X_train"], obj["X_val"]], axis=0).reset_index(drop=True)
                y_dev = pd.concat([obj["y_train"], obj["y_val"]], axis=0).reset_index(drop=True)
                rc = repeated_cv_robust(obj["best_estimator"], X_dev, y_dev)
                rc.update({"Label": label, "Model": meta["Model"], "Feature_Set": meta["Feature_Set"],
                           "Validation_R2": meta.get("Validation_R2", np.nan),
                           "TrainOOF_R2": meta.get("TrainOOF_R2", np.nan)})
                robust_rows.append(rc)
                print(f"  {label}: RepeatedCV R2={rc['RepeatedCV_Mean_R2']:.4f} +/- {rc['RepeatedCV_SD_R2']:.4f} "
                      f"(min {rc['RepeatedCV_Min_R2']:.4f}, n={rc['RepeatedCV_N']})")
            except Exception as e:
                print(f"  Repeated CV failed for {label}: {type(e).__name__}: {e}")
    robust_df = pd.DataFrame(robust_rows)

    # ---- Honest CV leaderboard (RepeatedCV is the fair, split-robust ranking) ----
    honest_cv_df = pd.DataFrame()
    if not robust_df.empty:
        cols = ["Model", "Feature_Set", "Label", "RepeatedCV_Mean_R2", "RepeatedCV_SD_R2",
                "RepeatedCV_Min_R2", "Validation_R2", "TrainOOF_R2", "RepeatedCV_N"]
        cols = [c for c in cols if c in robust_df.columns]
        honest_cv_df = robust_df[cols].sort_values("RepeatedCV_Mean_R2", ascending=False).reset_index(drop=True)
        honest_cv_df.insert(0, "Rank", range(1, len(honest_cv_df) + 1))
        print("\nHonest CV leaderboard (ranked by RepeatedCV mean R2, 25 dev folds):")
        for _, r in honest_cv_df.iterrows():
            print(f"  {int(r['Rank'])}. {r['Model']:<22} {r['RepeatedCV_Mean_R2']:.4f} "
                  f"+/- {r['RepeatedCV_SD_R2']:.4f} (min {r['RepeatedCV_Min_R2']:.4f})")

    # ---- Nested CV (unbiased robustness; dev set only, no test leakage) ----
    nested_folds_df, nested_summary_df = pd.DataFrame(), pd.DataFrame()
    if RUN_NESTED_CV and NESTED_CV_FEATURE_SET in FEATURE_SETS:
        print(f"\nNested CV on {NESTED_CV_FEATURE_SET} | {NESTED_CV_MODEL} "
              f"(outer {NESTED_CV_OUTER_SPLITS}x{NESTED_CV_OUTER_REPEATS}, inner {NESTED_CV_INNER_SPLITS}-fold)...")
        try:
            Xn, n_num, n_cat, _, _ = get_X(df, FEATURE_SETS[NESTED_CV_FEATURE_SET])
            Xn_dev = Xn.iloc[dev_idx_all].reset_index(drop=True)
            yn_dev = y.iloc[dev_idx_all].reset_index(drop=True)
            nested_folds_df, nested_summary_df = nested_cv(NESTED_CV_FEATURE_SET, NESTED_CV_MODEL,
                                                          Xn_dev, yn_dev, n_num, n_cat)
            if not nested_summary_df.empty:
                r = nested_summary_df.iloc[0]
                print(f"  Nested CV R2 = {r['NestedCV_Mean_R2']:.4f} +/- {r['NestedCV_SD_R2']:.4f} "
                      f"(min {r['NestedCV_Min_R2']:.4f}, {int(r['NestedCV_Outer_Folds'])} outer folds) -- UNBIASED estimate")
        except Exception as e:
            print(f"  Nested CV failed: {type(e).__name__}: {e}")

    # ---- Select final model ----
    # Prefer the forced single tree model (so SHAP/PDP run); else automatic robust-score top.
    selected = None
    if FORCE_FINAL_FEATURE_SET and FORCE_FINAL_MODEL:
        forced_label = f"{FORCE_FINAL_FEATURE_SET} | {FORCE_FINAL_MODEL}"
        cand = results_df[results_df["Label"] == forced_label]
        if not cand.empty:
            selected = cand.iloc[0]
            print(f"\nFinal model FORCED to interpretable single tree model: {forced_label}")
        else:
            print(f"\nForced label {forced_label!r} not found; falling back to robust-score top.")
    if selected is None:
        selected = results_df.iloc[0]
    selected_label = selected["Label"]
    obj = trained_objects[selected_label]
    print("\n" + "=" * 100)
    print("FINAL SELECTED MODEL (locked test NOT used for selection)")
    print("=" * 100)
    print(selected[["Feature_Set", "Model", "Weighted_High_Rut", "TrainOOF_R2", "Validation_R2",
                    "Train_R2", "Gap_TrainMinusValidation_R2", "Robust_Selection_Score"]].to_string())

    # Series used by RBR-sensitivity / applicability (defined before the branch).
    band_series_full = df["RBR_band"] if "RBR_band" in df.columns else pd.Series(["NA"] * len(df))
    rbr_pct_full = pd.to_numeric(df.get("RBR_JMF_percent", pd.Series([np.nan] * len(df))), errors="coerce")
    rapxac_full = pd.to_numeric(df.get("RAP_pct_x_ACinRAP", pd.Series([np.nan] * len(df))), errors="coerce")
    is_tree = obj["model_name"] in ["XGBoost", "LightGBM", "HistGradientBoosting", "CatBoost", "GradientBoosting"]
    diag = PATHS["figures"] / "diagnostics"

    # =========================================================================
    # BRANCH ON THE LOCKED-TEST SWITCH
    # =========================================================================
    if SCORE_LOCKED_TEST:
        mode_label = "FINAL EVALUATION (locked 20% test revealed once)"
        print("\n*** SCORE_LOCKED_TEST=True: revealing the locked 20% test ONE TIME. ***")
        final_est, X_pdp, y_dev, final_pred_df, final_metrics_df = final_refit_and_test(
            obj["best_estimator"], obj["X_train"], obj["y_train"], obj["X_val"], obj["y_val"],
            obj["X_test"], obj["y_test"])
        final_model_path = PATHS["models"] / "FINAL_SELECTED_MODEL_refit_on_dev80.joblib"
        joblib.dump(final_est, final_model_path)
        print("\nFinal metrics (LOCKED TEST scored once):")
        print(final_metrics_df.to_string(index=False))
        diag_splits = ["Train70", "Validation10", "Dev80_FinalFit", "LockedTest20"]
        X_explain = pd.concat([obj["X_val"], obj["X_test"]], axis=0).reset_index(drop=True)
        explain_pred = pd.concat([final_pred_df[final_pred_df["Dataset"] == "Validation10"],
                                  final_pred_df[final_pred_df["Dataset"] == "LockedTest20"]], ignore_index=True)
        band_explain = pd.concat([band_series_full.iloc[val_idx], band_series_full.iloc[test_idx]],
                                 axis=0).reset_index(drop=True)
        band_split_idx = {"Train70": train_idx, "Validation10": val_idx, "LockedTest20": test_idx}
        ad_targets = [("Validation10", val_idx), ("LockedTest20", test_idx)]
        metrics_sheet = "Final_TrainValTest_Metrics"
        workbook_name = f"Rut20k_v3_{SPLIT_TAG}_WITH_LOCKED_TEST_Results.xlsx"
    else:
        mode_label = "DEVELOPMENT ONLY (locked 20% test untouched)"
        print("\n*** SCORE_LOCKED_TEST=False: DEVELOPMENT-ONLY run. "
              "The 20% test is NOT read, scored, explained, or plotted. ***")
        # Selected model stays fit on the 70% TRAIN set only; validation is genuinely held out.
        final_est = obj["fitted"]
        final_model_path = PATHS["models"] / "SELECTED_MODEL_dev_only_fit_on_train70.joblib"
        joblib.dump(final_est, final_model_path)
        final_pred_df = obj["cand_pred"].copy()
        rows = []
        for ds, sub in final_pred_df.groupby("Dataset"):
            m = metrics(sub["Measured"], sub["Predicted"])
            rel = relative_error_summary(sub["Measured"], sub["Predicted"], "")
            bf = best_fit(sub["Measured"], sub["Predicted"])
            rows.append({"Dataset": ds, "Rows": len(sub), **m,
                         "HighRut_MAE_q80": high_rut_mae(sub["Measured"], sub["Predicted"], 0.80),
                         **rel, "BestFit_Equation": bf["equation"]})
        final_metrics_df = pd.DataFrame(rows)
        print("\nDevelopment metrics (train / train-OOF / validation — NO test):")
        print(final_metrics_df.to_string(index=False))
        X_pdp = pd.concat([obj["X_train"], obj["X_val"]], axis=0).reset_index(drop=True)
        diag_splits = ["Train70", "TrainOOF70", "Validation10"]
        X_explain = obj["X_val"].reset_index(drop=True)
        explain_pred = final_pred_df[final_pred_df["Dataset"] == "Validation10"].reset_index(drop=True)
        band_explain = band_series_full.iloc[val_idx].reset_index(drop=True)
        band_split_idx = {"Train70": train_idx, "Validation10": val_idx}
        ad_targets = [("Validation10", val_idx)]
        metrics_sheet = "Dev_Train_Val_Metrics"
        workbook_name = f"Rut20k_v3_{SPLIT_TAG}_DEV_ONLY_train_val_Results.xlsx"

    # ---- Real-time serving metrics for the selected model (latency + throughput) ----
    # Measured on the validation feature sample only — never the locked test.
    realtime_df = pd.DataFrame()
    try:
        realtime_df = inference_performance(final_est, X_explain, final_model_path)
        if not realtime_df.empty:
            print("\nReal-time serving metrics (latency / throughput, validation sample, CPU):")
            print(realtime_df.to_string(index=False))
    except Exception as e:
        print(f"Inference-performance timing failed: {type(e).__name__}: {e}")

    # ---- Diagnostics for the selected model (test only touched when SCORE_LOCKED_TEST) ----
    for split in diag_splits:
        graph_rows += plot_residuals(final_pred_df, split, diag)
    graph_rows += plot_fold_performance(obj.get("folds"), diag)
    graph_rows += plot_learning_curve(obj["best_estimator"], obj["X_train"], obj["y_train"], PATHS["figures"] / "learning")
    if obj.get("best_params"):
        graph_rows += plot_bias_variance(obj["best_params"], obj["numerical"], obj["categorical"],
                                         obj["X_train"], obj["y_train"], obj["X_val"], obj["y_val"],
                                         PATHS["figures"] / "bias_variance")
    graph_rows += plot_roc_risk(final_pred_df, PATHS["figures"] / "roc")

    perm_df = run_permutation(final_est, obj["X_val"], obj["y_val"], PATHS["figures"] / "feature_sets")
    top_features = list(perm_df["Feature"].head(12)) if not perm_df.empty else list(obj["X_train"].columns[:12])
    graph_rows += run_pdp(final_est, X_pdp, top_features, PATHS["figures"] / "pdp")

    shap_df, shap_graphs, sv, names_idx = (run_shap(final_est, X_explain, explain_pred, PATHS["figures"] / "shap")
                                           if is_tree else (pd.DataFrame(), [], None, None))
    graph_rows += shap_graphs

    # ---- RBR sensitivity + applicability domain ----
    shap_band_df = shap_by_rbr_band(sv, names_idx, band_explain, PATHS["figures"] / "rbr_sensitivity") if sv is not None else pd.DataFrame()

    range_parts, band_parts = [], []
    for ds, idxs in band_split_idx.items():
        sub = final_pred_df[final_pred_df["Dataset"] == ds].reset_index(drop=True).copy()
        if sub.empty:
            continue
        bands = band_series_full.iloc[idxs].reset_index(drop=True)
        sub["Rut_Range"] = sub["Measured"].apply(rut_range_label)
        sub["RBR_band"] = bands.values[:len(sub)]
        range_parts.append(error_by_group(sub, "Rut_Range", ds))
        band_parts.append(error_by_group(sub, "RBR_band", ds))
    error_by_rut_range = pd.concat([d for d in range_parts if not d.empty], ignore_index=True) if range_parts else pd.DataFrame()
    error_by_rbr_band = pd.concat([d for d in band_parts if not d.empty], ignore_index=True) if band_parts else pd.DataFrame()

    # Applicability-domain flag (sparse extreme regions) + inside/outside error comparison.
    rbr_hi = AD_RBR_PERCENT_HIGH if AD_RBR_PERCENT_HIGH is not None else float(np.nanquantile(rbr_pct_full.iloc[dev_idx_all], 0.975))
    rapxac_hi = AD_RAPxAC_HIGH if AD_RAPxAC_HIGH is not None else float(np.nanquantile(rapxac_full.iloc[dev_idx_all], 0.975))
    ad_rows = []
    for ds, idxs in ad_targets:
        sub = final_pred_df[final_pred_df["Dataset"] == ds].reset_index(drop=True).copy()
        if sub.empty:
            continue
        rp = rbr_pct_full.iloc[idxs].reset_index(drop=True).values[:len(sub)]
        rx = rapxac_full.iloc[idxs].reset_index(drop=True).values[:len(sub)]
        outside = (rp > rbr_hi) | (rx > rapxac_hi)
        for flag, name in [(~outside, "Inside_AD"), (outside, "Outside_AD_extreme")]:
            if flag.sum() < 2:
                continue
            m = metrics(sub["Measured"][flag], sub["Predicted"][flag])
            ad_rows.append({"Dataset": ds, "Domain": name, "Rows": int(flag.sum()),
                            "RBR_pct_threshold": rbr_hi, "RAPxAC_threshold": rapxac_hi, **m})
    applicability_df = pd.DataFrame(ad_rows)
    if not applicability_df.empty:
        print("\nApplicability domain (inside vs sparse-extreme outside):")
        print(applicability_df.to_string(index=False))

    # ---- Where is the model weak? (auto summary) ----
    weakness_rows = []
    if not error_by_rut_range.empty:
        for _, r in error_by_rut_range[error_by_rut_range["Dataset"] == "Validation10"].iterrows():
            weakness_rows.append({"Dimension": "Rut range", "Group": r["Rut_Range"], "Rows": r["Rows"],
                                  "R2": r["R2"], "RMSE": r["RMSE"], "MAE": r["MAE"], "Mean_Bias": r["Mean_Bias_PredMinusMeas"]})
    if not error_by_rbr_band.empty:
        for _, r in error_by_rbr_band[error_by_rbr_band["Dataset"] == "Validation10"].iterrows():
            weakness_rows.append({"Dimension": "RBR band", "Group": r["RBR_band"], "Rows": r["Rows"],
                                  "R2": r["R2"], "RMSE": r["RMSE"], "MAE": r["MAE"], "Mean_Bias": r["Mean_Bias_PredMinusMeas"]})
    weakness_df = pd.DataFrame(weakness_rows).sort_values("MAE", ascending=False) if weakness_rows else pd.DataFrame()

    # ---- Decision table vs advisor targets ----
    dec_rows = []
    for _, r in results_df.iterrows():
        dec_rows.append({
            "Label": r["Label"], "Feature_Set": r["Feature_Set"], "Model": r["Model"],
            "Weighted": r["Weighted_High_Rut"], "Val_R2": r["Validation_R2"], "Val_RMSE": r["Validation_RMSE"],
            "Val_MAE": r["Validation_MAE"], "Gap": r["Gap_TrainMinusValidation_R2"],
            "Val_HighRut_MAE": r.get("Validation_HighRut_MAE_q80", np.nan), "Min_Fold_R2": r.get("TrainOOF_Fold_R2_Min", np.nan),
            "Meets_R2>=0.80": r["Validation_R2"] >= TARGET_VAL_R2, "Meets_Gap<=0.16": r["Gap_TrainMinusValidation_R2"] <= TARGET_GAP,
            "Meets_RMSE<1.03": r["Validation_RMSE"] < TARGET_RMSE, "Meets_MAE<0.73": r["Validation_MAE"] < TARGET_MAE,
        })
    decision_df = pd.DataFrame(dec_rows)
    decision_df["Targets_Met_Count"] = decision_df[
        ["Meets_R2>=0.80", "Meets_Gap<=0.16", "Meets_RMSE<1.03", "Meets_MAE<0.73"]].sum(axis=1)

    settings_df = pd.DataFrame([
        {"Setting": "Input file", "Value": str(input_path)},
        {"Setting": "Target", "Value": TARGET},
        {"Setting": "Rows", "Value": len(df)},
        {"Setting": "Split", "Value": f"{int(TRAIN_SIZE*100)}% train / {int(VALIDATION_SIZE*100)}% validation / {int(TEST_SIZE*100)}% LOCKED test"},
        {"Setting": "Split method", "Value": "Two-stage target-bin stratified random; fixed RANDOM_STATE"},
        {"Setting": "Nested CV", "Value": (f"{NESTED_CV_FEATURE_SET}|{NESTED_CV_MODEL}, outer {NESTED_CV_OUTER_SPLITS}x{NESTED_CV_OUTER_REPEATS}, inner {NESTED_CV_INNER_SPLITS}-fold" if RUN_NESTED_CV else "off")},
        {"Setting": "Forced final model", "Value": f"{FORCE_FINAL_FEATURE_SET} | {FORCE_FINAL_MODEL}" if FORCE_FINAL_FEATURE_SET else "auto robust-score"},
        {"Setting": "High-rut weighting", "Value": RUN_HIGH_RUT_WEIGHTING},
        {"Setting": "Replicate averaging (Option B)", "Value": f"{AVERAGE_REPLICATES} (one averaged target row per {ID_COL}; removes replicate leakage)"},
        {"Setting": "Log-target (Option B)", "Value": f"{LOG_TARGET} (train on log1p(target); metrics back on original scale)"},
        {"Setting": "Random state", "Value": RANDOM_STATE},
        {"Setting": "Train/Val/Test rows", "Value": f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}"},
        {"Setting": "Training CV", "Value": f"{CV_FOLDS}-fold target-bin StratifiedKFold inside 70% train"},
        {"Setting": "Repeated CV", "Value": f"{CV_FOLDS}x{REPEATED_CV_REPEATS} on 80% dev for best instance per model family + ensemble"},
        {"Setting": "N_ITER_XGB", "Value": N_ITER_XGB},
        {"Setting": "GPU status", "Value": gpu_status_string()},
        {"Setting": "True RBR", "Value": "RBR_JMF_fraction (=RBR_decimal), RBR_JMF_percent (=RBR_percent)"},
        {"Setting": "Stacking", "Value": f"{RUN_STACKING} ({list(best_params_for_stack.keys())})"},
        {"Setting": "Selection", "Value": "Validation/Robust score; locked test never used for selection"},
        {"Setting": "SCORE_LOCKED_TEST", "Value": SCORE_LOCKED_TEST},
        {"Setting": "Run mode", "Value": mode_label},
        {"Setting": "Final fit", "Value": ("Selected model refit on 80% dev, tested once on 20% locked test"
                                           if SCORE_LOCKED_TEST else
                                           "Selected model fit on 70% train; validated on 10%; 20% test UNTOUCHED")},
    ])
    rec = {
        "Run_Mode": mode_label,
        "Final_Selected_Label": selected_label, "Selected_Feature_Set": selected["Feature_Set"],
        "Selected_Model": selected["Model"], "Selected_Weighted": selected["Weighted_High_Rut"],
        "Validation_R2": selected["Validation_R2"], "TrainOOF_R2": selected["TrainOOF_R2"],
        "Train_R2": selected["Train_R2"], "Gap_TrainMinusValidation": selected["Gap_TrainMinusValidation_R2"],
        "Final_Model_File": str(final_model_path),
    }
    if SCORE_LOCKED_TEST:
        rec["LockedTest_R2"] = float(final_metrics_df.loc[final_metrics_df["Dataset"] == "LockedTest20", "R2"].iloc[0])
        rec["LockedTest_RMSE"] = float(final_metrics_df.loc[final_metrics_df["Dataset"] == "LockedTest20", "RMSE"].iloc[0])
        rec["LockedTest_MAE"] = float(final_metrics_df.loc[final_metrics_df["Dataset"] == "LockedTest20", "MAE"].iloc[0])
    else:
        rec["LockedTest_R2"] = rec["LockedTest_RMSE"] = rec["LockedTest_MAE"] = "NOT SCORED (test untouched)"
    final_recommendation = pd.DataFrame([rec])

    # ---- "How to reach R2 >= 0.80" data-collection recommendation ----
    data_reco_df = pd.DataFrame([
        {"Priority": 1, "New_Predictor": "Continuous PG / binder rheology (DSR G*/sin d)",
         "Why": "Binder stiffness governs rutting; current PG is a rounded 67-76 integer."},
        {"Priority": 2, "New_Predictor": "Binder aging state (RTFO / PAV)",
         "Why": "Aged binder resists rutting; not captured today."},
        {"Priority": 3, "New_Predictor": "LWT test temperature",
         "Why": "Rutting is highly temperature dependent; missing as a feature."},
        {"Priority": 4, "New_Predictor": "Effective binder content / film thickness (Pbe)",
         "Why": "Separates total AC from absorbed AC; drives rutting susceptibility."},
        {"Priority": 5, "New_Predictor": "Traffic level as numeric ESALs",
         "Why": "Load magnitude; only a coarse category exists now."},
        {"Priority": 6, "New_Predictor": "Aggregate source / mineralogy",
         "Why": "Angularity and texture affect shear resistance."},
        {"Priority": "Note", "New_Predictor": "Feature engineering on the current 18 columns",
         "Why": "Elbow curve shows OOF flat at ~0.50 regardless of feature count -> the ceiling is the data, not the features."},
    ])

    graph_index_df = pd.DataFrame(graph_rows)

    # ---- Save workbook (separate file per run mode) ----
    workbook = OUTPUT_FOLDER / workbook_name
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        settings_df.to_excel(writer, sheet_name="Settings", index=False)
        split_summary.to_excel(writer, sheet_name="Split_Summary", index=False)
        feature_sets_df.to_excel(writer, sheet_name="Feature_Sets", index=False)
        results_df.to_excel(writer, sheet_name="Candidate_Results", index=False)
        decision_df.to_excel(writer, sheet_name="Decision_Table", index=False)
        if not honest_cv_df.empty:
            honest_cv_df.to_excel(writer, sheet_name="Honest_CV_Leaderboard", index=False)
        if not robust_df.empty:
            robust_df.to_excel(writer, sheet_name="RepeatedCV_Robustness", index=False)
        if not nested_summary_df.empty:
            nested_summary_df.to_excel(writer, sheet_name="NestedCV_Summary", index=False)
        if not nested_folds_df.empty:
            nested_folds_df.to_excel(writer, sheet_name="NestedCV_OuterFolds", index=False)
        data_reco_df.to_excel(writer, sheet_name="Reach_0.80_DataPlan", index=False)
        final_recommendation.to_excel(writer, sheet_name="Final_Selected_Model", index=False)
        final_metrics_df.to_excel(writer, sheet_name=metrics_sheet, index=False)
        if not realtime_df.empty:
            realtime_df.to_excel(writer, sheet_name="Realtime_Latency_Throughput", index=False)
        final_pred_df.to_excel(writer, sheet_name="Final_Predictions", index=False)
        if not error_by_rut_range.empty:
            error_by_rut_range.to_excel(writer, sheet_name="Error_By_Rut_Range", index=False)
        if not error_by_rbr_band.empty:
            error_by_rbr_band.to_excel(writer, sheet_name="Error_By_RBR_Band", index=False)
        if not applicability_df.empty:
            applicability_df.to_excel(writer, sheet_name="Applicability_Domain", index=False)
        if not weakness_df.empty:
            weakness_df.to_excel(writer, sheet_name="Model_Weakness_Map", index=False)
        if not shap_df.empty:
            shap_df.to_excel(writer, sheet_name="SHAP_Importance", index=False)
        if not shap_band_df.empty:
            shap_band_df.to_excel(writer, sheet_name="SHAP_By_RBR_Band", index=False)
        if not perm_df.empty:
            perm_df.to_excel(writer, sheet_name="Permutation_Importance", index=False)
        desc_df.to_excel(writer, sheet_name="Descriptive_Stats", index=False)
        corr_df.to_excel(writer, sheet_name="Corr_With_Target", index=False)
        if not vif_df.empty:
            vif_df.to_excel(writer, sheet_name="VIF_Multicollinearity", index=False)
        if not folds_df.empty:
            folds_df.to_excel(writer, sheet_name="CV_Folds_Train70", index=False)
        if not predictions_df.empty:
            predictions_df.head(200000).to_excel(writer, sheet_name="All_Candidate_Predictions", index=False)
        if not graph_index_df.empty:
            graph_index_df.to_excel(writer, sheet_name="Graph_Index", index=False)

    results_df.to_csv(PATHS["tables"] / "candidate_results.csv", index=False)
    decision_df.to_csv(PATHS["tables"] / "decision_table.csv", index=False)
    final_metrics_df.to_csv(PATHS["tables"] / "final_train_val_test_metrics.csv", index=False)
    final_pred_df.to_csv(PATHS["tables"] / "final_predictions.csv", index=False)

    locked_test_file = PATHS["splits"] / "LOCKED_test_20pct_DO_NOT_USE_FOR_TUNING.xlsx"
    elapsed = time.time() - t0
    print("\n" + "=" * 100)
    print(f"FINISHED RUT_20K v2 70/10/20 WORKFLOW — {mode_label}")
    print("=" * 100)
    print(f"Workbook: {workbook}")
    print(f"Selected model: {final_model_path}")
    print(f"Figures: {PATHS['figures']}")
    print("\nFinal recommendation:")
    print(final_recommendation.to_string(index=False))
    if not weakness_df.empty:
        print("\nWhere the model is weakest (validation, highest MAE first):")
        print(weakness_df.head(6).to_string(index=False))
    print(f"\nElapsed: {elapsed/60:.2f} minutes")
    best_val = float(results_df["Validation_R2"].max())
    if not SCORE_LOCKED_TEST:
        print("\n" + "-" * 100)
        print("DEVELOPMENT-ONLY RUN COMPLETE — the 20% test set was NOT used in any way.")
        print(f"  It is saved, untouched, in its own Excel file:\n    {locked_test_file}")
        print("  Use the train / train-OOF / validation metrics, the RepeatedCV_Robustness sheet, the")
        print("  CV fold stability, and the learning/bias-variance curves to confirm the model is stable")
        print("  and leakage-free. When you are satisfied, set SCORE_LOCKED_TEST = True and re-run ONCE")
        print("  to reveal the final locked-test score.")
        print("-" * 100)
    if best_val < TARGET_VAL_R2:
        print(f"\nNOTE: Best validation R2 = {best_val:.3f} < target {TARGET_VAL_R2:.2f}. "
              "As the advisor doc anticipated, reaching 0.80 likely requires NEW predictors "
              "(binder rheology, aging, test conditions) rather than further tuning.")


if __name__ == "__main__":
    main()
