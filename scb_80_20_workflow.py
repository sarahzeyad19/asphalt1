# -*- coding: utf-8 -*-
"""
SCB (Jc) MODELING WORKFLOW v3 — 80/20 SPLIT (TARGET INSIDE THE 80% TRAINING DATA)
                               + COMBINED CLEANED + NEW DATA + TRUE RBR + PHYSICS (Pbe/AFT)
                               + STACKING + REPEATED-CV ROBUSTNESS + FULL DIAGNOSTICS
                               + APPLICABILITY DOMAIN
Author: Updated for Sarah Al-Jezawi

SCB variant of the 80/20 rutting workflow. Same machinery (multi-file combining, physics
chain, compact mechanism-based feature selector, group-safe split), but the target is the
SCB critical strain-energy release rate Jc.

WHAT IS SPECIFIC TO SCB HERE (the requested "revised spec")
-----------------------------------------------------------
1. TARGET = SCB Jc (aliases: SCB, SCB_Jc, SCB_Result_Extracted, Jc).
2. COMBINE the cleaned SCB data with the new LaPave data (audited master SCB_Master_666 +
   matched-validation SCB_Full_Mixes), de-duplicated on mix/report identity.
3. USE THE FULL DATA RANGE — the old "Jc < 1" hard restriction is REMOVED
   (RESTRICT_TARGET_RANGE = False), so high-Jc mixes are kept.
4. DROP Jc == 0 — SCB rows with a zero result are invalid/missing measurements, not real
   fracture values, so they are dropped before modelling (DROP_TARGET_LE_ZERO = True).
5. Pass/fail SPEC threshold for risk screening uses the LaDOTD-style minimum Jc
   (SCB_SPEC_MIN_JC): a mix is flagged at-risk when Jc < spec.
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
    StratifiedGroupKFold,
    cross_val_score,
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
TARGET = "SCB"
# Target column aliases: the LaPave master stores the SCB result as "SCB_Result_Extracted",
# the validation file as "SCB_Jc", older cleaned files as "SCB"/"Jc". First present is renamed
# to TARGET in load_data so the same script runs on every file.
TARGET_ALIASES = ["SCB", "SCB_Jc", "SCB_Result_Extracted", "Jc", "SCB_Value"]
ID_COL = "MixDesignKey"
# Prefer a MIX-level identity (so all replicate reports of one mix share a group and can be
# kept together in a split / CV fold) before falling back to a per-report key.
ID_COL_ALIASES = [ID_COL, "Unified_Mix_ID", "Base_Mix_ID", "Mix_ID", "JMF_Record_Key", "JMF_Number"]

# ---- DATA SPLIT (requested: 80% training WITH the target / 20% locked test) ----
# VALIDATION_SIZE = 0 means NO separate validation holdout: the model trains on the full 80%
# (features X + Rut_20k target y together, as required for supervised learning) and every
# "validation" score becomes the honest out-of-fold 5-fold CV score inside that 80%.
TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.00
TEST_SIZE = 0.20
HAS_VAL_HOLDOUT = VALIDATION_SIZE > 0
SPLIT_TAG = (f"{int(TRAIN_SIZE*100)}_{int(TEST_SIZE*100)}" if not HAS_VAL_HOLDOUT
             else f"{int(TRAIN_SIZE*100)}_{int(VALIDATION_SIZE*100)}_{int(TEST_SIZE*100)}")

# Split names used in every table / plot / workbook sheet.
TRAIN_SPLIT_NAME = f"Train{int(TRAIN_SIZE*100)}"                    # e.g. "Train80"
OOF_SPLIT_NAME = f"TrainOOF{int(TRAIN_SIZE*100)}"                   # honest CV view of the training data
VAL_SPLIT_NAME = f"Validation{int(VALIDATION_SIZE*100)}"            # only exists when HAS_VAL_HOLDOUT
DEV_SPLIT_NAME = f"Dev{int((TRAIN_SIZE + VALIDATION_SIZE)*100)}_FinalFit"
TEST_SPLIT_NAME = f"LockedTest{int(TEST_SIZE*100)}"                 # "LockedTest20"

# ---- LOCKED-TEST SWITCH ----
# False: DEVELOPMENT-ONLY run. Train on the 80% (target included), check stability / leakage
#       via OOF CV. The 20% test is split off and saved to its own Excel file but is NEVER
#       read, scored, explained, or plotted. Results go to a separate DEV-ONLY workbook.
# True (current request): reveal the locked 20% test ONCE — refit the selected model on the
#       full 80% training data and score + explain the test a single time (FINAL EVALUATION).
SCORE_LOCKED_TEST = True

CV_FOLDS = 5
N_TARGET_BINS = 5

# Tuning budget. Lower these for a fast check; raise for final runs (bigger search = better
# hyper-parameter calibration, longer runtime). GPU makes the larger XGBoost search affordable.
N_ITER_XGB = 160
N_ITER_OTHER = 70
N_JOBS_SEARCH = 1
N_JOBS_MODEL = -1

# ---- Leakage control: keep every replicate of a mix in the SAME split AND the SAME CV fold ----
# When True, MixDesignKey (ID_COL) is used as a GROUP so the same physical mix can never appear
# in train and test (or in train and validation, or across CV folds). This is the rigorous,
# no-leakage protocol — train/val/test split, the inner training CV, RepeatedCV and Nested CV all
# become group-aware (StratifiedGroupKFold). Honest note: removing this leakage usually LOWERS the
# reported R2 slightly (the previous number was mildly inflated), but it is the correct estimate.
# DEFAULT True now that the audited master file is combined in: it carries multiple replicate
# reports of the same mix (identical design-extracted target), so a row-level split WOULD leak
# and inflate R2. Grouping on the mix ID keeps every replicate together and gives the honest
# score while still using all the rows. Set False only for a single file with no replicates.
GROUP_SPLIT_BY_MIX = True
GROUP_COL = ID_COL
_GROUPS_FULL = None  # set in main() from df[GROUP_COL]; None disables grouping at runtime

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
AVERAGE_REPLICATES = False
LOG_TARGET = False  # raw target. (Log lowered rutting numbers — the target isn't skewed enough
                    # to benefit. Set True only for a strongly right-skewed target like SCB.)

# ---- Final-model selection ----
# AUTO_SELECT_HIGHEST_VALIDATION: pick the final model as the candidate with the HIGHEST
#   validation R2 among SHAP-capable single models (so SHAP/PDP still run on it). This honours
#   "choose the model with the higher validation and test on it". Overrides FORCE_FINAL_* below.
#   The Stacking ensemble is excluded only because it cannot be SHAP-explained.
AUTO_SELECT_HIGHEST_VALIDATION = True
# AUTO_SELECT_METRIC: WHICH score the auto-selection ranks by.
#   "TrainOOF_R2" (default, HONEST) -> out-of-fold CV R2 on the training set: a stable, leakage-free
#       estimate that tracks the locked test far better than a single small validation slice, so it
#       avoids crowning a model that only got lucky on one fold.
#   "Validation_R2" -> old behaviour (highest single-split validation); noisy on small validation
#       sets and can over-reward a lucky slice.
AUTO_SELECT_METRIC = "TrainOOF_R2"
# Models that shap.TreeExplainer can explain (so SHAP/PDP run on the selected model).
# StackingRegressor and HistGradientBoosting are excluded (TreeExplainer cannot handle them).
SHAP_CAPABLE_MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "GradientBoostingHuber"]
# ---- Clean-SHAP option -----------------------------------------------------------------------
# ExtraTrees / RandomForest are BAGGED, deeply-grown randomized trees. They predict well, but
# their SHAP beeswarm looks muddy: high (red) and low (blue) feature values overlap because the
# attributions are non-monotonic and split arbitrarily among correlated features. GRADIENT-BOOSTED
# models (XGBoost / LightGBM / CatBoost) give clean, monotonic, well-separated SHAP colors.
# When True, the final SHAP-explained model is chosen ONLY among gradient boosters, so the SHAP
# plots are interpretable (tiny cost in CV R2 vs the bagging models). Set False to keep the raw
# highest-CV model (which may be ExtraTrees) for the explanation.
SHAP_PREFER_BOOSTER = True
SHAP_BOOSTER_MODELS = ["XGBoost", "LightGBM", "CatBoost", "GradientBoostingHuber"]
# Explain SHAP on the FULL data sample (train + test) instead of only the small locked-test slice,
# so the beeswarm has enough points to be representative.
SHAP_EXPLAIN_ON_ALL_ROWS = True

# ---- FORCE a specific final model (used only when AUTO_SELECT_HIGHEST_VALIDATION = False) ----
# Set both to None to fall back to automatic robust-score selection.
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

# GPU options. USE_GPU=True runs XGBoost (device="cuda") and CatBoost (task_type="GPU") on the
# GPU. Requires an NVIDIA GPU with CUDA drivers. If the GPU isn't usable, GPU_FALLBACK_TO_CPU
# automatically retries that model on CPU (no crash). LightGBM stays on CPU because the standard
# pip wheel is not built with GPU support; set USE_LIGHTGBM_GPU=True only if you installed a
# GPU-enabled LightGBM build yourself.
USE_GPU = True
GPU_DEVICE = "0"
GPU_FALLBACK_TO_CPU = True
USE_LIGHTGBM_GPU = False


def _nvidia_gpu_available() -> bool:
    """True only if an NVIDIA GPU is actually usable (nvidia-smi present and returns 0)."""
    try:
        import shutil as _sh, subprocess as _sp
        if _sh.which("nvidia-smi") is None:
            return False
        return _sp.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


# Auto-disable GPU when no usable GPU is present, so the SAME file runs cleanly on a GPU laptop
# AND a CPU-only machine (no CatBoost CUDA tracebacks). Set USE_GPU=False to force CPU always.
if USE_GPU and not _nvidia_gpu_available():
    print("USE_GPU=True but no usable NVIDIA GPU detected -> running on CPU.")
    USE_GPU = False

# Targets from the advisor decision table.
# Advisor targets, rescaled for SCB Jc (target sd ~0.16, so RMSE/MAE targets are ~10x smaller
# than the rutting ones which were in mm).
TARGET_VAL_R2 = 0.80
TARGET_GAP = 0.16
TARGET_RMSE = 0.12
TARGET_MAE = 0.09

# ---- SCB pass/fail spec (revised) ----------------------------------------------------------
# LaDOTD-style minimum SCB Jc for acceptance. Used for risk screening (ROC): a mix is flagged
# at-risk when its Jc is BELOW this value (low fracture energy = crack-prone). 0.5 kJ/m^2 is the
# common Louisiana minimum for wearing/binder courses; change to your governing spec if needed.
SCB_SPEC_MIN_JC = 0.5

# Applicability domain: flag sparse extreme regions.
AD_RBR_PERCENT_HIGH = None       # auto = 97.5th percentile if None
AD_RAPxAC_HIGH = None            # auto = 97.5th percentile if None

# =============================================================================
# RESTRICTED-RANGE REPORTING + CALIBRATION + IMPORTANCE-BASED FEATURE SELECTION
# (requested: keep the range where the model predicts well; adjust the final numbers with a
#  calibration equation; select the highest-impact features; keep the STRATIFIED split.)
# =============================================================================
# REVISED SCB SPEC: use the FULL Jc data range. The old workflow dropped Jc >= 1
# (RESTRICT_TARGET_RANGE + TARGET_RANGE_MAX=1.0); that restriction is now OFF so every mix,
# including high-Jc ones, is kept.
RESTRICT_TARGET_RANGE = False        # SCB: keep the full range (do NOT drop Jc >= 1 any more)
TARGET_RANGE_MIN = None              # inclusive lower bound (None = -inf)
TARGET_RANGE_MAX = None              # upper bound (None = +inf) -> full range
TARGET_RANGE_MAX_INCLUSIVE = True

# ---- Drop invalid zero-Jc rows (revised SCB spec) ----
# SCB rows with Jc == 0 are missing/invalid fracture results, not real measurements. Drop them
# (any row with target <= DROP_TARGET_LE_VALUE) before modelling so they cannot distort training.
DROP_TARGET_LE_ZERO = True
DROP_TARGET_LE_VALUE = 0.0           # drop rows with Jc <= this (0.0 -> drop exact zeros / negatives)

# REPORT (do NOT drop) a reliable band beside the full range. None = full range only (SCB request).
REPORT_INRANGE_BAND = None           # (min, max) or None; kept None so only the FULL range is reported

# Compression-calibration equation. Fit  Measured = a * Predicted + b  on the DEV predictions and
# apply it to every split so Predicted approx Measured (fixes the slope<1 compression -> lower
# RMSE / MAE / bias, best-fit slope -> 1). HONEST NOTE: the reported R2 here is the coefficient of
# determination (r2_score = 1 - SSres/SStot), which is NOT invariant to rescaling the prediction --
# so correcting the compression typically RAISES the test R2 as well. This is legitimate: a and b
# are learned on the DEV set only and applied once to the locked test (it is part of the model, not
# tuning on the test). Reported as extra "*_Calibrated" rows so raw and calibrated are both visible.
APPLY_CALIBRATION = True

# Importance-based feature selection: rank the pooled candidate features by RandomForest importance
# on the DEV rows only (no locked-test leakage) and register a reduced "TopImpact_Selected" set of
# the K most impactful features, run through the normal pipeline alongside the other feature sets.
# CORRECTION: the earlier pool name "Engineering_Core_WithOptionalPhysics" was never defined in
# FEATURE_SETS, so the selector silently fell back to another set. Both selectors now point at the
# explicitly defined "RutMechanism_CompactPool" (see FEATURE_SETS below).
ADD_TOP_IMPACT_FEATURE_SET = True
TOP_IMPACT_K = 10
TOP_IMPACT_POOL = "RutMechanism_CompactPool"

# ---- Compact mechanism-based feature selector (stability + redundancy + size sweep) ----
# Stage 1-2: per-CV-fold permutation importance PI on the TRAINING rows only; each feature gets
#            Stability = max(0, mean(PI)) * PositiveFoldFraction / (1 + SD(PI))
#            so a feature cannot rank highly because of one lucky fold.
# Stage 3:   redundancy pruning — |Spearman rho| >= COMPACT_CORRELATION_THRESHOLD (training data
#            only) drops the LOWER-ranked member of the pair (e.g. keep RBR_JMF_fraction OR
#            RAP_pct_x_ACinRAP, not both).
# Stage 4:   size sweep — for every k in COMPACT_CANDIDATE_COUNTS compute training OOF
#            (grouped/stratified CV) R2, RMSE, MAE of the top-k surviving features.
# Stage 5:   pick the SMALLEST subset with R2_k >= R2_best - COMPACT_R2_TOLERANCE and
#            RMSE_k <= COMPACT_RMSE_TOLERANCE * RMSE_reference (so 9 inputs beat 15 when the
#            extra 6 add only a negligible improvement).
# The winning subset is registered as feature set "Compact_Selected" and competes in the normal
# pipeline. All selector tables are written to the results workbook.
RUN_COMPACT_SELECTOR = True
COMPACT_POOL = "RutMechanism_CompactPool"
COMPACT_CORRELATION_THRESHOLD = 0.92
COMPACT_CANDIDATE_COUNTS = [5, 7, 9, 12, 15]
COMPACT_R2_TOLERANCE = 0.01
COMPACT_RMSE_TOLERANCE = 1.02
COMPACT_PI_REPEATS = 5           # permutation repeats per fold for the stability score

# Graph / analysis switches.
SHOW_PLOTS_IN_SPYDER = True
FIGURE_DPI = 200
RUN_PERMUTATION_IMPORTANCE = True
RUN_SHAP = True
RUN_PDP = True
RUN_LEARNING_CURVE = True
RUN_BIAS_VARIANCE_CURVES = True
RUN_ROC_RISK_SCREENING = True
RUT_RISK_THRESHOLD_MM = SCB_SPEC_MIN_JC   # reused by the ROC screening (SCB: at-risk if Jc < spec)
MAX_SHAP_ROWS = 700
MAX_PDP_FEATURES = 8
MAX_PDP_PAIRS = 4

# Fast smoke test (verify the script runs end-to-end). Set False for the real run.
QUICK_SMOKE_TEST = False

# ---- Input file resolution ----
HOME = Path.home()
# Input data + outputs live here. Falls back to the current user's Downloads folder (or the
# working directory) automatically when this path does not exist on the machine.
DOWNLOADS = Path(r"C:\Users\H0012066\Downloads")
if not DOWNLOADS.exists():
    DOWNLOADS = HOME / "Downloads" if (HOME / "Downloads").exists() else Path.cwd()
# Preferred: the combined cleaned workbook with the physics columns (AFT_micron, Pbe_pct, Gse,
# Dust_Pbe_ratio, SurfaceArea_m2kg, Grad_* gradation, additives), sheet "RUT". The older
# Rutting_Cleaned_with_RBR.xlsx stays as a fallback — missing physics columns are recomputed
# from the equations sheet formulas where possible.
# Primary = the cleaned SCB workbook (with the physics columns Pbe/AFT/Gse, gradation, additives).
# Falls back to the LaPave workbooks if the cleaned SCB file is not present on your machine.
RUT_FILENAME = "SCB_Cleaned_with_RBR.xlsx"
RUT_FILE = DOWNLOADS / RUT_FILENAME
RUT_FILE_FALLBACKS = [
    DOWNLOADS / "LWT__SCB_CLEANED.xlsx",
    DOWNLOADS / "LWT__SCB_CLEANED (1).xlsx",
    DOWNLOADS / "LaPave_Audited_Master_RUT_SCB.xlsx",
    DOWNLOADS / "LaPave_Validation_Full_Matched_RUT_SCB.xlsx",
    DOWNLOADS / "SCB_Cleaned_SpecBased.xlsx",
]
# SCB sheet names first (cleaned file), then the LaPave master/validation SCB sheets.
SHEET_CANDIDATES = ["SCB", "Cleaned_With_RBR", "Cleaned_Dataset", "Cleaned_Data_Kept",
                    "SCB_Master_666", "SCB_Full_Mixes", "LWT_Clean_Modeling", "Sheet1", 0]

# ---- Combine several data files (union of rows) --------------------------------------------
# When COMBINE_ADDITIONAL_FILES is True the primary file above is loaded FIRST and then every
# source below is appended: each file's columns are harmonized to the canonical names, the rows
# are concatenated, and duplicate mixes/reports are removed (on JMF_Record_Key, else Mix_ID +
# target) so a mix present in two files is not double-counted. Missing files are skipped with a
# warning, so the script still runs if you only have some of them.
# For SCB we combine the CLEANED SCB data with the NEW LaPave data (audited master SCB_Master_666
# + matched-validation SCB_Full_Mixes).
COMBINE_ADDITIONAL_FILES = True
ADDITIONAL_DATA_SOURCES = [
    # 666-mix audited master (SCB sheet).
    (["LaPave_Audited_Master_RUT_SCB.xlsx", "LaPave_Audited_Master_RUT_SCB (1).xlsx"],
     ["SCB_Master_666", "SCB_Master", "SCB_Full_Mixes"]),
    # 177-mix matched validation (SCB sheet).
    (["LaPave_Validation_Full_Matched_RUT_SCB.xlsx"],
     ["SCB_Full_Mixes", "SCB"]),
    # Cleaned SCB file (kept as an additional source in case the primary resolved elsewhere).
    (["SCB_Cleaned_with_RBR.xlsx", "LWT__SCB_CLEANED.xlsx"],
     ["SCB", "Cleaned_With_RBR", "Cleaned_Dataset"]),
]
# De-duplication key priority when merging files (first present wins).
DEDUP_KEYS = ["JMF_Record_Key", "Mix_ID", "Unified_Mix_ID"]

OUTPUT_FOLDER = DOWNLOADS / f"SCB_v3_{SPLIT_TAG}_RBR_outputs"

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
        candidates += list(script_dir.glob("*LaPave_Validation*Matched*RUT*SCB*.xlsx"))
        candidates += list(script_dir.glob("*LWT__SCB_CLEANED*.xlsx"))
        candidates += list(script_dir.glob("*Rutting_Cleaned_with_RBR*.xlsx"))
    except Exception:
        pass
    candidates += [Path.cwd() / p.name for p in list(candidates)]
    candidates += list(Path.cwd().glob("*LaPave_Validation*Matched*RUT*SCB*.xlsx"))
    candidates += list(Path.cwd().glob("*LWT__SCB_CLEANED*.xlsx"))
    candidates += list(Path.cwd().glob("*Rutting_Cleaned_with_RBR*.xlsx"))
    seen, out = set(), []
    for p in candidates:
        if str(p) in seen:
            continue
        seen.add(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find the data Excel file. Put 'LaPave_Validation_Full_Matched_RUT_SCB.xlsx' "
        "(preferred), 'LWT__SCB_CLEANED.xlsx', or 'Rutting_Cleaned_with_RBR.xlsx' in your "
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
    # Three source layouts are supported and harmonized to these canonical names:
    #   * LaPave VALIDATION file: "Validation_Average__*", "Combined_Aggregate_*",
    #     "Validation_Average__Passing_*".
    #   * LaPave AUDITED MASTER file: "Design_Submission__*" volumetrics/gradation, "*_Recalc"
    #     / "*_Final" physics, plus bare PG_HighTemp / NMAS_mm.
    #   * older LWT/Rutting cleaned files: "Grad_*", "RBR_percent", etc.
    "AsphaltContent_Design": ["AC_design", "AC_Design", "AsphaltContentDesign", "Design_AC",
                              "Validation_Average__Asphalt_Content", "Design_Submission__Percent_AC"],
    "Pass4_75mm": ["P4.75", "P4_75", "Pass_4_75mm", "Passing_4.75mm", "Grad_No4",
                   "Validation_Average__Passing_No4", "Design_Submission__Pass_No_4"],
    "Pass0_075mm": ["P0.075", "P0_075", "Pass_0_075mm", "Passing_0.075mm", "Grad_No200",
                    "Validation_Average__Passing_No200", "Design_Submission__Pass_No_200"],
    "NMAS (mm)": ["NMAS", "NMAS_mm", "NMAS(mm)"],  # else parsed from Nominal_Aggregate_Size (text)
    "PG_HighTemp": ["PG High", "PG_High", "PGHigh", "PG_High_Temp", "PG_HighTemp"],  # else parsed from PG grade
    "PG_LowTemp": ["PG_LowTemp", "PG Low", "PG_Low"],
    "RAP_pct": ["RAP", "RAP%", "RAP_Percent", "RAP_Pct"],
    "ACinRAP": ["AC_in_RAP", "AC_RAP", "ACin_RAP", "ACinRAP_Recalc"],
    "Dust_Binder": ["DustBinder", "Dust_to_Binder", "Dust/Binder"],
    "SandEq": ["Sand_Equivalent", "SandEQ", "SE", "Combined_Aggregate_Sand_Equivalent"],
    "FAA": ["Combined_Aggregate_FAA"],
    "CAA": ["Combined_Aggregate_CAA"],
    "Absorption": ["Combined_Aggregate_Absorption"],
    "VMA": ["Validation_Average__VMA", "Design_Submission__VMA"],
    "VFA": ["Validation_Average__VFA", "Design_Submission__VFA"],
    "Va": ["Air_Voids", "AirVoids", "Validation_Average__Air_Voids", "Design_Submission__Percent_Voids"],
    "Gmm": ["Validation_Average__Gmm", "Design_Submission__Gmm"],
    "Gmb": ["Validation_Average__Gmb", "Design_Submission__Gmb_Nd"],
    "Gsb": ["Combined_Aggregate_Bulk_Gravity"],
    "Gse": ["Validation_Average__Gse", "Design_Submission__Gse"],
    "Pba_pct": ["Validation_Average__Pba", "Design_Submission__Pba"],
    "Pbe_pct": ["Validation_Average__Pbe", "Pbe_Final", "Pbe_Reported", "Design_Submission__Pbe"],
    "Dust_Pbe_ratio": ["Validation_Average__Dust_to_Effective_Binder", "Dust_to_Effective_Binder",
                       "Design_Submission__Dust_Pbeff"],
    "SurfaceArea_m2kg": ["SurfaceArea_m2kg_Recalc"],
    "AFT_micron": ["AFT_micron_Recalc"],
    "DesignLev": ["DesignLevel", "Design_Level", "TrafficLevel"],
    "MixType": ["Mix_Type", "Mixture_Type", "Type"],
    "RAP_Class": ["RAPClass", "RAP class", "RAP_Classification", "RBR_Band", "RBR_band",
                  "RBR_Class", "RAP_Spec_Class"],
    "Has_Additive": ["Has_Antistrip"],
    "Additive_Type": ["Antistrip_Type", "Additive_Type_clean", "Antistrip_Family_Standardized",
                      "Additive_Family"],
    # RBR: master ships RBR_percent_Recalc; older files RBR_percent / RBR_decimal.
    "RBR_JMF_fraction": ["RBR_decimal", "RBR_fraction", "RBR"],
    "RBR_JMF_percent": ["RBR_percent", "RBR_pct", "RBR_percent_Recalc"],
    # Full gradation -> canonical Grad_* names used by the SA / gradation-area calcs.
    "Grad_1_1_2in": ["Validation_Average__Passing_1_1_2in", "Design_Submission__Pass_1_1_2in"],
    "Grad_1in": ["Validation_Average__Passing_1in", "Design_Submission__Pass_1in"],
    "Grad_3_4in": ["Validation_Average__Passing_3_4in", "Design_Submission__Pass_3_4in"],
    "Grad_1_2in": ["Validation_Average__Passing_1_2in", "Design_Submission__Pass_1_2in"],
    "Grad_3_8in": ["Validation_Average__Passing_3_8in", "Design_Submission__Pass_3_8in"],
    "Grad_No4": ["Validation_Average__Passing_No4", "Design_Submission__Pass_No_4"],
    "Grad_No8": ["Validation_Average__Passing_No8", "Design_Submission__Pass_No_8"],
    "Grad_No16": ["Validation_Average__Passing_No16", "Design_Submission__Pass_No_16"],
    "Grad_No30": ["Validation_Average__Passing_No30", "Design_Submission__Pass_No_30"],
    "Grad_No50": ["Validation_Average__Passing_No50", "Design_Submission__Pass_No_50"],
    "Grad_No100": ["Validation_Average__Passing_No100", "Design_Submission__Pass_No_100"],
    "Grad_No200": ["Validation_Average__Passing_No200", "Design_Submission__Pass_No_200"],
}

NUMERIC_HINTS = [
    "ACinRAP", "PG_HighTemp", "SandEq", "ADT_DOTD_ord", "Dust_Binder", "VFA",
    "RAP_pct", "Pass4_75mm", "Pass0_075mm", "FAA", "CAA", "Absorption", "VMA", "Va",
    "Gmb", "Gmm", "Gsb", "AsphaltContent_Design", "RAP_pct_x_ACinRAP", "NMAS (mm)",
    "RBR_JMF_fraction", "RBR_JMF_percent", "RAP_Binder_Contribution",
    "PG_x_RBR", "PG_x_RAPAC", "Abs_x_RBR", "SandEq_x_DustBinder",
    "Va_x_Gmm", "VFA_x_AC", "P0075_x_DustBinder",
    # Physics columns shipped in (or recomputed for) the LWT__SCB_CLEANED workbook.
    "AFT_micron", "Pbe_pct", "Pba_pct", "Dust_Pbe_ratio", "Gse", "SurfaceArea_m2kg",
    "SurfaceArea_ft2lb", "Additive_Rate_clean", "MixTemperature_F_clean",
    "Grad_3_4in", "Grad_1_2in", "Grad_3_8in", "Grad_No4", "Grad_No8", "Grad_No16",
    "Grad_No30", "Grad_No50", "Grad_No100", "Grad_No200",
    # Engineered mechanism parameters.
    "AggregateSkeletonIndex", "CompactionInstabilityIndex", "MasticStabilityIndex",
    "RecycleFilmSeverity", "GradationArea_LogSieve", "EffectiveBinderAvailability",
    "VirginBinderProxy", "PG_RecycleFilmSeverity", "BinderLubricationDemand",
    "SkeletonLubricationBalance",
]
CATEGORICAL_HINTS = ["MixType", "DesignLev", "RAP_Class", "Additive_Type_clean", "Additive_Type", "Has_Additive"]

DROP_ALWAYS = [
    ID_COL, "Rut_20k", "SCB", "LWT_Record_ID", "SCB_Record_ID", "LWT_LastUpdated",
    "SCB_LastUpdated", "Rut_Replicate_Number", "SCB_Replicate_Number", "Predictor_Source",
    "Gmm_record_date", "Gmb_record_date", "Gmb_specimen_AC", "Review_Flags",
    "Aggregate_components_used", "IsLeft", "ADT", "RBR_band",
    "Flag_RBR_out_of_range", "Flag_missing_input",
    "MixTemperature_flag_invalid", "MixTemperature_F", "Additive_Product", "Additive_Rate",
    "Source_File", "Unified_Mix_ID", "Base_Mix_ID", "Mix_ID", "JMF_Record_Key",
    "LWT_Design_Extracted", "SCB_Result_Extracted", "LWT_Validation_Rut", "SCB_Jc",
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


def parse_pg_high_temp(value: Any) -> float:
    """High-temperature PG number from a grade string, e.g. 'PG 70-22' -> 70, 'PG 76-22M' -> 76.
    Returns NaN for 'PG grade not normalized' and other unparseable values."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    m = re.search(r"(?i)PG\s*(\d{2,3})\s*[-–]\s*(-?\d{1,3})", str(value))
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d{2,3})\s*[-–]\s*-?\d{1,3}\b", str(value))
    return float(m.group(1)) if m else np.nan


_NMAS_TEXT_TO_MM = {"1 1/2": 37.5, "1-1/2": 37.5, "1.5": 37.5, "1 in": 25.0, "1in": 25.0,
                    "3/4": 19.0, "1/2": 12.5, "3/8": 9.5, "1": 25.0}


def parse_nmas_to_mm(value: Any) -> float:
    """NMAS in mm from text like '3/4 in.' -> 19.0, '1 in.' -> 25.0, '1/2 in.' -> 12.5.
    A bare number is treated as already-mm when plausible (<=50)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if v <= 50 else np.nan
    s = str(value).lower().replace("in.", "").replace("inch", "").replace("in", "").strip()
    if s in _NMAS_TEXT_TO_MM:
        return _NMAS_TEXT_TO_MM[s]
    m = re.match(r"^\s*(\d+)\s+(\d+)/(\d+)", s)      # mixed fraction e.g. "1 1/2"
    if m:
        inches = float(m.group(1)) + float(m.group(2)) / float(m.group(3))
        return round(inches * 25.4, 1)
    m = re.match(r"^\s*(\d+)/(\d+)", s)              # simple fraction e.g. "3/4"
    if m:
        return round(float(m.group(1)) / float(m.group(2)) * 25.4, 1)
    m = re.match(r"^\s*(\d+(?:\.\d+)?)", s)          # decimal inches or mm
    if m:
        v = float(m.group(1))
        return round(v * 25.4, 1) if v <= 3 else v   # <=3 -> inches, else already mm
    return np.nan


def normalize_yes_no(value: Any) -> Any:
    """'Yes'/'No' -> 1/0; pass numeric through; anything else -> as-is (for categorical use)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return 1
    if s in ("no", "n", "false", "0", "no antistrip listed in jmf"):
        return 0
    return value


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

    # ---- LaPave text columns -> numeric canonical inputs ----
    if "PG_HighTemp" not in df.columns:
        for src in ["PG_Grade_Normalized", "PG_Grade"]:
            if src in df.columns:
                df["PG_HighTemp"] = df[src].apply(parse_pg_high_temp)
                break
    if "NMAS (mm)" not in df.columns and "Nominal_Aggregate_Size" in df.columns:
        df["NMAS (mm)"] = df["Nominal_Aggregate_Size"].apply(parse_nmas_to_mm)
    if "Has_Additive" in df.columns:
        df["Has_Additive"] = df["Has_Additive"].apply(normalize_yes_no)
    if "MixTemperature_F_clean" not in df.columns and "Mix_Temperature" in df.columns:
        t = pd.to_numeric(df["Mix_Temperature"], errors="coerce")
        df["MixTemperature_F_clean"] = t.where((t >= 250) & (t <= 375))  # implausible -> blank

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

    # =========================================================================
    # PHYSICS CHAIN — equations from the LWT__SCB_CLEANED "Added variables" sheet.
    # Each quantity is computed ONLY when the loaded file does not already ship it,
    # so the same script runs on both cleaned data files.
    # =========================================================================
    GB_BINDER = 1.03     # binder specific gravity Gb used by the workbook equations
    EPS = 1e-6
    Pb = (pd.to_numeric(df["AsphaltContent_Design"], errors="coerce")
          if "AsphaltContent_Design" in df.columns else None)

    # Gse = (100 - Pb) / (100/Gmm - Pb/Gb)           effective aggregate specific gravity
    if "Gse" not in df.columns and Pb is not None and "Gmm" in df.columns:
        Gmm_ = pd.to_numeric(df["Gmm"], errors="coerce")
        df["Gse"] = (100.0 - Pb) / (100.0 / Gmm_ - Pb / GB_BINDER)
    # Pba = 100 * (Gse - Gsb) / (Gse * Gsb) * Gb     absorbed binder (%)
    if "Pba_pct" not in df.columns and {"Gse", "Gsb"}.issubset(df.columns):
        Gse_ = pd.to_numeric(df["Gse"], errors="coerce")
        Gsb_ = pd.to_numeric(df["Gsb"], errors="coerce")
        df["Pba_pct"] = 100.0 * (Gse_ - Gsb_) / (Gse_ * Gsb_) * GB_BINDER
    # Pbe = Pb - (Pba/100) * (100 - Pb)              effective binder (%)
    if "Pbe_pct" not in df.columns and Pb is not None and "Pba_pct" in df.columns:
        df["Pbe_pct"] = Pb - (pd.to_numeric(df["Pba_pct"], errors="coerce") / 100.0) * (100.0 - Pb)
    # Dust/Pbe = P0.075 / Pbe
    if "Dust_Pbe_ratio" not in df.columns and {"Pass0_075mm", "Pbe_pct"}.issubset(df.columns):
        df["Dust_Pbe_ratio"] = safe_divide(df["Pass0_075mm"], df["Pbe_pct"])
    # Hveem surface area SA(ft2/lb) = 2 + 2P4 + 4P8 + 8P16 + 14P30 + 30P50 + 60P100 + 160P200
    # (P as decimals); SA_m2kg = SA_ft2lb * 0.20482.
    SA_FACTORS = {"Grad_No4": 2.0, "Grad_No8": 4.0, "Grad_No16": 8.0, "Grad_No30": 14.0,
                  "Grad_No50": 30.0, "Grad_No100": 60.0, "Grad_No200": 160.0}
    if "SurfaceArea_ft2lb" not in df.columns:
        if "SurfaceArea_m2kg" in df.columns:
            df["SurfaceArea_ft2lb"] = pd.to_numeric(df["SurfaceArea_m2kg"], errors="coerce") / 0.20482
        elif set(SA_FACTORS).issubset(df.columns):
            sa = 2.0
            for c, f in SA_FACTORS.items():
                sa = sa + f * (pd.to_numeric(df[c], errors="coerce") / 100.0)
            df["SurfaceArea_ft2lb"] = sa
            df["SurfaceArea_m2kg"] = sa * 0.20482
    # AFT (um) = Pbe * 4870 / (100 * Ps * SA)   with Ps = (100 - Pb)/100 and SA in ft2/lb
    if ("AFT_micron" not in df.columns and Pb is not None
            and {"Pbe_pct", "SurfaceArea_ft2lb"}.issubset(df.columns)):
        Ps = (100.0 - Pb) / 100.0
        df["AFT_micron"] = pd.to_numeric(df["Pbe_pct"], errors="coerce") * 4870.0 / (
            100.0 * Ps * pd.to_numeric(df["SurfaceArea_ft2lb"], errors="coerce"))

    # =========================================================================
    # NEW RUTTING PARAMETERS (mechanism indices; each built only when its source
    # columns exist)
    # =========================================================================
    # 1. Effective binder availability  EBA = Pbe / (1 + Absorption)
    if has("Pbe_pct", "Absorption"):
        df["EffectiveBinderAvailability"] = pd.to_numeric(df["Pbe_pct"], errors="coerce") / (
            1.0 + pd.to_numeric(df["Absorption"], errors="coerce"))
    # 2. Virgin-binder proxy  VB = AC_design - RAP_pct * ACinRAP / 100
    if has("AsphaltContent_Design", "RAP_pct", "ACinRAP"):
        df["VirginBinderProxy"] = Pb - (pd.to_numeric(df["RAP_pct"], errors="coerce")
                                        * pd.to_numeric(df["ACinRAP"], errors="coerce") / 100.0)
    # 3. Recycled-binder film severity  RFS = RBR_percent / AFT
    if has("RBR_JMF_percent", "AFT_micron"):
        df["RecycleFilmSeverity"] = safe_divide(df["RBR_JMF_percent"], df["AFT_micron"])
    # 4. PG x recycled-film severity  PG_RFS = PG_high * RFS
    if has("PG_HighTemp", "RecycleFilmSeverity"):
        df["PG_RecycleFilmSeverity"] = (pd.to_numeric(df["PG_HighTemp"], errors="coerce")
                                        * pd.to_numeric(df["RecycleFilmSeverity"], errors="coerce"))
    # 5. Mastic stability index  MSI = Pbe*AFT / ((1 + Dust/Pbe)(1 + Absorption))
    #    (falls back to SandEq/(1 + Dust/Binder) when Pbe/AFT are unavailable)
    if has("Pbe_pct", "AFT_micron", "Dust_Pbe_ratio", "Absorption"):
        df["MasticStabilityIndex"] = (pd.to_numeric(df["Pbe_pct"], errors="coerce")
                                      * pd.to_numeric(df["AFT_micron"], errors="coerce")) / (
            (1.0 + pd.to_numeric(df["Dust_Pbe_ratio"], errors="coerce"))
            * (1.0 + pd.to_numeric(df["Absorption"], errors="coerce")))
    elif has("SandEq", "Dust_Binder"):
        df["MasticStabilityIndex"] = pd.to_numeric(df["SandEq"], errors="coerce") / (
            1.0 + pd.to_numeric(df["Dust_Binder"], errors="coerce"))
    # 6. Aggregate-skeleton index
    #    ASI = ((100 - P4.75)/100) * (NMAS/12.5) * (1 + CAA/100) * (1 + FAA/100)
    if has("Pass4_75mm", "NMAS (mm)", "CAA", "FAA"):
        df["AggregateSkeletonIndex"] = (
            (100.0 - pd.to_numeric(df["Pass4_75mm"], errors="coerce")) / 100.0
            * pd.to_numeric(df["NMAS (mm)"], errors="coerce") / 12.5
            * (1.0 + pd.to_numeric(df["CAA"], errors="coerce") / 100.0)
            * (1.0 + pd.to_numeric(df["FAA"], errors="coerce") / 100.0))
    elif has("CAA", "FAA"):
        df["AggregateSkeletonIndex"] = (pd.to_numeric(df["CAA"], errors="coerce")
                                        * pd.to_numeric(df["FAA"], errors="coerce") / 100.0)
    # 7. Binder-lubrication demand  BLD = Pbe / (P0.075 + Absorption + eps)
    if has("Pbe_pct", "Pass0_075mm", "Absorption"):
        df["BinderLubricationDemand"] = pd.to_numeric(df["Pbe_pct"], errors="coerce") / (
            pd.to_numeric(df["Pass0_075mm"], errors="coerce")
            + pd.to_numeric(df["Absorption"], errors="coerce") + EPS)
    # 8. Skeleton-lubrication balance  SLB = ln((ASI + eps) / (BLD + eps))
    if has("AggregateSkeletonIndex", "BinderLubricationDemand"):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (pd.to_numeric(df["AggregateSkeletonIndex"], errors="coerce") + EPS) / (
                pd.to_numeric(df["BinderLubricationDemand"], errors="coerce") + EPS)
            df["SkeletonLubricationBalance"] = np.log(ratio.where(ratio > 0))
    # 9. Compaction-instability index  CII = (Va / VMA) * (1 + RBR)
    if has("Va", "VMA"):
        cii = safe_divide(df["Va"], df["VMA"])
        if "RBR_JMF_fraction" in df.columns:
            cii = cii * (1.0 + pd.to_numeric(df["RBR_JMF_fraction"], errors="coerce"))
        df["CompactionInstabilityIndex"] = cii

    # Gradation area under the %-passing curve on a log-sieve axis (named Grad_* sieves).
    try:
        _SIEVE_MM = {"Grad_1_1_2in": 37.5, "Grad_1in": 25.0, "Grad_3_4in": 19.0,
                     "Grad_1_2in": 12.5, "Grad_3_8in": 9.5, "Grad_No4": 4.75,
                     "Grad_No8": 2.36, "Grad_No16": 1.18, "Grad_No30": 0.60,
                     "Grad_No50": 0.30, "Grad_No100": 0.15, "Grad_No200": 0.075}
        grad_pairs = sorted((mm, c) for c, mm in _SIEVE_MM.items() if c in df.columns)
        if len(grad_pairs) >= 4:
            sizes = np.log10([s for s, _ in grad_pairs])
            passing = df[[c for _, c in grad_pairs]].apply(pd.to_numeric, errors="coerce").values
            trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            df["GradationArea_LogSieve"] = trap(passing, x=sizes, axis=1) / (sizes[-1] - sizes[0])
    except Exception:
        pass

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

# Explicitly defined pool for the compact mechanism-based selector (and the TopImpact selector):
# mechanism-relevant raw inputs + physics columns (Pbe/AFT chain) + physics interactions + the
# engineered mechanism indices. Features missing from the loaded file are dropped by get_X.
RUT_MECHANISM_COMPACT_POOL = [
    "PG_HighTemp", "RBR_JMF_fraction", "RAP_pct_x_ACinRAP", "ACinRAP",
    "AsphaltContent_Design", "VFA", "VMA", "Va", "Gmm", "Absorption",
    "SandEq", "Dust_Binder", "FAA", "CAA", "Pass4_75mm", "Pass0_075mm", "NMAS (mm)",
    "Pbe_pct", "AFT_micron", "Dust_Pbe_ratio", "SurfaceArea_m2kg", "Gse",
    "EffectiveBinderAvailability", "VirginBinderProxy", "RecycleFilmSeverity",
    "PG_RecycleFilmSeverity", "MasticStabilityIndex", "AggregateSkeletonIndex",
    "BinderLubricationDemand", "SkeletonLubricationBalance", "CompactionInstabilityIndex",
    "GradationArea_LogSieve",
    "PG_x_RBR", "Abs_x_RBR", "SandEq_x_DustBinder", "Va_x_Gmm", "VFA_x_AC", "P0075_x_DustBinder",
]

# The nine new rutting parameters + the core physics columns as a stand-alone candidate set,
# so their contribution is measured directly against the older feature sets.
RUT_PHYSICS_NEW_PARAMS = [
    "PG_HighTemp", "RBR_JMF_fraction", "Va", "VFA", "SandEq",
    "Pbe_pct", "AFT_micron", "Dust_Pbe_ratio",
    "EffectiveBinderAvailability", "VirginBinderProxy", "RecycleFilmSeverity",
    "PG_RecycleFilmSeverity", "MasticStabilityIndex", "AggregateSkeletonIndex",
    "BinderLubricationDemand", "SkeletonLubricationBalance", "CompactionInstabilityIndex",
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
    "RutMechanism_CompactPool": RUT_MECHANISM_COMPACT_POOL,
    "RutPhysics_NewParams": RUT_PHYSICS_NEW_PARAMS,
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
    "RutPhysics_NewParams",
]
if QUICK_SMOKE_TEST:
    FEATURE_SETS_TO_RUN = ["VolumetricsB_NoADT_RBR_Both", "RutPhysics_NewParams"]


# =============================================================================
# DATA PREP + SPLITTING (80% train with target / 20% locked test)
# =============================================================================

def _resolve_named_file(names: List[str]) -> Optional[Path]:
    """Find the first existing file among `names`, searched in DOWNLOADS, the script folder and cwd."""
    roots = [DOWNLOADS, Path.cwd()]
    try:
        roots.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    for nm in names:
        p = Path(nm)
        if p.is_absolute() and p.exists():
            return p
        for r in roots:
            if (r / nm).exists():
                return r / nm
    return None


def _read_sheet(path: Path, sheets: List[Any]) -> pd.DataFrame:
    """Read the first matching sheet from `sheets`, else the first sheet."""
    xls = pd.ExcelFile(path)
    for s in sheets:
        if s != 0 and str(s) in xls.sheet_names:
            print(f"  {path.name}: sheet {s}")
            return pd.read_excel(path, sheet_name=s)
    print(f"  {path.name}: first sheet {xls.sheet_names[0]}")
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])


def _harmonize_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Clean columns, rename the target to TARGET, assign a mix ID, and copy alias columns so
    every source file ends up in the SAME canonical column space before rows are concatenated."""
    df = clean_column_names(df)
    if TARGET not in df.columns:
        for a in TARGET_ALIASES:
            if a in df.columns:
                if a != TARGET:
                    df = df.rename(columns={a: TARGET})
                    print(f"    target {a!r} -> {TARGET!r}")
                break
    if ID_COL not in df.columns:
        for a in ID_COL_ALIASES:
            if a in df.columns:
                df[ID_COL] = df[a].astype(str)
                break
    df = copy_alias_columns(df)
    df["Source_File"] = source_name
    return df


def load_data() -> Tuple[pd.DataFrame, pd.Series, Path]:
    # ---- Build the list of data sources: primary file first, then the additional ones ----
    path = resolve_file_path(RUT_FILE, RUT_FILE_FALLBACKS)
    sources: List[Tuple[Path, List[Any]]] = [(path, SHEET_CANDIDATES)]
    if COMBINE_ADDITIONAL_FILES:
        for names, sheets in ADDITIONAL_DATA_SOURCES:
            p = _resolve_named_file(names)
            if p is None:
                print(f"Additional data source not found (skipped): {names[0]}")
                continue
            if p.resolve() == path.resolve():
                continue  # already loaded as the primary
            sources.append((p, sheets))

    print("Loading data sources:")
    frames, per_source = [], []
    for p, sheets in sources:
        raw = _read_sheet(p, sheets)
        h = _harmonize_source(raw, p.name)
        h = create_engineered_columns(h)          # canonical inputs now exist -> engineer per source
        if TARGET not in h.columns:
            print(f"    WARNING: no target in {p.name}; skipped.")
            continue
        h = h.loc[pd.to_numeric(h[TARGET], errors="coerce").notna()].reset_index(drop=True)
        frames.append(h)
        per_source.append((p.name, len(h)))
    if not frames:
        raise KeyError(f"No data source produced a usable {TARGET!r} column (tried {TARGET_ALIASES}).")

    # ---- Concatenate (union of columns) and de-duplicate mixes/reports across files ----
    df = pd.concat(frames, ignore_index=True, sort=False)
    n_before = len(df)
    dedup_key = next((k for k in DEDUP_KEYS if k in df.columns), None)
    if dedup_key is not None:
        key = df[dedup_key].astype(str)
        if dedup_key != ID_COL:  # combine identity with the target so distinct results are kept
            key = key + "|" + pd.to_numeric(df[TARGET], errors="coerce").round(3).astype(str)
        df = df.loc[~key.duplicated(keep="first")].reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)

    print(f"Combined {len(frames)} source(s): "
          + ", ".join(f"{nm}={n}" for nm, n in per_source)
          + f" -> {n_before} rows; removed {n_dupes} duplicate "
          + (f"mixes/reports (key={dedup_key})" if dedup_key else "rows")
          + f" -> {len(df)} unique rows.")

    if TARGET not in df.columns:
        raise KeyError(f"Target not found (tried {TARGET_ALIASES}). Columns: {list(df.columns)[:40]}")
    # Drop rows with no target BEFORE averaging.
    df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)

    # ---- Revised SCB spec: drop invalid zero (or non-positive) Jc rows ----
    if DROP_TARGET_LE_ZERO:
        yv = pd.to_numeric(df[TARGET], errors="coerce")
        keep = yv > DROP_TARGET_LE_VALUE
        n_drop = int((~keep).sum())
        df = df.loc[keep.values].reset_index(drop=True)
        print(f"Dropped {n_drop} rows with {TARGET} <= {DROP_TARGET_LE_VALUE} "
              f"(invalid/missing SCB result) -> {len(df)} rows kept.")

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

    # ---- HARD restricted-range filter (drop rows outside the reliable target range) ----
    if RESTRICT_TARGET_RANGE:
        lo = -np.inf if TARGET_RANGE_MIN is None else float(TARGET_RANGE_MIN)
        hi = np.inf if TARGET_RANGE_MAX is None else float(TARGET_RANGE_MAX)
        in_range = (y >= lo) & (y <= hi if TARGET_RANGE_MAX_INCLUSIVE else y < hi)
        before = len(df)
        df = df.loc[in_range.values].reset_index(drop=True)
        y = y.loc[in_range.values].reset_index(drop=True)
        bound = f"{TARGET} in [{lo}, {hi}{']' if TARGET_RANGE_MAX_INCLUSIVE else ')'}"
        print(f"Restricted-range filter ON: {bound} -> kept {len(df)} of {before} rows "
              f"({100*len(df)/max(before,1):.1f}%). Model is valid ONLY on this range.")

    print("\n" + "=" * 100)
    print(f"Loaded {TARGET} data")
    print("=" * 100)
    print(f"Primary file: {path}")
    if "Source_File" in df.columns:
        print("Rows per source (after dedup): "
              + ", ".join(f"{s}={n}" for s, n in df["Source_File"].value_counts().items()))
    print(f"Rows: {len(df)} | Columns after engineering: {df.shape[1]}")
    print(f"Replicate averaging: {AVERAGE_REPLICATES} | Log-target: {LOG_TARGET}")
    if ID_COL in df.columns:
        print(f"Unique mixes ({ID_COL}): {df[ID_COL].nunique()} of {len(df)} rows"
              + ("  <-- replicates present; set GROUP_SPLIT_BY_MIX=True or AVERAGE_REPLICATES=True "
                 "to avoid leakage" if df[ID_COL].nunique() < len(df) else ""))
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


def _first_fold_holdout(bins, groups, holdout_size, seed):
    """Return (keep_idx, holdout_idx) where the holdout is ~holdout_size of the data, target-
    stratified, and GROUP-safe (no group spans the two sides) when groups is not None."""
    n = len(bins)
    n_splits = max(2, int(round(1.0 / holdout_size)))
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        keep, hold = next(iter(sgkf.split(np.zeros(n), bins, groups)))
    else:
        keep, hold = train_test_split(np.arange(n), test_size=holdout_size,
                                      random_state=seed, shuffle=True, stratify=bins)
    return np.array(keep), np.array(hold)


def stratified_train_test_split(df: pd.DataFrame, y: pd.Series):
    """Hold out the locked 20% test FIRST (target-bin stratified). The remaining 80% is the
    TRAINING data (features + Rut_20k target together — supervised learning). When
    VALIDATION_SIZE > 0 a validation slice is additionally carved out of the 80%; with the
    requested 80/20 protocol VALIDATION_SIZE = 0 and the validation index is empty (model
    selection then relies on out-of-fold CV inside the 80%).
    When GROUP_SPLIT_BY_MIX, the split is GROUP-aware (StratifiedGroupKFold on MixDesignKey) so
    no physical mix appears in more than one split — eliminating replicate leakage."""
    bins = make_target_bins(y)
    groups = None
    if GROUP_SPLIT_BY_MIX and GROUP_COL in df.columns:
        groups = df[GROUP_COL].astype(str).values

    # Stage 1: peel off the locked test (group-safe, stratified).
    dev_idx, test_idx = _first_fold_holdout(bins, groups, TEST_SIZE, RANDOM_STATE)

    if not HAS_VAL_HOLDOUT:
        # 80/20 mode: the whole 80% dev set IS the training set; no validation holdout.
        return np.array(dev_idx), np.array([], dtype=int), np.array(test_idx)

    # Stage 2 (only when VALIDATION_SIZE > 0): split dev into train + validation.
    dev_bins = bins.iloc[dev_idx].reset_index(drop=True)
    dev_groups = groups[dev_idx] if groups is not None else None
    val_fraction_of_dev = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
    train_pos, val_pos = _first_fold_holdout(dev_bins, dev_groups, val_fraction_of_dev, RANDOM_STATE)
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


def importance_ranked_feature_set(df: pd.DataFrame, y: pd.Series, dev_idx, pool_name: str, k: int):
    """Select the K most impactful features by RandomForest importance, fit on the DEV rows ONLY
    (train+validation — never the locked test). Honest 'pick the features that actually affect the
    model' step. Returns (top_feature_names, importance_series)."""
    pool = FEATURE_SETS.get(pool_name, VOLUMETRICS_B_RBR_BOTH)
    X, numerical, categorical, available, _ = get_X(df, pool)
    dev_idx = np.asarray(dev_idx)
    Xdev = X.iloc[dev_idx].reset_index(drop=True)
    ydev = pd.Series(y).iloc[dev_idx].reset_index(drop=True)
    frames = [Xdev[numerical].apply(lambda s: s.fillna(s.median()))]
    for c in categorical:                                   # encode categoricals as integer codes
        frames.append(pd.Series(pd.factorize(Xdev[c].astype(str))[0], name=c))
    Xmat = pd.concat(frames, axis=1)
    rf = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL)
    rf.fit(Xmat, ydev)
    imp = pd.Series(rf.feature_importances_, index=Xmat.columns).sort_values(ascending=False)
    top = list(imp.head(k).index)
    print(f"\nImportance-based feature selection (RandomForest on DEV, pool={pool_name}, top {k}):")
    for f in top:
        print(f"    {f:<28} importance={imp[f]:.4f}")
    return top, imp


def _selector_estimator():
    """Fixed lightweight CPU model used ONLY inside the compact selector (fast + stable;
    the winning subset is re-tuned later by the normal RandomizedSearch pipeline)."""
    if HAS_XGBOOST:
        return XGBRegressor(objective="reg:squarederror", n_estimators=500, learning_rate=0.03,
                            max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=30.0, tree_method="hist", random_state=RANDOM_STATE,
                            n_jobs=N_JOBS_MODEL)
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                 random_state=RANDOM_STATE, n_jobs=N_JOBS_MODEL)


def _numeric_matrix(X: pd.DataFrame, numerical: List[str], categorical: List[str]) -> pd.DataFrame:
    """Median-impute numerics and integer-encode categoricals into one numeric matrix."""
    frames = []
    if numerical:
        frames.append(X[numerical].apply(lambda s: s.fillna(s.median())))
    for c in categorical:
        frames.append(pd.Series(pd.factorize(X[c].astype(str))[0], name=c, index=X.index))
    return pd.concat(frames, axis=1)


def compact_feature_selector(df: pd.DataFrame, y: pd.Series, train_idx):
    """Compact mechanism-based feature selector. TRAINING rows only — the locked test never
    enters any stage.

    Stage 1-2: per-CV-fold permutation importance PI (each fold's PI measured on that fold's
               held-out part), then
                   Stability = max(0, mean(PI)) * PositiveFoldFraction / (1 + SD(PI))
               so a feature cannot rank highly because of one lucky fold.
    Stage 3:   |Spearman rho| >= COMPACT_CORRELATION_THRESHOLD drops the lower-ranked member
               of every redundant pair.
    Stage 4:   for every k in COMPACT_CANDIDATE_COUNTS, training OOF (grouped/stratified CV)
               R2 / RMSE / MAE of the top-k surviving features.
    Stage 5:   smallest k with R2_k >= R2_best - COMPACT_R2_TOLERANCE and
               RMSE_k <= COMPACT_RMSE_TOLERANCE * RMSE_reference.
    Returns (selected_features, tables_dict)."""
    pool = FEATURE_SETS.get(COMPACT_POOL, [])
    X, numerical, categorical, available, missing = get_X(df, pool)
    if len(available) < 2:
        raise ValueError(f"Compact pool {COMPACT_POOL!r} has <2 available features.")
    train_idx = np.asarray(train_idx)
    Xtr = X.iloc[train_idx].reset_index(drop=True)
    ytr = pd.Series(y).iloc[train_idx].reset_index(drop=True)
    Xmat = _numeric_matrix(Xtr, numerical, categorical)
    groups_train = _GROUPS_FULL[train_idx] if _GROUPS_FULL is not None else None
    splits, cv_name = make_cv_splits_for_training(ytr, groups_train)
    print(f"\nCompact selector: pool={COMPACT_POOL} ({len(Xmat.columns)} available, "
          f"missing: {missing or 'None'}) | {cv_name}")

    # ---- Stage 1-2: fold-wise permutation importance -> stability score ----
    pi = np.full((len(splits), Xmat.shape[1]), np.nan)
    for i, (tr, va) in enumerate(splits):
        est = clone(_selector_estimator())
        est.fit(Xmat.iloc[tr], ytr.iloc[tr])
        r = permutation_importance(est, Xmat.iloc[va], ytr.iloc[va],
                                   n_repeats=COMPACT_PI_REPEATS, random_state=RANDOM_STATE,
                                   scoring="r2", n_jobs=1)
        pi[i] = r.importances_mean
    mean_pi = np.nanmean(pi, axis=0)
    sd_pi = np.nanstd(pi, axis=0, ddof=1)
    pos_frac = np.mean(pi > 0, axis=0)
    stability = np.maximum(0.0, mean_pi) * pos_frac / (1.0 + sd_pi)
    stability_df = pd.DataFrame({
        "Feature": Xmat.columns, "Mean_PI": mean_pi, "SD_PI": sd_pi,
        "PositiveFoldFraction": pos_frac, "Stability": stability,
    }).sort_values("Stability", ascending=False).reset_index(drop=True)
    stability_df.insert(0, "Rank", range(1, len(stability_df) + 1))
    print("  Stability ranking (top 10):")
    for _, r in stability_df.head(10).iterrows():
        print(f"    {int(r['Rank']):>2}. {r['Feature']:<28} Stability={r['Stability']:.5f} "
              f"(meanPI={r['Mean_PI']:.5f}, +folds={r['PositiveFoldFraction']:.2f}, SD={r['SD_PI']:.5f})")

    # ---- Stage 3: Spearman redundancy pruning (training data only, best rank wins) ----
    corr = Xmat.corr(method="spearman").abs()
    ranked = list(stability_df["Feature"])
    kept, removed_rows = [], []
    for f in ranked:
        partner = next((k for k in kept if float(corr.loc[f, k]) >= COMPACT_CORRELATION_THRESHOLD), None)
        if partner is None:
            kept.append(f)
        else:
            removed_rows.append({"Removed_Feature": f, "Kept_Partner": partner,
                                 "Abs_Spearman_rho": float(corr.loc[f, partner]),
                                 "Threshold": COMPACT_CORRELATION_THRESHOLD})
    removed_df = pd.DataFrame(removed_rows)
    if removed_rows:
        print(f"  Redundancy pruning (|rho| >= {COMPACT_CORRELATION_THRESHOLD}): removed "
              + ", ".join(f"{r['Removed_Feature']} (kept {r['Kept_Partner']}, rho={r['Abs_Spearman_rho']:.3f})"
                          for r in removed_rows))

    # Keep only features with a non-zero stability score, in rank order.
    kept = [f for f in kept if float(stability_df.loc[stability_df["Feature"] == f, "Stability"].iloc[0]) > 0] or kept

    # ---- Stage 4: size sweep with training OOF scores ----
    counts = sorted({k for k in COMPACT_CANDIDATE_COUNTS if k <= len(kept)} | {min(len(kept), min(COMPACT_CANDIDATE_COUNTS))})
    sweep_rows = []
    for k in counts:
        feats = kept[:k]
        oof, _folds = oof_predict(_selector_estimator(), Xmat[feats], ytr, splits)
        m = metrics(ytr, oof)
        sweep_rows.append({"N_Features": k, "OOF_R2": m["R2"], "OOF_RMSE": m["RMSE"],
                           "OOF_MAE": m["MAE"], "Features": ", ".join(feats)})
        print(f"  k={k:>2}: OOF R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f}")
    sweep_df = pd.DataFrame(sweep_rows)

    # ---- Stage 5: smallest competitive subset ----
    best_row = sweep_df.loc[sweep_df["OOF_R2"].idxmax()]
    r2_best, rmse_ref = float(best_row["OOF_R2"]), float(best_row["OOF_RMSE"])
    ok = sweep_df[(sweep_df["OOF_R2"] >= r2_best - COMPACT_R2_TOLERANCE)
                  & (sweep_df["OOF_RMSE"] <= COMPACT_RMSE_TOLERANCE * rmse_ref)]
    chosen = ok.loc[ok["N_Features"].idxmin()] if not ok.empty else best_row
    selected = [f.strip() for f in chosen["Features"].split(",")]
    sweep_df["Selected"] = sweep_df["N_Features"] == chosen["N_Features"]
    summary_df = pd.DataFrame([{
        "Pool": COMPACT_POOL, "Pool_Available": len(Xmat.columns), "Kept_After_Pruning": len(kept),
        "Best_OOF_R2": r2_best, "Best_N_Features": int(best_row["N_Features"]),
        "Selected_N_Features": int(chosen["N_Features"]), "Selected_OOF_R2": float(chosen["OOF_R2"]),
        "Selected_OOF_RMSE": float(chosen["OOF_RMSE"]), "Selected_OOF_MAE": float(chosen["OOF_MAE"]),
        "R2_Tolerance": COMPACT_R2_TOLERANCE, "RMSE_Tolerance_Factor": COMPACT_RMSE_TOLERANCE,
        "Selected_Features": ", ".join(selected),
    }])
    print(f"  Selected the smallest competitive subset: {int(chosen['N_Features'])} features "
          f"(best R2={r2_best:.4f} at k={int(best_row['N_Features'])}; "
          f"selected R2={float(chosen['OOF_R2']):.4f}).")
    for i, f in enumerate(selected, 1):
        print(f"    {i}. {f}")
    tables = {"Compact_Stability": stability_df, "Compact_Corr_Pruning": removed_df,
              "Compact_Size_Sweep": sweep_df, "Compact_Selection": summary_df}
    return selected, tables


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


def make_cv_splits_for_training(y_train: pd.Series, groups_train=None):
    bins = make_target_bins(y_train)
    if GROUP_SPLIT_BY_MIX and groups_train is not None:
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(np.zeros(len(y_train)), bins, groups_train))
        name = f"Target-bin StratifiedGroupKFold (group=mix) inside training set ({CV_FOLDS} folds)"
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(np.zeros(len(y_train)), bins))
        name = f"Target-bin StratifiedKFold inside training set ({CV_FOLDS} folds, {N_TARGET_BINS} bins)"
    return splits, name


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
         "Note": "Measured on the explanation feature sample (timing only), single-threaded CPU."},
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
    # SCB Jc bands (kJ/m^2). Spec minimum is SCB_SPEC_MIN_JC.
    if y < SCB_SPEC_MIN_JC:
        return f"Below spec <{SCB_SPEC_MIN_JC:g}"
    if y < 0.75:
        return f"Marginal {SCB_SPEC_MIN_JC:g}-0.75"
    if y < 1.0:
        return "Good 0.75-1.0"
    return "High >=1.0"


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
        # NOTE: tuned for the RAW target scale; if you set LOG_TARGET=True, lower reg_lambda/
        # reg_alpha/gamma (the log target is ~10x smaller in scale).
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


def repeated_cv_robust(estimator, X, y, repeats=REPEATED_CV_REPEATS, groups=None) -> Dict[str, float]:
    """Repeated CV on the dev set for a stable generalization estimate (no test leakage).
    Group-aware (StratifiedGroupKFold on mix) when grouping is on, so no mix spans a fold."""
    bins = make_target_bins(y)
    scores = []
    for r in range(repeats):
        if GROUP_SPLIT_BY_MIX and groups is not None:
            cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
            splitter = cv.split(np.zeros(len(y)), bins, groups)
        else:
            cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
            splitter = cv.split(X)
        for tr, va in splitter:
            est = clone(estimator)
            est.fit(X.iloc[tr], y.iloc[tr])
            scores.append(r2_score(y.iloc[va], est.predict(X.iloc[va])))
    scores = np.array(scores, dtype=float)
    return {"RepeatedCV_Mean_R2": float(scores.mean()), "RepeatedCV_SD_R2": float(scores.std(ddof=1)),
            "RepeatedCV_Min_R2": float(scores.min()), "RepeatedCV_N": int(len(scores))}


def nested_cv(feature_set, model_name, X_dev, y_dev, numerical, categorical, groups=None):
    """Proper nested CV on the dev set: outer folds for an UNBIASED generalization estimate,
    inner RandomizedSearchCV for tuning inside each outer fold. Fully leakage-safe — all
    preprocessing/tuning happens inside the Pipeline inside each fold, and when grouping is on
    both the outer and inner folds are StratifiedGroupKFold so no mix crosses a fold boundary."""
    spec = define_models(feature_set).get(model_name)
    if spec is None:
        return pd.DataFrame(), pd.DataFrame()
    space = int(np.prod([len(v) for v in spec["params"].values()])) if spec["params"] else 1
    n_iter = min(NESTED_CV_INNER_NITER, max(1, space))
    bins_dev = make_target_bins(y_dev)
    grouped = GROUP_SPLIT_BY_MIX and groups is not None

    # Build the outer folds: repeats of (Stratified)GroupKFold with different seeds.
    outer_folds = []
    for rep in range(NESTED_CV_OUTER_REPEATS):
        if grouped:
            ocv = StratifiedGroupKFold(n_splits=NESTED_CV_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE + rep)
            outer_folds += list(ocv.split(np.zeros(len(y_dev)), bins_dev, groups))
        else:
            ocv = StratifiedKFold(n_splits=NESTED_CV_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE + rep)
            outer_folds += list(ocv.split(np.zeros(len(y_dev)), bins_dev))

    rows = []
    for i, (tr, te) in enumerate(outer_folds, start=1):
        Xtr, Xte = X_dev.iloc[tr], X_dev.iloc[te]
        ytr, yte = y_dev.iloc[tr], y_dev.iloc[te]
        if grouped:
            g_tr = np.asarray(groups)[tr]
            inner = list(StratifiedGroupKFold(n_splits=NESTED_CV_INNER_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE + i).split(np.zeros(len(ytr)), make_target_bins(ytr), g_tr))
        else:
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
    if X_val is not None and len(X_val) > 0:
        # Classic mode: score on the separate validation holdout.
        val_pred = candidate_est.predict(X_val)
        y_val_eval = y_val
        val_is_holdout = True
    else:
        # 80/20 mode (no validation holdout): the "validation" score is the honest
        # out-of-fold CV prediction inside the 80% training data — every row is predicted
        # by a model that never saw it during that fold's fit.
        val_pred = oof
        y_val_eval = y_train
        val_is_holdout = False
    tm, om, vm = metrics(y_train, train_pred), metrics(y_train, oof), metrics(y_val_eval, val_pred)
    gap_train_val = tm["R2"] - vm["R2"]
    fold_sd = float(folds["Fold_R2"].std(ddof=1)) if len(folds) > 1 else np.nan
    bf_oof = best_fit(y_train, oof)
    bf_val = best_fit(y_val_eval, val_pred)
    label = f"{feature_set} | {model_name}" + (" | WeightedHighRut" if weighted else "")

    row = {
        "Target": TARGET, "Feature_Set": feature_set, "Model": model_name,
        "Weighted_High_Rut": bool(weighted), "Label": label, "Device_Used": device_used,
        "N_Features_Raw": int(X_train.shape[1]), "Features_Used": ", ".join(X_train.columns),
        "Numerical_Features": ", ".join(numerical), "Categorical_Features": ", ".join(categorical),
        "CV_Method_on_Training": cv_name,
        "Validation_Source": "holdout" if val_is_holdout else "OOF 5-fold CV inside training 80%",
        "Train_R2": tm["R2"], "Train_RMSE": tm["RMSE"], "Train_MAE": tm["MAE"],
        "TrainOOF_R2": om["R2"], "TrainOOF_RMSE": om["RMSE"], "TrainOOF_MAE": om["MAE"],
        "Validation_R2": vm["R2"], "Validation_RMSE": vm["RMSE"], "Validation_MAE": vm["MAE"],
        "Validation_HighRut_MAE_q80": high_rut_mae(y_val_eval, val_pred, 0.80),
        "Gap_TrainMinusValidation_R2": float(gap_train_val),
        "Gap_TrainMinusTrainOOF_R2": float(tm["R2"] - om["R2"]),
        "TrainOOF_Fold_R2_SD": fold_sd, "TrainOOF_Fold_R2_Min": float(folds["Fold_R2"].min()) if len(folds) else np.nan,
        "Robust_Selection_Score": robust_score(vm["R2"], om["R2"], gap_train_val, fold_sd, X_train.shape[1]),
        "Validation_BestFit_Equation": bf_val["equation"], "TrainOOF_BestFit_Equation": bf_oof["equation"],
        "Meets_Val_R2_0.80": bool(vm["R2"] >= TARGET_VAL_R2),
        "Best_Params": json.dumps(search.best_params_, default=str),
    }
    row.update(relative_error_summary(y_val_eval, val_pred, "Validation_"))

    split_list = [(TRAIN_SPLIT_NAME, y_train, train_pred), (OOF_SPLIT_NAME, y_train, oof)]
    if val_is_holdout:
        split_list.append((VAL_SPLIT_NAME, y_val_eval, val_pred))
    pred_parts = []
    for split_name, ys, ps in split_list:
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
    use_holdout = X_val is not None and len(X_val) > 0

    def _bv_point(p):
        est = build_pipeline(XGBRegressor(**p), numerical, categorical)
        est.fit(X_train, y_train)
        tr = metrics(y_train, est.predict(X_train))["R2"]
        if use_holdout:
            va = metrics(y_val, est.predict(X_val))["R2"]
        else:
            # No validation holdout (80/20 mode): score by quick 3-fold CV on the training set.
            va = float(np.mean(cross_val_score(build_pipeline(XGBRegressor(**p), numerical, categorical),
                                               X_train, y_train, cv=3, scoring="r2")))
        return tr, va

    curves = {"max_depth": [1, 2, 3, 4, 5, 6], "n_estimators": [200, 400, 700, 900, 1200, 1500]}
    for param, values in curves.items():
        rows = []
        for v in values:
            p = base.copy()
            p[param] = v
            try:
                tr, va = _bv_point(p)
                rows.append({"value": v, "Train_R2": tr, "Validation_R2": va, "Gap": tr - va})
            except Exception as e:
                if is_gpu_error(e):
                    p.pop("device", None)
                    try:
                        tr, va = _bv_point(p)
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
        plt.title("Final model permutation importance")
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
    for ds in [VAL_SPLIT_NAME, OOF_SPLIT_NAME, TEST_SPLIT_NAME]:
        d = pred_df[pred_df["Dataset"] == ds].copy()
        if d.empty:
            continue
        # SCB: a mix is AT-RISK (positive class) when its Jc is BELOW the spec minimum
        # (low fracture energy = crack-prone). We therefore score the NEGATED prediction so a
        # lower predicted Jc yields a higher risk score.
        y_cls = (d["Measured"] < RUT_RISK_THRESHOLD_MM).astype(int)
        if y_cls.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_cls, -d["Predicted"].astype(float))
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"Threshold ROC: {ds}, below-spec Jc < {RUT_RISK_THRESHOLD_MM:g} kJ/m2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = out_dir / f"roc_{safe_name(ds)}.png"
        save_fig(p)
        graphs.append({"Graph": p.name, "Type": "ROC below-spec risk", "AUC": roc_auc})
    return graphs


# =============================================================================
# FINAL REFIT + LOCKED TEST (scored once)
# =============================================================================

def final_refit_and_test(final_estimator, X_train, y_train, X_val, y_val, X_test, y_test, oof_df=None):
    """Refit the selected model on the full training data (features + Rut_20k target — the
    target is what the model learns from) and score the locked 20% test EXACTLY ONCE.
    oof_df: optional DataFrame with Measured/Predicted out-of-fold CV predictions on the
    training rows — carried into the final report (and used for calibration when there is no
    validation holdout) as the honest, leakage-free view of training performance."""
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_dev = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_dev = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    else:
        # 80/20 mode: the training set IS the full 80% development data.
        X_dev = X_train.reset_index(drop=True)
        y_dev = y_train.reset_index(drop=True)
    final_est = clone(final_estimator)
    final_est.fit(X_dev, y_dev)

    split_list = [(TRAIN_SPLIT_NAME, X_train, y_train)]
    if has_val:
        split_list += [(VAL_SPLIT_NAME, X_val, y_val), (DEV_SPLIT_NAME, X_dev, y_dev)]
    split_list.append((TEST_SPLIT_NAME, X_test, y_test))
    pred_parts = []
    for ds, Xs, ys in split_list:
        ps = final_est.predict(Xs)
        d = pd.DataFrame({"Dataset": ds, "Measured": np.asarray(ys, float), "Predicted": np.asarray(ps, float)})
        pred_parts.append(add_error_columns(d))
    if oof_df is not None and len(oof_df) > 0:
        d = pd.DataFrame({"Dataset": OOF_SPLIT_NAME,
                          "Measured": np.asarray(oof_df["Measured"], float),
                          "Predicted": np.asarray(oof_df["Predicted"], float)})
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
    metrics_df = pd.DataFrame(rows)

    # ---- Compression-calibration equation (fit on DEV predictions, applied to every split) ----
    # With a validation holdout the calibration is fit on the refit model's DEV predictions;
    # in 80/20 mode (no holdout) it is fit on the OUT-OF-FOLD CV predictions when available
    # (honest — each row predicted by a model that never saw it), else on the training fit.
    if APPLY_CALIBRATION:
        if has_val:
            calib_ds = DEV_SPLIT_NAME
        elif oof_df is not None and len(oof_df) > 0:
            calib_ds = OOF_SPLIT_NAME
        else:
            calib_ds = TRAIN_SPLIT_NAME
        dev = pred_df[pred_df["Dataset"] == calib_ds]
        a, b = np.polyfit(dev["Predicted"].values, dev["Measured"].values, 1)  # Measured = a*Pred + b
        a, b = float(a), float(b)
        pred_df["Predicted_Calibrated"] = a * pred_df["Predicted"] + b
        cal_rows = []
        for ds, sub in pred_df.groupby("Dataset"):
            m = metrics(sub["Measured"], sub["Predicted_Calibrated"])
            bf = best_fit(sub["Measured"], sub["Predicted_Calibrated"])
            cal_rows.append({"Dataset": f"{ds}_Calibrated", "Rows": len(sub), **m,
                             "HighRut_MAE_q80": high_rut_mae(sub["Measured"], sub["Predicted_Calibrated"], 0.80),
                             "BestFit_Equation": bf["equation"]})
        metrics_df = pd.concat([metrics_df, pd.DataFrame(cal_rows)], ignore_index=True)
        print(f"\nCalibration equation  Measured = {a:.4f} * Predicted + {b:.4f}  "
              f"(learned on {calib_ds} — never the locked test; corrects compression -> "
              f"lower RMSE/MAE/bias AND typically higher coefficient-of-determination R2).")

    # ---- In-range band report (do not drop; show reliable band beside the full range) ----
    if REPORT_INRANGE_BAND:
        lo, hi = REPORT_INRANGE_BAND
        band_rows = []
        for ds, sub in pred_df.groupby("Dataset"):
            m_in = sub[(sub["Measured"] >= lo) & (sub["Measured"] <= hi)]
            if len(m_in) < 3:
                continue
            m = metrics(m_in["Measured"], m_in["Predicted"])
            bf = best_fit(m_in["Measured"], m_in["Predicted"])
            band_rows.append({"Dataset": f"{ds}_InRange_{lo:g}-{hi:g}", "Rows": len(m_in), **m,
                              "HighRut_MAE_q80": high_rut_mae(m_in["Measured"], m_in["Predicted"], 0.80),
                              "BestFit_Equation": bf["equation"]})
        if band_rows:
            metrics_df = pd.concat([metrics_df, pd.DataFrame(band_rows)], ignore_index=True)
            print(f"In-range report added for {TARGET} in [{lo:g}, {hi:g}] "
                  f"(shown beside the full range on every split).")
    return final_est, X_dev, y_dev, pred_df, metrics_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print(f"SCB (Jc) WORKFLOW v3 — {SPLIT_TAG.replace('_', '/')} + COMBINED DATA + PHYSICS + STACKING + FULL DIAGNOSTICS")
    print("=" * 100)
    print(f"Goal: push CV/validation R2 toward {TARGET_VAL_R2:.2f}; keep the 20% test locked/hidden.")
    print(gpu_status_string())
    if HAS_VAL_HOLDOUT:
        print(f"Split: {int(TRAIN_SIZE*100)}% train, {int(VALIDATION_SIZE*100)}% val, {int(TEST_SIZE*100)}% locked test")
    else:
        print(f"Split: {int(TRAIN_SIZE*100)}% training (features + {TARGET} target together — "
              f"supervised learning), {int(TEST_SIZE*100)}% locked test for the FINAL EVALUATION.")
        print("No separate validation holdout: model/feature selection uses out-of-fold "
              f"{CV_FOLDS}-fold CV inside the {int(TRAIN_SIZE*100)}% training data.")
    print(f"XGBoost={HAS_XGBOOST} LightGBM={HAS_LIGHTGBM} CatBoost={HAS_CATBOOST} SHAP={HAS_SHAP} | QUICK_SMOKE_TEST={QUICK_SMOKE_TEST}")
    print(f"Output: {OUTPUT_FOLDER}")

    df, y, input_path = load_data()
    global _GROUPS_FULL
    _GROUPS_FULL = df[GROUP_COL].astype(str).values if (GROUP_SPLIT_BY_MIX and GROUP_COL in df.columns) else None
    train_idx, val_idx, test_idx = stratified_train_test_split(df, y)
    if _GROUPS_FULL is not None:
        # verify zero mix overlap across splits (proves no replicate leakage)
        g = _GROUPS_FULL
        leak = (len(set(g[train_idx]) & set(g[test_idx])) + len(set(g[val_idx]) & set(g[test_idx]))
                + len(set(g[train_idx]) & set(g[val_idx])))
        print(f"Group leakage control ON (group={GROUP_COL}): mixes shared across splits = {leak} (must be 0)")

    split_rows = [
        {"Split": TRAIN_SPLIT_NAME, "Rows": len(train_idx), "Percent": 100 * len(train_idx) / len(df),
         "Purpose": f"fit/tune WITH the {TARGET} target as the label ({CV_FOLDS}-fold CV + OOF validation)"},
    ]
    if HAS_VAL_HOLDOUT:
        split_rows.append({"Split": VAL_SPLIT_NAME, "Rows": len(val_idx), "Percent": 100 * len(val_idx) / len(df),
                           "Purpose": "model/feature selection"})
    split_rows.append({"Split": TEST_SPLIT_NAME, "Rows": len(test_idx), "Percent": 100 * len(test_idx) / len(df),
                       "Purpose": "final one-time evaluation (target HIDDEN until scoring)"})
    split_summary = pd.DataFrame(split_rows)
    print("\nSplit summary:")
    print(split_summary.to_string(index=False))

    df.iloc[train_idx].to_excel(PATHS["splits"] / f"train_{int(TRAIN_SIZE*100)}pct_rows.xlsx", index=False)
    if len(val_idx) > 0:
        df.iloc[val_idx].to_excel(PATHS["splits"] / f"validation_{int(VALIDATION_SIZE*100)}pct_rows.xlsx", index=False)
    df.iloc[test_idx].to_excel(PATHS["splits"] / "LOCKED_test_20pct_DO_NOT_USE_FOR_TUNING.xlsx", index=False)

    # Statistical analysis on DEV data only (no test leakage).
    dev_idx_all = np.concatenate([train_idx, val_idx])

    # ---- Importance-based feature selection: register a reduced high-impact feature set ----
    if ADD_TOP_IMPACT_FEATURE_SET:
        try:
            top_feats, _imp = importance_ranked_feature_set(df, y, dev_idx_all, TOP_IMPACT_POOL, TOP_IMPACT_K)
            FEATURE_SETS["TopImpact_Selected"] = top_feats
            if "TopImpact_Selected" not in FEATURE_SETS_TO_RUN:
                FEATURE_SETS_TO_RUN.append("TopImpact_Selected")
        except Exception as e:
            print(f"TopImpact feature selection skipped: {type(e).__name__}: {e}")

    # ---- Compact mechanism-based selector: stability -> redundancy pruning -> size sweep ----
    compact_tables: Dict[str, pd.DataFrame] = {}
    if RUN_COMPACT_SELECTOR:
        try:
            compact_feats, compact_tables = compact_feature_selector(df, y, train_idx)
            if compact_feats:
                FEATURE_SETS["Compact_Selected"] = compact_feats
                if "Compact_Selected" not in FEATURE_SETS_TO_RUN:
                    FEATURE_SETS_TO_RUN.append("Compact_Selected")
        except Exception as e:
            print(f"Compact selector skipped: {type(e).__name__}: {e}")

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
        groups_train = _GROUPS_FULL[train_idx] if _GROUPS_FULL is not None else None
        cv_splits, cv_name = make_cv_splits_for_training(y_train, groups_train)

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
                                             "X_test": X_test, "y_test": y_test, "feature_set": fs_name,
                                             "cv_splits": cv_splits}
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
                if HAS_VAL_HOLDOUT and len(sc["X_val"]) > 0:
                    val_pred = stack.predict(sc["X_val"])
                    y_val_eval = sc["y_val"]
                else:
                    # 80/20 mode: score the stack honestly by out-of-fold CV on the training 80%.
                    print("    (no validation holdout: scoring the stack by out-of-fold CV on the training set)")
                    val_pred, _stack_folds = oof_predict(stack, sc["X_train"], sc["y_train"], sc["cv_splits"])
                    y_val_eval = sc["y_train"]
                tm, vm = metrics(sc["y_train"], train_pred), metrics(y_val_eval, val_pred)
                gap = tm["R2"] - vm["R2"]
                row = {"Target": TARGET, "Feature_Set": sc["feature_set"], "Model": "StackingRegressor",
                       "Weighted_High_Rut": False, "Label": f"{sc['feature_set']} | StackingRegressor",
                       "Device_Used": "stacking", "N_Features_Raw": int(sc["X_train"].shape[1]),
                       "Features_Used": ", ".join(sc["X_train"].columns),
                       "Numerical_Features": ", ".join(sc["numerical"]), "Categorical_Features": ", ".join(sc["categorical"]),
                       "CV_Method_on_Training": "StackingRegressor internal cv=5 OOF",
                       "Validation_Source": ("holdout" if HAS_VAL_HOLDOUT
                                             else "OOF 5-fold CV inside training 80%"),
                       "Train_R2": tm["R2"], "Train_RMSE": tm["RMSE"], "Train_MAE": tm["MAE"],
                       "TrainOOF_R2": vm["R2"], "TrainOOF_RMSE": vm["RMSE"], "TrainOOF_MAE": vm["MAE"],
                       "Validation_R2": vm["R2"], "Validation_RMSE": vm["RMSE"], "Validation_MAE": vm["MAE"],
                       "Validation_HighRut_MAE_q80": high_rut_mae(y_val_eval, val_pred, 0.80),
                       "Gap_TrainMinusValidation_R2": float(gap), "Gap_TrainMinusTrainOOF_R2": float(gap),
                       "TrainOOF_Fold_R2_SD": np.nan, "TrainOOF_Fold_R2_Min": np.nan,
                       "Robust_Selection_Score": robust_score(vm["R2"], vm["R2"], gap, 0.0, sc["X_train"].shape[1]),
                       "Validation_BestFit_Equation": best_fit(y_val_eval, val_pred)["equation"],
                       "TrainOOF_BestFit_Equation": "NA (stacking)",
                       "Meets_Val_R2_0.80": bool(vm["R2"] >= TARGET_VAL_R2), "Best_Params": "stacking"}
                row.update(relative_error_summary(y_val_eval, val_pred, "Validation_"))
                results_rows.append(row)
                stack_split_list = [(TRAIN_SPLIT_NAME, sc["y_train"], train_pred)]
                stack_split_list.append((VAL_SPLIT_NAME, y_val_eval, val_pred) if HAS_VAL_HOLDOUT
                                        else (OOF_SPLIT_NAME, y_val_eval, val_pred))
                pp = []
                for sn, ys, ps in stack_split_list:
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
                # groups aligned to X_dev (train rows then val rows), so no mix spans a CV fold
                groups_dev = (np.concatenate([_GROUPS_FULL[train_idx], _GROUPS_FULL[val_idx]])
                              if _GROUPS_FULL is not None else None)
                rc = repeated_cv_robust(obj["best_estimator"], X_dev, y_dev, groups=groups_dev)
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
            groups_nested = _GROUPS_FULL[dev_idx_all] if _GROUPS_FULL is not None else None
            nested_folds_df, nested_summary_df = nested_cv(NESTED_CV_FEATURE_SET, NESTED_CV_MODEL,
                                                          Xn_dev, yn_dev, n_num, n_cat, groups=groups_nested)
            if not nested_summary_df.empty:
                r = nested_summary_df.iloc[0]
                print(f"  Nested CV R2 = {r['NestedCV_Mean_R2']:.4f} +/- {r['NestedCV_SD_R2']:.4f} "
                      f"(min {r['NestedCV_Min_R2']:.4f}, {int(r['NestedCV_Outer_Folds'])} outer folds) -- UNBIASED estimate")
        except Exception as e:
            print(f"  Nested CV failed: {type(e).__name__}: {e}")

    # ---- Select final model ----
    selected = None
    if AUTO_SELECT_HIGHEST_VALIDATION:
        # honest CV score among SHAP-capable single models (keeps SHAP/PDP working)
        sort_col = AUTO_SELECT_METRIC if AUTO_SELECT_METRIC in results_df.columns else "Validation_R2"
        # SHAP_PREFER_BOOSTER: restrict to gradient boosters so the final SHAP beeswarm is clean
        # (ExtraTrees/RandomForest give muddy, non-monotonic SHAP colours). Falls back to all
        # SHAP-capable models if no booster finished.
        allowed = SHAP_BOOSTER_MODELS if SHAP_PREFER_BOOSTER else SHAP_CAPABLE_MODELS
        cand = results_df[results_df["Model"].isin(allowed)].sort_values(sort_col, ascending=False)
        if cand.empty and SHAP_PREFER_BOOSTER:
            cand = results_df[results_df["Model"].isin(SHAP_CAPABLE_MODELS)].sort_values(
                sort_col, ascending=False)
        if not cand.empty:
            selected = cand.iloc[0]
            print(f"\nFinal model AUTO-SELECTED by HIGHEST {sort_col} "
                  f"({'gradient booster for clean SHAP' if SHAP_PREFER_BOOSTER else 'SHAP-capable'}, "
                  f"honest CV): {selected['Label']} ({sort_col}={selected[sort_col]:.4f}, "
                  f"Validation R2={selected['Validation_R2']:.4f})")
    if selected is None and FORCE_FINAL_FEATURE_SET and FORCE_FINAL_MODEL:
        forced_label = f"{FORCE_FINAL_FEATURE_SET} | {FORCE_FINAL_MODEL}"
        cand = results_df[results_df["Label"] == forced_label]
        if not cand.empty:
            selected = cand.iloc[0]
            print(f"\nFinal model FORCED to: {forced_label}")
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
    is_tree = obj["model_name"] in SHAP_CAPABLE_MODELS
    diag = PATHS["figures"] / "diagnostics"

    # =========================================================================
    # BRANCH ON THE LOCKED-TEST SWITCH
    # =========================================================================
    if SCORE_LOCKED_TEST:
        mode_label = "FINAL EVALUATION (locked 20% test revealed once)"
        print("\n*** SCORE_LOCKED_TEST=True: revealing the locked 20% test ONE TIME. ***")
        # Honest OOF CV predictions on the training rows, carried into the final report
        # (and used to fit the calibration when there is no validation holdout).
        oof_pred_rows = obj["cand_pred"][obj["cand_pred"]["Dataset"] == OOF_SPLIT_NAME]
        oof_pred_rows = oof_pred_rows if len(oof_pred_rows) > 0 else None
        final_est, X_pdp, y_dev, final_pred_df, final_metrics_df = final_refit_and_test(
            obj["best_estimator"], obj["X_train"], obj["y_train"], obj["X_val"], obj["y_val"],
            obj["X_test"], obj["y_test"], oof_df=oof_pred_rows)
        final_model_path = PATHS["models"] / f"FINAL_SELECTED_MODEL_refit_on_train{int((TRAIN_SIZE+VALIDATION_SIZE)*100)}.joblib"
        joblib.dump(final_est, final_model_path)
        print("\nFinal metrics (LOCKED TEST scored once):")
        print(final_metrics_df.to_string(index=False))
        diag_splits = list(final_pred_df["Dataset"].unique())
        if HAS_VAL_HOLDOUT:
            X_explain = pd.concat([obj["X_val"], obj["X_test"]], axis=0).reset_index(drop=True)
            explain_pred = pd.concat([final_pred_df[final_pred_df["Dataset"] == VAL_SPLIT_NAME],
                                      final_pred_df[final_pred_df["Dataset"] == TEST_SPLIT_NAME]], ignore_index=True)
            band_explain = pd.concat([band_series_full.iloc[val_idx], band_series_full.iloc[test_idx]],
                                     axis=0).reset_index(drop=True)
            band_split_idx = {TRAIN_SPLIT_NAME: train_idx, VAL_SPLIT_NAME: val_idx, TEST_SPLIT_NAME: test_idx}
            ad_targets = [(VAL_SPLIT_NAME, val_idx), (TEST_SPLIT_NAME, test_idx)]
        else:
            # 80/20 mode: explain on the revealed test set; error tables also cover the
            # honest OOF view of the training rows.
            X_explain = obj["X_test"].reset_index(drop=True)
            explain_pred = final_pred_df[final_pred_df["Dataset"] == TEST_SPLIT_NAME].reset_index(drop=True)
            band_explain = band_series_full.iloc[test_idx].reset_index(drop=True)
            band_split_idx = {TRAIN_SPLIT_NAME: train_idx, TEST_SPLIT_NAME: test_idx}
            if oof_pred_rows is not None:
                band_split_idx[OOF_SPLIT_NAME] = train_idx
            ad_targets = ([(OOF_SPLIT_NAME, train_idx)] if oof_pred_rows is not None else []) + \
                         [(TEST_SPLIT_NAME, test_idx)]
        if SHAP_EXPLAIN_ON_ALL_ROWS:
            # Representative beeswarm: explain on ALL rows (train + test), not just the small
            # locked-test slice, so the SHAP colours are dense enough to read.
            X_explain = pd.concat([obj["X_train"], obj["X_test"]], axis=0).reset_index(drop=True)
            explain_pred = pd.concat(
                [final_pred_df[final_pred_df["Dataset"] == TRAIN_SPLIT_NAME],
                 final_pred_df[final_pred_df["Dataset"] == TEST_SPLIT_NAME]], ignore_index=True)
            band_explain = pd.concat([band_series_full.iloc[train_idx], band_series_full.iloc[test_idx]],
                                     axis=0).reset_index(drop=True)
        metrics_sheet = "Final_TrainValTest_Metrics" if HAS_VAL_HOLDOUT else "Final_TrainTest_Metrics"
        workbook_name = f"SCB_v3_{SPLIT_TAG}_WITH_LOCKED_TEST_Results.xlsx"
    else:
        mode_label = "DEVELOPMENT ONLY (locked 20% test untouched)"
        print("\n*** SCORE_LOCKED_TEST=False: DEVELOPMENT-ONLY run. "
              "The 20% test is NOT read, scored, explained, or plotted. ***")
        # Selected model stays fit on the TRAIN set only; the test stays genuinely hidden.
        final_est = obj["fitted"]
        final_model_path = PATHS["models"] / f"SELECTED_MODEL_dev_only_fit_on_train{int(TRAIN_SIZE*100)}.joblib"
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
        print("\nDevelopment metrics (train / train-OOF" + (" / validation" if HAS_VAL_HOLDOUT else "")
              + " — NO test):")
        print(final_metrics_df.to_string(index=False))
        X_pdp = pd.concat([obj["X_train"], obj["X_val"]], axis=0).reset_index(drop=True)
        diag_splits = list(final_pred_df["Dataset"].unique())
        if HAS_VAL_HOLDOUT:
            X_explain = obj["X_val"].reset_index(drop=True)
            explain_pred = final_pred_df[final_pred_df["Dataset"] == VAL_SPLIT_NAME].reset_index(drop=True)
            band_explain = band_series_full.iloc[val_idx].reset_index(drop=True)
            band_split_idx = {TRAIN_SPLIT_NAME: train_idx, VAL_SPLIT_NAME: val_idx}
            ad_targets = [(VAL_SPLIT_NAME, val_idx)]
        else:
            # 80/20 mode: explain via the honest OOF view of the training rows.
            X_explain = obj["X_train"].reset_index(drop=True)
            explain_pred = final_pred_df[final_pred_df["Dataset"] == OOF_SPLIT_NAME].reset_index(drop=True)
            band_explain = band_series_full.iloc[train_idx].reset_index(drop=True)
            band_split_idx = {TRAIN_SPLIT_NAME: train_idx, OOF_SPLIT_NAME: train_idx}
            ad_targets = [(OOF_SPLIT_NAME, train_idx)]
        metrics_sheet = "Dev_Train_Val_Metrics" if HAS_VAL_HOLDOUT else "Dev_Train_OOF_Metrics"
        workbook_name = f"SCB_v3_{SPLIT_TAG}_DEV_ONLY_train_val_Results.xlsx"

    # ---- Real-time serving metrics for the selected model (latency + throughput) ----
    # Timed on the explanation feature sample (timing only — target values are not used).
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

    if HAS_VAL_HOLDOUT and len(obj["X_val"]) > 0:
        perm_X, perm_y = obj["X_val"], obj["y_val"]
    else:
        # No validation holdout: permutation importance on the training rows.
        perm_X, perm_y = obj["X_train"], obj["y_train"]
    perm_df = run_permutation(final_est, perm_X, perm_y, PATHS["figures"] / "feature_sets")
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
    weakness_split = VAL_SPLIT_NAME if HAS_VAL_HOLDOUT else OOF_SPLIT_NAME
    if not error_by_rut_range.empty:
        for _, r in error_by_rut_range[error_by_rut_range["Dataset"] == weakness_split].iterrows():
            weakness_rows.append({"Dimension": "Rut range", "Group": r["Rut_Range"], "Rows": r["Rows"],
                                  "R2": r["R2"], "RMSE": r["RMSE"], "MAE": r["MAE"], "Mean_Bias": r["Mean_Bias_PredMinusMeas"]})
    if not error_by_rbr_band.empty:
        for _, r in error_by_rbr_band[error_by_rbr_band["Dataset"] == weakness_split].iterrows():
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
        {"Setting": "Split", "Value": (f"{int(TRAIN_SIZE*100)}% train / {int(VALIDATION_SIZE*100)}% validation / {int(TEST_SIZE*100)}% LOCKED test"
                                       if HAS_VAL_HOLDOUT else
                                       f"{int(TRAIN_SIZE*100)}% training (WITH {TARGET} target as label) / {int(TEST_SIZE*100)}% LOCKED test; validation = OOF {CV_FOLDS}-fold CV inside training")},
        {"Setting": "Split method", "Value": "Target-bin stratified random; fixed RANDOM_STATE"},
        {"Setting": "Nested CV", "Value": (f"{NESTED_CV_FEATURE_SET}|{NESTED_CV_MODEL}, outer {NESTED_CV_OUTER_SPLITS}x{NESTED_CV_OUTER_REPEATS}, inner {NESTED_CV_INNER_SPLITS}-fold" if RUN_NESTED_CV else "off")},
        {"Setting": "Forced final model", "Value": f"{FORCE_FINAL_FEATURE_SET} | {FORCE_FINAL_MODEL}" if FORCE_FINAL_FEATURE_SET else "auto robust-score"},
        {"Setting": "High-rut weighting", "Value": RUN_HIGH_RUT_WEIGHTING},
        {"Setting": "Replicate averaging (Option B)", "Value": f"{AVERAGE_REPLICATES} (one averaged target row per {ID_COL}; removes replicate leakage)"},
        {"Setting": "Log-target (Option B)", "Value": f"{LOG_TARGET} (train on log1p(target); metrics back on original scale)"},
        {"Setting": "Random state", "Value": RANDOM_STATE},
        {"Setting": "Train/Val/Test rows", "Value": f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}"},
        {"Setting": "Training CV", "Value": f"{CV_FOLDS}-fold target-bin StratifiedKFold inside {int(TRAIN_SIZE*100)}% train"},
        {"Setting": "Leakage control (group by mix)", "Value": (f"ON — group={GROUP_COL}; split + training CV + RepeatedCV + Nested CV are all StratifiedGroupKFold so no mix spans any fold" if GROUP_SPLIT_BY_MIX else "OFF (row-level split)")},
        {"Setting": "Repeated CV", "Value": f"{CV_FOLDS}x{REPEATED_CV_REPEATS} on 80% dev for best instance per model family + ensemble"},
        {"Setting": "N_ITER_XGB", "Value": N_ITER_XGB},
        {"Setting": "GPU status", "Value": gpu_status_string()},
        {"Setting": "True RBR", "Value": "RBR_JMF_fraction (=RBR_decimal), RBR_JMF_percent (=RBR_percent)"},
        {"Setting": "Compact selector", "Value": (f"{RUN_COMPACT_SELECTOR} — pool={COMPACT_POOL}; stability-scored fold PI, "
                                                  f"|rho|>={COMPACT_CORRELATION_THRESHOLD} pruning, counts {COMPACT_CANDIDATE_COUNTS}, "
                                                  f"smallest subset within R2_best-{COMPACT_R2_TOLERANCE} and "
                                                  f"{COMPACT_RMSE_TOLERANCE}x RMSE_ref (training rows only)")},
        {"Setting": "Stacking", "Value": f"{RUN_STACKING} ({list(best_params_for_stack.keys())})"},
        {"Setting": "Selection", "Value": ("Validation/Robust score; locked test never used for selection"
                                           if HAS_VAL_HOLDOUT else
                                           "OOF CV score; locked test never used for selection")},
        {"Setting": "SCORE_LOCKED_TEST", "Value": SCORE_LOCKED_TEST},
        {"Setting": "Run mode", "Value": mode_label},
        {"Setting": "Final fit", "Value": (f"Selected model refit on the full {int((TRAIN_SIZE+VALIDATION_SIZE)*100)}% training data "
                                           f"(features + {TARGET} target), tested once on the {int(TEST_SIZE*100)}% locked test"
                                           if SCORE_LOCKED_TEST else
                                           f"Selected model fit on {int(TRAIN_SIZE*100)}% train; {int(TEST_SIZE*100)}% test UNTOUCHED")},
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
        rec["LockedTest_R2"] = float(final_metrics_df.loc[final_metrics_df["Dataset"] == TEST_SPLIT_NAME, "R2"].iloc[0])
        rec["LockedTest_RMSE"] = float(final_metrics_df.loc[final_metrics_df["Dataset"] == TEST_SPLIT_NAME, "RMSE"].iloc[0])
        rec["LockedTest_MAE"] = float(final_metrics_df.loc[final_metrics_df["Dataset"] == TEST_SPLIT_NAME, "MAE"].iloc[0])
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
        for _sheet, _tdf in compact_tables.items():
            if _tdf is not None and not _tdf.empty:
                _tdf.to_excel(writer, sheet_name=_sheet[:31], index=False)
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
            folds_df.to_excel(writer, sheet_name=f"CV_Folds_Train{int(TRAIN_SIZE*100)}", index=False)
        if not predictions_df.empty:
            predictions_df.head(200000).to_excel(writer, sheet_name="All_Candidate_Predictions", index=False)
        if not graph_index_df.empty:
            graph_index_df.to_excel(writer, sheet_name="Graph_Index", index=False)

    for _sheet, _tdf in compact_tables.items():
        if _tdf is not None and not _tdf.empty:
            _tdf.to_csv(PATHS["tables"] / f"{_sheet.lower()}.csv", index=False)
    results_df.to_csv(PATHS["tables"] / "candidate_results.csv", index=False)
    decision_df.to_csv(PATHS["tables"] / "decision_table.csv", index=False)
    final_metrics_df.to_csv(PATHS["tables"] / "final_train_val_test_metrics.csv", index=False)
    final_pred_df.to_csv(PATHS["tables"] / "final_predictions.csv", index=False)

    locked_test_file = PATHS["splits"] / "LOCKED_test_20pct_DO_NOT_USE_FOR_TUNING.xlsx"
    elapsed = time.time() - t0
    print("\n" + "=" * 100)
    print(f"FINISHED SCB (Jc) v3 {SPLIT_TAG.replace('_', '/')} WORKFLOW — {mode_label}")
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
