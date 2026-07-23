

from __future__ import annotations

import json
import math
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# Avoid nested process pools on Windows/Spyder. Cross-validation runs serially,
# while tree estimators can use their own internal threads.
SEARCH_N_JOBS = 1
MODEL_N_JOBS = -1

# ---- FULL rutting + SCB dataset ----
# The full cleaned data lives in the Design_Validation workbook. Both the Design_Complete and
# Validation_Complete sheets carry BOTH targets (Rut_20k and SCB); COMBINE_FULL_DATA stacks them
# into one dataframe so the model trains on ALL available mixes for both targets.
# The path auto-falls back to the current user's Downloads folder (or the working directory) when
# the hard-coded path does not exist, so it runs on any machine.
input_file = Path(
    r"C:\Users\lenovo\Downloads\Design_Validation_Separated_Complete_Cleaned.xlsx"
)
if not input_file.exists():
    for _cand in [
        Path.home() / "Downloads" / "Design_Validation_Separated_Complete_Cleaned.xlsx",
        Path.cwd() / "Design_Validation_Separated_Complete_Cleaned.xlsx",
    ]:
        if _cand.exists():
            input_file = _cand
            break

# Combine Design_Complete + Validation_Complete for the FULL dataset (both targets in each sheet).
COMBINE_FULL_DATA = True
FULL_DATA_SHEETS = ["Design_Complete", "Validation_Complete"]
# Content-based de-duplication: remove rows that are byte-for-byte identical on the modelling
# features + target (the SAME mix reported twice), so the random train/test split cannot place an
# identical mix in both train and test. Strongly recommended (prevents data leakage).
DEDUP_FULL_DATA = True

preferred_sheet = "Design_Complete"

output_folder = Path(
    r"C:\Users\lenovo\Downloads\Publication_Modeling_Rut_SCB_Outputs"
)
if not output_folder.parent.exists():
    output_folder = (Path.home() / "Downloads" / "Publication_Modeling_Rut_SCB_Outputs")

# Full publication run controls.
SHOW_PLOTS_IN_SPYDER = True
SAVE_OUTPUTS = True
INCLUDE_OPTIONAL_MODELS = False
RUN_ABLATION_STUDY = True
RUN_SHAP = True

# Controlled search sizes for a 535-row dataset.
# Active primary models are intentionally limited to regularized boosting models.
N_ITER_GBR = 50
N_ITER_XGB = 50
N_ITER_OPTIONAL = 20

FIGURE_DPI = 300
SHAP_MAX_SAMPLES = 200
RELATIVE_ERROR_EPSILON = 1e-9

sns.set_theme(style="whitegrid", context="talk")
np.random.seed(RANDOM_STATE)


# =============================================================================
# 2. TARGETS, FEATURES, AND COLUMN ALIASES
# =============================================================================

TARGET_ALIASES = {
    "Rut_20k": [
        "Rut_20k",
        "Rut 20k",
        "Rut20k",
        "LWT Rut_20k",
        "LWT_Rut_20k",
        "rut depth 20k",
    ],
    "SCB": [
        "SCB",
        "SCB Jc",
        "SCB_Jc",
        "Jc",
        "SCB fracture",
        "SCB_Jc_kJ_m2",
    ],
}

FEATURE_ALIASES = {
    "ADT_DOTD_ord": [
        "ADT_DOTD_ord",
        "ADT DOTD ord",
        "ADT_ord",
        "ADT encoded",
    ],
    "PG Grade": [
        "PG Grade",
        "PGGrade",
        "PG_Grade",
        "PG_HighTemp",
        "Binder PG",
        "AsphaltMaterialName",
    ],
    "RAP_pct": [
        "RAP_pct",
        "RAP pct",
        "RAP %",
        "RAP",
        "MixRapTotal",
    ],
    "ACinRAP": [
        "ACinRAP",
        "AC in RAP",
        "AC_RAP",
        "RAP binder content",
    ],
    "Pass_4.75mm": [
        "Pass_4.75mm",
        "Pass4.75",
        "Passing 4.75 mm",
        "No4",
    ],
    "Pass_0.075mm": [
        "Pass_0.075mm",
        "Pass0.075",
        "Passing 0.075 mm",
        "No200",
    ],
    "Va": [
        "Va",
        "VTM",
        "AirVoids",
        "Air Voids",
        "DesignAirVoids",
    ],
    "VMA": ["VMA", "DesignVMA"],
    "Dust_Binder": [
        "Dust_Binder",
        "Dust Binder",
        "Dust/Binder",
        "Dust_to_Binder",
    ],
    "Gmm": ["Gmm", "Theoretical Maximum Specific Gravity"],
    "Gsb": ["Gsb", "Aggregate Bulk Specific Gravity"],
    "SandEq": ["SandEq", "Sand Equivalent", "Sand Eq"],
    "FAA": ["FAA", "Fine Aggregate Angularity"],
    "CAA": ["CAA", "Coarse Aggregate Angularity"],
    "NMAS": [
        "NMAS",
        "NMAS (mm)",
        "NMAS_mm",
        "Nominal Max Agg Size",
        "Nominal Aggregate Size",
    ],
    "Absorption": ["Absorption", "Aggregate Absorption"],
    "AsphaltContent_Design": [
        "AsphaltContent_Design",
        "Asphalt Content",
        "Design Asphalt Content",
        "DesignGmmAsphaltContent",
        "Design AC",
        "AC Design",
        "AC",
    ],
}

TARGET_CONFIGS = {
    "Rut_20k": {
        "target_aliases": TARGET_ALIASES["Rut_20k"],
        "requested_features": [
            "ADT_DOTD_ord",
            "PG Grade",
            "RAP_pct",
            "ACinRAP",
            "Pass_4.75mm",
            "Va",
            "VMA",
            "Dust_Binder",
            "Gmm",
            "SandEq",
            "FAA",
            "NMAS",
            "Absorption",
        ],
        "units": "mm",
        "meaning": (
            "Higher Rut_20k indicates greater permanent deformation and worse "
            "rutting resistance."
        ),
    },
    "SCB": {
        "target_aliases": TARGET_ALIASES["SCB"],
        "requested_features": [
            "AsphaltContent_Design",
            "Va",
            "VMA",
            "Dust_Binder",
            "Pass_0.075mm",
            "Gsb",
            "Gmm",
            "Pass_4.75mm",
            "CAA",
            "SandEq",
            "RAP_pct",
            "ACinRAP",
            "PG Grade",
            "ADT_DOTD_ord",
        ],
        "units": "kJ/m2",
        "meaning": (
            "Higher SCB Jc generally indicates greater fracture energy and "
            "better cracking resistance."
        ),
    },
}

PAPER_METHOD_ROWS = [
    {
        "Method_Element": "Input-output structure",
        "Paper": "Material and mixture variables predict performance indicators.",
        "Current_Study": "Selected JMF variables predict Rut_20k and SCB Jc separately.",
    },
    {
        "Method_Element": "Scaling",
        "Paper": "Min-Max normalization",
        "Current_Study": "MinMaxScaler fitted inside each training/CV pipeline.",
    },
    {
        "Method_Element": "Data split",
        "Paper": "75% training and 25% testing",
        "Current_Study": "Stratified-regression 75/25 split where feasible.",
    },
    {
        "Method_Element": "Validation",
        "Paper": "Five-fold cross-validation",
        "Current_Study": "Five-fold shuffled CV on training data only.",
    },
    {
        "Method_Element": "Primary models",
        "Paper": "Boosting/ensemble tree models for nonlinear asphalt performance prediction",
        "Current_Study": "Regularized GradientBoostingRegressor and XGBRegressor",
    },
    {
        "Method_Element": "Evaluation",
        "Paper": "R2, RMSE, and MAE",
        "Current_Study": "Train, CV, and locked-test metrics plus overfit gap.",
    },
    {
        "Method_Element": "Interpretation",
        "Paper": "SHAP",
        "Current_Study": "SHAP for the selected tree model when package is installed.",
    },
]


# =============================================================================
# 3. OUTPUT DIRECTORIES
# =============================================================================

def make_output_directories():
    paths = {
        "root": output_folder,
        "figures": output_folder / "figures",
        "rut_figures": output_folder / "figures" / "Rut_20k",
        "scb_figures": output_folder / "figures" / "SCB_Jc",
        "shap": output_folder / "shap",
        "rut_shap": output_folder / "shap" / "Rut_20k",
        "scb_shap": output_folder / "shap" / "SCB_Jc",
        "tables": output_folder / "tables",
        "models": output_folder / "models",
    }

    if SAVE_OUTPUTS:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    return paths


# =============================================================================
# 4. DATA LOADING AND FLEXIBLE MATCHING
# =============================================================================

def normalize_column_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_column(columns, requested_name, aliases=None):
    aliases = aliases or []
    normalized_lookup = {}
    for column in columns:
        normalized_lookup.setdefault(normalize_column_name(column), column)

    for candidate in [requested_name] + list(aliases):
        if candidate in columns:
            return candidate
        normalized = normalize_column_name(candidate)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]
    return None


def detect_modeling_sheet(excel_file, manual_sheet=None):
    if manual_sheet and manual_sheet in excel_file.sheet_names:
        return manual_sheet

    preferred_names = [
        "Cleaned_Data_Kept",
        "Cleaned Data Kept",
        "Cleaned_Data",
        "Cleaned Data",
        "Modeling_Data",
        "Modeling Data",
    ]
    for name in preferred_names:
        if name in excel_file.sheet_names:
            return name

    best_sheet = excel_file.sheet_names[0]
    best_score = -1
    for sheet in excel_file.sheet_names:
        preview = pd.read_excel(input_file, sheet_name=sheet, nrows=20)
        target_score = sum(
            resolve_column(preview.columns, target, aliases) is not None
            for target, aliases in TARGET_ALIASES.items()
        )
        score = 1000 * target_score + len(preview.columns)
        if score > best_score:
            best_sheet = sheet
            best_score = score
    return best_sheet


def load_excel_dataset():
    if not input_file.exists():
        raise FileNotFoundError(f"Input file was not found:\n{input_file}")

    excel_file = pd.ExcelFile(input_file)
    print("Available sheets:")
    for sheet in excel_file.sheet_names:
        print(" -", sheet)

    # ---- FULL DATA: combine the Design + Validation sheets (both carry both targets) ----
    if COMBINE_FULL_DATA:
        sheets_present = [s for s in FULL_DATA_SHEETS if s in excel_file.sheet_names]
        if not sheets_present:
            sheets_present = [detect_modeling_sheet(excel_file, preferred_sheet)]
        frames = []
        for s in sheets_present:
            part = pd.read_excel(input_file, sheet_name=s)
            part["Data_Source_Sheet"] = s
            frames.append(part)
        data = pd.concat(frames, ignore_index=True, sort=False)
        selected_sheet = " + ".join(sheets_present)
        print("\nCombined full-data sheets:", selected_sheet,
              "->", {s: len(f) for s, f in zip(sheets_present, frames)})

        # Content-based de-duplication so identical mixes cannot leak across the train/test split.
        if DEDUP_FULL_DATA:
            key_col = next((c for c in ["MixDesignKey", "Mix_ID"] if c in data.columns), None)
            num_cols = [c for c in data.select_dtypes("number").columns
                        if c not in ("Rut_20k", "SCB")]
            sig = pd.Series(
                ["|".join(map(str, row)) for row in data[num_cols].round(4).fillna(-999999).values],
                index=data.index,
            )
            before = len(data)
            data = data.loc[~sig.duplicated(keep="first")].reset_index(drop=True)
            print(f"Content de-duplication: {before} rows -> {len(data)} unique mixes "
                  + (f"(key={key_col})" if key_col else "")
                  + f" | Rut_20k mixes={data['Rut_20k'].notna().sum()} "
                    f"SCB mixes={(pd.to_numeric(data['SCB'], errors='coerce') > 0).sum()}")
    else:
        selected_sheet = detect_modeling_sheet(excel_file, preferred_sheet)
        data = pd.read_excel(input_file, sheet_name=selected_sheet)

    print("\nSelected modeling sheet:", selected_sheet)
    print("Dataset shape:", data.shape)
    print("\nColumn names:")
    print(list(data.columns))
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print("\nData types:")
    print(data.dtypes.to_string())

    return data, selected_sheet, excel_file.sheet_names


def match_requested_features(data, requested_features):
    rows = []
    matched_actual = []
    canonical_to_actual = {}

    for canonical in requested_features:
        # Prefer the explicit numerical NMAS column when both a text label and
        # an engineering value in millimeters are present.
        if canonical == "NMAS" and "NMAS (mm)" in data.columns:
            actual = "NMAS (mm)"
        else:
            actual = resolve_column(
                data.columns,
                canonical,
                FEATURE_ALIASES.get(canonical, []),
            )
        status = "Matched" if actual is not None else "Missing"
        rows.append(
            {
                "Requested_Feature": canonical,
                "Matched_Column": actual if actual is not None else "NOT FOUND",
                "Status": status,
            }
        )
        if actual is not None:
            matched_actual.append(actual)
            canonical_to_actual[canonical] = actual

    # Preserve order while preventing aliases from selecting the same column twice.
    matched_actual = list(dict.fromkeys(matched_actual))
    missing = [
        row["Requested_Feature"]
        for row in rows
        if row["Status"] == "Missing"
    ]
    return matched_actual, missing, canonical_to_actual, pd.DataFrame(rows)


def make_regression_strata(y, n_bins=5):
    y_series = pd.Series(y).reset_index(drop=True)
    if y_series.nunique() < 2:
        return None
    try:
        strata = pd.qcut(
            y_series,
            q=min(n_bins, y_series.nunique()),
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return None
    counts = pd.Series(strata).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return np.asarray(strata)


# =============================================================================
# 5. PREPROCESSING
# =============================================================================

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_feature_types(data, feature_columns):
    categorical = []
    numerical = []
    for column in feature_columns:
        if (
            pd.api.types.is_object_dtype(data[column])
            or pd.api.types.is_string_dtype(data[column])
            or pd.api.types.is_categorical_dtype(data[column])
        ):
            categorical.append(column)
        else:
            numerical.append(column)
    return numerical, categorical


def build_preprocessor(numerical_features, categorical_features):
    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ]
        )
        transformers.append(("numeric", numerical_pipeline, numerical_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_one_hot_encoder()),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_processed_feature_names(fitted_pipeline):
    try:
        names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
        return [
            str(name)
            .replace("numeric__", "")
            .replace("categorical__", "")
            for name in names
        ]
    except Exception:
        model = fitted_pipeline.named_steps["model"]
        count = getattr(model, "n_features_in_", 0)
        return [f"Feature_{index + 1}" for index in range(count)]


# =============================================================================
# 6. MODELS AND SEARCH SPACES
# =============================================================================

def define_models():
    """Define active publication models.

    The previous run still trained removed low-value candidates, which kept the output
    in the same range and selected a higher-gap model.  The active set below
    removes those two candidates and focuses on regularized boosting models that
    use shrinkage, shallow trees, subsampling, and leaf-size controls.
    """
    model_specs = {
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                loss="huber",
            ),
            "params": {
                "model__n_estimators": [150, 250, 400, 650, 900],
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [5, 8, 12, 20, 30],
                "model__min_samples_split": [10, 20, 40, 60],
                "model__subsample": [0.65, 0.80, 0.90, 1.00],
                "model__max_features": ["sqrt", 0.60, 0.75, None],
                "model__alpha": [0.80, 0.85, 0.90, 0.95],
            },
            "n_iter": N_ITER_GBR,
            "primary": True,
        }
    }

    if HAS_XGBOOST:
        model_specs["XGBoost"] = {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                eval_metric="rmse",
                random_state=RANDOM_STATE,
                n_jobs=MODEL_N_JOBS,
            ),
            "params": {
                "model__n_estimators": [150, 250, 400, 650, 900],
                "model__max_depth": [1, 2, 3, 4],
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
                "model__subsample": [0.65, 0.80, 0.90, 1.00],
                "model__colsample_bytree": [0.65, 0.80, 0.90, 1.00],
                "model__min_child_weight": [3, 5, 8, 12, 20],
                "model__reg_alpha": [0, 0.01, 0.05, 0.10, 0.50, 1.00],
                "model__reg_lambda": [1, 2, 5, 10, 20, 40],
                "model__gamma": [0, 0.03, 0.05, 0.10, 0.20],
            },
            "n_iter": N_ITER_XGB,
            "primary": True,
        }
    else:
        print("XGBoost is not installed. Install it with: conda install -c conda-forge xgboost")
        print("XGBoost will be skipped without stopping the workflow.")

    if INCLUDE_OPTIONAL_MODELS:
        model_specs.update(
            {
                "ExtraTrees": {
                    "estimator": ExtraTreesRegressor(
                        random_state=RANDOM_STATE,
                        n_jobs=MODEL_N_JOBS,
                    ),
                    "params": {
                        "model__n_estimators": [200, 500, 800],
                        "model__max_depth": [5, 8, 12, None],
                        "model__min_samples_leaf": [1, 2, 5, 8],
                        "model__max_features": ["sqrt", 0.5, 0.7, 1.0],
                    },
                    "n_iter": N_ITER_OPTIONAL,
                    "primary": False,
                },
                "SVR_RBF": {
                    "estimator": SVR(kernel="rbf"),
                    "params": {
                        "model__C": [0.1, 0.5, 1, 5, 10, 25],
                        "model__epsilon": [0.005, 0.01, 0.05, 0.10, 0.20],
                        "model__gamma": ["scale", "auto", 0.01, 0.05, 0.10],
                    },
                    "n_iter": N_ITER_OPTIONAL,
                    "primary": False,
                },
                "Ridge": {
                    "estimator": Ridge(),
                    "params": {"model__alpha": np.logspace(-4, 3, 20)},
                    "n_iter": N_ITER_OPTIONAL,
                    "primary": False,
                },
                "Lasso": {
                    "estimator": Lasso(max_iter=50000, random_state=RANDOM_STATE),
                    "params": {"model__alpha": np.logspace(-5, 0, 20)},
                    "n_iter": N_ITER_OPTIONAL,
                    "primary": False,
                },
            }
        )

    return model_specs


# =============================================================================
# 7. METRICS AND BEST-FIT LINE
# =============================================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = errors**2

    valid_relative = np.abs(y_true) > RELATIVE_ERROR_EPSILON
    relative_errors = np.full(len(y_true), np.nan)
    relative_errors[valid_relative] = (
        absolute_errors[valid_relative] / np.abs(y_true[valid_relative]) * 100.0
    )

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "Mean_Relative_Error_pct": float(np.nanmean(relative_errors)),
        "Median_Relative_Error_pct": float(np.nanmedian(relative_errors)),
        "Min_Relative_Error_pct": float(np.nanmin(relative_errors)),
        "Max_Relative_Error_pct": float(np.nanmax(relative_errors)),
        "Residual_Mean": float(np.mean(errors)),
        "Residual_SD": float(np.std(errors, ddof=1)),
    }


def calculate_best_fit_line(measured, predicted):
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    slope, intercept = np.polyfit(measured, predicted, 1)
    fitted = slope * measured + intercept
    line_r2 = r2_score(predicted, fitted)
    return {
        "Slope": float(slope),
        "Intercept": float(intercept),
        "Line_R2": float(line_r2),
        "Equation": f"Predicted = {slope:.4f} x Measured + {intercept:.4f}",
    }


def relative_error_frame(y_true, y_pred, target, model_name, dataset):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.abs(y_true) > RELATIVE_ERROR_EPSILON
    relative = np.full(len(y_true), np.nan)
    relative[valid] = np.abs(y_true[valid] - y_pred[valid]) / np.abs(
        y_true[valid]
    ) * 100.0
    return pd.DataFrame(
        {
            "Target": target,
            "Model": model_name,
            "Dataset": dataset,
            "Measured": y_true,
            "Predicted": y_pred,
            "Residual_PredMinusMeasured": y_pred - y_true,
            "Relative_Error_pct": relative,
        }
    )


# =============================================================================
# 8. FIGURE HELPERS
# =============================================================================

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def finish_figure(fig, path=None):
    fig.tight_layout()
    if SAVE_OUTPUTS and path is not None:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if SHOW_PLOTS_IN_SPYDER:
        plt.show()
    else:
        plt.close(fig)


def parity_plot(
    measured,
    predicted,
    target,
    model_name,
    dataset_label,
    units,
    figure_path,
    figure_number,
):
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    metrics = calculate_metrics(measured, predicted)
    line = calculate_best_fit_line(measured, predicted)

    low = float(min(measured.min(), predicted.min()))
    high = float(max(measured.max(), predicted.max()))
    padding = 0.05 * (high - low) if high > low else 1.0
    x_line = np.array([low - padding, high + padding])

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(
        measured,
        predicted,
        alpha=0.72,
        edgecolor="black",
        linewidth=0.35,
        color="#2878B5",
    )
    ax.plot(x_line, x_line, "r--", lw=2, label="1:1 reference line")
    ax.plot(
        x_line,
        line["Slope"] * x_line + line["Intercept"],
        color="#173F5F",
        lw=2,
        label="Best-fit line",
    )
    ax.set_xlabel(f"Measured {target} ({units})")
    ax.set_ylabel(f"Predicted {target} ({units})")
    ax.set_title(
        f"Figure {figure_number}. {target} - {model_name}\n{dataset_label}"
    )
    ax.text(
        0.04,
        0.96,
        f"{line['Equation']}\n"
        f"Best-fit line R2 = {line['Line_R2']:.3f}\n"
        f"Model R2 = {metrics['R2']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"MAE = {metrics['MAE']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.28)
    finish_figure(fig, figure_path)
    return line


def combined_parity_plot(
    y_train,
    pred_train,
    y_test,
    pred_test,
    target,
    model_name,
    units,
    figure_path,
    figure_number,
):
    all_values = np.concatenate(
        [
            np.asarray(y_train, dtype=float),
            np.asarray(pred_train, dtype=float),
            np.asarray(y_test, dtype=float),
            np.asarray(pred_test, dtype=float),
        ]
    )
    low, high = float(all_values.min()), float(all_values.max())
    padding = 0.05 * (high - low) if high > low else 1.0
    line = np.array([low - padding, high + padding])

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(
        y_train,
        pred_train,
        alpha=0.55,
        label="Training",
        color="#4C78A8",
    )
    ax.scatter(
        y_test,
        pred_test,
        alpha=0.80,
        label="Testing",
        color="#F58518",
        edgecolor="black",
        linewidth=0.35,
    )
    ax.plot(line, line, "k--", lw=1.8, label="1:1 reference line")
    ax.set_xlabel(f"Measured {target} ({units})")
    ax.set_ylabel(f"Predicted {target} ({units})")
    ax.set_title(
        f"Figure {figure_number}. {target} - {model_name}\n"
        "Training and independent testing predictions"
    )
    ax.legend()
    ax.grid(alpha=0.28)
    finish_figure(fig, figure_path)


def residual_plot(
    y_true,
    y_pred,
    target,
    model_name,
    dataset_label,
    units,
    figure_path,
    figure_number,
):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(y_pred, residuals, alpha=0.75, color="#2A9D8F")
    ax.axhline(0, color="black", linestyle="--", lw=1.5)
    ax.set_xlabel(f"Predicted {target} ({units})")
    ax.set_ylabel(f"Residual: predicted - measured ({units})")
    ax.set_title(
        f"Figure {figure_number}. {target} - {model_name}\n"
        f"{dataset_label} residuals"
    )
    ax.grid(alpha=0.28)
    finish_figure(fig, figure_path)


def residual_distribution_plot(
    y_true,
    y_pred,
    target,
    model_name,
    dataset_label,
    units,
    figure_path,
    figure_number,
):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.histplot(residuals, bins=20, kde=True, ax=ax, color="#8E6C8A")
    ax.axvline(0, color="black", linestyle="--", lw=1.5)
    ax.set_xlabel(f"Residual ({units})")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Figure {figure_number}. {target} - {model_name}\n"
        f"{dataset_label} residual distribution"
    )
    finish_figure(fig, figure_path)


def relative_error_boxplot(
    relative_df,
    target,
    model_name,
    figure_path,
    figure_number,
):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.boxplot(
        data=relative_df,
        x="Dataset",
        y="Relative_Error_pct",
        ax=ax,
        color="#E9C46A",
    )
    ax.set_ylabel("Absolute relative error (%)")
    ax.set_title(
        f"Figure {figure_number}. {target} - {model_name}\n"
        "Relative prediction error"
    )
    finish_figure(fig, figure_path)


def model_comparison_plots(performance_df, target, target_figure_dir):
    target_df = performance_df[performance_df["Target"] == target].copy()
    if target_df.empty:
        return

    long_r2 = target_df.melt(
        id_vars=["Model"],
        value_vars=["Training_R2", "CV_Mean_R2", "Testing_R2"],
        var_name="Evaluation",
        value_name="R2",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=long_r2, x="Model", y="R2", hue="Evaluation", ax=ax)
    ax.set_title(f"{target}: training, CV, and independent-test R2")
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(0, color="black", lw=0.8)
    finish_figure(fig, target_figure_dir / f"{target}_R2_comparison.png")

    error_long = target_df.melt(
        id_vars=["Model"],
        value_vars=["Testing_RMSE", "Testing_MAE"],
        var_name="Metric",
        value_name="Error",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=error_long, x="Model", y="Error", hue="Metric", ax=ax)
    ax.set_title(f"{target}: independent-test error comparison")
    ax.tick_params(axis="x", rotation=30)
    finish_figure(fig, target_figure_dir / f"{target}_error_comparison.png")


def measured_prediction_curve(
    predictions_df,
    target,
    target_figure_dir,
):
    target_df = predictions_df[
        (predictions_df["Target"] == target)
        & (predictions_df["Dataset"] == "Testing")
    ].copy()
    if target_df.empty:
        return

    models = list(target_df["Model"].unique())
    measured = (
        target_df[target_df["Model"] == models[0]]
        .sort_values("Measured")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        np.arange(len(measured)),
        measured["Measured"],
        color="black",
        lw=2.2,
        label="Measured",
    )
    for model_name in models:
        model_df = (
            target_df[target_df["Model"] == model_name]
            .sort_values("Measured")
            .reset_index(drop=True)
        )
        ax.plot(
            np.arange(len(model_df)),
            model_df["Predicted"],
            marker="o",
            markersize=3,
            lw=1.2,
            label=model_name,
        )
    ax.set_xlabel("Testing mixtures ordered by measured response")
    ax.set_ylabel(target)
    ax.set_title(f"{target}: measured and predicted testing values")
    ax.legend(fontsize=9)
    finish_figure(fig, target_figure_dir / f"{target}_test_prediction_curve.png")


# =============================================================================
# 9. MODEL TRAINING AND EVALUATION
# =============================================================================

def tune_and_evaluate_model(
    model_name,
    model_spec,
    X_train,
    X_test,
    y_train,
    y_test,
    numerical_features,
    categorical_features,
    target,
    units,
    target_figure_dir,
):
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", clone(model_spec["estimator"])),
        ]
    )

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=model_spec["params"],
        n_iter=model_spec["n_iter"],
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS,
        verbose=1,
        return_train_score=True,
        error_score=np.nan,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    train_pred = best_pipeline.predict(X_train)
    test_pred = best_pipeline.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    test_line = calculate_best_fit_line(y_test, test_pred)

    best_index = search.best_index_
    cv_mean = float(search.cv_results_["mean_test_score"][best_index])
    cv_sd = float(search.cv_results_["std_test_score"][best_index])
    tuning_train_r2 = float(search.cv_results_["mean_train_score"][best_index])

    relative_train = relative_error_frame(
        y_train,
        train_pred,
        target,
        model_name,
        "Training",
    )
    relative_test = relative_error_frame(
        y_test,
        test_pred,
        target,
        model_name,
        "Testing",
    )
    relative_all = pd.concat([relative_train, relative_test], ignore_index=True)

    prefix = f"{safe_name(target)}_{safe_name(model_name)}"
    figure_counter = 1
    train_line = parity_plot(
        y_train,
        train_pred,
        target,
        model_name,
        "Training data",
        units,
        target_figure_dir / f"{prefix}_01_training_parity.png",
        figure_counter,
    )
    figure_counter += 1
    parity_plot(
        y_test,
        test_pred,
        target,
        model_name,
        "Independent testing data",
        units,
        target_figure_dir / f"{prefix}_02_testing_parity.png",
        figure_counter,
    )
    figure_counter += 1
    combined_parity_plot(
        y_train,
        train_pred,
        y_test,
        test_pred,
        target,
        model_name,
        units,
        target_figure_dir / f"{prefix}_03_combined_parity.png",
        figure_counter,
    )
    figure_counter += 1
    residual_plot(
        y_test,
        test_pred,
        target,
        model_name,
        "Independent testing data",
        units,
        target_figure_dir / f"{prefix}_04_test_residuals.png",
        figure_counter,
    )
    figure_counter += 1
    residual_distribution_plot(
        y_test,
        test_pred,
        target,
        model_name,
        "Independent testing data",
        units,
        target_figure_dir / f"{prefix}_05_test_residual_distribution.png",
        figure_counter,
    )
    figure_counter += 1
    relative_error_boxplot(
        relative_all,
        target,
        model_name,
        target_figure_dir / f"{prefix}_06_relative_error.png",
        figure_counter,
    )

    performance_row = {
        "Target": target,
        "Model": model_name,
        "Primary_Paper_Model": bool(model_spec["primary"]),
        "N_Training": int(len(X_train)),
        "N_Testing": int(len(X_test)),
        "N_Features_Raw": int(X_train.shape[1]),
        "Training_R2": train_metrics["R2"],
        "CV_Mean_R2": cv_mean,
        "CV_SD_R2": cv_sd,
        "Tuning_Mean_Train_R2": tuning_train_r2,
        "Testing_R2": test_metrics["R2"],
        "Training_RMSE": train_metrics["RMSE"],
        "Testing_RMSE": test_metrics["RMSE"],
        "Training_MAE": train_metrics["MAE"],
        "Testing_MAE": test_metrics["MAE"],
        "Training_MSE": train_metrics["MSE"],
        "Testing_MSE": test_metrics["MSE"],
        "Overfit_Gap_TrainMinusTest": (
            train_metrics["R2"] - test_metrics["R2"]
        ),
        "Overfit_Gap_TrainMinusCV": train_metrics["R2"] - cv_mean,
        "Mean_Relative_Error_pct": test_metrics[
            "Mean_Relative_Error_pct"
        ],
        "Median_Relative_Error_pct": test_metrics[
            "Median_Relative_Error_pct"
        ],
        "Min_Relative_Error_pct": test_metrics["Min_Relative_Error_pct"],
        "Max_Relative_Error_pct": test_metrics["Max_Relative_Error_pct"],
        "BestFit_Slope_Test": test_line["Slope"],
        "BestFit_Intercept_Test": test_line["Intercept"],
        "BestFit_Line_R2_Test": test_line["Line_R2"],
        "BestFit_Equation_Test": test_line["Equation"],
    }

    best_parameter_row = {
        "Target": target,
        "Model": model_name,
        "Best_CV_R2": cv_mean,
        "Best_Params_JSON": json.dumps(search.best_params_, default=str),
    }

    equation_rows = [
        {
            "Target": target,
            "Model": model_name,
            "Dataset": "Training",
            **train_line,
            "Model_R2": train_metrics["R2"],
            "RMSE": train_metrics["RMSE"],
            "MAE": train_metrics["MAE"],
        },
        {
            "Target": target,
            "Model": model_name,
            "Dataset": "Testing",
            **test_line,
            "Model_R2": test_metrics["R2"],
            "RMSE": test_metrics["RMSE"],
            "MAE": test_metrics["MAE"],
        },
    ]

    return {
        "pipeline": best_pipeline,
        "performance": performance_row,
        "best_parameters": best_parameter_row,
        "equations": equation_rows,
        "relative_errors": relative_all,
        "train_predictions": relative_train,
        "test_predictions": relative_test,
        "search_results": pd.DataFrame(search.cv_results_),
    }


# =============================================================================
# 10. FEATURE IMPORTANCE AND SHAP
# =============================================================================

def extract_feature_importance(
    fitted_pipeline,
    X_test,
    y_test,
    target,
    model_name,
    target_figure_dir,
):
    processed_names = get_processed_feature_names(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "Native tree importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
        method = "Absolute coefficient"
    else:
        permutation = permutation_importance(
            fitted_pipeline,
            X_test,
            y_test,
            scoring="r2",
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=SEARCH_N_JOBS,
        )
        # Pipeline-level permutation importance corresponds to raw columns.
        processed_names = list(X_test.columns)
        values = permutation.importances_mean
        method = "Test permutation importance"

    if len(processed_names) != len(values):
        processed_names = [f"Feature_{i + 1}" for i in range(len(values))]

    importance_df = pd.DataFrame(
        {
            "Target": target,
            "Model": model_name,
            "Feature": processed_names,
            "Importance": values,
            "Method": method,
        }
    ).sort_values("Importance", ascending=False)
    importance_df["Rank"] = np.arange(1, len(importance_df) + 1)

    plot_df = importance_df.head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["Feature"], plot_df["Importance"], color="#2A9D8F")
    ax.set_xlabel("Importance")
    ax.set_title(f"{target} - {model_name}: top feature importance")
    finish_figure(
        fig,
        target_figure_dir
        / f"{safe_name(target)}_{safe_name(model_name)}_importance.png",
    )
    return importance_df


def run_shap_analysis(
    fitted_pipeline,
    X_train,
    target,
    model_name,
    target_shap_dir,
):
    if not RUN_SHAP:
        return pd.DataFrame()
    if not HAS_SHAP:
        print(
            "SHAP skipped because the package is not installed.\n"
            "Install with: conda install -c conda-forge shap"
        )
        return pd.DataFrame(
            [
                {
                    "Target": target,
                    "Model": model_name,
                    "Feature": "SHAP not available",
                    "Mean_Absolute_SHAP": np.nan,
                    "Rank": np.nan,
                }
            ]
        )

    model = fitted_pipeline.named_steps["model"]
    tree_types = (
            ExtraTreesRegressor,
        GradientBoostingRegressor,
    )
    is_tree = isinstance(model, tree_types)
    if HAS_XGBOOST:
        is_tree = is_tree or isinstance(model, XGBRegressor)
    if not is_tree:
        print(f"SHAP skipped for non-tree selected model: {model_name}")
        return pd.DataFrame()

    sample_n = min(SHAP_MAX_SAMPLES, len(X_train))
    X_sample = X_train.sample(sample_n, random_state=RANDOM_STATE)
    transformed = fitted_pipeline.named_steps["preprocess"].transform(X_sample)
    transformed = np.asarray(transformed)
    feature_names = get_processed_feature_names(fitted_pipeline)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    mean_absolute = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "Target": target,
            "Model": model_name,
            "Feature": feature_names,
            "Mean_Absolute_SHAP": mean_absolute,
        }
    ).sort_values("Mean_Absolute_SHAP", ascending=False)
    shap_df["Rank"] = np.arange(1, len(shap_df) + 1)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    fig = plt.gcf()
    fig.suptitle(f"{target} - {model_name}: SHAP direction", y=1.02)
    finish_figure(
        fig,
        target_shap_dir / f"{safe_name(target)}_SHAP_beeswarm.png",
    )

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    fig = plt.gcf()
    fig.suptitle(f"{target} - {model_name}: mean absolute SHAP", y=1.02)
    finish_figure(
        fig,
        target_shap_dir / f"{safe_name(target)}_SHAP_bar.png",
    )

    for feature in shap_df.head(3)["Feature"]:
        feature_index = feature_names.index(feature)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature_index,
            shap_values,
            transformed,
            feature_names=feature_names,
            show=False,
            interaction_index=None,
        )
        fig = plt.gcf()
        fig.suptitle(f"{target}: SHAP dependence - {feature}", y=1.02)
        finish_figure(
            fig,
            target_shap_dir
            / f"{safe_name(target)}_SHAP_dependence_{safe_name(feature)}.png",
        )

    return shap_df


# =============================================================================
# 11. FEATURE-GROUP ABLATION
# =============================================================================

def build_feature_groups(target, full_features, canonical_to_actual):
    def actual(canonical_names):
        return [
            canonical_to_actual[name]
            for name in canonical_names
            if name in canonical_to_actual
        ]

    if target == "Rut_20k":
        canonical_groups = {
            "Full selected set": list(TARGET_CONFIGS[target]["requested_features"]),
            "Traffic only": ["ADT_DOTD_ord"],
            "Binder only": ["PG Grade"],
            "RAP only": ["RAP_pct", "ACinRAP"],
            "Gradation only": ["Pass_4.75mm"],
            "Volumetric only": ["Va", "VMA"],
            "Dust Binder only": ["Dust_Binder"],
            "Density only": ["Gmm"],
            "Aggregate quality only": ["SandEq", "FAA"],
            "NMAS Absorption only": ["NMAS", "Absorption"],
            "Gradation plus volumetric": ["Pass_4.75mm", "Va", "VMA"],
            "Binder plus RAP": ["PG Grade", "RAP_pct", "ACinRAP"],
            "Full without RAP": [
                name
                for name in TARGET_CONFIGS[target]["requested_features"]
                if name not in {"RAP_pct", "ACinRAP"}
            ],
            "Full without gradation": [
                name
                for name in TARGET_CONFIGS[target]["requested_features"]
                if name != "Pass_4.75mm"
            ],
            "Full without volumetrics": [
                name
                for name in TARGET_CONFIGS[target]["requested_features"]
                if name not in {"Va", "VMA"}
            ],
        }
    else:
        canonical_groups = {
            "Full selected set": list(TARGET_CONFIGS[target]["requested_features"]),
            "Binder content only": ["AsphaltContent_Design"],
            "Volumetric only": ["Va", "VMA"],
            "Dust fines only": ["Dust_Binder", "Pass_0.075mm"],
            "Density gradation only": ["Gsb", "Gmm", "Pass_4.75mm"],
            "Aggregate quality only": ["CAA", "SandEq"],
            "RAP only": ["RAP_pct", "ACinRAP"],
            "PG Grade only": ["PG Grade"],
            "Traffic only": ["ADT_DOTD_ord"],
            "Gradation plus volumetric": [
                "Pass_0.075mm",
                "Pass_4.75mm",
                "Va",
                "VMA",
            ],
            "Binder plus RAP": [
                "AsphaltContent_Design",
                "RAP_pct",
                "ACinRAP",
                "PG Grade",
            ],
            "Full without RAP": [
                name
                for name in TARGET_CONFIGS[target]["requested_features"]
                if name not in {"RAP_pct", "ACinRAP"}
            ],
            "Full without gradation": [
                name
                for name in TARGET_CONFIGS[target]["requested_features"]
                if name not in {"Pass_0.075mm", "Pass_4.75mm"}
            ],
            "Full without volumetrics": [
                name
                for name in TARGET_CONFIGS[target]["requested_features"]
                if name not in {"Va", "VMA"}
            ],
        }

    groups = {}
    for group_name, canonical_names in canonical_groups.items():
        columns = list(dict.fromkeys(actual(canonical_names)))
        if columns:
            groups[group_name] = columns
    groups["Full selected set"] = full_features
    return groups


def run_feature_group_ablation(
    target,
    feature_groups,
    model_results,
    X_train_full,
    X_test_full,
    y_train,
    y_test,
):
    if not RUN_ABLATION_STUDY:
        return pd.DataFrame()

    rows = []
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for group_name, features in feature_groups.items():
        X_train = X_train_full[features].copy()
        X_test = X_test_full[features].copy()
        numerical, categorical = infer_feature_types(X_train, features)

        for model_name in ["GradientBoosting", "XGBoost"]:
            if model_name not in model_results:
                continue

            tuned_model = clone(
                model_results[model_name]["pipeline"].named_steps["model"]
            )
            pipeline = Pipeline(
                [
                    ("preprocess", build_preprocessor(numerical, categorical)),
                    ("model", tuned_model),
                ]
            )

            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "r2": "r2",
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                },
                return_train_score=True,
                n_jobs=SEARCH_N_JOBS,
                error_score=np.nan,
            )
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)
            train_metrics = calculate_metrics(y_train, train_pred)
            test_metrics = calculate_metrics(y_test, test_pred)

            rows.append(
                {
                    "Target": target,
                    "Feature_Group": group_name,
                    "Features_Used": ", ".join(features),
                    "Number_of_Features": len(features),
                    "Model": model_name,
                    "Training_R2": train_metrics["R2"],
                    "CV_Mean_R2": float(np.nanmean(cv_scores["test_r2"])),
                    "CV_SD_R2": float(np.nanstd(cv_scores["test_r2"], ddof=1)),
                    "Testing_R2": test_metrics["R2"],
                    "Testing_RMSE": test_metrics["RMSE"],
                    "Testing_MAE": test_metrics["MAE"],
                    "Overfit_Gap_TrainMinusCV": (
                        train_metrics["R2"]
                        - float(np.nanmean(cv_scores["test_r2"]))
                    ),
                    "Selection_Rule": (
                        "Rank by training CV first; test metrics are confirmatory."
                    ),
                }
            )

    ablation_df = pd.DataFrame(rows)
    if not ablation_df.empty:
        ablation_df["CV_Rank"] = ablation_df.groupby("Target")[
            "CV_Mean_R2"
        ].rank(ascending=False, method="min")
    return ablation_df


# =============================================================================
# 12. TARGET WORKFLOW
# =============================================================================

def run_target_workflow(data, target, paths):
    config = TARGET_CONFIGS[target]
    target_column = resolve_column(
        data.columns,
        target,
        config["target_aliases"],
    )
    if target_column is None:
        raise KeyError(
            f"Could not locate the target {target}. "
            f"Tried aliases: {config['target_aliases']}"
        )

    matched, missing, canonical_to_actual, matching_df = (
        match_requested_features(data, config["requested_features"])
    )
    matched = [
        column
        for column in matched
        if not data[column].isna().all()
    ]

    print("\n" + "#" * 96)
    print("TARGET:", target)
    print("#" * 96)
    print("Target column:", target_column)
    print("1. Requested features:")
    print(config["requested_features"])
    print("2. Matched features:")
    print(matched)
    print("3. Missing features:")
    print(missing if missing else "None")
    print("4. Final features used:")
    print(matched)

    y_numeric = pd.to_numeric(data[target_column], errors="coerce")
    eligible = y_numeric.notna()
    X = data.loc[eligible, matched].copy()
    y = y_numeric.loc[eligible].astype(float).reset_index(drop=True)
    X = X.reset_index(drop=True)

    numerical, categorical = infer_feature_types(X, matched)
    for column in numerical:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    strata = make_regression_strata(y, n_bins=5)
    # ---- Leakage-safe split: group by physical mix so no mix (or its design/validation
    #      replicates) appears in both train and test. Falls back to a plain stratified split
    #      when no mix key is available. ----
    group_key = None
    if "MixDesignKey" in data.columns:
        group_key = data.loc[eligible, "MixDesignKey"].astype(str).reset_index(drop=True)
    if group_key is not None and group_key.nunique() < len(group_key):
        from sklearn.model_selection import StratifiedGroupKFold
        n_splits = max(2, int(round(1.0 / TEST_SIZE)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(iter(sgkf.split(X, strata, group_key)))
        X_train, X_test = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)
        shared = len(set(group_key.iloc[tr_idx]) & set(group_key.iloc[te_idx]))
        print(f"Grouped split by MixDesignKey: train={len(tr_idx)} test={len(te_idx)} "
              f"| unique mixes={group_key.nunique()} | mixes shared across splits={shared} (must be 0)")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=strata,
        )

    # Explicit supervised-learning split tables:
    # The saved training and locked-test tables contain X plus y for traceability.
    # The model still receives X and y separately during fitting, and only X_test
    # during prediction. This prevents target leakage while documenting that the
    # target is used as the supervised-learning label.
    selected_features = list(matched)
    train_data = X_train.copy().reset_index(drop=True)
    train_data[target] = y_train.reset_index(drop=True)
    locked_test_data = X_test.copy().reset_index(drop=True)
    locked_test_data[target] = y_test.reset_index(drop=True)

    if target == "SCB":
        selected_scb_features = selected_features
        X_train = train_data[selected_scb_features].copy()
        y_train = train_data["SCB"].astype(float).reset_index(drop=True)
        X_test = locked_test_data[selected_scb_features].copy()
        y_test = locked_test_data["SCB"].astype(float).reset_index(drop=True)
        print("SCB supervised-learning structure:")
        print("X_train = train_data[selected_scb_features]")
        print('y_train = train_data["SCB"]')
        print("model.fit(X_train, y_train)")
        print("model.predict(X_test) receives only X_test.")
    elif target == "Rut_20k":
        selected_rut_features = selected_features
        X_train = train_data[selected_rut_features].copy()
        y_train = train_data["Rut_20k"].astype(float).reset_index(drop=True)
        X_test = locked_test_data[selected_rut_features].copy()
        y_test = locked_test_data["Rut_20k"].astype(float).reset_index(drop=True)
        print("Rut_20k supervised-learning structure:")
        print("X_train = train_data[selected_rut_features]")
        print('y_train = train_data["Rut_20k"]')
        print("model.fit(X_train, y_train)")
        print("model.predict(X_test) receives only X_test.")

    if SAVE_OUTPUTS:
        split_dir = output_folder / "supervised_learning_splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(
            split_dir / f"{safe_name(target)}_train_X_plus_y.csv",
            index=False,
        )
        locked_test_data.to_csv(
            split_dir / f"{safe_name(target)}_locked_test_X_plus_y.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "Target": target,
                    "Step": "Training",
                    "Code": "model.fit(X_train, y_train)",
                    "Explanation": "The model receives X_train predictors and y_train target labels during supervised learning.",
                },
                {
                    "Target": target,
                    "Step": "Validation",
                    "Code": "Five-fold CV on the training data",
                    "Explanation": "Each CV fold contains X_validation and y_validation internally, but predictions are made from X_validation only.",
                },
                {
                    "Target": target,
                    "Step": "Locked testing",
                    "Code": "model.predict(X_test)",
                    "Explanation": "The locked-test file contains X_test and y_test for evaluation, but the model receives only X_test when predicting.",
                },
            ]
        ).to_csv(
            split_dir / f"{safe_name(target)}_supervised_learning_structure.csv",
            index=False,
        )

    print("Total samples:", len(X))
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Raw features:", len(matched))
    print("Numerical features:", numerical)
    print("Categorical features:", categorical)
    print("Engineering meaning:", config["meaning"])

    target_figure_dir = (
        paths["rut_figures"] if target == "Rut_20k" else paths["scb_figures"]
    )
    target_shap_dir = (
        paths["rut_shap"] if target == "Rut_20k" else paths["scb_shap"]
    )

    model_specs = define_models()
    model_results = {}
    performance_rows = []
    parameter_rows = []
    equation_rows = []
    relative_frames = []
    prediction_frames = []
    search_frames = []

    for model_name, model_spec in model_specs.items():
        print("\n" + "-" * 96)
        print(f"{target} - tuning and evaluating {model_name}")
        print("-" * 96)
        result = tune_and_evaluate_model(
            model_name=model_name,
            model_spec=model_spec,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            numerical_features=numerical,
            categorical_features=categorical,
            target=target,
            units=config["units"],
            target_figure_dir=target_figure_dir,
        )
        model_results[model_name] = result
        performance_rows.append(result["performance"])
        parameter_rows.append(result["best_parameters"])
        equation_rows.extend(result["equations"])
        relative_frames.append(result["relative_errors"])
        prediction_frames.extend(
            [result["train_predictions"], result["test_predictions"]]
        )
        search_df = result["search_results"].copy()
        search_df.insert(0, "Model", model_name)
        search_df.insert(0, "Target", target)
        search_frames.append(search_df)

        if SAVE_OUTPUTS:
            model_path = paths["models"] / (
                f"{safe_name(target)}_{safe_name(model_name)}_tuned.pkl"
            )
            joblib.dump(result["pipeline"], model_path)

    performance_df = pd.DataFrame(performance_rows)
    # Selection is CV-first. Testing results are not the selection criterion.
    performance_df = performance_df.sort_values(
        ["CV_Mean_R2", "CV_SD_R2", "Overfit_Gap_TrainMinusCV"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    performance_df["CV_Based_Rank"] = np.arange(1, len(performance_df) + 1)

    selected_model_name = performance_df.iloc[0]["Model"]
    selected_pipeline = model_results[selected_model_name]["pipeline"]
    print("\nCV-selected model:", selected_model_name)
    print(
        performance_df[
            [
                "Model",
                "Training_R2",
                "CV_Mean_R2",
                "CV_SD_R2",
                "Testing_R2",
                "Testing_RMSE",
                "Testing_MAE",
                "Overfit_Gap_TrainMinusCV",
            ]
        ].to_string(index=False)
    )

    importance_df = extract_feature_importance(
        fitted_pipeline=selected_pipeline,
        X_test=X_test,
        y_test=y_test,
        target=target,
        model_name=selected_model_name,
        target_figure_dir=target_figure_dir,
    )
    shap_df = run_shap_analysis(
        fitted_pipeline=selected_pipeline,
        X_train=X_train,
        target=target,
        model_name=selected_model_name,
        target_shap_dir=target_shap_dir,
    )

    feature_groups = build_feature_groups(
        target,
        matched,
        canonical_to_actual,
    )
    ablation_df = run_feature_group_ablation(
        target=target,
        feature_groups=feature_groups,
        model_results=model_results,
        X_train_full=X_train,
        X_test_full=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    relative_df = pd.concat(relative_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    model_comparison_plots(performance_df, target, target_figure_dir)
    measured_prediction_curve(predictions_df, target, target_figure_dir)

    if not relative_df.empty:
        test_relative = relative_df[relative_df["Dataset"] == "Testing"]
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(
            data=test_relative,
            x="Model",
            y="Relative_Error_pct",
            ax=ax,
            color="#E9C46A",
        )
        ax.set_title(f"{target}: testing relative error across models")
        ax.tick_params(axis="x", rotation=30)
        finish_figure(
            fig,
            target_figure_dir / f"{safe_name(target)}_relative_error_models.png",
        )

    selected_row = performance_df.iloc[0].to_dict()
    top_importance = importance_df.head(5)["Feature"].tolist()
    if (
        not shap_df.empty
        and "Mean_Absolute_SHAP" in shap_df
        and shap_df["Mean_Absolute_SHAP"].notna().any()
    ):
        top_shap = shap_df.dropna(subset=["Mean_Absolute_SHAP"]).head(5)[
            "Feature"
        ].tolist()
    else:
        top_shap = ["SHAP unavailable"]

    if not ablation_df.empty:
        best_group_row = ablation_df.sort_values(
            ["CV_Mean_R2", "CV_SD_R2"],
            ascending=[False, True],
        ).iloc[0]
        best_group = best_group_row["Feature_Group"]
    else:
        best_group = "Not run"

    final_summary = {
        "Target": target,
        "Selected_Model_CV_Based": selected_model_name,
        "Training_R2": selected_row["Training_R2"],
        "CV_Mean_R2": selected_row["CV_Mean_R2"],
        "CV_SD_R2": selected_row["CV_SD_R2"],
        "Testing_R2": selected_row["Testing_R2"],
        "Testing_RMSE": selected_row["Testing_RMSE"],
        "Testing_MAE": selected_row["Testing_MAE"],
        "Overfit_Gap_TrainMinusCV": selected_row[
            "Overfit_Gap_TrainMinusCV"
        ],
        "BestFit_Equation_Test": selected_row["BestFit_Equation_Test"],
        "Top_5_Importance_Features": ", ".join(top_importance),
        "Top_5_SHAP_Features": ", ".join(top_shap),
        "Best_Feature_Group_By_CV": best_group,
        "Engineering_Meaning": config["meaning"],
    }

    return {
        "target": target,
        "target_column": target_column,
        "features": matched,
        "matching": matching_df,
        "performance": performance_df,
        "best_parameters": pd.DataFrame(parameter_rows),
        "equations": pd.DataFrame(equation_rows),
        "relative_errors": relative_df,
        "predictions": predictions_df,
        "search_results": pd.concat(search_frames, ignore_index=True),
        "importance": importance_df,
        "shap": shap_df,
        "ablation": ablation_df,
        "selected_model_name": selected_model_name,
        "selected_pipeline": selected_pipeline,
        "summary": final_summary,
        "split": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_data_with_target": train_data,
            "locked_test_data_with_target": locked_test_data,
            "selected_features": selected_features,
        },
    }


# =============================================================================
# 13. EXCEL REPORT AND FINAL SUMMARY
# =============================================================================

def dataset_summary_table(data, selected_sheet, all_sheets):
    rows = [
        {"Item": "Input file", "Value": str(input_file)},
        {"Item": "Selected sheet", "Value": selected_sheet},
        {"Item": "Available sheets", "Value": ", ".join(all_sheets)},
        {"Item": "Rows", "Value": len(data)},
        {"Item": "Columns", "Value": len(data.columns)},
        {"Item": "Total missing cells", "Value": int(data.isna().sum().sum())},
        {"Item": "Train fraction", "Value": 1 - TEST_SIZE},
        {"Item": "Test fraction", "Value": TEST_SIZE},
        {"Item": "CV folds", "Value": CV_FOLDS},
        {"Item": "Random state", "Value": RANDOM_STATE},
        {"Item": "XGBoost available", "Value": HAS_XGBOOST},
        {"Item": "SHAP available", "Value": HAS_SHAP},
        {
            "Item": "Selection rule",
            "Value": "Best model and feature group selected by training CV.",
        },
    ]
    return pd.DataFrame(rows)


def modeling_recommendations():
    return pd.DataFrame(
        [
            {
                "Priority": 1,
                "Recommendation": (
                    "Treat the 25% test results as locked confirmation. Do not "
                    "retune after viewing them."
                ),
            },
            {
                "Priority": 2,
                "Recommendation": (
                    "Select models and feature groups using CV mean, CV spread, "
                    "error, overfit gap, and engineering plausibility together."
                ),
            },
            {
                "Priority": 3,
                "Recommendation": (
                    "If repeated records later exist for the same JMF, project, "
                    "or roadway section, replace random splitting with grouped CV."
                ),
            },
            {
                "Priority": 4,
                "Recommendation": (
                    "Do not claim field long-term performance prediction from "
                    "laboratory Rut_20k and SCB targets alone."
                ),
            },
            {
                "Priority": 5,
                "Recommendation": (
                    "Add traffic history, climate, layer structure, age, binder "
                    "rheology, and repeated field condition records when available."
                ),
            },
            {
                "Priority": 6,
                "Recommendation": (
                    "Interpret SHAP associations as model behavior, not proof of "
                    "causality."
                ),
            },
        ]
    )


def save_publication_workbook(
    data,
    selected_sheet,
    all_sheets,
    rut_result,
    scb_result,
    final_summary_df,
    paths,
):
    if not SAVE_OUTPUTS:
        return None

    workbook_path = paths["root"] / "Publication_Modeling_Results_Rut_SCB.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        dataset_summary_table(data, selected_sheet, all_sheets).to_excel(
            writer,
            sheet_name="Dataset_Summary",
            index=False,
        )
        pd.DataFrame(PAPER_METHOD_ROWS).to_excel(
            writer,
            sheet_name="Paper_Methodology",
            index=False,
        )
        rut_result["matching"].to_excel(
            writer,
            sheet_name="Rut_Selected_Features",
            index=False,
        )
        scb_result["matching"].to_excel(
            writer,
            sheet_name="SCB_Selected_Features",
            index=False,
        )
        rut_result["performance"].to_excel(
            writer,
            sheet_name="Rut_Model_Performance",
            index=False,
        )
        scb_result["performance"].to_excel(
            writer,
            sheet_name="SCB_Model_Performance",
            index=False,
        )
        rut_result["best_parameters"].to_excel(
            writer,
            sheet_name="Rut_Best_Parameters",
            index=False,
        )
        scb_result["best_parameters"].to_excel(
            writer,
            sheet_name="SCB_Best_Parameters",
            index=False,
        )
        rut_result["equations"].to_excel(
            writer,
            sheet_name="Rut_BestFit_Equations",
            index=False,
        )
        scb_result["equations"].to_excel(
            writer,
            sheet_name="SCB_BestFit_Equations",
            index=False,
        )
        rut_result["relative_errors"].to_excel(
            writer,
            sheet_name="Rut_Relative_Error",
            index=False,
        )
        scb_result["relative_errors"].to_excel(
            writer,
            sheet_name="SCB_Relative_Error",
            index=False,
        )
        rut_result["importance"].to_excel(
            writer,
            sheet_name="Rut_Feature_Importance",
            index=False,
        )
        scb_result["importance"].to_excel(
            writer,
            sheet_name="SCB_Feature_Importance",
            index=False,
        )
        rut_result["shap"].to_excel(
            writer,
            sheet_name="Rut_SHAP_Importance",
            index=False,
        )
        scb_result["shap"].to_excel(
            writer,
            sheet_name="SCB_SHAP_Importance",
            index=False,
        )
        rut_result["ablation"].to_excel(
            writer,
            sheet_name="Rut_Feature_Group_Study",
            index=False,
        )
        scb_result["ablation"].to_excel(
            writer,
            sheet_name="SCB_Feature_Group_Study",
            index=False,
        )
        pd.concat(
            [
                rut_result["performance"],
                scb_result["performance"],
            ],
            ignore_index=True,
        ).to_excel(
            writer,
            sheet_name="Overfitting_Check",
            index=False,
        )
        final_summary_df.to_excel(
            writer,
            sheet_name="Final_Best_Model_Summary",
            index=False,
        )
        modeling_recommendations().to_excel(
            writer,
            sheet_name="Modeling_Recommendations",
            index=False,
        )
    return workbook_path


def print_final_summary(final_summary_df):
    print("\n" + "=" * 100)
    print("FINAL PUBLICATION MODELING SUMMARY")
    print("=" * 100)
    for row in final_summary_df.itertuples(index=False):
        print(f"\nTarget: {row.Target}")
        print("CV-selected model:", row.Selected_Model_CV_Based)
        print(f"Training R2: {row.Training_R2:.4f}")
        print(
            f"Five-fold CV R2: {row.CV_Mean_R2:.4f} "
            f"+/- {row.CV_SD_R2:.4f}"
        )
        print(f"Independent testing R2: {row.Testing_R2:.4f}")
        print(f"Independent testing RMSE: {row.Testing_RMSE:.4f}")
        print(f"Independent testing MAE: {row.Testing_MAE:.4f}")
        print(
            "Train-CV overfit gap:",
            f"{row.Overfit_Gap_TrainMinusCV:.4f}",
        )
        print("Testing best-fit equation:", row.BestFit_Equation_Test)
        print("Top five importance features:", row.Top_5_Importance_Features)
        print("Top five SHAP features:", row.Top_5_SHAP_Features)
        print("Best feature group by CV:", row.Best_Feature_Group_By_CV)
        print("Engineering interpretation:", row.Engineering_Meaning)

    print("\nPaper discussion recommendation:")
    print(
        "Report CV-based selection, locked-test confirmation, error metrics, "
        "overfit gaps, and whether feature directions agree with asphalt "
        "engineering. Code execution alone does not establish publication "
        "readiness."
    )


# =============================================================================
# 14. MAIN
# =============================================================================

def main():
    start_time = time.time()
    paths = make_output_directories()

    print("=" * 100)
    print("PUBLICATION MODELING: RUT_20K AND SCB JC")
    print("=" * 100)
    print("Input:", input_file)
    print("Output:", output_folder)
    print("75/25 split:", f"random_state={RANDOM_STATE}")
    print("Five-fold CV tuning: YES")
    print("Primary models: Gradient Boosting, XGBoost")
    print("XGBoost available:", HAS_XGBOOST)
    print("SHAP available:", HAS_SHAP)

    data, selected_sheet, all_sheets = load_excel_dataset()
    rut_result = run_target_workflow(data, "Rut_20k", paths)
    scb_result = run_target_workflow(data, "SCB", paths)

    final_summary_df = pd.DataFrame(
        [rut_result["summary"], scb_result["summary"]]
    )

    if SAVE_OUTPUTS:
        for result in (rut_result, scb_result):
            target_label = safe_name(result["target"])
            result["performance"].to_csv(
                paths["tables"] / f"{target_label}_model_performance.csv",
                index=False,
            )
            result["predictions"].to_csv(
                paths["tables"] / f"{target_label}_predictions.csv",
                index=False,
            )
            result["search_results"].to_csv(
                paths["tables"] / f"{target_label}_tuning_search_results.csv",
                index=False,
            )
            result["importance"].to_csv(
                paths["tables"] / f"{target_label}_feature_importance.csv",
                index=False,
            )
            result["ablation"].to_csv(
                paths["tables"] / f"{target_label}_feature_group_study.csv",
                index=False,
            )

    workbook_path = save_publication_workbook(
        data=data,
        selected_sheet=selected_sheet,
        all_sheets=all_sheets,
        rut_result=rut_result,
        scb_result=scb_result,
        final_summary_df=final_summary_df,
        paths=paths,
    )

    print_final_summary(final_summary_df)
    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"\nElapsed time: {elapsed_minutes:.2f} minutes")
    if workbook_path is not None:
        print("Excel workbook:", workbook_path)
        print("Figures:", paths["figures"])
        print("SHAP outputs:", paths["shap"])
        print("Saved models:", paths["models"])

    return {
        "Rut_20k": rut_result,
        "SCB": scb_result,
        "Final_Summary": final_summary_df,
        "Workbook": workbook_path,
        "Output_Paths": paths,
    }


if __name__ == "__main__":
    PUBLICATION_RESULTS = main()
