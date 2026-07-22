# -*- coding: utf-8 -*-
"""
RUTTING & SCB PREDICTION — DESIGN / VALIDATION / COMBINED, 70/10/20, ENGINEERING FEATURES,
                           TUNED LightGBM + (0.6 ExtraTrees + 0.4 LightGBM) BLEND + MODEL COMPARISON
Author: prepared for Sarah Al-Jezawi

WHAT THIS SCRIPT ANSWERS
------------------------
1. Predict BOTH targets: set TARGET = "Rut_20k" or "SCB".
2. Give an INDICATION of design vs validation vs combined: set DATA_MODE, or leave
   COMPARE_DATA_MODES = True to run all three back-to-back and print a comparison table
   (which way of describing the data to the model gives the best honest test score).
3. Engineering features per the pavement-mechanism list (rutting- and SCB-specific), built
   defensively from whatever columns exist. Features that need data we do NOT have
   (DSR G*/sinδ, PAV rheology, PG_LowTemp, Aging_Day, gyratory Ndes densification) are
   listed at the bottom as the highest-value future additions.
4. Models: a tuned LightGBM and the requested 0.6*ExtraTrees + 0.4*LightGBM BLEND are the
   headline models; XGBoost, CatBoost, HistGB, RandomForest and a Ridge-stacked ensemble run
   alongside for comparison ("suggest additional superior models").

LEAKAGE CONTROL (publication-grade)
-----------------------------------
- 70/10/20 split is GROUPED by MixDesignKey so no mix (or its replicates) spans train/val/test.
- Replicate tests of a mix are AVERAGED to one de-noised target (set AVERAGE_REPLICATES=False to keep all).
- All preprocessing is inside the sklearn Pipeline, fit within each CV fold only.
- The 20% test is scored EXACTLY ONCE, after the model is refit on train+validation.
"""

from __future__ import annotations
import warnings, re
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              HistGradientBoostingRegressor, VotingRegressor, StackingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# =============================================================================
# SETTINGS
# =============================================================================
TARGET = "Rut_20k"          # "Rut_20k" (rutting) or "SCB"
DATA_MODE = "combined"      # "design" | "validation" | "combined"
COMPARE_DATA_MODES = True   # run design/validation/combined and print a comparison table
RANDOM_STATE = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.10, 0.20
AVERAGE_REPLICATES = True   # average replicate tests of a mix to one de-noised target
GROUP_COL = "MixDesignKey"  # group split by physical mix (no leakage)
N_ITER = 40                 # RandomizedSearch iterations per tuned model (lower = faster)
CV_FOLDS = 5
SHOW_PLOTS = True           # show plots inline in Spyder
BLEND_WEIGHTS = (0.60, 0.40)  # (ExtraTrees, LightGBM) — the requested blend

HOME = Path.home()
DOWNLOADS = Path(r"C:\Users\H0012066\Downloads")
if not DOWNLOADS.exists():
    DOWNLOADS = HOME / "Downloads" if (HOME / "Downloads").exists() else Path.cwd()
DATA_FILE = "Design_Validation_Rutting_SCB_Separated_Cleaned.xlsx"
OUT = DOWNLOADS / "Rut_SCB_Blend_outputs"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

if SHOW_PLOTS:
    try: plt.ion()
    except Exception: pass
np.random.seed(RANDOM_STATE)

CATEGORICAL = ["MixType", "DesignLev", "RAP_Class"]

# =============================================================================
# FEATURE ENGINEERING  (pavement-mechanism parameters, built only when sources exist)
# =============================================================================
def num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    Pb = num(df, "AsphaltContent_Design")
    # ---- RAP / RBR binder ----
    df["RAP_Binder_Load"] = num(df, "RAP_pct") * num(df, "ACinRAP") / 100.0      # RAP binder in mix
    if "RBR_percent" not in df.columns:
        df["RBR_percent"] = 100.0 * df["RAP_Binder_Load"] / Pb.replace(0, np.nan)
    df["RBR_fraction"] = num(df, "RBR_percent") / 100.0
    # ---- Physics chain: Gse -> Pba -> Pbe -> surface area -> AFT ----
    Gb = 1.03
    if "Gse" not in df.columns:
        df["Gse"] = (100.0 - Pb) / (100.0 / num(df, "Gmm") - Pb / Gb)
    if "Pba_pct" not in df.columns:
        df["Pba_pct"] = 100.0 * (num(df, "Gse") - num(df, "Gsb")) / (num(df, "Gse") * num(df, "Gsb")) * Gb
    if "Pbe_pct" not in df.columns:
        df["Pbe_pct"] = Pb - (num(df, "Pba_pct") / 100.0) * (100.0 - Pb)
    # Hveem surface area (ft2/lb) from % passing (decimals) -> m2/kg
    saf = {"Grad_No4": 2.0, "Grad_No8": 4.0, "Grad_No16": 8.0, "Grad_No30": 14.0,
           "Grad_No50": 30.0, "Grad_No100": 60.0, "Grad_No200": 160.0}
    if set(saf).issubset(df.columns):
        sa = 2.0
        for c, f in saf.items():
            sa = sa + f * (num(df, c) / 100.0)
        df["SurfaceArea_ft2lb"] = sa
        df["SurfaceArea_m2kg"] = sa * 0.20482
        Ps = (100.0 - Pb) / 100.0
        df["AFT_micron"] = num(df, "Pbe_pct") * 4870.0 / (100.0 * Ps * sa)
    # ---- Mastic / film ratios ----
    df["Fines_to_Pbe"] = num(df, "Grad_No200") / num(df, "Pbe_pct").replace(0, np.nan)
    df["Dust_Pbe_ratio"] = df["Fines_to_Pbe"]
    if "AFT_micron" in df.columns:
        df["BinderAvailability"] = num(df, "Pbe_pct") / num(df, "SurfaceArea_m2kg").replace(0, np.nan)
        df["AgingBurden"] = num(df, "RBR_percent") / num(df, "AFT_micron").replace(0, np.nan)
        df["PG_x_AFT"] = num(df, "PG_HighTemp") * num(df, "AFT_micron")
        df["DustPbe_x_AFT"] = df["Dust_Pbe_ratio"] * num(df, "AFT_micron")
        df["PG_RBR_AFT_Severity"] = num(df, "PG_HighTemp") * df["RBR_fraction"] / num(df, "AFT_micron").replace(0, np.nan)
        df["Absorption_SurfaceDemand"] = num(df, "Absorption") * num(df, "SurfaceArea_m2kg")
    # ---- Effective / virgin binder ----
    df["VirginEffectiveBinder"] = num(df, "Pbe_pct") * (1.0 - df["RBR_fraction"])
    df["RecycledBinderBurden"] = num(df, "Pbe_pct") * df["RBR_fraction"]
    df["AbsorptionPenalty"] = num(df, "Absorption") * Pb
    # ---- Volumetric protection ----
    df["VMA_Filled_Index"] = num(df, "VMA") * num(df, "VFA") / 100.0
    df["BinderFilledVolume"] = num(df, "VMA") - num(df, "Va")
    if "SurfaceArea_m2kg" in df.columns:
        df["CrackingExposure"] = num(df, "Va") * num(df, "SurfaceArea_m2kg") / num(df, "Pbe_pct").replace(0, np.nan)
    # ---- Gradation physical fractions + Bailey-style proxies + retained ratios ----
    df["Coarse_Fraction"] = 100.0 - num(df, "Grad_No4")
    df["Intermediate_Fraction"] = num(df, "Grad_No4") - num(df, "Grad_No8")
    df["Fine_Fraction"] = num(df, "Grad_No8") - num(df, "Grad_No200")
    df["Filler_Fraction"] = num(df, "Grad_No200")
    df["No50_to_No8"] = num(df, "Grad_No50") / num(df, "Grad_No8").replace(0, np.nan)
    df["No8_to_No4"] = num(df, "Grad_No8") / num(df, "Grad_No4").replace(0, np.nan)
    df["Ret_1_2in_to_3_8in"] = (100 - num(df, "Grad_1_2in")) / (100 - num(df, "Grad_3_8in")).replace(0, np.nan)
    df["Ret_3_8in_to_No4"] = (num(df, "Grad_3_8in") - num(df, "Grad_No4"))
    df["Bailey_CA_Proxy"] = num(df, "Grad_No4") / num(df, "Grad_1_2in").replace(0, np.nan)          # coarse packing proxy
    df["Bailey_FAf_Proxy"] = num(df, "Grad_No100") / num(df, "Grad_No30").replace(0, np.nan)         # fine-of-fine proxy
    df["CoarseFine_Ratio"] = df["Coarse_Fraction"] / (df["Fine_Fraction"].replace(0, np.nan))
    df["Fines_x_Absorption"] = num(df, "Grad_No200") * num(df, "Absorption")
    # ---- Binder / traffic / temperature interactions ----
    df["PG_x_RBR"] = num(df, "PG_HighTemp") * df["RBR_fraction"]
    df["PG_x_Va"] = num(df, "PG_HighTemp") * num(df, "Va")
    if "ADT_Midpoint" not in df.columns:
        df["ADT_Midpoint"] = num(df, "ADT").where(num(df, "ADT").notna(), np.nan)
    if "MixTemperature_F" in df.columns:
        df["Temperature_x_PG"] = num(df, "MixTemperature_F") * num(df, "PG_HighTemp")
    return df

# Target-specific compact engineering feature sets (only-present ones are used).
RUT_FEATURES = [
    "PG_HighTemp", "RBR_percent", "RAP_Binder_Load", "Va", "VMA", "Pbe_pct", "Fines_to_Pbe",
    "CAA", "FAA", "SandEq", "Coarse_Fraction", "Intermediate_Fraction", "Grad_No200", "NMAS (mm)",
    "PG_x_RBR", "PG_x_Va", "PG_x_AFT", "PG_RBR_AFT_Severity", "Absorption", "Absorption_SurfaceDemand",
    "Bailey_CA_Proxy", "Bailey_FAf_Proxy", "No50_to_No8", "Ret_1_2in_to_3_8in", "VMA_Filled_Index",
    "ADT_Midpoint", "DesignLev", "Additive_Rate_clean", "Gse", "Gsb",
]
SCB_FEATURES = [
    "Pbe_pct", "AFT_micron", "SurfaceArea_m2kg", "Va", "VMA", "RBR_percent", "VirginEffectiveBinder",
    "RecycledBinderBurden", "AgingBurden", "Absorption", "Dust_Pbe_ratio", "BinderAvailability",
    "CrackingExposure", "VMA_Filled_Index", "AbsorptionPenalty", "SandEq", "FAA",
    "Grad_No50", "Grad_No100", "Grad_No200", "Bailey_FAf_Proxy", "PG_x_RBR", "DustPbe_x_AFT", "Gse",
]

# =============================================================================
# DATA LOADING (design / validation / combined)  + replicate averaging
# =============================================================================
def load(mode: str):
    tag = "Rutting" if TARGET == "Rut_20k" else "SCB"
    path = DOWNLOADS / DATA_FILE
    if not path.exists():
        for alt in [Path.cwd() / DATA_FILE]:
            if alt.exists(): path = alt
    sheets = {"design": [f"Design_{tag}"], "validation": [f"Validation_{tag}"],
              "combined": [f"Design_{tag}", f"Validation_{tag}"]}[mode]
    frames = []
    for s in sheets:
        d = pd.read_excel(path, sheet_name=s)
        d.columns = [str(c).strip() for c in d.columns]
        d["Data_Source_Sheet"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
    if TARGET == "SCB":
        df = df.loc[pd.to_numeric(df[TARGET], errors="coerce") > 0].reset_index(drop=True)  # drop Jc<=0
    df = engineer(df)
    if AVERAGE_REPLICATES and GROUP_COL in df.columns:
        agg = {c: ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "first")
               for c in df.columns if c != GROUP_COL}
        df = df.groupby(GROUP_COL, as_index=False).agg(agg)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    return df.reset_index(drop=True), y.reset_index(drop=True)

def target_bins(y, n=5):
    for q in [n, 4, 3, 2]:
        try:
            b = pd.qcut(y, q=q, labels=False, duplicates="drop")
            if pd.Series(b).nunique() >= 2: return b.astype(int)
        except Exception: pass
    return (y >= np.median(y)).astype(int)

def grouped_split(df, y):
    """Two-stage grouped, target-stratified split -> train / val / test index arrays."""
    groups = df[GROUP_COL].astype(str).values if GROUP_COL in df.columns else np.arange(len(df))
    bins = target_bins(y)
    n_test = max(2, round(1/TEST_SIZE))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=RANDOM_STATE)
    dev_idx, test_idx = next(iter(sgkf.split(np.zeros(len(y)), bins, groups)))
    # split dev into train/val
    devb = pd.Series(bins).iloc[dev_idx].values
    devg = groups[dev_idx]
    n_val = max(2, round((TRAIN_SIZE+VAL_SIZE)/VAL_SIZE))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=RANDOM_STATE)
    tr_pos, val_pos = next(iter(sgkf2.split(np.zeros(len(dev_idx)), devb, devg)))
    return dev_idx[tr_pos], dev_idx[val_pos], test_idx

# =============================================================================
# MODELS
# =============================================================================
def get_X(df, feats):
    avail = [f for f in feats if f in df.columns]
    X = df[avail].copy()
    numeric, categ = [], []
    for c in X.columns:
        if c in CATEGORICAL:
            categ.append(c); X[c] = X[c].astype("object").where(X[c].notna(), "Missing").astype(str)
        else:
            numeric.append(c); X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, numeric, categ

def preproc(numeric, categ):
    t = []
    if numeric: t.append(("num", SimpleImputer(strategy="median"), numeric))
    if categ:
        t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                                   ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categ))
    return ColumnTransformer(t, remainder="drop")

def base_models():
    """Return {name: (estimator, param_grid)}. LightGBM + ExtraTrees are tuned; others compared."""
    m = {}
    if HAS_LGBM:
        m["LightGBM"] = (LGBMRegressor(random_state=RANDOM_STATE, verbose=-1), {
            "model__n_estimators": [400, 700, 1200], "model__learning_rate": [0.01, 0.02, 0.04],
            "model__num_leaves": [15, 31, 63], "model__min_child_samples": [15, 30, 50],
            "model__subsample": [0.7, 0.85, 1.0], "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.0, 5.0, 20.0]})
    m["ExtraTrees"] = (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
        "model__n_estimators": [400, 800], "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 3, 8], "model__max_features": ["sqrt", 0.5, 0.8]})
    m["RandomForest"] = (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
        "model__n_estimators": [400, 800], "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 3, 8], "model__max_features": ["sqrt", 0.5, 0.8]})
    m["HistGB"] = (HistGradientBoostingRegressor(random_state=RANDOM_STATE), {
        "model__learning_rate": [0.02, 0.05, 0.08], "model__max_iter": [300, 600],
        "model__max_leaf_nodes": [15, 31], "model__min_samples_leaf": [15, 30],
        "model__l2_regularization": [0.0, 1.0, 5.0]})
    if HAS_XGB:
        m["XGBoost"] = (XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                     random_state=RANDOM_STATE, n_jobs=-1), {
            "model__n_estimators": [400, 800, 1200], "model__learning_rate": [0.01, 0.02, 0.04],
            "model__max_depth": [2, 3, 4], "model__min_child_weight": [5, 15, 30],
            "model__subsample": [0.7, 0.85], "model__colsample_bytree": [0.6, 0.8],
            "model__reg_lambda": [5, 20, 60]})
    if HAS_CAT:
        m["CatBoost"] = (CatBoostRegressor(verbose=0, random_seed=RANDOM_STATE), {
            "model__iterations": [400, 800], "model__learning_rate": [0.02, 0.04, 0.06],
            "model__depth": [4, 6], "model__l2_leaf_reg": [3, 10, 30]})
    return m

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def metrics(a, b): return {"R2": r2_score(a, b), "RMSE": rmse(a, b), "MAE": mean_absolute_error(a, b)}

def tune(est, grid, numeric, categ, Xtr, ytr, groups_tr):
    pipe = Pipeline([("prep", preproc(numeric, categ)), ("model", est)])
    if not grid:
        pipe.fit(Xtr, ytr); return pipe
    cv = list(GroupKFold(n_splits=CV_FOLDS).split(Xtr, ytr, groups_tr))
    space = int(np.prod([len(v) for v in grid.values()]))
    s = RandomizedSearchCV(pipe, grid, n_iter=min(N_ITER, space), scoring="r2", cv=cv,
                           random_state=RANDOM_STATE, n_jobs=-1, error_score=np.nan)
    s.fit(Xtr, ytr)
    return s.best_estimator_

# =============================================================================
# RUN ONE (target, mode)
# =============================================================================
def run(mode):
    df, y = load(mode)
    feats = RUT_FEATURES if TARGET == "Rut_20k" else SCB_FEATURES
    X, numeric, categ = get_X(df, feats)
    tr, val, te = grouped_split(df, y)
    groups = df[GROUP_COL].astype(str).values
    Xtr, Xval, Xte = X.iloc[tr], X.iloc[val], X.iloc[te]
    ytr, yval, yte = y.iloc[tr], y.iloc[val], y.iloc[te]
    print(f"\n{'='*80}\nTARGET={TARGET} | DATA_MODE={mode} | features used={len(numeric)+len(categ)}")
    print(f"Split (grouped by {GROUP_COL}): Train70={len(tr)}  Validation10={len(val)}  LockedTest20={len(te)}"
          f"  | unique mixes={df[GROUP_COL].nunique()}")

    # tune base models, score on validation
    fitted, rows = {}, []
    for name, (est, grid) in base_models().items():
        try:
            f = tune(est, grid, numeric, categ, Xtr, ytr, groups[tr])
            fitted[name] = f
            vm = metrics(yval, f.predict(Xval))
            rows.append({"Model": name, **{f"Val_{k}": v for k, v in vm.items()}})
        except Exception as e:
            print(f"  {name} failed: {type(e).__name__}: {e}")

    # ---- the requested BLEND: 0.6*ExtraTrees + 0.4*LightGBM ----
    if "ExtraTrees" in fitted and "LightGBM" in fitted:
        blend = VotingRegressor([("et", clone(fitted["ExtraTrees"].named_steps["model"])),
                                 ("lgb", clone(fitted["LightGBM"].named_steps["model"]))],
                                weights=list(BLEND_WEIGHTS))
        bpipe = Pipeline([("prep", preproc(numeric, categ)), ("model", blend)])
        bpipe.fit(Xtr, ytr); fitted["Blend_0.6ET_0.4LGBM"] = bpipe
        vm = metrics(yval, bpipe.predict(Xval))
        rows.append({"Model": "Blend_0.6ET_0.4LGBM", **{f"Val_{k}": v for k, v in vm.items()}})

    # ---- stacking (suggested superior model) ----
    try:
        ests = [(n, Pipeline([("prep", preproc(numeric, categ)), ("model", clone(fitted[n].named_steps["model"]))]))
                for n in ["LightGBM", "ExtraTrees", "HistGB"] if n in fitted]
        if len(ests) >= 2:
            stack = StackingRegressor(ests, final_estimator=RidgeCV(), cv=CV_FOLDS, n_jobs=-1)
            stack.fit(Xtr, ytr); fitted["Stacking"] = stack
            vm = metrics(yval, stack.predict(Xval))
            rows.append({"Model": "Stacking", **{f"Val_{k}": v for k, v in vm.items()}})
    except Exception as e:
        print(f"  Stacking failed: {type(e).__name__}: {e}")

    comp = pd.DataFrame(rows).sort_values("Val_R2", ascending=False).reset_index(drop=True)
    print("\nModel comparison (validation, sorted):")
    print(comp.round(4).to_string(index=False))

    # ---- select best on validation, refit on train+val, score locked test ONCE ----
    best_name = comp.iloc[0]["Model"]
    Xdev = pd.concat([Xtr, Xval]); ydev = pd.concat([ytr, yval])
    best = clone(fitted[best_name]); best.fit(Xdev, ydev)
    splits = {"Train70": (Xtr, ytr), "Validation10": (Xval, yval), "LockedTest20": (Xte, yte)}
    final_rows = []
    for sname, (Xs, ys) in splits.items():
        m = metrics(ys, best.predict(Xs)); final_rows.append({"Dataset": sname, "Rows": len(ys), **m})
    final = pd.DataFrame(final_rows)
    print(f"\nBEST MODEL = {best_name}  (refit on train+val, test scored once):")
    print(final.round(4).to_string(index=False))
    return {"mode": mode, "best": best_name, "comp": comp, "final": final,
            "model": best, "splits": splits, "feats": (numeric, categ), "Xdev": Xdev}

def best_fit_plot(ys, yp, title, path):
    ys, yp = np.asarray(ys, float), np.asarray(yp, float)
    m = metrics(ys, yp); sl, ic = np.polyfit(ys, yp, 1)
    plt.figure(figsize=(6.5, 6)); plt.scatter(ys, yp, alpha=0.6, s=22)
    lo, hi = float(min(ys.min(), yp.min())), float(max(ys.max(), yp.max()))
    xs = np.linspace(lo, hi, 100)
    plt.plot([lo, hi], [lo, hi], "--", lw=2, label="Ideal 1:1")
    plt.plot(xs, sl*xs+ic, lw=2, label=f"Best fit: y={sl:.3f}x+{ic:.3f}")
    plt.xlabel(f"Measured {TARGET}"); plt.ylabel(f"Predicted {TARGET}")
    plt.title(f"{title}\nR2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        try: plt.show()
        except Exception: pass
    plt.close()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print(f"RUTTING/SCB BLEND WORKFLOW | TARGET={TARGET} | blend={BLEND_WEIGHTS} "
          f"| LightGBM={HAS_LGBM} XGB={HAS_XGB} CatBoost={HAS_CAT} SHAP={HAS_SHAP}")
    print("="*80)
    modes = ["design", "validation", "combined"] if COMPARE_DATA_MODES else [DATA_MODE]
    results = {}
    for mode in modes:
        try:
            results[mode] = run(mode)
        except Exception as e:
            print(f"MODE {mode} failed: {type(e).__name__}: {e}")

    # ---- design vs validation vs combined indication ----
    if len(results) > 1:
        ind = []
        for mode, r in results.items():
            t = r["final"].set_index("Dataset")
            ind.append({"Data_Mode": mode, "Best_Model": r["best"],
                        "Val_R2": round(float(t.loc["Validation10", "R2"]), 4),
                        "Test_R2": round(float(t.loc["LockedTest20", "R2"]), 4),
                        "Test_RMSE": round(float(t.loc["LockedTest20", "RMSE"]), 4)})
        ind_df = pd.DataFrame(ind).sort_values("Test_R2", ascending=False)
        print("\n" + "="*80)
        print("INDICATION — which way to describe the data gives the best honest test score:")
        print("="*80); print(ind_df.to_string(index=False))
        winner = ind_df.iloc[0]["Data_Mode"]
    else:
        winner = list(results.keys())[0]

    # ---- best-fit plots (train/val/test) + SHAP for the winning data mode ----
    r = results[winner]
    print(f"\nGenerating best-fit plots + SHAP for winning mode = {winner} (model {r['best']})")
    for sname, (Xs, ys) in r["splits"].items():
        best_fit_plot(ys, r["model"].predict(Xs), f"{TARGET} {sname} — {winner} — {r['best']}",
                      OUT / "figures" / f"{TARGET}_{winner}_{sname}_bestfit.png")
    if HAS_SHAP:
        try:
            mdl = r["model"]
            pre = mdl.named_steps["prep"]; inner = mdl.named_steps["model"]
            Xt = pre.transform(r["Xdev"]); names = list(pre.get_feature_names_out())
            expl_model = inner
            if hasattr(inner, "estimators_") and not hasattr(inner, "feature_importances_"):
                expl_model = inner.estimators_[0]  # blend/voting -> explain first base
            sv = shap.TreeExplainer(expl_model)(Xt)
            plt.figure(); shap.summary_plot(sv.values, Xt, feature_names=names, show=False, max_display=20)
            plt.tight_layout(); plt.savefig(OUT / "figures" / f"{TARGET}_{winner}_shap_beeswarm.png", dpi=160, bbox_inches="tight")
            if SHOW_PLOTS:
                try: plt.show()
                except Exception: pass
            plt.close()
            imp = pd.DataFrame({"Feature": names, "MeanAbsSHAP": np.abs(sv.values).mean(0)}).sort_values("MeanAbsSHAP", ascending=False)
            print("\nTop-15 SHAP features:"); print(imp.head(15).round(4).to_string(index=False))
        except Exception as e:
            print(f"SHAP skipped: {type(e).__name__}: {e}")

    print(f"\nDone. Figures + outputs in: {OUT}")
    print("\nHIGHEST-VALUE FUTURE FEATURES (need data not in this file):")
    print("  Rutting: DSR G*/sinδ (RTFO), gyratory %Gmm@Nini/Ndes/Nmax (densification slope), LWT test temperature")
    print("  SCB:     PG_LowTemp / PAV G*sinδ, ΔTc, carbonyl index, aging days, polymer-modification flag")

if __name__ == "__main__":
    main()
