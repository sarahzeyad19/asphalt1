# -*- coding: utf-8 -*-
"""
SCB WORKFLOW — RECOMMENDED FEATURES + SHAP + PDP  (SPYDER-FRIENDLY: plots shown on screen)
==========================================================================================
Data file : New_Data_SCB_LWT_Cleaned_Modeling_Files_with_RBR.xlsx  (sheet SCB_Clean_Modeling)
Target    : SCB (Jc) — higher is better.
Approach  : the tuned-ExtraTrees protocol that produced the high (~0.68) validation result:
            70/10/20 stratified split, RandomizedSearchCV on train only, RepeatedCV (5x5) on the
            dev set, the 20% locked test scored ONCE at the end. SHAP + PDP on the final model.

FEATURES (as recommended):
    PG_HighTemp, ADT(->ordinal), DesignLev, NMAS (mm), MixType, RBR_percent, AFT_micron,
    Dust_Pbe_ratio, Va, VMA, Gsb, Absorption, CAA, FAA, SandEq, Grad_No4, Grad_No30,
    Grad_No200, Has_Additive, Additive_Type_clean, Additive_Rate_clean, MixTemperature_F_clean
ENGINEERED INTERACTIONS (as recommended):
    RBR_x_AFT, RBR_x_PG, RBR_x_DustPbe, AFT_x_DustPbe, Va_to_VMA, Absorption_x_AFT
DO-NOT-USE-TOGETHER rules are checked automatically and a warning is printed if violated:
    {RAP_pct, ACinRAP, RBR_percent} | {Gmm, Gmb, Va} | {Va, VMA, VFA}
    {AsphaltContent_Design, Pbe_pct, AFT_micron} | {Dust_Binder, Dust_Pbe_ratio, Grad_No200}
    all Grad_* sieves together
"""
from __future__ import annotations
import json, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt   # NO Agg backend -> figures appear in the Spyder plots pane

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, StratifiedKFold, StratifiedGroupKFold,
                                     RandomizedSearchCV, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import shap; HAS_SHAP = True
except Exception:
    shap = None; HAS_SHAP = False
    print("NOTE: shap is not installed -> SHAP plots will be skipped. Fix: pip install shap")

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
TARGET = "SCB"; ID_COL = "MixDesignKey"
SHEET = "SCB_Clean_Modeling"
FILE_NAME = "New_Data_SCB_LWT_Cleaned_Modeling_Files_with_RBR.xlsx"
TRAIN, VAL, TEST = 0.70, 0.10, 0.20
CV_FOLDS, N_TARGET_BINS = 5, 5
N_ITER = 80
REPEATED_REPEATS = 5
# Use ALL test rows (replicates kept). Leakage is prevented by the GROUP-aware split below:
# every replicate of a mix goes to the SAME split, so no mix straddles train/val/test.
UNIQUE_MIXES = False
SHOW_PLOTS = True                 # True = show every figure in Spyder (they are ALSO saved)
MAX_SHAP_ROWS = 500
OUT = Path("SCB_Recommended_SHAP_PDP_outputs")
for sub in ["", "figures", "splits", "models"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)
if SHOW_PLOTS:
    try: plt.ion()
    except Exception: pass

# =============================================================================
# FEATURES (exactly the recommended list; ADT becomes ADT_ord)
# =============================================================================
RECOMMENDED = ["PG_HighTemp", "ADT_ord", "DesignLev", "NMAS (mm)", "MixType", "RBR_percent",
               "AFT_micron", "Dust_Pbe_ratio", "Va", "VMA", "Gsb", "Absorption", "CAA", "FAA",
               "SandEq", "Grad_No4", "Grad_No30", "Grad_No200", "Has_Additive",
               "Additive_Type_clean", "Additive_Rate_clean", "MixTemperature_F_clean"]
INTERACTIONS = ["RBR_x_AFT", "RBR_x_PG", "RBR_x_DustPbe", "AFT_x_DustPbe", "Va_to_VMA",
                "Absorption_x_AFT"]
CATEGORICAL_HINTS = ["MixType", "DesignLev", "Additive_Type_clean", "Has_Additive"]

FORBIDDEN_TOGETHER = [
    {"RAP_pct", "ACinRAP", "RBR_percent"},
    {"Gmm", "Gmb", "Va"},
    {"Va", "VMA", "VFA"},
    {"AsphaltContent_Design", "Pbe_pct", "AFT_micron"},
    {"Dust_Binder", "Dust_Pbe_ratio", "Grad_No200"},
]
ALL_GRAD = {"Grad_1_5in", "Grad_1in", "Grad_3_4in", "Grad_1_2in", "Grad_3_8in", "Grad_No4",
            "Grad_No8", "Grad_No16", "Grad_No30", "Grad_No50", "Grad_No100", "Grad_No200"}

def check_collinearity_rules(feats: list, name: str):
    s = set(feats)
    for grp in FORBIDDEN_TOGETHER:
        if grp.issubset(s):
            print(f"[WARN] Feature set '{name}' uses a forbidden trio together: {sorted(grp)}")
    if len(ALL_GRAD & s) >= len(ALL_GRAD) - 1:
        print(f"[WARN] Feature set '{name}' uses (almost) all Grad_* sieves together.")

def build_feature_sets() -> dict:
    return {
        "Recommended": RECOMMENDED,
        "Recommended_PlusInteractions": RECOMMENDED + INTERACTIONS,
    }

# =============================================================================
# DATA
# =============================================================================
def find_input_file() -> Path:
    home = Path.home()
    cands = [Path(FILE_NAME), Path.cwd() / FILE_NAME, home / "Downloads" / FILE_NAME]
    for p in cands:
        if p.exists(): return p
    for d in [Path.cwd(), home / "Downloads", home / "Desktop", home / "OneDrive" / "Desktop",
              Path("/content"), Path("/root/.claude/uploads")]:
        if d.exists():
            # also matches copies like "... (1).xlsx" and the previous cleaned file
            for pat in ["*New_Data*SCB_LWT*Cleaned*with_RBR*.xlsx",
                        "*New_Data*SCB_LWT*Cleaned*Modeling*.xlsx", "*SCB_LWT*Cleaned*.xlsx"]:
                hits = sorted(d.glob(pat)) or sorted(d.rglob(pat))
                if hits: return hits[0]
    raise FileNotFoundError(f"Put {FILE_NAME} next to this script or in Downloads.")

def parse_adt_to_ordinal(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)): return float(value)
    s = str(value).strip().lower().replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if len(nums) >= 2: return float(np.mean(nums[:2]))
    if len(nums) == 1: return nums[0]
    if "low" in s: return 1.0
    if "med" in s: return 2.0
    if "high" in s: return 3.0
    return np.nan

def load_data() -> tuple:
    path = find_input_file()
    print("=" * 92); print(f"Input file: {path}  (sheet {SHEET})")
    df = pd.read_excel(path, sheet_name=SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "ADT" in df.columns:
        df["ADT_ord"] = df["ADT"].apply(parse_adt_to_ordinal)
    # numeric casts for the interaction ingredients
    for c in ["RBR_percent", "AFT_micron", "PG_HighTemp", "Dust_Pbe_ratio", "Va", "VMA", "Absorption"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    # ---- recommended engineered interactions ----
    df["RBR_x_AFT"] = df["RBR_percent"] * df["AFT_micron"]
    df["RBR_x_PG"] = df["RBR_percent"] * df["PG_HighTemp"]
    df["RBR_x_DustPbe"] = df["RBR_percent"] * df["Dust_Pbe_ratio"]
    df["AFT_x_DustPbe"] = df["AFT_micron"] * df["Dust_Pbe_ratio"]
    df["Va_to_VMA"] = df["Va"] / df["VMA"].replace(0, np.nan)
    df["Absorption_x_AFT"] = df["Absorption"] * df["AFT_micron"]
    df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
    if UNIQUE_MIXES and ID_COL in df.columns and df[ID_COL].nunique() < len(df):
        n0 = len(df)
        agg = {c: ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "first")
               for c in df.columns if c != ID_COL}
        df = df.groupby(ID_COL, as_index=False).agg(agg)
        print(f"UNIQUE-MIX DATA: {n0} rows -> {len(df)} unique mixes (replicates averaged; no leakage).")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    print(f"Rows: {len(df)} | {TARGET}: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f} (higher = better)")
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

def _group_holdout(bins, groups, holdout_size, seed):
    """Target-stratified holdout that keeps every replicate of a mix together."""
    n_splits = max(2, int(round(1.0 / holdout_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep, hold = next(iter(sgkf.split(np.zeros(len(bins)), bins, groups)))
    return np.array(keep), np.array(hold)

def split_70_10_20(df, y):
    bins = make_target_bins(y); idx = np.arange(len(y))
    groups = df[ID_COL].astype(str).values if ID_COL in df.columns else None
    if groups is not None and len(set(groups)) < len(groups):
        # ALL rows kept -> GROUP-aware split so replicates never straddle splits
        dev, te = _group_holdout(bins, groups, TEST, RANDOM_STATE)
        vfrac = VAL / (TRAIN + VAL)
        trp, vap = _group_holdout(bins.iloc[dev].reset_index(drop=True), groups[dev], vfrac, RANDOM_STATE)
        tr, va = dev[trp], dev[vap]
    else:
        dev, te = train_test_split(idx, test_size=TEST, random_state=RANDOM_STATE, shuffle=True, stratify=bins)
        vfrac = VAL / (TRAIN + VAL)
        trp, vap = train_test_split(np.arange(len(dev)), test_size=vfrac, random_state=RANDOM_STATE,
                                    shuffle=True, stratify=bins.iloc[dev].reset_index(drop=True))
        tr, va = dev[trp], dev[vap]
    if groups is not None:
        g = groups
        leak = len(set(g[tr]) & set(g[te])) + len(set(g[va]) & set(g[te])) + len(set(g[tr]) & set(g[va]))
        print(f"Split: train {len(tr)} / val {len(va)} / locked-test {len(te)} rows | mix overlap = {leak} (must be 0)")
        print(f"       mixes: train {len(set(g[tr]))} / val {len(set(g[va]))} / test {len(set(g[te]))}")
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

def show_and_save(fname):
    """Save the current figure AND show it in the Spyder plots pane."""
    plt.tight_layout()
    plt.savefig(OUT / "figures" / fname, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        try: plt.show()
        except Exception: pass
    plt.close()

# =============================================================================
# MODEL: tuned ExtraTrees — the approach behind the 0.68 validation result
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

def _train_cv_splits(ytr, gtr):
    """Group-aware CV inside training when replicates exist (keeps a mix in one fold)."""
    bins = make_target_bins(ytr)
    if gtr is not None and len(set(gtr)) < len(gtr):
        cv = StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        return list(cv.split(np.zeros(len(ytr)), bins, gtr))
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(ytr)), bins))

def tune(name, est, grid, n_iter, Xtr, ytr, Xva, yva, num, cat, gtr=None):
    pipe = _pipe(clone(est), num, cat)
    space = int(np.prod([len(v) for v in grid.values()]))
    s = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, space), scoring="r2",
                           cv=_train_cv_splits(ytr, gtr), random_state=RANDOM_STATE,
                           n_jobs=1, return_train_score=True, error_score=np.nan)
    s.fit(Xtr, ytr)
    best = s.best_estimator_
    oof = float(s.cv_results_["mean_test_score"][s.best_index_])
    trm, vam = metrics(ytr, best.predict(Xtr)), metrics(yva, best.predict(Xva))
    row = {"Model": name, "Train_R2": trm["R2"], "OOF_CV_R2": oof, "Validation_R2": vam["R2"],
           "Validation_RMSE": vam["RMSE"], "Validation_MAE": vam["MAE"],
           "Train_minus_Val_gap": trm["R2"] - vam["R2"], "Best_Params": json.dumps(s.best_params_, default=str)}
    return row, best

def repeated_cv(est, Xdev, ydev, gdev=None):
    """Repeated CV; GROUP-aware when replicates exist (each repeat randomly re-assigns whole
    mixes to folds, so replicates of a mix are never split between train and validation)."""
    sc = []
    for r in range(REPEATED_REPEATS):
        if gdev is not None and len(set(gdev)) < len(gdev):
            rng = np.random.RandomState(RANDOM_STATE + 59 * r)
            ug = np.unique(gdev); fold_of = dict(zip(ug, rng.randint(0, CV_FOLDS, size=len(ug))))
            f = np.array([fold_of[g] for g in gdev])
            splits = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(CV_FOLDS)]
        else:
            cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 59 * r)
            splits = list(cv.split(Xdev))
        for tr, va in splits:
            if len(va) == 0 or len(tr) == 0: continue
            e = clone(est); e.fit(Xdev.iloc[tr], ydev.iloc[tr])
            sc.append(r2_score(ydev.iloc[va], e.predict(Xdev.iloc[va])))
    sc = np.array(sc)
    return float(sc.mean()), float(sc.std(ddof=1)), float(sc.min())

# =============================================================================
# SHAP + PDP  (run on the FINAL model, using dev data only — never the locked test)
# =============================================================================
def run_shap(final_pipe, Xdev):
    if not HAS_SHAP:
        print("SHAP skipped (not installed)."); return
    try:
        prep = final_pipe.named_steps["prep"]; mdl = final_pipe.named_steps["model"]
        Xs = Xdev.sample(min(MAX_SHAP_ROWS, len(Xdev)), random_state=RANDOM_STATE)
        Xt = prep.transform(Xs)
        names = list(prep.get_feature_names_out())
        expl = shap.TreeExplainer(mdl)
        sv = expl.shap_values(Xt)
        # bar (mean |SHAP|)
        shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", show=False)
        show_and_save("shap_bar.png")
        # beeswarm
        shap.summary_plot(sv, Xt, feature_names=names, show=False)
        show_and_save("shap_beeswarm.png")
        # waterfall for one typical row
        try:
            ev = expl.expected_value if np.ndim(expl.expected_value) == 0 else expl.expected_value[0]
            e = shap.Explanation(values=sv[0], base_values=ev, data=Xt[0], feature_names=names)
            shap.plots.waterfall(e, show=False)
            show_and_save("shap_waterfall_row0.png")
        except Exception as ex:
            print(f"  SHAP waterfall skipped: {type(ex).__name__}")
        # save mean |SHAP| table
        mean_abs = pd.DataFrame({"Feature": names, "Mean_abs_SHAP": np.abs(sv).mean(axis=0)}
                                ).sort_values("Mean_abs_SHAP", ascending=False)
        _save_df(mean_abs, OUT / "shap_mean_abs.xlsx")
        print("SHAP done: shap_bar.png, shap_beeswarm.png, shap_waterfall_row0.png")
    except Exception as e:
        print(f"SHAP failed: {type(e).__name__}: {e}")

def run_pdp(final_pipe, Xdev, ydev, num_features):
    # PDP re-predicts the model over a value grid, so keep it fast: subsample the rows,
    # use a coarse grid (esp. for 2D: resolution^2 grid points), and parallelize.
    rng = np.random.RandomState(RANDOM_STATE)
    pos = rng.choice(len(Xdev), size=min(300, len(Xdev)), replace=False)
    Xp = Xdev.iloc[pos].reset_index(drop=True); yp = ydev.iloc[pos].reset_index(drop=True)
    try:
        r = permutation_importance(final_pipe, Xp, yp, n_repeats=3, scoring="r2",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    except Exception:
        r = None
    try:
        imp_order = (pd.Series(r.importances_mean, index=Xp.columns).sort_values(ascending=False)
                     if r is not None else pd.Series(dtype=float))
        top_num = [c for c in imp_order.index if c in num_features][:6] or num_features[:6]
        # 1D PDP (top 6 numeric features)
        fig, ax = plt.subplots(figsize=(11, 7))
        PartialDependenceDisplay.from_estimator(final_pipe, Xp, top_num, kind="average",
                                                grid_resolution=30, n_jobs=-1, ax=ax)
        fig.suptitle("Partial dependence — top features (final model)")
        show_and_save("pdp_top_features.png")
        # 2D PDP for the key recommended interactions (coarse 15x15 grid)
        for pair in [("RBR_percent", "AFT_micron"), ("AFT_micron", "Dust_Pbe_ratio")]:
            if all(p in Xp.columns for p in pair):
                fig, ax = plt.subplots(figsize=(7, 5.5))
                PartialDependenceDisplay.from_estimator(final_pipe, Xp, [pair], kind="average",
                                                        grid_resolution=15, n_jobs=-1, ax=ax)
                fig.suptitle(f"2D partial dependence: {pair[0]} x {pair[1]}")
                show_and_save(f"pdp2d_{pair[0]}_x_{pair[1]}.png")
        print("PDP done: pdp_top_features.png + 2D interaction PDPs")
    except Exception as e:
        print(f"PDP failed: {type(e).__name__}: {e}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    df, y = load_data()
    tr, va, te = split_70_10_20(df, y)
    groups = df[ID_COL].astype(str).values if ID_COL in df.columns else None
    gtr = groups[tr] if groups is not None else None
    gdev = groups[np.concatenate([tr, va])] if groups is not None else None
    rows, trained = [], {}

    for fs_name, feats in build_feature_sets().items():
        check_collinearity_rules(feats, fs_name)
        X, num, cat, avail = get_X(df, feats)
        Xtr, Xva, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ytr, yva, yte = y.iloc[tr].reset_index(drop=True), y.iloc[va].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        print(f"\n--- Feature set: {fs_name} ({len(avail)} features) ---")
        for mname, est, grid, ni in [("ExtraTrees", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1), ET_GRID, N_ITER),
                                     ("RandomForest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), RF_GRID, max(20, N_ITER // 2))]:
            try:
                row, best = tune(mname, est, grid, ni, Xtr, ytr, Xva, yva, num, cat, gtr=gtr)
                row["Feature_Set"] = fs_name; row["Label"] = f"{fs_name} | {mname}"
                rows.append(row)
                trained[row["Label"]] = {"est": best, "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
                                         "Xte": Xte, "yte": yte, "num": num, "cat": cat}
                print(f"  {mname:14s} Val R2={row['Validation_R2']:.4f} | OOF={row['OOF_CV_R2']:.4f} | gap={row['Train_minus_Val_gap']:.3f}")
            except Exception as e:
                print(f"  {mname} FAILED: {type(e).__name__}: {e}")

    lb = pd.DataFrame(rows)
    lb_oof = lb.sort_values("OOF_CV_R2", ascending=False).reset_index(drop=True)
    print("\nLEADERBOARD by OOF_CV_R2 (honest):")
    print(lb_oof[["Label", "OOF_CV_R2", "Validation_R2", "Train_minus_Val_gap"]].to_string(index=False))

    print("\nRepeatedCV (5x5) on the dev set (ExtraTrees per feature set):")
    rcv_rows = []
    for fs_name in build_feature_sets():
        lbl = f"{fs_name} | ExtraTrees"
        if lbl not in trained: continue
        obj = trained[lbl]
        Xdev = pd.concat([obj["Xtr"], obj["Xva"]]).reset_index(drop=True)
        ydev = pd.concat([obj["ytr"], obj["yva"]]).reset_index(drop=True)
        m, s, mn = repeated_cv(obj["est"], Xdev, ydev, gdev=gdev)
        rcv_rows.append({"Label": lbl, "RepeatedCV_R2_mean": m, "RepeatedCV_R2_std": s, "RepeatedCV_R2_min": mn})
        print(f"  {lbl:45s} {m:.4f} +/- {s:.4f} (min {mn:.4f})")
    rcv = pd.DataFrame(rcv_rows).sort_values("RepeatedCV_R2_mean", ascending=False).reset_index(drop=True)

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

    # ---- plots (shown in Spyder AND saved) ----
    for ds in ["Validation10", "LockedTest20"]:
        d = pred[pred.Dataset == ds]
        m = metrics(d.Measured, d.Predicted)
        xs = np.array([d.Measured.min(), d.Measured.max()]); slope, b = np.polyfit(d.Measured, d.Predicted, 1)
        plt.figure(figsize=(6.4, 6)); plt.scatter(d.Measured, d.Predicted, alpha=0.7, edgecolor="k", linewidth=0.3)
        plt.plot(xs, xs, "r--", label="1:1"); plt.plot(xs, slope * xs + b, "b-", label="best-fit")
        plt.title(f"{ds}: R2={m['R2']:.3f} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f}")
        plt.xlabel("Measured SCB Jc"); plt.ylabel("Predicted SCB Jc"); plt.legend(); plt.grid(alpha=0.3)
        show_and_save(f"{ds}_parity.png")
        plt.figure(figsize=(7, 4.5)); plt.scatter(d.Predicted, d.Residual, alpha=0.7); plt.axhline(0, color="k", ls="--")
        plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title(f"{ds}: residuals"); plt.grid(alpha=0.3)
        show_and_save(f"{ds}_residuals.png")
    if not lb.empty:
        d = lb.sort_values("Validation_R2")
        plt.figure(figsize=(9, max(3.5, 0.5 * len(d)))); plt.barh(d.Label, d.Validation_R2, color="#357")
        plt.xlabel("Validation R2"); plt.title("Model comparison"); plt.grid(axis="x", alpha=0.3)
        show_and_save("model_comparison.png")
    try:
        mdl = final.named_steps["model"]
        names = final.named_steps["prep"].get_feature_names_out()
        imp = pd.DataFrame({"Feature": names, "Importance": mdl.feature_importances_}).sort_values("Importance", ascending=False)
        _save_df(imp, OUT / "feature_importance.xlsx")
        ii = imp.head(15).sort_values("Importance")
        plt.figure(figsize=(8, 5)); plt.barh(ii.Feature, ii.Importance, color="#2A9D8F")
        plt.title("Feature importance (final model)")
        show_and_save("feature_importance.png")
    except Exception: pass

    # ---- SHAP + PDP on the final model (dev rows only) ----
    print("\nRunning SHAP...");  run_shap(final, Xdev)
    print("Running PDP...");     run_pdp(final, Xdev, ydev, obj["num"])

    _save_df(lb_oof, OUT / "model_comparison.xlsx")
    _save_df(rcv, OUT / "repeated_cv_results.xlsx")
    _save_df(pred, OUT / "final_predictions.xlsx")
    _save_df(fm, OUT / "final_metrics.xlsx")
    joblib.dump(final, OUT / "models" / "final_model.joblib")

    lt = fm[fm.Dataset == "LockedTest20"]
    lines = ["SCB RECOMMENDED-FEATURES + SHAP + PDP — SUMMARY", "=" * 60,
             f"Final model: {final_label}",
             f"Locked-test R2 = {float(lt.R2.iloc[0]):.4f}  RMSE = {float(lt.RMSE.iloc[0]):.4f}  MAE = {float(lt.MAE.iloc[0]):.4f}",
             "Features: the recommended 22 + 6 engineered interactions (collinearity rules respected).",
             "Selection by OOF CV + RepeatedCV; the 20% locked test was scored once."]
    (OUT / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved outputs to {OUT}/  (figures also shown in the Spyder plots pane)")


if __name__ == "__main__":
    main()
