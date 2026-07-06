# -*- coding: utf-8 -*-
"""
RUT_20k — TabPFN (foundation model) STRONG + TUNED WORKFLOW
===========================================================
TabArena shows TabPFN is the top model for small tabular data (your rutting set = 1898 rows).
TabPFN is PRETRAINED, so it is not hyper-parameter tuned the usual way; you make it STRONGER by:
  * a bigger internal ensemble  (n_estimators; default 8 for regression -> use 16/32),
  * optional POST-HOC ENSEMBLING (AutoTabPFNRegressor from tabpfn-extensions), which blends many
    TabPFN configurations under a time budget.

This script:
  * loads the rutting data (Rutting_Cleaned_with_RBR.xlsx), engineers true RBR,
  * makes a target-stratified 75/25 split (train / one-time test),
  * fits TabPFN (n_estimators = TABPFN_ENSEMBLE) and, if available, AutoTabPFN (post-hoc),
  * benchmarks a tuned XGBoost and a TabPFN+XGBoost average on the SAME split,
  * reports the honest ladder (RepeatedCV + one-time test) and a parity plot.

ONE-TIME TabPFN license (Prior Labs, NOT Hugging Face):
  1) pip install tabpfn           (optional stronger: pip install tabpfn-extensions)
  2) open https://ux.priorlabs.ai , log in, accept the license on the "Licenses" tab
  3) copy your API key from https://ux.priorlabs.ai/account
  4) Anaconda Prompt ->  setx TABPFN_TOKEN "your-key-here"   then restart Spyder
If TabPFN is unavailable/unauthenticated, the script still runs the XGBoost baseline.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# ---- optional deps ----
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def _detect_device(pref="auto"):
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# ---- TabPFN backend ----
# "client" = CLOUD API (easiest: pip install tabpfn-client; runs on Prior Labs servers, no local
#            weights download). "local" = local weights (pip install tabpfn; needs the license).
TABPFN_BACKEND = "client"
# Paste your token here (from https://ux.priorlabs.ai/account, starts with tabpfn_sk_...). Leave
# empty to instead read env var TABPFN_TOKEN, or (client) get an interactive login prompt.
# SECURITY: do not commit a real token to git.
TABPFN_API_TOKEN = r""

import os as _os
TabPFNRegressor = None
AutoTabPFNRegressor = None
_BACKEND = None

def _apply_token():
    tok = TABPFN_API_TOKEN or _os.environ.get("TABPFN_TOKEN", "")
    return tok

if TABPFN_BACKEND == "client":
    try:
        from tabpfn_client import TabPFNRegressor as _TabPFN, set_access_token as _set_tok
        _tok = _apply_token()
        if _tok:
            try:
                _set_tok(_tok)
            except Exception as _e:
                print(f"tabpfn_client token not set ({type(_e).__name__}); you may get a login prompt.")
        TabPFNRegressor = _TabPFN
        _BACKEND = "client"
    except Exception:
        TabPFNRegressor = None

if TabPFNRegressor is None:      # fall back to local weights
    try:
        from tabpfn import TabPFNRegressor as _TabPFN
        TabPFNRegressor = _TabPFN
        _BACKEND = "local"
        try:
            try:
                from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor as _Auto
            except Exception:
                from tabpfn_extensions import AutoTabPFNRegressor as _Auto
            AutoTabPFNRegressor = _Auto
        except Exception:
            AutoTabPFNRegressor = None
    except Exception:
        TabPFNRegressor = None

# =============================================================================
# SETTINGS
# =============================================================================
RANDOM_STATE = 42
TARGET, UNITS = "Rut_20k", "mm"
TEST_SIZE, CV_FOLDS = 0.25, 5
TABPFN_ENSEMBLE = 32          # bigger internal ensemble = stronger TabPFN (default 8)
TABPFN_DEVICE = "auto"        # "auto" -> cuda if available else cpu
USE_AUTO_TABPFN = True        # use AutoTabPFNRegressor (post-hoc) if tabpfn-extensions is installed
AUTO_TABPFN_MAX_TIME = 120    # seconds budget for the post-hoc ensemble search
REPEATED_K, REPEATED_REPEATS = 10, 3     # honest RepeatedCV (kept modest: TabPFN refits each fold)
N_ITER_XGB = 60               # XGBoost RandomizedSearch budget (baseline)

# The CLOUD client makes an API call per fit/predict, so heavy RepeatedCV can hit rate limits.
# When True and the backend is "client", TabPFN uses a single light 5-fold CV instead of 10x3.
TABPFN_CLIENT_LIGHT_CV = True

OLD_FILE = r""                # paste full path to Rutting_Cleaned_with_RBR.xlsx if auto-find fails
RUT_FEATURES = ["ADT_DOTD_ord", "PG Grade", "RAP_pct", "ACinRAP", "Pass_4.75mm", "Va", "VMA",
                "Dust_Binder", "Gmm", "SandEq", "FAA", "NMAS", "Absorption"]
ALIASES = {
    "ADT_DOTD_ord": ["ADT_DOTD_ord", "ADT_ord", "ADT_enc"], "PG Grade": ["PG Grade", "PG_HighTemp", "PG_Grade"],
    "RAP_pct": ["RAP_pct", "RAP"], "ACinRAP": ["ACinRAP"], "Pass_4.75mm": ["Pass_4.75mm", "Pass4_75mm"],
    "Va": ["Va", "AirVoids"], "VMA": ["VMA"], "Dust_Binder": ["Dust_Binder", "DustBinder"], "Gmm": ["Gmm"],
    "SandEq": ["SandEq"], "FAA": ["FAA"], "NMAS": ["NMAS", "NMAS (mm)"], "Absorption": ["Absorption"],
    "AsphaltContent_Design": ["AsphaltContent_Design", "AC_Design"], "Rut_20k": ["Rut_20k", "Rut20k"],
}
_DESK = Path.home() / "Desktop"; _ONEDESK = Path.home() / "OneDrive" / "Desktop"
OLD_CANDIDATES = [Path("Rutting_Cleaned_with_RBR.xlsx"), Path.home() / "Downloads" / "Rutting_Cleaned_with_RBR.xlsx",
                  _DESK / "Rutting_Cleaned_with_RBR.xlsx", _ONEDESK / "Rutting_Cleaned_with_RBR.xlsx",
                  Path("/tmp/claude-0/-home-user-asphalt1/2de3e2fd-e268-5a6b-af7f-ada01b8f275d/scratchpad/Rutting_Cleaned_with_RBR.xlsx")]
OLD_PATTERNS = ["*Rutting_Cleaned_with_RBR*.xlsx", "*Rutting_Cleaned*.xlsx"]
OUT = Path("Rut_TabPFN_Outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)


def resolve(cands, patterns):
    for p in cands:
        try:
            if p.exists(): return p
        except Exception: pass
    dirs = [Path.cwd(), Path.home() / "Downloads", _DESK, _ONEDESK]
    for d in dirs:
        if d.exists():
            for pat in patterns:
                hits = sorted(d.glob(pat))
                if hits: return hits[0]
    for d in dirs:
        if d.exists():
            for pat in patterns:
                try: hits = sorted(d.rglob(pat))
                except Exception: hits = []
                if hits: return hits[0]
    return None

def read_best_sheet(path):
    xl = pd.ExcelFile(path)
    for s in xl.sheet_names:
        if "lean" in s.lower(): return pd.read_excel(path, sheet_name=s)
    return pd.read_excel(path, sheet_name=xl.sheet_names[0])

def harmonize(df):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    out = {}
    for canon, names in ALIASES.items():
        for nm in names:
            if nm in df.columns: out[canon] = df[nm]; break
    h = pd.DataFrame(out)
    for c in ["RAP_pct", "ACinRAP", "AsphaltContent_Design", "Rut_20k"]:
        if c in h.columns: h[c] = pd.to_numeric(h[c], errors="coerce")
    if {"RAP_pct", "ACinRAP", "AsphaltContent_Design"}.issubset(h.columns):
        h["RBR_JMF_fraction"] = ((h["RAP_pct"] * h["ACinRAP"] / 100.0) / h["AsphaltContent_Design"]).replace([np.inf, -np.inf], np.nan)
    return h

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p))}

def num_pipe(est, feats):
    return Pipeline([("prep", ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", MinMaxScaler())]), feats)], remainder="drop")), ("model", est)])

def build_tabpfn(feats):
    """Return (pipeline, label) for the strongest available TabPFN.
    CLIENT (cloud) backend: TabPFNRegressor() runs on Prior Labs servers (no device arg).
    LOCAL backend: post-hoc AutoTabPFN if tabpfn-extensions is present, else plain TabPFN with a
    larger internal ensemble on the detected device."""
    if _BACKEND == "client":
        for kw in ({"n_estimators": TABPFN_ENSEMBLE}, {}):     # cloud may/ may not accept n_estimators
            try:
                lab = f"TabPFN-cloud(n_estimators={kw.get('n_estimators', 'def')})"
                return num_pipe(TabPFNRegressor(**kw), feats), lab
            except TypeError:
                continue
        return num_pipe(TabPFNRegressor(), feats), "TabPFN-cloud"
    # local backend
    dev = _detect_device(TABPFN_DEVICE)
    if USE_AUTO_TABPFN and AutoTabPFNRegressor is not None:
        try:
            return num_pipe(AutoTabPFNRegressor(max_time=AUTO_TABPFN_MAX_TIME, device=dev), feats), "AutoTabPFN(post-hoc)"
        except Exception:
            pass
    for kw in ({"n_estimators": TABPFN_ENSEMBLE, "device": dev, "random_state": RANDOM_STATE},
               {"n_estimators": TABPFN_ENSEMBLE, "device": dev},
               {"device": dev}):
        try:
            return num_pipe(TabPFNRegressor(**kw), feats), f"TabPFN(n_estimators={kw.get('n_estimators', 'def')})"
        except TypeError:
            continue
    raise RuntimeError("Could not construct TabPFNRegressor")

def make_xgb():
    return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                        random_state=RANDOM_STATE, n_jobs=-1)

XGB_GRID = {"model__n_estimators": [400, 700, 1000], "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.01, 0.02, 0.04], "model__subsample": [0.6, 0.8],
            "model__colsample_bytree": [0.5, 0.7, 0.9], "model__min_child_weight": [5, 12, 20],
            "model__reg_lambda": [2, 10, 40], "model__gamma": [0.0, 0.1, 0.2]}

def repeated_cv(est, X, y, k=REPEATED_K, reps=REPEATED_REPEATS):
    sc = []
    for r in range(reps):
        cv = KFold(k, shuffle=True, random_state=RANDOM_STATE + 41 * r)
        for tr, va in cv.split(X):
            e = clone(est); e.fit(X.iloc[tr], y.iloc[tr]); sc.append(r2_score(y.iloc[va], e.predict(X.iloc[va])))
    sc = np.array(sc); return {"Mean": float(sc.mean()), "SD": float(sc.std(ddof=1)), "Min": float(sc.min())}

def parity(y, p, title, path):
    m = metrics(y, p); s, b = np.polyfit(np.asarray(y, float), np.asarray(p, float), 1)
    xs = np.array([float(min(np.min(y), np.min(p))), float(max(np.max(y), np.max(p)))])
    plt.figure(figsize=(6.5, 6)); plt.scatter(y, p, alpha=0.6, edgecolor="k", linewidth=0.3, color="#2878B5")
    plt.plot(xs, xs, "r--", lw=2, label="1:1"); plt.plot(xs, s * xs + b, color="#173F5F", lw=2, label="Best-fit")
    plt.title(f"{title}\nR2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
    plt.xlabel(f"Measured {TARGET} ({UNITS})"); plt.ylabel(f"Predicted {TARGET} ({UNITS})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# =============================================================================
def main():
    old_p = Path(OLD_FILE) if OLD_FILE and Path(OLD_FILE).exists() else resolve(OLD_CANDIDATES, OLD_PATTERNS)
    if old_p is None:
        raise FileNotFoundError("Could not find Rutting_Cleaned_with_RBR.xlsx. Put it next to the script "
                                "or paste its full path into OLD_FILE at the top.")
    print("=" * 92); print(f"RUT TabPFN | file: {old_p}")
    print(f"TabPFN available: {TabPFNRegressor is not None} | backend: {_BACKEND} | "
          f"AutoTabPFN: {AutoTabPFNRegressor is not None} | device={_detect_device(TABPFN_DEVICE)} | XGBoost={HAS_XGB}")
    df = harmonize(read_best_sheet(old_p))
    df = df.loc[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    feats = [c for c in RUT_FEATURES if c in df.columns] + (["RBR_JMF_fraction"] if "RBR_JMF_fraction" in df.columns else [])
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    print(f"Rows: {len(df)} | Features ({len(feats)}): {feats}"); print("=" * 92)

    strata = pd.qcut(y, q=5, labels=False, duplicates="drop")
    itr, ite = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=strata)
    Xtr, Xte = X.iloc[itr].reset_index(drop=True), X.iloc[ite].reset_index(drop=True)
    ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
    print(f"Split: train {len(itr)} / test {len(ite)} (stratified 75/25)\n")

    rows, preds = [], {}

    # ---- TabPFN (strong) ----
    if TabPFNRegressor is not None:
        try:
            tp, tp_label = build_tabpfn(feats)
            tp.fit(Xtr, ytr)
            p_te = tp.predict(Xte); preds["TabPFN"] = p_te
            tr_m, te_m = metrics(ytr, tp.predict(Xtr)), metrics(yte, p_te)
            if _BACKEND == "client" and TABPFN_CLIENT_LIGHT_CV:
                print("  (cloud backend: using a light single 5-fold CV to limit API calls)")
                rc = repeated_cv(tp, Xtr, ytr, k=CV_FOLDS, reps=1)
            else:
                rc = repeated_cv(tp, Xtr, ytr)
            rows.append({"Model": tp_label, "Train_R2": tr_m["R2"], "RepeatedCV_R2": rc["Mean"], "RepeatedCV_SD": rc["SD"],
                         "Test_R2": te_m["R2"], "Test_RMSE": te_m["RMSE"], "Overfit_Gap": tr_m["R2"] - rc["Mean"]})
            print(f"  {tp_label:26s} Train={tr_m['R2']:.3f} | RepeatedCV={rc['Mean']:.3f}±{rc['SD']:.3f} | Test={te_m['R2']:.3f}")
            parity(yte, p_te, f"Rut test — {tp_label}", OUT / "figures" / "tabpfn_test_parity.png")
        except Exception as e:
            print(f"  TabPFN SKIPPED ({type(e).__name__}): {str(e).splitlines()[0]}")
            print("  -> license needed: https://ux.priorlabs.ai (accept license), copy API key from /account,\n"
                  "     then  setx TABPFN_TOKEN \"your-key\"  and restart Spyder.")
    else:
        print("  TabPFN not installed -> pip install tabpfn (and optionally tabpfn-extensions).")

    # ---- XGBoost baseline (tuned) ----
    if HAS_XGB:
        s = RandomizedSearchCV(num_pipe(make_xgb(), feats), XGB_GRID, n_iter=N_ITER_XGB, scoring="r2",
                               cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                               random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan)
        s.fit(Xtr, ytr); xgb = s.best_estimator_
        p_te = xgb.predict(Xte); preds["XGBoost"] = p_te
        tr_m, te_m = metrics(ytr, xgb.predict(Xtr)), metrics(yte, p_te); rc = repeated_cv(xgb, Xtr, ytr)
        rows.append({"Model": "XGBoost(tuned)", "Train_R2": tr_m["R2"], "RepeatedCV_R2": rc["Mean"], "RepeatedCV_SD": rc["SD"],
                     "Test_R2": te_m["R2"], "Test_RMSE": te_m["RMSE"], "Overfit_Gap": tr_m["R2"] - rc["Mean"]})
        print(f"  {'XGBoost(tuned)':26s} Train={tr_m['R2']:.3f} | RepeatedCV={rc['Mean']:.3f}±{rc['SD']:.3f} | Test={te_m['R2']:.3f}")

    # ---- TabPFN + XGBoost average blend ----
    if "TabPFN" in preds and "XGBoost" in preds:
        blend = 0.5 * preds["TabPFN"] + 0.5 * preds["XGBoost"]
        te_m = metrics(yte, blend)
        rows.append({"Model": "Blend TabPFN+XGB (0.5/0.5)", "Train_R2": np.nan, "RepeatedCV_R2": np.nan,
                     "RepeatedCV_SD": np.nan, "Test_R2": te_m["R2"], "Test_RMSE": te_m["RMSE"], "Overfit_Gap": np.nan})
        print(f"  {'Blend TabPFN+XGB':26s} Test={te_m['R2']:.3f}")

    res = pd.DataFrame(rows)
    if res.empty:
        print("\nNo model ran. Install tabpfn and/or xgboost."); return
    res = res.sort_values("Test_R2", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 92); print("RESULTS (sorted by test R2):"); print(res.to_string(index=False)); print("=" * 92)
    print(f"\nBest test R2: {res['Test_R2'].max():.3f}  |  prior rutting locked test was ~0.58 (row-level).")

    with pd.ExcelWriter(OUT / "Rut_TabPFN_Results.xlsx", engine="openpyxl") as w:
        res.to_excel(w, sheet_name="Models", index=False)
        pd.DataFrame({"Feature": feats}).to_excel(w, sheet_name="Features", index=False)
    print(f"Saved: {OUT/'Rut_TabPFN_Results.xlsx'} | figures in {OUT/'figures'}")


if __name__ == "__main__":
    main()
