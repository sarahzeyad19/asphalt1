# -*- coding: utf-8 -*-
"""
SCB (Jc) — ANTI-OVERFIT + FULL PLOTS COMPANION
==============================================
Additive companion focused on the two asks: (1) reduce the large train-vs-CV overfit gap and push
the honest score up, and (2) produce model plots/graphs. Your original code is not modified.

Anti-overfitting levers (all standard, see sources in the chat):
  * GAP-PENALIZED Optuna objective:  score = CV_mean - OVERFIT_PENALTY * max(0, Train - CV)
      -> Optuna is rewarded for hyper-parameters that GENERALIZE, not ones that memorize.
  * Early stopping for HistGB (built-in internal validation).
  * Wide L1/L2 + shallow trees + high min_samples_leaf / min_child_weight search ranges.
  * Feature selection: keep the top-K stable RandomForest-importance features (drops noise).
  * Seed-bagged final model (average of BAG_SEEDS fits) -> variance reduction.

Plots written to figures/:
  1. overfit_gap.png        Train vs RepeatedCV vs Test R2 per model (the overfit picture)
  2. cv_fold_boxplot.png    RepeatedCV fold-R2 distribution per model (stability)
  3. learning_curve.png     Train vs CV R2 vs training-set size for the best model
  4. parity_best.png        Measured vs predicted (train + test) for the best model
  5. residuals_best.png     Residual vs predicted (test) for the best model
  6. importance.png         Feature importance of the best model
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

from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, cross_val_score, cross_val_predict,
                                     learning_curve, train_test_split)
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
    from tabpfn import TabPFNRegressor          # pip install tabpfn  (foundation model for small tables)
    HAS_TABPFN = True
except Exception:
    HAS_TABPFN = False

# ---------------- settings ----------------
RANDOM_STATE = 42
TARGET, UNITS = "SCB", "kJ/m2"
TEST_SIZE, CV_FOLDS = 0.25, 5
N_TRIALS = 150
REPEATED_K, REPEATED_REPEATS = 10, 5
USE_TABPFN = True               # add TabPFN as a candidate + stacking base (no tuning needed)
TABPFN_DEVICE = "auto"          # "auto" -> cuda if available else cpu; or force "cuda"/"cpu"
# TabPFN one-time setup (this tabpfn build uses PRIOR LABS auth, not Hugging Face):
#   1) pip install tabpfn
#   2) open https://ux.priorlabs.ai , log in / register, accept the license on the "Licenses" tab
#   3) copy your API key from https://ux.priorlabs.ai/account
#   4) set it once:  Anaconda Prompt ->  setx TABPFN_TOKEN "your-key-here"   then restart Spyder
#      (or in Python before running:  import os; os.environ["TABPFN_TOKEN"] = "your-key-here")
#   If TabPFN is absent or unauthenticated, the script now SKIPS it and runs everything else.
OVERFIT_PENALTY = 0.15          # weight on the train-CV gap (0 = pure CV; higher = simpler/less overfit).
                                # 0.10-0.20 shrinks the gap without crushing the score; 0.30+ is aggressive.
FEATURE_SELECTION, TOP_K = True, 12
BAG_SEEDS = 5                   # seed-bagged final model
PAPER14 = ["AsphaltContent_Design", "Va", "VMA", "Dust_Binder", "Pass_0.075mm", "Gsb", "Gmm",
           "Pass_4.75mm", "CAA", "SandEq", "RAP_pct", "ACinRAP", "PG Grade", "ADT_DOTD_ord"]
INPUT_CANDIDATES = [
    Path("Final_Cleaned_590Dataset_User_Removal_Rules.xlsx"),
    Path.home() / "Downloads" / "Final_Cleaned_590Dataset_User_Removal_Rules.xlsx",
    Path("/root/.claude/uploads/2de3e2fd-e268-5a6b-af7f-ada01b8f275d/b2f34a91-Final_Cleaned_590Dataset_User_Removal_Rules.xlsx"),
]
SHEET = "Cleaned_Data_Kept"
OUT = Path("SCB_AntiOverfit_Outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})


def resolve_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Put the 590 Excel next to the script.")

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}

def make_pipe(est, feats):
    return Pipeline([("prep", ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]), feats)],
        remainder="drop")), ("model", est)])


class SeedBaggingRegressor(BaseEstimator, RegressorMixin):
    """Average of the same pipeline fit under several seeds -> lower variance / less overfit."""
    def __init__(self, base, seeds): self.base, self.seeds = base, seeds
    def fit(self, X, y):
        self.models_ = []
        for s in self.seeds:
            m = clone(self.base)
            for step in ("model",):
                est = m.named_steps[step]
                if hasattr(est, "random_state"): est.set_params(random_state=s)
            self.models_.append(m.fit(X, y))
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def suggest(trial, name):
    if name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            max_features=trial.suggest_float("max_features", 0.3, 0.9),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 5, 25))
    if name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            max_features=trial.suggest_float("max_features", 0.3, 0.9),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20))
    if name == "HistGB":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error",
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            max_iter=trial.suggest_int("max_iter", 300, 1000, step=100),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 6, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 15, 45),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-2, 20.0, log=True))
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1,
            n_estimators=trial.suggest_int("n_estimators", 300, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 0.85),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.85),
            min_child_weight=trial.suggest_int("min_child_weight", 5, 30),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 60.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-2, 0.5, log=True))
    raise ValueError(name)


def tune(name, Xtr, ytr, feats):
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    def gap_penalized(est):
        pipe = make_pipe(est, feats)
        cvm = float(np.mean(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="r2", n_jobs=1)))
        pipe.fit(Xtr, ytr)
        train = r2_score(ytr, pipe.predict(Xtr))
        return cvm - OVERFIT_PENALTY * max(0.0, train - cvm)     # reward generalization
    if HAS_OPTUNA:
        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        st.optimize(lambda t: gap_penalized(suggest(t, name)), n_trials=N_TRIALS, show_progress_bar=False)
        best = make_pipe(suggest(optuna.trial.FixedTrial(st.best_params), name), feats)
    else:
        best = make_pipe(suggest(optuna.trial.FixedTrial({}), name), feats)  # not reached without optuna
    best.fit(Xtr, ytr)
    return best


def repeated_cv_scores(est, X, y):
    sc = []
    for r in range(REPEATED_REPEATS):
        cv = KFold(REPEATED_K, shuffle=True, random_state=RANDOM_STATE + 41 * r)
        for tr, va in cv.split(X):
            e = clone(est); e.fit(X.iloc[tr], y.iloc[tr])
            sc.append(r2_score(y.iloc[va], e.predict(X.iloc[va])))
    return np.array(sc)


# =============================================================================
def main():
    path = resolve_input()
    df = pd.read_excel(path, sheet_name=SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["RAP_pct", "ACinRAP", "AsphaltContent_Design"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["RBR_JMF_fraction"] = ((df["RAP_pct"] * df["ACinRAP"] / 100) / df["AsphaltContent_Design"]).replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[y.notna()].reset_index(drop=True); y = y.loc[y.notna()].reset_index(drop=True)
    feats_all = [c for c in PAPER14 if c in df.columns] + ["RBR_JMF_fraction"]
    X = df[feats_all].apply(pd.to_numeric, errors="coerce")

    strata = pd.qcut(y, q=min(5, y.nunique()), labels=False, duplicates="drop")
    itr, ite = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                shuffle=True, stratify=strata)
    Xtr_all, Xte_all = X.iloc[itr].reset_index(drop=True), X.iloc[ite].reset_index(drop=True)
    ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)

    # ---- feature selection (drop noise) ----
    feats = feats_all
    if FEATURE_SELECTION:
        rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(Xtr_all.fillna(Xtr_all.median()), ytr)
        imp = pd.Series(rf.feature_importances_, index=feats_all).sort_values(ascending=False)
        feats = list(imp.head(TOP_K).index)
    Xtr, Xte = Xtr_all[feats], Xte_all[feats]

    print("=" * 92)
    print(f"SCB ANTI-OVERFIT | rows {len(df)} | features {len(feats)}/{len(feats_all)} "
          f"| Optuna={HAS_OPTUNA}({N_TRIALS}) | gap_penalty={OVERFIT_PENALTY} | bag={BAG_SEEDS}")
    print(f"Selected features: {feats}"); print("=" * 92)

    def make_tabpfn():
        dev = TABPFN_DEVICE
        if dev == "auto":
            try:
                import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
        try:
            reg = TabPFNRegressor(device=dev, random_state=RANDOM_STATE)
        except TypeError:
            reg = TabPFNRegressor(device=dev)      # older/newer API without random_state
        return make_pipe(reg, feats)

    names = ["RandomForest", "ExtraTrees", "HistGB"] + (["XGBoost"] if HAS_XGB else [])
    if USE_TABPFN and HAS_TABPFN:
        names.append("TabPFN")
    elif USE_TABPFN and not HAS_TABPFN:
        print("TabPFN requested but not installed -> skipping. Install with: pip install tabpfn")
    tuned, rows, cv_dist = {}, [], {}
    for nm in names:
        # TabPFN is a pretrained foundation model: fit/predict, NO hyper-parameter tuning.
        # Wrap the whole build so a TabPFN license/download error SKIPS it instead of killing
        # the run (you still get the other models, stacking, plots and the Excel).
        try:
            best = make_tabpfn() if nm == "TabPFN" else tune(nm, Xtr, ytr, feats)
            if nm == "TabPFN":
                best.fit(Xtr, ytr)
        except Exception as e:
            print(f"  {nm:14s} SKIPPED ({type(e).__name__}): {str(e).splitlines()[0]}")
            if nm == "TabPFN":
                print("  -> TabPFN needs a one-time license. Open https://ux.priorlabs.ai , log in, accept the\n"
                      "     license on the 'Licenses' tab, copy your API key from https://ux.priorlabs.ai/account ,\n"
                      "     then set it once:  Anaconda Prompt ->  setx TABPFN_TOKEN \"your-key-here\"  and restart Spyder.\n"
                      "     (Or set USE_TABPFN=False at the top to skip TabPFN entirely.)")
            continue
        tuned[nm] = best
        try:
            sc = repeated_cv_scores(best, Xtr, ytr)
        except Exception as e:
            print(f"  {nm}: RepeatedCV failed ({type(e).__name__}); using single 5-fold. {e}")
            sc = np.array([np.mean(cross_val_score(clone(best), Xtr, ytr,
                          cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring="r2"))])
        cv_dist[nm] = sc
        tr_m, te_m = metrics(ytr, best.predict(Xtr)), metrics(yte, best.predict(Xte))
        rows.append({"Model": nm, "Train_R2": tr_m["R2"], "RepeatedCV_R2": float(sc.mean()),
                     "RepeatedCV_SD": float(sc.std(ddof=1)) if len(sc) > 1 else 0.0, "Test_R2": te_m["R2"],
                     "Test_RMSE": te_m["RMSE"], "Overfit_Gap": tr_m["R2"] - float(sc.mean())})
        print(f"  {nm:14s} Train={tr_m['R2']:.3f} | RepeatedCV={sc.mean():.3f}±{sc.std(ddof=1) if len(sc)>1 else 0:.3f} "
              f"| Test={te_m['R2']:.3f} | gap={tr_m['R2']-sc.mean():.3f}")

    trained_names = [n for n in names if n in tuned]      # only models that actually fit (TabPFN may be skipped)
    stack = StackingRegressor([(n, clone(tuned[n])) for n in trained_names],
                              final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]), cv=CV_FOLDS, n_jobs=1)
    stack.fit(Xtr, ytr); sc = repeated_cv_scores(stack, Xtr, ytr); cv_dist["Stacking"] = sc
    tr_m, te_m = metrics(ytr, stack.predict(Xtr)), metrics(yte, stack.predict(Xte))
    rows.append({"Model": "Stacking", "Train_R2": tr_m["R2"], "RepeatedCV_R2": float(sc.mean()),
                 "RepeatedCV_SD": float(sc.std(ddof=1)), "Test_R2": te_m["R2"], "Test_RMSE": te_m["RMSE"],
                 "Overfit_Gap": tr_m["R2"] - float(sc.mean())}); tuned["Stacking"] = stack
    print(f"  {'Stacking':14s} Train={tr_m['R2']:.3f} | RepeatedCV={sc.mean():.3f}±{sc.std(ddof=1):.3f} | Test={te_m['R2']:.3f}")

    res = pd.DataFrame(rows).sort_values("RepeatedCV_R2", ascending=False).reset_index(drop=True)
    sel = res.iloc[0]["Model"]

    # ---- seed-bagged final of the selected single model (variance reduction) ----
    bag_note = ""
    if sel != "Stacking":
        bag = SeedBaggingRegressor(tuned[sel], seeds=list(range(RANDOM_STATE, RANDOM_STATE + BAG_SEEDS)))
        bag.fit(Xtr, ytr)
        bag_test = metrics(yte, bag.predict(Xte))["R2"]
        bag_note = f" | seed-bagged({BAG_SEEDS}) test={bag_test:.3f}"

    print("\n" + "=" * 92)
    print(f"SELECTED (RepeatedCV-first): {sel}  RepeatedCV={res.iloc[0]['RepeatedCV_R2']:.3f} "
          f"| Test={res.iloc[0]['Test_R2']:.3f} | gap={res.iloc[0]['Overfit_Gap']:.3f}{bag_note}")
    print("=" * 92)
    print("\nBest test across models:", round(res['Test_R2'].max(), 3),
          "| smallest overfit gap:", round(res['Overfit_Gap'].min(), 3))

    # =========================== PLOTS ===========================
    F = OUT / "figures"
    order = list(res["Model"])
    # 1) overfit gap: Train vs RepeatedCV vs Test
    x = np.arange(len(order)); w = 0.26
    plt.figure(figsize=(9, 5))
    plt.bar(x - w, [res[res.Model == m]["Train_R2"].iloc[0] for m in order], w, label="Train", color="#B4322E")
    plt.bar(x,     [res[res.Model == m]["RepeatedCV_R2"].iloc[0] for m in order], w, label="RepeatedCV", color="#2E5AAC")
    plt.bar(x + w, [res[res.Model == m]["Test_R2"].iloc[0] for m in order], w, label="Test", color="#1B7A43")
    plt.xticks(x, order, rotation=15); plt.ylabel("R²"); plt.ylim(0, 1)
    plt.title("Overfitting picture — Train vs RepeatedCV vs Test R²"); plt.legend()
    plt.tight_layout(); plt.savefig(F / "overfit_gap.png"); plt.close()

    # 2) CV fold-R2 distribution (stability)
    plt.figure(figsize=(9, 5))
    plt.boxplot([cv_dist[m] for m in order], showmeans=True)
    plt.xticks(range(1, len(order) + 1), order, rotation=15)
    plt.axhline(0, color="grey", lw=0.8); plt.ylabel("Fold R² (RepeatedCV)")
    plt.title("Model stability — RepeatedCV fold-R² distribution")
    plt.tight_layout(); plt.savefig(F / "cv_fold_boxplot.png"); plt.close()

    # 3) learning curve for the best model
    try:
        est = tuned[sel] if sel != "Stacking" else tuned[order[1]]
        sizes, tr_s, va_s = learning_curve(clone(est), Xtr, ytr, cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                                           train_sizes=np.linspace(0.3, 1.0, 6), scoring="r2", n_jobs=1)
        plt.figure(figsize=(8, 5))
        plt.plot(sizes, tr_s.mean(1), "o-", color="#B4322E", label="Train R²")
        plt.plot(sizes, va_s.mean(1), "o-", color="#2E5AAC", label="CV R²")
        plt.fill_between(sizes, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=0.15, color="#2E5AAC")
        plt.xlabel("Training examples"); plt.ylabel("R²"); plt.ylim(0, 1)
        plt.title(f"Learning curve — {sel} (gap between curves = overfitting)"); plt.legend()
        plt.tight_layout(); plt.savefig(F / "learning_curve.png"); plt.close()
    except Exception as e:
        print("learning curve skipped:", e)

    # 4) parity (train + test) for best model
    est = tuned[sel]; ptr, pte = est.predict(Xtr), est.predict(Xte)
    lo, hi = float(min(y.min(), min(ptr.min(), pte.min()))), float(max(y.max(), max(ptr.max(), pte.max())))
    plt.figure(figsize=(6.5, 6))
    plt.scatter(ytr, ptr, alpha=0.4, s=22, label="Train", color="#8Fb0d8")
    plt.scatter(yte, pte, alpha=0.85, s=28, edgecolor="k", linewidth=0.3, label="Test", color="#1B7A43")
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="1:1")
    plt.xlabel(f"Measured {TARGET} ({UNITS})"); plt.ylabel(f"Predicted {TARGET} ({UNITS})")
    plt.title(f"Parity — {sel}  (test R²={metrics(yte, pte)['R2']:.3f})"); plt.legend()
    plt.tight_layout(); plt.savefig(F / "parity_best.png"); plt.close()

    # 5) residuals (test)
    plt.figure(figsize=(7.5, 5))
    plt.scatter(pte, pte - np.asarray(yte, float), alpha=0.75, color="#2A9D8F")
    plt.axhline(0, color="k", ls="--", lw=1.3)
    plt.xlabel(f"Predicted {TARGET} ({UNITS})"); plt.ylabel("Residual (pred − meas)")
    plt.title(f"Test residuals — {sel}")
    plt.tight_layout(); plt.savefig(F / "residuals_best.png"); plt.close()

    # 6) feature importance
    model = est.named_steps["model"] if hasattr(est, "named_steps") else None
    if model is not None and hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feats).sort_values()
        plt.figure(figsize=(8, 5)); plt.barh(imp.index, imp.values, color="#2E5AAC")
        plt.xlabel("Importance"); plt.title(f"Feature importance — {sel}")
        plt.tight_layout(); plt.savefig(F / "importance.png"); plt.close()

    with pd.ExcelWriter(OUT / "SCB_AntiOverfit_Results.xlsx", engine="openpyxl") as w:
        res.to_excel(w, sheet_name="Models", index=False)
        pd.DataFrame({"Selected_Features": feats}).to_excel(w, sheet_name="Features", index=False)
    print(f"\nPlots saved to {F}")
    for p in sorted(F.glob("*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
