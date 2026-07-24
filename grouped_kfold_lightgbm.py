# -*- coding: utf-8 -*-
"""
RUTTING & SCB — GroupKFold CROSS-VALIDATION with LightGBM
=========================================================
Requested setup:
  * GroupKFold(5) for data splitting so every REPETITIVE / similar mix stays inside ONE fold
    (a mix used as a validation fold is never also in that fold's training part -> no leakage).
    Grouping key = Mixture_Family_ID (engineering-similarity family; groups all repeated
    reports, revisions, projects and near-duplicate mixes of the same design).
  * Model = LightGBM only.
  * Reports each fold's R2/RMSE/MAE and the pooled out-of-fold (OOF) R2 = the honest score.
  * Design mixes only; all report rows kept; identifiers never used as predictors.
  * Best-fit plot (OOF measured vs predicted) + LightGBM importance.

Reuses the data loading / family construction / preprocessing from family_aware_pipeline.py,
so keep both files in the same folder.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
try: matplotlib.use("TkAgg")
except Exception: pass
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

import family_aware_pipeline as fa   # load(), get_X(), prune(), prep(), OUT, CATEG

CV_FOLDS = 5
GROUP_COL = "Mixture_Family_ID"      # repetitive/similar mixes grouped together
SHOW_PLOTS = True

def _save(name):
    plt.tight_layout(); plt.savefig(fa.OUT/name, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        try: plt.show()
        except Exception: pass
    plt.close()

def run(target_name, sheet, target_col, feats):
    print("\n"+"#"*84+f"\n{target_name} — GroupKFold({CV_FOLDS}) CV with LightGBM  (group = {GROUP_COL})\n"+"#"*84)
    d = fa.load(sheet, target_col, target_name)
    y = d[target_name].reset_index(drop=True)
    X, num, cat = fa.get_X(d, feats)
    keep, dropped = fa.prune(X, y); X = X[keep]
    num = [c for c in keep if c not in fa.CATEG]; cat = [c for c in keep if c in fa.CATEG]
    groups = d[GROUP_COL].astype(str).values
    print(f"rows={len(d)} | groups({GROUP_COL})={pd.Series(groups).nunique()} "
          f"| dropped same-effect: {', '.join(a+'~'+b for a,b in dropped) or 'none'}")

    lgbm = fa.prep(LGBMRegressor(n_estimators=700, learning_rate=0.02, num_leaves=31,
                                 min_child_samples=30, subsample=0.85, colsample_bytree=0.8,
                                 reg_lambda=10, random_state=42, verbose=-1), num, cat)
    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof = np.full(len(y), np.nan)
    print(f"  {'Fold':<6}{'train':>7}{'test':>7}{'grp_test':>9}{'leak':>6}{'R2':>8}{'RMSE':>8}{'MAE':>8}")
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        leak = len(set(groups[tr]) & set(groups[te]))     # must be 0: no repetitive mix across the fold
        m = clone(lgbm); m.fit(X.iloc[tr], y.iloc[tr]); p = m.predict(X.iloc[te]); oof[te] = p
        r2 = r2_score(y.iloc[te], p); rmse = np.sqrt(mean_squared_error(y.iloc[te], p)); mae = mean_absolute_error(y.iloc[te], p)
        print(f"  {k:<6}{len(tr):>7}{len(te):>7}{pd.Series(groups[te]).nunique():>9}{leak:>6}{r2:>8.3f}{rmse:>8.3f}{mae:>8.3f}")
    # pooled out-of-fold metrics (the honest cross-validated score)
    R2 = r2_score(y, oof); RMSE = float(np.sqrt(mean_squared_error(y, oof))); MAE = mean_absolute_error(y, oof)
    print(f"  {'ALL':<6}{'':>7}{len(y):>7}{'':>9}{'':>6}{R2:>8.3f}{RMSE:>8.3f}{MAE:>8.3f}   <- pooled OOF (honest)")

    # ---- plots ----
    unit = "mm" if target_name == "Rut_20k" else ""
    sl, ic = np.polyfit(y.values, oof, 1)
    plt.figure(figsize=(6, 5.6)); plt.scatter(y, oof, alpha=0.5, s=18)
    lo, hi = float(min(y.min(), oof.min())), float(max(y.max(), oof.max())); xs = np.linspace(lo, hi, 100)
    plt.plot([lo, hi], [lo, hi], "--", lw=2, label="Ideal 1:1"); plt.plot(xs, sl*xs+ic, lw=2, label=f"Fit y={sl:.2f}x+{ic:.2f}")
    plt.xlabel(f"Measured {target_name} {unit}"); plt.ylabel(f"OOF-predicted {target_name} {unit}")
    plt.title(f"{target_name} — GroupKFold({CV_FOLDS}) LightGBM (OOF)\nR2={R2:.3f} RMSE={RMSE:.3f} MAE={MAE:.3f}")
    plt.legend(); plt.grid(alpha=0.3); _save(f"{target_name}_groupkfold_lgbm_oof.png")

    m = clone(lgbm); m.fit(X, y)
    try:
        names = list(m.named_steps["prep"].get_feature_names_out()); imp = m.named_steps["model"].feature_importances_
        s = pd.Series(imp, index=names).sort_values(ascending=False).head(16)[::-1]
        plt.figure(figsize=(7, max(4, len(s)*0.32))); plt.barh(s.index, s.values); plt.xlabel("LightGBM importance")
        plt.title(f"{target_name} — LightGBM importance"); plt.grid(axis="x", alpha=0.3); _save(f"{target_name}_groupkfold_lgbm_importance.png")
    except Exception: pass
    print(f"  Figures saved to: {fa.OUT}")

def main():
    run("Rut_20k", "Rutting_Design", "LWT_Design_Result", fa.RUT_FEATURES)
    run("SCB", "SCB_Design", "SCB_Result", fa.SCB_FEATURES)

if __name__ == "__main__":
    main()
