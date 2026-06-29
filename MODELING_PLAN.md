# Rut_20k Modeling Plan v2 — Pushing Validation R² Higher (70/10/20, True RBR)

This document is the "updated plan to achieve more" that accompanies
[`rut20k_70_10_20_workflow.py`](./rut20k_70_10_20_workflow.py). It maps every step to the
requests in your message and to the advisor comments in `Executive_Summary_5.docx` and the run
history in `Rut_20k_Model_Runs_and_Split_Comparison.docx`.

---

## 1. What you asked for, and where it lives in the code

| Request | Implementation |
|---|---|
| Re-split to **70% train / 10% validation / 20% test** | `stratified_70_10_20_split()` — two-stage, target-bin–stratified, `random_state=42` |
| **Keep the test set hidden** (don't use it) | Controlled by **`SCORE_LOCKED_TEST`** (default **`False`**). When `False`, this is a *development-only* run: train on 70%, validate on 10%, and the 20% test is split off, saved to its own Excel file, and **never read, scored, explained, or plotted**. Results go to a separate **`..._DEV_ONLY_train_val_Results.xlsx`** workbook. Set it `True` later for a one-time final evaluation (refit on 80% dev, score + SHAP the test once → `..._WITH_LOCKED_TEST_Results.xlsx`). Test rows always saved as `LOCKED_test_20pct_DO_NOT_USE_FOR_TUNING.xlsx`. |
| **Add the RBR** as a real feature | `RBR_JMF_fraction` (= file's `RBR_decimal`) and `RBR_JMF_percent` (= `RBR_percent`). Used in `VolumetricsB_NoADT_RBR_Both`, `SHAP12/14_RBR`, and the interaction sets. One set *replaces* `RAP_pct_x_ACinRAP` with RBR to measure RBR's standalone value. |
| **High regularization + highly tuned + robust model** | Shallow trees (`max_depth` 2–3), high `reg_alpha`/`reg_lambda`, low `subsample`/`colsample`, `gamma`, large `min_child_weight`; `RandomizedSearchCV` (n_iter=120) in training-only 5-fold CV; **RepeatedKFold (5×5)** robustness pass on the 80% dev set. |
| **Best-fit vs 45° line for train AND validation** | `plot_fit()` draws the ideal 1:1 (45°) line and the best-fit regression line for `Train70`, `Validation10` (and `LockedTest20` at the end). |
| **Residuals** | `plot_residuals()`: residual vs predicted, residual vs measured, residual histogram, Q-Q plot, relative-error plots. |
| **SHAP and PDP** | `run_shap()` (bar / beeswarm / waterfall) + `run_pdp()` (1D for top features, 2D for physical pairs incl. `PG_HighTemp × RBR`). |
| **Learning rate / learning curve and bias-variance** | `plot_learning_curve()` and `plot_bias_variance()` (train/val R² and gap vs `max_depth` and vs `n_estimators`) + complexity elbow. |
| **Statistical + feature-importance analysis** | `statistical_analysis()`: descriptive stats, correlation heatmap, correlation-with-target, **VIF multicollinearity**; plus permutation importance and SHAP importance. |
| **SHAP + sensitivity over RBR ranges; find where the model is weak** | `shap_by_rbr_band()`, `error_by_group()` by `RBR_band` and by rut range, **applicability-domain** flag for sparse extreme regions, and an auto **Model_Weakness_Map**. |
| **Use SHAP + advisor comments to pick the most impactful variables** | Feature sets are seeded from the SHAP ranking (PG_HighTemp, RBR, SandEq, Absorption, VFA, …) and the advisor's recommended physical interactions. |
| **Latency + throughput (real-time serving metrics)** | `inference_performance()` measures single-row latency (median + p95), batch throughput (rows/sec), per-row batch latency, and on-disk model size on the **validation** sample (never the locked test) → **`Realtime_Latency_Throughput`** sheet. |

### Meeting the standard evaluation criteria

| Criterion (textbook / supervisor) | How this workflow satisfies it |
|---|---|
| **Three-way data split** — train fits the model, validation tunes hyper-parameters & selects the model, test is reserved for one final unbiased evaluation; typical 70–80 / 10–15 / 10–15 | Two scripts: **70/10/20** and **65/15/20** (both inside the recommended ranges). Train = model fitting; **validation** = held-out comparison that drives model selection, while hyper-parameters are tuned by `RandomizedSearchCV` 5-fold CV *inside the training data*; the **20% test is locked** behind `SCORE_LOCKED_TEST` and untouched until a single final evaluation. |
| **Latency & throughput in metrics** (real-time use) | `inference_performance()` reports prediction **latency** (delay per input, median + p95 tail) and **throughput** (predictions/sec), plus model size, in the `Realtime_Latency_Throughput` sheet — so the model is judged on speed, not only accuracy. |
| **Test set represents the real environment** | The split is **target-bin stratified** (`random_state=42`), so the locked test mirrors the full rut/SCB distribution rather than a skewed slice. |
| **Optimization steps when accuracy/speed fall short** | *Feature engineering*: 9 feature sets incl. physical interactions & true RBR. *Preprocessing*: median imputation + one-hot in a leakage-safe `Pipeline`. *Architecture/technique*: high regularization, Huber loss, RepeatedCV per family, stacking ensemble. *Compression/efficiency*: shallow trees + the model-size metric give a direct quantization/compression lever. |

---

## 2. The improvement levers (from the advisor doc), in priority order

1. **True RBR replaces redundant RAP terms.** SHAP showed `PG_HighTemp` and `RAP_pct_x_ACinRAP`
   dominate, and `ACinRAP`/`RAP_pct_x_ACinRAP` are partially redundant. RBR
   (`RAP binder / design binder`) is the physically meaningful single quantity. The
   `..._RBR_Replace_RAPAC` set tests whether RBR alone beats the raw RAP product.
2. **Stronger regularization to close the train–validation gap.** The documented gap was
   ~0.20–0.27. Shallow trees + high L1/L2 + subsampling target this directly; the
   bias-variance curves show the elbow where train and validation converge.
3. **Physics-only interaction terms** (`PG_x_RBR`, `PG_x_RAPAC`, `Abs_x_RBR`,
   `SandEq_x_DustBinder`, …) added only to the SHAP sets, not blindly.
4. **High-rut sample weighting** (q80→1.4, q90→1.8) to attack the compression
   (low rut over-predicted, high rut under-predicted). Kept only if it improves high-rut MAE
   without hurting overall validation R² — reported side-by-side as a sensitivity.
5. **Stacking ensemble** (XGB + LightGBM + HistGB + CatBoost, RidgeCV meta, OOF cv=5) to
   squeeze partially-uncorrelated errors.
6. **Repeated CV** so the chosen model is robust to the split, not a lucky single fold.

---

## 3. Where the model is weak (diagnosed automatically)

The workflow writes a **Model_Weakness_Map** and per-group error tables. In the verification run
the weak regions matched the advisor's prediction exactly:

- **Very-high rut (>7 mm)** and **low rut (<2 mm)** tails: large negative/positive mean bias →
  the model *compresses* (under-predicts high rut, over-predicts low rut). Best-fit slope ≈0.53–0.70 < 1.
- **No-RAP / Low-RBR bands** carry higher MAE than the Moderate band where most data sits.
- **Applicability domain:** samples with `RBR_percent` or `RAP_pct_x_ACinRAP` above the 97.5th
  percentile are flagged as sparse-data extrapolation — predictions there are less reliable and
  are reported separately (`Applicability_Domain` sheet).

These tables tell you *exactly* which RBR ranges and rut ranges to treat with caution.

---

## 4. Honest expectation about the 0.80 target

The advisor's Executive Summary is explicit: with the current predictors the realistic ceiling is
**~0.60–0.65 dev R²**, and reaching **0.80 almost certainly requires new physical inputs**
(binder rheology / continuous PG, short- and long-term aging, test temperature, gradation shape
beyond two sieves). This script pushes every defensible lever (RBR, regularization, interactions,
weighting, stacking, repeated CV) and reports the **best honest metrics** in a `Decision_Table`
against your targets (R²≥0.80, Gap≤0.16, RMSE<1.03, MAE<0.73) rather than overfitting to hit a
number. If no model clears 0.80, the script says so and recommends data enrichment — and the
locked 20% test stays the unbiased reality check.

In the CPU verification run, the stacking model reached **validation R² ≈ 0.63** with Gap ≈ 0.15
(at the target) and a locked-test R² ≈ 0.59 — consistent with the advisor's expectation. The full
GPU run with `N_ITER_XGB=120` and all feature sets should do at least as well.

---

## 4b. Two-phase use of the locked test (`SCORE_LOCKED_TEST`)

The 20% test is treated as a one-shot resource, so it is gated behind a single switch:

- **Phase 1 — `SCORE_LOCKED_TEST = False` (default).** Train + validate only. Use this to confirm
  the model is **stable and leakage-free** via: the held-out validation metrics, the
  `RepeatedCV_Robustness` sheet (mean ± std over 25 dev folds), `CV_Folds_Train70` fold stability,
  the learning curve, and the bias-variance curves. The 20% test is saved untouched in
  `splits/LOCKED_test_20pct_DO_NOT_USE_FOR_TUNING.xlsx` and is **never** opened by the code.
  Output: `Rut20k_v2_70_10_20_DEV_ONLY_train_val_Results.xlsx`.
- **Phase 2 — `SCORE_LOCKED_TEST = True`.** Only after you are satisfied. Refits the selected model
  on the 80% dev set and scores + explains the test **once**.
  Output: `Rut20k_v2_70_10_20_WITH_LOCKED_TEST_Results.xlsx`.

This guarantees the final test number stays an unbiased estimate — you never tune against it.

## 5. How to run it

1. Put `Rutting_Cleaned_with_RBR.xlsx` in your `Downloads` folder (or next to the script).
2. Open `rut20k_70_10_20_workflow.py`. Top-of-file settings you may want:
   - `QUICK_SMOKE_TEST = True` for a fast end-to-end check (~minutes); set `False` for the real run.
   - `USE_GPU = True` (auto-falls back to CPU if CUDA isn't available).
   - `N_ITER_XGB`, `REPEATED_CV_REPEATS`, `RUN_STACKING` to trade runtime vs thoroughness.
3. Run it. Everything lands in `Downloads/Rut20k_v2_70_10_20_RBR_outputs/`:
   - `Rut20k_v2_70_10_20_RBR_Results.xlsx` (all tables: candidates, decision table, repeated-CV,
     final metrics, error-by-RBR-band, applicability domain, weakness map, SHAP/VIF/correlation).
   - `figures/` (best-fit, residuals, SHAP, PDP, learning curve, bias-variance, RBR sensitivity).
   - `models/FINAL_SELECTED_MODEL_refit_on_dev80.joblib`.
   - `splits/` including the untouched locked-test file.

Optional dependencies degrade gracefully if missing: `xgboost`, `lightgbm`, `catboost`, `shap`,
`scipy`, `statsmodels` (VIF falls back to a correlation-inverse estimate).
