# Task 3: 30-Day Monthly Spending Prediction

**Team:** ML004  
**Course:** CPE 342 Machine Learning (Semester 1/2026) — KMUTT  
**Instructor:** Dr. Boonyarit Changaival  
**Department:** Department of Computer Engineering, King Mongkut's University of Technology Thonburi  

---

## 1. Problem Formulation & Metric

- **Target Variable:** `spending_30d` ($\ge 0.0$ THB).
- **Data Characteristics:** Zero-inflated continuous distribution (~50.5% exact zero spending, heavy Pareto tail among positive spenders up to 265,404 THB).
- **Official Competition Metric:** Normalized Mean Absolute Error (NMAE)

$$
\text{NMAE} = \frac{1}{1 + \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}}
$$

- Baseline (Predicting Global Median): $\text{NMAE} = 0.5000$
- Challenge Target: $\text{NMAE} \ge 0.7500$
- Our Ensembled Solution: **$\text{NMAE} = 0.7889$** (MAE: **2,809.54 THB**)

---

## 2. Methodology & Architecture: Two-Stage Tri-Model Hurdle Ensemble

Standard single-stage regression models fail on zero-inflated heavy-tailed distributions due to gradient conflict between mass at zero and extreme positive values. We implement a decoupled Two-Stage Hurdle Model:

1. **Stage 1 (Binary Gatekeeper Classifier):**
   - Predicts propensity to spend $P(Y > 0 \mid X)$.
   - Algorithms: LightGBM (`binary_logloss`), CatBoost (`Logloss`), XGBoost (`binary:logistic`).
   - Objective: Distinguish non-payers ($0$ THB) from active spenders.

2. **Stage 2 (Positive Spending Regressor):**
   - Trained exclusively on samples with $y > 0$ using logarithmic target $\log(1 + y)$.
   - Loss Function: $L_1$ Regression (MAE / Regression-L1) to align directly with the Bayes-optimal median estimator required by MAE evaluation.
   - Algorithms: LightGBM (`regression_l1`), CatBoost (`MAE`), XGBoost (`reg:absoluteerror`).

3. **Stage 3 (Bayes-Optimal Hurdle Post-Processing):**
   - Reconstructs spending via optimal gating threshold $\tau^*$ and blending parameter $\alpha^*$:

$$
\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \tau^*) \cdot \left[ \alpha \hat{y}_i^{\text{positive}} + (1 - \alpha) \hat{y}_i^{\text{direct}} \right]
$$

   - Hyperparameters $\tau^* = 0.50$, $\alpha^* = 0.95$ tuned via grid search on Out-Of-Fold (OOF) predictions.

---

## 3. Validation Results Summary (5-Fold Stratified CV)

| Model Architecture | OOF MAE (THB) | OOF NMAE | Status |
| :--- | :---: | :---: | :---: |
| Baseline (Empirical Global Median) | 10,495.48 | 0.5000 | Reference Baseline |
| Standalone LightGBM Hurdle | 2,879.38 | 0.7847 | Target Exceeded (+0.0347) |
| Standalone CatBoost Hurdle | 2,862.14 | 0.7858 | Target Exceeded (+0.0358) |
| **Tri-Model Hurdle Ensemble (LGBM + CatBoost + XGBoost)** | **2,809.54** | **0.7889** | **Top Performance (+0.0389)** |

---

## 4. Execution & Artifacts

- **Jupyter Notebook:** `task3_monthly_spending.ipynb` (Fully self-contained, automated dataset resolution for Local, Google Colab, and Kaggle runtimes).
- **Test Set Predictions:** `predictions/test_predictions.csv` (25,889 rows with `player_id` and `spending_30d`).
- **Diagnostic Visualizations:** Saved in `assets/`:
  - `figure1-spending-eda.png`
  - `figure2-hurdle-calibration-diagnostics.png`
  - `figure3-residual-error-distribution.png`
  - `figure4-test-spending-prediction-density.png`
