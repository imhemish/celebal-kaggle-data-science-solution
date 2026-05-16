"""
Anomaly Detection - Energy Manufacturing Plant
Evaluation Metric: F1 Score (binary, class 1 = anomaly)
Usage: python solution.py
Outputs: submission.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
train      = pd.read_parquet("train.parquet")
test       = pd.read_parquet("test.parquet")
sample_sub = pd.read_parquet("sample_submission.parquet")

train["Date"] = pd.to_datetime(train["Date"], unit="ms")
test["Date"]  = pd.to_datetime(test["Date"],  unit="ms")
train["target"] = train["target"].astype(int)

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Anomaly rate: {train['target'].mean()*100:.2f}%  "
      f"({train['target'].sum():,} anomalies / {len(train):,} total)")
print(f"Class imbalance: {(1-train['target'].mean())/train['target'].mean():.0f}:1\n")


# ── 2. KEY OBSERVATIONS FROM EDA ─────────────────────────────────────────────
#
#  • X1-X5 appear to be LOGARITHMS of raw sensor readings (most values are e^n)
#    e.g. X4 = 1.0 (=e^0), 2.718 (=e^1), 7.389 (=e^2) in normal rows
#         X4 = 4.3e15, 5.5e34 etc. in anomalous / fault rows  →  HUGE signal
#
#  • X3 behaves similarly: mostly 1.0 (normal), explodes to e^16, e^38 in faults
#
#  • Anomaly rate correlates STRONGLY with log magnitude of X3 and X4:
#    log(X3) or log(X4) > 10  →  anomaly rate jumps from ~0.4% to 17-100%
#
#  • Date spans 2020-12-16 to 2024-12-11 (4 years, daily readings)
#  • X5 takes discrete values (0, log(2), log(18.5) etc.) — likely categorical
#  • X2 is narrow range (~5.45-5.47) — low signal on its own


# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    # --- Core log transforms ---
    # X3 and X4 are the dominant signals. Taking log exposes the "normal" baseline
    # and separates extreme fault values from benign ones.
    for col in ["X1", "X2", "X3", "X4", "X5"]:
        df[f"log_{col}"] = np.log(df[col].clip(lower=1e-9))

    # --- Extremity flags (binary, high precision for anomaly class) ---
    df["X3_extreme"]   = (df["log_X3"] > 10).astype(int)
    df["X4_extreme"]   = (df["log_X4"] > 10).astype(int)
    df["X3_very_ext"]  = (df["log_X3"] > 30).astype(int)  # 79-88 → ~100% anomaly
    df["X4_very_ext"]  = (df["log_X4"] > 50).astype(int)  # 72-80 → ~82% anomaly

    # --- Combined extremity score ---
    df["total_log_extremity"] = df["log_X3"] + df["log_X4"]
    df["max_log_extremity"]   = df[["log_X3", "log_X4"]].max(axis=1)

    # --- X1 extreme (log_X1 > 1.25 → 67% anomaly rate in EDA) ---
    df["X1_extreme"] = (df["log_X1"] > 1.25).astype(int)

    # --- "Normal" value flags (sensor stuck at baseline = NOT anomalous) ---
    # e^0=1, e^1=2.718, e^2=7.389 are the dominant X4 normal values
    normal_X4 = {1.0, 2.718281828459045, 7.38905609893065}
    df["X4_is_normal_val"] = df["X4"].isin(normal_X4).astype(int)
    df["X3_is_one"]        = (df["X3"] == 1.0).astype(int)

    # --- Date features ---
    df["year"]       = df["Date"].dt.year
    df["month"]      = df["Date"].dt.month
    df["dayofyear"]  = df["Date"].dt.dayofyear
    df["dayofweek"]  = df["Date"].dt.dayofweek

    # --- Pairwise log ratios (deviation between sensors) ---
    df["log_X4_minus_X3"] = df["log_X4"] - df["log_X3"]
    df["log_X1_minus_X2"] = df["log_X1"] - df["log_X2"]

    return df


print("Engineering features...")
train_fe = engineer_features(train)
test_fe  = engineer_features(test)

FEATURE_COLS = [
    # Raw sensor values
    "X1", "X2", "X3", "X4", "X5",
    # Log transforms
    "log_X1", "log_X2", "log_X3", "log_X4", "log_X5",
    # Extremity flags
    "X3_extreme", "X4_extreme", "X3_very_ext", "X4_very_ext", "X1_extreme",
    # Combined extremity
    "total_log_extremity", "max_log_extremity",
    # Normal value flags
    "X4_is_normal_val", "X3_is_one",
    # Date
    "year", "month", "dayofyear", "dayofweek",
    # Ratios
    "log_X4_minus_X3", "log_X1_minus_X2",
]

X      = train_fe[FEATURE_COLS]
y      = train_fe["target"]
X_test = test_fe[FEATURE_COLS]

print(f"Feature matrix: {X.shape}  |  Test: {X_test.shape}\n")


# ── 4. CROSS-VALIDATION + THRESHOLD TUNING ───────────────────────────────────
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# scale_pos_weight handles 115:1 imbalance. Alternatively try class_weight='balanced'.
MODEL_PARAMS = dict(
    n_estimators      = 600,
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = 8,
    min_child_samples = 20,
    scale_pos_weight  = 115,   # ~(# negatives / # positives) in train
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs  = np.zeros(len(X))     # out-of-fold probabilities
test_probs = np.zeros(len(X_test))
f1_scores  = []

print("Running 5-fold cross-validation...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
    y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[],                  # remove early stopping to keep it simple
    )

    val_prob  = model.predict_proba(X_val)[:, 1]
    oof_probs[val_idx] = val_prob

    # Default threshold = 0.5 for a quick fold F1
    val_pred  = (val_prob >= 0.5).astype(int)
    f1 = f1_score(y_val, val_pred)
    f1_scores.append(f1)

    test_probs += model.predict_proba(X_test)[:, 1] / 5   # average over folds

    print(f"  Fold {fold+1}: F1 = {f1:.4f}")

print(f"\nCV Mean F1 (threshold=0.5): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


# ── 5. THRESHOLD SEARCH ON OOF PREDICTIONS ───────────────────────────────────
# The default 0.5 threshold is rarely optimal for imbalanced problems.
# We sweep thresholds on OOF probs (which are honest — never seen the model).

print("\nSearching best threshold on OOF probabilities...")
best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.01, 0.90, 0.01):
    preds = (oof_probs >= t).astype(int)
    f1    = f1_score(y, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"Best OOF threshold: {best_thresh:.2f}  →  F1 = {best_f1:.4f}")
print("\nOOF classification report (best threshold):")
print(classification_report(y, (oof_probs >= best_thresh).astype(int),
                             target_names=["Normal", "Anomaly"]))


# ── 6. FEATURE IMPORTANCE ────────────────────────────────────────────────────
# (Uses the last fold's model as a proxy — retrain on full data for production)
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
print("Top 10 features by importance:")
print(importances.sort_values(ascending=False).head(10).to_string())
print()


# ── 7. OPTIONAL: RETRAIN ON FULL TRAIN DATA ──────────────────────────────────
# Uncomment to get slightly stronger test predictions (no CV estimate though).
#
print("Retraining on full training data...")
final_model = LGBMClassifier(**MODEL_PARAMS)
final_model.fit(X, y)
test_probs = final_model.predict_proba(X_test)[:, 1]


# ── 8. GENERATE SUBMISSION ───────────────────────────────────────────────────
test_preds = (test_probs >= best_thresh).astype(int)

submission = pd.DataFrame({
    "ID":     test_fe["ID"],
    "target": test_preds.astype(str),   # match sample_submission dtype (str)
})

submission.to_csv("submission.csv", index=False)
print(f"submission.csv saved → {len(submission):,} rows")
print(f"Predicted anomalies: {test_preds.sum():,} "
      f"({test_preds.mean()*100:.2f}%)")
print("\nDone! Upload submission.csv to Kaggle.")


# ── 9. TIPS FOR IMPROVING SCORE ──────────────────────────────────────────────
"""
1. XGBoost / CatBoost ensemble: average their test_probs with LightGBM's before
   applying the threshold. Diversity between models usually helps F1.

2. Time-based validation: instead of random KFold, use the last 3 months of
   training dates as the validation set (TimeSeriesSplit). This tests
   generalisation more honestly if the test set is future data.

3. Rolling features: group by Date and compute rolling std of X3/X4 over
   past 7 days. Sudden spikes (high std) often precede anomalies.

4. Isolation Forest as an extra feature: fit sklearn IsolationForest on all
   X1-X5, add its anomaly score as a feature to LightGBM.

5. Post-processing: if consecutive rows with the same Date all have high
   probability, flip borderline 0 predictions to 1 (anomalies tend to cluster).
"""