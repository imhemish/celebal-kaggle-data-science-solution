"""
Anomaly Detection v2 — Energy Manufacturing Plant
Improvements over v1:
  - Daily aggregate features (per-day max/mean/std of sensors)
  - Rolling window features (past 3/7/14 days) — captures fault build-up
  - Row-level deviation from daily baseline
  - LightGBM + XGBoost ensemble
  - OOF threshold search (3-fold, fast)
  - Final model trained on 100% data

Install: pip install lightgbm xgboost scikit-learn pandas pyarrow
Run:     python solution_v2.py
Output:  submission.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet("train.parquet")
test  = pd.read_parquet("test.parquet")

train["Date"] = pd.to_datetime(train["Date"], unit="ms")
test["Date"]  = pd.to_datetime(test["Date"],  unit="ms")
train["target"] = train["target"].astype(int)

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Anomaly rate: {train['target'].mean()*100:.3f}%\n")


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

def add_base_features(df):
    """Row-level features."""
    df = df.copy()

    for col in ["X1", "X2", "X3", "X4", "X5"]:
        df[f"log_{col}"] = np.log(df[col].clip(lower=1e-9))

    # Extremity flags — core signal from EDA
    df["X3_extreme"]      = (df["log_X3"] > 10).astype(int)
    df["X4_extreme"]      = (df["log_X4"] > 10).astype(int)
    df["X3_very_extreme"] = (df["log_X3"] > 30).astype(int)
    df["X4_very_extreme"] = (df["log_X4"] > 50).astype(int)
    df["X1_extreme"]      = (df["log_X1"] > 1.25).astype(int)

    # Combined sensor fault score
    df["total_log_extremity"] = df["log_X3"] + df["log_X4"]
    df["max_log_extremity"]   = df[["log_X3", "log_X4"]].max(axis=1)
    df["any_extreme"]         = ((df["X3_extreme"] == 1) | (df["X4_extreme"] == 1)).astype(int)

    # Normal value flags
    _normal_X4 = {1.0, 2.718281828459045, 7.38905609893065}
    df["X4_is_normal"] = df["X4"].isin(_normal_X4).astype(int)
    df["X3_is_one"]    = (df["X3"] == 1.0).astype(int)

    # Sensor ratios
    df["log_X4_minus_X3"] = df["log_X4"] - df["log_X3"]
    df["log_X1_minus_X2"] = df["log_X1"] - df["log_X2"]
    df["log_X3_times_X4"] = df["log_X3"] * df["log_X4"]

    # Date
    df["year"]      = df["Date"].dt.year
    df["month"]     = df["Date"].dt.month
    df["dayofyear"] = df["Date"].dt.dayofyear
    df["dayofweek"] = df["Date"].dt.dayofweek

    return df


def build_daily_features(df_train, df_test):
    """
    Daily aggregate + rolling features.

    KEY INSIGHT: anomalies cluster on specific days (some days have 10%+ anomaly rate).
    Knowing how extreme yesterday's / last week's sensor readings were is very predictive.

    We build these from train+test combined so test rows get valid rolling stats.
    Rolling uses .shift(1) so each day only sees PAST days — no leakage.
    """
    # Combine for consistent date coverage
    combined = pd.concat([
        df_train[["Date", "log_X1", "log_X2", "log_X3", "log_X4", "log_X5"]],
        df_test [["Date", "log_X1", "log_X2", "log_X3", "log_X4", "log_X5"]],
    ], ignore_index=True)

    daily = combined.groupby("Date").agg(
        d_X3_max  = ("log_X3", "max"),
        d_X4_max  = ("log_X4", "max"),
        d_X3_mean = ("log_X3", "mean"),
        d_X4_mean = ("log_X4", "mean"),
        d_X3_std  = ("log_X3", "std"),
        d_X4_std  = ("log_X4", "std"),
        d_X1_std  = ("log_X1", "std"),
        d_X2_std  = ("log_X2", "std"),
        d_n       = ("log_X3", "count"),
    ).sort_index().reset_index()

    daily["d_X3_std"] = daily["d_X3_std"].fillna(0)
    daily["d_X4_std"] = daily["d_X4_std"].fillna(0)

    # Rolling windows — shift(1) means each day sees only past days (no leakage)
    for w in [3, 7, 14]:
        daily[f"r{w}_X3_max"]  = daily["d_X3_max"].shift(1).rolling(w, min_periods=1).max()
        daily[f"r{w}_X4_max"]  = daily["d_X4_max"].shift(1).rolling(w, min_periods=1).max()
        daily[f"r{w}_X3_mean"] = daily["d_X3_mean"].shift(1).rolling(w, min_periods=1).mean()
        daily[f"r{w}_X4_mean"] = daily["d_X4_mean"].shift(1).rolling(w, min_periods=1).mean()
        daily[f"r{w}_X3_std"]  = daily["d_X3_std"].shift(1).rolling(w, min_periods=1).mean()
        daily[f"r{w}_X4_std"]  = daily["d_X4_std"].shift(1).rolling(w, min_periods=1).mean()

    # How much did yesterday's max spike vs the prior 7-day baseline?
    daily["X3_spike_ratio"] = (daily["d_X3_max"].shift(1) /
                                (daily[f"r7_X3_mean"].shift(1) + 1e-6))
    daily["X4_spike_ratio"] = (daily["d_X4_max"].shift(1) /
                                (daily[f"r7_X4_mean"].shift(1) + 1e-6))

    drop_cols = ["d_n"]
    daily = daily.drop(columns=drop_cols)

    return daily


def add_daily_deviation(df, daily_df):
    """How much does THIS ROW deviate from today's daily average?"""
    df = df.merge(daily_df, on="Date", how="left")

    df["X3_dev_from_daily_mean"] = df["log_X3"] - df["d_X3_mean"]
    df["X4_dev_from_daily_mean"] = df["log_X4"] - df["d_X4_mean"]
    df["X3_dev_from_daily_max"]  = df["d_X3_max"] - df["log_X3"]
    df["X4_dev_from_daily_max"]  = df["d_X4_max"] - df["log_X4"]

    return df


# Build features
print("Engineering features...")
train_fe = add_base_features(train)
test_fe  = add_base_features(test)

daily = build_daily_features(train_fe, test_fe)

train_fe = add_daily_deviation(train_fe, daily)
test_fe  = add_daily_deviation(test_fe,  daily)

FEATURE_COLS = [
    # Raw + log
    "X1", "X2", "X3", "X4", "X5",
    "log_X1", "log_X2", "log_X3", "log_X4", "log_X5",
    # Extremity
    "X3_extreme", "X4_extreme", "X3_very_extreme", "X4_very_extreme", "X1_extreme",
    "total_log_extremity", "max_log_extremity", "any_extreme",
    # Normal flags
    "X4_is_normal", "X3_is_one",
    # Ratios
    "log_X4_minus_X3", "log_X1_minus_X2", "log_X3_times_X4",
    # Date
    "year", "month", "dayofyear", "dayofweek",
    # Daily stats
    "d_X3_max", "d_X4_max", "d_X3_mean", "d_X4_mean",
    "d_X3_std", "d_X4_std", "d_X1_std", "d_X2_std",
    # Rolling windows
    "r3_X3_max",  "r3_X4_max",  "r3_X3_mean", "r3_X4_mean",
    "r7_X3_max",  "r7_X4_max",  "r7_X3_mean", "r7_X4_mean",
    "r14_X3_max", "r14_X4_max", "r14_X3_mean","r14_X4_mean",
    "r3_X3_std",  "r3_X4_std",  "r7_X3_std",  "r7_X4_std",
    # Spike ratios
    "X3_spike_ratio", "X4_spike_ratio",
    # Row deviation from daily
    "X3_dev_from_daily_mean", "X4_dev_from_daily_mean",
    "X3_dev_from_daily_max",  "X4_dev_from_daily_max",
]

X      = train_fe[FEATURE_COLS].fillna(0)
y      = train_fe["target"]
X_test = test_fe[FEATURE_COLS].fillna(0)

print(f"Features: {len(FEATURE_COLS)}  |  Train: {X.shape}  |  Test: {X_test.shape}\n")


# ── 3. OOF THRESHOLD SEARCH (3-fold, fast) ───────────────────────────────────
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

lgb_params = dict(
    n_estimators     = 500,
    learning_rate    = 0.05,
    num_leaves       = 127,
    max_depth        = 9,
    min_child_samples= 20,
    scale_pos_weight = 115,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)

xgb_params = dict(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 8,
    min_child_weight = 10,
    scale_pos_weight = 115,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    tree_method      = "hist",   # fast on CPU
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

oof_probs = np.zeros(len(X))

print("Running 3-fold CV for threshold search...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    lgb = LGBMClassifier(**lgb_params)
    xgb = XGBClassifier(**xgb_params)

    lgb.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)

    # Average the two models
    p_lgb = lgb.predict_proba(X_val)[:, 1]
    p_xgb = xgb.predict_proba(X_val)[:, 1]
    oof_probs[val_idx] = 0.5 * p_lgb + 0.5 * p_xgb

    f1 = f1_score(y_val, (oof_probs[val_idx] >= 0.5).astype(int))
    print(f"  Fold {fold+1}: F1 = {f1:.4f} (threshold=0.5)")

# Search best threshold on OOF
print("\nSearching best threshold...")
best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.01, 0.90, 0.005):
    f1 = f1_score(y, (oof_probs >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"Best threshold: {best_thresh:.3f}  →  OOF F1 = {best_f1:.4f}")


# ── 4. FINAL MODELS ON FULL TRAIN DATA ───────────────────────────────────────
print("\nTraining final models on full data...")

lgb_final = LGBMClassifier(**lgb_params)
xgb_final = XGBClassifier(**xgb_params)

lgb_final.fit(X, y)
xgb_final.fit(X, y)

p_lgb_test = lgb_final.predict_proba(X_test)[:, 1]
p_xgb_test = xgb_final.predict_proba(X_test)[:, 1]

# Ensemble: equal weight (tune this ratio if you want)
test_probs = 0.5 * p_lgb_test + 0.5 * p_xgb_test
test_preds = (test_probs >= best_thresh).astype(int)

print(f"\nPredicted anomalies in test: {test_preds.sum():,} "
      f"({test_preds.mean()*100:.2f}%)")


# ── 5. FEATURE IMPORTANCE ────────────────────────────────────────────────────
imp = pd.Series(lgb_final.feature_importances_, index=FEATURE_COLS)
print("\nTop 15 features (LightGBM):")
print(imp.sort_values(ascending=False).head(15).to_string())


# ── 6. SUBMISSION ─────────────────────────────────────────────────────────────
submission = pd.DataFrame({
    "ID":     test_fe["ID"],
    "target": test_preds.astype(str),
})
submission.to_csv("submission.csv", index=False)
print(f"\nsubmission.csv saved — {len(submission):,} rows.")
print("Upload to Kaggle. Good luck!")