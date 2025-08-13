import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

DATA_PATH = "data/processed/model_df.parquet"
MODEL_PATH = "models/model_xgb.joblib"
CAL_MODEL_PATH = "models/model_xgb_calibrated.joblib"
THRESHOLD_PATH = "models/model_threshold.joblib"
FEATURES_PATH = "models/feature_names.joblib"
METADATA_PATH = "models/model_training_metadata.joblib"


# ---- XGBoost Best Parameters ----
xgb_params = {
    "tree_method": "hist",
    #"device": "cuda",
    "objective": "binary:logistic",
    "max_depth": 4,
    "learning_rate": 0.06296939501995362,
    "gamma": 0.49470084266224107,
    "min_child_weight": 7,
    "subsample": 0.8800704280225073,
    "colsample_bytree": 0.5857661985145072,
    "reg_alpha": 0.12002226134106342,
    "reg_lambda": 2.839977470486064,
    "scale_pos_weight": 4.851299799430592,
    "eval_metric": "auc",
    "random_state": 42
}

TARGET = "loan_status"
CALIBRATION_METHOD = "isotonic"
TEST_SIZE = 0.18
RANDOM_STATE = 42
TARGET_BAD_RATE = 0.10
MIN_APPROVAL_RATE = 0.05

def train_and_save_model():
    # Load processed data
    df = pd.read_parquet(DATA_PATH)
    
    # Separate features and target
    feature_names = [col for col in df.columns if col != TARGET]
    X = df[feature_names]
    y = df[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Train best XGBoost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train)

    # Calibrate probabilities
    cal_model = CalibratedClassifierCV(xgb, method=CALIBRATION_METHOD, cv="prefit")
    cal_model.fit(X_train, y_train)
    cal_pred_probs = cal_model.predict_proba(X_test)[:, 1]

    # Find optimal threshold using approval/bad rate tradeoff
    results = pd.DataFrame({
        "pred_pd": cal_pred_probs,
        "actual": y_test.values
    })
    thresholds = np.linspace(0, 1, 101)
    approval_rates, bad_rates = [], []
    for t in thresholds:
        approved = results[results["pred_pd"] <= t]
        approval_rates.append(len(approved) / len(results))
        if len(approved) > 0:
            bad_rates.append(approved["actual"].mean())
        else:
            bad_rates.append(np.nan)
    bad_rates_array = np.array(bad_rates)
    thresholds_array = np.array(thresholds)
    safe_indices = np.where(
        (bad_rates_array <= TARGET_BAD_RATE) &
        (np.array(approval_rates) >= MIN_APPROVAL_RATE)
    )[0]
    if len(safe_indices) > 0:
        safe_idx = safe_indices[-1]
        safe_threshold = thresholds_array[safe_idx]
        print(f"Max safe threshold: {safe_threshold:.3f}")
        print(f"Bad rate at cutoff: {bad_rates_array[safe_idx]:.2%}")
        print(f"Approval rate at cutoff: {approval_rates[safe_idx]:.2%}")
    else:
        safe_threshold = 0.5  # fallback default
        print("No threshold meets target; fallback to 0.5.")

    # Save artifacts
    joblib.dump(xgb, MODEL_PATH)
    joblib.dump(cal_model, CAL_MODEL_PATH)
    joblib.dump(safe_threshold, THRESHOLD_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    joblib.dump({
        "target_bad_rate": TARGET_BAD_RATE,
        "min_approval_rate": MIN_APPROVAL_RATE,
        "test_size": TEST_SIZE,
        "calibration_method": CALIBRATION_METHOD
    }, METADATA_PATH)
    print("✅ Model, threshold, and metadata saved.")

if __name__ == "__main__":
    train_and_save_model()