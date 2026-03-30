from utils.dataset import load_data
from utils.data_preprocess import clean_data, remove_outliers
from utils.bias import add_noise
from utils.metrics import rsme_metrics, mae_metrics, r2_score_metrics
from models.xgboost_model import XGBoostModel
from utils.visualization import calibration_plot, distribution_errors, scatter_plot

import numpy as np


DATASET_PATH = "data/"


# =====================================
# FEATURE ENGINEERING (LAG FEATURES)
# =====================================
def create_features(signal, window=20):
    X = []

    for i in range(window, len(signal)):
        row = []

        # current value
        row.append(signal[i])

        # past values (temporal memory)
        for j in range(1, window + 1):
            row.append(signal[i - j])

        X.append(row)

    return np.array(X)


# =====================================
# MAIN TRAINING FUNCTION
# =====================================
def run():
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    dataset = load_data(DATASET_PATH)

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    dataset = clean_data(dataset)
    dataset = remove_outliers(dataset)

    # -----------------------------
    # ADD DRIFT + NOISE
    # -----------------------------
    dataset['PM2.5_drifted'] = add_noise(dataset['PM2.5'].values)

    # -----------------------------
    # FEATURES + TARGET
    # -----------------------------
    window = 20
    signal = dataset['PM2.5_drifted'].values

    X = create_features(signal, window=window)
    y = dataset['PM2.5'].values[window:]  # align with lag

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    model = XGBoostModel()
    model.train(X_train, y_train)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    rmse = rsme_metrics(y_test, y_pred)
    mae = mae_metrics(y_test, y_pred)
    r2 = r2_score_metrics(y_test, y_pred)

    print("\n===== XGBoost Calibration Results =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # -----------------------------
    # FIXED VISUALIZATION ALIGNMENT
    # -----------------------------
    drifted = signal[window:]                  # align with y
    drifted_test = drifted[split_idx:]         # align with test set

    calibration_plot(
        y_true=y_test,
        y_drifted=drifted_test,
        y_pred=y_pred,
        title="Sensor Calibration (XGBoost)",
        save_path="results/xgboost_calibration.png"
    )

    scatter_plot(
        y_test,
        y_pred,
        title="True vs Predicted (XGBoost)",
        save_path="results/xgboost_scatter.png"
    )

    distribution_errors(
        y_test,
        y_pred,
        title="XGBoost Error Distribution",
        save_path="results/xgboost_errors.png"
    )


# =====================================
# ENTRY POINT
# =====================================
if __name__ == "__main__":
    run()