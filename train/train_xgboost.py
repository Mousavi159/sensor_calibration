from utils.dataset import load_data
from utils.data_preprocess import clean_data, remove_outliers
from utils.bias import(
    add_sinusoidal_drift_noise,
    add_mixed_drift_noise,
    add_linear_drift_noise
)
from utils.metrics import rsme_metrics, mae_metrics, r2_score_metrics
from models.xgboost_model import XGBoostModel
from utils.visualization import calibration_plot, distribution_errors, scatter_plot

from sklearn.preprocessing import StandardScaler
import numpy as np

DATASET_PATH = "data/"


# =====================================
# FEATURE ENGINEERING (LAG FEATURES)
# =====================================
def create_features(signal, window=20):
    X = []

    for i in range(window, len(signal)):
        row = [signal[i]]

        for j in range(1, window + 1):
            row.append(signal[i - j])

        X.append(row)

    return np.array(X)


def run():
    # -----------------------------
    # 🔒 FIX RANDOMNESS
    # -----------------------------
    np.random.seed(42)

    # -----------------------------
    # LOAD + CLEAN
    # -----------------------------
    dataset = load_data(DATASET_PATH)
    dataset = clean_data(dataset)
    dataset = remove_outliers(dataset)

    print("\nAFTER CLEANING:")
    print(dataset['PM2.5'].describe())

    # -----------------------------
    # ADD DRIFT + NOISE
    # -----------------------------
    dataset['PM2.5_drifted'] = add_linear_drift_noise(dataset['PM2.5'].values)

    signal = dataset['PM2.5_drifted'].values

    print("Drifted signal min/max:", signal.min(), signal.max())

    # -----------------------------
    # FEATURES
    # -----------------------------
    window = 20
    X = create_features(signal, window=window)
    y = dataset['PM2.5'].values[window:]

    # -----------------------------
    # SPLIT
    # -----------------------------
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = XGBoostModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Predictions min/max:", y_pred.min(), y_pred.max())

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
    # ALIGNMENT
    # -----------------------------
    drifted = signal[window:]
    drifted_test = drifted[split_idx:]

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    calibration_plot(
        y_true=y_test,
        y_drifted=drifted_test,
        y_pred=y_pred,
        title="Sensor Calibration (XGBoost)",
        save_path="results/xgboost_calibration2.png"
    )

    scatter_plot(
        y_test,
        y_pred,
        title="True vs Predicted (XGBoost)",
        save_path="results/xgboost_scatter2.png"
    )

    distribution_errors(
        y_test,
        y_pred,
        title="XGBoost Error Distribution",
        save_path="results/xgboost_errors2.png"
    )


if __name__ == "__main__":
    run()

"""
===== XGBoost Calibration Results =====
RMSE: 3.0387
MAE:  2.5761
R2 Score: 0.9490

2

===== XGBoost Calibration Results =====
RMSE: 3.6190
MAE:  3.0235
R2 Score: 0.9277
"""