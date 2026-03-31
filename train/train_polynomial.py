from utils.dataset import load_data
from utils.data_preprocess import clean_data, remove_outliers
from utils.bias import(
    add_sinusoidal_drift_noise,
    add_mixed_drift_noise,
    add_linear_drift_noise
)
from utils.metrics import rsme_metrics, mae_metrics, r2_score_metrics
from models.polynomial_model import PolynomialModel
from utils.visualization import calibration_plot, distribution_errors, scatter_plot

from sklearn.preprocessing import StandardScaler
import numpy as np

DATASET_PATH = "data/"


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
    dataset['PM2.5_drifted'] = add_mixed_drift_noise(dataset['PM2.5'].values)

    # -----------------------------
    # FEATURES
    # -----------------------------
    X = dataset[['PM2.5_drifted']].values
    y = dataset['PM2.5'].values

    # -----------------------------
    # SPLIT
    # -----------------------------
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = PolynomialModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    rmse = rsme_metrics(y_test, y_pred)
    mae = mae_metrics(y_test, y_pred)
    r2 = r2_score_metrics(y_test, y_pred)

    print("\n===== Polynomial Calibration Results =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    calibration_plot(
        y_true=y_test,
        y_drifted=X_test.flatten(),
        y_pred=y_pred,
        title="Sensor Calibration (Polynomial Model)",
        save_path="results/Polynomial_plot2.png"
    )

    scatter_plot(
        y_test,
        y_pred,
        title="True vs Predicted",
        save_path="results/scatter_plot2.png"
    )

    distribution_errors(
        y_test,
        y_pred,
        title="Distribution Errors",
        save_path="results/polynomial_errors2.png"
    )


if __name__ == "__main__":
    run()

"""
===== Polynomial Calibration Results =====
RMSE: 3.3890
MAE:  2.9830
R2 Score: 0.9366
"""

"""
2
===== Polynomial Calibration Results =====
RMSE: 3.8750
MAE:  3.2606
R2 Score: 0.9171
"""