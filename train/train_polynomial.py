from utils.dataset import load_data
from utils.data_preprocess import clean_data, remove_outliers
from utils.bias import add_noise
from utils.metrics import rsme_metrics
from utils.metrics import mae_metrics
from utils.metrics import r2_score_metrics
from models.polynomial_model import PolynomialModel
from utils.visualization import calibration_plot
from utils.visualization import distribution_errors
from utils.visualization import scatter_plot

from sklearn.preprocessing import StandardScaler

import numpy as np

DATASET_PATH = "data/"


def run():
    dataset = load_data(DATASET_PATH)

    dataset = clean_data(dataset)
    dataset = remove_outliers(dataset)

    dataset['PM2.5_drifted'] = add_noise(dataset['PM2.5'].values)

    X = dataset[['PM2.5_drifted']].values
    y = dataset['PM2.5'].values

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = PolynomialModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = rsme_metrics(y_test, y_pred)
    mae = mae_metrics(y_test, y_pred)
    r2 = r2_score_metrics(y_test, y_pred)


    print("\n===== Polynomial Calibration Results =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    calibration_plot(
        y_true=y_test,
        y_drifted=X_test.flatten(),
        y_pred=y_pred,
        title="Sensor Calibration (Polynomial Model)",
        save_path="results/Polynomial_plot.png"
    )

    scatter_plot(
        y_test,
        y_pred,
        title="True vs Predicted",
        save_path="results/scatter_plot.png"
    )

    distribution_errors(
        y_test,
        y_pred,
        title="Distribution Errors",
        save_path="results/polynomial_errors.png"
    )


if __name__ == "__main__":
    run()


"""
===== Polynomial Calibration Results =====
RMSE: 34.0403
MAE:  31.2970
R2 Score: -5.7452
"""