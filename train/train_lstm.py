import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.dataset import load_data
from utils.data_preprocess import clean_data, remove_outliers
from utils.bias import add_mixed_drift_noise
from utils.metrics import rsme_metrics, mae_metrics, r2_score_metrics
from utils.visualization import (
    calibration_plot,
    distribution_errors,
    scatter_plot,
    plot_learning_curves
)

from models.lstm_model import LSTMModel


DATASET_PATH = "data/"


# ==============================
# SEQUENCE CREATION
# ==============================
def create_sequences(X, y, seq_length=20):
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])

    return np.array(X_seq), np.array(y_seq)


# ==============================
# TRAINING
# ==============================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # ===== TRAIN LOOP =====
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)           # shape [batch, 1]
            loss = criterion(outputs, y_batch) # shape matches

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ===== VALIDATION LOOP (FIXED MEMORY ISSUE) =====
        model.eval()
        val_loss_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_total += loss.item()

        train_losses.append(total_loss)
        val_losses.append(val_loss_total)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss_total:.4f}")

    return train_losses, val_losses


# ==============================
# EVALUATION
# ==============================
def evaluate(model, X_test, y_test, scaler_y):
    model.eval()
    preds_list = []

    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch = X_test[i:i+64]
            preds = model(batch)
            preds_list.append(preds.cpu().numpy())

    preds = np.vstack(preds_list)
    y_true = y_test.cpu().numpy()

    # Inverse scaling
    preds = scaler_y.inverse_transform(preds)
    y_true = scaler_y.inverse_transform(y_true)

    rmse = rsme_metrics(y_true, preds)
    mae = mae_metrics(y_true, preds)
    r2 = r2_score_metrics(y_true, preds)

    return preds, y_true, rmse, mae, r2


# ==============================
# MAIN PIPELINE
# ==============================
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== LOAD DATA =====
    dataset = load_data(DATASET_PATH)
    dataset = clean_data(dataset)
    dataset = remove_outliers(dataset)

    print("\nAFTER CLEANING:")
    print(dataset['PM2.5'].describe())

    # ===== ADD DRIFT =====
    dataset['PM2.5_drifted'] = add_mixed_drift_noise(dataset['PM2.5'].values)

    # ===== FEATURES =====
    X = dataset[['PM2.5_drifted', 'PM10', 'NO2', 'O3', 'CO']].values
    y = dataset['PM2.5'].values.reshape(-1, 1)

    # ===== NORMALIZATION =====
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # ===== SEQUENCES =====
    seq_length = 20
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

    # ===== SPLIT =====
    split = int(len(X_seq) * 0.8)

    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # ===== TORCH =====
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # ===== DATALOADERS =====
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ===== MODEL =====
    model = LSTMModel(
        input_size=X_train.shape[2],   # correct feature size
        hidden_size=64,
        num_layers=2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ===== TRAIN =====
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=25
    )

    # ===== EVALUATE =====
    preds, y_true, rmse, mae, r2 = evaluate(
        model,
        X_test,
        y_test_tensor,
        scaler_y
    )

    print("\n===== LSTM RESULTS =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # ===== VISUALIZATION =====
    drifted_last = scaler_X.inverse_transform(
        X_test[:, -1, :].cpu().numpy()
    )[:, 0]

    calibration_plot(
        y_true=y_true,
        y_drifted=drifted_last,
        y_pred=preds,
        title="Sensor Calibration (LSTM)",
        save_path="results/lstm_calibration.png"
    )

    scatter_plot(
        y_true,
        preds,
        title="True vs Predicted (LSTM)",
        save_path="results/lstm_scatter.png"
    )

    distribution_errors(
        y_true,
        preds,
        title="LSTM Error Distribution",
        save_path="results/lstm_errors.png"
    )

    plot_learning_curves(
        train_losses,
        val_losses,
        title="LSTM Learning Curves",
        save_path="results/lstm_learning_curve.png"
    )


if __name__ == "__main__":
    run()