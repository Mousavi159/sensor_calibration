import matplotlib.pyplot as plt

def calibration_plot(y_true, y_drifted, y_pred, title=None, num_points=500, save_path=None):
    plt.figure(figsize=(10, 5))
    
    plt.plot(y_true[:num_points], label="True")
    plt.plot(y_drifted[:num_points], label="Drifted")
    plt.plot(y_pred[:num_points], label="Predicted")
    
    if title:
        plt.title(title)

    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("PM2.5")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def distribution_errors(y_true, y_pred, title=None, save_path=None):
    errors = y_true - y_pred

    plt.hist(errors, bins=50)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def scatter_plot(y_test, y_pred, title=None, save_path=None):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)

    if title:
        plt.title(title)

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.plot([0,100],[0,100],'r--')  # perfect line
    plt.grid()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_learning_curves(train_losses, val_losses=None, title=None, save_path=None):

    plt.figure(figsize=(8, 5))

    # Plot training loss
    plt.plot(train_losses, label="Train Loss")

    # Plot validation loss (if available)
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss")
    if title:
        plt.title(title)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()