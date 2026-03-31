import numpy as np

def add_sinusoidal_drift_noise(signal):
    t = np.arange(len(signal))

    # 🔥 bounded drift (does NOT explode)
    drift = 5 * np.sin(t / 200)

    noise = np.random.normal(0, 0.5, len(signal))

    return signal + drift + noise

def add_mixed_drift_noise(signal):
    t = np.arange(len(signal))

    drift = 3 * np.sin(t / 300) + 0.00001 * t
    noise = np.random.normal(0, 0.5, len(signal))

    return signal + drift + noise

def add_linear_drift_noise(signal):
    t = np.arange(len(signal))

    drift = 0.00002 * t   # 🔥 VERY SMALL slope
    noise = np.random.normal(0, 0.5, len(signal))

    return signal + drift + noise

def add_temporal_drift(signal):
    drift = np.zeros_like(signal)

    for i in range(1, len(signal)):
        drift[i] = drift[i-1] + np.random.normal(0, 0.1)

    noise = np.random.normal(0, 2, len(signal))

    return signal + drift + noise
