import numpy as np

def add_noise(signal):
    t = np.arange(len(signal))

    # realistic small drift
    drift = 0.02 * t + 2 * np.sin(t / 50)

    # small noise
    noise = np.random.normal(0, 0.5, len(signal))

    return signal + drift + noise