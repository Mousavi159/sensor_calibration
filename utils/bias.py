import numpy as np

def add_noise(signal):
    t = np.arange(len(signal))

    # 🔥 bounded drift (does NOT explode)
    drift = 5 * np.sin(t / 200)

    noise = np.random.normal(0, 0.5, len(signal))

    return signal + drift + noise