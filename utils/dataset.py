import os
import pandas as pd 

def load_data(data_path):
    csv_path = os.path.join(data_path, "Measurement_summary.csv")
    return pd.read_csv(csv_path)