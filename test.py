import os 
import numpy as np 
import pandas as pd 
DATASET_PATH = "H:/project/data" 
def load_pollution_data(dataset_path=DATASET_PATH): 
    csv_path = os.path.join(dataset_path, "Measurement_summary.csv") 
    return pd.read_csv(csv_path) 
    

dataset = load_pollution_data() 
#print(dataset.head()) 
dataset.info() 
print(dataset.describe())