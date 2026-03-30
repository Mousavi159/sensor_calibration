import numpy as np
import pandas as pd

def clean_data(dataset):
    cols = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    
    dataset[cols] = dataset[cols].replace(-1, np.nan) # Replace -1 with NaN
    dataset = dataset.dropna()
    
    
    dataset['Measurement date'] = pd.to_datetime(dataset['Measurement date']) # Convert time

    # Sort
    dataset = dataset.sort_values(by='Measurement date')

    return dataset


def remove_outliers(dataset):
    return dataset[
        (dataset['PM2.5'] < dataset['PM2.5'].quantile(0.99)) &
        (dataset['PM10'] < dataset['PM10'].quantile(0.99))
    ]