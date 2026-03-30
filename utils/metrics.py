import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def rsme_metrics(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae_metrics(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2_score_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2