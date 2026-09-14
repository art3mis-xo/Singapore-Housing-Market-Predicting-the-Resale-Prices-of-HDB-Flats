import numpy as np
import pandas as pd
from sklearn.base import clone

def calculate_rmse(y_true, y_pred, squared=False):
    mse = np.mean((y_true - y_pred) ** 2)
    if squared:
        return mse
    return np.sqrt(mse)





# Ignore the code below, its unused
def fit_eval(model, X_train, y_train, X_val=None, y_val=None, X_test=None, 
             post_map_fn=lambda pred_raw,helper_df: pred_raw,
             helper_train_df=None, helper_val_df=None, helper_test_df=None):
    """
    Fit a model or pipeline, and return results in a dict.

    Returns a dict in the following format:
    {
      "preds": {
        "train": <np.array>
        "val": <np.array>
        "test:" <np.array>
      }
      "scores": {
        "train_rmse": <np.float>
        "val_rmse": <np.float>
      }
    }
    
    "preds" contains the predicted values for the corrosponding dataset, if specified.
    "scores" contains the rmse for the corrosponding dataset, if specified.
    Test scores are not included because we don't have y_test.
    """
    model = clone(model)
    model.fit(X_train, y_train)

    results = {
      'preds': {},
      'scores': {},
      'model': model
    }

    pred_train_raw = model.predict(X_train)
    pred_train = post_map_fn(pred_train_raw, helper_train_df)
    results['preds']['train'] = pred_train
    results['scores']['train_rmse'] = calculate_rmse(y_train, pred_train)

    if X_val is not None and y_val is not None:
      pred_val_raw = model.predict(X_val)
      pred_val = post_map_fn(pred_val_raw, helper_val_df)
      results['preds']['val'] = pred_val
      results['scores']['val_rmse'] = calculate_rmse(y_val, pred_val)

    if X_test is not None:
      pred_test_raw = model.predict(X_test)
      pred_test = post_map_fn(pred_test_raw, helper_test_df)
      results['preds']['test'] = pred_test

    return results