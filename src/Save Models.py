import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, HistGradientBoostingRegressor, \
    GradientBoostingRegressor

from dev.model_fitting import calculate_rmse

path = '../'
src_path = '.'
sys.path.append(src_path)
print(os.getcwd())

np.set_printoptions(precision=3)

intermediate_csv_path = path + 'intermediate_csvs/'
data_path = path + 'data/'
auxiliary_data_path = path + 'data/auxiliary_data/'
models_path = path + 'models'

t = time.time()

orig_train_df = pd.read_csv(intermediate_csv_path + 'orig_train.csv', parse_dates=['MONTH'])
orig_val_df = pd.read_csv(intermediate_csv_path + 'orig_val.csv', parse_dates=['MONTH'])
orig_test_df = pd.read_csv(intermediate_csv_path + 'orig_test.csv', parse_dates=['MONTH'])

sqm_train_df = pd.read_csv(intermediate_csv_path + 'sqm_train.csv', parse_dates=['MONTH'])
sqm_val_df = pd.read_csv(intermediate_csv_path + 'sqm_val.csv', parse_dates=['MONTH'])
sqm_test_df = pd.read_csv(intermediate_csv_path + 'sqm_test.csv', parse_dates=['MONTH'])

rpi_train_df = pd.read_csv(intermediate_csv_path + 'rpi_train.csv', parse_dates=['MONTH'])
rpi_val_df = pd.read_csv(intermediate_csv_path + 'rpi_val.csv', parse_dates=['MONTH'])
rpi_test_df = pd.read_csv(intermediate_csv_path + 'rpi_test.csv', parse_dates=['MONTH'])

sqmrpi_train_df = pd.read_csv(intermediate_csv_path + 'sqmrpi_train.csv', parse_dates=['MONTH'])
sqmrpi_val_df = pd.read_csv(intermediate_csv_path + 'sqmrpi_val.csv', parse_dates=['MONTH'])
sqmrpi_test_df = pd.read_csv(intermediate_csv_path + 'sqmrpi_test.csv', parse_dates=['MONTH'])

n_jobs = os.cpu_count() - 1

all_pipelines = [
    ("Bagging", BaggingRegressor(n_estimators=1600, random_state=42, n_jobs=n_jobs)),
    ("RandomForest", RandomForestRegressor(n_estimators=1600, random_state=42, n_jobs=n_jobs)),
    # ("GB", GradientBoostingRegressor(n_estimators=6400, random_state=42, learning_rate=0.3)),
    ("HistGB", HistGradientBoostingRegressor(max_iter=3200, random_state=42, learning_rate=0.1, early_stopping=False,
                                             max_leaf_nodes=22, min_samples_leaf=336, max_features=0.6))
]

proposed_features2 = ['MONTH_SIN1', 'MONTH_COS1', 'RESALE_QUARTER_RPI', 'RENTAL_PER_SQM', 'REMAINING_LEASE',
                      'TOWN_TE', 'SUBZONE_TE', 'REGION_TE', 'MAX_FLOOR',
                      'NUM_STN_1KM_OPEN', 'NUM_STN_2KM_OPEN',
                      'NUM_MRT_STOPS_1KM_OPEN', 'NUM_MRT_STOPS_2KM_OPEN',
                      'NUM_MRT_INT_1KM_OPEN', 'NUM_MRT_INT_2KM_OPEN',
                      'NUM_MRT_LINES_1KM_OPEN',
                      'NUM_MALL_1KM', 'NUM_MALL_2KM', 'NUM_HAKWER_1KM', 'NUM_HAWKER_2KM',
                      'NUM_PRI_SCH_1KM', 'NUM_PRI_SCH_2KM', 'NUM_SEC_SCH_1KM', 'NUM_SEC_SCH_2KM',
                      'DIST_NEAREST_HAWKER_WALK', 'DIST_NEAREST_MRT_WALK', 'DIST_NEAREST_PRI_SCH_WALK',
                      'DIST_NEAREST_SEC_SCH_WALK', 'DIST_NEAREST_MALL_WALK', 'DIST_TO_CBD', 'DIST_TO_CENTER',
                      'TYPEMODEL_TE', 'FLOOR_MID', 'RESALE_PRICE_PER_SQM', "FLOOR_AREA_SQM",
                      'RESALE_PRICE']  # 'FLOOR_AREA_SQM' will be removed from X

new_train_df = sqm_train_df.copy()[proposed_features2]
X_train = new_train_df.iloc[:, :-3]
y_train = new_train_df["RESALE_PRICE_PER_SQM"]

new_val_df = sqm_val_df.copy()[proposed_features2]

X_val = new_val_df.iloc[:, :-3]
y_val = new_val_df["RESALE_PRICE"]

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

for name, pipeline in all_pipelines:
    with parallel_backend("threading"):
        model = clone(pipeline).fit(X_train, y_train, sample_weight=new_train_df["FLOOR_AREA_SQM"] ** 2)
        y_pred = model.predict(X_val)
    error = calculate_rmse(y_val, y_pred * new_val_df["FLOOR_AREA_SQM"])
    print(error)
    model_filename = os.path.join(models_path, f"model3_{name}_model.pkl")
    joblib.dump(model, model_filename)
    print(f"✅ Model saved: {model_filename}")
