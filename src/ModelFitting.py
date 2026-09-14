#!/usr/bin/env python
# Change directory
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
# Uncomment respective models to test them
all_pipelines = [
    HistGradientBoostingRegressor(max_iter=1600, learning_rate=0.1, early_stopping=False),
    HistGradientBoostingRegressor(max_iter=3200, learning_rate=0.1, early_stopping=False),
    HistGradientBoostingRegressor(max_iter=6400, learning_rate=0.1, early_stopping=False),
    HistGradientBoostingRegressor(max_iter=3200, random_state=42, learning_rate=0.1, early_stopping=False,
                                  max_leaf_nodes=22, min_samples_leaf=336, max_features=0.6),

    # BaggingRegressor(n_estimators=800, random_state=42, n_jobs=n_jobs),
    # BaggingRegressor(n_estimators=1600, random_state=42, n_jobs=n_jobs),
    # BaggingRegressor(n_estimators=3200, random_state=42, n_jobs=n_jobs),

    # RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=n_jobs),
    # RandomForestRegressor(n_estimators=1600, random_state=42, n_jobs=n_jobs),
    # RandomForestRegressor(n_estimators=3200, random_state=42, n_jobs=n_jobs),

    # The following models require renaming all 4 "sample_weight" to "[model name lowercase]__sample_weight" in the file
    # for scikit-learn to work properly due to the use of make_pipeline().

    # make_pipeline(StandardScaler(), LinearRegression()),

    # make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
    # make_pipeline(StandardScaler(), Ridge(alpha=1)),
    # make_pipeline(StandardScaler(), Ridge(alpha=10)),
    # make_pipeline(StandardScaler(), Ridge(alpha=100)),
    # make_pipeline(StandardScaler(), Ridge(alpha=1000)),

    # make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    # make_pipeline(StandardScaler(), Lasso(alpha=1)),
    # make_pipeline(StandardScaler(), Lasso(alpha=10)),
    # make_pipeline(StandardScaler(), Lasso(alpha=100)),
    # make_pipeline(StandardScaler(), Lasso(alpha=1000)),

    # make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,), max_iter=100, learning_rate=0.3)),
    # make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, learning_rate=0.3)),
    # make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,), max_iter=400, learning_rate=0.3)),
    # make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,), max_iter=800, learning_rate=0.3)),
    # make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=800, learning_rate=0.1)),
    # make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256, 128), max_iter=1000, learning_rate=0.1)),

    # make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
    # make_pipeline(StandardScaler(), SVR(C=32, epsilon=0.1)),
    # make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1)),
    # make_pipeline(StandardScaler(), SVR(C=316, epsilon=0.1)),
    # make_pipeline(StandardScaler(), SVR(C=1000, epsilon=0.1)),
]

errs = []

# ### Model 1: Original features to predict RESALE_PRICE


orig_features = ['RESALE_MONTH', 'RESALE_YEAR', 'LEASE_COMMENCE_DATA',
                 'TOWN_TE', 'SUBZONE_TE', 'REGION_TE', 'MAX_FLOOR',
                 'ROOM_QTY', 'MODEL_TE', 'FLOOR_MID', 'FLOOR_AREA_SQM', 'RESALE_PRICE']

new_train_df = orig_train_df.copy()[orig_features]
X_train = new_train_df.iloc[:, :-1]
y_train = new_train_df["RESALE_PRICE"]

new_val_df = orig_val_df.copy()[orig_features]
X_val = new_val_df.iloc[:, :-1]
y_val = new_val_df["RESALE_PRICE"]

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# In[9]:


for pipeline in all_pipelines:
    with parallel_backend("threading"):
        model = clone(pipeline).fit(X_train, y_train)
        y_pred = model.predict(X_val)
    error = calculate_rmse(y_val, y_pred)

    errs.append(error)

# ### Model 2: Processed features to predict RESALE_PRICE


proposed_features1 = ['MONTH_SIN1', 'MONTH_COS1', 'RESALE_QUARTER_RPI', 'RENTAL', 'REMAINING_LEASE',
                      'TOWN_TE', 'SUBZONE_TE', 'REGION_TE', 'MAX_FLOOR',
                      'NUM_STN_1KM_OPEN', 'NUM_STN_2KM_OPEN',
                      'NUM_MRT_STOPS_1KM_OPEN', 'NUM_MRT_STOPS_2KM_OPEN',
                      'NUM_MRT_INT_1KM_OPEN', 'NUM_MRT_INT_2KM_OPEN',
                      'NUM_MRT_LINES_1KM_OPEN',
                      'NUM_MALL_1KM', 'NUM_MALL_2KM', 'NUM_HAKWER_1KM', 'NUM_HAWKER_2KM',
                      'NUM_PRI_SCH_1KM', 'NUM_PRI_SCH_2KM', 'NUM_SEC_SCH_1KM', 'NUM_SEC_SCH_2KM',
                      'DIST_NEAREST_HAWKER_WALK', 'DIST_NEAREST_MRT_WALK', 'DIST_NEAREST_PRI_SCH_WALK',
                      'DIST_NEAREST_SEC_SCH_WALK', 'DIST_NEAREST_MALL_WALK', 'DIST_TO_CBD', 'DIST_TO_CENTER',
                      'TYPEMODEL_TE', 'FLOOR_MID', 'FLOOR_AREA_SQM', 'RESALE_PRICE']

new_train_df = orig_train_df.copy()[proposed_features1]
X_train = new_train_df.iloc[:, :-1]
y_train = new_train_df["RESALE_PRICE"]

new_val_df = orig_val_df.copy()[proposed_features1]
X_val = new_val_df.iloc[:, :-1]
y_val = new_val_df["RESALE_PRICE"]

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# In[21]:


for pipeline in all_pipelines:
    with parallel_backend("threading"):
        model = clone(pipeline).fit(X_train, y_train)
        y_pred = model.predict(X_val)
    error = calculate_rmse(y_val, y_pred)
    errs.append(error)

# ### Model 3: Proposed features to predict RESALE_PRICE_PER_SQM
# (to adjust back to RESALE_PRICE before computing MSE)

# In[26]:


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

# In[27]:


new_val_df.columns

# In[28]:


for pipeline in all_pipelines:
    with parallel_backend("threading"):
        model = clone(pipeline).fit(X_train, y_train, sample_weight=new_train_df["FLOOR_AREA_SQM"] ** 2)
        y_pred = model.predict(X_val)
    error = calculate_rmse(y_val, y_pred * new_val_df["FLOOR_AREA_SQM"])
    errs.append(error)

# MSE(price_per_sqm)=\sum_i(y_true-y_pred)^2/n      (un-reweighted)
# MSE(raw price)=\sum_i(Area*y_true-Area*y_pred)^2/n
#               =\sum_i(Area^2)(y_true-y_pred)^2/n  (reweighted)

# RMSE(raw price) ∝ weighted_RMSE(price_per_sqm)

# ### Model 4: Proposed features to predict RESALE_PRICE_RPI_ADJ
# (to adjust back to RESALE_PRICE before computing MSE)

# In[29]:


rpi_train_df.columns

# In[30]:


proposed_features4 = ['MONTH_SIN1', 'MONTH_COS1', 'RENTAL_RPI_ADJ', 'REMAINING_LEASE',
                      'TOWN_TE', 'SUBZONE_TE', 'REGION_TE', 'MAX_FLOOR',
                      'NUM_STN_1KM_OPEN', 'NUM_STN_2KM_OPEN',
                      'NUM_MRT_STOPS_1KM_OPEN', 'NUM_MRT_STOPS_2KM_OPEN',
                      'NUM_MRT_INT_1KM_OPEN', 'NUM_MRT_INT_2KM_OPEN',
                      'NUM_MRT_LINES_1KM_OPEN',
                      'NUM_MALL_1KM', 'NUM_MALL_2KM', 'NUM_HAKWER_1KM', 'NUM_HAWKER_2KM',
                      'NUM_PRI_SCH_1KM', 'NUM_PRI_SCH_2KM', 'NUM_SEC_SCH_1KM', 'NUM_SEC_SCH_2KM',
                      'DIST_NEAREST_HAWKER_WALK', 'DIST_NEAREST_MRT_WALK',
                      'DIST_NEAREST_PRI_SCH_WALK', 'DIST_NEAREST_SEC_SCH_WALK',
                      'DIST_NEAREST_MALL_WALK', 'DIST_NEAREST_MRT_OPEN',
                      'DIST_TO_CBD', 'DIST_TO_CENTER',
                      'TYPEMODEL_TE', 'FLOOR_AREA_SQM', 'FLOOR_MID', 'RESALE_PRICE_RPI_ADJ', 'RESALE_QUARTER_RPI',
                      'RESALE_PRICE']  # 'RESALE_QUARTER_RPI' will be removed from X

# Uncomment the stuff below if you want to recreate the previous NUM_MRT which just adds all the open and planned mrt together
# Probably unneeded, so I'll remove it once I'm done testing -Nic
new_train_df = rpi_train_df.copy()[proposed_features4]
# new_train_df.insert(10, 'NUM_STN_1KM', new_train_df['NUM_STN_1KM_OPEN'] + new_train_df['NUM_STN_1KM_PLAN'])
# new_train_df.insert(11, 'NUM_STN_2KM', new_train_df['NUM_STN_2KM_OPEN'] + new_train_df['NUM_STN_2KM_PLAN'])
# new_train_df = new_train_df.drop(['NUM_STN_1KM_OPEN', 'NUM_STN_1KM_PLAN', 'NUM_STN_2KM_OPEN', 'NUM_STN_2KM_PLAN'], axis=1)
X_train = new_train_df.iloc[:, :-3]
y_train = new_train_df["RESALE_PRICE_RPI_ADJ"]

new_val_df = rpi_val_df.copy()[proposed_features4]
# new_val_df.insert(10, 'NUM_STN_1KM', new_val_df['NUM_STN_1KM_OPEN'] + new_val_df['NUM_STN_1KM_PLAN'])
# new_val_df.insert(11, 'NUM_STN_2KM', new_val_df['NUM_STN_2KM_OPEN'] + new_val_df['NUM_STN_2KM_PLAN'])
# new_val_df = new_val_df.drop(['NUM_STN_1KM_OPEN', 'NUM_STN_1KM_PLAN', 'NUM_STN_2KM_OPEN', 'NUM_STN_2KM_PLAN'], axis=1)
X_val = new_val_df.iloc[:, :-3]
y_val = new_val_df["RESALE_PRICE"]

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# In[31]:


for pipeline in all_pipelines:
    with parallel_backend("threading"):
        model = clone(pipeline).fit(X_train, y_train, sample_weight=new_train_df['RESALE_QUARTER_RPI'] ** 2)
        y_pred = model.predict(X_val)
    error = calculate_rmse(y_val, y_pred * new_val_df['RESALE_QUARTER_RPI'] / 100)

    errs.append(error)

# ### Model 5: Proposed features to predict RESALE_PRICE_PER_SQM_RPI_ADJ
# (to adjust back to RESALE_PRICE before computing MSE)

# In[32]:


proposed_features5 = ['MONTH_SIN1', 'MONTH_COS1', 'RENTAL_PER_SQM_RPI_ADJ', 'REMAINING_LEASE',
                      'TOWN_TE', 'SUBZONE_TE', 'REGION_TE', 'MAX_FLOOR',
                      'NUM_STN_1KM_OPEN', 'NUM_STN_2KM_OPEN',
                      'NUM_MRT_STOPS_1KM_OPEN', 'NUM_MRT_STOPS_2KM_OPEN',
                      'NUM_MRT_INT_1KM_OPEN', 'NUM_MRT_INT_2KM_OPEN',
                      'NUM_MRT_LINES_1KM_OPEN',
                      'NUM_MALL_1KM', 'NUM_MALL_2KM', 'NUM_HAKWER_1KM', 'NUM_HAWKER_2KM',
                      'NUM_PRI_SCH_1KM', 'NUM_PRI_SCH_2KM', 'NUM_SEC_SCH_1KM', 'NUM_SEC_SCH_2KM',
                      'DIST_NEAREST_HAWKER_WALK', 'DIST_NEAREST_MRT_WALK', 'DIST_NEAREST_PRI_SCH_WALK',
                      'DIST_NEAREST_SEC_SCH_WALK', 'DIST_NEAREST_MALL_WALK', 'DIST_TO_CBD', 'DIST_TO_CENTER',
                      'TYPEMODEL_TE', 'FLOOR_MID', 'RESALE_PRICE_PER_SQM_RPI_ADJ', 'FLOOR_AREA_SQM',
                      'RESALE_QUARTER_RPI',
                      'RESALE_PRICE']  # 'FLOOR_AREA_SQM' and 'RESALE_QUARTER_RPI' will be removed from X

new_train_df = sqmrpi_train_df.copy()[proposed_features5]
X_train = new_train_df.iloc[:, :-4]
y_train = new_train_df["RESALE_PRICE_PER_SQM_RPI_ADJ"]

new_val_df = sqmrpi_val_df.copy()[proposed_features5]
X_val = new_val_df.iloc[:, :-4]
y_val = new_val_df['RESALE_PRICE']

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# In[33]:

train_weight = new_train_df['RESALE_QUARTER_RPI'] * new_train_df['FLOOR_AREA_SQM']
for pipeline in all_pipelines:
    with parallel_backend("threading"):
        model = clone(pipeline).fit(X_train, y_train, sample_weight=train_weight ** 2)
        y_pred = model.predict(X_val)
    error = calculate_rmse(y_val, y_pred * new_val_df['RESALE_QUARTER_RPI'] / 100 * new_val_df['FLOOR_AREA_SQM'])

    errs.append(error)

# ### Model 6: Proposed features to predict RESALE_PRICE_PER_SQM_RPI_ADJ + PCA
# (to adjust back to RESALE_PRICE before computing MSE)

# In[34]:


proposed_features5 = ['MONTH_SIN1', 'MONTH_COS1', 'RENTAL_PER_SQM_RPI_ADJ', 'REMAINING_LEASE',
                      'TOWN_TE', 'SUBZONE_TE', 'REGION_TE', 'MAX_FLOOR',
                      'NUM_STN_1KM_OPEN', 'NUM_STN_2KM_OPEN',
                      'NUM_MRT_STOPS_1KM_OPEN', 'NUM_MRT_STOPS_2KM_OPEN',
                      'NUM_MRT_INT_1KM_OPEN', 'NUM_MRT_INT_2KM_OPEN',
                      'NUM_MRT_LINES_1KM_OPEN',
                      'NUM_MALL_1KM', 'NUM_MALL_2KM', 'NUM_HAKWER_1KM', 'NUM_HAWKER_2KM',
                      'NUM_PRI_SCH_1KM', 'NUM_PRI_SCH_2KM', 'NUM_SEC_SCH_1KM', 'NUM_SEC_SCH_2KM',
                      'DIST_NEAREST_HAWKER_WALK', 'DIST_NEAREST_MRT_WALK', 'DIST_NEAREST_PRI_SCH_WALK',
                      'DIST_NEAREST_SEC_SCH_WALK', 'DIST_NEAREST_MALL_WALK', 'DIST_TO_CBD', 'DIST_TO_CENTER',
                      'TYPEMODEL_TE', 'FLOOR_MID', 'RESALE_PRICE_PER_SQM_RPI_ADJ', 'FLOOR_AREA_SQM',
                      'RESALE_QUARTER_RPI',
                      'RESALE_PRICE']  # 'FLOOR_AREA_SQM' and 'RESALE_QUARTER_RPI' will be removed from X

new_train_df = sqmrpi_train_df.copy()[proposed_features5]
X_train = new_train_df.iloc[:, :-4]
y_train = new_train_df["RESALE_PRICE_PER_SQM_RPI_ADJ"]

new_val_df = sqmrpi_val_df.copy()[proposed_features5]
X_val = new_val_df.iloc[:, :-4]
y_val = new_val_df['RESALE_PRICE']

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# In[35]:


# PCA with 5 PCs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# --- Standardize features (fit on training only) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- Fit PCA on TRAIN only ---
pca = PCA(n_components=5, random_state=0)
X_train_pca = pca.fit_transform(X_train_scaled)

# --- Apply PCA to VALIDATION set ---
X_val_pca = pca.transform(X_val_scaled)

# --- Results ---
# Transformed components
X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f"PC{i + 1}" for i in range(5)])
X_val_pca_df = pd.DataFrame(X_val_pca, columns=[f"PC{i + 1}" for i in range(5)])

# Loadings (feature weights)
loadings = pd.DataFrame(pca.components_.T, index=X_train.columns, columns=X_train_pca_df.columns)

print("Train transformed shape:", X_train_pca_df.shape, sep='\n')
print("Validation transformed shape:", X_val_pca_df.shape, sep='\n')
print("Explained variance (absolute):", pca.explained_variance_, sep='\n')
print("Explained variance ratio (per component):", pca.explained_variance_ratio_, sep='\n')
print("Cumulative variance explained:", pca.explained_variance_ratio_.cumsum(), sep='\n')
print("PCA Components (which direction in feature space defines each component):", pca.components_, sep='\n')
print("Loadings matrix (how much each original feature contributes to that component, considering the strength "
      "(variance) of that component):", loadings, sep='\n')

# In[ ]:

train_multiplier = new_train_df['RESALE_QUARTER_RPI'] * new_train_df['FLOOR_AREA_SQM']
for pipeline in all_pipelines:
    with parallel_backend("threading"):
        model = clone(pipeline).fit(X_train_pca_df, y_train, sample_weight=train_multiplier ** 2)
        y_pred = model.predict(X_val_pca_df)
    error = calculate_rmse(y_val, y_pred * new_val_df['RESALE_QUARTER_RPI'] / 100 * new_val_df['FLOOR_AREA_SQM'])

    errs.append(error)

for i, err in enumerate(errs):
    print(int(round(err)), end='\n' if (i + 1) % len(all_pipelines) == 0 else '\t')

print("Program duration:", time.time() - t, "seconds")
