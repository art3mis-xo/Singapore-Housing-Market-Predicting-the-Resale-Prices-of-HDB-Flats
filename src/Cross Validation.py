import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from dev.model_fitting import calculate_rmse

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
                      'TYPEMODEL_TE', 'FLOOR_MID', 'RESALE_PRICE_PER_SQM', "FLOOR_AREA_SQM", 'RESALE_PRICE']

if __name__ == "__main__":
    intermediate_csv_path = '../intermediate_csvs/'
    data_path: str = '../data/'
    auxiliary_data_path: str = '../data/auxiliary_data/'
    sqm_train_df = pd.read_csv(intermediate_csv_path + 'sqm_train.csv', parse_dates=['MONTH'])
    sqm_val_df = pd.read_csv(intermediate_csv_path + 'sqm_val.csv', parse_dates=['MONTH'])
    sqm_val_df = sqm_val_df[proposed_features2]
    X_val = sqm_val_df.iloc[:, :-3]
    y_val = sqm_val_df["RESALE_PRICE"]
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for cv_train_idx, cv_val_idx in kf.split(sqm_train_df):
        cv_train_df: DataFrame = sqm_train_df.iloc[cv_train_idx]
        cv_val_df: DataFrame = sqm_train_df.iloc[cv_val_idx]

        model = HistGradientBoostingRegressor(max_iter=3200, random_state=42, learning_rate=0.1, early_stopping=False,
                                              max_leaf_nodes=22, min_samples_leaf=336)

        cv_train_df = cv_train_df[proposed_features2]
        X_train = cv_train_df.iloc[:, :-3]
        y_train = cv_train_df["RESALE_PRICE_PER_SQM"]
        cv_val_df = cv_val_df[proposed_features2]
        X_cv_val = cv_val_df.iloc[:, :-3]
        y_cv_val = cv_val_df["RESALE_PRICE"]

        model = model.fit(X_train, y_train, sample_weight=cv_train_df["FLOOR_AREA_SQM"] ** 2)
        cv_pred = model.predict(X_cv_val)
        cv_error = calculate_rmse(y_cv_val, cv_pred * cv_val_df["FLOOR_AREA_SQM"])
        print(round(cv_error), end='\t')
        pred = model.predict(X_val)
        error = calculate_rmse(y_val, pred * sqm_val_df["FLOOR_AREA_SQM"])
        print(round(error))
