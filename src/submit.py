import os

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import HistGradientBoostingRegressor

import src.dev.data_prep as dataprep

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

if __name__ == '__main__':
    path = '../'
    intermediate_csv_path = path + 'intermediate_csvs/'
    data_path = path + 'data/'
    auxiliary_data_path = path + 'data/auxiliary_data/'
    models_path = path + 'models'

    train_val_df: DataFrame = pd.read_csv(data_path + 'original_dataset/train.csv')
    test_df = pd.read_csv(data_path + 'original_dataset/test.csv')
    clean_train_val_df: DataFrame = dataprep.clean_data(train_val_df)
    clean_test_df = dataprep.clean_data(test_df)
    train_val_no_duplicates: DataFrame = clean_train_val_df.drop_duplicates()

    hdb_df: DataFrame = pd.read_csv(auxiliary_data_path + 'sg-hdb-block-details.csv')
    mrt_df: DataFrame = pd.read_csv(auxiliary_data_path + 'sg-mrt-stations.csv')
    mall_df: DataFrame = pd.read_csv(auxiliary_data_path + 'sg-shopping-malls.csv')
    hawker_df: DataFrame = pd.read_csv(auxiliary_data_path + 'sg-gov-hawkers.csv')
    pri_sch_df: DataFrame = pd.read_csv(auxiliary_data_path + 'sg-primary-schools.csv')
    sec_sch_df: DataFrame = pd.read_csv(auxiliary_data_path + 'sg-secondary-schools.csv')
    walk_dist_df: DataFrame = pd.read_csv(auxiliary_data_path + 'hdb-block-walking-distance.csv')

    processed_mrt_df: DataFrame = dataprep.process_mrt_df(mrt_df)
    processed_mall_df_dict: DataFrame = dataprep.process_mall_df(mall_df)
    processed_hawker_df: DataFrame = dataprep.process_hawker_df(hawker_df)
    processed_p_sch_df: DataFrame = dataprep.process_sch_df(pri_sch_df)
    processed_s_sch_df: DataFrame = dataprep.process_sch_df(sec_sch_df)
    processed_walk_dist_df: DataFrame = dataprep.process_walk_dist_df(walk_dist_df)
    processed_hdb_df: DataFrame = dataprep.process_hdb_df(hdb_df, processed_mrt_df, processed_mall_df_dict,
                                                          processed_hawker_df,
                                                          processed_p_sch_df, processed_s_sch_df,
                                                          processed_walk_dist_df)

    train_df: DataFrame = train_val_no_duplicates
    integrated_train_df: DataFrame = dataprep.integrate_aux_data(train_df, processed_hdb_df)
    integrated_test_df = dataprep.integrate_aux_data(clean_test_df, processed_hdb_df)
    processed_train_df: DataFrame = dataprep.process_data(integrated_train_df)
    processed_test_df = dataprep.process_data(integrated_test_df)
    added_train_df: DataFrame = dataprep.add_new_features(processed_train_df)
    added_test_df = dataprep.add_new_features(processed_test_df, is_test=True)
    sqm_train_df, model_means, typemodel_means, town_means, subzone_means, planning_area_means, region_means = dataprep.finalise_data(added_train_df, is_train=True, adj='sqm')

    sqm_test_df = dataprep.finalise_data(added_test_df, is_test=True, adj='sqm', model_means=model_means, typemodel_means=typemodel_means, town_means=town_means, subzone_means=subzone_means, planning_area_means=planning_area_means, region_means=region_means)

    model = HistGradientBoostingRegressor(max_iter=3200, random_state=42, learning_rate=0.1, early_stopping=False,
                                              max_leaf_nodes=22, min_samples_leaf=336)

    new_train_df: DataFrame = sqm_train_df.copy()[proposed_features2]
    new_test_df: DataFrame = sqm_test_df.copy()[proposed_features2[:-3] + ["FLOOR_AREA_SQM"]]
    X_train = new_train_df.iloc[:, :-3]
    y_train = new_train_df["RESALE_PRICE_PER_SQM"]
    X_test = new_test_df.iloc[:, :-1]
    print(X_train.shape, y_train.shape, X_test.shape)

    model = model.fit(X_train, y_train, sample_weight=new_train_df["FLOOR_AREA_SQM"] ** 2)
    y_pred = model.predict(X_test) * new_test_df["FLOOR_AREA_SQM"]
    pd.DataFrame({
        "Id": range(len(y_pred)),
        "Predicted": y_pred
    }).to_csv("pred.csv", index=False)

    model_filename = os.path.join(models_path, f"model3_HistGradientBoostingRegressor_model.pkl")
    joblib.dump(model, model_filename)
    print(f"✅ Model saved: {model_filename}")
